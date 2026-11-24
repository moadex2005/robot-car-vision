"""Live frames from the phone, the laptop webcam or a recorded clip.

A background thread reads the camera as fast as frames arrive and keeps only
the newest one. Whoever asks for a frame always gets the latest picture, so a
slow model never makes the video fall behind the real world.

Every frame is shrunk to fit inside a box, stream.width x stream.height from
config.yaml to begin with. set_box() changes that box while the video runs:
the pipeline does that when a picture size is chosen in the Camera box.

The phone is read with FFmpeg (through OpenCV), asked to keep no buffer.
FFmpeg reads and decodes outside Python's global lock, so it does not slow
the detector running in another thread. A reader written in Python was tried
and measured on the phone: no less delay (59 to 79 ms against FFmpeg's 66),
and the pipeline fell from 25.8 to 14.5 frames a second, so it was taken out.

Run it on its own to check the stream:

    .venv\\Scripts\\python.exe -m car_vision.stream
    .venv\\Scripts\\python.exe -m car_vision.stream --source phone
    .venv\\Scripts\\python.exe -m car_vision.stream --source file

Keys in the window: q quits, c shows a millisecond clock for the delay test.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from car_vision.config import ConfigError, load_config, require, resolve_path, section

log = logging.getLogger(__name__)

SOURCES = ("phone", "webcam", "file")
OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 5000
RECORD_FPS = 15.0
FFMPEG_OPTIONS_VARIABLE = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
FFMPEG_LOW_DELAY = "fflags;nobuffer|flags;low_delay"

# Only one reader may have the webcam at a time, from before it opens the camera until
# after it has let it go. OpenCV's webcam driver on Windows keeps one shared record per
# camera: opening the camera while another capture of it is still being closed corrupts
# the program's memory and ends Python at once (seen live: off and on pressed quickly).
# A new reader therefore waits here until the one before it has finished.
WEBCAM_GATE = threading.Lock()
GATE_CHECK_SECONDS = 0.05       # a waiting reader checks this often whether it was stopped
GATE_SLOW_SECONDS = 3.0         # after waiting this long it says so in the log, once


class Capture(Protocol):
    """The small part of cv2.VideoCapture this module uses. Tests pass a fake."""

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...

    def isOpened(self) -> bool: ...


class FpsCounter:
    """Frames per second over the last few seconds."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        self._window = window_seconds
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self, now: float | None = None) -> None:
        stamp = time.monotonic() if now is None else now
        with self._lock:
            self._stamps.append(stamp)
            self._trim(stamp)

    def value(self, now: float | None = None) -> float:
        stamp = time.monotonic() if now is None else now
        with self._lock:
            self._trim(stamp)
            if len(self._stamps) < 2:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def _trim(self, now: float) -> None:
        while self._stamps and now - self._stamps[0] > self._window:
            self._stamps.popleft()


def fitted_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """The largest size with the same shape that fits inside max_width x max_height.

    Never enlarges, and keeps both sides even because video writers need that.
    """
    scale = min(max_width / width, max_height / height, 1.0)
    fitted_width = max(2, int(round(width * scale / 2)) * 2)
    fitted_height = max(2, int(round(height * scale / 2)) * 2)
    return fitted_width, fitted_height


def open_ffmpeg(url: str) -> Capture:
    """Open a stream with FFmpeg, asking it to keep no buffer."""
    # OpenCV reads this variable each time it opens an FFmpeg capture, and it
    # holds for the whole process: every FFmpeg capture opened later uses it too.
    # A value someone set on purpose is left alone.
    if FFMPEG_OPTIONS_VARIABLE not in os.environ:
        os.environ[FFMPEG_OPTIONS_VARIABLE] = FFMPEG_LOW_DELAY
    timeouts = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, timeouts)


class FrameSource:
    """Keeps the newest camera frame ready, and reconnects when the stream drops."""

    def __init__(
        self,
        settings: dict[str, Any],
        opener: Callable[[], Capture | None] | None = None,
        gate: Any = None,
    ) -> None:
        self.source: str = require(settings, "source", str)
        if self.source not in SOURCES:
            raise ConfigError(f"stream.source must be one of {SOURCES}, got {self.source!r}")
        self.width: int = require(settings, "width", int)
        self.height: int = require(settings, "height", int)
        self.reconnect_seconds: float = float(require(settings, "reconnect_seconds", (int, float)))
        self.loop_file: bool = require(settings, "loop_file", bool)
        self._settings = settings
        # The real webcam is used by one reader at a time (see WEBCAM_GATE). A test that
        # hands in its own opener has no real camera, so it gets no gate unless it asks.
        if gate is None and opener is None and self.source == "webcam":
            gate = WEBCAM_GATE
        self._gate = gate
        self._opener = opener if opener is not None else self._default_opener()

        self.fps = FpsCounter()
        self._lock = threading.Lock()
        # Frames are shrunk to fit inside this (width, height). It starts as the
        # configured size; set_box() may change it while the reader runs.
        self._box: tuple[int, int] = (self.width, self.height)
        self._first_frame = threading.Event()
        self._reopen = threading.Event()   # set by reopen(); the reader thread opens the stream again
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._connected = False
        self._ended = False
        self._frame_interval = 0.0   # seconds between frames; only used for a clip

    # ---- public --------------------------------------------------------

    def start(self) -> "FrameSource":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="frame-source", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=READ_TIMEOUT_MS / 1000 + 2)
            self._thread = None

    def read(self) -> tuple[np.ndarray | None, int]:
        """Return a copy of the newest frame and its number.

        The number goes up by one for every new frame, so a caller can tell
        whether it has already processed this picture. The frame is None until
        the first one arrives.
        """
        with self._lock:
            if self._frame is None:
                return None, self._sequence
            return self._frame.copy(), self._sequence

    def set_box(self, max_width: int, max_height: int) -> None:
        """Fit the frames from the next one on inside max_width x max_height.

        Safe to call from another thread while the reader runs: the width and
        the height are stored together, so a frame is never fitted with the
        new width and the old height. Frames are still never enlarged.
        Raises ValueError for a side that is not a whole number above 0.
        """
        for side in (max_width, max_height):
            if not isinstance(side, int) or isinstance(side, bool) or side < 1:
                raise ValueError(f"set_box needs whole numbers above 0, got {max_width!r} x {max_height!r}")
        with self._lock:
            self._box = (max_width, max_height)

    def reopen(self) -> None:
        """Close the stream and open it again at once, without calling it lost.

        Needed when the phone starts sending another picture size: OpenCV's
        FFmpeg reader fixes the frame size when the stream is opened and keeps
        scaling every later picture to that size (seen on the real phone: after
        asking for 1920 x 1080 the frames stayed 640 x 360). Safe to call from
        another thread; the reader thread does the work before its next frame.
        """
        self._reopen.set()

    def wait_for_frame(self, timeout: float = 10.0) -> bool:
        """Block until the first frame arrives. False when it never did."""
        return self._first_frame.wait(timeout)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def ended(self) -> bool:
        """True once a clip has played to its end with looping switched off."""
        return self._ended

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # ---- the reader thread ---------------------------------------------

    def _run(self) -> None:
        """The reader thread: wait for the camera to be free, read it, then say it is free again."""
        if not self._wait_for_gate():
            return                                     # stopped before the camera was free
        try:
            self._read_frames()
        finally:
            if self._gate is not None:
                self._gate.release()

    def _wait_for_gate(self) -> bool:
        """Wait until no other reader has the camera. False when stop() came first.

        The wait is in short steps, so stop() never has to wait for another
        reader: a source that is switched off again before its turn never
        touches the camera at all.
        """
        if self._gate is None:
            return True
        started = time.monotonic()
        told = False
        while not self._gate.acquire(timeout=GATE_CHECK_SECONDS):
            if self._stop.is_set():
                return False
            if not told and time.monotonic() - started > GATE_SLOW_SECONDS:
                told = True
                log.info("Waiting for the %s to be let go by its last reader", self.source)
        if self._stop.is_set():
            self._gate.release()
            return False
        return True

    def _read_frames(self) -> None:
        capture: Capture | None = None
        frames_since_open = 0
        try:
            while not self._stop.is_set():
                if self._reopen.is_set():              # reopen() was called: start the stream afresh
                    self._reopen.clear()
                    if capture is not None:
                        capture.release()
                        capture = None
                if capture is None:
                    capture = self._open()
                    frames_since_open = 0
                    if capture is None:
                        self._connected = False
                        self._stop.wait(self.reconnect_seconds)
                        continue

                started = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    capture.release()
                    capture = None
                    clip_finished = self.source == "file" and frames_since_open > 0
                    if clip_finished and self.loop_file:
                        continue                      # start the clip again at once
                    self._connected = False
                    if clip_finished:
                        self._ended = True
                        return
                    log.warning("Lost the %s stream; trying again in %.1f s",
                                self.source, self.reconnect_seconds)
                    self._stop.wait(self.reconnect_seconds)
                    continue

                frames_since_open += 1
                fitted = self._fit(frame)
                with self._lock:
                    self._frame = fitted
                    self._sequence += 1
                self._connected = True
                self.fps.tick()
                self._first_frame.set()

                if self._frame_interval > 0:           # play a clip at its own speed
                    spare = self._frame_interval - (time.monotonic() - started)
                    if spare > 0:
                        self._stop.wait(spare)
        finally:
            if capture is not None:
                capture.release()
            self._connected = False

    def _open(self) -> Capture | None:
        try:
            capture = self._opener()
        except ConfigError:
            raise
        except Exception:                              # camera drivers raise many kinds
            log.exception("Could not open the %s stream", self.source)
            return None
        if capture is None or not capture.isOpened():
            log.warning("The %s stream did not open", self.source)
            if capture is not None:
                capture.release()
            return None
        if self.source == "file":
            get = getattr(capture, "get", None)
            clip_fps = float(get(cv2.CAP_PROP_FPS)) if callable(get) else 0.0
            self._frame_interval = 1.0 / clip_fps if clip_fps > 1.0 else 1.0 / 30.0
        log.info("Opened the %s stream", self.source)
        return capture

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        """Shrink the frame to fit inside the box without changing its shape.

        With the 640 x 480 box from config.yaml, a wide 1920 x 1080 phone picture
        becomes 640 x 360, not a squashed 640 x 480: squashed faces are harder
        to recognise.
        """
        with self._lock:
            max_width, max_height = self._box       # both sides read together
        size = fitted_size(frame.shape[1], frame.shape[0], max_width, max_height)
        if size == (frame.shape[1], frame.shape[0]):
            return frame
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    # ---- real cameras ----------------------------------------------------

    def _default_opener(self) -> Callable[[], Capture | None]:
        settings = self._settings
        if self.source == "phone":
            url: str = require(settings, "phone_url", str)

            def open_phone_stream() -> Capture:
                return open_ffmpeg(url)

            return open_phone_stream

        if self.source == "webcam":
            index: int = require(settings, "webcam_index", int)

            def open_webcam() -> Capture:
                capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # opens fast on Windows
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return capture

            return open_webcam

        path = resolve_path(require(settings, "file_path", str))
        if not path.is_file():
            raise ConfigError(
                f"stream.source is 'file' but the clip does not exist: {path}\n"
                "Record a clip into data/samples/ or set stream.source to 'webcam'."
            )

        def open_file() -> Capture:
            return cv2.VideoCapture(str(path))

        return open_file


# ---- running this file on its own ----------------------------------------


def draw_status(frame: np.ndarray, text: str) -> None:
    """Write one line of status in the top-left corner, readable on any background."""
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def clock_image() -> np.ndarray:
    """A big clock with milliseconds. Point the phone at it to measure the delay:
    the clock itself shows now, the video of the clock shows then."""
    image = np.zeros((160, 560, 3), dtype=np.uint8)
    text = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(image, text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show the live camera stream.")
    parser.add_argument("--source", choices=SOURCES, help="override stream.source from config.yaml")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop by itself after this long")
    parser.add_argument("--no-window", action="store_true", help="print the FPS instead of showing video")
    parser.add_argument("--record", metavar="FILE", help="also save the video, e.g. data/samples/car_clip.mp4")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = dict(section(load_config(), "stream"))
    if args.source:
        settings["source"] = args.source

    source = FrameSource(settings)
    show_clock = False
    with source:
        if not source.wait_for_frame(timeout=15.0):
            print(f"No frame arrived from the {source.source} stream after 15 seconds.")
            if source.source == "phone":
                print("Check that IP Webcam is running, that the phone and the laptop are on "
                      "the same network, and that stream.phone_url in config.yaml is right.")
            return 1

        writer = None
        if args.record:
            first_frame, _ = source.read()
            record_path = resolve_path(args.record)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            frame_size = (first_frame.shape[1], first_frame.shape[0])
            writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     RECORD_FPS, frame_size)
            if not writer.isOpened():
                print(f"Could not create the recording file: {record_path}")
                return 1
            print(f"Recording {frame_size[0]} x {frame_size[1]} video to {record_path}")

        started = time.monotonic()
        last_print = started
        last_sequence = 0
        next_record = started
        while not source.ended:
            now = time.monotonic()
            if args.seconds and now - started >= args.seconds:
                break
            frame, sequence = source.read()
            # Save at a steady rate, so the clip plays back at the speed it was filmed.
            if writer is not None and frame is not None and now >= next_record and sequence != last_sequence:
                writer.write(frame)
                last_sequence = sequence
                next_record = max(next_record + 1.0 / RECORD_FPS, now)
            status = f"FPS {source.fps.value():.1f} | {source.source}"
            if writer is not None:
                status += " | recording"
            if not source.connected:
                status += " | reconnecting"
            if args.no_window:
                if now - last_print >= 1.0:
                    print(status)
                    last_print = now
                time.sleep(0.05)
                continue
            if frame is not None:
                draw_status(frame, status)
                cv2.imshow("Car camera", frame)
            if show_clock:
                cv2.imshow("Clock", clock_image())
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                show_clock = not show_clock
                if not show_clock:
                    cv2.destroyWindow("Clock")
        if writer is not None:
            writer.release()
        print(f"Final FPS {source.fps.value():.1f} from the {source.source} stream")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
