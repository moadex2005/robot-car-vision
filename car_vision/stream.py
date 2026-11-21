"""Live frames from the phone, the laptop webcam or a recorded clip.

A background thread reads the camera as fast as frames arrive and keeps only
the newest one. Whoever asks for a frame always gets the latest picture, so a
slow model never makes the video fall behind the real world.

Run it on its own to check the stream:

    .venv\\Scripts\\python.exe -m car_vision.stream
    .venv\\Scripts\\python.exe -m car_vision.stream --source phone
    .venv\\Scripts\\python.exe -m car_vision.stream --source file

Keys in the window: q quits, c shows a millisecond clock for the delay test.
"""

from __future__ import annotations

import argparse
import logging
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


class FrameSource:
    """Keeps the newest camera frame ready, and reconnects when the stream drops."""

    def __init__(
        self,
        settings: dict[str, Any],
        opener: Callable[[], Capture | None] | None = None,
    ) -> None:
        self.source: str = require(settings, "source", str)
        if self.source not in SOURCES:
            raise ConfigError(f"stream.source must be one of {SOURCES}, got {self.source!r}")
        self.width: int = require(settings, "width", int)
        self.height: int = require(settings, "height", int)
        self.reconnect_seconds: float = float(require(settings, "reconnect_seconds", (int, float)))
        self.loop_file: bool = require(settings, "loop_file", bool)
        self._settings = settings
        self._opener = opener if opener is not None else self._default_opener()

        self.fps = FpsCounter()
        self._lock = threading.Lock()
        self._first_frame = threading.Event()
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
        capture: Capture | None = None
        frames_since_open = 0
        try:
            while not self._stop.is_set():
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
        if frame.shape[1] == self.width and frame.shape[0] == self.height:
            return frame
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

    # ---- real cameras ----------------------------------------------------

    def _default_opener(self) -> Callable[[], Capture | None]:
        settings = self._settings
        if self.source == "phone":
            url: str = require(settings, "phone_url", str)

            def open_phone() -> Capture:
                timeouts = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
                return cv2.VideoCapture(url, cv2.CAP_FFMPEG, timeouts)

            return open_phone

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
            record_path = resolve_path(args.record)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     RECORD_FPS, (source.width, source.height))
            if not writer.isOpened():
                print(f"Could not create the recording file: {record_path}")
                return 1
            print(f"Recording to {record_path}")

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
