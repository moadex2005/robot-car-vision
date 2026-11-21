"""Finds objects in camera frames with a pretrained YOLO model and tracks them.

Every object gets a label ("person", "chair", ...), a confidence and a box.
The tracker gives each object a number that stays the same while it is in view.

This is the only module that imports ultralytics or torch. Both are imported
inside the functions that load the model or check the graphics card, so
importing this file stays cheap and the unit tests can pass a fake model.

Run it on its own to see the live result:

    .venv\\Scripts\\python.exe -m car_vision.detector --source webcam
    .venv\\Scripts\\python.exe -m car_vision.detector --source webcam --seconds 20 --no-window
    .venv\\Scripts\\python.exe -m car_vision.detector --file data/samples/people_clip.mp4

The q key quits the window; Ctrl+C stops it in the console.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from car_vision import draw
from car_vision.config import ConfigError, load_config, require, resolve_path, section
from car_vision.state import Detection, WorldState
from car_vision.stream import SOURCES, FpsCounter, FrameSource

log = logging.getLogger(__name__)

DEVICES = ("auto", "cuda", "cpu")
VOTE_MEMORY_SECONDS = 30.0      # forget a track's label votes after this long unseen
IDLE_WAIT_SECONDS = 0.005       # pause when no new frame has arrived yet
PHONE_HINT = ("Check that IP Webcam is running, that the phone and the laptop are on "
              "the same network, and that stream.phone_url in config.yaml is right.")


def cuda_available() -> bool:
    """True when PyTorch can see a graphics card. Tests replace this function."""
    import torch

    return bool(torch.cuda.is_available())


def tracker_names() -> list[str]:
    """The tracker settings files that come with the installed Ultralytics.

    They live in ultralytics/cfg/trackers. The folder is found without
    importing ultralytics, so this stays cheap.
    """
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        return []
    folder = Path(spec.origin).parent / "cfg" / "trackers"
    return sorted(path.name for path in folder.glob("*.yaml"))


class LabelVotes:
    """Remembers which labels each track has had, so a still object keeps one name.

    The detector sometimes calls the same couch a dog for a frame or two. Each
    track reports the label it has had most often; a tie goes to the newest.
    """

    def __init__(self, memory_seconds: float = VOTE_MEMORY_SECONDS) -> None:
        self.memory_seconds = memory_seconds
        self._counts: dict[int, dict[str, int]] = {}
        self._newest: dict[int, dict[str, int]] = {}   # label -> number of its latest vote
        self._last_seen: dict[int, float] = {}
        self._votes_cast = 0

    def vote(self, track_id: int, label: str, now: float) -> str:
        """Count this label for the track and return the track's winning label."""
        counts = self._counts.setdefault(track_id, {})
        newest = self._newest.setdefault(track_id, {})
        self._votes_cast += 1
        counts[label] = counts.get(label, 0) + 1
        newest[label] = self._votes_cast
        self._last_seen[track_id] = now
        return max(counts, key=lambda name: (counts[name], newest[name]))

    def forget_old(self, now: float) -> None:
        """Drop the votes of tracks not seen for memory_seconds."""
        old = [track for track, seen in self._last_seen.items() if now - seen > self.memory_seconds]
        for track in old:
            del self._counts[track], self._newest[track], self._last_seen[track]

    def clear(self) -> None:
        self._counts.clear()
        self._newest.clear()
        self._last_seen.clear()

    def __len__(self) -> int:
        return len(self._counts)


class Detector:
    """Wraps the YOLO model and its tracker. One call to detect() per frame."""

    def __init__(self, settings: dict[str, Any], model: Any = None) -> None:
        self.model_name: str = require(settings, "model", str)
        self.models_dir: str = require(settings, "models_dir", str)
        self.confidence: float = float(require(settings, "confidence", (int, float)))
        self.image_size: int = require(settings, "image_size", int)
        wanted_device: str = require(settings, "device", str)
        self.tracker: str = require(settings, "tracker", str)
        self._check_settings()

        self._device = self._choose_device(wanted_device, injected=model is not None)
        self._votes = LabelVotes()
        self._frame_size: tuple[int, int] | None = None
        self._clock = time.monotonic
        if model is not None:
            self._model = model
        else:
            self._model = self._load_model()
            self._warm_up()

    @property
    def device(self) -> str:
        """The device actually in use: "cuda" or "cpu"."""
        return self._device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run the model and the tracker on one frame and return what it found."""
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("detect() needs a camera frame, got an empty one")
        height, width = frame.shape[:2]
        self._reset_if_size_changed((width, height))
        results = self._model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self._device,
            verbose=False,
        )
        found = to_detections(results[0], width, height) if results else []
        return self._steady_labels(found)

    # ---- setup -----------------------------------------------------------

    def _check_settings(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ConfigError(f"detector.confidence must be above 0 and at most 1, got {self.confidence}")
        if self.image_size <= 0 or self.image_size % 32 != 0:
            raise ConfigError(
                f"detector.image_size must be a positive multiple of 32 such as 320 or 640, "
                f"got {self.image_size}"
            )
        names = tracker_names()
        if not names:
            raise ConfigError("Could not find the tracker settings of Ultralytics. Is it installed in .venv?")
        if self.tracker not in names:
            raise ConfigError(f"detector.tracker must be one of {', '.join(names)}; got {self.tracker!r}")

    @staticmethod
    def _choose_device(wanted: str, injected: bool) -> str:
        if wanted not in DEVICES:
            raise ConfigError(f"detector.device must be one of {DEVICES}, got {wanted!r}")
        if wanted == "cpu":
            return "cpu"
        if wanted == "auto" and injected:
            return "cpu"              # a test model: do not import torch to look for a card
        has_card = cuda_available()
        if wanted == "cuda" and not has_card:
            raise ConfigError(
                "detector.device is 'cuda' but PyTorch cannot see a graphics card. "
                "Set detector.device to 'auto' or 'cpu' in config.yaml."
            )
        return "cuda" if has_card else "cpu"

    def _load_model(self) -> Any:
        from ultralytics import YOLO

        folder = resolve_path(self.models_dir)
        folder.mkdir(parents=True, exist_ok=True)
        weights = folder / self.model_name
        if not weights.is_file():
            log.info("Downloading %s into %s (first run only)", self.model_name, folder)
        # A full path makes the download land in models_dir, not the project root.
        # Half precision stays off: GTX 16-series cards give empty or NaN results with it.
        return YOLO(str(weights))

    def _warm_up(self) -> None:
        """Run once on a blank picture, so the first camera frame is not held up
        for seconds while the graphics card gets ready. Leaves no tracks or votes."""
        print(f"Warming up the model on {self._device} ...")
        self.detect(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        self._reset_tracking()

    # ---- tracking --------------------------------------------------------

    def _reset_if_size_changed(self, size: tuple[int, int]) -> None:
        if self._frame_size is not None and size != self._frame_size:
            log.info("Frame size changed from %s to %s; starting the tracker again", self._frame_size, size)
            self._reset_tracking()
        self._frame_size = size

    def _reset_tracking(self) -> None:
        """Forget every track and every label vote.

        Ultralytics keeps its trackers on model.predictor.trackers, and every
        tracker it ships has reset(). A fake model without them is left alone.
        """
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        self._votes.clear()
        self._frame_size = None

    def _steady_labels(self, found: list[Detection]) -> list[Detection]:
        """Give each tracked object the label its track has had most often."""
        now = self._clock()
        self._votes.forget_old(now)          # first, so a track back after 30 s starts afresh
        steady = []
        for item in found:
            if item.track_id is None:
                steady.append(item)
                continue
            label = self._votes.vote(item.track_id, item.label, now)
            steady.append(item if label == item.label else replace(item, label=label))
        return steady


# ---- turning the library's result into Detection objects ------------------


def to_numpy(values: Any) -> np.ndarray:
    """Turn a torch tensor (on any device) or a plain list into a numpy array."""
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def label_for(names: Any, class_index: int) -> str:
    """Class name for a class number. names is a dict or a list in Ultralytics."""
    try:
        return str(names[class_index])
    except (KeyError, IndexError, TypeError):
        return str(class_index)


def clip_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Round a box to whole pixels and keep it inside the frame."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1 = min(max(x1, 0), width - 1)
    x2 = min(max(x2, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    y2 = min(max(y2, 0), height - 1)
    return x1, y1, x2, y2


def to_detections(result: Any, width: int, height: int) -> list[Detection]:
    """Read result.boxes (xyxy, conf, cls, id) and result.names into Detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    corners = to_numpy(boxes.xyxy)
    confidences = to_numpy(boxes.conf)
    classes = to_numpy(boxes.cls)
    ids = None if boxes.id is None else to_numpy(boxes.id)   # None until a track is confirmed

    detections = []
    for row in range(len(corners)):
        track_id = None if ids is None else int(ids[row])
        detections.append(Detection(
            label=label_for(result.names, int(classes[row])),
            confidence=float(confidences[row]),
            box=clip_box(corners[row], width, height),
            track_id=track_id,
        ))
    return detections


# ---- running this file on its own -----------------------------------------


def describe(detections: list[Detection]) -> str:
    """Short text such as 'person #1, cup' for the console."""
    parts = []
    for item in detections:
        text = item.label if item.track_id is None else f"{item.label} #{item.track_id}"
        if text not in parts:
            parts.append(text)
    return ", ".join(parts) if parts else "nothing"


class LiveStats:
    """Counts detect() calls and their times, for the console and the summary line."""

    def __init__(self, keep_labels: bool) -> None:
        self.keep_labels = keep_labels      # only the console mode prints what was seen
        self.calls = 0
        self.timed_ms = 0.0                 # every call except the first
        self.first_at = 0.0
        self.last_at = 0.0
        self.second_ms: list[float] = []
        self.second_seen: list[Detection] = []

    def record(self, took_ms: float, detections: list[Detection], now: float) -> None:
        self.calls += 1
        self.last_at = now
        if self.calls == 1:
            self.first_at = now             # left out of the averages
        else:
            self.timed_ms += took_ms
            self.second_ms.append(took_ms)
        if self.keep_labels:
            self.second_seen.extend(detections)

    def second_average_ms(self) -> float:
        return sum(self.second_ms) / len(self.second_ms) if self.second_ms else 0.0

    def new_second(self) -> None:
        self.second_ms.clear()
        self.second_seen.clear()

    def summary(self, device: str) -> str:
        timed = max(self.calls - 1, 0)
        span = self.last_at - self.first_at
        fps = timed / span if timed and span > 0 else 0.0
        average_ms = self.timed_ms / timed if timed else 0.0
        return (f"Average detection FPS {fps:.1f} | average detect {average_ms:.1f} ms "
                f"over {timed} calls (first call left out) | device {device}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show live object detection and tracking.")
    parser.add_argument("--source", choices=SOURCES, help="override stream.source from config.yaml")
    parser.add_argument("--file", metavar="PATH",
                        help="play this clip instead of stream.file_path (implies --source file); "
                             "a relative path starts at the project folder")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop by itself after this long")
    parser.add_argument("--no-window", action="store_true", help="print one line per second instead of video")
    args = parser.parse_args(argv)
    if args.file and args.source not in (None, "file"):
        parser.error(f"--file plays a clip, so it cannot be used with --source {args.source}")
    return args


def stream_settings_for(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """The stream: section with the command-line overrides applied. config.yaml is not changed."""
    settings = dict(section(config, "stream"))
    if args.source:
        settings["source"] = args.source
    if args.file:
        settings["source"] = "file"
        settings["file_path"] = args.file
    return settings


def run(detector: Detector, source: FrameSource, args: argparse.Namespace, stats: LiveStats) -> int:
    """Detect on every new frame until the time is up, the clip ends or q is pressed."""
    if not source.wait_for_frame(timeout=15.0):
        print(f"No frame arrived from the {source.source} stream after 15 seconds.")
        if source.source == "phone":
            print(PHONE_HINT)
        return 1

    world = WorldState()
    detect_fps = FpsCounter()
    started = time.monotonic()
    last_print = started
    last_sequence = 0
    while not source.ended:
        now = time.monotonic()
        if args.seconds and now - started >= args.seconds:
            break

        frame, sequence = source.read()
        is_new = frame is not None and sequence != last_sequence
        if is_new:
            last_sequence = sequence
            began = time.perf_counter()
            detections = detector.detect(frame)
            stats.record((time.perf_counter() - began) * 1000.0, detections, time.monotonic())
            detect_fps.tick()
            world.update_detections(detections, (frame.shape[1], frame.shape[0]))

        status = (f"Detect FPS {detect_fps.value():.1f} | camera {source.fps.value():.1f} | "
                  f"{stats.second_average_ms():.1f} ms | {detector.device}")
        if not source.connected:
            status += " | reconnecting"
        if now - last_print >= 1.0:
            if args.no_window:
                print(f"{status} | seen: {describe(stats.second_seen)}")
            stats.new_second()
            last_print = now

        if not args.no_window:
            if is_new:
                draw.draw_detections(frame, world.snapshot().detections)
                draw.draw_status(frame, status)
                cv2.imshow("Car camera: objects", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if not is_new:
            time.sleep(IDLE_WAIT_SECONDS)     # do not spin a whole processor core
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config()
        source = FrameSource(stream_settings_for(config, args))   # a missing clip fails here, before the model loads
        detector = Detector(section(config, "detector"))
    except ConfigError as error:
        print(f"Cannot start: {error}")
        return 1
    print(f"Model {detector.model_name} ready on {detector.device}")

    stats = LiveStats(keep_labels=args.no_window)
    source.start()
    try:
        return run(detector, source, args, stats)
    except KeyboardInterrupt:
        print("Stopped with Ctrl+C")
        return 0
    except ConfigError as error:
        print(f"Cannot go on: {error}")
        return 1
    finally:
        source.stop()
        cv2.destroyAllWindows()
        print(stats.summary(detector.device))


if __name__ == "__main__":
    raise SystemExit(main())
