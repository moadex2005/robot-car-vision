"""Finds faces in camera frames, encodes them with InsightFace, and names them.

No model is trained here. Enrolment (car_vision.enroll) turned each person's
photos into face vectors once; this module turns each new face into a vector
the same way and asks car_vision.faces which saved vector it is closest to.

This module and car_vision.enroll are the only ones that import insightface or
onnxruntime. Both are imported inside load_face_model, so importing this file
stays cheap and the unit tests can pass a fake face model instead.

Run it on its own to see the live result:

    .venv\\Scripts\\python.exe -m car_vision.recognizer --source webcam
    .venv\\Scripts\\python.exe -m car_vision.recognizer --source file --seconds 20 --no-window

The q key quits the window.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from car_vision import draw
from car_vision.config import ConfigError, load_config, require, resolve_path, section
from car_vision.detector import Detector, clip_box
from car_vision.faces import FaceMatcher, FaceMemory, Person, load_embeddings, load_people
from car_vision.state import UNKNOWN, Box, Detection, RecognizedFace, WorldState
from car_vision.stream import SOURCES, FpsCounter, FrameSource

log = logging.getLogger(__name__)

DEVICES = ("auto", "cuda", "cpu")
FACE_TASKS = ["detection", "recognition"]       # skip landmarks, gender and age
GPU_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


# ---- loading the InsightFace models (shared with car_vision.enroll) -----------


def choose_providers(device: str, available: Sequence[str]) -> list[str]:
    """ONNX Runtime providers to try, best first, for the device setting."""
    if device not in DEVICES:
        raise ConfigError(f"recognizer.device must be one of {DEVICES}, got {device!r}")
    if device != "cpu" and GPU_PROVIDER in available:
        return [GPU_PROVIDER, CPU_PROVIDER]
    if device == "cuda":
        log.warning("recognizer.device is 'cuda' but ONNX Runtime offers no graphics card "
                    "provider; the face models run on the processor")
    return [CPU_PROVIDER]


def face_model_providers(face_model: Any) -> dict[str, str]:
    """The provider each loaded face model really runs on, e.g. {'detection': 'CUDAExecutionProvider'}."""
    used = {}
    for task, model in getattr(face_model, "models", {}).items():
        session = getattr(model, "session", None)
        if session is not None:
            used[task] = session.get_providers()[0]
    return used


def load_face_model(settings: dict) -> Any:
    """InsightFace FaceAnalysis with only its detection and recognition models.

    The model pack is downloaded into models_dir on the first run. The graphics
    card is used when device allows it; if it cannot be used, the models run on
    the processor and a warning is logged.
    """
    pack: str = require(settings, "model_pack", str)
    models_dir = resolve_path(require(settings, "models_dir", str))
    detection_size: int = require(settings, "detection_size", int)
    device: str = require(settings, "device", str)

    import onnxruntime
    from insightface.app import FaceAnalysis

    providers = choose_providers(device, onnxruntime.get_available_providers())
    if GPU_PROVIDER in providers:
        onnxruntime.preload_dlls()       # use the CUDA libraries that PyTorch ships

    models_dir.mkdir(parents=True, exist_ok=True)
    # InsightFace keeps packs in <root>/models/<pack>; root is our models_dir.
    if not (models_dir / "models" / pack).is_dir():
        log.info("Downloading the %s face models into %s (first run only, about 330 MB)", pack, models_dir)

    def build(chosen: list[str]) -> Any:
        model = FaceAnalysis(name=pack, root=str(models_dir), allowed_modules=FACE_TASKS,
                             providers=chosen)
        model.prepare(ctx_id=0 if GPU_PROVIDER in chosen else -1,
                      det_size=(detection_size, detection_size))
        return model

    try:
        model = build(providers)
    except Exception as error:                       # a broken CUDA setup can raise many kinds
        if GPU_PROVIDER not in providers:
            raise
        log.warning("The face models could not start on the graphics card (%s); "
                    "using the processor instead", error)
        model = build([CPU_PROVIDER])

    used = face_model_providers(model)
    if GPU_PROVIDER in providers and any(p != GPU_PROVIDER for p in used.values()):
        log.warning("The graphics card provider did not load for every face model; "
                    "running on: %s. Face checks will be slow.", used)
    return model


# ---- naming faces ---------------------------------------------------------------


def embedding_of(face: Any) -> np.ndarray | None:
    """The face vector of an InsightFace face: normed_embedding, else embedding."""
    vector = getattr(face, "normed_embedding", None)
    if vector is None:
        vector = getattr(face, "embedding", None)
    return vector


def track_for(face_box: Box, detections: Sequence[Detection]) -> int | None:
    """Track number of the smallest person box that contains the middle of the face."""
    middle_x = (face_box[0] + face_box[2]) / 2
    middle_y = (face_box[1] + face_box[3]) / 2
    best: Detection | None = None
    best_area = 0
    for detection in detections:
        if detection.label != "person":
            continue
        x1, y1, x2, y2 = detection.box
        if not (x1 <= middle_x <= x2 and y1 <= middle_y <= y2):
            continue
        area = (x2 - x1) * (y2 - y1)
        if best is None or area < best_area:
            best, best_area = detection, area
    return None if best is None else best.track_id


class Recognizer:
    """Finds the faces in a frame and names the enrolled people among them."""

    def __init__(self, settings: dict, face_model: Any = None,
                 people: dict[str, Person] | None = None,
                 matcher: FaceMatcher | None = None) -> None:
        threshold = float(require(settings, "match_threshold", (int, float)))
        self.min_face_pixels: int = require(settings, "min_face_pixels", int)
        people_file: str = require(settings, "people_file", str)
        embeddings_file: str = require(settings, "embeddings_file", str)

        # Cheap files first, so a mistake in them shows before the slow model load.
        self._people = people if people is not None else load_people(resolve_path(people_file))
        if matcher is None:
            ids, embeddings = load_embeddings(resolve_path(embeddings_file))
            matcher = FaceMatcher(ids, embeddings, threshold)
        self._matcher = matcher
        self._warn_about_mismatches()
        self._model = face_model if face_model is not None else load_face_model(settings)

    @property
    def providers(self) -> dict[str, str]:
        """The provider each face model runs on; empty for a fake model."""
        return face_model_providers(self._model)

    def person(self, person_id: str) -> Person | None:
        """The info card of an enrolled person, or None."""
        return self._people.get(person_id)

    def recognize(self, frame: np.ndarray,
                  detections: Sequence[Detection] = ()) -> list[RecognizedFace]:
        """One RecognizedFace per face big enough to trust, named or UNKNOWN."""
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("recognize() needs a camera frame, got an empty one")
        height, width = frame.shape[:2]
        found: list[tuple[Box, str | None, float]] = []
        for face in self._model.get(frame):
            corners = np.asarray(face.bbox, dtype=np.float32)[:4]
            if min(corners[2] - corners[0], corners[3] - corners[1]) < self.min_face_pixels:
                continue                               # too far away to trust
            vector = embedding_of(face)
            person_id, score = (None, 0.0) if vector is None else self._matcher.match(vector)
            found.append((clip_box(corners, width, height), person_id, score))

        found = self._one_face_per_person(found)
        return [
            RecognizedFace(
                name=self._display_name(person_id),
                similarity=score,
                box=box,
                person_id=person_id,
                track_id=track_for(box, detections) if detections else None,
            )
            for box, person_id, score in found
        ]

    # ---- helpers ------------------------------------------------------------------

    @staticmethod
    def _one_face_per_person(found: list[tuple[Box, str | None, float]]) -> list[tuple[Box, str | None, float]]:
        """When two faces match the same person, only the better match keeps the name."""
        best_score: dict[str, float] = {}
        for _box, person_id, score in found:
            if person_id is not None and score > best_score.get(person_id, -2.0):
                best_score[person_id] = score
        result = []
        claimed: set[str] = set()
        for box, person_id, score in found:
            if person_id is not None and (person_id in claimed or score < best_score[person_id]):
                person_id = None
            if person_id is not None:
                claimed.add(person_id)
            result.append((box, person_id, score))
        return result

    def _display_name(self, person_id: str | None) -> str:
        if person_id is None:
            return UNKNOWN
        person = self._people.get(person_id)
        return person.name if person is not None else person_id

    def _warn_about_mismatches(self) -> None:
        enrolled = self._matcher.person_ids
        no_photos = sorted(set(self._people) - enrolled)
        no_card = sorted(enrolled - set(self._people))
        problems = []
        if no_photos:
            problems.append(f"no saved face for {', '.join(no_photos)} (run the enroll command)")
        if no_card:
            problems.append(f"saved faces for {', '.join(no_card)} who are not in the people file "
                            "(they are shown by id, without an info card)")
        if problems:
            log.warning("Face data does not match the people file: %s", "; ".join(problems))


# ---- running this file on its own -----------------------------------------------


def load_known_faces(settings: dict) -> tuple[dict[str, Person], FaceMatcher]:
    """People and saved faces for the live view. A missing or broken file is
    reported in plain words and everyone is shown as Unknown, instead of stopping."""
    threshold = float(require(settings, "match_threshold", (int, float)))
    try:
        people = load_people(resolve_path(require(settings, "people_file", str)))
    except ConfigError as error:
        print(f"Note: {error}")
        people = {}
    try:
        ids, embeddings = load_embeddings(resolve_path(require(settings, "embeddings_file", str)))
    except ConfigError as error:
        print(f"Note: {error}")
        ids, embeddings = [], np.zeros((0, 0), dtype=np.float32)
    if not ids:
        print("Nobody is enrolled yet, so every face is shown as Unknown.")
    return people, FaceMatcher(ids, embeddings, threshold)


def describe_faces(faces: Sequence[RecognizedFace]) -> str:
    """Short text such as 'Sara Ahmed 0.63, Unknown 0.21' with each name's best score."""
    best: dict[str, float] = {}
    for face in faces:
        best[face.name] = max(best.get(face.name, -1.0), face.similarity)
    return ", ".join(f"{name} {score:.2f}" for name, score in best.items()) if best else "no faces"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show live face recognition with info cards.")
    parser.add_argument("--source", choices=SOURCES, help="override stream.source from config.yaml")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop by itself after this long")
    parser.add_argument("--no-window", action="store_true", help="print one line per second instead of video")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config()
        stream_settings = dict(section(config, "stream"))
        if args.source:
            stream_settings["source"] = args.source
        settings = section(config, "recognizer")
        every_n: int = require(settings, "every_n_frames", int)
        if every_n < 1:
            raise ConfigError("recognizer.every_n_frames must be 1 or more")
        people, matcher = load_known_faces(settings)
        detector = Detector(section(config, "detector"))
        recognizer = Recognizer(settings, people=people, matcher=matcher)
    except ConfigError as error:
        print(f"Cannot start: {error}")
        return 1

    print(f"Detector {detector.model_name} on {detector.device} | face models on {recognizer.providers}")
    world = WorldState()
    memory = FaceMemory()
    loop_fps = FpsCounter()

    frames = 0                    # new frames processed
    face_calls = 0
    face_ms_total = 0.0           # every face check except the first (warm-up)
    frame_ms_total = 0.0          # whole work per frame, except the first two
    first_frame_at = 0.0
    last_frame_at = 0.0
    second_face_ms: list[float] = []
    second_seen: list[RecognizedFace] = []

    try:
        source = FrameSource(stream_settings)
    except ConfigError as error:
        print(f"Cannot start: {error}")
        return 1
    with source:
        if not source.wait_for_frame(timeout=15.0):
            print(f"No frame arrived from the {source.source} stream after 15 seconds.")
            return 1

        started = time.monotonic()
        last_print = started
        last_sequence = 0
        while not source.ended:
            now = time.monotonic()
            if args.seconds and now - started >= args.seconds:
                break

            frame, sequence = source.read()
            if frame is not None and sequence != last_sequence:
                last_sequence = sequence
                began = time.perf_counter()
                detections = detector.detect(frame)
                world.update_detections(detections, (frame.shape[1], frame.shape[0]))
                if frames % every_n == 0:
                    face_began = time.perf_counter()
                    faces = recognizer.recognize(frame, detections)
                    face_ms = (time.perf_counter() - face_began) * 1000.0
                    memory.remember(faces, detections, time.monotonic())
                    face_calls += 1
                    if face_calls > 1:
                        face_ms_total += face_ms
                        second_face_ms.append(face_ms)
                    second_seen.extend(faces)
                else:
                    faces = memory.current(detections, time.monotonic())
                world.update_faces(faces)
                frames += 1
                last_frame_at = time.monotonic()
                if frames == 1:
                    first_frame_at = last_frame_at
                elif frames > 2:
                    frame_ms_total += (time.perf_counter() - began) * 1000.0
                loop_fps.tick()
            else:
                frame = None          # nothing new to show

            face_ms_now = sum(second_face_ms) / len(second_face_ms) if second_face_ms else 0.0
            status = (f"FPS {loop_fps.value():.1f} | camera {source.fps.value():.1f} | "
                      f"faces every {every_n} frames, {face_ms_now:.0f} ms")
            if not source.connected:
                status += " | reconnecting"

            if args.no_window:
                if now - last_print >= 1.0:
                    print(f"{status} | seen: {describe_faces(second_seen)}")
                    second_face_ms.clear()
                    second_seen.clear()
                    last_print = now
                if frame is None:
                    time.sleep(0.005)
                continue

            if frame is not None:
                snapshot = world.snapshot()
                draw.draw_detections(frame, snapshot.detections)
                draw.draw_faces(frame, snapshot.faces, show_scores=True)
                for face in snapshot.known_faces():
                    person = recognizer.person(face.person_id)
                    if person is not None:
                        draw.draw_info_card(frame, face, person)
                draw.draw_status(frame, status)
                cv2.imshow("Car camera: faces", frame)
                if now - last_print >= 1.0:
                    second_face_ms.clear()
                    second_seen.clear()
                    last_print = now
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    span = last_frame_at - first_frame_at
    average_fps = (frames - 1) / span if frames > 1 and span > 0 else 0.0
    average_face_ms = face_ms_total / (face_calls - 1) if face_calls > 1 else 0.0
    average_frame_ms = frame_ms_total / (frames - 2) if frames > 2 else 0.0
    capacity = 1000.0 / average_frame_ms if average_frame_ms > 0 else 0.0
    print(f"Average FPS {average_fps:.1f} | average face check {average_face_ms:.1f} ms over "
          f"{max(face_calls - 1, 0)} checks (first left out) | average work per frame "
          f"{average_frame_ms:.1f} ms, enough for {capacity:.0f} FPS | face models on {recognizer.providers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
