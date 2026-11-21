"""The see-decide-show loop: camera frames in, annotated JPEG pictures and a status out.

One background thread takes every new frame from the camera and
1. finds the objects in it (the detector),
2. checks the faces on every few frames (the recognizer) and keeps them on
   screen in between (FaceMemory),
3. writes what it saw into the shared WorldState, which the assistant reads,
4. asks the ReactionEngine which greetings and warnings to show,
5. draws the boxes and names and keeps the picture as JPEG bytes for the page.

The info cards are not drawn on the video: the demo page shows them.

Next to the drawn picture the loop keeps one View: a clean copy of the same
frame (nothing drawn on it) with that frame's detections and faces. When a
visitor clicks the video, target() finds what is under the click in that view
and cuts a small photo of it out of the clean frame.

Every part can be handed in from outside, so the tests run the loop with
fakes and need no camera, model or graphics card.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from car_vision import draw
from car_vision.config import ConfigError, require, section
from car_vision.detector import Detector
from car_vision.faces import FaceMemory, Person
from car_vision.reactions import Reaction, ReactionEngine
from car_vision.recognizer import Recognizer, load_known_faces
from car_vision.state import Box, Detection, RecognizedFace, WorldSnapshot, WorldState
from car_vision.stream import FpsCounter, FrameSource

log = logging.getLogger(__name__)

IDLE_WAIT_SECONDS = 0.005      # pause when no new frame has arrived yet
STOP_WAIT_SECONDS = 2.0        # stop() waits at most this long for the loop to end
MAX_FAILURES_IN_A_ROW = 10     # the loop gives up after this many broken frames in a row
STALE_SECONDS = 2.0            # status() leaves out objects, people and reactions older than this
NOBODY_ENROLLED = "Nobody is enrolled yet, so every face is shown as Unknown."
NO_INFO_CARDS = "No about.md was found for the enrolled people, so recognised faces show only a name."

PHOTO_MARGIN = 0.15            # the photo of a target is its box grown by 15 % on each side
PHOTO_MIN_PIXELS = 160         # ... and at least this wide and high, where the frame allows
CLIP_FINISHED = "Clip finished"   # drawn and shown when a clip has played to its end
AREA_SHARE = 0.5               # a click on no box takes half the frame wide and half the frame high
FACE_LABEL = "face"            # the label of a target that is a face box with no person box around it


@dataclass(frozen=True)
class Target:
    """What a click on the video points at, with a small photo of it."""

    kind: str                    # "object", "person" or "area"
    label: str                   # the detector's label; "" for an area
    confidence: float            # 0.0 for an area
    track_id: int | None
    photo: bytes                 # JPEG cut from the clean frame
    person: Person | None = None # the recognised person, for kind "person"


@dataclass(frozen=True)
class View:
    """One frame as the camera gave it (nothing drawn on it) and what was found in it.

    The loop replaces the whole view at once and never writes into its frame
    afterwards, so a web-server thread can read it without holding the lock.
    """

    frame: np.ndarray
    detections: tuple[Detection, ...]
    faces: tuple[RecognizedFace, ...]

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the frame in pixels."""
        height, width = self.frame.shape[:2]
        return width, height


class Pipeline:
    """Runs the camera, the models and the reactions in one background thread."""

    def __init__(self, config: dict, source=None, detector=None, recognizer=None,
                 reactions=None, state: WorldState | None = None) -> None:
        recognizer_settings = section(config, "recognizer")
        self.every_n_frames: int = require(recognizer_settings, "every_n_frames", int)
        if self.every_n_frames < 1:
            raise ConfigError("recognizer.every_n_frames must be 1 or more")
        self.jpeg_quality: int = require(section(config, "app"), "jpeg_quality", int)
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError(f"app.jpeg_quality must be from 1 to 100, got {self.jpeg_quality}")

        self._notes: list[str] = []
        # Cheap parts first, so a wrong setting or a missing clip shows before the slow model loads.
        self._source = source if source is not None else FrameSource(section(config, "stream"))
        self._reactions = reactions if reactions is not None else ReactionEngine(section(config, "reactions"))
        self._state = state if state is not None else WorldState()
        # A detector that cannot load is a real failure: its error is not caught here.
        self._detector = detector if detector is not None else Detector(section(config, "detector"))
        self._recognizer = recognizer if recognizer is not None else self._build_recognizer(recognizer_settings)
        for note in self._notes:
            log.warning("%s", note)

        self._memory = FaceMemory()
        self._fps = FpsCounter()
        self._lock = threading.Lock()          # guards the four values the web server reads
        self._jpeg: bytes | None = None
        self._view: View | None = None
        self._on_screen: list[Reaction] = []
        self._error: str | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Only the loop thread uses these.
        self._frames = 0                       # frames handed to the models so far
        self._failures = 0                     # broken frames in a row
        self._last_drawn: np.ndarray | None = None
        self._lost_line: str | None = None     # the line already drawn on the last frame, if any

    # ---- public ------------------------------------------------------------

    @property
    def state(self) -> WorldState:
        """What the camera sees right now. The assistant reads the same object."""
        return self._state

    def start(self) -> "Pipeline":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._source.start()
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """End the loop (within two seconds), then let go of the camera."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=STOP_WAIT_SECONDS)
            if self._thread.is_alive():
                log.warning("The video loop did not stop within %.0f seconds", STOP_WAIT_SECONDS)
            self._thread = None
        self._source.stop()

    def latest_jpeg(self) -> bytes | None:
        """The newest annotated frame as JPEG bytes; None until the first frame.

        Every new picture is a new bytes object, so a caller can tell a new
        frame from one it has already sent by comparing with `is`.
        """
        with self._lock:
            return self._jpeg

    def status(self) -> dict[str, Any]:
        """Everything the page shows besides the video, as plain JSON-ready values."""
        snapshot = self._state.snapshot()
        with self._lock:
            on_screen = list(self._on_screen)
            error = self._error
        # While the stream is down nothing new is seen, so old sightings are left out.
        fresh = snapshot.detections_at > 0 and time.monotonic() - snapshot.detections_at <= STALE_SECONDS
        return {
            "source": self._source.source,
            "connected": bool(self._source.connected) and error is None,   # a stopped loop is not live
            "ended": bool(self._source.ended),        # a clip played to its end (loop_file: false)
            "fps": round(self._fps.value(), 1),
            "device": self._detector.device,
            "recognition": self._recognizer is not None,
            "notes": list(self._notes),
            "people": self._people(snapshot) if fresh else [],
            "objects": objects_in_view(snapshot, self._person_card) if fresh else [],
            "reactions": [{"kind": r.kind, "text": r.text} for r in on_screen] if fresh else [],
            "error": error,
        }

    def target(self, x: float | None = None, y: float | None = None,
               track_id: int | None = None, label: str | None = None) -> Target | None:
        """What a click points at, with a photo cut from the clean frame; None when it is gone.

        x and y are the click as fractions of the picture (0 to 1 from the
        top-left corner). With a track_id the point is not used. Called from
        the web server's threads while the loop runs.
        """
        with self._lock:
            view = self._view                   # the view is never changed, only replaced
        if view is None:
            return None
        found = find_target(view, x, y, track_id, label, self._person_card)
        if found is None:
            return None
        target, box = found
        photo = self._encode(cut_out(view.frame, photo_box(box, view.size)))
        return replace(target, photo=photo)

    # ---- start-up --------------------------------------------------------------

    def _build_recognizer(self, settings: dict) -> Recognizer | None:
        """The real face recogniser, or None when it cannot start. Never stops the program.

        Missing people or a missing embeddings file are handled the way the
        recognizer's own live view handles them: a note, and everyone is Unknown.
        """
        try:
            people, matcher = load_known_faces(settings)
            recognizer = Recognizer(settings, people=people, matcher=matcher)
        except Exception as error:          # a broken model pack or graphics card setup raises many kinds
            log.exception("Face recognition could not start")
            self._notes.append(f"Face recognition is off: {first_line(error)}")
            return None
        if not matcher.person_ids:
            self._notes.append(NOBODY_ENROLLED)
        elif not people:
            self._notes.append(NO_INFO_CARDS)
        return recognizer

    # ---- the loop thread -----------------------------------------------------

    def _run(self) -> None:
        last_sequence = 0
        while not self._stop.is_set():
            frame, sequence = self._source.read()
            if frame is None or sequence == last_sequence:
                if not self._source.connected:
                    self._show_stream_lost()
                self._stop.wait(IDLE_WAIT_SECONDS)
                continue
            last_sequence = sequence
            try:
                self._process(frame)
            except Exception as error:      # one bad frame must not end the demo
                if self._too_many_failures(error):
                    return
            else:
                self._failures = 0

    def _process(self, frame: np.ndarray) -> None:
        """See, decide and show for one frame. Nothing is written until the models are done."""
        now = time.monotonic()
        index = self._frames
        self._frames += 1

        detections = self._detector.detect(frame)
        if self._recognizer is not None and index % self.every_n_frames == 0:
            found = self._recognizer.recognize(frame, detections)
            self._memory.remember(found, detections, now)
        faces = self._memory.current(detections, now)

        height, width = frame.shape[:2]
        self._state.update_detections(detections, (width, height), now)
        self._state.update_faces(faces, now)
        on_screen = self._reactions.update(self._state.snapshot(), now)

        # The boxes are drawn on a copy: the camera's own frame goes into the view untouched.
        drawn = frame.copy()
        draw.draw_detections(drawn, detections)
        draw.draw_faces(drawn, faces)
        jpeg = self._encode(drawn)
        view = View(frame, tuple(detections), tuple(faces))
        with self._lock:                        # picture, view and reactions change together
            self._jpeg = jpeg
            self._view = view
            self._on_screen = list(on_screen)
        self._last_drawn = drawn
        self._lost_line = None
        self._fps.tick(now)

    def _too_many_failures(self, error: Exception) -> bool:
        """Log a skipped frame. True when too many failed in a row and the loop must stop."""
        self._failures += 1
        if self._failures == 1:
            log.exception("Skipped a frame because processing it failed")
        else:
            log.warning("Skipped another frame (%d in a row): %s: %s",
                        self._failures, type(error).__name__, error)
        if self._failures < MAX_FAILURES_IN_A_ROW:
            return False
        message = (f"The video loop stopped after {MAX_FAILURES_IN_A_ROW} frames in a row failed "
                   f"({type(error).__name__}: {first_line(error)})")
        log.error("%s", message)
        with self._lock:
            self._error = message
        self._show_line_on_last_frame(f"Stopped: {MAX_FAILURES_IN_A_ROW} frames in a row failed")
        return True

    def _show_stream_lost(self) -> None:
        """Keep serving the last picture, with a line on it saying why nothing new comes.

        Each line is drawn once; it is drawn again only when the reason changes,
        e.g. from reconnecting to "Clip finished" when a clip has played to its end.
        """
        text = CLIP_FINISHED if self._source.ended else f"{self._source.source} stream lost - reconnecting"
        if self._lost_line == text:
            return
        self._lost_line = text
        self._show_line_on_last_frame(text)

    def _show_line_on_last_frame(self, text: str) -> None:
        if self._last_drawn is None:
            return                              # no picture yet: nothing to write on
        frame = self._last_drawn.copy()
        draw.draw_status(frame, text)
        jpeg = self._encode(frame)
        with self._lock:
            self._jpeg = jpeg

    def _encode(self, frame: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("OpenCV could not encode the frame as JPEG")
        return buffer.tobytes()

    # ---- status pieces ---------------------------------------------------------

    def _people(self, snapshot: WorldSnapshot) -> list[dict[str, Any]]:
        """One info card per recognised person in view."""
        people: list[dict[str, Any]] = []
        seen: set[str] = set()
        for face in snapshot.known_faces():
            if face.person_id in seen:
                continue
            seen.add(face.person_id)
            card = self._person_card(face)
            people.append({
                "id": face.person_id,
                "name": card.name,
                "role": card.role,
                "lines": list(card.lines),
                "similarity": round(float(face.similarity), 2),
            })
        return people

    def _person_card(self, face: RecognizedFace) -> Person:
        """The person a known face belongs to; just the name when they have no about.md."""
        card = self._recognizer.person(face.person_id) if self._recognizer is not None else None
        return card if card is not None else Person(face.person_id, face.name)


CardOf = Callable[[RecognizedFace], Person]     # the person a known face belongs to


# ---- the objects list on the page --------------------------------------------------


def objects_in_view(snapshot: WorldSnapshot, card_of: CardOf) -> list[dict[str, Any]]:
    """One entry per detection: people first, then by label, then by track number."""
    owners = face_owners(snapshot.detections, snapshot.faces)
    order = sorted(range(len(snapshot.detections)), key=lambda i: list_order(snapshot.detections[i]))
    return [object_entry(snapshot.detections[i], owners.get(i), snapshot.frame_size, card_of) for i in order]


def list_order(detection: Detection) -> tuple[bool, str, bool, int]:
    track = detection.track_id
    return detection.label != "person", detection.label, track is None, track if track is not None else 0


def object_entry(detection: Detection, face: RecognizedFace | None, size: tuple[int, int],
                 card_of: CardOf) -> dict[str, Any]:
    """One entry, e.g. {"track_id": 7, "label": "laptop", "confidence": 0.83, "box": [...], "name": None}.

    `face` is the known face that belongs to this box (see face_owners), or None.
    """
    return {
        "track_id": int(detection.track_id) if detection.track_id is not None else None,
        "label": detection.label,
        "confidence": round(float(detection.confidence), 2),
        "box": box_fractions(detection.box, size),
        "name": card_of(face).name if face is not None else None,
    }


def box_fractions(box: Box, size: tuple[int, int]) -> list[float]:
    """A pixel box as [x1, y1, x2, y2] fractions of the frame, each from 0 to 1."""
    width, height = size
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = box
    return [round(min(max(value / whole, 0.0), 1.0), 3)
            for value, whole in ((x1, width), (y1, height), (x2, width), (y2, height))]


# ---- what a click points at ----------------------------------------------------------


def find_target(view: View, x: float | None, y: float | None, track_id: int | None,
                label: str | None, card_of: CardOf) -> tuple[Target, Box] | None:
    """The target (its photo still empty) and the box to cut it from, or None.

    1. A track id: that detection (with that label, when one is given).
    2. Else the smallest detection or face box that holds the point.
    3. Else, when no label was asked for, the area around the point.
    """
    if track_id is not None:
        index = by_track(view.detections, track_id, label)
        return detection_target(view, index, card_of) if index is not None else None
    if x is None or y is None:
        return None
    width, height = view.size
    px, py = x * width, y * height
    hit = smallest_box_at(view, px, py, label)
    if isinstance(hit, RecognizedFace):
        return face_target(hit, view, card_of)
    if isinstance(hit, Detection):
        return detection_target(view, index_of(view.detections, hit), card_of)
    if label is not None:
        return None                             # the thing asked for is no longer there
    return Target("area", "", 0.0, None, b""), area_box(view.size, px, py)


def by_track(detections: Sequence[Detection], track_id: int, label: str | None) -> int | None:
    """The position of the detection with that track number (and label), or None."""
    for index, detection in enumerate(detections):
        if detection.track_id == track_id and (label is None or detection.label == label):
            return index
    return None


def index_of(detections: Sequence[Detection], detection: Detection) -> int:
    """The position of this very detection (two detections can be equal, so `is` is used)."""
    return next(index for index, item in enumerate(detections) if item is detection)


def smallest_box_at(view: View, px: float, py: float,
                    label: str | None) -> Detection | RecognizedFace | None:
    """The smallest box holding the point, so a phone in a hand beats the person holding it.

    Only detections with `label` count when one is given; face boxes count
    when no label is given or the label is "person".
    """
    boxes: list[Detection | RecognizedFace] = [
        d for d in view.detections if (label is None or d.label == label) and contains(d.box, px, py)]
    if label is None or label == "person":
        boxes += [f for f in view.faces if contains(f.box, px, py)]
    return min(boxes, key=lambda item: box_area(item.box), default=None)


def detection_target(view: View, index: int, card_of: CardOf) -> tuple[Target, Box]:
    """An object, or a person (recognised when a known face belongs to their box)."""
    detection = view.detections[index]
    confidence = float(detection.confidence)
    if detection.label != "person":
        return Target("object", detection.label, confidence, detection.track_id, b""), detection.box
    face = face_owners(view.detections, view.faces).get(index)
    person = card_of(face) if face is not None else None
    return Target("person", detection.label, confidence, detection.track_id, b"", person), detection.box


def face_target(face: RecognizedFace, view: View, card_of: CardOf) -> tuple[Target, Box]:
    """A click on a face means that person: their person box when there is one, else the face box."""
    index = body_of(face, view.detections, view.faces)
    if index is not None:
        return detection_target(view, index, card_of)
    person = card_of(face) if face.is_known else None
    return Target("person", FACE_LABEL, 0.0, face.track_id, b"", person), face.box


# ---- which face belongs to which person box ------------------------------------------
#
# The rules, so a stranger standing behind a recognised person never gets their name:
# - a face with a track number belongs only to the person box with that track number;
# - a face without one belongs to the smallest person box around its centre that has
#   no face of its own;
# - one person is never given to two boxes in the same view.


def face_owners(detections: Sequence[Detection], faces: Sequence[RecognizedFace]) -> dict[int, RecognizedFace]:
    """The known face of each person box, by the box's position in `detections`."""
    owners: dict[int, RecognizedFace] = {}
    given: set[str] = set()                     # person ids already given to a box
    known = [face for face in faces if face.is_known]
    for face in known:                          # matching track numbers come first and win
        index = body_of(face, detections, faces) if face.track_id is not None else None
        if index is not None and index not in owners and face.person_id not in given:
            owners[index] = face
            given.add(face.person_id)
    untracked = [(boxes_around(face, detections, faces), face) for face in known if face.track_id is None]
    untracked.sort(key=lambda item: box_area(detections[item[0][0]].box) if item[0] else 0)
    for boxes, face in untracked:               # then the smallest box around the face
        free = [index for index in boxes if index not in owners]
        if free and face.person_id not in given:
            owners[free[0]] = face
            given.add(face.person_id)
    return owners


def body_of(face: RecognizedFace, detections: Sequence[Detection],
            faces: Sequence[RecognizedFace]) -> int | None:
    """The position of the person box a face belongs to, or None."""
    if face.track_id is not None:
        return next((index for index, d in enumerate(detections)
                     if d.label == "person" and d.track_id == face.track_id), None)
    boxes = boxes_around(face, detections, faces)
    return boxes[0] if boxes else None


def boxes_around(face: RecognizedFace, detections: Sequence[Detection],
                 faces: Sequence[RecognizedFace]) -> list[int]:
    """Person boxes with no face of their own that hold the face's centre, smallest first."""
    tracks_with_a_face = {f.track_id for f in faces if f.track_id is not None}
    middle = centre(face.box)
    found = [index for index, d in enumerate(detections)
             if d.label == "person" and d.track_id not in tracks_with_a_face and contains(d.box, *middle)]
    return sorted(found, key=lambda index: box_area(detections[index].box))


def contains(box: Box, px: float, py: float) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def centre(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


# ---- the photo of a target -----------------------------------------------------------


def area_box(size: tuple[int, int], px: float, py: float) -> Box:
    """Half the frame wide and half the frame high, centred on the point, moved to stay inside."""
    width, height = size
    box_width, box_height = round(width * AREA_SHARE), round(height * AREA_SHARE)
    x1 = slide_inside(round(px - box_width / 2), box_width, width)
    y1 = slide_inside(round(py - box_height / 2), box_height, height)
    return x1, y1, x1 + box_width, y1 + box_height


def slide_inside(start: float, length: float, limit: int) -> float:
    """Move a stretch of `length` so it starts at 0 or later and ends at `limit` or earlier."""
    return max(0, min(start, limit - length))


def photo_box(box: Box, size: tuple[int, int]) -> Box:
    """The part of the frame to cut: the box grown by 15 %, at least 160 pixels, inside the frame."""
    width, height = size
    x1, x2 = photo_span(box[0], box[2], width)
    y1, y2 = photo_span(box[1], box[3], height)
    return x1, y1, x2, y2


def photo_span(start: float, end: float, limit: int) -> tuple[int, int]:
    """One side of photo_box, from left to right or from top to bottom."""
    margin = (end - start) * PHOTO_MARGIN
    start, end = start - margin, end + margin
    short = min(PHOTO_MIN_PIXELS, limit) - (end - start)
    if short > 0:                               # too small: widen it, staying inside the frame
        start = slide_inside(start - short / 2, end - start + short, limit)
        end = start + min(PHOTO_MIN_PIXELS, limit)
    begin = min(max(0, round(start)), limit - 1)
    finish = max(min(limit, round(end)), begin + 1)
    return begin, finish


def cut_out(frame: np.ndarray, box: Box) -> np.ndarray:
    """The pixels inside the box, at their own size (never enlarged)."""
    x1, y1, x2, y2 = box
    return np.ascontiguousarray(frame[y1:y2, x1:x2])


def first_line(error: Exception) -> str:
    """The first line of an error message, or the error's type when it has no message."""
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__
