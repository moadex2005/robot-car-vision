"""The one shared record of what the camera currently sees.

The vision side (detector, recognizer) writes into a WorldState.
The assistant, the reaction rules and the demo page only read from it.
Nothing else is shared between the two tracks, so each can be built alone.

Boxes are always (x1, y1, x2, y2) in pixels of the frame the model saw.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

Box = tuple[int, int, int, int]

UNKNOWN = "Unknown"


@dataclass(frozen=True)
class Detection:
    """One object found by the detector in one frame."""

    label: str            # class name, e.g. "person", "chair"
    confidence: float     # 0.0 to 1.0
    box: Box
    track_id: int | None = None   # same number while the tracker follows the object


@dataclass(frozen=True)
class RecognizedFace:
    """One face found by the recognizer in one frame."""

    name: str                     # display name, or UNKNOWN
    similarity: float             # cosine similarity to the best enrolled match
    box: Box
    person_id: str | None = None  # the person's folder name; None when unknown
    track_id: int | None = None   # track of the person box this face sits in

    @property
    def is_known(self) -> bool:
        return self.person_id is not None


@dataclass(frozen=True)
class WorldSnapshot:
    """A consistent copy of the state at one moment. Safe to keep and read."""

    detections: tuple[Detection, ...] = ()
    faces: tuple[RecognizedFace, ...] = ()
    frame_size: tuple[int, int] = (0, 0)   # (width, height)
    detections_at: float = 0.0             # time.monotonic() of the last update
    faces_at: float = 0.0

    def known_faces(self) -> tuple[RecognizedFace, ...]:
        return tuple(f for f in self.faces if f.is_known)

    def people(self) -> tuple[Detection, ...]:
        return tuple(d for d in self.detections if d.label == "person")


@dataclass
class WorldState:
    """Thread-safe holder. Writers replace whole lists; readers take snapshots."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _snapshot: WorldSnapshot = field(default_factory=WorldSnapshot)
    _memory: tuple[str, ...] = field(default=(), repr=False)

    def update_detections(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int],
        now: float | None = None,
    ) -> None:
        stamp = time.monotonic() if now is None else now
        with self._lock:
            old = self._snapshot
            self._snapshot = WorldSnapshot(
                detections=tuple(detections),
                faces=old.faces,
                frame_size=frame_size,
                detections_at=stamp,
                faces_at=old.faces_at,
            )

    def update_faces(self, faces: list[RecognizedFace], now: float | None = None) -> None:
        stamp = time.monotonic() if now is None else now
        with self._lock:
            old = self._snapshot
            self._snapshot = WorldSnapshot(
                detections=old.detections,
                faces=tuple(faces),
                frame_size=old.frame_size,
                detections_at=old.detections_at,
                faces_at=stamp,
            )

    def snapshot(self) -> WorldSnapshot:
        with self._lock:
            return self._snapshot

    def set_memory(self, lines: Sequence[str]) -> None:
        """Replace the lines the camera remembers (newest first), for the assistant."""
        copied = tuple(lines)
        with self._lock:
            self._memory = copied

    def memory(self) -> tuple[str, ...]:
        """The lines the camera remembers, newest first; empty until the pipeline sets them."""
        with self._lock:
            return self._memory

    def current_people(self, max_age_seconds: float = 3.0, now: float | None = None) -> list[str]:
        """Names of enrolled people seen within the last few seconds, no repeats.

        This is what the assistant is told, so "who is this?" needs no typed name.
        """
        stamp = time.monotonic() if now is None else now
        snap = self.snapshot()
        if stamp - snap.faces_at > max_age_seconds:
            return []
        names: list[str] = []
        for face in snap.known_faces():
            if face.name not in names:
                names.append(face.name)
        return names
