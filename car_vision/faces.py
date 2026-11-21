"""Who the team is, their saved face vectors, and matching a new face against them.

Pure logic with no camera and no model, so all of it can be tested with
made-up numbers:

- load_people reads data/people.json, the hand-written info cards.
- save_embeddings and load_embeddings keep the face vectors made by enroll.
- FaceMatcher tells which enrolled person a face vector belongs to, if any.
- FaceMemory keeps faces on screen between the frames where faces are checked.
"""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from car_vision.config import ConfigError
from car_vision.state import Box, Detection, RecognizedFace

ENROLL_COMMAND = r".venv\Scripts\python.exe -m car_vision.enroll"


# ---- the people file --------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """One enrolled person and the text on their info card."""

    person_id: str
    name: str
    role: str = ""
    lines: tuple[str, ...] = ()


def load_people(path: Path) -> dict[str, Person]:
    """Read the people file. Every mistake in it becomes a ConfigError that names the file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"People file not found: {path}\n"
            "Copy data/people.example.json to data/people.json and write in your team."
        )
    try:
        # utf-8-sig also accepts the invisible mark Windows Notepad may put at the start.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"Could not read {path.name}: {error}") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"{path.name} is not valid JSON: {error.msg} at line {error.lineno}, column {error.colno}. "
            "Check for a missing comma or quote, or a comma after the last item."
        ) from error
    if not isinstance(data, dict):
        raise ConfigError(
            f'{path.name} must be one object of people, like {{"sara": {{"name": "Sara Ahmed"}}}}'
        )
    return {person_id: _person_from(path.name, person_id, entry) for person_id, entry in data.items()}


def _person_from(file_name: str, person_id: str, entry: object) -> Person:
    where = f"{file_name}, person '{person_id}'"
    if not person_id.strip():
        raise ConfigError(f"{file_name} has a person with an empty id")
    if not isinstance(entry, dict):
        raise ConfigError(f'{where}: must be an object like {{"name": "Sara Ahmed"}}')
    if "name" not in entry:
        raise ConfigError(f"{where}: 'name' is missing")
    name = entry["name"]
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{where}: 'name' must be text, got {name!r}")
    role = entry.get("role", "")
    if not isinstance(role, str):
        raise ConfigError(f"{where}: 'role' must be text, got {role!r}")
    lines = entry.get("lines", [])
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        raise ConfigError(f"{where}: 'lines' must be a list of text, like [\"Built the chassis\"]")
    return Person(person_id=person_id, name=name.strip(), role=role.strip(),
                  lines=tuple(line.strip() for line in lines))


# ---- the embeddings file ------------------------------------------------------


def save_embeddings(path: Path, ids: list[str], embeddings: np.ndarray) -> None:
    """Write one person id and one face vector per row into an .npz file."""
    path = Path(path)
    rows = np.asarray(embeddings, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] != len(ids):
        raise ValueError(f"need one embedding row per id: {len(ids)} ids, array of shape {rows.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write a temporary file first, so a crash never leaves a half-written file behind.
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez(temporary, ids=np.array(ids, dtype=str), embeddings=rows)
    os.replace(temporary, path)


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Read the file written by save_embeddings."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"No saved faces at {path}. Run the enroll command first:\n  {ENROLL_COMMAND}")
    try:
        with np.load(path, allow_pickle=False) as data:
            ids = [str(value) for value in data["ids"]]
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ConfigError(
            f"{path.name} is damaged or not an embeddings file ({error}). "
            f"Run the enroll command again:\n  {ENROLL_COMMAND}"
        ) from error
    if len(embeddings) != len(ids):
        raise ConfigError(
            f"{path.name} has {len(ids)} ids but {len(embeddings)} face vectors. "
            f"Run the enroll command again:\n  {ENROLL_COMMAND}"
        )
    return ids, embeddings


# ---- matching -----------------------------------------------------------------


def normalized(vector: np.ndarray) -> np.ndarray | None:
    """The vector scaled to length 1, or None for a zero or broken vector."""
    flat = np.asarray(vector, dtype=np.float32).ravel()
    length = float(np.linalg.norm(flat))
    if flat.size == 0 or not np.isfinite(length) or length == 0.0:
        return None
    return flat / length


class FaceMatcher:
    """Compares a face vector with every enrolled vector and names the closest person."""

    def __init__(self, ids: list[str], embeddings: np.ndarray, threshold: float) -> None:
        self.threshold = float(threshold)
        rows = np.asarray(embeddings, dtype=np.float32)
        if len(ids) == 0 or rows.size == 0:
            self._ids: list[str] = []
            self._rows = np.zeros((0, 0), dtype=np.float32)
            return
        if rows.ndim != 2 or rows.shape[0] != len(ids):
            raise ValueError(f"need one embedding row per id: {len(ids)} ids, array of shape {rows.shape}")
        lengths = np.linalg.norm(rows, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0                    # a zero row stays zero and never matches
        self._ids = list(ids)
        self._rows = rows / lengths

    @property
    def person_ids(self) -> set[str]:
        """Every person id that has at least one stored row."""
        return set(self._ids)

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """(person_id, score) when the best person scores at least the threshold, else (None, score)."""
        if not self._ids:
            return None, 0.0
        vector = normalized(embedding)
        if vector is None:
            return None, 0.0
        if vector.shape[0] != self._rows.shape[1]:
            raise ValueError(
                f"face vector has {vector.shape[0]} numbers but the saved ones have "
                f"{self._rows.shape[1]}; run the enroll command again"
            )
        similarities = self._rows @ vector
        best_by_person: dict[str, float] = {}
        for person_id, similarity in zip(self._ids, similarities):
            best_by_person[person_id] = max(best_by_person.get(person_id, -1.0), float(similarity))
        best_id = max(best_by_person, key=best_by_person.get)
        best_score = best_by_person[best_id]
        if best_score >= self.threshold:
            return best_id, best_score
        return None, best_score


# ---- keeping faces on screen between face checks ------------------------------


def person_boxes(detections: Sequence[Detection]) -> dict[int, Box]:
    """Box of every tracked person, by track number."""
    return {d.track_id: d.box for d in detections if d.label == "person" and d.track_id is not None}


def shifted(box: Box, old_anchor: Box, new_anchor: Box) -> Box:
    """Move box the way the anchor box moved.

    Sideways it follows the middle of the person box; up and down it follows
    the top edge, where the head is, so a box that grows downward as a person
    walks closer does not drag the face down with it.
    """
    dx = round((new_anchor[0] + new_anchor[2]) / 2 - (old_anchor[0] + old_anchor[2]) / 2)
    dy = new_anchor[1] - old_anchor[1]
    x1, y1, x2, y2 = box
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


class FaceMemory:
    """Keeps the last recognised faces on screen between face checks."""

    def __init__(self, max_age_seconds: float = 1.0) -> None:
        self.max_age_seconds = max_age_seconds
        # Each entry: the face, the box of its person track when it was seen, and when.
        self._faces: list[tuple[RecognizedFace, Box | None, float]] = []

    def remember(self, faces: Sequence[RecognizedFace], detections: Sequence[Detection], now: float) -> None:
        """Store the faces of a fresh face check. The faces stored before are forgotten."""
        boxes = person_boxes(detections)
        self._faces = [
            (face, boxes.get(face.track_id) if face.track_id is not None else None, now)
            for face in faces
        ]

    def current(self, detections: Sequence[Detection], now: float) -> list[RecognizedFace]:
        """The remembered faces, moved along with their person boxes in this frame."""
        boxes = person_boxes(detections)
        kept: list[tuple[RecognizedFace, Box | None, float]] = []
        result: list[RecognizedFace] = []
        for face, anchor, seen_at in self._faces:
            if face.track_id is not None:
                new_anchor = boxes.get(face.track_id)
                if new_anchor is None:
                    continue                           # the person left: forget the face
                box = face.box if anchor is None else shifted(face.box, anchor, new_anchor)
                result.append(RecognizedFace(face.name, face.similarity, box,
                                             face.person_id, face.track_id))
            else:
                if now - seen_at > self.max_age_seconds:
                    continue                           # too old to trust
                result.append(face)
            kept.append((face, anchor, seen_at))
        self._faces = kept
        return result
