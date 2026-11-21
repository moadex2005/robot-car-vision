"""Who the team is, their saved face vectors, and matching a new face against them.

Pure logic with no camera and no model, so all of it can be tested with
made-up numbers:

- load_people reads the person folders under data/people/: each folder is one
  person, described by an optional hand-written about.md.
- save_embeddings and load_embeddings keep the face vectors made by enroll.
- FaceMatcher tells which enrolled person a face vector belongs to, if any.
- FaceMemory keeps faces on screen between the frames where faces are checked.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from car_vision.config import ConfigError
from car_vision.state import Box, Detection, RecognizedFace

log = logging.getLogger(__name__)

ENROLL_COMMAND = r".venv\Scripts\python.exe -m car_vision.enroll"
ABOUT_FILE = "about.md"
PERSON_ID = re.compile(r"[a-z0-9_]+")            # a folder name: small letters, digits, underscores
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# "Role: x", "Role : x", "**Role:** x" and "**Role**: x", in any letter case.
ROLE_LINE = re.compile(r"\*{0,2}role\s*\*{0,2}\s*:\s*\*{0,2}\s*(.*)", re.IGNORECASE)
# A bullet: -, *, + or a bullet sign, then a space or a tab. A bare mark is an empty bullet.
BULLET_LINE = re.compile(r"[-*+•](?:[ \t]+(.*))?")


# ---- the person folders ------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """One person the camera may recognise, and what the team wrote about them."""

    person_id: str
    name: str
    role: str = ""
    lines: tuple[str, ...] = ()     # every bullet line of about.md, in order
    story: str = ""                 # the text that is left: no heading, no Role line, no bullets
    brief: str = ""                 # the whole about.md as written


def load_people(photos_dir: Path, problems: dict[str, str] | None = None) -> dict[str, Person]:
    """One Person per sub-folder of photos_dir, in name order.

    A missing photos_dir gives nobody. A folder without about.md is a person
    known only by the folder name.

    Two problems do not stop the others, and each is reported once:
    - a folder name that is not a valid id: the folder is skipped;
    - an about.md that cannot be read (for example, not saved as UTF-8): that
      person is known only by the folder name.
    The report goes into `problems` (folder name -> a full sentence for the
    user) when the caller gives one, else into the log as a warning.
    """
    photos_dir = Path(photos_dir)
    if not photos_dir.is_dir():
        return {}
    people: dict[str, Person] = {}
    for folder in sorted(p for p in photos_dir.iterdir() if p.is_dir()):
        if not PERSON_ID.fullmatch(folder.name):
            report(problems, folder.name,
                   f"Skipped the folder '{folder.name}' in {photos_dir}: a person folder name may only "
                   "have small letters, digits and underscores, like sara_ahmed. Rename it to use it.")
            continue
        people[folder.name] = read_person(folder, problems)
    return people


def read_person(folder: Path, problems: dict[str, str] | None = None) -> Person:
    """The Person described by folder/about.md, or just the folder name when there is
    none or it cannot be read (see load_people for where the problem goes)."""
    about = folder / ABOUT_FILE
    if not about.is_file():
        return Person(person_id=folder.name, name=folder.name)
    try:
        text = read_about(about)
    except ConfigError as error:
        report(problems, folder.name,
               f"{error} Until then {folder.name} is shown by folder name, with no card text.")
        return Person(person_id=folder.name, name=folder.name)
    return parse_about(folder.name, text)


def report(problems: dict[str, str] | None, folder_name: str, message: str) -> None:
    """Put a problem in the caller's dict, or in the log when there is none."""
    if problems is None:
        log.warning("%s", message)
    else:
        problems[folder_name] = message


def has_about_text(person: Person) -> bool:
    """True when the person's about.md says something besides HTML comments."""
    return bool(HTML_COMMENT.sub("", person.brief).strip())


def read_about(path: Path) -> str:
    """The text of an about.md, read as UTF-8 and accepting a byte-order mark."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        problem = ConfigError(f"{path} is not saved as UTF-8. Open it in a text editor "
                              "and save it again with the UTF-8 encoding.")
    except OSError as error:
        problem = ConfigError(f"Could not read {path}: {error.strerror or error}")
    raise problem from None


def parse_about(person_id: str, text: str) -> Person:
    """Build a Person from the text of an about.md.

    The first '# ' heading is the name (else the person id), the first
    'Role:' line is the role, bullet lines (-, *, + or a bullet sign) are the
    card lines, and the rest is the story. HTML comments are left out of all
    of them.
    """
    name, role, role_seen = "", "", False
    lines: list[str] = []
    story: list[str] = []
    for line in HTML_COMMENT.sub("", text).splitlines():
        stripped = line.strip()
        role_match = ROLE_LINE.fullmatch(stripped)
        bullet_match = BULLET_LINE.fullmatch(stripped)
        if not name and heading_text(line):
            name = heading_text(line)
        elif role_match and not role_seen:
            role, role_seen = role_match.group(1).strip(" *"), True
        elif bullet_match:
            if (bullet_match.group(1) or "").strip():
                lines.append(bullet_match.group(1).strip())
        else:
            story.append(line.rstrip())
    return Person(person_id=person_id, name=name or person_id, role=role,
                  lines=tuple(lines), story=join_paragraphs(story), brief=text)


def heading_text(line: str) -> str:
    """The text of a '# ' heading without closing #s, or '' when the line is not one."""
    if not line.startswith("# "):
        return ""
    return line[2:].strip().rstrip("#").strip()


def join_paragraphs(lines: list[str]) -> str:
    """The lines as text, with one blank line between paragraphs and none at the ends."""
    paragraphs: list[list[str]] = [[]]
    for line in lines:
        if line:
            paragraphs[-1].append(line)
        elif paragraphs[-1]:
            paragraphs.append([])
    return "\n\n".join("\n".join(paragraph) for paragraph in paragraphs if paragraph)


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


def load_embeddings(path: Path, vector_size: int | None = None) -> tuple[list[str], np.ndarray]:
    """Read the file written by save_embeddings.

    With vector_size, every stored face vector must have that many numbers
    (the size the face model makes), else a ConfigError says to enroll again.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"No saved faces at {path}. Run the enroll command first:\n  {ENROLL_COMMAND}")
    try:
        with np.load(path, allow_pickle=False) as data:
            ids = [str(value) for value in data["ids"]]
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    except (OSError, EOFError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        raise ConfigError(
            f"{path.name} is damaged or not an embeddings file ({error}). "
            f"Run the enroll command again:\n  {ENROLL_COMMAND}"
        ) from error
    if embeddings.ndim != 2:
        raise ConfigError(f"{path.name} does not hold one face vector per row. "
                          f"Run the enroll command again:\n  {ENROLL_COMMAND}")
    if len(embeddings) != len(ids):
        raise ConfigError(
            f"{path.name} has {len(ids)} ids but {len(embeddings)} face vectors. "
            f"Run the enroll command again:\n  {ENROLL_COMMAND}"
        )
    if vector_size is not None and ids and embeddings.shape[1] != vector_size:
        raise ConfigError(
            f"{path.name} holds face vectors of {embeddings.shape[1]} numbers, but the face model "
            f"makes {vector_size}. Run the enroll command again:\n  {ENROLL_COMMAND}"
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


TRACK_GAP_SECONDS = 0.3      # how long a face waits for its person track to come back


@dataclass
class RememberedFace:
    """One face kept by FaceMemory, and what it needs to follow its person box."""

    face: RecognizedFace
    anchor: Box | None              # the person box when the face was seen
    seen_at: float
    box: Box                        # where the face was last shown
    track_lost_at: float | None = None


class FaceMemory:
    """Keeps the last recognised faces on screen between face checks."""

    def __init__(self, max_age_seconds: float = 1.0) -> None:
        self.max_age_seconds = max_age_seconds
        self._faces: list[RememberedFace] = []

    def remember(self, faces: Sequence[RecognizedFace], detections: Sequence[Detection], now: float) -> None:
        """Store the faces of a fresh face check. The faces stored before are forgotten."""
        boxes = person_boxes(detections)
        self._faces = [
            RememberedFace(face, boxes.get(face.track_id) if face.track_id is not None else None,
                           now, face.box)
            for face in faces
        ]

    def current(self, detections: Sequence[Detection], now: float) -> list[RecognizedFace]:
        """The remembered faces, moved along with their person boxes in this frame."""
        boxes = person_boxes(detections)
        self._faces = [entry for entry in self._faces if self._still_valid(entry, boxes, now)]
        return [RecognizedFace(e.face.name, e.face.similarity, e.box, e.face.person_id, e.face.track_id)
                for e in self._faces]

    def _still_valid(self, entry: RememberedFace, boxes: dict[int, Box], now: float) -> bool:
        """Move the entry with its person box; False when it should be forgotten."""
        if entry.face.track_id is None:
            return now - entry.seen_at <= self.max_age_seconds      # else too old to trust
        new_anchor = boxes.get(entry.face.track_id)
        if new_anchor is None:
            # The track is missing: keep the face where it was for a moment, in
            # case the detector only missed the person in one or two frames.
            if entry.track_lost_at is None:
                entry.track_lost_at = now
            return now - entry.track_lost_at <= TRACK_GAP_SECONDS
        entry.track_lost_at = None
        if entry.anchor is not None:
            entry.box = shifted(entry.face.box, entry.anchor, new_anchor)
        return True
