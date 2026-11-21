"""Turns each person's photos into saved face vectors, once, before the live view.

Every sub-folder of data/people/ is one person; the folder name is their id
(small letters, digits, underscores). Put 3 to 10 clear photos of each person
in their folder, next to their about.md, then run:

    .venv\\Scripts\\python.exe -m car_vision.enroll

A photo is used only when one face stands out in it. Every skipped photo is
listed with the reason. The face vectors are written to data/embeddings.npz;
they are never printed.

This module and car_vision.recognizer are the only ones that use insightface;
the face model is loaded with recognizer.load_face_model, only when there are
photos to read.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from car_vision.config import ConfigError, load_config, require, resolve_path, section
from car_vision.faces import ABOUT_FILE, Person, has_about_text, load_people, normalized, save_embeddings
from car_vision.recognizer import FaceModelError, embedding_of, load_face_model

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png")
SYSTEM_FILES = ("thumbs.db", "desktop.ini")
ENOUGH_PHOTOS = 3
CLEAR_WINNER = 2.0        # the largest face must be at least twice the area of the next


def photos_in(folder: Path) -> list[Path]:
    """Every jpg, jpeg or png in the folder, any letter case, in name order."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES)


def read_photo(path: Path) -> np.ndarray | None:
    """The photo as a BGR image, or None when it cannot be read.

    np.fromfile + imdecode also works for folder names with letters outside
    English, which cv2.imread cannot open on Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error):
        return None
    return image if image is not None and image.size else None


def box_area(face: Any) -> float:
    x1, y1, x2, y2 = (float(v) for v in np.asarray(face.bbox).ravel()[:4])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def pick_face(faces: list[Any]) -> tuple[Any | None, str]:
    """The one face to use from a photo, or None and the reason to skip it."""
    if not faces:
        return None, "no face found"
    if len(faces) == 1:
        return faces[0], ""
    by_size = sorted(faces, key=box_area, reverse=True)
    if box_area(by_size[0]) >= CLEAR_WINNER * box_area(by_size[1]):
        return by_size[0], ""
    return None, f"several faces ({len(faces)}) and none clearly the largest"


def enroll(settings: dict, face_model: Any = None) -> dict[str, int]:
    """Encode every usable photo and save the vectors. Returns person id -> photos used.

    The people are the sub-folders of photos_dir.
    """
    photos_dir = resolve_path(require(settings, "photos_dir", str))
    embeddings_path = resolve_path(require(settings, "embeddings_file", str))
    problems: dict[str, str] = {}             # folder name -> what is wrong with it
    people = load_people(photos_dir, problems)

    found = {person_id: photos_in(photos_dir / person_id) for person_id in people}
    if not photos_dir.is_dir():
        print(f"The photo folder does not exist: {photos_dir}")
    elif not people:
        print(f"There are no person folders in {photos_dir}.")
    print_ignored_files(photos_dir, people)

    ids: list[str] = []
    rows: list[np.ndarray] = []
    used = {person_id: 0 for person_id in people}
    if any(found.values()):
        if face_model is None:
            face_model = load_face_model(settings)
        for person_id, photos in found.items():
            for photo in photos:
                vector, reason = encode_photo(face_model, photo)
                if vector is None:
                    print(f"Skipped {person_id}/{photo.name}: {reason}")
                    continue
                ids.append(person_id)
                rows.append(vector)
                used[person_id] += 1

    if people:
        print_table(people, found, used)
    print_about_notes(people, problems)

    if rows:
        save_embeddings(embeddings_path, ids, np.vstack(rows))
        enrolled = sum(1 for count in used.values() if count > 0)
        print(f"Saved {len(rows)} face vectors for {enrolled} people to {embeddings_path}")
    else:
        print(f"Nobody was enrolled, so {embeddings_path.name} was not written.\n"
              f"Put 3 to 10 clear photos of each person in {photos_dir}\\<person id>\\ "
              "(jpg or png), one face per photo, then run this again.")
        remove_old_embeddings(embeddings_path)
    return used


def remove_old_embeddings(path: Path) -> None:
    """Delete faces saved by an earlier run, so people who were removed are not still recognised."""
    if not path.is_file():
        return
    try:
        path.unlink()
    except OSError as error:
        print(f"Could not remove the old {path.name} ({error.strerror or error}); "
              "delete it by hand, or people from an earlier run are still recognised.")
        return
    print(f"Removed the old {path.name}, so nobody is recognised until people are enrolled again.")


def encode_photo(face_model: Any, photo: Path) -> tuple[np.ndarray | None, str]:
    """The normalised face vector of one photo, or None and the reason it was skipped."""
    image = read_photo(photo)
    if image is None:
        return None, "unreadable (not a jpg or png image, or damaged)"
    try:
        faces = list(face_model.get(image))
    except Exception:                                # one strange photo must not stop the run
        return None, "could not be processed"
    face, reason = pick_face(faces)
    if face is None:
        return None, reason
    raw = embedding_of(face)
    vector = None if raw is None else normalized(raw)
    if vector is None:
        return None, "no face found (the face could not be encoded)"
    return vector, ""


def print_table(people: dict, found: dict[str, list[Path]], used: dict[str, int]) -> None:
    width = max([len("person")] + [len(person_id) for person_id in people])
    print()
    print(f"{'person':<{width}}  photos found  photos used")
    for person_id in people:
        print(f"{person_id:<{width}}  {len(found[person_id]):>12}  {used[person_id]:>11}")
    print()
    for person_id in people:
        if used[person_id] == 0:
            reason = "no photos found" if not found[person_id] else "no usable photo"
            print(f"Not enrolled: {person_id} ({reason})")
        elif used[person_id] < ENOUGH_PHOTOS:
            print(f"Warning: {person_id} has only {used[person_id]} usable photo(s); "
                  f"{ENOUGH_PHOTOS} or more make recognition more reliable")


def ignored_files(folder: Path) -> list[str]:
    """Names of the files in a person folder that enrol does not read.

    about.md and the hidden files Windows makes (Thumbs.db, desktop.ini) are left out.
    """
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() not in PHOTO_SUFFIXES
                  and p.name != ABOUT_FILE and p.name.lower() not in SYSTEM_FILES
                  and not p.name.startswith("."))


def print_ignored_files(photos_dir: Path, people: dict[str, Person]) -> None:
    """One line per person with files in other formats, such as .heic or .webp."""
    for person_id in people:
        names = ignored_files(photos_dir / person_id)
        if names:
            print(f"Ignored (not jpg or png): {', '.join(f'{person_id}/{name}' for name in names)}")


def print_about_notes(people: dict[str, Person], problems: dict[str, str]) -> None:
    """A note for each folder or about.md with a problem, and for each about.md missing or empty."""
    for problem in problems.values():
        print(f"Note: {problem}")
    for person_id, person in people.items():
        if not has_about_text(person) and person_id not in problems:
            print(f"Note: {person_id} has no about.md (or an empty one), so they are shown "
                  "by folder name, with no card text.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn the photos in data/people/ into saved face vectors.")
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        used = enroll(section(load_config(), "recognizer"))
    except ConfigError as error:
        print(f"Cannot enroll: {error}")
        return 1
    except FaceModelError as error:
        print(f"Cannot start: {error}")
        return 1
    return 0 if any(count > 0 for count in used.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
