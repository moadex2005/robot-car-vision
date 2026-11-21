"""The person folders as the page edits them: add a person, change their about.md, remove them.

Each person is one folder under data/people/, named by their id, holding
their photos and an about.md (see car_vision.faces for how it is read). This
module makes those folders and files so nobody has to by hand:

- make_id turns a name into a folder name (small letters, digits, underscores).
- about_text writes an about.md that faces.parse_about reads back exactly.
  Whatever a person types can only end up in its own field: it can never add
  a card line, change the role, or hide text in an HTML comment.
- PeopleBook lists, saves and removes people. Removing moves the folder
  aside into data/people.removed/; nothing is ever deleted.

Every path is built from a checked id inside the photos folder, so no input
can reach a file outside it. Only the standard library and car_vision.faces
are used; taking photos and enrolling are done by car_vision.pipeline.
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from car_vision.faces import (
    ABOUT_FILE,
    BULLET_LINE,
    PERSON_ID,
    ROLE_LINE,
    Person,
    join_paragraphs,
    load_people,
    parse_about,
    photos_in,
)

MAX_NAME = 80
MAX_ROLE = 80
MAX_LINES = 12
MAX_LINE = 120
MAX_STORY = 4000
MAX_PEOPLE = 50

FALLBACK_ID = "person"                 # the id of a name with no usable letters, e.g. one in Arabic
COMMENT_MARKS = ("<!--", "-->")
REMOVED_STAMP = "%Y%m%d-%H%M%S"
TEMPORARY_ABOUT = ".about.md.partial"  # a dot first, so enrolment never lists it as an ignored file
# Folder names Windows does not allow, whatever the letter case.
RESERVED_NAMES = frozenset({"con", "prn", "aux", "nul"}
                           | {f"com{n}" for n in range(1, 10)} | {f"lpt{n}" for n in range(1, 10)}
                           # "photos" is part of the page's addresses (/api/people/photos), so a
                           # person called Photos gets the id photos_2.
                           | {"photos"})


@dataclass(frozen=True)
class PersonEntry:
    """One person as the page lists them."""

    person: Person          # as faces.py reads it from about.md
    photos: int             # jpg/png files in the folder


# ---- ids ---------------------------------------------------------------------------


def make_id(name: str, taken: Collection[str]) -> str:
    """A folder name for a new person, e.g. "Sára Ahmed" -> "sara_ahmed".

    Accents are dropped from letters; spaces, dots and dashes become "_";
    every other character is left out. A name with nothing usable gives
    "person". An id already taken (in any letter case) gets _2, _3, ...
    """
    base = id_letters(name) or FALLBACK_ID
    taken_ids = {str(item).lower() for item in taken} | RESERVED_NAMES
    candidate, number = base, 1
    while candidate in taken_ids:
        number += 1
        candidate = f"{base}_{number}"
    return candidate


def id_letters(name: str) -> str:
    """The small letters, digits and single underscores an id is made of; "" when none are left."""
    plain = unicodedata.normalize("NFKD", str(name)).lower()
    kept = []
    for character in plain:
        if character.isascii() and character.isalnum():
            kept.append(character)
        elif character.isspace() or character in "-_.":
            kept.append("_")
    return re.sub(r"_+", "_", "".join(kept)).strip("_")


def person_folder(photos_dir: Path, person_id: str) -> Path:
    """The folder of an existing person. KeyError for an id that is not valid or has no folder."""
    if not isinstance(person_id, str) or not PERSON_ID.fullmatch(person_id):
        raise KeyError(f"There is no person with the id {person_id!r}.")
    folder = Path(photos_dir) / person_id
    if not folder.is_dir():
        raise KeyError(f"There is no person with the id {person_id!r}.")
    return folder


# ---- about.md ---------------------------------------------------------------------


def about_text(name: str, role: str, lines: Sequence[str], story: str) -> str:
    """The about.md for these details, cleaned so faces.parse_about reads them back as written.

    The cleaning: no line breaks in the name, the role or a card line; no
    HTML comment marks anywhere; empty card lines left out; a story line that
    the reader would take for a card line or a Role line gets a backslash in
    front (as Markdown escapes it), so it stays part of the story.
    """
    name, role, lines, story = clean_name(name), clean_role(role), clean_lines(lines), clean_story(story)
    if not name:
        raise ValueError("The name is needed.")
    # A name that ends in # would lose it: the reader drops closing #s of a heading.
    parts = [f"# {name} #" if name.endswith("#") else f"# {name}"]
    if role:
        parts.append(f"Role: {role}")
    if lines:
        parts.append("")
        parts.extend(f"- {line}" for line in lines)
    if story:
        parts.append("")
        parts.append(story)
    return "\n".join(parts) + "\n"


def without_comment_marks(text: str) -> str:
    """The text with every <!-- and --> taken out, including ones that form as others go."""
    text = str(text)
    while any(mark in text for mark in COMMENT_MARKS):
        for mark in COMMENT_MARKS:
            text = text.replace(mark, "")
    return text


def one_line(text: str) -> str:
    """The text on one line, with no comment marks and no spaces at the ends."""
    return " ".join(without_comment_marks(text).splitlines()).strip()


def clean_name(name: str) -> str:
    return one_line(name)


def clean_role(role: str) -> str:
    """One line, without the spaces and stars at its ends that the reader would drop."""
    return re.sub(r"^[\s*]+|[\s*]+$", "", one_line(role))


def clean_lines(lines: Sequence[str]) -> list[str]:
    """Each card line on one line; empty ones left out."""
    cleaned = (one_line(line) for line in lines)
    return [line for line in cleaned if line]


def clean_story(story: str) -> str:
    """The story with comment marks taken out, card-line and Role lines escaped,
    and blank lines as the reader keeps them (one between paragraphs, none at the ends)."""
    lines = [escaped(line.rstrip()) for line in without_comment_marks(story).splitlines()]
    return join_paragraphs(lines)


def escaped(line: str) -> str:
    """A story line with a backslash before its first letter when the reader would not
    keep it as story (a bullet such as "- x", or a line such as "Role: x")."""
    stripped = line.strip()
    if not (BULLET_LINE.fullmatch(stripped) or ROLE_LINE.fullmatch(stripped)):
        return line
    indent = len(line) - len(line.lstrip())
    return line[:indent] + "\\" + line[indent:]


def checked_fields(name: str, role: str, lines: Sequence[str],
                   story: str) -> tuple[str, str, list[str], str]:
    """The cleaned fields, or ValueError with a plain message when one is the wrong kind or too long."""
    for value, what in ((name, "The name"), (role, "The role"), (story, "The about text")):
        if not isinstance(value, str):
            raise ValueError(f"{what} must be text.")
    if isinstance(lines, str) or not all(isinstance(line, str) for line in lines):
        raise ValueError("The card lines must be a list of lines of text.")
    name, role, lines, story = clean_name(name), clean_role(role), clean_lines(lines), clean_story(story)
    if not name:
        raise ValueError("The name is needed.")
    if len(name) > MAX_NAME:
        raise ValueError(f"The name may be at most {MAX_NAME} characters long.")
    if len(role) > MAX_ROLE:
        raise ValueError(f"The role may be at most {MAX_ROLE} characters long.")
    if len(lines) > MAX_LINES:
        raise ValueError(f"At most {MAX_LINES} card lines, please.")
    if any(len(line) > MAX_LINE for line in lines):
        raise ValueError(f"Each card line may be at most {MAX_LINE} characters long.")
    if len(story) > MAX_STORY:
        raise ValueError(f"The about text may be at most {MAX_STORY} characters long.")
    return name, role, lines, story


def write_about(folder: Path, text: str) -> None:
    """Write folder/about.md as UTF-8 through a temporary file, so it is never half written."""
    temporary = folder / TEMPORARY_ABOUT
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        file.write(text)
    os.replace(temporary, folder / ABOUT_FILE)


# ---- the book of people --------------------------------------------------------------


class PeopleBook:
    """Lists, saves and removes the person folders under photos_dir. Safe to use from several threads."""

    def __init__(self, photos_dir: Path, removed_dir: Path) -> None:
        self.photos_dir = Path(photos_dir)
        self.removed_dir = Path(removed_dir)
        self._lock = threading.Lock()          # one save or remove at a time

    def all(self) -> list[PersonEntry]:
        """Every person with a valid folder, sorted by id."""
        problems: dict[str, str] = {}          # enrolment reports these; the list just leaves them out
        people = load_people(self.photos_dir, problems)
        return [PersonEntry(people[person_id], len(photos_in(self.photos_dir / person_id)))
                for person_id in sorted(people)]

    def save(self, name: str, role: str = "", lines: Sequence[str] = (), story: str = "",
             person_id: str | None = None) -> Person:
        """Make a new person (no person_id) or rewrite that person's about.md; the photos stay.

        ValueError with a plain message for details that break a limit, and
        for a 51st person; KeyError for an unknown person_id.
        """
        fields = checked_fields(name, role, lines, story)
        text = about_text(*fields)
        with self._lock:
            if person_id is None:
                folder = self._new_folder(fields[0])
            else:
                folder = person_folder(self.photos_dir, person_id)
            write_about(folder, text)
        return parse_about(folder.name, text)

    def remove(self, person_id: str) -> None:
        """Move the person's folder to removed_dir/<id>-<YYYYmmdd-HHMMSS>/. Nothing is deleted.

        KeyError for an unknown id.
        """
        with self._lock:
            folder = person_folder(self.photos_dir, person_id)
            self.removed_dir.mkdir(parents=True, exist_ok=True)
            folder.rename(self._removed_place(person_id))

    # ---- helpers ------------------------------------------------------------------

    def _new_folder(self, name: str) -> Path:
        """Make the folder for a new person. Call with the lock held."""
        existing = [p.name for p in self.photos_dir.iterdir()] if self.photos_dir.is_dir() else []
        people = [entry for entry in existing if PERSON_ID.fullmatch(entry)
                  and (self.photos_dir / entry).is_dir()]
        if len(people) >= MAX_PEOPLE:
            raise ValueError(f"There are already {MAX_PEOPLE} people. Remove someone before adding another.")
        folder = self.photos_dir / make_id(name, existing)
        folder.mkdir(parents=True)
        return folder

    def _removed_place(self, person_id: str) -> Path:
        """A new folder name in removed_dir: the id and the time, with -2, -3 ... when that is taken."""
        base = f"{person_id}-{datetime.now().strftime(REMOVED_STAMP)}"
        place, number = self.removed_dir / base, 1
        while place.exists():
            number += 1
            place = self.removed_dir / f"{base}-{number}"
        return place
