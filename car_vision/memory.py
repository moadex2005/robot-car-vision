"""The camera's memory: what was seen, and when.

The pipeline tells a SightingLog about every tracked thing in every processed
frame ("a bottle, track 7, on the left"). The log turns that stream of frames
into a few rows, one per visit, kept in one SQLite file on this laptop.

- A *candidate* is a thing that has just come into view. It lives only in
  memory, never in the file.
- Once a candidate has stayed in view for ``min_seconds`` it becomes a
  *sighting*: a row with when it was first and last seen, where in the picture
  it was, and the detector's best confidence. While the thing stays in view the
  row is extended. When it has not been seen for 3 seconds the row is final; if
  the thing comes back after that, it is a new sighting.

Why a short flicker is not remembered: a detector sometimes gives a wrong label
for a frame or two (a cushion called "cat"). That label never stays for a full
``min_seconds``, so it remains a candidate and is forgotten without a trace.

Threads: the video loop calls ``seen`` many times a second; the web server
calls ``recent``, ``picture``, ``summary`` and ``clear``. They share one SQLite
connection guarded by one lock. ``seen`` mostly changes things in memory: the
file is written at most once every 2 seconds, in one short transaction, so the
video never waits long for the disk.

Failure is quiet: when the file cannot be created, is locked or is broken, the
problem is logged once, the video goes on and reads give empty results. A
broken file is moved aside as ``memory.sqlite3.corrupt-<date>-<time>`` and a
fresh one is started.

Pictures are kept only for objects, never for people. Only the standard
library is used.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KINDS = ("object", "person")
PLACE_WORDS = {"left": "on the left", "middle": "in the middle", "right": "on the right"}

WRITE_EVERY_SECONDS = 2.0       # the file is written at most this often
ROW_GAP_SECONDS = 3.0           # a sighting not seen for this long is final
CANDIDATE_GAP_SECONDS = 1.0     # a candidate not seen for this long is forgotten
DELETE_OLD_EVERY_SECONDS = 3600.0
MAX_PICTURE_BYTES = 40_000      # larger pictures are refused quietly
MAX_FOLLOWED = 200              # things followed in memory at once, at most
BUSY_TIMEOUT_SECONDS = 0.1      # how long to wait for a file locked by another program

MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused, even after deletes
    kind        TEXT NOT NULL,                      -- "object" or "person"
    label       TEXT NOT NULL,                      -- "person" for people
    name        TEXT,                               -- the recognised person's name, else NULL
    first_seen  REAL NOT NULL,                      -- seconds since 1970
    last_seen   REAL NOT NULL,
    place       TEXT NOT NULL,                      -- "left", "middle" or "right"
    confidence  REAL NOT NULL                       -- the best one during the sighting
);
CREATE INDEX IF NOT EXISTS sightings_by_last_seen ON sightings (last_seen);
CREATE INDEX IF NOT EXISTS sightings_by_thing ON sightings (kind, label, name, last_seen);
CREATE TABLE IF NOT EXISTS pictures (
    sighting_id INTEGER PRIMARY KEY,                -- the id of the sighting it shows
    jpeg        BLOB NOT NULL
);
"""


@dataclass(frozen=True)
class Sighting:
    """One remembered visit of one thing, as stored in the file."""

    id: int
    kind: str              # "object" or "person"
    label: str             # the detector's label or the typed name; "person" for people
    name: str | None       # the recognised person's display name, else None
    first_seen: float      # seconds since 1970, as time.time() gives
    last_seen: float
    place: str             # "left", "middle" or "right" of the picture when last seen
    confidence: float
    has_picture: bool


@dataclass
class _Followed:
    """A thing the log follows right now: a candidate, or a sighting still open."""

    kind: str
    label: str
    name: str | None
    first_seen: float
    last_seen: float
    place: str
    confidence: float
    is_row: bool = False             # False while it is only a candidate
    row_id: int | None = None        # set once the row is in the file
    changed: bool = False            # something new to write to the file
    picture: bytes | None = None     # a picture waiting to be written
    picture_confidence: float = -1.0


class SightingLog:
    """Remembers what the camera saw, in one SQLite file. Safe to use from several threads."""

    def __init__(self, path: Path, keep_days: float = 7, min_seconds: float = 1.0) -> None:
        if keep_days <= 0:
            raise ValueError("keep_days must be more than 0")
        if min_seconds < 0:
            raise ValueError("min_seconds must be 0 or more")
        self.path = Path(path)
        self.keep_seconds = float(keep_days) * DAY
        self.min_seconds = float(min_seconds)
        self._lock = threading.Lock()
        self._followed: dict[Hashable, _Followed] = {}
        self._finished: list[_Followed] = []     # final rows not yet written
        self._written_at: float | None = None
        self._old_deleted_at: float | None = None
        self._warned: set[str] = set()
        self._closed = False
        self._db = self._open()
        start_up = time.time()
        self._use_db("deleting old sightings", lambda db: self._delete_old(db, start_up), None)

    # ----- what the video loop calls -----

    def seen(self, key: Hashable, kind: str, label: str, name: str | None, place: str,
             confidence: float, now: float, picture: bytes | None = None) -> None:
        """Note that a thing is in view now. Call it for every frame the thing is in.

        ``key`` stays the same while the tracker follows the same thing. A
        picture is kept only for objects and only up to 40 kB.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be 'object' or 'person', not {kind!r}")
        if place not in PLACE_WORDS:
            raise ValueError(f"place must be 'left', 'middle' or 'right', not {place!r}")
        picture = _usable_picture(kind, picture)
        with self._lock:
            if self._closed:
                return
            thing = self._followed.get(key)
            if thing is not None and self._has_gone(thing, now):
                self._let_go(key)
                thing = None
            if thing is None:
                thing = self._start(key, kind, label, place, confidence, now)
            self._extend(thing, name, place, float(confidence), now, picture)
            if self._write_due(now):
                self._write_and_tidy(now)

    # ----- what the web server and the assistant call -----

    def recent(self, limit: int = 50, text: str | None = None) -> list[Sighting]:
        """The newest sightings first. ``text`` keeps those whose label or name contains it.

        The file is at most 2 seconds behind what is in view right now.
        """
        limit = max(0, int(limit))
        needle = (text or "").strip().casefold()
        if limit == 0:
            return []
        query = ("SELECT s.id, s.kind, s.label, s.name, s.first_seen, s.last_seen, s.place,"
                 " s.confidence, EXISTS (SELECT 1 FROM pictures p WHERE p.sighting_id = s.id)"
                 " FROM sightings s")
        values: tuple[Any, ...] = ()
        if needle:
            query += " WHERE instr(folded(s.label), ?) > 0 OR instr(folded(s.name), ?) > 0"
            values = (needle, needle)
        query += " ORDER BY s.last_seen DESC, s.id DESC LIMIT ?"
        with self._lock:
            rows = self._use_db("reading the memory",
                                lambda db: db.execute(query, (*values, limit)).fetchall(), [])
        return [_sighting(row) for row in rows]

    def picture(self, sighting_id: int) -> bytes | None:
        """The small JPEG of one sighting, or None when there is none."""
        query = "SELECT jpeg FROM pictures WHERE sighting_id = ?"
        with self._lock:
            row = self._use_db("reading a picture",
                               lambda db: db.execute(query, (int(sighting_id),)).fetchone(), None)
        return bytes(row[0]) if row else None

    def summary(self, now: float, limit: int = 30) -> list[str]:
        """Plain lines for the assistant, newest first, one per thing or person.

        Example: ``laptop: last seen 3 minutes ago on the left, in view for 2 minutes``.
        """
        # With a single MAX() in a GROUP BY, SQLite takes the other plain columns
        # (first_seen, place) from the row that has that latest last_seen.
        query = ("SELECT kind, label, name, first_seen, MAX(last_seen) AS latest, place"
                 " FROM sightings GROUP BY kind, label, name"
                 " ORDER BY latest DESC LIMIT ?")
        with self._lock:
            rows = self._use_db("reading the memory",
                                lambda db: db.execute(query, (max(0, int(limit)),)).fetchall(), [])
        return [_summary_line(*row, now=now) for row in rows]

    def clear(self) -> None:
        """Forget everything: the file is emptied and nothing in view is carried over."""
        with self._lock:
            self._followed.clear()
            self._finished.clear()
            self._use_db("emptying the memory", _delete_everything, None)
            # Also empty the write-ahead file, so the old rows do not linger on disk.
            self._use_db("emptying the memory",
                         lambda db: db.execute("PRAGMA wal_checkpoint(TRUNCATE)"), None)

    def close(self) -> None:
        """Write what is still open and close the file. Later calls do nothing."""
        with self._lock:
            if self._closed:
                return
            for key in list(self._followed):
                self._let_go(key)
            self._write_rows()
            self._closed = True
            if self._db is not None:
                _close_quietly(self._db)
                self._db = None

    # ----- following things in memory -----

    def _start(self, key: Hashable, kind: str, label: str, place: str,
               confidence: float, now: float) -> _Followed:
        """Begin a new candidate, making room first when too many are followed."""
        if len(self._followed) >= MAX_FOLLOWED:
            stalest = min(self._followed, key=lambda k: self._followed[k].last_seen)
            self._let_go(stalest)
        thing = _Followed(kind, label, None, now, now, place, float(confidence))
        self._followed[key] = thing
        return thing

    def _extend(self, thing: _Followed, name: str | None, place: str, confidence: float,
                now: float, picture: bytes | None) -> None:
        """Add one more moment in view to a candidate or an open sighting."""
        thing.last_seen = max(thing.last_seen, now)
        thing.place = place
        thing.confidence = max(thing.confidence, confidence)
        if name:
            thing.name = name          # a name recognised part-way stays from then on
        if picture is not None and confidence > thing.picture_confidence:
            thing.picture = picture    # keep the picture taken when the detector was surest
            thing.picture_confidence = confidence
        thing.changed = True
        if not thing.is_row and thing.last_seen - thing.first_seen >= self.min_seconds:
            thing.is_row = True        # stayed long enough: from now on it is remembered

    def _has_gone(self, thing: _Followed, now: float) -> bool:
        gap = ROW_GAP_SECONDS if thing.is_row else CANDIDATE_GAP_SECONDS
        return now - thing.last_seen > gap

    def _let_go(self, key: Hashable) -> None:
        """Stop following a thing. A sighting is kept to be written; a candidate is forgotten."""
        thing = self._followed.pop(key)
        if thing.is_row:
            self._finished.append(thing)

    # ----- writing the file -----

    def _write_due(self, now: float) -> bool:
        if self._written_at is None or now < self._written_at:
            self._written_at = now
            return False
        return now - self._written_at >= WRITE_EVERY_SECONDS

    def _write_and_tidy(self, now: float) -> None:
        """Every 2 seconds: let go of what is gone, write the changes, and once an hour delete old rows."""
        self._written_at = now
        for key in [k for k, thing in self._followed.items() if self._has_gone(thing, now)]:
            self._let_go(key)
        self._write_rows()
        if self._old_deleted_at is None or now - self._old_deleted_at >= DELETE_OLD_EVERY_SECONDS:
            self._old_deleted_at = now
            self._use_db("deleting old sightings", lambda db: self._delete_old(db, now), None)

    def _write_rows(self) -> None:
        """Write every changed sighting in one transaction. On failure, try again next time."""
        open_rows = [thing for thing in self._followed.values() if thing.is_row]
        waiting = [thing for thing in self._finished + open_rows if thing.changed]
        written = self._use_db("writing to the memory",
                               lambda db: [(thing, _write_one(db, thing)) for thing in waiting], None)
        if written is None:
            del self._finished[:-MAX_FOLLOWED]   # never let unwritten rows pile up
            return
        for thing, row_id in written:
            thing.row_id = row_id
            thing.changed = False
            thing.picture = None
        self._finished.clear()

    def _delete_old(self, db: sqlite3.Connection, now: float) -> None:
        cutoff = now - self.keep_seconds
        db.execute("DELETE FROM pictures WHERE sighting_id IN"
                   " (SELECT id FROM sightings WHERE last_seen < ?)", (cutoff,))
        db.execute("DELETE FROM sightings WHERE last_seen < ?", (cutoff,))

    # ----- the file itself, and its problems -----

    def _use_db(self, what: str, action: Callable[[sqlite3.Connection], Any], default: Any) -> Any:
        """Run ``action`` in one transaction. On a file problem, log once and return ``default``."""
        if self._db is None:
            return default
        try:
            with self._db:              # commits at the end, or rolls back on an error
                return action(self._db)
        except sqlite3.Error as error:
            self._warn_once(what, error)
            if _is_broken_file(error):
                self._start_fresh()
            return default

    def _open(self) -> sqlite3.Connection | None:
        """Open the file, creating it and its folder. A broken file is moved aside."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return self._connect()
        except sqlite3.Error as error:
            if not _is_broken_file(error):
                self._warn_once("opening the memory file", error)
                return None
        except OSError as error:
            self._warn_once("opening the memory file", error)
            return None
        self._move_aside()
        try:
            return self._connect()
        except (sqlite3.Error, OSError) as error:
            self._warn_once("opening the memory file", error)
            return None

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=2.0, check_same_thread=False)
        try:
            db.create_function("folded", 1, _folded, deterministic=True)
            db.execute("PRAGMA journal_mode = WAL")      # cheap writes, readers never wait
            db.execute("PRAGMA synchronous = NORMAL")    # no waiting for the disk on each write
            db.execute("PRAGMA secure_delete = ON")      # deleted rows are overwritten
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")
        except BaseException:
            _close_quietly(db)
            raise
        return db

    def _start_fresh(self) -> None:
        """The file broke while in use: move it aside and write everything open to a new one."""
        if self._db is not None:
            _close_quietly(self._db)
        self._db = None
        self._move_aside()
        self._db = self._open()
        for thing in [*self._followed.values(), *self._finished]:
            thing.row_id = None
            thing.changed = True

    def _move_aside(self) -> None:
        """Rename a broken file (and its helper files) so a fresh one can start."""
        target = _free_name(self.path, time.strftime("%Y%m%d-%H%M%S"))
        try:
            for suffix in ("", "-wal", "-shm"):
                part = Path(f"{self.path}{suffix}")
                if part.exists():
                    part.rename(Path(f"{target}{suffix}"))
        except OSError as error:
            self._warn_once("moving the broken memory file aside", error)
            return
        log.warning("The memory file %s was broken. It was moved to %s and a fresh one was started.",
                    self.path, target.name)

    def _warn_once(self, what: str, error: Exception) -> None:
        if what in self._warned:
            return
        self._warned.add(what)
        log.warning("The camera's memory had a problem %s (%s): %s. The video goes on;"
                    " this problem is not logged again.", what, self.path, error)


# ----- small helpers -----

def _usable_picture(kind: str, picture: bytes | None) -> bytes | None:
    """The one place that decides whether a picture may be kept: objects only, 40 kB at most."""
    if kind != "object" or not picture:
        return None
    if len(picture) > MAX_PICTURE_BYTES:
        return None
    return bytes(picture)


def _write_one(db: sqlite3.Connection, thing: _Followed) -> int:
    """Insert or update one sighting (and its picture). Gives the row's id."""
    if thing.row_id is None:
        row_id = db.execute(
            "INSERT INTO sightings (kind, label, name, first_seen, last_seen, place, confidence)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (thing.kind, thing.label, thing.name, thing.first_seen, thing.last_seen,
             thing.place, thing.confidence)).lastrowid
    else:
        row_id = thing.row_id
        db.execute("UPDATE sightings SET name = ?, last_seen = ?, place = ?, confidence = ?"
                   " WHERE id = ?",
                   (thing.name, thing.last_seen, thing.place, thing.confidence, row_id))
    if thing.picture is not None:
        db.execute("INSERT OR REPLACE INTO pictures (sighting_id, jpeg) VALUES (?, ?)",
                   (row_id, thing.picture))
    return row_id


def _delete_everything(db: sqlite3.Connection) -> None:
    db.execute("DELETE FROM pictures")
    db.execute("DELETE FROM sightings")


def _sighting(row: tuple) -> Sighting:
    row_id, kind, label, name, first_seen, last_seen, place, confidence, has_picture = row
    return Sighting(row_id, kind, label, name, first_seen, last_seen, place, confidence,
                    bool(has_picture))


def _summary_line(kind: str, label: str, name: str | None, first_seen: float, last_seen: float,
                  place: str, now: float) -> str:
    if kind == "person":
        who = f"{name} (person)" if name else "an unrecognised person"
    else:
        who = label
    where = PLACE_WORDS.get(place, "somewhere in the picture")
    return (f"{who}: last seen {time_ago(now - last_seen)} {where},"
            f" in view for {duration(last_seen - first_seen)}")


def time_ago(seconds: float) -> str:
    """How long ago, in words: 'just now', '45 seconds ago', 'yesterday', '3 days ago'."""
    if seconds < 10:
        return "just now"
    if seconds < MINUTE:
        return f"{_count(seconds, 'second')} ago"
    if seconds < HOUR:
        return f"{_count(seconds / MINUTE, 'minute')} ago"
    if seconds < DAY:
        return f"{_count(seconds / HOUR, 'hour')} ago"
    if seconds < 2 * DAY:
        return "yesterday"
    return f"{_count(seconds / DAY, 'day')} ago"


def duration(seconds: float) -> str:
    """How long, in words: 'less than a second', '40 seconds', '2 minutes', '1 hour'."""
    if seconds < 1:
        return "less than a second"
    if seconds < MINUTE:
        return _count(seconds, "second")
    if seconds < HOUR:
        return _count(seconds / MINUTE, "minute")
    return _count(seconds / HOUR, "hour")


def _count(amount: float, unit: str) -> str:
    whole = int(amount)                 # rounded down: 3 min 59 s is "3 minutes"
    return f"{whole} {unit}" if whole == 1 else f"{whole} {unit}s"


def _folded(text: str | None) -> str | None:
    """Lower case for any script, used by SQL to match text without regard to case."""
    return text.casefold() if isinstance(text, str) else None


def _is_broken_file(error: sqlite3.Error) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    return code is not None and (code & 0xFF) in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB)


def _free_name(path: Path, stamp: str) -> Path:
    """A name next to ``path`` for a broken copy that is not taken yet."""
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    number = 2
    while target.exists():
        target = path.with_name(f"{path.name}.corrupt-{stamp}-{number}")
        number += 1
    return target


def _close_quietly(db: sqlite3.Connection) -> None:
    try:
        db.close()
    except sqlite3.Error:
        pass
