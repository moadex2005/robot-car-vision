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

Threads: the video loop calls ``seen`` many times a second and ``tick`` on
every turn, also when nothing is in view; ``tick`` does the periodic work
(writing, finishing rows whose thing has gone, the hourly delete). ``seen``
mostly changes things in memory: the file is written at most once every 2
seconds, in one short transaction, so the video never waits long for the disk.

The web server and the assistant's summary read through a second connection
of their own, with a lock of their own (``_side_lock``). The file is in WAL
mode, so a slow filter query there never holds up the video's writes, and
``clear`` empties the file there too: the video's lock is held only for the
moment it takes to forget what is followed in memory.

At most MAX_ROWS sightings are kept (the newest); older ones go at start-up
and at the hourly delete, so the file and its queries stay small.

Failure is quiet: when the file cannot be created, is locked or is broken, the
problem is logged once, the video goes on and reads give empty results. A
broken file is moved aside as ``memory.sqlite3.corrupt-<date>-<time>`` and a
fresh one is started.

Pictures are kept only for objects, never for people. Only the standard
library is used.

A watched thing that appeared (M25, ``appeared``) is one more kind of row in
the same table: kind ``watch``, first and last seen both the moment it
appeared, never a picture. It is listed, deleted and cleared with the
sightings, so the table needs no new column and a file from before still opens
as it is.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KINDS = ("object", "person")
WATCH = "watch"                 # the kind of a row that says a watched thing appeared (M25)
PLACE_WORDS = {"left": "on the left", "middle": "in the middle", "right": "on the right"}

WRITE_EVERY_SECONDS = 2.0       # the file is written at most this often
ROW_GAP_SECONDS = 3.0           # a sighting not seen for this long is final
CANDIDATE_GAP_SECONDS = 1.0     # a candidate not seen for this long is forgotten
DELETE_OLD_EVERY_SECONDS = 3600.0
MAX_PICTURE_BYTES = 40_000      # larger pictures are refused quietly
MAX_FOLLOWED = 200              # things followed in memory at once, at most
MAX_ROWS = 20_000               # sightings kept in the file, the newest
BUSY_TIMEOUT_SECONDS = 0.1      # how long the video waits for a file locked by another program
SIDE_BUSY_TIMEOUT_SECONDS = 2.0 # ... and how long a web thread waits

MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused, even after deletes
    kind        TEXT NOT NULL,                      -- "object", "person" or "watch" (a watched thing appeared)
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
    kind: str              # "object", "person" or "watch"
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
        self._clearing = False          # clear() is emptying the file: the video does not write meanwhile
        self._db = self._open()
        self._db_number = 0             # goes up when the file is replaced; the side connection follows
        self._side_lock = threading.Lock()
        self._side: sqlite3.Connection | None = None
        self._side_number = -1          # the _db_number the side connection was opened for
        start_up = time.time()
        self._use_db("deleting old sightings", lambda db: self._delete_old(db, start_up), None)

    # ----- what the video loop calls -----

    def tick(self, now: float) -> None:
        """The periodic work, called on every turn of the video loop, also with nothing in view.

        Writes what changed (at most every 2 seconds), finishes the rows of
        things that have gone, and deletes old rows once an hour. Without it a
        thing that leaves the picture would wait for the next ``seen`` to be written.
        """
        with self._lock:
            if not self._closed and self._write_due(now):
                self._write_and_tidy(now)

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

    def appeared(self, label: str, place: str, now: float, name: str | None = None) -> None:
        """Note that a watched thing appeared (M25): one row of kind "watch", never with a picture.

        ``label`` is the watched name ("bottle"), or "person" with the person's
        ``name``. It is written with the next write, like a sighting that ended.
        """
        if place not in PLACE_WORDS:
            raise ValueError(f"place must be 'left', 'middle' or 'right', not {place!r}")
        with self._lock:
            if not self._closed:
                self._finished.append(_Followed(WATCH, label, name, now, now, place, 0.0, is_row=True, changed=True))

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
        rows = self._use_side("reading the memory",
                              lambda db: db.execute(query, (*values, limit)).fetchall(), [])
        return [_sighting(row) for row in rows]

    def picture(self, sighting_id: int) -> bytes | None:
        """The small JPEG of one sighting, or None when there is none."""
        query = "SELECT jpeg FROM pictures WHERE sighting_id = ?"
        row = self._use_side("reading a picture",
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
        rows = self._use_side("reading the memory",
                              lambda db: db.execute(query, (max(0, int(limit)),)).fetchall(), [])
        return [_summary_line(*row, now=now) for row in rows]

    def clear(self) -> None:
        """Forget everything: the file is emptied and nothing in view is carried over.

        The video's lock is held only to forget what is followed and to stop
        its writes; the slow emptying of the file happens on the side connection.
        """
        with self._lock:
            self._followed.clear()
            self._finished.clear()
            self._clearing = True
        try:
            self._use_side("emptying the memory", _delete_everything, None)
            # Also empty the write-ahead file, so the old rows do not linger on disk.
            self._use_side("emptying the memory",
                           lambda db: db.execute("PRAGMA wal_checkpoint(TRUNCATE)"), None)
        finally:
            with self._lock:
                self._clearing = False

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
        with self._side_lock:
            if self._side is not None:
                _close_quietly(self._side)
                self._side = None

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
        if self._clearing:
            return False                # what is in view now is written once the file is empty
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
        """Delete the rows older than keep_days, and all but the newest MAX_ROWS."""
        cutoff = now - self.keep_seconds
        db.execute("DELETE FROM pictures WHERE sighting_id IN"
                   " (SELECT id FROM sightings WHERE last_seen < ?)", (cutoff,))
        db.execute("DELETE FROM sightings WHERE last_seen < ?", (cutoff,))
        beyond = ("SELECT id FROM sightings ORDER BY last_seen DESC, id DESC"
                  " LIMIT -1 OFFSET ?")       # every row after the newest MAX_ROWS
        db.execute(f"DELETE FROM pictures WHERE sighting_id IN ({beyond})", (MAX_ROWS,))
        db.execute(f"DELETE FROM sightings WHERE id IN ({beyond})", (MAX_ROWS,))

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

    def _use_side(self, what: str, action: Callable[[sqlite3.Connection], Any], default: Any) -> Any:
        """Like _use_db, on the side connection the web threads share (never the video's lock).

        Nothing is read when the video's own connection could not open the
        file. A problem is logged once; a broken file is left to the video's
        connection to replace.
        """
        with self._side_lock:
            db = self._side_connection()
            if db is None:
                return default
            try:
                with db:
                    return action(db)
            except sqlite3.Error as error:
                self._warn_once(what, error)
                _close_quietly(db)
                self._side = None       # opened afresh next time
                return default

    def _side_connection(self) -> sqlite3.Connection | None:
        """The side connection, opened (again) when needed. Call with _side_lock held."""
        if self._closed or self._db is None:
            return None
        if self._side is not None and self._side_number == self._db_number:
            return self._side
        if self._side is not None:
            _close_quietly(self._side)      # the file was replaced: follow it
            self._side = None
        try:
            self._side = self._connect(SIDE_BUSY_TIMEOUT_SECONDS)
        except (sqlite3.Error, OSError) as error:
            self._warn_once("reading the memory", error)
            return None
        self._side_number = self._db_number
        return self._side

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

    def _connect(self, busy_seconds: float = BUSY_TIMEOUT_SECONDS) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=2.0, check_same_thread=False)
        try:
            db.create_function("folded", 1, _folded, deterministic=True)
            db.execute("PRAGMA journal_mode = WAL")      # cheap writes, readers never wait
            db.execute("PRAGMA synchronous = NORMAL")    # no waiting for the disk on each write
            db.execute("PRAGMA secure_delete = ON")      # deleted rows are overwritten
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA busy_timeout = {int(busy_seconds * 1000)}")
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
        self._db_number += 1            # the side connection opens the new file next time
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
    if kind == WATCH:
        return _watch_line(f"person {name}" if name else label, last_seen, place, now)
    if kind == "person":
        who = f"{name} (person)" if name else "an unrecognised person"
    else:
        who = label
    where = PLACE_WORDS.get(place, "somewhere in the picture")
    return (f"{who}: last seen {time_ago(now - last_seen)} {where},"
            f" in view for {duration(last_seen - first_seen)}")


def _watch_line(what: str, at: float, place: str, now: float) -> str:
    """The latest appearance of a watched thing: '10:42 the watched bottle appeared, on the left'
    ('... the watched person Mohamed Adel appeared ...' for a person)."""
    where = PLACE_WORDS.get(place, "somewhere in the picture")
    return f"{clock_time(at, now)} the watched {what} appeared, {where}"


def clock_time(at: float, now: float) -> str:
    """The local time of a moment: '10:42' today, 'yesterday 10:42', else '21 November 10:42'."""
    moment, today = time.localtime(at), time.localtime(now)
    hours = time.strftime("%H:%M", moment)
    days = (_local_date(today) - _local_date(moment)).days
    if days == 0:
        return hours
    if days == 1:
        return f"yesterday {hours}"
    return f"{moment.tm_mday} {time.strftime('%B', moment)} {hours}"


def _local_date(moment: time.struct_time) -> date:
    return date(moment.tm_year, moment.tm_mon, moment.tm_mday)


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
