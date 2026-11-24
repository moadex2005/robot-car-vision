"""M25 Part B: the memory keeps one entry each time a watched thing appeared.

Every test uses a temporary file and passes the time in. The entry is a row of kind
"watch" in the same table as the sightings: kept, deleted and cleared by the same rules,
never with a picture, and a file made before this milestone opens as it is.
"""

import sqlite3
import time

import pytest

from car_vision.memory import SightingLog, clock_time
from test_memory import BASE, DAY, JPEG, reopened, watch
from test_memory import make_log, path  # noqa: F401  (the fixtures, used by name)

# The table as the code before this milestone made it (memory.SCHEMA at commit 3abe583).
OLD_SCHEMA = """
CREATE TABLE sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    label       TEXT NOT NULL,
    name        TEXT,
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    place       TEXT NOT NULL,
    confidence  REAL NOT NULL
);
CREATE INDEX sightings_by_last_seen ON sightings (last_seen);
CREATE INDEX sightings_by_thing ON sightings (kind, label, name, last_seen);
CREATE TABLE pictures (
    sighting_id INTEGER PRIMARY KEY,
    jpeg        BLOB NOT NULL
);
"""


def local(hour, minute, day=24):
    """Seconds since 1970 of a local time on a day in November 2026."""
    return time.mktime((2026, 11, day, hour, minute, 0, 0, 0, -1))


def test_an_appearance_is_stored_and_listed_among_the_sightings_in_time_order(make_log, path):
    log = make_log()
    watch(log, "cup", 0, 5, label="cup", place="right")
    log.appeared("bottle", "left", BASE + 10)
    watch(log, "laptop", 20, 25, label="laptop")
    log.appeared("person", "middle", BASE + 30, "Sara Ahmed")
    rows = reopened(log, make_log).recent()
    assert [(row.kind, row.label, row.name) for row in rows] == [
        ("watch", "person", "Sara Ahmed"), ("object", "laptop", None), ("watch", "bottle", None), ("object", "cup", None)]
    bottle = rows[2]
    assert (bottle.first_seen, bottle.last_seen, bottle.place, bottle.has_picture) == (BASE + 10, BASE + 10, "left", False)


def test_an_appearance_is_written_with_the_next_write_without_closing(make_log):
    log = make_log()
    log.appeared("bottle", "left", BASE)
    log.tick(BASE)                                          # the first tick only starts the write clock
    log.tick(BASE + 2.5)
    assert [(row.kind, row.label) for row in log.recent()] == [("watch", "bottle")]


def test_the_filter_finds_an_appearance_by_its_name(make_log, path):
    log = make_log()
    log.appeared("bottle", "left", BASE)
    log.appeared("person", "right", BASE + 1, "Sara Ahmed")
    log = reopened(log, make_log)
    assert [row.label for row in log.recent(text="bott")] == ["bottle"]
    assert [row.name for row in log.recent(text="sara")] == ["Sara Ahmed"]


def test_an_appearance_never_has_a_picture(make_log, path):
    log = make_log()
    log.appeared("bottle", "left", BASE)
    [row] = reopened(log, make_log).recent()
    assert row.has_picture is False and log.picture(row.id) is None


def test_appearances_are_deleted_with_the_sightings_after_keep_days(make_log, path):
    log = make_log(keep_days=1 / 24 / 60)                  # keep one minute only
    log.appeared("bottle", "left", BASE)
    log.tick(BASE)
    log.tick(BASE + 2)                                      # written, and the hourly delete has run
    watch(log, "new", 120, 125, label="new")               # the bottle is past a minute, but no hour yet
    assert {row.label for row in log.recent()} == {"bottle", "new"}
    watch(log, "newest", 3700, 3705, label="newest")       # an hour later: old rows go
    assert [row.label for row in reopened(log, make_log, keep_days=1).recent()] == ["newest"]


def test_appearances_older_than_keep_days_go_at_start_up(make_log):
    log = make_log(keep_days=1)
    now = time.time()
    log.appeared("old bottle", "left", now - 2 * DAY)
    log.appeared("new bottle", "left", now - 60)
    log.close()
    assert [row.label for row in make_log(keep_days=1).recent()] == ["new bottle"]


def test_clear_memory_empties_the_appearances_too(make_log, path):
    log = make_log()
    log.appeared("bottle", "left", BASE)
    log = reopened(log, make_log)
    log.appeared("cup", "right", BASE + 5)                  # not written yet: forgotten as well
    log.clear()
    assert log.recent() == []
    assert reopened(log, make_log).recent() == []


def test_a_wrong_place_is_refused_and_after_close_nothing_happens(make_log):
    log = make_log()
    with pytest.raises(ValueError):
        log.appeared("bottle", "under the table", BASE)
    log.close()
    log.appeared("bottle", "left", BASE)
    assert log.recent() == []


def test_the_assistants_lines_say_when_a_watched_thing_appeared(make_log, path):
    log = make_log()
    now = local(10, 50)
    watch(log, "cup", now - BASE - 300, now - BASE - 290, label="cup")
    log.appeared("bottle", "left", local(10, 42))
    log.appeared("bottle", "middle", local(9, 15))          # an earlier appearance: the newest is told
    log.appeared("person", "right", local(10, 30), "Sara Ahmed")
    lines = reopened(log, make_log).summary(now=now)
    assert lines == [
        "cup: last seen 4 minutes ago on the left, in view for 10 seconds",
        "10:42 the watched bottle appeared, on the left",
        "10:30 the watched person Sara Ahmed appeared, on the right",
    ]


def test_clock_time_names_the_day_when_it_was_not_today():
    now = local(10, 50)
    assert clock_time(local(10, 42), now) == "10:42"
    assert clock_time(local(23, 5, day=23), now) == "yesterday 23:05"
    assert clock_time(local(8, 0, day=20), now) == "20 November 08:00"


def test_a_memory_file_from_before_this_milestone_opens_and_takes_appearances(make_log, path):
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:                       # made the way the old code made it
        db.executescript(OLD_SCHEMA)
        db.execute("INSERT INTO sightings (kind, label, name, first_seen, last_seen, place, confidence)"
                   " VALUES ('object', 'cup', NULL, ?, ?, 'right', 0.7)", (BASE, BASE + 4))
        db.execute("INSERT INTO pictures (sighting_id, jpeg) VALUES (1, ?)", (JPEG,))
    db.close()
    log = make_log()
    log.appeared("bottle", "left", BASE + 10)
    rows = reopened(log, make_log).recent()
    assert [(row.kind, row.label) for row in rows] == [("watch", "bottle"), ("object", "cup")]
    assert rows[1].has_picture is True and log_picture(path, 1) == JPEG      # nothing old was lost


def log_picture(path, sighting_id):
    log = SightingLog(path)
    try:
        return log.picture(sighting_id)
    finally:
        log.close()
