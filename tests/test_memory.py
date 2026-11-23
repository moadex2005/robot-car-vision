"""The camera's memory: candidates, sightings, pictures, wording and quiet failure.

Every test uses a temporary file and passes the time in, so hours are played in
milliseconds and nothing touches the real data folder.
"""

import logging
import sqlite3
import threading
import time

import pytest

from car_vision import memory
from car_vision.memory import SightingLog, duration, time_ago

BASE = 1_800_000_000.0     # a fixed moment (January 2027), in seconds since 1970
DAY = 86400.0
JPEG = b"\xff\xd8 a small picture \xff\xd9"


@pytest.fixture
def path(tmp_path):
    return tmp_path / "data" / "memory.sqlite3"


@pytest.fixture
def make_log(path):
    logs = []

    def make(**settings):
        log = SightingLog(path, **settings)
        logs.append(log)
        return log

    yield make
    for log in logs:
        log.close()


def watch(log, key, start, end, label="bottle", kind="object", name=None, place="left",
          confidence=0.5, step=0.5, picture=None):
    """Call seen for one thing from start to end (seconds after BASE), every step seconds."""
    moments = [start + number * step for number in range(int((end - start) / step) + 1)]
    if moments[-1] < end:
        moments.append(end)
    for moment in moments:
        log.seen(key, kind, label, name, place, confidence, BASE + moment, picture)


def reopened(log, make_log, **settings):
    """Close the log (which writes what is open) and open the same file again."""
    log.close()
    return make_log(**settings)


def memory_warnings(caplog):
    return [r for r in caplog.records if r.name == "car_vision.memory" and r.levelno >= logging.WARNING]


# ----- candidates and sightings -----

def test_a_short_flicker_leaves_nothing(make_log, path):
    log = make_log()
    watch(log, "flicker", 0, 0.5)
    watch(log, "other", 10, 10)            # time moves on, the file is written
    assert log.recent() == []
    assert reopened(log, make_log).recent() == []


def test_a_candidate_seen_with_gaps_never_counts_as_in_view(make_log, path):
    log = make_log()
    for moment in (0, 0.5, 2.0, 2.5, 4.0, 4.5):    # never one second without a 1 second gap
        log.seen("flicker", "object", "cat", None, "left", 0.9, BASE + moment)
    assert reopened(log, make_log).recent() == []


def test_staying_min_seconds_makes_one_row(make_log, path):
    log = make_log(min_seconds=1.0)
    watch(log, "bottle", 0, 1.0)
    rows = reopened(log, make_log).recent()
    assert len(rows) == 1
    row = rows[0]
    assert (row.kind, row.label, row.name) == ("object", "bottle", None)
    assert (row.first_seen, row.last_seen) == (BASE, BASE + 1.0)
    assert not row.has_picture


def test_the_file_is_written_at_most_every_two_seconds(make_log):
    log = make_log()
    watch(log, "bottle", 0, 1.5)
    assert log.recent() == []               # a row already, but not written yet
    watch(log, "bottle", 2.0, 2.0)
    assert [r.last_seen for r in log.recent()] == [BASE + 2.0]
    watch(log, "bottle", 3.0, 3.0)
    assert [r.last_seen for r in log.recent()] == [BASE + 2.0]
    watch(log, "bottle", 4.0, 4.0)
    assert [r.last_seen for r in log.recent()] == [BASE + 4.0]


def test_later_calls_extend_last_seen_place_and_best_confidence(make_log, path):
    log = make_log()
    for moment, place, confidence in [(0, "left", 0.4), (0.5, "middle", 0.9), (1.0, "right", 0.6)]:
        log.seen("cup", "object", "cup", None, place, confidence, BASE + moment)
    watch(log, "cup", 1.5, 6.0, label="cup", place="right", confidence=0.3)
    row = reopened(log, make_log).recent()[0]
    assert (row.first_seen, row.last_seen) == (BASE, BASE + 6.0)
    assert row.place == "right"
    assert row.confidence == pytest.approx(0.9)


def test_a_gap_of_more_than_three_seconds_makes_a_new_row(make_log, path):
    log = make_log()
    watch(log, "bottle", 0, 5)
    watch(log, "bottle", 9, 12)            # same key, but gone for 4 seconds
    rows = reopened(log, make_log).recent()
    assert [(r.first_seen, r.last_seen) for r in rows] == [(BASE + 9, BASE + 12), (BASE, BASE + 5)]


def test_a_short_gap_extends_the_same_row(make_log, path):
    log = make_log()
    watch(log, "bottle", 0, 5)
    watch(log, "bottle", 7, 10)            # gone for 2 seconds only
    rows = reopened(log, make_log).recent()
    assert [(r.first_seen, r.last_seen) for r in rows] == [(BASE, BASE + 10)]


def test_a_name_recognised_part_way_stays_on_the_row(make_log, path):
    log = make_log()
    watch(log, "p1", 0, 2, kind="person", label="person")
    watch(log, "p1", 2.5, 4, kind="person", label="person", name="Sara Ahmed")
    watch(log, "p1", 4.5, 6, kind="person", label="person")     # the face turned away
    row = reopened(log, make_log).recent()[0]
    assert (row.kind, row.label, row.name) == ("person", "person", "Sara Ahmed")
    assert row.first_seen == BASE


def test_bad_kind_or_place_is_refused(make_log):
    log = make_log()
    with pytest.raises(ValueError):
        log.seen("k", "animal", "cat", None, "left", 0.5, BASE)
    with pytest.raises(ValueError):
        log.seen("k", "object", "cat", None, "top", 0.5, BASE)


def test_memory_stays_small_when_thousands_of_tracks_come_and_go(make_log):
    log = make_log()
    for number in range(5000):
        log.seen(("epoch", number, "cat"), "object", "cat", None, "left", 0.5, BASE + number * 0.001)
        assert len(log._followed) <= memory.MAX_FOLLOWED
    assert len(log._finished) <= memory.MAX_FOLLOWED


# ----- pictures -----

def test_an_object_keeps_its_picture(make_log, path):
    log = make_log()
    watch(log, "cup", 0, 1, label="cup", picture=JPEG)
    log = reopened(log, make_log)
    row = log.recent()[0]
    assert row.has_picture
    assert log.picture(row.id) == JPEG
    log.close()


def test_never_a_picture_of_a_person(make_log, path):
    log = make_log()
    watch(log, "p1", 0, 2, kind="person", label="person", name="Sara Ahmed", picture=JPEG)
    log = reopened(log, make_log)
    row = log.recent()[0]
    assert not row.has_picture
    assert log.picture(row.id) is None
    log.close()


def test_pictures_larger_than_40_kb_are_refused(make_log, path):
    log = make_log()
    watch(log, "big", 0, 1, label="big", picture=b"x" * (memory.MAX_PICTURE_BYTES + 1))
    watch(log, "fits", 0, 1, label="fits", picture=b"x" * memory.MAX_PICTURE_BYTES)
    log = reopened(log, make_log)
    pictures = {row.label: row.has_picture for row in log.recent()}
    assert pictures == {"big": False, "fits": True}
    log.close()


def test_the_most_confident_picture_is_kept(make_log, path):
    log = make_log()
    log.seen("cup", "object", "cup", None, "left", 0.5, BASE, b"first")
    log.seen("cup", "object", "cup", None, "left", 0.9, BASE + 0.5, b"clearest")
    log.seen("cup", "object", "cup", None, "left", 0.6, BASE + 1.0, b"later")
    log = reopened(log, make_log)
    assert log.picture(log.recent()[0].id) == b"clearest"
    log.close()


def test_no_picture_for_an_unknown_id(make_log):
    assert make_log().picture(12345) is None


# ----- recent -----

def test_recent_is_newest_first_and_filters_label_and_name(make_log, path):
    log = make_log()
    watch(log, "b", 0, 2, label="bottle")
    watch(log, "l", 10, 12, label="laptop")
    watch(log, "p", 20, 22, kind="person", label="person", name="Élodie Martin")
    log = reopened(log, make_log)
    assert [r.label for r in log.recent()] == ["person", "laptop", "bottle"]
    assert [r.label for r in log.recent(limit=2)] == ["person", "laptop"]
    assert [r.label for r in log.recent(text="BOT")] == ["bottle"]
    assert [r.name for r in log.recent(text="ÉLODIE")] == ["Élodie Martin"]
    assert [r.label for r in log.recent(text="PERSON")] == ["person"]
    assert len(log.recent(text="  ")) == 3
    assert log.recent(text="sofa") == []
    assert log.recent(limit=0) == []
    log.close()


# ----- summary -----

@pytest.mark.parametrize("later, words", [
    (3, "just now"),
    (45, "45 seconds ago"),
    (61, "1 minute ago"),
    (3 * 60 + 59, "3 minutes ago"),
    (3600, "1 hour ago"),
    (5 * 3600 + 10, "5 hours ago"),
    (DAY + 3600, "yesterday"),
    (3 * DAY, "3 days ago"),
])
def test_summary_says_when_in_words(make_log, path, later, words):
    log = make_log()
    watch(log, "laptop", 0, 120, label="laptop", step=0.5)
    log = reopened(log, make_log)
    assert log.summary(now=BASE + 120 + later) == [
        f"laptop: last seen {words} on the left, in view for 2 minutes"]
    log.close()


@pytest.mark.parametrize("seconds, step, words", [
    (40, 0.5, "40 seconds"),
    (120, 0.5, "2 minutes"),
    (3600, 2.5, "1 hour"),
])
def test_summary_says_how_long_in_view(make_log, path, seconds, step, words):
    log = make_log()
    watch(log, "cup", 0, 1, label="cup")
    watch(log, "cup", 1, seconds, label="cup", step=step)
    log = reopened(log, make_log)
    assert log.summary(now=BASE + seconds + 600) == [
        f"cup: last seen 10 minutes ago on the left, in view for {words}"]
    log.close()


def test_summary_names_people_and_the_unrecognised(make_log, path):
    log = make_log()
    watch(log, "sara", 0, 40, kind="person", label="person", name="Sara Ahmed", place="middle")
    watch(log, "who", 100, 110, kind="person", label="person", place="right")
    log = reopened(log, make_log)
    assert log.summary(now=BASE + 3700) == [
        "an unrecognised person: last seen 59 minutes ago on the right, in view for 10 seconds",
        "Sara Ahmed (person): last seen 1 hour ago in the middle, in view for 40 seconds",
    ]
    log.close()


def test_summary_gives_one_line_per_thing_newest_first(make_log, path):
    log = make_log()
    watch(log, ("e", 1, "bottle"), 0, 5, label="bottle", place="right")
    watch(log, ("e", 2, "laptop"), 10, 20, label="laptop")
    watch(log, ("e", 3, "bottle"), 30, 32, label="bottle", place="middle")   # the bottle again
    log = reopened(log, make_log)
    assert log.summary(now=BASE + 92) == [
        "bottle: last seen 1 minute ago in the middle, in view for 2 seconds",
        "laptop: last seen 1 minute ago on the left, in view for 10 seconds",
    ]
    assert log.summary(now=BASE + 92, limit=1) == [
        "bottle: last seen 1 minute ago in the middle, in view for 2 seconds"]
    log.close()


def test_time_words_at_the_edges():
    assert time_ago(-5) == "just now"
    assert time_ago(9.9) == "just now"
    assert time_ago(10) == "10 seconds ago"
    assert time_ago(2 * DAY - 1) == "yesterday"
    assert time_ago(2 * DAY) == "2 days ago"
    assert duration(0) == "less than a second"
    assert duration(1) == "1 second"
    assert duration(59.9) == "59 seconds"
    assert duration(7200) == "2 hours"


def test_summary_of_an_empty_memory_is_empty(make_log):
    assert make_log().summary(now=BASE) == []


# ----- old rows and clearing -----

def test_old_rows_are_deleted_once_an_hour(make_log, path):
    log = make_log(keep_days=1 / 24 / 60)                  # keep one minute only
    watch(log, "old", 0, 2, label="old")
    watch(log, "new", 120, 125, label="new")               # old is past a minute, but no hour yet
    assert {r.label for r in log.recent()} == {"old", "new"}
    watch(log, "newest", 3700, 3705, label="newest")       # an hour later: old rows go
    labels = [r.label for r in reopened(log, make_log, keep_days=1).recent()]
    assert labels == ["newest"]


def test_rows_older_than_keep_days_are_deleted_at_start_up(make_log, path):
    log = make_log(keep_days=7)
    real_now = time.time()
    for label, age in [("week old", 8 * DAY), ("day old", DAY)]:
        start = real_now - age - BASE
        watch(log, label, start, start + 2, label=label)
    log.close()
    log = make_log(keep_days=7)
    assert [r.label for r in log.recent()] == ["day old"]


def test_clear_empties_everything(make_log, path):
    log = make_log()
    watch(log, "cup", 0, 5, label="cup", picture=JPEG)
    watch(log, "cup", 5.5, 8, label="cup", picture=JPEG)
    old_id = log.recent()[0].id
    log.clear()
    assert log.recent() == [] and log.picture(old_id) is None
    assert log.summary(now=BASE + 10) == []
    watch(log, "cup", 8.5, 11, label="cup")                # still in view: a fresh row
    rows = reopened(log, make_log).recent()
    assert [(r.first_seen, r.last_seen) for r in rows] == [(BASE + 8.5, BASE + 11)]
    assert rows[0].id != old_id


def test_close_twice_and_seen_after_close_do_nothing(make_log):
    log = make_log()
    watch(log, "cup", 0, 2)
    log.close()
    log.close()
    log.seen("cup", "object", "cup", None, "left", 0.5, BASE + 3)
    assert log.recent() == [] and log.summary(now=BASE) == []


# ----- threads -----

def test_the_video_thread_and_web_threads_at_once(make_log, path):
    log = make_log()
    errors = []
    done = threading.Event()

    def video_loop():
        try:
            for frame in range(2000):
                now = BASE + frame * 0.05
                for key in range(10):
                    log.seen(key, "object", f"thing {key}", None, "left", 0.5, now, JPEG)
        except Exception as error:      # noqa: BLE001  any error fails the test below
            errors.append(error)
        finally:
            done.set()

    def web_server():
        try:
            while not done.is_set():
                for row in log.recent(limit=20, text="thing"):
                    log.picture(row.id)
                log.summary(now=BASE + 100)
        except Exception as error:      # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=video_loop)] + [threading.Thread(target=web_server) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    rows = reopened(log, make_log).recent(limit=100)
    assert sorted(r.label for r in rows) == sorted(f"thing {key}" for key in range(10))
    assert all(r.last_seen == BASE + 1999 * 0.05 for r in rows)


# ----- quiet failure -----

@pytest.mark.parametrize("problem", ["folder is a file", "file is a folder"])
def test_a_file_that_cannot_be_opened_never_stops_the_video(tmp_path, caplog, problem):
    if problem == "folder is a file":
        (tmp_path / "data").write_text("not a folder")
        path = tmp_path / "data" / "memory.sqlite3"
    else:
        path = tmp_path / "memory.sqlite3"
        path.mkdir()
    with caplog.at_level(logging.WARNING, logger="car_vision.memory"):
        log = SightingLog(path)
        for number in range(400):          # 20 seconds of frames, past many writes
            log.seen(number // 50, "object", "cup", None, "left", 0.5, BASE + number * 0.05, JPEG)
        assert log.recent() == []
        assert log.summary(now=BASE) == []
        assert log.picture(1) is None
        log.clear()
        log.close()
    assert len(memory_warnings(caplog)) == 1
    assert len(log._finished) <= memory.MAX_FOLLOWED


def test_a_broken_file_is_moved_aside_and_a_fresh_one_started(make_log, path, caplog):
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not a database " * 200)
    with caplog.at_level(logging.WARNING, logger="car_vision.memory"):
        log = make_log()
    moved = list(path.parent.glob("memory.sqlite3.corrupt-*"))
    assert len(moved) == 1
    assert moved[0].read_bytes().startswith(b"this is not a database")
    assert any("moved" in r.getMessage() for r in memory_warnings(caplog))
    watch(log, "cup", 0, 2, label="cup")
    log = reopened(log, make_log)
    assert [r.label for r in log.recent()] == ["cup"]
    log.close()


def test_a_file_that_breaks_while_in_use_is_replaced_and_what_is_open_kept(
        make_log, path, caplog, monkeypatch):
    log = make_log()
    watch(log, "cup", 0, 4, label="cup")                   # written once already
    real_write = memory._write_one
    calls = []

    def breaks_once(db, thing):
        calls.append(thing.label)
        if len(calls) == 1:
            error = sqlite3.DatabaseError("database disk image is malformed")
            error.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
            raise error
        return real_write(db, thing)

    monkeypatch.setattr(memory, "_write_one", breaks_once)
    with caplog.at_level(logging.WARNING, logger="car_vision.memory"):
        watch(log, "cup", 4.5, 9, label="cup")
    assert len(list(path.parent.glob("memory.sqlite3.corrupt-*"))) == 1
    rows = reopened(log, make_log).recent()
    assert [(r.label, r.first_seen, r.last_seen) for r in rows] == [("cup", BASE, BASE + 9)]


def test_a_locked_file_slows_nothing_and_is_written_later(make_log, path, caplog):
    log = make_log()
    other = sqlite3.connect(path, timeout=0)
    other.execute("BEGIN EXCLUSIVE")                       # another program holds the file
    with caplog.at_level(logging.WARNING, logger="car_vision.memory"):
        slowest = 0.0
        for number in range(200):                          # 10 seconds of frames
            started = time.perf_counter()
            log.seen("cup", "object", "cup", None, "left", 0.5, BASE + number * 0.05)
            slowest = max(slowest, time.perf_counter() - started)
        warnings_while_locked = len(memory_warnings(caplog))
        other.rollback()
        other.close()
        watch(log, "cup", 10, 14, label="cup")
    assert slowest < 1.0
    assert 1 <= warnings_while_locked <= 2                 # writing (and deleting old rows), once each
    assert len(memory_warnings(caplog)) == warnings_while_locked
    rows = reopened(log, make_log).recent()
    assert [(r.first_seen, r.last_seen) for r in rows] == [(BASE, BASE + 14)]
