"""The shared state both tracks rely on."""

import threading

from car_vision.state import UNKNOWN, Detection, RecognizedFace, WorldState

PERSON = Detection("person", 0.9, (10, 10, 110, 310), track_id=1)
CHAIR = Detection("chair", 0.7, (200, 150, 300, 300), track_id=2)
SARA = RecognizedFace("Sara", 0.62, (40, 20, 90, 80), person_id="sara", track_id=1)
STRANGER = RecognizedFace(UNKNOWN, 0.12, (300, 20, 350, 80))


def test_starts_empty():
    snap = WorldState().snapshot()
    assert snap.detections == () and snap.faces == ()


def test_updates_do_not_erase_each_other():
    state = WorldState()
    state.update_detections([PERSON, CHAIR], (640, 480), now=1.0)
    state.update_faces([SARA], now=1.5)
    snap = state.snapshot()
    assert snap.detections == (PERSON, CHAIR)
    assert snap.faces == (SARA,)
    assert snap.frame_size == (640, 480)
    assert (snap.detections_at, snap.faces_at) == (1.0, 1.5)


def test_snapshot_does_not_change_after_later_updates():
    state = WorldState()
    state.update_detections([PERSON], (640, 480), now=1.0)
    before = state.snapshot()
    state.update_detections([], (640, 480), now=2.0)
    assert before.detections == (PERSON,)


def test_people_and_known_faces_filters():
    state = WorldState()
    state.update_detections([PERSON, CHAIR], (640, 480), now=1.0)
    state.update_faces([SARA, STRANGER], now=1.0)
    snap = state.snapshot()
    assert snap.people() == (PERSON,)
    assert snap.known_faces() == (SARA,)
    assert not STRANGER.is_known


def test_current_people_lists_known_names_once():
    state = WorldState()
    state.update_faces([SARA, SARA, STRANGER], now=10.0)
    assert state.current_people(max_age_seconds=3.0, now=11.0) == ["Sara"]


def test_current_people_forgets_stale_faces():
    state = WorldState()
    state.update_faces([SARA], now=10.0)
    assert state.current_people(max_age_seconds=3.0, now=20.0) == []


def test_memory_starts_empty():
    assert WorldState().memory() == ()


def test_memory_lines_are_kept_as_a_copy():
    state = WorldState()
    lines = ["laptop: last seen just now on the left, in view for 2 minutes"]
    state.set_memory(lines)
    lines.append("changed later")
    assert state.memory() == ("laptop: last seen just now on the left, in view for 2 minutes",)
    state.set_memory([])
    assert state.memory() == ()


def test_memory_does_not_disturb_the_snapshot():
    state = WorldState()
    state.update_detections([PERSON], (640, 480), now=1.0)
    state.set_memory(["bottle: last seen 1 minute ago in the middle, in view for 5 seconds"])
    assert state.snapshot().detections == (PERSON,)


def test_memory_can_be_set_and_read_from_two_threads():
    state = WorldState()
    seen_lengths = []

    def writer():
        for count in range(500):
            state.set_memory([f"line {n}" for n in range(count % 7)])

    def reader():
        for _ in range(500):
            seen_lengths.append(len(state.memory()))

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(0 <= length < 7 for length in seen_lengths)
