"""The shared state both tracks rely on."""

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
