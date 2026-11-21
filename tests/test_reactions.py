"""The reaction rules, played with made-up snapshots and times. No camera, model or clock."""

import logging

import pytest

from car_vision.config import ConfigError
from car_vision.reactions import Reaction, ReactionEngine
from car_vision.state import UNKNOWN, Detection, RecognizedFace, WorldSnapshot

SETTINGS = {
    "greet_cooldown_seconds": 60,
    "message_seconds": 4.0,
}
FRAME = (640, 480)


def engine(**changes):
    return ReactionEngine({**SETTINGS, **changes})


def face(person_id, name):
    return RecognizedFace(name, 0.7, (100, 100, 160, 170), person_id, track_id=1)


SARA = face("sara", "Sara Ahmed")
OMAR = face("omar", "Omar Ali")
STRANGER = RecognizedFace(UNKNOWN, 0.2, (300, 100, 360, 170))


def person(height, label="person", frame_height=480):
    """A box `height` pixels tall, standing on the bottom of the frame."""
    return Detection(label, 0.9, (200, frame_height - 1 - height, 300, frame_height - 1), 1)


def snap(*faces, detections=(), frame_size=FRAME):
    return WorldSnapshot(detections=tuple(detections), faces=tuple(faces), frame_size=frame_size)


def greetings(reactions):
    return [r.text for r in reactions if r.kind == "greeting"]


# ---- greetings -----------------------------------------------------------


def test_known_face_is_greeted_by_name():
    shown = engine().update(snap(SARA), now=0.0)
    assert shown == [Reaction("greeting", "Hello, Sara Ahmed!", "sara")]


def test_one_greeting_per_person_per_cooldown_however_many_frames():
    reactions = engine()
    started = []
    previous = []
    for step in range(0, 590):                          # 59 seconds in view at 10 frames a second
        now = step / 10
        shown = greetings(reactions.update(snap(SARA), now))
        if shown and not previous:
            started.append(now)
        previous = shown
    assert started == [0.0]


def test_greeted_again_after_the_cooldown():
    reactions = engine()
    assert greetings(reactions.update(snap(SARA), 0.0)) == ["Hello, Sara Ahmed!"]
    reactions.update(snap(), 10.0)                      # she walks away
    assert greetings(reactions.update(snap(SARA), 30.0)) == []    # back too soon
    reactions.update(snap(), 40.0)
    assert greetings(reactions.update(snap(SARA), 60.0)) == ["Hello, Sara Ahmed!"]


def test_person_staying_in_view_is_greeted_again_once_the_cooldown_has_passed():
    reactions = engine()
    reactions.update(snap(SARA), 0.0)
    assert greetings(reactions.update(snap(SARA), 59.9)) == []
    assert greetings(reactions.update(snap(SARA), 60.0)) == ["Hello, Sara Ahmed!"]


def test_two_known_people_get_one_greeting_each():
    reactions = engine()
    assert greetings(reactions.update(snap(SARA, OMAR), 0.0)) == ["Hello, Sara Ahmed!", "Hello, Omar Ali!"]
    assert greetings(reactions.update(snap(SARA, OMAR), 1.0)) == ["Hello, Sara Ahmed!", "Hello, Omar Ali!"]


def test_second_person_arriving_later_gets_their_own_greeting():
    reactions = engine()
    reactions.update(snap(SARA), 0.0)
    assert greetings(reactions.update(snap(SARA, OMAR), 2.0)) == ["Hello, Sara Ahmed!", "Hello, Omar Ali!"]
    assert greetings(reactions.update(snap(SARA, OMAR), 4.0)) == ["Hello, Omar Ali!"]   # Sara's ran out
    assert greetings(reactions.update(snap(SARA, OMAR), 6.0)) == []


def test_greeting_expires_after_message_seconds():
    reactions = engine()
    reactions.update(snap(SARA), 0.0)
    assert greetings(reactions.update(snap(SARA), 3.9)) == ["Hello, Sara Ahmed!"]
    assert greetings(reactions.update(snap(SARA), 4.0)) == []


def test_greeting_stays_its_full_time_after_the_person_leaves():
    reactions = engine()
    reactions.update(snap(SARA), 0.0)
    assert greetings(reactions.update(snap(), 2.0)) == ["Hello, Sara Ahmed!"]
    assert greetings(reactions.update(snap(), 4.0)) == []


def test_unknown_faces_are_never_greeted():
    reactions = engine()
    for step in range(100):
        assert greetings(reactions.update(snap(STRANGER), step * 1.0)) == []


# ---- no close warning any more -------------------------------------------


def test_a_person_filling_the_frame_gives_no_warning():
    """The red "Person close in front of the car" warning was taken out at the owner's request."""
    reactions = engine()
    for step in range(30):                              # 3 seconds very close to the camera
        shown = reactions.update(snap(detections=[person(470)]), step / 10)
        assert shown == []


def test_a_close_known_person_is_only_greeted():
    shown = engine().update(snap(SARA, detections=[person(470)]), 0.0)
    assert [r.kind for r in shown] == ["greeting"]


# ---- logging and settings ------------------------------------------------


def test_each_start_and_end_is_logged_once_with_the_reason(caplog):
    reactions = engine()
    with caplog.at_level(logging.INFO, logger="car_vision.reactions"):
        for step in range(60):                          # Sara in view for 6 seconds, close to the camera
            reactions.update(snap(SARA, detections=[person(400)]), step / 10)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(messages) == 2
    assert sum("Greeting started" in m and "Sara Ahmed" in m for m in messages) == 1
    assert sum("Greeting ended" in m and "Sara Ahmed" in m for m in messages) == 1


def test_missing_setting_is_reported():
    settings = dict(SETTINGS)
    del settings["message_seconds"]
    with pytest.raises(ConfigError, match="message_seconds"):
        ReactionEngine(settings)


def test_the_old_warning_settings_are_simply_ignored():
    """A config.yaml from before the warning was removed still starts the program."""
    engine(close_box_height_ratio=0.75, close_hold_seconds=0.5)


def test_negative_seconds_are_rejected():
    with pytest.raises(ConfigError, match="message_seconds"):
        engine(message_seconds=-1)
