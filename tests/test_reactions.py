"""The reaction rules, played with made-up snapshots and times. No camera, model or clock."""

import logging

import pytest

from car_vision.config import ConfigError
from car_vision.reactions import WARNING_TEXT, Reaction, ReactionEngine, tallest_person_ratio
from car_vision.state import UNKNOWN, Detection, RecognizedFace, WorldSnapshot

SETTINGS = {
    "greet_cooldown_seconds": 60,
    "close_box_height_ratio": 0.75,
    "close_hold_seconds": 0.5,
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


def warned(reactions):
    return any(r.kind == "warning" for r in reactions)


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


# ---- the close warning ---------------------------------------------------


def close():
    return [person(400)]            # 400 / 480 = 0.83 of the frame height


def far():
    return [person(200)]            # 0.42


def test_warning_only_after_the_hold_time():
    reactions = engine()
    assert not warned(reactions.update(snap(detections=close()), 0.0))
    assert not warned(reactions.update(snap(detections=close()), 0.25))
    shown = reactions.update(snap(detections=close()), 0.5)
    assert Reaction("warning", WARNING_TEXT) in shown
    assert warned(reactions.update(snap(detections=close()), 5.0))


def test_warning_clears_on_the_first_frame_nobody_is_close():
    reactions = engine()
    reactions.update(snap(detections=close()), 0.0)
    assert warned(reactions.update(snap(detections=close()), 1.0))
    assert not warned(reactions.update(snap(detections=far()), 1.1))
    assert not warned(reactions.update(snap(), 1.2))


def test_short_gap_resets_the_hold():
    reactions = engine()
    reactions.update(snap(detections=close()), 0.0)
    reactions.update(snap(detections=close()), 0.3)
    reactions.update(snap(detections=far()), 0.4)                   # one frame apart
    assert not warned(reactions.update(snap(detections=close()), 0.5))
    assert not warned(reactions.update(snap(detections=close()), 0.75))
    assert warned(reactions.update(snap(detections=close()), 1.0))  # 0.5 s since 0.5


def test_zero_frame_size_gives_no_warning():
    reactions = engine(close_hold_seconds=0)
    for step in range(10):
        assert not warned(reactions.update(snap(detections=close(), frame_size=(0, 0)), step * 0.5))


def test_box_exactly_at_the_ratio_counts_as_close():
    reactions = engine(close_hold_seconds=0)
    assert warned(reactions.update(snap(detections=[person(360)]), 0.0))       # 360 / 480 = 0.75
    assert not warned(reactions.update(snap(detections=[person(359)]), 0.1))


def test_tall_boxes_that_are_not_people_do_not_warn():
    reactions = engine(close_hold_seconds=0)
    assert not warned(reactions.update(snap(detections=[person(470, label="chair")]), 0.0))


def test_wide_frame_uses_its_own_height():
    detections = [person(300, frame_height=360)]     # 300 / 360 = 0.83 in a 640 x 360 picture
    assert tallest_person_ratio(snap(detections=detections, frame_size=(640, 360))) == pytest.approx(300 / 360)
    assert warned(engine(close_hold_seconds=0).update(snap(detections=detections, frame_size=(640, 360)), 0.0))


def test_warning_comes_before_greetings():
    reactions = engine(close_hold_seconds=0)
    shown = reactions.update(snap(SARA, detections=close()), 0.0)
    assert [r.kind for r in shown] == ["warning", "greeting"]


# ---- logging and settings ------------------------------------------------


def test_each_start_and_end_is_logged_once_with_the_reason(caplog):
    reactions = engine()
    with caplog.at_level(logging.INFO, logger="car_vision.reactions"):
        for step in range(20):                          # Sara close in view for 2 seconds
            reactions.update(snap(SARA, detections=close()), step / 10)
        for step in range(20, 60):                      # then she steps back
            reactions.update(snap(SARA, detections=far()), step / 10)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(messages) == 4
    assert sum("Greeting started" in m and "Sara Ahmed" in m for m in messages) == 1
    assert sum("Greeting ended" in m and "Sara Ahmed" in m for m in messages) == 1
    assert sum("Close warning started" in m and "0.83" in m for m in messages) == 1
    assert sum("Close warning ended" in m and "0.42" in m for m in messages) == 1


def test_missing_setting_is_reported():
    settings = dict(SETTINGS)
    del settings["close_hold_seconds"]
    with pytest.raises(ConfigError, match="close_hold_seconds"):
        ReactionEngine(settings)


@pytest.mark.parametrize("ratio", [0, -0.5, 1.5])
def test_ratio_outside_zero_to_one_is_rejected(ratio):
    with pytest.raises(ConfigError, match="close_box_height_ratio"):
        engine(close_box_height_ratio=ratio)


def test_negative_seconds_are_rejected():
    with pytest.raises(ConfigError, match="message_seconds"):
        engine(message_seconds=-1)
