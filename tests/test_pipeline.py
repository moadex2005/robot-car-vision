"""The see-decide-show loop, run with a fake camera, detector and recogniser.

No camera, model, network or graphics card is needed.
"""

import ast
import json
import logging
import sys
import threading
import time
import types
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from car_vision import enroll as enroll_module
from car_vision import pipeline as pipeline_module
from car_vision.config import ConfigError
from car_vision.faces import FaceMatcher, Person, load_embeddings
from car_vision.people import PeopleBook
from car_vision.phone import parse_size
from car_vision.pipeline import (HINT_CANCELLED, HINT_CLOSER, HINT_NO_FACE, HINT_ONE_PERSON, HINT_STILL,
                                 HINT_SWITCHED, NOBODY_ENROLLED, SHARPNESS_THRESHOLD, Pipeline,
                                 next_photo_number, photo_hint, save_photo, sharpness)
from car_vision.reactions import ReactionEngine
from car_vision.recognizer import FaceCheck, Recognizer
from car_vision.state import Detection, RecognizedFace, WorldState

PIPELINE_FILE = Path(__file__).resolve().parent.parent / "car_vision" / "pipeline.py"

CONFIG = {
    "stream": {"source": "file", "phone_url": "http://192.168.0.2:8080/video", "phone_picture": "auto",
               "webcam_index": 0,
               "file_path": "data/does-not-exist/clip.mp4", "width": 640, "height": 480,
               "reconnect_seconds": 0.01, "loop_file": True},
    "detector": {"model": "fake.pt", "models_dir": "models", "confidence": 0.4, "image_size": 640,
                 "device": "cpu", "tracker": "bytetrack.yaml",
                 "open_model": "fake-open.pt", "open_confidence": 0.25, "open_every_n_frames": 3},
    "recognizer": {"model_pack": "buffalo_l", "models_dir": "models", "detection_size": 640,
                   "device": "cpu", "match_threshold": 0.42, "every_n_frames": 3, "min_face_pixels": 40,
                   "people_file": "data/does-not-exist/people.json",
                   "photos_dir": "data/does-not-exist/people",
                   "embeddings_file": "data/does-not-exist/embeddings.npz"},
    # No memory file: the tests that need the memory hand in a fake or a file in tmp_path.
    "memory": {"file": None, "keep_days": 7, "min_seconds": 1.0, "pictures": True},
    "lookout": {"model": "fake/model:provider", "every_seconds": 10, "picture_width": 512, "max_names": 6,
                "daily_budget_usd": 0.5, "input_price_per_million": 0.04,
                "output_price_per_million": 0.05, "spend_file": "data/does-not-exist/lookout.json"},
    "codes": {"enabled": False, "every_n_frames": 5, "max_width": 960, "hold_seconds": 1.5},
    "reactions": {"greet_cooldown_seconds": 60, "message_seconds": 4.0},
    "app": {"host": "127.0.0.1", "port": 5000, "jpeg_quality": 80, "log_file": "logs/test.log"},
}

STATUS_KEYS = {"source", "connected", "fps", "device", "recognition", "notes",
               "people", "objects", "codes", "reactions", "phone_url", "phone_picture", "camera"}

PERSON = Detection("person", 0.9, (100, 60, 260, 300), 1)
CHAIRS = [Detection("chair", 0.8, (400, 200, 500, 340), 2), Detection("chair", 0.7, (520, 200, 600, 340), 3)]
SARA_FACE = RecognizedFace("Sara Ahmed", 0.6123, (150, 70, 210, 140), "sara", track_id=1)
SARA = Person("sara", "Sara Ahmed", "Team lead", ("Built the chassis", "Fourth-year student"))


# ---- fakes ---------------------------------------------------------------------


class FakeSource:
    """Hands out `frames` numbered frames, one per read, then repeats the last one."""

    def __init__(self, frames=5, size=(640, 360), picture=None):
        self.source = "file"
        self.connected = True
        self.ended = False
        self.frames = frames
        self.sequence = 0
        self.started = False
        self.stopped = False
        width, height = size
        self.picture = np.full((height, width, 3), 60, dtype=np.uint8) if picture is None else picture
        self.boxes = []                                 # every box set_box() was given, in order
        self._lock = threading.Lock()

    def set_box(self, max_width, max_height):
        self.boxes.append((max_width, max_height))

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    def reopen(self):
        self.reopens = getattr(self, "reopens", 0) + 1

    def read(self):
        with self._lock:
            if self.sequence < self.frames:
                self.sequence += 1
            if self.sequence == 0:
                return None, 0
            return self.picture.copy(), self.sequence


class FakeDetector:
    device = "cpu"

    def __init__(self, detections=None, fail_on=(), fail_always=False, delay=0.0):
        self.detections = [PERSON, *CHAIRS] if detections is None else list(detections)
        self.fail_on = set(fail_on)
        self.fail_always = fail_always
        self.delay = delay
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail_always or self.calls in self.fail_on:
            raise RuntimeError("the model broke")
        return list(self.detections)


class FakeRecognizer:
    def __init__(self, faces=(SARA_FACE,), people=None, fail_on=()):
        self.faces = list(faces)
        self.people = {"sara": SARA} if people is None else people
        self.fail_on = set(fail_on)
        self.calls = 0

    def recognize(self, frame, detections=()):
        self.calls += 1
        if self.calls in self.fail_on:
            raise ValueError("face model broke")
        return list(self.faces)

    def check_faces(self, frame, detections=()):
        faces = self.recognize(frame, detections)
        return FaceCheck(faces, [face.box for face in faces])

    def person(self, person_id):
        return self.people.get(person_id)


class NoPhone:
    """The network as the tests see it: nothing answers. Every address asked for is kept."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        raise ConnectionRefusedError("nothing listens here in a test")


def wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def make():
    """Build pipelines with fakes; every one is stopped at the end of the test."""
    made = []

    def build(source=None, detector=None, recognizer=None, config=CONFIG, phone_fetch=None, **kwargs):
        pipeline = Pipeline(config,
                            source=source if source is not None else FakeSource(),
                            detector=detector if detector is not None else FakeDetector(),
                            recognizer=recognizer if recognizer is not None else FakeRecognizer(),
                            phone_fetch=phone_fetch if phone_fetch is not None else NoPhone(),   # never the network
                            **kwargs)
        made.append(pipeline)
        return pipeline

    yield build
    for pipeline in made:
        pipeline.stop()


# ---- the loop ---------------------------------------------------------------------


@pytest.mark.parametrize("size", [(640, 360), (640, 480)])
def test_detections_and_faces_reach_the_world_state(make, size):
    state = WorldState()
    detector = FakeDetector()
    pipeline = make(source=FakeSource(frames=4, size=size), detector=detector, state=state).start()
    assert pipeline.state is state
    assert wait_until(lambda: detector.calls == 4 and state.snapshot().faces)
    snapshot = state.snapshot()
    assert list(snapshot.detections) == [PERSON, *CHAIRS]
    assert [f.person_id for f in snapshot.faces] == ["sara"]
    assert snapshot.frame_size == size                  # taken from the frame, not the settings
    assert state.current_people() == ["Sara Ahmed"]     # what the assistant will be told


def test_recogniser_runs_only_on_every_nth_frame(make):
    detector, recognizer = FakeDetector(), FakeRecognizer()
    pipeline = make(source=FakeSource(frames=7), detector=detector, recognizer=recognizer).start()
    assert wait_until(lambda: detector.calls == 7)
    time.sleep(0.05)
    assert recognizer.calls == 3                        # frames 1, 4 and 7 (every_n_frames is 3)
    assert [f.name for f in pipeline.state.snapshot().faces] == ["Sara Ahmed"]   # kept in between


def test_each_frame_is_processed_once(make):
    detector = FakeDetector()
    make(source=FakeSource(frames=3), detector=detector).start()
    assert wait_until(lambda: detector.calls == 3)
    time.sleep(0.1)                                     # the source keeps returning frame 3
    assert detector.calls == 3


def test_latest_jpeg_is_none_before_the_first_frame(make):
    assert make().latest_jpeg() is None


@pytest.mark.parametrize("size", [(640, 360), (640, 480)])
def test_latest_jpeg_returns_decodable_jpeg_bytes(make, size):
    pipeline = make(source=FakeSource(frames=2, size=size)).start()
    assert wait_until(lambda: pipeline.latest_jpeg() is not None)
    jpeg = pipeline.latest_jpeg()
    assert isinstance(jpeg, bytes) and jpeg[:2] == b"\xff\xd8"
    picture = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert picture.shape == (size[1], size[0], 3)
    assert (picture != 60).any()                        # boxes were drawn on it


def test_info_card_is_left_to_the_page(make, monkeypatch):
    cards = []
    monkeypatch.setattr(pipeline_module.draw, "draw_info_card", lambda *args: cards.append(args))
    detector = FakeDetector()
    make(source=FakeSource(frames=3), detector=detector).start()
    assert wait_until(lambda: detector.calls == 3)
    assert cards == []


# ---- status() -------------------------------------------------------------------------


def test_status_has_every_key_and_is_json_serialisable(make):
    # 20 ms per frame: Windows' monotonic clock only moves every 15 ms or so.
    detector = FakeDetector(delay=0.02)
    pipeline = make(source=FakeSource(frames=4), detector=detector).start()
    assert wait_until(lambda: detector.calls == 4 and pipeline.status()["people"])
    status = pipeline.status()
    assert STATUS_KEYS <= set(status)
    assert json.loads(json.dumps(status)) == status
    assert status["source"] == "file"
    assert status["phone_url"] == "http://192.168.0.2:8080/video"      # from the settings
    assert status["connected"] is True
    assert status["device"] == "cpu"
    assert status["recognition"] is True
    assert status["notes"] == []
    assert status["error"] is None
    assert isinstance(status["fps"], float) and status["fps"] > 0
    assert [(o["label"], o["track_id"], o["name"]) for o in status["objects"]] == [
        ("person", 1, "Sara Ahmed"), ("chair", 2, None), ("chair", 3, None)]
    assert status["people"] == [{"id": "sara", "name": "Sara Ahmed", "role": "Team lead",
                                 "lines": ["Built the chassis", "Fourth-year student"],
                                 "similarity": 0.61}]
    assert status["reactions"] == [{"kind": "greeting", "text": "Hello, Sara Ahmed!"}]


def test_status_before_any_frame_is_still_complete(make):
    status = make().status()
    assert STATUS_KEYS <= set(status)
    assert status["people"] == status["objects"] == status["reactions"] == []
    json.dumps(status)


def test_known_face_without_an_info_card_is_shown_by_its_id(make):
    ghost = RecognizedFace("ghost", 0.5, (150, 70, 210, 140), "ghost", track_id=1)
    pipeline = make(recognizer=FakeRecognizer(faces=[ghost], people={})).start()
    assert wait_until(lambda: pipeline.status()["people"])
    assert pipeline.status()["people"] == [{"id": "ghost", "name": "ghost", "role": "",
                                            "lines": [], "similarity": 0.5}]


def test_a_person_close_to_the_camera_gives_no_warning_in_the_status(make):
    """The close warning was removed at the owner's request: only greetings are reactions now."""
    tall = Detection("person", 0.9, (200, 10, 400, 350), 1)          # 340 / 360 of the frame
    detector = FakeDetector(detections=[tall])
    pipeline = make(detector=detector, recognizer=FakeRecognizer(faces=[]),
                    reactions=ReactionEngine(CONFIG["reactions"])).start()
    assert wait_until(lambda: detector.calls >= 3)
    assert pipeline.status()["reactions"] == []


# ---- the objects list ------------------------------------------------------------------------


def test_objects_has_one_entry_per_detection_people_first_then_label_then_track(make):
    detections = [Detection("chair", 0.71, (520, 200, 600, 340), 3),
                  Detection("laptop", 0.834, (300, 100, 380, 160), None),
                  Detection("person", 0.5, (400, 20, 500, 350), 5),
                  Detection("chair", 0.8, (400, 200, 500, 340), 2),
                  PERSON]
    pipeline = make(detector=FakeDetector(detections=detections)).start()
    assert wait_until(lambda: len(pipeline.status()["objects"]) == 5 and pipeline.status()["objects"][0]["name"])
    objects = pipeline.status()["objects"]
    assert [(o["label"], o["track_id"]) for o in objects] == [
        ("person", 1), ("person", 5), ("chair", 2), ("chair", 3), ("laptop", None)]
    assert objects[0] == {"track_id": 1, "label": "person", "confidence": 0.9,
                          "box": [round(100 / 640, 3), round(60 / 360, 3), round(260 / 640, 3), round(300 / 360, 3)],
                          "name": "Sara Ahmed"}
    assert objects[1]["name"] is None                   # nobody recognised in that box
    assert objects[4]["confidence"] == 0.83
    json.dumps(objects)


STRANGER_BEHIND = Detection("person", 0.7, (80, 40, 300, 340), 5)     # Sara's face centre lies inside it too


def test_a_stranger_behind_a_recognised_person_does_not_get_their_name(make):
    pipeline = started(make, detections=[PERSON, STRANGER_BEHIND])
    assert wait_until(lambda: pipeline.status()["objects"])
    names = {o["track_id"]: o["name"] for o in pipeline.status()["objects"]}
    assert names == {1: "Sara Ahmed", 5: None}          # one pill with her name, not two
    assert pipeline.target(track_id=1).person == SARA
    assert pipeline.target(track_id=5).person is None


def owners(detections, faces):
    return {i: face.person_id for i, face in pipeline_module.face_owners(detections, faces).items()}


def test_a_face_with_a_track_number_belongs_only_to_that_box():
    elsewhere = RecognizedFace("Sara Ahmed", 0.6, (150, 70, 210, 140), "sara", track_id=9)   # box 9 not in view
    assert owners([PERSON], [elsewhere]) == {}


def test_a_face_without_a_track_number_goes_to_the_smallest_box_around_it():
    untracked = RecognizedFace("Sara Ahmed", 0.6, (150, 70, 210, 140), "sara", track_id=None)
    assert owners([STRANGER_BEHIND, PERSON], [untracked]) == {1: "sara"}    # PERSON is the smaller box


def test_a_box_with_a_face_of_its_own_gets_no_other_face():
    own_unknown = RecognizedFace("Unknown", 0.2, (160, 80, 200, 130), None, track_id=1)
    untracked = RecognizedFace("Sara Ahmed", 0.6, (150, 70, 210, 140), "sara", track_id=None)
    assert owners([PERSON, STRANGER_BEHIND], [own_unknown, untracked]) == {1: "sara"}


def test_one_person_is_never_given_to_two_boxes():
    tracked = RecognizedFace("Sara Ahmed", 0.6, (150, 70, 210, 140), "sara", track_id=1)
    again = RecognizedFace("Sara Ahmed", 0.5, (420, 60, 460, 110), "sara", track_id=None)
    other = Detection("person", 0.8, (400, 40, 500, 340), 6)
    assert owners([other, PERSON], [again, tracked]) == {1: "sara"}         # the matching track wins
    small = Detection("person", 0.8, (390, 40, 480, 300), 7)
    large = Detection("person", 0.8, (380, 20, 600, 350), None)
    twice = [RecognizedFace("Sara Ahmed", 0.5, (420, 60, 460, 110), "sara", track_id=None),
             RecognizedFace("Sara Ahmed", 0.5, (540, 60, 580, 110), "sara", track_id=None)]
    assert owners([large, small], twice) == {1: "sara"}                     # else the smallest box


def test_box_fractions_stay_between_0_and_1():
    assert pipeline_module.box_fractions((-10, 0, 700, 360), (640, 360)) == [0.0, 0.0, 1.0, 1.0]
    assert pipeline_module.box_fractions((1, 2, 3, 4), (0, 0)) == [0.0, 0.0, 0.0, 0.0]


# ---- target(): what a click points at ---------------------------------------------------------


def started(make, detections=(PERSON, *CHAIRS), faces=(SARA_FACE,), people=None, **kwargs):
    """A pipeline that has processed its first frame, so target() has a view."""
    pipeline = make(detector=FakeDetector(detections=detections),
                    recognizer=FakeRecognizer(faces=faces, people=people), **kwargs).start()
    assert wait_until(lambda: pipeline.latest_jpeg() is not None)
    return pipeline


def photo_of(target):
    picture = cv2.imdecode(np.frombuffer(target.photo, np.uint8), cv2.IMREAD_COLOR)
    assert picture is not None
    return picture


def at(x, y, size=(640, 360)):
    """A pixel position as the fractions the page sends."""
    return {"x": x / size[0], "y": y / size[1]}


def test_target_is_none_before_the_first_frame(make):
    assert make().target(x=0.5, y=0.5) is None
    assert make().target(track_id=1) is None


def test_target_by_track_id(make):
    target = started(make).target(track_id=2)
    assert (target.kind, target.label, target.track_id, target.person) == ("object", "chair", 2, None)
    assert target.confidence == pytest.approx(0.8)
    # The chair box (400, 200, 500, 340) grown by 15 %, at least 160 wide, cut off at the frame's bottom.
    assert photo_of(target).shape == (181, 160, 3)


def test_target_by_track_id_and_label(make):
    pipeline = started(make)
    assert pipeline.target(track_id=2, label="chair").track_id == 2
    assert pipeline.target(track_id=2, label="laptop") is None        # that track is something else
    assert pipeline.target(track_id=99) is None                       # no longer in view


def test_target_by_track_id_ignores_the_point(make):
    target = started(make).target(track_id=3, **at(5, 5))
    assert (target.label, target.track_id) == ("chair", 3)


def test_target_by_point(make):
    target = started(make).target(**at(450, 270))
    assert (target.kind, target.label, target.track_id) == ("object", "chair", 2)


def test_the_smallest_box_wins(make):
    phone = Detection("cell phone", 0.6, (200, 150, 240, 200), 4)     # held in front of the person
    pipeline = started(make, detections=[PERSON, phone], faces=[])
    assert pipeline.target(**at(220, 175)).label == "cell phone"
    assert pipeline.target(**at(130, 250)).label == "person"


def test_a_label_limits_the_boxes_that_count(make):
    phone = Detection("cell phone", 0.6, (200, 150, 240, 200), 4)
    pipeline = started(make, detections=[PERSON, phone], faces=[])
    assert pipeline.target(label="person", **at(220, 175)).label == "person"


def test_a_label_that_is_no_longer_there_gives_none(make):
    pipeline = started(make)
    assert pipeline.target(label="laptop", **at(450, 270)) is None
    assert pipeline.target(label="chair", **at(20, 20)) is None       # no area when a label was asked for


def test_a_click_on_a_face_gives_its_person_box(make):
    target = started(make).target(**at(180, 105))                     # inside Sara's face box
    assert (target.kind, target.label, target.track_id) == ("person", "person", 1)
    assert target.person == SARA
    assert target.confidence == pytest.approx(0.9)


def test_a_face_box_with_no_person_box_is_the_target_itself(make):
    face = RecognizedFace("Sara Ahmed", 0.6, (300, 100, 360, 170), "sara", track_id=None)
    target = started(make, detections=list(CHAIRS), faces=[face]).target(**at(330, 135))
    assert (target.kind, target.label, target.track_id) == ("person", "face", None)
    assert target.person == SARA
    assert photo_of(target).shape == (160, 160, 3)                    # the face box, grown to 160 pixels


def test_a_recognised_person_by_track_id(make):
    target = started(make).target(**at(120, 280))                     # the person box, below the face
    assert target.kind == "person" and target.person == SARA


def test_a_known_face_counts_when_its_centre_is_inside_the_person_box(make):
    untracked = RecognizedFace("Sara Ahmed", 0.6, (150, 70, 210, 140), "sara", track_id=None)
    target = started(make, faces=[untracked]).target(track_id=1)
    assert target.person == SARA


def test_an_unrecognised_person(make):
    stranger = RecognizedFace("Unknown", 0.2, (150, 70, 210, 140), None, track_id=1)
    pipeline = started(make, faces=[stranger])
    target = pipeline.target(track_id=1)
    assert (target.kind, target.label, target.person) == ("person", "person", None)
    assert pipeline.target(**at(180, 105)).person is None              # a click on the unknown face


def test_a_known_face_without_an_about_file_still_gives_the_name(make):
    target = started(make, people={}).target(track_id=1)
    assert (target.person.person_id, target.person.name) == ("sara", "Sara Ahmed")


def test_a_click_on_nothing_gives_the_area_around_it(make):
    target = started(make, detections=[], faces=[]).target(x=0.5, y=0.5)
    assert (target.kind, target.label, target.confidence, target.track_id, target.person) == (
        "area", "", 0.0, None, None)
    # Half the frame (320 x 180) centred on the click, then grown by 15 % on each side.
    assert photo_of(target).shape == (234, 416, 3)


def test_the_area_is_centred_on_the_click_and_pushed_back_inside_at_an_edge():
    assert pipeline_module.area_box((640, 360), 320, 180) == (160, 90, 480, 270)
    assert pipeline_module.area_box((640, 360), 400, 100) == (240, 10, 560, 190)
    assert pipeline_module.area_box((640, 360), 5, 355) == (0, 180, 320, 360)
    assert pipeline_module.area_box((640, 360), 640, 0) == (320, 0, 640, 180)


def test_the_photo_is_cut_from_the_clean_frame(make):
    # Plain grey frame: the detector's coloured boxes and the face names must not be in the photo.
    pipeline = started(make)
    drawn = cv2.imdecode(np.frombuffer(pipeline.latest_jpeg(), np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(drawn.astype(int) - 60).max() > 50                  # the video has boxes on it
    for target in (pipeline.target(track_id=1), pipeline.target(track_id=2), pipeline.target(x=0.9, y=0.1)):
        assert np.abs(photo_of(target).astype(int) - 60).max() <= 3


def test_the_photo_shows_the_right_part_of_the_frame(make):
    picture = np.zeros((360, 640, 3), np.uint8)
    picture[200:340, 400:500] = (0, 0, 255)                           # a red block where chair 2 is
    pipeline = started(make, source=FakeSource(frames=1, picture=picture))
    photo = photo_of(pipeline.target(track_id=2)).astype(int)
    height, width = photo.shape[:2]
    assert photo[height // 2, width // 2, 2] > 200                    # red in the middle
    assert photo[2, 2].sum() < 30                                     # the margin around it is dark


def test_the_photo_is_never_larger_than_the_frame(make):
    whole = Detection("tv", 0.9, (0, 0, 100, 80), 1)
    tiny = Detection("mouse", 0.9, (10, 10, 20, 20), 2)
    pipeline = started(make, detections=[whole, tiny], faces=[], source=FakeSource(frames=1, size=(100, 80)))
    assert photo_of(pipeline.target(track_id=1)).shape == (80, 100, 3)
    assert photo_of(pipeline.target(track_id=2)).shape == (80, 100, 3)   # 160 is more than the frame allows


def test_photo_box_grows_by_15_percent_and_to_160_pixels():
    assert pipeline_module.photo_box((100, 100, 300, 300), (640, 480)) == (70, 70, 330, 330)
    assert pipeline_module.photo_box((300, 200, 320, 220), (640, 480)) == (230, 130, 390, 290)
    assert pipeline_module.photo_box((0, 0, 20, 20), (640, 480)) == (0, 0, 160, 160)   # kept inside
    assert pipeline_module.photo_box((500, 0, 640, 480), (640, 480)) == (479, 0, 640, 480)   # clipped


def test_target_is_safe_to_call_while_the_loop_runs(make):
    pipeline = started(make, source=FakeSource(frames=300))
    errors = []

    def click_many():
        try:
            for _ in range(200):
                target = pipeline.target(track_id=2)
                assert target is not None and target.label == "chair"
                photo_of(target)
        except Exception as error:                       # reported below, in the test's own thread
            errors.append(error)

    threads = [threading.Thread(target=click_many) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []


# ---- failures ---------------------------------------------------------------------------


def test_a_frame_that_makes_the_detector_raise_is_skipped(make):
    detector = FakeDetector(fail_on={2})
    pipeline = make(source=FakeSource(frames=5), detector=detector).start()
    assert wait_until(lambda: detector.calls == 5)
    time.sleep(0.05)
    status = pipeline.status()
    assert status["error"] is None
    assert status["objects"]
    assert pipeline.latest_jpeg() is not None


def test_a_frame_that_makes_the_recogniser_raise_is_skipped(make):
    detector, recognizer = FakeDetector(), FakeRecognizer(fail_on={1})
    pipeline = make(source=FakeSource(frames=6), detector=detector, recognizer=recognizer).start()
    assert wait_until(lambda: detector.calls == 6)
    time.sleep(0.05)
    assert pipeline.status()["error"] is None
    assert recognizer.calls >= 2


def test_ten_failures_in_a_row_stop_the_loop_with_an_error(make):
    detector = FakeDetector(fail_always=True)
    pipeline = make(source=FakeSource(frames=50), detector=detector).start()
    assert wait_until(lambda: pipeline.status()["error"] is not None)
    time.sleep(0.1)
    assert detector.calls == 10                         # nothing more after the tenth
    error = pipeline.status()["error"]
    assert "10 frames in a row" in error and "the model broke" in error
    assert pipeline.status()["connected"] is False      # the page must not say Live any more
    json.dumps(pipeline.status())


def test_failures_with_good_frames_between_them_do_not_stop_the_loop(make):
    detector = FakeDetector(fail_on=set(range(1, 40, 2)))       # every other frame fails
    pipeline = make(source=FakeSource(frames=40), detector=detector).start()
    assert wait_until(lambda: detector.calls == 40)
    time.sleep(0.05)
    assert pipeline.status()["error"] is None


def test_stream_lost_keeps_the_last_frame_with_a_reconnecting_line(make):
    source = FakeSource(frames=3)
    detector = FakeDetector()
    pipeline = make(source=source, detector=detector).start()
    assert wait_until(lambda: detector.calls == 3)
    time.sleep(0.05)
    before = pipeline.latest_jpeg()
    source.connected = False
    assert wait_until(lambda: pipeline.latest_jpeg() is not before)
    after = pipeline.latest_jpeg()
    first = cv2.imdecode(np.frombuffer(before, np.uint8), cv2.IMREAD_COLOR)
    second = cv2.imdecode(np.frombuffer(after, np.uint8), cv2.IMREAD_COLOR)
    assert first.shape == second.shape
    assert np.abs(first[:40].astype(int) - second[:40].astype(int)).sum() > 1000   # the text line
    time.sleep(0.05)
    assert pipeline.latest_jpeg() is after              # drawn once, not on every wait
    assert pipeline.status()["connected"] is False


def decoded(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).astype(int)


def test_a_finished_clip_says_so_instead_of_reconnecting(make):
    source = FakeSource(frames=3)
    detector = FakeDetector()
    pipeline = make(source=source, detector=detector).start()
    assert wait_until(lambda: detector.calls == 3)
    time.sleep(0.05)
    assert pipeline.status()["ended"] is False
    playing = pipeline.latest_jpeg()
    source.connected = False                            # the reader sees the end of the clip ...
    assert wait_until(lambda: pipeline.latest_jpeg() is not playing)
    reconnecting = pipeline.latest_jpeg()
    source.ended = True                                 # ... and a moment later knows it ended
    assert wait_until(lambda: pipeline.latest_jpeg() is not reconnecting)
    assert pipeline._lost_line == pipeline_module.CLIP_FINISHED
    assert np.abs(decoded(reconnecting)[:40] - decoded(pipeline.latest_jpeg())[:40]).sum() > 1000
    status = pipeline.status()
    assert status["ended"] is True and status["connected"] is False
    finished = pipeline.latest_jpeg()
    time.sleep(0.05)
    assert pipeline.latest_jpeg() is finished           # drawn once


def test_status_leaves_out_old_sightings_while_the_stream_is_down(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "STALE_SECONDS", 0.2)
    source = FakeSource(frames=3)
    pipeline = make(source=source).start()
    assert wait_until(lambda: pipeline.status()["objects"])
    source.connected = False
    assert wait_until(lambda: not pipeline.status()["objects"])
    status = pipeline.status()
    assert status["people"] == [] and status["reactions"] == []


# ---- use_phone(): switching to another phone while the loop runs ----------------------------

PHONE = "http://192.168.1.7:8080/video"


def phone_source(frames=3, size=(320, 240), value=200, connected=True):
    """A fake phone with its own picture size and colour, so its frames can be told apart."""
    width, height = size
    source = FakeSource(frames=frames, picture=np.full((height, width, 3), value, np.uint8))
    source.source = "phone"
    source.connected = connected
    return source


class SourceFactory:
    """Stands in for FrameSource: hands out prepared fakes in order and keeps the settings."""

    def __init__(self, *sources):
        self.sources = list(sources)
        self.settings = []

    def __call__(self, settings):
        self.settings.append(settings)
        return self.sources.pop(0)


def test_use_phone_uses_the_new_source_and_stops_the_old_one(make):
    old, new = FakeSource(frames=3), phone_source()
    factory = SourceFactory(new)
    pipeline = make(source=old, source_factory=factory).start()
    assert wait_until(lambda: pipeline.latest_jpeg() is not None)
    pipeline.use_phone(PHONE)
    assert factory.settings == [{**CONFIG["stream"], "source": "phone", "phone_url": PHONE}]
    assert new.started
    assert wait_until(lambda: old.stopped)
    # Frames from the new source reach the state and the view.
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (320, 240))
    assert wait_until(lambda: pipeline.target(track_id=2) is not None)
    assert np.abs(photo_of(pipeline.target(track_id=2)).astype(int) - 200).max() <= 3
    status = pipeline.status()
    assert (status["source"], status["phone_url"], status["connected"]) == ("phone", PHONE, True)
    assert not new.stopped


def test_use_phone_clears_the_view_and_the_old_sightings_at_once(make):
    new = phone_source(frames=0, connected=False)          # a phone that sends nothing yet
    pipeline = started(make, source=FakeSource(frames=3), source_factory=SourceFactory(new))
    assert pipeline.target(track_id=2) is not None
    assert wait_until(lambda: pipeline.status()["objects"])
    pipeline.use_phone(PHONE)
    assert pipeline.target(track_id=2) is None              # a click cannot hit the old camera's picture
    assert pipeline.target(x=0.5, y=0.5) is None
    status = pipeline.status()
    assert status["objects"] == status["people"] == status["reactions"] == []
    assert status["connected"] is False and status["error"] is None
    assert wait_until(lambda: pipeline._lost_line == "Connecting to the phone ...")


def test_use_phone_clears_what_the_assistant_is_told_at_once(make):
    new = phone_source(frames=0, connected=False)
    pipeline = started(make, source=FakeSource(frames=3), source_factory=SourceFactory(new))
    assert wait_until(lambda: pipeline.state.current_people() == ["Sara Ahmed"])
    pipeline.use_phone(PHONE)
    snapshot = pipeline.state.snapshot()
    assert snapshot.detections == () and snapshot.faces == ()
    assert pipeline.state.current_people() == []          # no "I see Sara" about a camera that is gone
    assert snapshot.frame_size == (640, 360)              # the size is left as it was


def test_while_switching_the_picture_is_plain_dark_with_the_connecting_line(make):
    new = phone_source(frames=0, connected=False)
    pipeline = started(make, source=FakeSource(frames=3), source_factory=SourceFactory(new))
    old_jpeg = pipeline.latest_jpeg()
    old = decoded(old_jpeg)
    pipeline.use_phone(PHONE)

    def dark_below_the_line():
        picture = decoded(pipeline.latest_jpeg())
        return np.abs(picture[60:] - pipeline_module.DARK_LEVEL).max() <= 4   # no boxes, nothing of the old camera

    assert wait_until(dark_below_the_line)
    assert pipeline._lost_line == "Connecting to the phone ..."
    picture = decoded(pipeline.latest_jpeg())
    assert picture.shape == old.shape                     # the same size as before
    assert np.abs(picture[:40] - pipeline_module.DARK_LEVEL).max() > 100    # the line is written on it
    time.sleep(0.05)
    assert dark_below_the_line()                          # and the old picture does not come back


def test_status_never_shows_a_login_from_the_settings(make):
    config = {**CONFIG, "stream": {**CONFIG["stream"], "phone_url": "http://me:secret@192.168.0.2:8080/video"}}
    pipeline = make(config=config)
    assert pipeline.status()["phone_url"] == "http://192.168.0.2:8080/video"


class GateDetector(FakeDetector):
    """Holds its second frame until the test lets it go."""

    def __init__(self):
        super().__init__()
        self.holding = threading.Event()
        self.go_on = threading.Event()

    def detect(self, frame):
        if self.calls == 1:
            self.holding.set()
            assert self.go_on.wait(5)
        return super().detect(frame)


def test_a_frame_of_the_old_camera_still_being_worked_on_is_dropped(make):
    detector = GateDetector()
    new = phone_source(frames=0, connected=False)
    pipeline = make(source=FakeSource(frames=2), detector=detector, source_factory=SourceFactory(new)).start()
    assert detector.holding.wait(3)                         # frame 2 of the old camera is in the models
    pipeline.use_phone(PHONE)
    detector.go_on.set()
    assert wait_until(lambda: detector.calls == 2)
    time.sleep(0.05)
    assert pipeline.target(track_id=2) is None
    assert pipeline.status()["objects"] == []


def test_the_loop_starts_again_after_ten_failures(make):
    detector = FakeDetector(fail_on=set(range(1, 11)))      # the first ten frames fail, then all is well
    pipeline = make(source=FakeSource(frames=50), detector=detector,
                    source_factory=SourceFactory(phone_source(frames=3))).start()
    assert wait_until(lambda: pipeline.status()["error"] is not None)
    time.sleep(0.05)
    assert detector.calls == 10
    pipeline.use_phone(PHONE)
    assert pipeline.status()["error"] is None
    assert wait_until(lambda: pipeline.target(track_id=2) is not None)
    assert pipeline.status()["connected"] is True
    assert pipeline.state.snapshot().frame_size == (320, 240)


def test_the_loop_goes_on_with_a_phone_after_a_clip_finished(make):
    clip = FakeSource(frames=2)
    pipeline = started(make, source=clip, source_factory=SourceFactory(phone_source(frames=3)))
    clip.connected, clip.ended = False, True
    assert wait_until(lambda: pipeline._lost_line == pipeline_module.CLIP_FINISHED)
    pipeline.use_phone(PHONE)
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (320, 240))
    assert pipeline.status()["ended"] is False


class SlowToStopSource(FakeSource):
    """stop() hangs like a FrameSource whose reader waits on a phone that is gone."""

    def __init__(self):
        super().__init__(frames=3)
        self.stopping = threading.Event()
        self.may_stop = threading.Event()

    def stop(self):
        self.stopping.set()
        self.may_stop.wait(5)
        super().stop()


def test_use_phone_does_not_wait_for_the_old_camera_to_stop(make):
    old = SlowToStopSource()
    pipeline = started(make, source=old, source_factory=SourceFactory(phone_source()))
    try:
        began = time.monotonic()
        pipeline.use_phone(PHONE)
        assert time.monotonic() - began < 0.5
        assert old.stopping.wait(2)                         # the old one is stopping ...
        began = time.monotonic()
        pipeline.status()                                   # ... and the lock is free meanwhile
        pipeline.latest_jpeg()
        pipeline.target(x=0.5, y=0.5)
        assert time.monotonic() - began < 0.5
        assert wait_until(lambda: pipeline.state.snapshot().frame_size == (320, 240))   # the new one runs
        assert not old.stopped
    finally:
        old.may_stop.set()
    assert wait_until(lambda: old.stopped)


def test_switches_at_the_same_time_are_safe_and_the_last_one_wins(make):
    phones = [phone_source(frames=3) for _ in range(6)]
    factory = SourceFactory(*phones)
    old = FakeSource(frames=1000)
    pipeline = started(make, source=old, source_factory=factory)
    addresses = [f"http://192.168.1.{n}:8080/video" for n in range(10, 16)]
    threads = [threading.Thread(target=pipeline.use_phone, args=(address,)) for address in addresses]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    last = phones[-1]                                       # the factory is called one switch at a time
    assert pipeline.status()["phone_url"] == factory.settings[-1]["phone_url"]
    assert wait_until(lambda: old.stopped and all(phone.stopped for phone in phones[:-1]))
    assert last.started and not last.stopped
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (320, 240))


def test_use_phone_before_start_starts_nothing_until_start(make):
    old, new = FakeSource(), phone_source()
    pipeline = make(source=old, source_factory=SourceFactory(new))
    pipeline.use_phone(PHONE)
    assert not new.started and not old.started
    pipeline.start()
    assert new.started and not old.started
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (320, 240))


def test_use_phone_after_stop_starts_nothing(make):
    new = phone_source()
    pipeline = started(make, source=FakeSource(frames=3), source_factory=SourceFactory(new))
    pipeline.stop()
    pipeline.use_phone(PHONE)
    assert not new.started
    assert pipeline.status()["phone_url"] == PHONE


def test_without_a_factory_the_phone_source_is_a_frame_source(make, monkeypatch):
    built = []
    monkeypatch.setattr(pipeline_module, "FrameSource", lambda settings: built.append(settings) or phone_source())
    pipeline = make(source=FakeSource(frames=1))
    pipeline.use_phone(PHONE)
    assert built == [{**CONFIG["stream"], "source": "phone", "phone_url": PHONE}]


# ---- the phone's picture size ---------------------------------------------------------------

REAL_OFFERED = ["960x540", "800x600", "3840x2160", "1920x1080", "1280x720", "1600x1200",
                "640x480", "640x360", "352x288", "320x240", "176x144"]
BY_AREA = ["176x144", "320x240", "352x288", "640x360", "640x480", "800x600", "960x540",
           "1280x720", "1600x1200", "1920x1080", "3840x2160"]


class FakePhoneApp:
    """Answers like IP Webcam: its status with the current size, and "Ok" to a new size.

    `lies` answers Ok but keeps the old size; `gate` holds every answer until it is set.
    """

    def __init__(self, current="1920x1080", offered=REAL_OFFERED, lies=False, gate=None):
        self.current = current
        self.offered = list(offered)
        self.lies = lies
        self.gate = gate
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, url, timeout):
        assert timeout == 3.0
        with self._lock:
            self.calls.append(url)
        if self.gate is not None:
            assert self.gate.wait(5)
        if url.endswith("/status.json?show_avail=1"):
            return json.dumps({"curvals": {"video_size": self.current}, "avail": {"video_size": self.offered}})
        size = url.partition("/settings/video_size?set=")[2]
        if size in self.offered and not self.lies:
            self.current = size
        return '<?xml version="1.0" ?>\n<result>Ok</result>\n'

    @property
    def sizes_asked(self):
        with self._lock:
            return [url.partition("set=")[2] for url in self.calls if "set=" in url]


class Network:
    """Several fake phones, each at its own "host:port"."""

    def __init__(self, apps_by_host):
        self.apps = apps_by_host

    def __call__(self, url, timeout):
        return self.apps[url.split("/")[2]](url, timeout)


def picture_config(choice):
    return {**CONFIG, "stream": {**CONFIG["stream"], "phone_picture": choice}}


def picture_of(pipeline):
    return pipeline.status()["phone_picture"]


def phone_at_start(make, app, choice="auto", **kwargs):
    """A pipeline whose source is the phone from the start."""
    return make(source=phone_source(), phone_fetch=app, config=picture_config(choice), **kwargs).start()


def test_auto_asks_the_phone_for_the_best_size_and_reports_what_it_reads_back(make):
    app = FakePhoneApp()
    pipeline = phone_at_start(make, app)
    assert wait_until(lambda: picture_of(pipeline) and picture_of(pipeline)["current"] == "640x360")
    assert picture_of(pipeline) == {"current": "640x360", "offered": BY_AREA, "choice": "auto"}
    phone = "http://192.168.0.2:8080"
    assert app.calls == [phone + "/status.json?show_avail=1", phone + "/settings/video_size?set=640x360",
                         phone + "/status.json?show_avail=1"]
    json.dumps(pipeline.status())


def test_the_offered_sizes_survive_a_phone_that_lists_one_size_while_its_camera_restarts(make):
    """Seen on the real phone: right after a size change its status listed only the old size."""
    class RestartingPhone(FakePhoneApp):
        def __call__(self, url, timeout):
            answer = super().__call__(url, timeout)
            if "set=" in url:
                self.offered = ["1920x1080"]          # what it lists for a moment afterwards
            return answer

    pipeline = phone_at_start(make, RestartingPhone())
    assert wait_until(lambda: picture_of(pipeline) and picture_of(pipeline)["current"] == "640x360")
    assert picture_of(pipeline) == {"current": "640x360", "offered": BY_AREA, "choice": "auto"}


def test_the_stream_is_opened_again_after_the_phone_changed_its_size(make, monkeypatch):
    """Seen on the real phone: the stream reader keeps the frame size it found when it opened
    the stream, so after a change the frames stayed 640 x 360 until the stream was reopened."""
    monkeypatch.setattr(pipeline_module, "STREAM_SETTLE_SECONDS", 0.01)
    source = phone_source(frames=3)
    pipeline = make(source=source, phone_fetch=FakePhoneApp(), config=picture_config("auto")).start()
    assert wait_until(lambda: getattr(source, "reopens", 0) == 1)      # 1920x1080 -> 640x360
    pipeline.set_phone_picture("1280x720")
    assert wait_until(lambda: getattr(source, "reopens", 0) == 2)


def test_the_stream_is_left_alone_when_the_phone_already_sends_the_chosen_size(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "STREAM_SETTLE_SECONDS", 0.01)
    source = phone_source(frames=3)
    app = FakePhoneApp(current="640x360")
    pipeline = make(source=source, phone_fetch=app, config=picture_config("auto")).start()
    assert wait_until(lambda: picture_of(pipeline) is not None)
    time.sleep(0.1)
    assert getattr(source, "reopens", 0) == 0 and app.sizes_asked == []


def test_the_chosen_size_is_asked_for_again_when_the_phone_drops_out_and_returns(make):
    """Seen on the real phone: its camera app restarted and sent 1920 x 1080 again."""
    app = FakePhoneApp()
    source = phone_source(frames=3)
    pipeline = make(source=source, phone_fetch=app, config=picture_config("auto")).start()
    assert wait_until(lambda: picture_of(pipeline) and picture_of(pipeline)["current"] == "640x360")
    assert app.sizes_asked == ["640x360"]

    source.connected = False                                  # the app restarts ...
    assert wait_until(lambda: pipeline._lost_line is not None)
    app.current = "1920x1080"                                 # ... and comes back with its own size
    source.connected = True
    source.frames += 3
    assert wait_until(lambda: app.sizes_asked == ["640x360", "640x360"])
    assert wait_until(lambda: picture_of(pipeline)["current"] == "640x360")


def test_auto_asks_once_for_each_connect(make):
    app = FakePhoneApp()
    pipeline = make(source=FakeSource(frames=3), phone_fetch=app,
                    source_factory=SourceFactory(phone_source(), phone_source())).start()
    pipeline.use_phone("http://192.168.0.2:8080/video")
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "640x360")
    time.sleep(0.05)
    assert app.sizes_asked == ["640x360"]
    app.current = "1920x1080"                          # someone changed it on the phone meanwhile
    pipeline.use_phone("http://192.168.0.2:8080/video")
    assert wait_until(lambda: app.sizes_asked == ["640x360", "640x360"])
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "640x360")


def test_a_phone_already_at_the_best_size_is_not_asked(make):
    app = FakePhoneApp(current="640x360")
    pipeline = phone_at_start(make, app)
    assert wait_until(lambda: picture_of(pipeline) is not None)
    time.sleep(0.05)
    assert app.sizes_asked == [] and len(app.calls) == 1
    assert picture_of(pipeline)["current"] == "640x360"


def test_current_is_what_the_phone_says_after_the_change_not_what_was_asked(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "READ_BACK_WAIT_SECONDS", 0.01)
    app = FakePhoneApp(lies=True)                      # says Ok, keeps sending 1920 x 1080
    pipeline = phone_at_start(make, app)
    assert wait_until(lambda: len(app.calls) == 4)     # status, ask, read back, read back once more
    time.sleep(0.05)
    assert len(app.calls) == 4
    assert picture_of(pipeline)["current"] == "1920x1080"


def test_keep_asks_nothing(make):
    app = FakePhoneApp()
    pipeline = phone_at_start(make, app, choice="keep")
    assert wait_until(lambda: picture_of(pipeline) is not None)
    time.sleep(0.05)
    assert app.sizes_asked == []
    assert picture_of(pipeline) == {"current": "1920x1080", "offered": BY_AREA, "choice": "keep"}


def test_a_fixed_size_is_asked_for(make):
    app = FakePhoneApp()
    pipeline = phone_at_start(make, app, choice="1280x720")
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "1280x720")
    assert app.sizes_asked == ["1280x720"]
    assert picture_of(pipeline)["choice"] == "1280x720"


def test_a_fixed_size_the_phone_does_not_offer_is_ignored_with_one_warning(make, caplog):
    caplog.set_level(logging.INFO, logger="car_vision.pipeline")
    app = FakePhoneApp()
    pipeline = phone_at_start(make, app, choice="1234x567")
    assert wait_until(lambda: picture_of(pipeline) is not None)
    time.sleep(0.05)
    assert app.sizes_asked == []
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and "1234x567" in warnings[0].getMessage()
    assert picture_of(pipeline)["current"] == "1920x1080"


@pytest.mark.parametrize("kind", ["webcam", "file"])
def test_the_webcam_and_a_clip_never_ask_anything(make, kind):
    app = FakePhoneApp()
    source = FakeSource(frames=3)
    source.source = kind
    pipeline = make(source=source, phone_fetch=app).start()
    assert wait_until(lambda: pipeline.latest_jpeg() is not None)
    pipeline.set_phone_picture("auto")
    pipeline.set_phone_picture("keep")
    time.sleep(0.05)
    assert app.calls == []
    assert picture_of(pipeline) is None
    assert source.boxes == []                               # its box is never changed


def test_another_camera_app_changes_nothing_and_shows_nothing(make):
    nothing = NoPhone()
    detector = FakeDetector()
    pipeline = make(source=phone_source(frames=5), detector=detector, phone_fetch=nothing).start()
    assert wait_until(lambda: nothing.calls and detector.calls == 5)
    time.sleep(0.05)
    # Asked at start-up and once more on the first picture (it might not have been serving yet),
    # then never again: not for each of the other four frames. (When the first picture comes
    # before the start-up question went out, that question gives way and it is asked once.)
    assert set(nothing.calls) == {"http://192.168.0.2:8080/status.json?show_avail=1"}
    assert 1 <= len(nothing.calls) <= 2
    assert picture_of(pipeline) is None
    assert pipeline.status()["error"] is None


def test_a_silent_phone_slows_neither_the_video_nor_the_server(make):
    gate = threading.Event()
    app = FakePhoneApp(gate=gate)
    detector = FakeDetector()
    factory = SourceFactory(phone_source(frames=5))
    pipeline = make(source=FakeSource(frames=1), detector=detector, phone_fetch=app, source_factory=factory).start()
    try:
        began = time.monotonic()
        pipeline.use_phone(PHONE)
        assert wait_until(lambda: app.calls)               # the thread is waiting on the phone ...
        pipeline.set_phone_picture("keep")
        pipeline.status()
        assert time.monotonic() - began < 0.5               # ... and nobody else waits for it
        assert wait_until(lambda: pipeline.state.snapshot().frame_size == (320, 240))   # the video goes on
        assert pipeline.target(track_id=2) is not None
    finally:
        gate.set()


def test_the_newest_switch_wins(make):
    gate = threading.Event()
    first, second = FakePhoneApp(gate=gate), FakePhoneApp(current="1280x720")
    network = Network({"192.168.1.10:8080": first, "192.168.1.11:8080": second})
    factory = SourceFactory(phone_source(), phone_source())
    pipeline = make(source=FakeSource(frames=1), phone_fetch=network, source_factory=factory).start()
    try:
        pipeline.use_phone("http://192.168.1.10:8080/video")
        assert wait_until(lambda: first.calls)              # the first phone is slow to answer
        pipeline.use_phone("http://192.168.1.11:8080/video")
    finally:
        gate.set()
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "640x360")
    time.sleep(0.1)
    assert first.sizes_asked == []                          # its answer came too late to matter
    assert second.sizes_asked == ["640x360"]
    assert picture_of(pipeline)["current"] == "640x360"


def test_a_switch_forgets_what_the_old_phone_said(make):
    old_phone = FakePhoneApp(current="640x360")
    new_phone = FakePhoneApp(gate=threading.Event())        # never answers in this test
    network = Network({"192.168.0.2:8080": old_phone, "192.168.1.7:8080": new_phone})
    pipeline = make(source=phone_source(), phone_fetch=network, source_factory=SourceFactory(phone_source())).start()
    assert wait_until(lambda: picture_of(pipeline) is not None)
    pipeline.use_phone(PHONE)
    assert picture_of(pipeline) is None
    new_phone.gate.set()


def test_set_phone_picture_each_form(make):
    app = FakePhoneApp()
    config = picture_config("keep")
    pipeline = make(source=phone_source(), phone_fetch=app, config=config).start()
    assert wait_until(lambda: picture_of(pipeline) is not None)
    pipeline.set_phone_picture("1280x720")
    assert picture_of(pipeline)["choice"] == "1280x720"    # applies at once ...
    assert wait_until(lambda: picture_of(pipeline)["current"] == "1280x720")   # ... and the phone follows
    pipeline.set_phone_picture(" AUTO ")
    assert wait_until(lambda: picture_of(pipeline)["current"] == "640x360")
    assert picture_of(pipeline)["choice"] == "auto"
    pipeline.set_phone_picture("keep")
    assert wait_until(lambda: picture_of(pipeline)["choice"] == "keep")
    time.sleep(0.05)
    assert app.sizes_asked == ["1280x720", "640x360"]
    assert config["stream"]["phone_picture"] == "keep"        # for this run only: the settings are not changed


@pytest.mark.parametrize("choice", ["1234x567", "huge", "", "640 x 360", "0x0", None, 640])
def test_set_phone_picture_refuses_what_the_phone_does_not_offer(make, choice):
    app = FakePhoneApp()
    pipeline = phone_at_start(make, app, choice="keep")
    assert wait_until(lambda: picture_of(pipeline) is not None)
    calls = len(app.calls)
    with pytest.raises(ValueError) as refused:
        pipeline.set_phone_picture(choice)
    assert "offer" in str(refused.value)
    assert picture_of(pipeline)["choice"] == "keep"
    time.sleep(0.05)
    assert len(app.calls) == calls                          # nothing was sent


def test_a_size_cannot_be_chosen_before_the_phone_said_what_it_offers(make):
    pipeline = make(source=FakeSource())
    with pytest.raises(ValueError):
        pipeline.set_phone_picture("640x360")
    pipeline.set_phone_picture("keep")                      # the two words always work
    assert pipeline._picture_choice == "keep"


@pytest.mark.parametrize("value", ["big", "", "640 by 360", 640, True, None, ["auto"]])
def test_a_wrong_phone_picture_setting_names_the_three_forms(value):
    with pytest.raises(ConfigError, match="auto, keep or a size like 1280x720"):
        Pipeline(picture_config(value), source=FakeSource(), detector=FakeDetector(), recognizer=FakeRecognizer())


def test_the_phone_picture_setting_is_required():
    stream = {key: value for key, value in CONFIG["stream"].items() if key != "phone_picture"}
    with pytest.raises(ConfigError, match="phone_picture"):
        Pipeline({**CONFIG, "stream": stream}, source=FakeSource(), detector=FakeDetector(),
                 recognizer=FakeRecognizer())


@pytest.mark.parametrize("text, choice", [("auto", "auto"), ("Keep", "keep"), (" 1280x720 ", "1280x720"),
                                          ("1280X720", "1280x720"), ("1280 x 720", None), ("max", None)])
def test_picture_choice(text, choice):
    assert pipeline_module.picture_choice(text) == choice


def test_picture_status_sorts_the_sizes_from_small_to_large():
    picture = pipeline_module.PhonePicture((1920, 1080), ((1280, 720), (640, 480), (800, 450), (640, 360)))
    assert pipeline_module.picture_status(picture, "auto") == {
        "current": "1920x1080", "offered": ["640x360", "640x480", "800x450", "1280x720"], "choice": "auto"}
    assert pipeline_module.picture_status(None, "auto") is None


# ---- the box the phone's pictures are fitted into -----------------------------------------

CONFIG_BOX = (640, 480)


def offering(*sizes):
    """What a phone sending 1920 x 1080 said, offering these sizes."""
    return pipeline_module.PhonePicture((1920, 1080), tuple(parse_size(size) for size in sizes))


def test_shown_box_for_each_kind_of_choice():
    picture = offering(*REAL_OFFERED)
    shown_box = pipeline_module.shown_box
    assert shown_box("auto", picture, CONFIG_BOX) == CONFIG_BOX
    assert shown_box("keep", picture, CONFIG_BOX) == CONFIG_BOX
    assert shown_box("1280x720", picture, CONFIG_BOX) == (1280, 720)
    assert shown_box("1920x1080", picture, CONFIG_BOX) == (1920, 1080)
    assert shown_box("320x240", picture, CONFIG_BOX) == (320, 240)        # smaller than the config box
    assert shown_box("3840x2160", picture, CONFIG_BOX) == (1920, 1080)    # at most 1920 x 1080 ...
    assert shown_box("1600x1200", picture, CONFIG_BOX) == (1440, 1080)    # ... with the shape kept
    assert pipeline_module.MAX_SHOWN_SIZE == (1920, 1080)


def test_shown_box_is_the_config_box_for_a_size_the_phone_does_not_offer_or_has_not_said():
    assert pipeline_module.shown_box("1234x567", offering(*REAL_OFFERED), CONFIG_BOX) == CONFIG_BOX
    assert pipeline_module.shown_box("1280x720", offering("640x360"), CONFIG_BOX) == CONFIG_BOX
    assert pipeline_module.shown_box("1280x720", None, CONFIG_BOX) == CONFIG_BOX


def test_a_chosen_size_sets_the_box_at_once_and_auto_puts_it_back(make):
    app, source = FakePhoneApp(), phone_source()
    pipeline = make(source=source, phone_fetch=app, config=picture_config("keep")).start()
    assert wait_until(lambda: picture_of(pipeline) is not None)
    assert source.boxes[-1] == CONFIG_BOX
    pipeline.set_phone_picture("1920x1080")
    assert source.boxes[-1] == (1920, 1080)                 # at once, without waiting for the phone
    pipeline.set_phone_picture("3840x2160")
    assert source.boxes[-1] == (1920, 1080)                 # the cap
    pipeline.set_phone_picture("auto")
    assert source.boxes[-1] == CONFIG_BOX                   # Automatic puts the box back at once
    assert wait_until(lambda: picture_of(pipeline)["current"] == "640x360")
    pipeline.set_phone_picture("1280x720")
    assert source.boxes[-1] == (1280, 720)
    pipeline.set_phone_picture("keep")
    assert source.boxes[-1] == CONFIG_BOX
    time.sleep(0.05)                                        # what the phone threads still do ...
    assert source.boxes[-1] == CONFIG_BOX                   # ... leaves the newest choice's box


@pytest.mark.parametrize("choice, box", [("1280x720", (1280, 720)), ("3840x2160", (1920, 1080))])
def test_a_configured_size_sets_the_box_once_the_phone_says_it_offers_it(make, choice, box):
    app, source = FakePhoneApp(gate=threading.Event()), phone_source()
    pipeline = make(source=source, phone_fetch=app, config=picture_config(choice)).start()
    assert wait_until(lambda: app.calls)
    assert source.boxes == []                               # the config box until the phone answers
    app.gate.set()
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == choice)
    assert set(source.boxes) == {box}


def test_a_configured_size_the_phone_does_not_offer_keeps_the_config_box(make):
    source = phone_source()
    pipeline = make(source=source, phone_fetch=FakePhoneApp(), config=picture_config("1234x567")).start()
    assert wait_until(lambda: picture_of(pipeline) is not None)
    time.sleep(0.05)
    assert set(source.boxes) == {CONFIG_BOX}


def test_the_box_stays_when_the_phone_drops_out_and_returns(make):
    app, source = FakePhoneApp(), phone_source(frames=3)
    pipeline = make(source=source, phone_fetch=app, config=picture_config("1280x720")).start()
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "1280x720")
    source.connected = False                                # the app restarts ...
    assert wait_until(lambda: pipeline._lost_line is not None)
    app.current = "1920x1080"                               # ... with its own size
    source.connected = True
    source.frames += 3
    assert wait_until(lambda: app.sizes_asked == ["1280x720", "1280x720"])
    assert wait_until(lambda: picture_of(pipeline)["current"] == "1280x720")
    assert set(source.boxes) == {(1280, 720)}               # never back to the config box meanwhile


def test_the_box_follows_the_choice_to_a_new_phone(make):
    first, second = FakePhoneApp(), FakePhoneApp()
    network = Network({"192.168.0.2:8080": first, "192.168.1.7:8080": second})
    old, new = phone_source(), phone_source()
    factory = SourceFactory(new)
    pipeline = make(source=old, phone_fetch=network, source_factory=factory, config=picture_config("keep")).start()
    assert wait_until(lambda: picture_of(pipeline) is not None)
    pipeline.set_phone_picture("1280x720")
    pipeline.use_phone(PHONE)
    assert (factory.settings[-1]["width"], factory.settings[-1]["height"]) == CONFIG_BOX   # it starts with the config box
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "1280x720")
    assert set(new.boxes) == {(1280, 720)}
    assert second.sizes_asked == ["1280x720"]


def test_a_new_phone_that_does_not_offer_the_chosen_size_gets_the_config_box(make):
    first, second = FakePhoneApp(), FakePhoneApp(offered=["1920x1080", "640x360"])
    network = Network({"192.168.0.2:8080": first, "192.168.1.7:8080": second})
    new = phone_source()
    pipeline = make(source=phone_source(), phone_fetch=network, source_factory=SourceFactory(new),
                    config=picture_config("keep")).start()
    assert wait_until(lambda: picture_of(pipeline) is not None)
    pipeline.set_phone_picture("1280x720")
    pipeline.use_phone(PHONE)
    assert wait_until(lambda: picture_of(pipeline) is not None and len(second.calls) == 1)
    time.sleep(0.05)
    assert set(new.boxes) == {CONFIG_BOX}
    assert picture_of(pipeline)["choice"] == "1280x720" and second.sizes_asked == []


@pytest.mark.parametrize("kind", ["webcam", "file"])
def test_a_clip_or_the_webcam_is_never_given_a_box_but_the_phone_after_it_is(make, kind):
    clip, phone = FakeSource(frames=3), phone_source()
    clip.source = kind
    pipeline = make(source=clip, phone_fetch=FakePhoneApp(), source_factory=SourceFactory(phone),
                    config=picture_config("1280x720")).start()
    assert wait_until(lambda: pipeline.latest_jpeg() is not None)
    pipeline.use_phone(PHONE)
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "1280x720")
    assert clip.boxes == []
    assert set(phone.boxes) == {(1280, 720)}


class FakeCapture:
    """A phone stream as FrameSource reads it: 1920 x 1080 pictures, as many as asked for."""

    def __init__(self, size=(1920, 1080)):
        width, height = size
        self.picture = np.full((height, width, 3), 60, np.uint8)

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.005)
        return True, self.picture.copy()

    def release(self):
        pass


def test_the_chosen_size_is_the_size_of_the_video_and_of_what_the_models_see(make):
    """With a real FrameSource (fed by a fake stream): the frames really change size."""
    settings = {**CONFIG["stream"], "source": "phone"}
    source = pipeline_module.FrameSource(settings, opener=FakeCapture)
    sizes = []

    class SizeDetector(FakeDetector):
        def detect(self, frame):
            sizes.append(frame.shape[:2])
            return super().detect(frame)

    pipeline = make(source=source, detector=SizeDetector(), phone_fetch=FakePhoneApp(),
                    config=picture_config("keep")).start()
    assert wait_until(lambda: picture_of(pipeline) is not None)
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (640, 360))      # the config box
    pipeline.set_phone_picture("1920x1080")
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (1920, 1080))
    assert wait_until(lambda: decoded(pipeline.latest_jpeg()).shape == (1080, 1920, 3))   # the page's video
    assert sizes[-1] == (1080, 1920)                                                   # the detector's
    assert pipeline.target(track_id=2) is not None                                     # clicks still work
    pipeline.set_phone_picture("auto")
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (640, 360))


class ResizingSource(FakeSource):
    """Sends `small` frames at 640 x 360, then the rest at 1280 x 720."""

    def __init__(self, small, frames):
        super().__init__(frames=frames)
        self.small = small
        self.large = np.full((720, 1280, 3), 60, np.uint8)

    def read(self):
        frame, sequence = super().read()
        if frame is not None and sequence > self.small:
            return self.large.copy(), sequence
        return frame, sequence


class RecognizesOnce(FakeRecognizer):
    """Finds Sara on its first check only."""

    def recognize(self, frame, detections=()):
        found = super().recognize(frame, detections)
        return found if self.calls == 1 else []


def test_a_new_frame_size_forgets_the_remembered_faces_and_checks_again_at_once(make):
    # every_n_frames is 3: without the new size, frame 3 would show Sara from memory.
    detector, recognizer = FakeDetector(), RecognizesOnce()
    pipeline = make(source=ResizingSource(small=2, frames=3), detector=detector, recognizer=recognizer).start()
    assert wait_until(lambda: detector.calls == 3)
    time.sleep(0.05)
    assert recognizer.calls == 2                            # frame 1, and frame 3 at once (the new size)
    assert pipeline.state.snapshot().faces == ()
    assert pipeline.state.snapshot().frame_size == (1280, 720)



# ---- talking with the phone: the harder cases --------------------------------------------


class NotServingYet(FakePhoneApp):
    """A phone whose camera app does not answer until `serving` is set."""

    serving = False

    def __call__(self, url, timeout):
        if not self.serving:
            with self._lock:
                self.calls.append(url)
            raise ConnectionRefusedError("the camera app is not serving yet")
        return super().__call__(url, timeout)


@pytest.mark.parametrize("when", ["start-up", "connect"])
def test_a_phone_that_was_not_serving_is_asked_on_its_first_picture(make, when):
    app, phone = NotServingYet(), phone_source(frames=0, connected=False)
    if when == "start-up":
        pipeline = make(source=phone, phone_fetch=app).start()
    else:
        pipeline = make(source=FakeSource(frames=3), phone_fetch=app, source_factory=SourceFactory(phone)).start()
        pipeline.use_phone("http://192.168.0.2:8080/video")
    assert wait_until(lambda: app.calls)                    # asked, and no answer
    time.sleep(0.05)
    assert picture_of(pipeline) is None
    app.serving = True                                      # the camera app starts serving ...
    phone.connected, phone.frames = True, 3                 # ... and its pictures arrive
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "640x360")
    time.sleep(0.05)
    assert app.sizes_asked == ["640x360"]                   # asked once, not once per frame


class AnswersOkLate(FakePhoneApp):
    """Takes a new size at once, but holds its "Ok" until the test lets it go."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.asked, self.answer = threading.Event(), threading.Event()

    def __call__(self, url, timeout):
        reply = super().__call__(url, timeout)
        if "set=" in url:
            self.asked.set()
            assert self.answer.wait(5)
        return reply


def test_the_stream_is_reopened_even_when_a_newer_round_replaced_the_one_that_asked(make, monkeypatch):
    """1280x720, then keep, then 1280x720 within a second: the phone sends 1280 x 720,
    and the stream must be opened again although the round that asked was replaced."""
    monkeypatch.setattr(pipeline_module, "STREAM_SETTLE_SECONDS", 0.01)
    app, source = AnswersOkLate(current="640x360"), phone_source()
    pipeline = make(source=source, phone_fetch=app, config=picture_config("keep")).start()
    assert wait_until(lambda: picture_of(pipeline) is not None)
    pipeline.set_phone_picture("1280x720")
    assert app.asked.wait(3)                                # the phone has the new size, its Ok is on the way
    pipeline.set_phone_picture("keep")
    pipeline.set_phone_picture("1280x720")
    app.answer.set()
    assert wait_until(lambda: getattr(source, "reopens", 0) == 1)
    time.sleep(0.1)
    assert getattr(source, "reopens", 0) == 1 and app.sizes_asked == ["1280x720"]


def test_offered_to_keep_keeps_the_longer_list_of_the_same_phone():
    earlier = offering("640x360", "1280x720", "1920x1080")
    sizes = [(640, 360), (1280, 720), (1920, 1080)]
    assert pipeline_module.offered_to_keep(None, [(1920, 1080)]) == ((1920, 1080),)
    assert pipeline_module.offered_to_keep(earlier, [(1920, 1080)]) == earlier.offered      # shorter
    assert pipeline_module.offered_to_keep(earlier, [(1920, 1080), (640, 480), (800, 600)]) == earlier.offered
    assert pipeline_module.offered_to_keep(earlier, sizes + [(800, 450)]) == (*sizes, (800, 450))   # more: taken


class ListsOneSizeOnce(FakePhoneApp):
    """Like the real phone while its camera restarts: the next status lists only the size it sends."""

    short_once = False

    def __call__(self, url, timeout):
        if self.short_once and url.endswith("show_avail=1"):
            self.short_once = False
            full, self.offered = self.offered, [self.current]
            try:
                return super().__call__(url, timeout)
            finally:
                self.offered = full
        return super().__call__(url, timeout)


def test_the_came_back_check_keeps_the_good_list_of_offered_sizes(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "STREAM_SETTLE_SECONDS", 0.01)
    app, source = ListsOneSizeOnce(), phone_source(frames=3)
    pipeline = make(source=source, phone_fetch=app, config=picture_config("1280x720")).start()
    assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "1280x720")
    source.connected = False                                # the camera app restarts ...
    assert wait_until(lambda: pipeline._lost_line is not None)
    app.current, app.short_once = "1920x1080", True         # ... sends its own size, lists only that
    source.connected = True
    source.frames += 3
    assert wait_until(lambda: app.sizes_asked == ["1280x720", "1280x720"])   # still "offered": asked again
    assert wait_until(lambda: picture_of(pipeline)["current"] == "1280x720")
    assert picture_of(pipeline)["offered"] == BY_AREA
    assert set(source.boxes) == {(1280, 720)}               # the box never fell back


class BouncingPhoneSource(FakeSource):
    """A phone that goes back to 1920 x 1080 whenever its stream is opened again,
    and whose stream drops for a moment each time."""

    def __init__(self, app):
        super().__init__(frames=1)
        self.source, self.app, self.reopens = "phone", app, 0

    def reopen(self):
        self.reopens += 1
        self.app.current = "1920x1080"
        self.connected = False
        threading.Timer(0.03, self.come_back).start()

    def come_back(self):
        with self._lock:
            self.frames += 1
        self.connected = True


def test_a_phone_that_keeps_changing_back_is_asked_at_most_three_times(make, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="car_vision.pipeline")
    monkeypatch.setattr(pipeline_module, "STREAM_SETTLE_SECONDS", 0.01)
    app = FakePhoneApp()
    source = BouncingPhoneSource(app)
    pipeline = make(source=source, phone_fetch=app, config=picture_config("auto")).start()
    assert wait_until(lambda: len(app.sizes_asked) == 3)
    assert wait_until(lambda: any("3 times" in record.getMessage() for record in caplog.records))
    time.sleep(0.3)                                         # several more drops and returns would fit here
    assert len(app.sizes_asked) == 3
    assert len([r for r in caplog.records if "3 times" in r.getMessage()]) == 1
    pipeline.set_phone_picture("auto")                      # a new choice may ask again
    assert wait_until(lambda: len(app.sizes_asked) >= 4)


def test_a_dead_phone_does_not_hold_up_the_next_one(make):
    dead, alive = FakePhoneApp(gate=threading.Event()), FakePhoneApp()
    network = Network({"192.168.1.10:8080": dead, "192.168.1.11:8080": alive})
    factory = SourceFactory(phone_source(), phone_source())
    pipeline = make(source=FakeSource(frames=1), phone_fetch=network, source_factory=factory).start()
    try:
        pipeline.use_phone("http://192.168.1.10:8080/video")
        assert wait_until(lambda: dead.calls)               # stuck waiting on the dead phone ...
        began = time.monotonic()
        pipeline.use_phone("http://192.168.1.11:8080/video")
        assert wait_until(lambda: (picture_of(pipeline) or {}).get("current") == "640x360", timeout=1.0)
        assert time.monotonic() - began < 0.5               # ... and the next phone did not wait for it
    finally:
        dead.gate.set()
    time.sleep(0.05)
    assert dead.sizes_asked == []


def test_a_round_waiting_to_read_back_gives_way_to_a_newer_one(make):
    app = FakePhoneApp(lies=True)                           # says Ok, keeps 1920 x 1080: a read-back waits 1 s
    pipeline = phone_at_start(make, app, choice="keep")
    assert wait_until(lambda: picture_of(pipeline) is not None and pipeline.latest_jpeg() is not None)
    time.sleep(0.05)                                        # the start-up talk is over
    base = len(app.calls)
    pipeline.set_phone_picture("1280x720")
    assert wait_until(lambda: len(app.calls) == base + 3)   # status, ask, first read-back: now it waits
    began = time.monotonic()
    pipeline.set_phone_picture("keep")
    assert wait_until(lambda: len(app.calls) == base + 4, timeout=1.0)
    assert time.monotonic() - began < 0.3                   # not after the older round's second
    time.sleep(1.2)
    assert len(app.calls) == base + 4                       # the older round never read back again


def test_the_track_epoch_goes_up_when_track_numbers_start_again(make):
    source = FakeSource(frames=2)
    pipeline = started(make, source=source, source_factory=SourceFactory(phone_source(frames=3)))
    assert pipeline.target(track_id=2).epoch == 0
    source.frames += 2                                      # more frames of the same size: same epoch
    assert wait_until(lambda: source.sequence == 4)
    time.sleep(0.05)
    assert pipeline.target(track_id=2).epoch == 0
    source.picture = np.full((720, 1280, 3), 60, np.uint8)  # a new frame size: the detector starts again
    source.frames += 1
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (1280, 720))
    assert pipeline.target(track_id=2).epoch == 1
    assert pipeline.target(x=0.5, y=0.5).epoch == 1         # every kind of target carries it
    pipeline.use_phone(PHONE)                               # a new camera
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (320, 240))
    assert wait_until(lambda: pipeline.target(track_id=2) is not None)
    assert pipeline.target(track_id=2).epoch == 2


def test_a_target_has_epoch_0_unless_told_otherwise():
    assert pipeline_module.Target("area", "", 0.0, None, b"").epoch == 0


# ---- start-up ---------------------------------------------------------------------------


def test_recogniser_that_fails_to_build_gives_recognition_false_and_a_note(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError("CUDA provider missing\nmore detail")

    monkeypatch.setattr(pipeline_module, "Recognizer", broken)
    detector = FakeDetector()
    pipeline = Pipeline(CONFIG, source=FakeSource(frames=3), detector=detector)
    try:
        status = pipeline.status()
        assert status["recognition"] is False
        assert status["notes"] == ["Face recognition is off: CUDA provider missing"]
        pipeline.start()                                # and it still runs without faces
        assert wait_until(lambda: detector.calls == 3)
        assert pipeline.state.snapshot().faces == ()
    finally:
        pipeline.stop()


def test_nobody_enrolled_runs_with_everyone_unknown_and_says_so(monkeypatch, capsys):
    built = {}

    class Recorder(FakeRecognizer):
        def __init__(self, settings, people=None, matcher=None):
            super().__init__(faces=[])
            built.update(people=people, matcher=matcher)

    monkeypatch.setattr(pipeline_module, "Recognizer", Recorder)
    pipeline = Pipeline(CONFIG, source=FakeSource(), detector=FakeDetector())
    status = pipeline.status()
    assert status["recognition"] is True
    assert status["notes"] == [NOBODY_ENROLLED]
    assert built["people"] == {} and built["matcher"].person_ids == set()
    assert "Nobody is enrolled yet" in capsys.readouterr().out   # the recognizer's own note


def test_detector_that_cannot_load_raises(monkeypatch):
    def broken(settings):
        raise RuntimeError("weights damaged")

    monkeypatch.setattr(pipeline_module, "Detector", broken)
    with pytest.raises(RuntimeError, match="weights damaged"):
        Pipeline(CONFIG, source=FakeSource(), recognizer=FakeRecognizer())


def test_missing_clip_fails_before_any_model_loads(monkeypatch):
    def must_not_load(settings):
        raise AssertionError("the detector was loaded")

    monkeypatch.setattr(pipeline_module, "Detector", must_not_load)
    with pytest.raises(ConfigError, match="does not exist"):
        Pipeline(CONFIG)


@pytest.mark.parametrize("section, key, value", [("recognizer", "every_n_frames", 0),
                                                 ("app", "jpeg_quality", 101)])
def test_bad_settings_are_rejected(section, key, value):
    config = {**CONFIG, section: {**CONFIG[section], key: value}}
    with pytest.raises(ConfigError, match=key):
        Pipeline(config, source=FakeSource(), detector=FakeDetector(), recognizer=FakeRecognizer())


# ---- stopping ---------------------------------------------------------------------------


@pytest.mark.parametrize("delay", [0.0, 0.3])
def test_stop_returns_within_two_seconds(delay):
    source, detector = FakeSource(frames=10_000), FakeDetector(delay=delay)
    pipeline = Pipeline(CONFIG, source=source, detector=detector, recognizer=FakeRecognizer()).start()
    assert source.started
    assert wait_until(lambda: detector.calls >= 2)
    began = time.monotonic()
    pipeline.stop()
    assert time.monotonic() - began < 2.0
    assert source.stopped
    calls = detector.calls
    time.sleep(0.4)
    assert detector.calls == calls                      # the loop really ended


def test_start_twice_runs_one_loop(make):
    pipeline = make()
    assert pipeline.start() is pipeline
    thread = pipeline._thread
    pipeline.start()
    assert pipeline._thread is thread


# ---- which libraries the pipeline uses ----------------------------------------------------


def test_pipeline_imports_no_third_party_library_but_cv2_and_numpy():
    allowed = set(sys.stdlib_module_names) | {"car_vision", "cv2", "numpy"}
    found = set()
    for node in ast.walk(ast.parse(PIPELINE_FILE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    assert found - allowed == set()


# ---- people: photos from the live camera, and enrolling again ------------------------------
#
# These tests use the real Recognizer with a fake face model, so the photo session sees the
# same face boxes the loop gets, and enrolling again really reads the saved photos.

FACE_BOX = (200, 60, 360, 260)                  # 160 pixels wide
TINY_FACE = (520, 60, 550, 90)                  # too small to name (min_face_pixels is 40), still a face
LINA = np.eye(8, dtype=np.float32)[2]
SARA_VECTOR = np.eye(8, dtype=np.float32)[0]
LINA_ID = "lina_haddad"                       # the id PeopleBook makes from "Lina Haddad"


def drawn_face(size, seed=0):
    """A cartoon face with a little camera noise: sharp edges, like a face in focus."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 90, np.uint8)
    middle = size // 2
    cv2.ellipse(image, (middle, middle), (int(size * 0.35), int(size * 0.45)), 0, 0, 360, (150, 170, 200), -1)
    for side in (-1, 1):
        eye = (middle + side * size // 6, middle - size // 8)
        cv2.circle(image, eye, size // 16, (40, 40, 40), -1)
        cv2.line(image, (eye[0] - size // 10, middle - size // 5), (eye[0] + size // 10, middle - size // 5),
                 (50, 50, 60), 2)
    cv2.ellipse(image, (middle, middle + size // 5), (size // 8, size // 20), 0, 0, 180, (60, 60, 120), 2)
    return np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)


def face_frame(blur=0.0, size=(640, 360)):
    """A camera frame with a drawn face inside FACE_BOX."""
    width, height = size
    frame = np.full((height, width, 3), 60, np.uint8)
    x1, y1, x2, y2 = FACE_BOX
    frame[y1:y2, x1:x2] = cv2.resize(drawn_face(x2 - x1), (x2 - x1, y2 - y1))
    return cv2.GaussianBlur(frame, (0, 0), blur) if blur else frame


class FakeFace:
    """Shaped like an InsightFace face."""

    def __init__(self, box, vector):
        self.bbox = np.asarray(box, dtype=np.float32)
        self.normed_embedding = np.asarray(vector, dtype=np.float32)


class ScriptedFaceModel:
    """Sees the faces at `boxes` in every picture. Whose face it is follows the picture's
    brightness: dark pictures show Sara, bright ones Lina (this survives saving as JPEG)."""

    def __init__(self, boxes=(FACE_BOX,), delay=0.0):
        self.boxes = list(boxes)
        self.delay = delay
        self.calls = 0

    def get(self, image):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        vector = SARA_VECTOR if image.mean() < 50 else LINA
        return [FakeFace(box, vector) for box in self.boxes]


def people_config(tmp_path, every_n=1):
    recognizer = {**CONFIG["recognizer"], "every_n_frames": every_n,
                  "photos_dir": str(tmp_path / "people"), "embeddings_file": str(tmp_path / "embeddings.npz")}
    return {**CONFIG, "recognizer": recognizer}


def real_recognizer(tmp_path, model):
    """The real Recognizer with a fake face model and nobody enrolled yet."""
    settings = people_config(tmp_path)["recognizer"]
    return Recognizer(settings, face_model=model, people={}, matcher=FaceMatcher([], np.zeros((0, 0)), 0.42))


@pytest.fixture
def book(tmp_path):
    """A photo folder with Lina in it (an about.md, no photos yet)."""
    book = PeopleBook(tmp_path / "people", tmp_path / "people.removed")
    book.save("Lina Haddad", "Driver", ["Drives the car in the demo"])
    return book


@pytest.fixture
def quick(monkeypatch):
    """No wait between photos and a short timeout; a test can change both."""
    monkeypatch.setattr(pipeline_module, "PHOTO_GAP_SECONDS", 0.0)
    monkeypatch.setattr(pipeline_module, "PHOTO_TIMEOUT_SECONDS", 30.0)
    return monkeypatch


def photo_pipeline(make, tmp_path, boxes=(FACE_BOX,), source=None, every_n=1, delay=0.0):
    model = ScriptedFaceModel(boxes, delay)
    pipeline = make(source=source if source is not None else FakeSource(frames=10_000, picture=face_frame()),
                    detector=FakeDetector(detections=[]), recognizer=real_recognizer(tmp_path, model),
                    config=people_config(tmp_path, every_n))
    return pipeline, model


def photos_of(book, person_id=LINA_ID):
    return sorted(p.name for p in (book.photos_dir / person_id).glob("photo_*.jpg"))


def progress_is(pipeline, **expected):
    def check():
        progress = pipeline.photo_progress()
        return progress is not None and all(progress[key] == value for key, value in expected.items())
    return check


# ---- the sharpness rule ------------------------------------------------------------


@pytest.mark.parametrize("size", [80, 160, 320])
def test_the_sharpness_threshold_lies_between_a_sharp_and_a_blurred_face(size):
    sharp = drawn_face(size)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2 * size / 112)     # 2 pixels at the size the model reads
    flat = np.full_like(sharp, 128)
    assert sharpness(sharp) > 10 * SHARPNESS_THRESHOLD
    assert sharpness(blurred) < 0.7 * SHARPNESS_THRESHOLD
    assert sharpness(flat) == 0.0
    assert sharpness(sharp[:0]) == 0.0                             # an empty cut-out


@pytest.mark.parametrize("boxes, blur, hint", [
    ([], 0.0, HINT_NO_FACE),
    ([FACE_BOX, TINY_FACE], 0.0, HINT_ONE_PERSON),
    ([(200, 60, 279, 140)], 0.0, HINT_CLOSER),                    # 79 pixels wide
    ([(200, 60, 280, 140)], 0.0, ""),                              # 80 is enough
    ([FACE_BOX], 3.0, HINT_STILL),
    ([FACE_BOX], 0.0, ""),
])
def test_photo_hint_for_each_rule(boxes, blur, hint):
    assert photo_hint(face_frame(blur), boxes) == hint


# ---- photo sessions ------------------------------------------------------------------


@pytest.mark.parametrize("boxes, blur, hint", [
    ([], 0.0, HINT_NO_FACE),
    ([FACE_BOX, TINY_FACE], 0.0, HINT_ONE_PERSON),
    ([(200, 60, 260, 120)], 0.0, HINT_CLOSER),
    ([FACE_BOX], 3.0, HINT_STILL),
])
def test_no_photo_is_taken_until_every_rule_holds(make, tmp_path, book, quick, boxes, blur, hint):
    source = FakeSource(frames=10_000, picture=face_frame(blur))
    pipeline, _model = photo_pipeline(make, tmp_path, boxes, source)
    pipeline.start()
    pipeline.start_photos(LINA_ID, 3)
    assert wait_until(progress_is(pipeline, hint=hint))
    time.sleep(0.1)
    assert pipeline.photo_progress() == {"person_id": LINA_ID, "taken": 0, "wanted": 3, "state": "taking",
                                         "hint": hint, "notes": []}
    assert photos_of(book) == []


def test_photos_are_taken_until_the_count_then_everyone_is_enrolled(make, tmp_path, book, quick, caplog):
    assert save_photo(book.photos_dir / LINA_ID, face_frame()).name == "photo_1.jpg"    # one from before
    pipeline, _model = photo_pipeline(make, tmp_path)
    assert pipeline.photo_progress() is None                       # no session yet
    pipeline.start()
    with caplog.at_level(logging.DEBUG):
        pipeline.start_photos(LINA_ID, 3)
        assert wait_until(progress_is(pipeline, state="done"))
    progress = pipeline.photo_progress()
    assert (progress["taken"], progress["wanted"], progress["hint"]) == (3, 3, "")
    assert f"Saved 4 face vectors for 1 people to {tmp_path / 'embeddings.npz'}" in progress["notes"]
    assert photos_of(book) == ["photo_1.jpg", "photo_2.jpg", "photo_3.jpg", "photo_4.jpg"]
    photo = cv2.imread(str(book.photos_dir / LINA_ID / "photo_4.jpg"))
    assert photo.shape == (360, 640, 3)                            # the clean frame at its own size
    assert np.abs(photo.astype(int) - face_frame().astype(int)).mean() < 3      # nothing drawn on it
    assert pipeline.enrolled_ids() == {LINA_ID}
    assert load_embeddings(tmp_path / "embeddings.npz")[0] == [LINA_ID] * 4
    # Lina is recognised in the video now, without a restart.
    assert wait_until(lambda: [f.name for f in pipeline.state.snapshot().faces] == ["Lina Haddad"])
    # Taking photos and enrolling log only the person id and counts. (The greeting the
    # reactions log once Lina is recognised is not part of the photo session.)
    logged = " ".join(record.getMessage() for record in caplog.records
                      if record.name in ("car_vision.pipeline", "car_vision.recognizer", "car_vision.enroll"))
    assert LINA_ID in logged
    assert "Lina Haddad" not in logged and "Driver" not in logged


def test_a_photo_is_not_taken_too_soon_after_the_last(make, tmp_path, book, quick):
    quick.setattr(pipeline_module, "PHOTO_GAP_SECONDS", 10.0)
    pipeline, _model = photo_pipeline(make, tmp_path)
    pipeline.start()
    pipeline.start_photos(LINA_ID, 3)
    assert wait_until(progress_is(pipeline, taken=1))
    time.sleep(0.2)
    assert pipeline.photo_progress()["taken"] == 1
    # Nothing is wrong, so the hint asks for the pose of the next photo.
    assert pipeline.photo_progress()["hint"] == pipeline_module.PHOTO_POSES[1]
    assert photos_of(book) == ["photo_1.jpg"]


def test_time_up_with_three_photos_enrols_and_says_fewer_were_taken(make, tmp_path, book, quick):
    quick.setattr(pipeline_module, "PHOTO_TIMEOUT_SECONDS", 0.5)
    pipeline, _model = photo_pipeline(make, tmp_path, source=FakeSource(frames=3, picture=face_frame()))
    pipeline.start_photos(LINA_ID, 5)                               # before start(): every frame counts
    pipeline.start()
    assert wait_until(progress_is(pipeline, state="done"))
    progress = pipeline.photo_progress()
    assert progress["taken"] == 3
    assert progress["notes"][0].startswith("Only 3 of 5 photos were taken within")
    assert pipeline.enrolled_ids() == {LINA_ID}


def test_time_up_with_two_photos_fails_and_keeps_them(make, tmp_path, book, quick):
    quick.setattr(pipeline_module, "PHOTO_TIMEOUT_SECONDS", 0.5)
    pipeline, _model = photo_pipeline(make, tmp_path, source=FakeSource(frames=2, picture=face_frame()))
    pipeline.start_photos(LINA_ID, 5)
    pipeline.start()
    assert wait_until(progress_is(pipeline, state="failed"))
    progress = pipeline.photo_progress()
    assert progress["taken"] == 2
    assert "Only 2 clear photo(s)" in progress["hint"] and "at least 3" in progress["hint"]
    assert photos_of(book) == ["photo_1.jpg", "photo_2.jpg"]
    assert pipeline.enrolled_ids() == set()
    assert not (tmp_path / "embeddings.npz").exists()


def test_the_time_runs_out_even_while_no_frames_come(make, tmp_path, book, quick):
    quick.setattr(pipeline_module, "PHOTO_TIMEOUT_SECONDS", 0.3)
    source = FakeSource(frames=0)                                  # the phone sends nothing
    source.connected = False
    pipeline, _model = photo_pipeline(make, tmp_path, source=source)
    pipeline.start()
    pipeline.start_photos(LINA_ID, 3)
    assert wait_until(progress_is(pipeline, state="failed"))
    assert "Only 0 clear photo(s)" in pipeline.photo_progress()["hint"]


def test_a_session_survives_a_phone_that_drops_out_and_returns(make, tmp_path, book, quick):
    source = FakeSource(frames=1, picture=face_frame())
    pipeline, _model = photo_pipeline(make, tmp_path, source=source)
    pipeline.start_photos(LINA_ID, 3)
    pipeline.start()
    assert wait_until(progress_is(pipeline, taken=1))
    source.connected = False                                       # gone for a while
    time.sleep(0.1)
    assert pipeline.photo_progress()["state"] == "taking"
    source.connected, source.frames = True, 3                      # and back with new frames
    assert wait_until(progress_is(pipeline, state="done"))
    assert pipeline.photo_progress()["taken"] == 3


class GrowingSource(FakeSource):
    """One frame at 640 x 360, then frames at 1280 x 720 (another picture size was chosen)."""

    def __init__(self):
        super().__init__(frames=10_000, picture=face_frame())
        self.large = face_frame(size=(1280, 720))

    def read(self):
        frame, sequence = super().read()
        return (self.large.copy() if sequence > 1 else frame), sequence


def test_a_session_survives_a_change_of_picture_size(make, tmp_path, book, quick):
    pipeline, _model = photo_pipeline(make, tmp_path, source=GrowingSource())
    pipeline.start_photos(LINA_ID, 3)
    pipeline.start()
    assert wait_until(progress_is(pipeline, state="done"))
    sizes = [cv2.imread(str(book.photos_dir / LINA_ID / name)).shape[:2] for name in photos_of(book)]
    assert sizes == [(360, 640), (720, 1280), (720, 1280)]


def test_cancel_ends_the_session_and_keeps_the_photos(make, tmp_path, book, quick):
    quick.setattr(pipeline_module, "PHOTO_GAP_SECONDS", 10.0)
    pipeline, _model = photo_pipeline(make, tmp_path)
    pipeline.cancel_photos()                                       # nothing to cancel: no error
    assert pipeline.photo_progress() is None
    pipeline.start()
    pipeline.start_photos(LINA_ID, 3)
    assert wait_until(progress_is(pipeline, taken=1))
    pipeline.cancel_photos()
    assert pipeline.photo_progress() == {"person_id": LINA_ID, "taken": 1, "wanted": 3, "state": "failed",
                                         "hint": HINT_CANCELLED, "notes": []}
    time.sleep(0.1)
    assert pipeline.photo_progress()["state"] == "failed"
    assert photos_of(book) == ["photo_1.jpg"]


def test_a_second_session_is_refused_while_one_runs(make, tmp_path, book, quick):
    quick.setattr(pipeline_module, "PHOTO_GAP_SECONDS", 10.0)
    book.save("Omar Ali")
    pipeline, _model = photo_pipeline(make, tmp_path)
    pipeline.start_photos(LINA_ID, 3)
    with pytest.raises(ValueError, match="already"):
        pipeline.start_photos("omar_ali", 3)
    pipeline.cancel_photos()
    pipeline.start_photos("omar_ali", 4)                           # once it ended, a new one may begin
    assert pipeline.photo_progress()["person_id"] == "omar_ali"
    assert pipeline.photo_progress()["taken"] == 0


@pytest.mark.parametrize("person_id", ["nobody", "../people", "Lina", "", None])
def test_start_photos_for_an_unknown_person_is_a_key_error(make, tmp_path, book, person_id):
    pipeline, _model = photo_pipeline(make, tmp_path)
    with pytest.raises(KeyError):
        pipeline.start_photos(person_id)
    assert pipeline.photo_progress() is None


@pytest.mark.parametrize("count", [2, 11, 0, -5, "5", 4.0, True, None])
def test_start_photos_takes_3_to_10(make, tmp_path, book, count):
    pipeline, _model = photo_pipeline(make, tmp_path)
    with pytest.raises(ValueError, match="3 to 10"):
        pipeline.start_photos(LINA_ID, count)


def test_use_phone_cancels_a_session(make, tmp_path, book, quick):
    quick.setattr(pipeline_module, "PHOTO_GAP_SECONDS", 10.0)
    model = ScriptedFaceModel()
    pipeline = make(source=FakeSource(frames=10_000, picture=face_frame()), detector=FakeDetector(detections=[]),
                    recognizer=real_recognizer(tmp_path, model), config=people_config(tmp_path),
                    source_factory=SourceFactory(phone_source(frames=10_000)))
    pipeline.start()
    pipeline.start_photos(LINA_ID, 3)
    assert wait_until(progress_is(pipeline, taken=1))
    pipeline.use_phone(PHONE)
    progress = pipeline.photo_progress()
    assert (progress["state"], progress["hint"]) == ("failed", HINT_SWITCHED)
    assert progress["hint"].startswith("Cancelled")


def test_a_photo_that_cannot_be_saved_ends_the_session_as_failed(make, tmp_path, book, quick):
    pipeline, _model = photo_pipeline(make, tmp_path)
    pipeline.start_photos(LINA_ID, 3)
    book.remove(LINA_ID)                                            # the folder went away meanwhile
    pipeline.start()
    assert wait_until(progress_is(pipeline, state="failed"))
    assert pipeline.photo_progress()["hint"].startswith("Could not save the photo")


def test_next_photo_number_and_save_photo_never_write_over_a_file(tmp_path):
    assert next_photo_number(tmp_path) == 1
    for name in ("photo_2.jpg", "photo_9.JPG", "photo_x.jpg", "other_20.jpg"):
        (tmp_path / name).write_bytes(b"old")
    assert next_photo_number(tmp_path) == 10
    assert save_photo(tmp_path, face_frame()).name == "photo_10.jpg"
    assert (tmp_path / "photo_9.JPG").read_bytes() == b"old"


# ---- enrolling again ------------------------------------------------------------------


def plain_jpeg(path, value):
    ok, data = cv2.imencode(".jpg", np.full((120, 160, 3), value, np.uint8))
    path.write_bytes(data.tobytes())


def test_reload_people_recognises_a_new_person_and_forgets_a_removed_one(make, tmp_path, book, monkeypatch):
    monkeypatch.setattr(enroll_module, "load_face_model", lambda settings: pytest.fail("a second model was loaded"))
    sara = book.save("Sara Ahmed", "Team lead")
    plain_jpeg(book.photos_dir / "sara_ahmed" / "photo_1.jpg", 20)       # dark: the fake model sees Sara
    book.save("Omar Ali")                                               # no photos
    # The face is checked only on the first frame and when told to: the memory keeps the rest.
    pipeline, model = photo_pipeline(make, tmp_path, every_n=1_000_000,
                                     source=FakeSource(frames=10_000, picture=np.full((360, 640, 3), 200, np.uint8)))
    pipeline.start()
    assert wait_until(lambda: [f.name for f in pipeline.state.snapshot().faces] == ["Unknown"])

    plain_jpeg(book.photos_dir / LINA_ID / "photo_1.jpg", 200)           # bright: Lina
    notes = pipeline.reload_people()
    assert pipeline.enrolled_ids() == {LINA_ID, "sara_ahmed"}
    assert "Not enrolled: omar_ali (no photos found)" in notes
    assert load_embeddings(tmp_path / "embeddings.npz")[0] == [LINA_ID, "sara_ahmed"]
    assert wait_until(lambda: [f.name for f in pipeline.state.snapshot().faces] == ["Lina Haddad"])
    assert "Nobody is enrolled yet" not in " ".join(pipeline.status()["notes"])

    book.remove(LINA_ID)
    pipeline.reload_people()
    assert pipeline.enrolled_ids() == {"sara_ahmed"}
    assert wait_until(lambda: [f.name for f in pipeline.state.snapshot().faces] == ["Unknown"])
    dark = np.full((360, 640, 3), 20, np.uint8)
    assert [f.name for f in pipeline._recognizer.recognize(dark)] == [sara.name]

    book.remove("sara_ahmed")
    pipeline.reload_people()
    assert pipeline.enrolled_ids() == set()
    assert not (tmp_path / "embeddings.npz").exists()
    assert NOBODY_ENROLLED in pipeline.status()["notes"]


def test_enrolling_again_keeps_the_video_running(make, tmp_path, book):
    for number in range(1, 9):
        plain_jpeg(book.photos_dir / LINA_ID / f"photo_{number}.jpg", 200)
    detector = FakeDetector(detections=[])
    model = ScriptedFaceModel(delay=0.03)                          # each picture takes 30 ms
    pipeline = make(source=FakeSource(frames=100_000, picture=face_frame()), detector=detector,
                    recognizer=real_recognizer(tmp_path, model), config=people_config(tmp_path, every_n=1))
    pipeline.start()
    assert wait_until(lambda: detector.calls > 0)
    before = detector.calls
    began = time.monotonic()
    pipeline.reload_people()                                       # 8 photos: at least 240 ms
    assert time.monotonic() - began >= 0.24
    assert detector.calls - before >= 3                            # frames went on while it ran
    assert pipeline.enrolled_ids() == {LINA_ID}


def test_with_recognition_off_every_people_method_fails_plainly(monkeypatch, tmp_path, book):
    def broken(*args, **kwargs):
        raise RuntimeError("CUDA provider missing\nmore detail")

    monkeypatch.setattr(pipeline_module, "Recognizer", broken)
    pipeline = Pipeline(people_config(tmp_path), source=FakeSource(frames=3), detector=FakeDetector())
    try:
        with pytest.raises(RuntimeError, match="^Face recognition is off: CUDA provider missing$"):
            pipeline.reload_people()
        with pytest.raises(ValueError, match="Face recognition is off: CUDA provider missing"):
            pipeline.start_photos(LINA_ID)
        assert pipeline.enrolled_ids() == set()
        assert pipeline.photo_progress() is None
        pipeline.cancel_photos()
        pipeline.start()
        assert wait_until(lambda: pipeline.scene_photo() is not None)
    finally:
        pipeline.stop()


# ---- the picture for "Describe the scene" ------------------------------------------------


def jpeg_size(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return image.shape[1], image.shape[0]


@pytest.mark.parametrize("size, max_side, expected", [
    ((640, 360), 960, (640, 360)),          # never enlarged
    ((640, 360), 640, (640, 360)),
    ((640, 360), 320, (320, 180)),
    ((1920, 1080), 960, (960, 540)),
    ((360, 640), 320, (180, 320)),          # the longer side counts, also when it is the height
])
def test_scene_photo_sizes(make, size, max_side, expected):
    width, height = size
    picture = np.full((height, width, 3), 90, np.uint8)
    pipeline = make(source=FakeSource(frames=3, picture=picture))
    assert pipeline.scene_photo() is None                          # before the first frame
    pipeline.start()
    assert wait_until(lambda: pipeline.scene_photo() is not None)
    assert jpeg_size(pipeline.scene_photo(max_side)) == expected


def test_scene_photo_is_the_clean_frame_and_none_while_the_view_is_cleared(make):
    picture = np.full((360, 640, 3), 90, np.uint8)
    pipeline = make(source=FakeSource(frames=3, picture=picture),
                    source_factory=SourceFactory(phone_source(frames=0, connected=False))).start()
    assert wait_until(lambda: pipeline.scene_photo() is not None)
    image = cv2.imdecode(np.frombuffer(pipeline.scene_photo(), np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(image.astype(int) - 90).max() <= 3                # no boxes drawn on it
    pipeline.use_phone(PHONE)
    assert pipeline.scene_photo() is None


@pytest.mark.parametrize("max_side", [0, -1, 2.5, "960", True])
def test_scene_photo_refuses_a_wrong_size(make, max_side):
    with pytest.raises(ValueError):
        make().scene_photo(max_side)


# ---- looking for things by name (M13) ------------------------------------------------------
#
# A fake open detector stands in for detector.OpenDetector, so no model, network or graphics
# card is needed. It notes every call and whether two calls ever overlapped.

KEYS = Detection("keys", 0.7, (20, 300, 60, 340), 7)                    # on the left
ROUTERS = [Detection("wifi router", 0.5, (500, 20, 560, 60), 3),         # on the right ...
           Detection("wifi router", 0.8, (300, 20, 340, 60), 4)]         # ... and, more sure, in the middle
OFFSET = pipeline_module.NAMED_ID_OFFSET


class FakeOpenDetector:
    """Shaped like detector.OpenDetector: finds those of `finds` whose label is one of its names.

    `load_gate` / `names_gate` hold load() / set_names() until they are set. `load_error`
    and `detect_error` are raised by those calls.
    """

    def __init__(self, finds=(KEYS, *ROUTERS), load_gate=None, names_gate=None,
                 load_error=None, detect_error=None):
        self.finds = list(finds)
        self.load_gate, self.names_gate = load_gate, names_gate
        self.load_error, self.detect_error = load_error, detect_error
        self._loaded = False
        self._names = ()
        self.calls = []
        self.overlaps = []                  # calls made while another call was still running
        self._busy = threading.Lock()
        self.inside = threading.Event()     # set while load() or set_names() waits on its gate

    @property
    def loaded(self):
        return self._loaded

    @property
    def names(self):
        return self._names

    def _enter(self, what):
        if not self._busy.acquire(blocking=False):
            self.overlaps.append(what)
            return False
        self.calls.append(what)
        return True

    def _leave(self, entered):
        if entered:
            self._busy.release()

    def load(self):
        entered = self._enter("load")
        try:
            if self.load_gate is not None:
                self.inside.set()
                assert self.load_gate.wait(5)
            if self.load_error is not None:
                raise self.load_error
            self._loaded = True
        finally:
            self._leave(entered)

    def set_names(self, names):
        entered = self._enter(("set_names", tuple(names)))
        try:
            if self.names_gate is not None:
                self.inside.set()
                assert self.names_gate.wait(5)
            self._names = tuple(names)
        finally:
            self._leave(entered)

    def detect(self, frame):
        entered = self._enter("detect")
        try:
            if self.detect_error is not None:
                raise self.detect_error
            return [find for find in self.finds if find.label in self._names]
        finally:
            self._leave(entered)

    def reset_tracks(self):
        self._leave(self._enter("reset_tracks"))

    def warm_up(self, frame_size):
        entered = self._enter(("warm_up", tuple(frame_size)))
        self._leave(entered)

    def count(self, what):
        return self.calls.count(what)


def search_of(pipeline):
    return pipeline.status()["search"]


def ready(pipeline):
    return lambda: search_of(pipeline)["state"] == "ready"


def searching(make, names=("keys", "wifi router"), open_detector=None, frames=10_000, **kwargs):
    """A running pipeline whose search for `names` was ready before the first frame."""
    open_detector = open_detector if open_detector is not None else FakeOpenDetector()
    pipeline = make(source=kwargs.pop("source", None) or FakeSource(frames=frames),
                    open_detector=open_detector, **kwargs)
    pipeline.search(list(names))
    assert wait_until(ready(pipeline))
    pipeline.start()
    return pipeline, open_detector


def relabelled(detection, label):
    return Detection(label, detection.confidence, detection.box, detection.track_id)


NO_TARGET = pipeline_module.Target("area", "", 0.0, None, b"", epoch=-1)


# ---- the name rules ---------------------------------------------------------------------


@pytest.mark.parametrize("names, cleaned", [
    (["power bank"], ("power bank",)),
    (["  Power   Bank  ", "keys"], ("Power Bank", "keys")),          # trimmed, single spaces
    (["Keys", "keys", "KEYS"], ("Keys",)),                            # case does not matter; the first stays
    (["wi-fi router", "O'Brien's mug", "Sara’s phone"], ("wi-fi router", "O'Brien's mug", "Sara’s phone")),
    (["مفاتيح", "चाबी", "clé", "USB 3"], ("مفاتيح", "चाबी", "clé", "USB 3")),   # any script, digits
    (["a" * 40], ("a" * 40,)),
    (["a", "b", "c", "d", "e", "f"], ("a", "b", "c", "d", "e", "f")),
    (["a", "b", "c", "d", "e", "f", "F"], ("a", "b", "c", "d", "e", "f")),      # 7 with a repeat is 6
    ([], ()),                                                         # stops looking
    (("keys",), ("keys",)),                                           # any sequence of text
])
def test_clean_names_keeps_good_names(names, cleaned):
    assert pipeline_module.clean_names(names) == cleaned


@pytest.mark.parametrize("names, words", [
    (["a", "b", "c", "d", "e", "f", "g"], "at most 6"),
    ([""], "1 to 40"),
    (["   "], "1 to 40"),
    (["a" * 41], "1 to 40"),
    (["keys!"], "letters, digits"),
    (["keys; drop"], "letters, digits"),
    (["<script>"], "letters, digits"),
    (["tab\there"], "letters, digits"),
    (["new\nline"], "letters, digits"),
    (["½ cup"], "letters, digits"),                                  # a number, but not a digit
    (["keys", 5], "text"),
    ([None], "text"),
    ("keys", "list"),                                                 # a bare string is not a list of names
    (None, "list"),
    ({"names": ["keys"]}, "list"),
])
def test_clean_names_refuses_bad_names_plainly(names, words):
    with pytest.raises(ValueError, match=words):
        pipeline_module.clean_names(names)


def test_search_refuses_bad_names_and_changes_nothing(make):
    open_detector = FakeOpenDetector()
    pipeline = make(open_detector=open_detector)
    with pytest.raises(ValueError):
        pipeline.search(["keys?"])
    assert search_of(pipeline)["state"] == "off"
    time.sleep(0.05)
    assert open_detector.calls == []


# ---- the search states ------------------------------------------------------------------


def test_status_search_is_off_before_any_search_and_json_ready(make):
    status = make().status()
    assert status["search"] == {"names": [], "state": "off", "note": "", "found": [], "missing": [],
                                 "seen": False}
    json.dumps(status)


def test_search_returns_at_once_then_loads_in_the_background(make):
    gate = threading.Event()
    open_detector = FakeOpenDetector(load_gate=gate)
    pipeline = make(open_detector=open_detector)
    began = time.monotonic()
    pipeline.search(["keys"])
    assert time.monotonic() - began < 0.2
    assert open_detector.inside.wait(2)                     # load() is under way ...
    assert search_of(pipeline) == {"names": ["keys"], "state": "loading", "note": "", "found": [], "missing": [],
                                 "seen": False}
    gate.set()                                              # ... and done
    assert wait_until(ready(pipeline))
    assert open_detector.calls == ["load", ("set_names", ("keys",)), ("warm_up", (640, 480))]


def test_a_loaded_open_detector_is_not_loaded_again(make):
    pipeline, open_detector = searching(make, names=["keys"])
    pipeline.search(["wallet"])
    assert wait_until(ready(pipeline))
    assert open_detector.count("load") == 1
    assert ("set_names", ("wallet",)) in open_detector.calls


def test_an_empty_search_stops_looking_at_once(make):
    pipeline, open_detector = searching(make)
    assert wait_until(lambda: search_of(pipeline)["found"])
    pipeline.search([])
    assert search_of(pipeline) == {"names": [], "state": "off", "note": "", "found": [], "missing": [],
                                 "seen": False}
    assert wait_until(lambda: not any(o.get("found") for o in pipeline.status()["objects"]))
    time.sleep(0.05)                                        # a frame already under way may still run it once
    calls = open_detector.count("detect")
    time.sleep(0.1)
    assert open_detector.count("detect") == calls          # the open detector is not used any more
    assert all(o["track_id"] < OFFSET for o in pipeline.status()["objects"])


def test_the_newest_search_wins(make):
    gate = threading.Event()
    open_detector = FakeOpenDetector(load_gate=gate)
    pipeline = make(open_detector=open_detector)
    pipeline.search(["keys"])
    assert open_detector.inside.wait(2)                     # the first round is loading the model ...
    pipeline.search(["wallet"])
    pipeline.search(["wifi router", "phone"])               # ... while two newer searches come in
    assert search_of(pipeline)["names"] == ["wifi router", "phone"]
    assert search_of(pipeline)["state"] == "loading"
    gate.set()
    assert wait_until(ready(pipeline))
    time.sleep(0.05)
    # The older rounds gave up: only the newest names were ever set, and the model was loaded once.
    assert open_detector.calls == ["load", ("set_names", ("wifi router", "phone")), ("warm_up", (640, 480))]
    assert search_of(pipeline)["names"] == ["wifi router", "phone"]
    assert open_detector.overlaps == []


def test_a_failed_load_says_why_in_the_status_and_the_video_goes_on(make):
    message = "Could not load the open detector yoloe-26s-seg.pt; its first use needs the internet."
    open_detector = FakeOpenDetector(load_error=RuntimeError(message + "\nmore detail"))
    detector = FakeDetector()
    pipeline = make(source=FakeSource(frames=10_000), detector=detector, open_detector=open_detector).start()
    pipeline.search(["keys"])
    assert wait_until(lambda: search_of(pipeline)["state"] == "failed")
    assert search_of(pipeline) == {"names": ["keys"], "state": "failed", "note": message,
                                   "found": [], "missing": [], "seen": False}
    calls = detector.calls
    assert wait_until(lambda: detector.calls > calls + 3)  # the video goes on
    assert pipeline.status()["error"] is None
    open_detector.load_error = None                         # e.g. the internet is back
    pipeline.search(["keys"])
    assert wait_until(ready(pipeline))
    assert search_of(pipeline)["note"] == ""


def test_an_open_detector_that_cannot_be_built_fails_the_search_only(make, monkeypatch):
    from car_vision import detector as detector_module

    def broken(settings):
        raise ConfigError("detector.open_confidence must be above 0 and at most 1, got 5")

    monkeypatch.setattr(detector_module, "OpenDetector", broken, raising=False)
    pipeline = make()
    pipeline.search(["keys"])
    assert wait_until(lambda: search_of(pipeline)["state"] == "failed")
    assert "open_confidence" in search_of(pipeline)["note"]


def test_a_detect_that_raises_fails_the_search_but_not_the_video(make):
    open_detector = FakeOpenDetector(detect_error=RuntimeError("CUDA out of memory"))
    pipeline, _ = searching(make, open_detector=open_detector)
    assert wait_until(lambda: search_of(pipeline)["state"] == "failed")
    assert search_of(pipeline)["note"] == "CUDA out of memory"
    time.sleep(0.1)
    assert open_detector.count("detect") == 1               # not tried again on every frame
    assert pipeline.status()["error"] is None and pipeline.status()["objects"]


# ---- the loop and the open detector ------------------------------------------------------


def test_the_loop_skips_the_named_search_while_the_background_thread_works(make):
    gate = threading.Event()
    open_detector = FakeOpenDetector(names_gate=gate)
    detector = FakeDetector()
    pipeline = make(source=FakeSource(frames=10_000), detector=detector, open_detector=open_detector).start()
    pipeline.search(["keys"])
    assert open_detector.inside.wait(2)                     # set_names() is under way
    calls = detector.calls
    assert wait_until(lambda: detector.calls > calls + 5)   # the video goes on meanwhile ...
    assert open_detector.count("detect") == 0               # ... without the named search
    assert search_of(pipeline)["state"] == "loading"
    gate.set()
    assert wait_until(lambda: open_detector.count("detect") > 0)
    assert open_detector.overlaps == []


def test_the_loop_never_waits_for_the_open_detector(make):
    pipeline, open_detector = searching(make)
    assert wait_until(lambda: open_detector.count("detect") > 0)
    detector = pipeline._detector
    with pipeline._open_lock:                               # as if the background thread held it
        detects = open_detector.count("detect")
        calls = detector.calls
        assert wait_until(lambda: detector.calls > calls + 10)
        assert open_detector.count("detect") == detects
    assert wait_until(lambda: open_detector.count("detect") > detects)


def test_the_open_detector_is_never_used_by_two_threads_at_once(make):
    pipeline, open_detector = searching(make)
    for names in (["wallet"], ["keys"], ["keys", "wifi router"], ["phone"], ["keys"]):
        pipeline.search(names)
        time.sleep(0.02)
    assert wait_until(ready(pipeline))
    assert wait_until(lambda: open_detector.calls[-1] == "detect")
    assert open_detector.overlaps == []


def test_the_named_search_runs_on_every_nth_frame(make):
    detector = FakeDetector()
    pipeline, open_detector = searching(make, frames=10, detector=detector)   # open_every_n_frames is 3
    assert wait_until(lambda: detector.calls == 10)
    time.sleep(0.05)
    assert open_detector.count("detect") == 4               # frames 1, 4, 7 and 10


def test_a_new_search_runs_on_its_first_ready_frame(make):
    detector = FakeDetector(delay=0.005)
    pipeline, open_detector = searching(make, detector=detector)
    assert wait_until(lambda: open_detector.count("detect") >= 1)
    pipeline.search(["wallet"])
    assert wait_until(ready(pipeline))
    position = len(open_detector.calls)
    calls = detector.calls
    assert wait_until(lambda: detector.calls >= calls + 7)
    assert open_detector.calls[position:].count("detect") >= 2


def test_the_finds_stay_on_screen_between_runs_and_are_highlighted(make, monkeypatch):
    wanted_per_frame = []
    real_draw = pipeline_module.draw.draw_detections

    def recording_draw(frame, detections, wanted=()):
        wanted_per_frame.append(sorted(wanted))
        real_draw(frame, detections, wanted)

    monkeypatch.setattr(pipeline_module.draw, "draw_detections", recording_draw)
    detector = FakeDetector()
    pipeline, open_detector = searching(make, frames=7, detector=detector)
    assert wait_until(lambda: detector.calls == 7)
    time.sleep(0.05)
    assert open_detector.count("detect") == 3               # frames 1, 4 and 7 ...
    assert wanted_per_frame == [[OFFSET + 3, OFFSET + 4, OFFSET + 7]] * 7    # ... shown on all seven
    drawn = decoded(pipeline.latest_jpeg())
    yellow = np.array(pipeline_module.draw.WANTED_COLOR)
    assert np.abs(drawn[300:341, 20] - yellow).sum(axis=1).min() < 60       # the keys' box is yellow


def test_without_a_search_draw_gets_no_highlights_and_nothing_is_found(make, monkeypatch):
    wanted_per_frame = []
    monkeypatch.setattr(pipeline_module.draw, "draw_detections",
                        lambda frame, detections, wanted=(): wanted_per_frame.append(list(wanted)))
    detector = FakeDetector()
    open_detector = FakeOpenDetector()
    pipeline = make(source=FakeSource(frames=5), detector=detector, open_detector=open_detector).start()
    assert wait_until(lambda: detector.calls == 5)
    time.sleep(0.05)
    assert wanted_per_frame == [[]] * 5
    assert open_detector.calls == []
    assert not any("found" in entry for entry in pipeline.status()["objects"])


# ---- what the page sees -----------------------------------------------------------------------


def test_status_says_what_was_found_where_and_what_is_missing(make):
    finds = [KEYS, *[relabelled(router, "Wifi Router") for router in ROUTERS]]
    pipeline, _ = searching(make, names=["Wifi Router", "keys", "power bank"],
                            open_detector=FakeOpenDetector(finds=finds))
    assert wait_until(lambda: search_of(pipeline)["found"])
    assert search_of(pipeline) == {"names": ["Wifi Router", "keys", "power bank"], "state": "ready", "note": "",
                                   "found": [{"name": "Wifi Router", "count": 2, "place": "middle"},  # the surer one
                                             {"name": "keys", "count": 1, "place": "left"}],
                                   "missing": ["power bank"], "seen": True}
    json.dumps(pipeline.status())


def test_found_objects_are_marked_in_the_list_and_can_be_clicked(make):
    pipeline, _ = searching(make, names=["keys"])
    assert wait_until(lambda: any(o["label"] == "keys" for o in pipeline.status()["objects"]))
    objects = pipeline.status()["objects"]
    keys = next(o for o in objects if o["label"] == "keys")
    assert keys["track_id"] == OFFSET + 7 and keys["found"] is True
    assert all("found" not in o for o in objects if o["label"] != "keys")
    target = pipeline.target(track_id=OFFSET + 7)
    assert (target.kind, target.label, target.track_id) == ("object", "keys", OFFSET + 7)
    assert pipeline.target(**at(40, 320)).label == "keys"    # a click on it
    assert "keys" in [d.label for d in pipeline.state.snapshot().detections]   # what the assistant is told


def test_a_find_replaces_the_everyday_box_it_overlaps(make):
    office_chair = Detection("office chair", 0.6, (402, 198, 500, 342), 1)    # over chair #2
    pipeline, _ = searching(make, names=["office chair"], open_detector=FakeOpenDetector(finds=[office_chair]))
    assert wait_until(lambda: any(o["label"] == "office chair" for o in pipeline.status()["objects"]))
    pairs = [(o["label"], o["track_id"]) for o in pipeline.status()["objects"]]
    assert pairs == [("person", 1), ("chair", 3), ("office chair", OFFSET + 1)]
    assert pipeline.target(track_id=2) is None              # the chair it replaced cannot be clicked either


def test_the_search_status_finds_nothing_while_the_stream_is_down(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "STALE_SECONDS", 0.2)
    pipeline, _ = searching(make, names=["keys"], frames=3)
    assert wait_until(lambda: search_of(pipeline)["found"])
    pipeline._source.connected = False
    assert wait_until(lambda: search_of(pipeline)["found"] == [])
    assert search_of(pipeline)["missing"] == [] and search_of(pipeline)["seen"] is False   # not "not in view"


def test_finds_of_replaced_names_never_reach_the_page(make):
    """Other names are asked for while a frame's named finds are being worked out:
    that frame is dropped, so the old names' finds never show after search() returned."""
    class HoldingOpenDetector(FakeOpenDetector):
        def __init__(self):
            super().__init__()
            self.holding, self.go_on = threading.Event(), threading.Event()

        def detect(self, frame):
            found = super().detect(frame)
            if "keys" in self.names and not self.go_on.is_set():
                self.holding.set()
                assert self.go_on.wait(5)
            return found

    open_detector = HoldingOpenDetector()
    pipeline, _ = searching(make, names=["keys"], open_detector=open_detector)
    assert open_detector.holding.wait(3)                    # the first named run is under way
    pipeline.search(["wallet"])
    open_detector.go_on.set()
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        assert not any(o["label"] == "keys" for o in pipeline.status()["objects"])
        assert pipeline.target(track_id=OFFSET + 7) is None
        time.sleep(0.005)


# ---- the track epoch ---------------------------------------------------------------------------


def test_the_epoch_goes_up_when_the_names_change(make):
    pipeline, _ = searching(make, names=["keys"])
    assert wait_until(lambda: pipeline.target(track_id=OFFSET + 7) is not None)
    first = pipeline.target(track_id=OFFSET + 7).epoch
    assert first == 1                                       # setting names the first time started tracks
    assert pipeline.target(track_id=2).epoch == first       # one epoch for every target of a view
    pipeline.search(["keys"])                               # the same names: the tracks go on
    assert wait_until(ready(pipeline))
    time.sleep(0.05)
    assert pipeline.target(track_id=OFFSET + 7).epoch == first
    pipeline.search(["keys", "wifi router"])                # other names: the track numbers start again
    assert wait_until(ready(pipeline))
    assert wait_until(lambda: (pipeline.target(track_id=OFFSET + 7) or NO_TARGET).epoch == first + 1)


def test_a_new_frame_size_starts_the_open_detectors_tracks_again(make):
    source = FakeSource(frames=4)
    pipeline, open_detector = searching(make, names=["keys"], source=source)
    assert wait_until(lambda: source.sequence == 4)
    time.sleep(0.05)
    resets = open_detector.count("reset_tracks")
    source.picture = np.full((720, 1280, 3), 60, np.uint8)
    source.frames += 4
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (1280, 720))
    assert wait_until(lambda: open_detector.count("reset_tracks") == resets + 1)
    last_reset = len(open_detector.calls) - 1 - open_detector.calls[::-1].index("reset_tracks")
    assert open_detector.calls[last_reset + 1] == "detect"  # before the first detect at the new size


def test_a_camera_switch_drops_the_finds_and_keeps_looking(make):
    pipeline, _ = searching(make, names=["keys"],
                            source_factory=SourceFactory(phone_source(frames=0, connected=False)))
    assert wait_until(lambda: search_of(pipeline)["found"])
    pipeline.use_phone(PHONE)
    assert pipeline.status()["objects"] == []
    assert search_of(pipeline)["found"] == [] and search_of(pipeline)["state"] == "ready"


# ---- merge_named: the rules on their own --------------------------------------------------------

CHAIR = Detection("chair", 0.8, (400, 200, 500, 340), 2)
CUP = Detection("cup", 0.6, (100, 100, 140, 150), 5)
NOTHING_KEPT = pipeline_module.NamedFinds(1)


def merge(everyday, kept, fresh, search_round=1, now=10.0):
    return pipeline_module.merge_named(everyday, kept, fresh, search_round, now)


def test_merge_adds_the_named_finds_with_their_ids_kept_apart():
    shown, kept = merge([CHAIR, CUP], NOTHING_KEPT, [Detection("keys", 0.7, (0, 0, 9, 9), 2)])
    assert shown == [CHAIR, CUP, Detection("keys", 0.7, (0, 0, 9, 9), OFFSET + 2)]   # not a second #2
    assert kept.search_round == 1 and kept.finds == ((shown[-1], 10.0),)
    untracked, _ = merge([], NOTHING_KEPT, [Detection("keys", 0.7, (0, 0, 9, 9))])
    assert untracked[0].track_id is None


def test_merge_with_nothing_named_gives_the_everyday_detections():
    shown, kept = merge([CHAIR, CUP], NOTHING_KEPT, None)
    assert shown == [CHAIR, CUP] and kept.finds == ()
    shown, _ = merge([CHAIR, CUP], NOTHING_KEPT, [])
    assert shown == [CHAIR, CUP]


@pytest.mark.parametrize("box, replaced", [
    ((400, 200, 500, 340), True),           # the same box: 1.0
    ((410, 210, 500, 340), True),           # 0.84
    ((400, 200, 500, 285), True),           # 0.61
    ((400, 200, 500, 284), False),          # 0.60: not more than 0.6
    ((450, 200, 550, 340), False),          # 0.33
    ((0, 0, 50, 50), False),                # apart
])
def test_merge_replaces_an_everyday_box_only_when_it_overlaps_more_than_06(box, replaced):
    shown, _ = merge([CHAIR, CUP], NOTHING_KEPT, [Detection("office chair", 0.5, box, 1)])
    assert (CHAIR not in shown) is replaced
    assert CUP in shown and shown[-1].label == "office chair"


def test_overlap():
    assert pipeline_module.overlap((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert pipeline_module.overlap((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert pipeline_module.overlap((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert pipeline_module.overlap((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
    assert pipeline_module.overlap(CHAIR.box, (400, 200, 500, 284)) == pytest.approx(0.6)


def test_merge_keeps_finds_up_to_a_second_between_runs_then_drops_them():
    keys = Detection("keys", 0.7, (0, 0, 9, 9), 7)
    _, kept = merge([], NOTHING_KEPT, [keys], now=10.0)
    shown, kept = merge([], kept, None, now=10.5)
    assert [d.track_id for d in shown] == [OFFSET + 7]
    shown, kept = merge([], kept, None, now=11.0)           # exactly a second: still there
    assert [d.track_id for d in shown] == [OFFSET + 7]
    shown, kept = merge([], kept, None, now=11.01)          # longer: gone
    assert shown == [] and kept.finds == ()


def test_a_run_that_misses_a_tracked_find_keeps_it_for_its_second():
    keys = Detection("keys", 0.7, (0, 0, 9, 9), 7)
    wallet = Detection("wallet", 0.6, (100, 100, 150, 150), 8)
    _, kept = merge([], NOTHING_KEPT, [keys, wallet], now=10.0)
    shown, kept = merge([], kept, [replace(keys, confidence=0.9)], now=10.4)     # the wallet was missed
    assert [(d.label, d.confidence) for d in shown] == [("keys", 0.9), ("wallet", 0.6)]
    shown, kept = merge([], kept, [keys], now=11.2)          # missed again, last seen 1.2 s ago
    assert [d.label for d in shown] == ["keys"]


def test_a_run_drops_an_old_find_a_fresh_one_covers_and_old_untracked_ones():
    old = Detection("keys", 0.7, (0, 0, 100, 100), 7)
    untracked = Detection("keys", 0.5, (300, 300, 340, 340), None)
    _, kept = merge([], NOTHING_KEPT, [old, untracked], now=10.0)
    renumbered = Detection("keys", 0.7, (2, 0, 100, 100), 9)                   # the same thing, a new track
    shown, _ = merge([], kept, [renumbered], now=10.2)
    assert [d.track_id for d in shown] == [OFFSET + 9]


def test_merge_keeps_nothing_across_a_change_of_names():
    keys = Detection("keys", 0.7, (0, 0, 9, 9), 7)
    _, kept = merge([], NOTHING_KEPT, [keys], search_round=1, now=10.0)
    shown, kept = merge([CHAIR], kept, None, search_round=2, now=10.1)
    assert shown == [CHAIR] and kept == pipeline_module.NamedFinds(2)


@pytest.mark.parametrize("x1, x2, place", [(0, 10, "left"), (200, 220, "left"), (210, 220, "middle"),
                                           (300, 340, "middle"), (420, 440, "right"), (630, 639, "right")])
def test_place_of_is_the_third_of_the_picture_the_centre_is_in(x1, x2, place):
    assert pipeline_module.place_of((x1, 0, x2, 10), 640) == place


def test_search_status_lists_found_and_missing_only_when_ready():
    finds = [Detection("keys", 0.5, (500, 0, 520, 10), 1), Detection("keys", 0.9, (0, 0, 20, 10), 2)]
    loading = pipeline_module.Search(("keys", "wallet"), "loading", "", 3)
    assert pipeline_module.search_status(loading, finds, 640) == {
        "names": ["keys", "wallet"], "state": "loading", "note": "", "found": [], "missing": [], "seen": False}
    ready_search = replace(loading, state="ready")
    assert pipeline_module.search_status(ready_search, finds, 640)["found"] == [
        {"name": "keys", "count": 2, "place": "left"}]
    assert pipeline_module.search_status(ready_search, finds, 640)["missing"] == ["wallet"]


# ---- start-up ------------------------------------------------------------------------------------


def test_start_up_builds_no_open_detector_and_the_first_search_does(make, monkeypatch):
    from car_vision import detector as detector_module
    built = []

    def recording(settings):
        built.append(settings)
        return FakeOpenDetector()

    monkeypatch.setattr(detector_module, "OpenDetector", recording, raising=False)
    detector = FakeDetector()
    pipeline = make(source=FakeSource(frames=5), detector=detector).start()
    assert wait_until(lambda: detector.calls == 5)
    assert built == [] and pipeline._open_detector is None
    pipeline.search(["keys"])
    assert wait_until(ready(pipeline))
    pipeline.search(["wallet"])
    assert wait_until(ready(pipeline))
    assert built == [CONFIG["detector"]]                    # built once, from the detector settings


@pytest.mark.parametrize("value", [0, -1, "3", 2.5, True])
def test_a_bad_open_every_n_frames_is_rejected(value):
    config = {**CONFIG, "detector": {**CONFIG["detector"], "open_every_n_frames": value}}
    with pytest.raises(ConfigError, match="open_every_n_frames"):
        Pipeline(config, source=FakeSource(), detector=FakeDetector(), recognizer=FakeRecognizer())


@pytest.mark.parametrize("key", ["file", "keep_days", "min_seconds", "pictures"])
def test_every_memory_setting_is_required(key):
    memory = {name: value for name, value in CONFIG["memory"].items() if name != key}
    with pytest.raises(ConfigError, match=key):
        Pipeline({**CONFIG, "memory": memory}, source=FakeSource(), detector=FakeDetector(),
                 recognizer=FakeRecognizer())


# ---- the memory ------------------------------------------------------------------------------------


class FakeSightings:
    """Shaped like memory.SightingLog: keeps every seen() call; summary() gives `lines`."""

    def __init__(self, lines=("laptop: last seen just now on the left, in view for 2 seconds",), fail=False):
        self.lines = list(lines)
        self.fail = fail
        self.seen_calls = []
        self.summaries = []

    def seen(self, key, kind, label, name, place, confidence, now, picture=None):
        if self.fail:
            raise RuntimeError("database is locked")
        self.seen_calls.append({"key": key, "kind": kind, "label": label, "name": name, "place": place,
                                "confidence": confidence, "now": now, "picture": picture})

    def summary(self, now, limit=30):
        self.summaries.append(now)
        return list(self.lines)

    def tick(self, now):
        self.ticks = getattr(self, "ticks", 0) + 1

    def close(self):
        self.closed = True


class RecordingState(WorldState):
    """A WorldState that keeps the memory lines it is given (whether or not Part B has landed)."""

    def set_memory(self, lines):
        self.memory_lines = tuple(lines)


def memory_config(min_seconds=0.0, pictures=True, file=None):
    return {**CONFIG, "memory": {**CONFIG["memory"], "file": file, "min_seconds": min_seconds,
                                 "pictures": pictures}}


def remembering(make, sightings, detections=None, frames=4, config=None, **kwargs):
    detector = FakeDetector(detections=detections)
    pipeline = make(source=FakeSource(frames=frames), detector=detector, sightings=sightings,
                    config=config or memory_config(), state=RecordingState(), **kwargs).start()
    assert wait_until(lambda: detector.calls == frames)
    time.sleep(0.05)
    return pipeline


UNTRACKED_CUP = Detection("cup", 0.6, (100, 100, 140, 150), None)


def test_every_tracked_detection_is_told_to_the_memory_and_untracked_ones_are_not(make):
    sightings = FakeSightings()
    before = time.time()
    pipeline = remembering(make, sightings, detections=[PERSON, *CHAIRS, UNTRACKED_CUP])
    assert pipeline.sightings() is sightings
    calls = sightings.seen_calls
    assert len(calls) == 4 * 3                              # 3 tracked things on each of 4 frames
    assert {call["key"] for call in calls} == {(0, 1, "person"), (0, 2, "chair"), (0, 3, "chair")}
    assert "cup" not in {call["label"] for call in calls}
    person = next(call for call in calls if call["label"] == "person")
    assert (person["kind"], person["name"], person["place"]) == ("person", "Sara Ahmed", "left")
    chair = calls[-1]
    assert (chair["kind"], chair["label"], chair["name"], chair["place"]) == ("object", "chair", None, "right")
    assert chair["confidence"] == pytest.approx(0.7)
    assert before - 1 <= chair["now"] <= time.time()        # seconds since 1970, like time.time()


def test_an_unrecognised_person_is_remembered_without_a_name(make):
    sightings = FakeSightings()
    remembering(make, sightings, detections=[PERSON],
                recognizer=FakeRecognizer(faces=[RecognizedFace("Unknown", 0.2, SARA_FACE.box, None, 1)]))
    assert {call["name"] for call in sightings.seen_calls} == {None}


def test_people_never_get_a_picture_and_objects_get_one_now_and_then(make):
    sightings = FakeSightings()
    big_chair = Detection("chair", 0.8, (0, 0, 640, 300), 2)
    remembering(make, sightings, detections=[PERSON, big_chair], frames=6)
    assert all(call["picture"] is None for call in sightings.seen_calls if call["kind"] == "person")
    pictures = [call["picture"] for call in sightings.seen_calls if call["picture"] is not None]
    assert len(pictures) == 1                               # once within PICTURE_EVERY_SECONDS, not 6 times
    picture = cv2.imdecode(np.frombuffer(pictures[0], np.uint8), cv2.IMREAD_COLOR)
    assert picture.shape == (75, 160, 3)                    # 640 x 300, its longer side made 160
    assert len(pictures[0]) <= 40_000


def test_the_first_picture_waits_for_min_seconds(make):
    sightings = FakeSightings()
    remembering(make, sightings, config=memory_config(min_seconds=30.0))
    assert sightings.seen_calls and all(call["picture"] is None for call in sightings.seen_calls)


def test_no_pictures_when_memory_pictures_is_off(make):
    sightings = FakeSightings()
    remembering(make, sightings, config=memory_config(pictures=False))
    assert sightings.seen_calls and all(call["picture"] is None for call in sightings.seen_calls)


def test_named_finds_are_remembered_with_the_typed_name(make):
    sightings = FakeSightings()
    searching(make, names=["keys"], sightings=sightings, config=memory_config(), state=RecordingState())
    assert wait_until(lambda: any(call["label"] == "keys" for call in sightings.seen_calls))
    keys = next(call for call in sightings.seen_calls if call["label"] == "keys")
    assert keys["key"] == (0, 1, OFFSET + 7, "keys")        # everyday epoch, named epoch, track, label
    assert keys["kind"] == "object" and keys["place"] == "left"
    assert keys["picture"] is not None


def test_a_new_frame_size_gives_new_memory_keys(make):
    sightings = FakeSightings()
    source = FakeSource(frames=2)
    pipeline = make(source=source, sightings=sightings, config=memory_config(), state=RecordingState()).start()
    assert wait_until(lambda: source.sequence == 2)
    time.sleep(0.05)
    source.picture = np.full((720, 1280, 3), 60, np.uint8)
    source.frames += 2
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (1280, 720))
    time.sleep(0.05)
    assert {call["key"][0] for call in sightings.seen_calls} == {0, 1}


def test_the_memory_summary_reaches_the_world_state(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "MEMORY_SHARE_SECONDS", 0.05)
    lines = ["laptop: last seen just now on the left, in view for 2 seconds",
             "an unrecognised person (person): last seen 1 minute ago in the middle, in view for 5 seconds"]
    sightings = FakeSightings(lines)
    state = RecordingState()
    source = FakeSource(frames=0)
    source.connected = False                                # also while no frames come
    make(source=source, sightings=sightings, config=memory_config(), state=state).start()
    assert wait_until(lambda: getattr(state, "memory_lines", None) == tuple(lines))
    count = len(sightings.summaries)
    time.sleep(0.25)
    assert 2 <= len(sightings.summaries) - count <= 8       # about every 0.05 s, not on every turn
    assert abs(sightings.summaries[-1] - time.time()) < 60  # asked with the wall clock


def test_the_summary_is_shared_every_5_seconds_by_default():
    assert pipeline_module.MEMORY_SHARE_SECONDS == 5.0


def test_a_broken_memory_never_stops_the_video(make, caplog):
    caplog.set_level(logging.WARNING, logger="car_vision.pipeline")
    pipeline = remembering(make, FakeSightings(fail=True), frames=30)
    assert pipeline.status()["error"] is None and pipeline.status()["objects"]
    logged = [record for record in caplog.records if "memory" in record.getMessage()]
    assert len(logged) == 1                                 # once, not on every frame


def fake_memory_module(monkeypatch, sighting_log):
    import types
    module = types.ModuleType("car_vision.memory")
    module.SightingLog = sighting_log
    monkeypatch.setitem(sys.modules, "car_vision.memory", module)


def test_a_memory_that_cannot_start_leaves_a_note_and_the_video_runs(make, monkeypatch, tmp_path):
    def cannot_open(path, keep_days=7, min_seconds=1.0):
        raise OSError("unable to open database file")

    fake_memory_module(monkeypatch, cannot_open)
    detector = FakeDetector()
    pipeline = make(source=FakeSource(frames=3), detector=detector,
                    config=memory_config(file=str(tmp_path / "memory.sqlite3"))).start()
    assert pipeline.sightings() is None
    assert pipeline.status()["notes"] == ["The camera's memory is off: unable to open database file"]
    assert wait_until(lambda: detector.calls == 3)


def test_the_memory_file_from_the_settings_is_opened_at_start_up(make, monkeypatch, tmp_path):
    opened = []

    def recording(path, keep_days=7, min_seconds=1.0):
        opened.append((path, keep_days, min_seconds))
        return FakeSightings()

    fake_memory_module(monkeypatch, recording)
    pipeline = make(config=memory_config(min_seconds=1.5, file=str(tmp_path / "memory.sqlite3")))
    assert opened == [(tmp_path / "memory.sqlite3", 7.0, 1.5)]
    assert isinstance(pipeline.sightings(), FakeSightings)


@pytest.mark.parametrize("file", [None, ""])
def test_no_memory_file_means_no_memory(make, file):
    pipeline = make(config=memory_config(file=file))
    assert pipeline.sightings() is None
    assert pipeline.status()["notes"] == []


def test_with_the_real_sighting_log_things_are_remembered(make, tmp_path):
    memory = pytest.importorskip("car_vision.memory")
    if not hasattr(WorldState, "set_memory"):
        pytest.skip("WorldState.set_memory is not there yet (Part B)")
    sightings = memory.SightingLog(tmp_path / "memory.sqlite3", keep_days=7, min_seconds=0.0)
    try:
        make(source=FakeSource(frames=10_000), detector=FakeDetector(delay=0.01), sightings=sightings,
             config=memory_config()).start()
        assert wait_until(lambda: {"chair", "person"} <= {s.label for s in sightings.recent()}, timeout=5)
        people = [s for s in sightings.recent() if s.kind == "person"]
        assert not any(s.has_picture for s in people)
        assert [s.name for s in people] == ["Sara Ahmed"]
    finally:
        sightings.close()


# ---- the picture timer ----------------------------------------------------------------------------


def test_the_picture_timer_waits_for_min_seconds_then_gives_one_every_few_seconds():
    timer = pipeline_module.PictureTimer(first_after=1.0, every=5.0)
    assert [timer.due("chair", t) for t in (0.0, 0.5, 0.99)] == [False, False, False]
    assert timer.due("chair", 1.0) is True                  # in view for min_seconds: the first picture
    assert [timer.due("chair", t) for t in (1.1, 3.0, 5.9)] == [False, False, False]
    assert timer.due("chair", 6.0) is True
    assert timer.due("cup", 6.0) is False                   # each thing has its own times


def test_the_picture_timer_forgets_things_not_seen_for_a_while():
    timer = pipeline_module.PictureTimer(first_after=0.0, every=5.0, forget_after=10.0)
    assert timer.due("chair", 0.0) is True
    timer.due("cup", 5.0)
    timer.forget_old(10.5)
    assert len(timer) == 1                                  # the chair, last seen at 0.0, is gone
    assert timer.due("chair", 10.5) is True                 # so it is a new thing again


# ---- with the real OpenDetector (Part A) on a fake model --------------------------------------------


class YoloeShaped:
    """Just enough of ultralytics.YOLOE for OpenDetector: text calls, and track() finding
    the first name at the keys' place with track number 5."""

    def __init__(self):
        self.classes = []

    def get_text_pe(self, names):
        return list(names)

    def set_classes(self, names, embeddings):
        self.classes.append(list(names))

    def track(self, frame, **kwargs):
        assert "half" not in kwargs

        class Boxes:
            xyxy, conf, cls, id = [list(KEYS.box)], [0.7], [0], [5]

            def __len__(self):
                return 1

        class Result:
            boxes, names = Boxes(), {}

        return [Result()]


def test_with_the_real_open_detector_a_typed_name_is_found(make, tmp_path):
    from car_vision import detector as detector_module
    if not hasattr(detector_module, "OpenDetector"):
        pytest.skip("detector.OpenDetector is not there yet (Part A)")
    settings = {**CONFIG["detector"], "models_dir": str(tmp_path)}    # its encoder folder: not the project's
    open_detector = detector_module.OpenDetector(settings, model=YoloeShaped())
    pipeline, _ = searching(make, names=["power bank"], open_detector=open_detector)
    assert wait_until(lambda: search_of(pipeline)["found"])
    assert search_of(pipeline)["found"] == [{"name": "power bank", "count": 1, "place": "left"}]
    found = [o for o in pipeline.status()["objects"] if o.get("found")]
    assert [(o["label"], o["track_id"]) for o in found] == [("power bank", OFFSET + 5)]
    pipeline.search(["keys", "power bank"])
    assert wait_until(ready(pipeline))
    assert open_detector.names == ("keys", "power bank")


# ---- memory keys follow what really restarts ------------------------------------------------------


def test_memory_key_keeps_everyday_and_named_tracks_apart():
    chair = Detection("chair", 0.8, (0, 0, 9, 9), 2)
    keys = Detection("keys", 0.7, (0, 0, 9, 9), OFFSET + 7)
    assert pipeline_module.memory_key(chair, False, 3, 5) == (3, 2, "chair")      # the named epoch plays no part
    assert pipeline_module.memory_key(keys, True, 3, 5) == (3, 5, OFFSET + 7, "keys")


def change_the_search_while_in_view(pipeline):
    """Start a search, change its names, then stop it, while the frames go on."""
    for names in (["keys"], ["keys", "wifi router"], []):
        pipeline.search(names)
        if names:
            assert wait_until(ready(pipeline))
        time.sleep(0.15)


def test_searching_does_not_change_the_memory_keys_of_everyday_objects(make):
    sightings = FakeSightings()
    pipeline = make(source=FakeSource(frames=10_000), detector=FakeDetector(delay=0.005), sightings=sightings,
                    open_detector=FakeOpenDetector(), config=memory_config(), state=RecordingState()).start()
    assert wait_until(lambda: sightings.seen_calls)
    change_the_search_while_in_view(pipeline)
    chair_keys = {call["key"] for call in sightings.seen_calls if call["label"] == "chair" and call["key"][1] == 2}
    assert chair_keys == {(0, 2, "chair")}
    named_epochs = {call["key"][1] for call in sightings.seen_calls if call["label"] == "keys"}
    assert named_epochs == {1, 2}                           # the named tracks did start again with new names
    assert pipeline.target(track_id=2).epoch == 2           # Target.epoch still counts every restart


def test_a_search_started_changed_and_stopped_gives_one_sighting_of_an_object_in_view(make, tmp_path):
    memory = pytest.importorskip("car_vision.memory")
    sightings = memory.SightingLog(tmp_path / "memory.sqlite3", keep_days=7, min_seconds=0.0)
    try:
        couch = Detection("couch", 0.8, (300, 150, 600, 340), 4)
        pipeline = make(source=FakeSource(frames=10_000), detector=FakeDetector(detections=[couch], delay=0.005),
                        recognizer=FakeRecognizer(faces=[]), sightings=sightings,
                        open_detector=FakeOpenDetector(), config=memory_config()).start()
        assert wait_until(lambda: any(s.label == "couch" for s in sightings.recent()), timeout=5)
        change_the_search_while_in_view(pipeline)
        time.sleep(2.2)                                     # the log writes at most every 2 seconds
        pipeline.stop()
        sightings.close()                                   # writes what is still open
        reopened = memory.SightingLog(tmp_path / "memory.sqlite3", keep_days=7, min_seconds=0.0)
        try:
            assert [s.label for s in reopened.recent()].count("couch") == 1
        finally:
            reopened.close()
    finally:
        sightings.close()


# ---- review fixes (M13) ---------------------------------------------------------------------------

# Fix 1: never a person as an "object"


@pytest.mark.parametrize("name", ["man", "Women", "old man", "man's", "faces", "someone", "Baby",
                                  "red-hand", "PEOPLE", "anybody"])
def test_clean_names_refuses_words_for_people(name):
    with pytest.raises(ValueError) as refused:
        pipeline_module.clean_names([name])
    assert str(refused.value) == pipeline_module.NO_PEOPLE


@pytest.mark.parametrize("name", ["headphones", "mandarin", "keyboard", "facial cream"])
def test_names_that_only_contain_the_letters_of_a_people_word_are_fine(name):
    assert pipeline_module.clean_names([name]) == (name,)


PERSON_SQUARE = (0, 0, 100, 100)


@pytest.mark.parametrize("box, dropped", [
    ((20, 20, 60, 60), True),          # inside the person
    ((-10, -10, 110, 110), True),      # around the person: its centre is inside
    ((40, 0, 170, 100), True),         # centre (105, 50) outside, but overlapping 0.35
    ((60, 0, 200, 100), False),        # centre (130, 50) outside, overlapping 0.2
    ((110, 0, 200, 100), False),       # beside the person
])
def test_on_a_person(box, dropped):
    assert pipeline_module.on_a_person(Detection("coat", 0.7, box, OFFSET + 1), [PERSON_SQUARE]) is dropped


def test_merge_drops_a_find_on_a_person_and_the_person_keeps_their_box():
    on_her = Detection("jacket", 0.9, (110, 70, 250, 290), 3)        # nearly the person box
    shown, kept = merge([PERSON, CUP], NOTHING_KEPT, [on_her])
    assert shown == [PERSON, CUP] and kept.finds == ()


def test_a_named_find_on_a_person_never_turns_them_into_an_object(make):
    sightings = FakeSightings()
    on_sara = Detection("jacket", 0.9, (102, 62, 258, 298), 3)
    pipeline, open_detector = searching(make, names=["jacket"], open_detector=FakeOpenDetector(finds=[on_sara]),
                                        sightings=sightings, config=memory_config(), state=RecordingState())
    assert wait_until(lambda: open_detector.count("detect") >= 2)
    objects = pipeline.status()["objects"]
    assert [(o["label"], o["track_id"], o["name"]) for o in objects][0] == ("person", 1, "Sara Ahmed")
    assert not any(o["label"] == "jacket" for o in objects)
    assert pipeline.target(track_id=1).kind == "person"
    assert not any(call["label"] == "jacket" for call in sightings.seen_calls)
    assert search_of(pipeline)["missing"] == ["jacket"]


# Fix 2: the memory's periodic work and closing


def test_the_memory_is_ticked_also_while_nothing_comes_and_closed_at_stop(make):
    sightings = FakeSightings()
    source = FakeSource(frames=0)
    source.connected = False
    pipeline = make(source=source, sightings=sightings, config=memory_config(), state=RecordingState()).start()
    assert wait_until(lambda: getattr(sightings, "ticks", 0) > 20)
    assert sightings.seen_calls == []
    pipeline.stop()
    assert sightings.closed is True                         # handed in, and closed all the same


def test_a_thing_seen_briefly_reaches_the_file_without_another_sighting(make, tmp_path):
    memory = pytest.importorskip("car_vision.memory")
    sightings = memory.SightingLog(tmp_path / "memory.sqlite3", keep_days=7, min_seconds=0.2)
    bottle = Detection("bottle", 0.8, (300, 100, 340, 200), 9)
    source = FakeSource(frames=10_000)
    detector = FakeDetector(detections=[bottle], delay=0.01)
    pipeline = make(source=source, detector=detector, recognizer=FakeRecognizer(faces=[]),
                    sightings=sightings, config=memory_config()).start()
    time.sleep(0.6)
    detector.detections = []                                # taken away; nothing else is in view
    try:
        assert wait_until(lambda: [s.label for s in sightings.recent()] == ["bottle"], timeout=6)
        row = sightings.recent()[0]
        assert row.last_seen - row.first_seen >= 0.4         # its real time in view, not cut short
    finally:
        pipeline.stop()


# Fix 3: "seen": the finder has looked


def test_the_search_is_not_seen_until_the_finder_has_looked_at_a_frame(make):
    open_detector = FakeOpenDetector()
    source = FakeSource(frames=0)
    pipeline = make(source=source, open_detector=open_detector).start()
    pipeline.search(["power bank"])
    assert wait_until(ready(pipeline))
    time.sleep(0.05)
    search = search_of(pipeline)
    assert (search["seen"], search["found"], search["missing"]) == (False, [], [])   # no frame yet
    source.frames = 3
    assert wait_until(lambda: search_of(pipeline)["seen"])
    assert search_of(pipeline)["missing"] == ["power bank"]


def test_the_search_is_not_seen_while_the_stream_is_down(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "STALE_SECONDS", 0.2)
    pipeline, _ = searching(make, names=["keys"], frames=3)
    assert wait_until(lambda: search_of(pipeline)["seen"])
    pipeline._source.connected = False
    assert wait_until(lambda: search_of(pipeline)["seen"] is False)
    assert search_of(pipeline)["found"] == search_of(pipeline)["missing"] == []


def test_new_own_names_are_not_seen_until_looked_for(make):
    pipeline, _ = searching(make, names=["keys"], frames=3)
    assert wait_until(lambda: search_of(pipeline)["seen"] and pipeline._source.sequence == 3)
    time.sleep(0.05)                                        # the last frame is done: the camera is stalled
    pipeline.search(["wallet"])                             # the camera is stalled: no new frame
    assert wait_until(ready(pipeline))
    time.sleep(0.05)
    assert search_of(pipeline)["seen"] is False


# Fix 4: old finds vanish at once, also with a stalled camera


@pytest.mark.parametrize("names", [["wallet"], []])
def test_old_finds_vanish_at_once_from_the_list_and_the_clicks(make, names):
    pipeline, _ = searching(make, names=["keys"], frames=3)  # then the camera stalls
    assert wait_until(lambda: any(o["label"] == "keys" for o in pipeline.status()["objects"]))
    time.sleep(0.05)
    pipeline.search(names)
    assert not any(o["label"] == "keys" for o in pipeline.status()["objects"])
    assert pipeline.target(track_id=OFFSET + 7) is None
    assert pipeline.target(**at(40, 320)).label != "keys"
    assert [(o["label"], o["track_id"]) for o in pipeline.status()["objects"]] == [
        ("person", 1), ("chair", 2), ("chair", 3)]             # the everyday ones stay


def test_finds_of_names_still_looked_for_stay_when_others_are_added(make):
    pipeline, _ = searching(make, names=["keys"], frames=3)
    assert wait_until(lambda: any(o["label"] == "keys" for o in pipeline.status()["objects"]))
    pipeline.search(["keys", "wallet"])
    assert any(o["label"] == "keys" for o in pipeline.status()["objects"])


# Fix 5: the summary is read off the video loop


def test_the_summary_is_read_on_a_thread_of_its_own(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "MEMORY_SHARE_SECONDS", 0.05)
    threads = []

    class WhereAsked(FakeSightings):
        def summary(self, now, limit=30):
            threads.append(threading.current_thread().name)
            return super().summary(now, limit)

    make(source=FakeSource(frames=50), sightings=WhereAsked(), config=memory_config(),
         state=RecordingState()).start()
    assert wait_until(lambda: len(threads) >= 3)
    assert set(threads) == {"memory-summary"}


# Fix 6: warm-up off the loop


def test_the_open_detector_is_warmed_up_before_the_search_is_ready_and_only_for_new_names(make):
    pipeline, open_detector = searching(make, names=["keys"], frames=3)
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (640, 360))
    assert open_detector.calls[:3] == ["load", ("set_names", ("keys",)), ("warm_up", (640, 480))]
    pipeline.search(["wallet"])
    assert wait_until(ready(pipeline))
    assert open_detector.calls[-1] == ("warm_up", (640, 360))    # the size of the frames now
    warm_ups = len([c for c in open_detector.calls if c[0] == "warm_up"])
    pipeline.search(["wallet"])                                  # the same names: nothing to warm up
    assert wait_until(ready(pipeline))
    time.sleep(0.05)
    assert len([c for c in open_detector.calls if c[0] == "warm_up"]) == warm_ups


# Fix 7: one Unicode form, and a letter or digit


def test_names_are_compared_in_one_unicode_form():
    composed, decomposed = "café mug", "café mug"
    assert pipeline_module.clean_names([decomposed, composed]) == (composed,)


@pytest.mark.parametrize("name", ["'", "- -", "''-", "’"])
def test_a_name_without_a_letter_or_digit_is_refused(name):
    with pytest.raises(ValueError, match="letter or digit"):
        pipeline_module.clean_names([name])


# Fix 8: stop() during a load


def test_a_search_loading_at_stop_is_never_marked_ready(make):
    gate = threading.Event()
    open_detector = FakeOpenDetector(load_gate=gate)
    pipeline = make(open_detector=open_detector).start()
    pipeline.search(["keys"])
    assert open_detector.inside.wait(2)
    pipeline.stop()
    gate.set()
    time.sleep(0.1)
    assert search_of(pipeline)["state"] == "off"
    assert pipeline._search.state == "off"
    assert ("set_names", ("keys",)) not in open_detector.calls


# ---- auto-discover (M14 part B) ------------------------------------------------------------------
#
# A fake lookout stands in for lookout.Lookout: no network, no key, no .env.


class FakeLookoutError(Exception):
    """Shaped like lookout.LookoutError: a plain message and a kind."""

    def __init__(self, message, kind="other"):
        super().__init__(message)
        self.kind = kind


class FakeLookout:
    """Gives the answers in turn (the last one again and again); an Exception in the list is raised."""

    def __init__(self, *answers, budget=0.5, gate=None):
        self.answers = list(answers) or [["router"]]
        self.budget = budget
        self.gate = gate
        self.pictures = []
        self.looks_today = 0
        self.spent_today = 0.0

    def names(self, jpeg):
        self.pictures.append(jpeg)
        if self.gate is not None:
            assert self.gate.wait(5)
        answer = self.answers[min(len(self.pictures), len(self.answers)) - 1]
        self.looks_today += 1
        self.spent_today += 0.000013
        if isinstance(answer, Exception):
            raise answer
        return list(answer)

    @property
    def looks(self):
        return len(self.pictures)


ROUTER = Detection("router", 0.6, (480, 40, 560, 90), 11)       # on the right, away from everyone
CURTAIN = Detection("curtain", 0.5, (560, 0, 639, 200), 12)


def discover_config(every=0.02, **changes):
    return {**CONFIG, "lookout": {**CONFIG["lookout"], "every_seconds": every, **changes}}


def discover_of(pipeline):
    return pipeline.status()["discover"]


def discovering(make, lookout, names=(), finds=(KEYS, ROUTER, CURTAIN), every=0.02, frames=10_000, **kwargs):
    """A running pipeline with auto-discover on (and the visitor's own `names`, if any)."""
    open_detector = FakeOpenDetector(finds=finds)
    pipeline = make(source=FakeSource(frames=frames), detector=FakeDetector(delay=0.005),
                    open_detector=open_detector, lookout=lookout,
                    config=kwargs.pop("config", None) or discover_config(every), **kwargs).start()
    if names:
        pipeline.search(list(names))
    assert wait_until(lambda: pipeline.latest_jpeg() is not None)
    pipeline.discover(True)
    return pipeline, open_detector


def set_names_calls(open_detector):
    return [call[1] for call in open_detector.calls if isinstance(call, tuple) and call[0] == "set_names"]


def no_more_detects(open_detector):
    """True when the open detector is not run any more (after a frame that was under way)."""
    time.sleep(0.05)
    detects = open_detector.count("detect")
    time.sleep(0.1)
    return open_detector.count("detect") == detects


def test_auto_discover_is_off_at_start_up_and_its_status_is_json_ready(make):
    lookout = FakeLookout()
    pipeline = make(source=FakeSource(frames=5), lookout=lookout, config=discover_config()).start()
    time.sleep(0.1)
    assert discover_of(pipeline) == {"on": False, "state": "off", "names": [], "note": "", "looks": 0,
                                     "spent_usd": 0.0, "budget_usd": 0.5}
    assert lookout.looks == 0
    json.dumps(pipeline.status())


def test_the_lookout_is_built_on_the_first_look_not_at_start_up(make, monkeypatch):
    built = []
    lookout_module = types.ModuleType("car_vision.lookout")
    lookout_module.Lookout = lambda settings: built.append(settings) or FakeLookout()
    monkeypatch.setitem(sys.modules, "car_vision.lookout", lookout_module)
    pipeline = make(source=FakeSource(frames=10_000), open_detector=FakeOpenDetector(),
                    config=discover_config()).start()
    time.sleep(0.05)
    assert built == []
    pipeline.discover(True)
    assert wait_until(lambda: built)
    assert built == [pipeline._lookout_settings]


def test_discover_takes_only_true_or_false(make):
    with pytest.raises(ValueError):
        make().discover("yes")


def test_looks_come_every_few_seconds_on_a_clean_shrunk_picture(make):
    lookout = FakeLookout(["router"])
    pipeline, _ = discovering(make, lookout, every=0.05)
    assert wait_until(lambda: lookout.looks >= 1)
    began = lookout.looks
    time.sleep(0.5)
    assert 4 <= lookout.looks - began <= 12                 # about every 0.05 s
    picture = decoded(lookout.pictures[0])
    assert picture.shape == (288, 512, 3)                   # 640 x 360 shrunk to 512 wide
    assert np.abs(picture - 60).max() <= 4                  # no boxes drawn on it
    assert discover_of(pipeline)["state"] == "looking"
    assert discover_of(pipeline)["looks"] == lookout.looks_today


def test_the_look_thread_is_not_the_video_loop(make):
    threads = []

    class WhereAsked(FakeLookout):
        def names(self, jpeg):
            threads.append(threading.current_thread().name)
            return super().names(jpeg)

    discovering(make, WhereAsked(["router"]))
    assert wait_until(lambda: threads)
    assert set(threads) == {"lookout"}


def test_discovered_names_reach_the_open_detector_and_its_finds_the_page(make):
    lookout = FakeLookout(["router", "curtain"])
    pipeline, open_detector = discovering(make, lookout)
    assert wait_until(lambda: any(o["label"] == "router" for o in pipeline.status()["objects"]))
    assert set_names_calls(open_detector)[0] == ("router", "curtain")
    router = next(o for o in pipeline.status()["objects"] if o["label"] == "router")
    assert router["discovered"] is True and "found" not in router
    assert router["track_id"] == OFFSET + 11
    assert discover_of(pipeline)["names"] == ["router", "curtain"]
    assert search_of(pipeline) == {"names": [], "state": "off", "note": "", "found": [], "missing": [],
                                   "seen": False}               # not the visitor's search
    assert pipeline.target(track_id=OFFSET + 11).label == "router"


def test_discovered_finds_are_not_drawn_in_the_search_colour(make, monkeypatch):
    wanted_per_frame = []
    real_draw = pipeline_module.draw.draw_detections

    def recording_draw(frame, detections, wanted=()):
        wanted_per_frame.append((sorted(wanted), sorted(d.label for d in detections)))
        real_draw(frame, detections, wanted)

    monkeypatch.setattr(pipeline_module.draw, "draw_detections", recording_draw)
    pipeline, _ = discovering(make, FakeLookout(["router"]), names=["keys"])
    assert wait_until(lambda: any("router" in labels for _w, labels in wanted_per_frame))
    wanted, labels = next(item for item in reversed(wanted_per_frame) if "router" in item[1])
    assert wanted == [OFFSET + 7]                           # only the visitor's keys stand out
    # A frame is drawn before it is published: wait until the page sees it too.
    assert wait_until(lambda: {"keys", "router"} <= {o["label"] for o in pipeline.status()["objects"]})
    objects = {o["label"]: o for o in pipeline.status()["objects"]}
    assert objects["keys"].get("found") is True and "discovered" not in objects["keys"]
    assert objects["router"].get("discovered") is True
    assert search_of(pipeline)["names"] == ["keys"]


def test_own_names_come_first_and_six_in_all(make):
    lookout = FakeLookout(["router", "curtain", "cushion", "lamp", "rug", "plant pot", "cable"])
    pipeline, open_detector = discovering(make, lookout, names=["keys", "wallet"])
    assert wait_until(lambda: any(len(names) == 6 for names in set_names_calls(open_detector)))
    assert set_names_calls(open_detector)[-1] == ("keys", "wallet", "router", "curtain", "cushion", "lamp")
    assert discover_of(pipeline)["names"][:4] == ["router", "curtain", "cushion", "lamp"]


def test_names_the_everyday_detector_knows_and_words_for_people_are_dropped(make):
    lookout = FakeLookout(["tv", "Television", "chairs", "monitor", "man", "hand", "router", "cell phone"])
    pipeline, open_detector = discovering(make, lookout)
    assert wait_until(lambda: discover_of(pipeline)["names"])
    assert discover_of(pipeline)["names"] == ["router"]
    assert wait_until(lambda: set_names_calls(open_detector))
    assert set_names_calls(open_detector) == [("router",)]


def test_set_names_only_when_the_list_really_changes(make):
    lookout = FakeLookout(["router", "curtain"], ["curtain", "router"], ["router", "curtain"])
    pipeline, open_detector = discovering(make, lookout)
    assert wait_until(lambda: lookout.looks >= 8)
    assert set_names_calls(open_detector) == [("router", "curtain")]   # the same two, in any order: once


def test_a_discovered_name_stays_a_while_after_it_was_last_mentioned(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "DISCOVERED_KEEP_SECONDS", 0.3)
    lookout = FakeLookout(["router"], [], [])
    pipeline, open_detector = discovering(make, lookout)
    assert wait_until(lambda: discover_of(pipeline)["names"] == ["router"])
    began = time.monotonic()
    assert wait_until(lambda: discover_of(pipeline)["names"] == [])
    assert time.monotonic() - began >= 0.2                  # kept, then dropped
    assert pipeline._search.finder_names == ()              # nothing left to look for
    assert no_more_detects(open_detector)
    assert not any(o["label"] == "router" for o in pipeline.status()["objects"])


def test_a_discovered_find_on_a_person_is_dropped(make):
    on_sara = Detection("scarf", 0.8, (120, 80, 240, 280), 13)
    pipeline, open_detector = discovering(make, FakeLookout(["scarf"]), finds=[on_sara])
    assert wait_until(lambda: open_detector.count("detect") >= 3)
    assert not any(o["label"] == "scarf" for o in pipeline.status()["objects"])
    assert pipeline.target(track_id=1).person == SARA


def test_switching_off_stops_the_looks_and_forgets_the_names(make):
    lookout = FakeLookout(["router"])
    pipeline, open_detector = discovering(make, lookout, names=["keys"])
    assert wait_until(lambda: any(o["label"] == "router" for o in pipeline.status()["objects"]))
    pipeline.discover(False)
    looks = lookout.looks
    assert discover_of(pipeline)["on"] is False and discover_of(pipeline)["names"] == []
    assert not any(o["label"] == "router" for o in pipeline.status()["objects"])   # at once
    assert wait_until(lambda: open_detector.names == ("keys",))
    time.sleep(0.15)
    assert lookout.looks <= looks + 1                       # at most the one already under way
    assert search_of(pipeline)["state"] == "ready"


def test_switching_on_twice_starts_one_thread(make):
    lookout = FakeLookout(["router"])
    pipeline, _ = discovering(make, lookout, every=0.1)
    pipeline.discover(True)
    time.sleep(0.45)
    assert lookout.looks <= 7                               # one thread: about 5, not 10


@pytest.mark.parametrize("kind", ["budget", "no_key", "key_refused", "no_credit", "not_available"])
def test_a_lasting_problem_pauses_until_switched_off_and_on(make, kind):
    message = f"A plain message about {kind}."
    lookout = FakeLookout(["router"], FakeLookoutError(message, kind), ["router"])
    pipeline, open_detector = discovering(make, lookout)
    assert wait_until(lambda: discover_of(pipeline)["state"] == "paused")
    assert discover_of(pipeline)["note"] == message and discover_of(pipeline)["on"] is True
    assert discover_of(pipeline)["names"] == []             # nothing more is looked for
    assert pipeline._search.finder_names == ()
    assert no_more_detects(open_detector)
    looks = lookout.looks
    time.sleep(0.15)
    assert lookout.looks == looks                           # no more looks ...
    pipeline.discover(False)
    pipeline.discover(True)                                 # ... until switched off and on
    assert wait_until(lambda: lookout.looks > looks)
    assert wait_until(lambda: discover_of(pipeline)["state"] == "looking")


def test_three_passing_failures_in_a_row_wait_a_while(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "FAILURE_WAIT_SECONDS", 0.4)
    busy = FakeLookoutError("Hugging Face is busy right now. Try again later.", "busy")
    lookout = FakeLookout(busy, busy, busy, ["router"])
    pipeline, _ = discovering(make, lookout)
    assert wait_until(lambda: lookout.looks == 3)
    assert discover_of(pipeline)["state"] == "paused"
    assert discover_of(pipeline)["note"] == "Hugging Face is busy right now. Try again later."
    began = time.monotonic()
    assert wait_until(lambda: lookout.looks == 4, timeout=2)
    assert time.monotonic() - began >= 0.3                  # waited, not every 0.02 s
    assert wait_until(lambda: discover_of(pipeline)["state"] == "looking")
    assert discover_of(pipeline)["names"] == ["router"]


def test_one_passing_failure_pauses_with_its_note_and_the_next_look_comes_as_usual(make):
    lookout = FakeLookout(FakeLookoutError("Could not reach Hugging Face.", "network"), ["router"])
    pipeline, _ = discovering(make, lookout)
    assert wait_until(lambda: lookout.looks >= 2, timeout=1)
    assert wait_until(lambda: discover_of(pipeline)["state"] == "looking")


def test_a_look_still_under_way_is_skipped_without_counting(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "FAILURE_WAIT_SECONDS", 5.0)
    still = FakeLookoutError("The previous look has not finished yet.", "still_looking")
    lookout = FakeLookout(still, still, still, still, ["router"])
    pipeline, _ = discovering(make, lookout)
    assert wait_until(lambda: lookout.looks >= 5, timeout=1)  # no 5 s wait after three
    assert discover_of(pipeline)["state"] == "looking" and discover_of(pipeline)["note"] == ""


def test_an_unexpected_error_pauses_with_a_plain_note(make):
    lookout = FakeLookout(RuntimeError("secret detail from deep inside"))
    pipeline, _ = discovering(make, lookout)
    assert wait_until(lambda: discover_of(pipeline)["state"] == "paused")
    assert discover_of(pipeline)["note"] == pipeline_module.LOOK_FAILED


def test_no_look_after_stop(make):
    lookout = FakeLookout(["router"])
    pipeline, _ = discovering(make, lookout)
    assert wait_until(lambda: lookout.looks >= 2)
    pipeline.stop()
    looks = lookout.looks
    time.sleep(0.15)
    assert lookout.looks == looks
    assert discover_of(pipeline)["on"] is False


def test_stop_does_not_wait_for_a_look_under_way(make):
    gate = threading.Event()
    lookout = FakeLookout(["router"], gate=gate)
    pipeline, _ = discovering(make, lookout)
    try:
        assert wait_until(lambda: lookout.looks == 1)       # the request is under way ...
        began = time.monotonic()
        pipeline.stop()
        assert time.monotonic() - began < 1.0               # ... and stop() does not wait for it
    finally:
        gate.set()
    time.sleep(0.05)
    assert discover_of(pipeline)["names"] == []             # its answer came too late to count


def test_an_answer_about_a_replaced_camera_is_dropped(make):
    gate = threading.Event()
    lookout = FakeLookout(["router"], gate=gate)
    pipeline, open_detector = discovering(make, lookout,
                                          source_factory=SourceFactory(phone_source(frames=0, connected=False)))
    assert wait_until(lambda: lookout.looks == 1)
    pipeline.use_phone(PHONE)
    gate.set()
    time.sleep(0.1)
    assert discover_of(pipeline)["names"] == []
    assert set_names_calls(open_detector) == []


def test_the_finder_failing_for_discovered_names_pauses_discover(make):
    open_detector = FakeOpenDetector(load_error=RuntimeError("Could not load the named search model."))
    pipeline = make(source=FakeSource(frames=10_000), open_detector=open_detector,
                    lookout=FakeLookout(["router"]), config=discover_config()).start()
    pipeline.discover(True)
    assert wait_until(lambda: discover_of(pipeline)["state"] == "paused")
    assert discover_of(pipeline)["note"] == "Could not load the named search model."
    assert search_of(pipeline)["state"] == "off"            # the visitor searched for nothing


def test_with_the_real_lookout_and_a_fake_network(make, monkeypatch, tmp_path):
    lookout_module = pytest.importorskip("car_vision.lookout")
    monkeypatch.setattr(lookout_module, "read_key", lambda: "hf_" + "x" * 20)   # never the real .env
    sent = []

    def fake_fetch(url, headers, body, timeout):
        sent.append(url)
        answer = {"choices": [{"message": {"content": "TV, Router, a man, Curtains"}}],
                  "usage": {"prompt_tokens": 312, "completion_tokens": 17}}
        return 200, json.dumps(answer)

    settings = {**CONFIG["lookout"], "spend_file": str(tmp_path / "lookout.json")}
    lookout = lookout_module.Lookout(settings, fetch=fake_fetch)
    pipeline, open_detector = discovering(make, lookout, every=0.1)
    assert wait_until(lambda: discover_of(pipeline)["names"])
    assert discover_of(pipeline)["names"] == ["router", "curtains"]
    assert discover_of(pipeline)["looks"] >= 1 and discover_of(pipeline)["spent_usd"] > 0
    assert wait_until(lambda: set_names_calls(open_detector))
    assert set_names_calls(open_detector)[0] == ("router", "curtains")
    assert all(url.startswith("https://router.huggingface.co/") for url in sent)


@pytest.mark.parametrize("names, own, expected", [
    (["a", "b"], (), ("a", "b")),
    (["Keys", "b"], ("keys",), ("b",)),                     # not twice
    (["a", "b", "c", "d", "e", "f", "g"], ("x", "y"), ("a", "b", "c", "d")),
])
def test_names_after(names, own, expected):
    assert pipeline_module.names_after(own, names) == expected


def test_in_earlier_order_keeps_the_names_already_given_in_place():
    assert pipeline_module.in_earlier_order(("b", "a", "c"), ("a", "b")) == ("a", "b", "c")
    assert pipeline_module.in_earlier_order(("c",), ("a", "b")) == ("c",)


@pytest.mark.parametrize("name, boxed", [("tv", True), ("Televisions", True), ("couches", True),
                                         ("sofa", True), ("router", False), ("curtain", False),
                                         ("power bank", False)])
def test_boxed_every_day(name, boxed):
    assert pipeline_module.boxed_every_day(name) is boxed


def test_every_lookout_setting_the_pipeline_reads_is_checked():
    for key, value in [("every_seconds", 0), ("picture_width", 8), ("every_seconds", "10")]:
        with pytest.raises(ConfigError, match=key):
            Pipeline(discover_config(**{key: value}), source=FakeSource(), detector=FakeDetector(),
                     recognizer=FakeRecognizer())
