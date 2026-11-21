"""The see-decide-show loop, run with a fake camera, detector and recogniser.

No camera, model, network or graphics card is needed.
"""

import ast
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from car_vision import pipeline as pipeline_module
from car_vision.config import ConfigError
from car_vision.faces import Person
from car_vision.pipeline import NOBODY_ENROLLED, Pipeline
from car_vision.reactions import WARNING_TEXT, ReactionEngine
from car_vision.state import Detection, RecognizedFace, WorldState

PIPELINE_FILE = Path(__file__).resolve().parent.parent / "car_vision" / "pipeline.py"

CONFIG = {
    "stream": {"source": "file", "phone_url": "http://192.168.0.2:8080/video", "webcam_index": 0,
               "file_path": "data/does-not-exist/clip.mp4", "width": 640, "height": 480,
               "reconnect_seconds": 0.01, "loop_file": True},
    "detector": {"model": "fake.pt", "models_dir": "models", "confidence": 0.4, "image_size": 640,
                 "device": "cpu", "tracker": "bytetrack.yaml"},
    "recognizer": {"model_pack": "buffalo_l", "models_dir": "models", "detection_size": 640,
                   "device": "cpu", "match_threshold": 0.42, "every_n_frames": 3, "min_face_pixels": 40,
                   "people_file": "data/does-not-exist/people.json",
                   "photos_dir": "data/does-not-exist/people",
                   "embeddings_file": "data/does-not-exist/embeddings.npz"},
    "reactions": {"greet_cooldown_seconds": 60, "close_box_height_ratio": 0.75,
                  "close_hold_seconds": 0.0, "message_seconds": 4.0},
    "app": {"host": "127.0.0.1", "port": 5000, "jpeg_quality": 80, "log_file": "logs/test.log"},
}

STATUS_KEYS = {"source", "connected", "fps", "device", "recognition", "notes",
               "people", "objects", "reactions"}

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
        self.frames = frames
        self.sequence = 0
        self.started = False
        self.stopped = False
        width, height = size
        self.picture = np.full((height, width, 3), 60, dtype=np.uint8) if picture is None else picture
        self._lock = threading.Lock()

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

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

    def person(self, person_id):
        return self.people.get(person_id)


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

    def build(source=None, detector=None, recognizer=None, config=CONFIG, **kwargs):
        pipeline = Pipeline(config,
                            source=source if source is not None else FakeSource(),
                            detector=detector if detector is not None else FakeDetector(),
                            recognizer=recognizer if recognizer is not None else FakeRecognizer(),
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


def test_close_warning_reaches_the_status(make):
    tall = Detection("person", 0.9, (200, 10, 400, 350), 1)          # 340 / 360 of the frame
    pipeline = make(detector=FakeDetector(detections=[tall]), recognizer=FakeRecognizer(faces=[]),
                    reactions=ReactionEngine(CONFIG["reactions"])).start()
    assert wait_until(lambda: pipeline.status()["reactions"])
    assert pipeline.status()["reactions"] == [{"kind": "warning", "text": WARNING_TEXT}]


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


def test_status_leaves_out_old_sightings_while_the_stream_is_down(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "STALE_SECONDS", 0.2)
    source = FakeSource(frames=3)
    pipeline = make(source=source).start()
    assert wait_until(lambda: pipeline.status()["objects"])
    source.connected = False
    assert wait_until(lambda: not pipeline.status()["objects"])
    status = pipeline.status()
    assert status["people"] == [] and status["reactions"] == []


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
