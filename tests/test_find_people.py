"""M20 Parts A and C: a Find that works around people, and a known person's name in Look for.

The promise these tests prove: a person is never treated as an object. Nothing that is a
person or on a person gets a picture in the memory, and a click on it never sends a
cut-out to the assistant. People are only ever recognised by their enrolled face.

Everything is a fake: no camera, model, network or graphics card is needed.
"""

import types

import cv2
import numpy as np
import pytest

from car_vision import pipeline as pipeline_module
from car_vision.app import create_app
from car_vision.enroll import Enrolment
from car_vision.faces import Person
from car_vision.pipeline import NamedFinds, Pipeline, merge_named, on_person, same_as_person, share_inside
from car_vision.recognizer import FaceCheck
from car_vision.state import Detection, RecognizedFace
from test_pipeline import (CONFIG, OFFSET, PERSON, SARA, SARA_FACE, FakeDetector, FakeLookout, FakeOpenDetector,
                           FakeRecognizer, FakeSightings, FakeSource, NoPhone, RecordingState, discover_config,
                           memory_config, search_of, wait_until)
from test_pipeline import make  # noqa: F401  (the fixture, used by name)

def frames_later(pipeline, count=2):
    """Wait until `count` more frames have started: the frames before them are then done (no sleeping)."""
    start = pipeline._detector.calls
    assert wait_until(lambda: pipeline._detector.calls >= start + count)


def every_frame(pipeline, monkeypatch, do):
    """Call do() once on every frame, right after the frame was published and told to the memory."""
    real = pipeline._remember

    def remember(*args, **kwargs):
        real(*args, **kwargs)
        do()

    monkeypatch.setattr(pipeline, "_remember", remember)


# ---- Part A: the boxes of the table in the spec (the owner in front of a window, 640 x 360) ----

OWNER = (200, 60, 440, 360)                         # the person box
WINDOW = Detection("window", 0.26, (20, 0, 620, 340), 3)       # most of the picture, the person inside it
GLASSES = Detection("glasses", 0.75, (280, 100, 360, 130), 4)  # on the face
WALL = Detection("wall", 0.35, (0, 0, 640, 360), 5)            # the whole picture
CURTAIN = Detection("curtain", 0.70, (460, 0, 600, 300), 6)    # beside the person
THE_OWNER = Detection("human shape", 0.8, (205, 65, 435, 355), 7)   # a box that is the person


@pytest.mark.parametrize("find, dropped, on_her", [
    (WINDOW, False, False),             # a window behind a person: an ordinary find
    (GLASSES, False, True),             # glasses on a face: kept, and marked
    (WALL, False, False),
    (CURTAIN, False, False),
    (THE_OWNER, True, True),            # the person: dropped
])
def test_the_three_cases_with_the_boxes_of_the_table(find, dropped, on_her):
    assert same_as_person(find, [OWNER]) is dropped
    assert on_person(find, [OWNER]) is on_her


def test_share_inside_on_touching_nested_and_apart_boxes():
    assert share_inside((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0        # touching: nothing inside
    assert share_inside((2, 2, 4, 4), (0, 0, 10, 10)) == 1.0           # nested: all of it
    assert share_inside((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(0.04)   # the other way round
    assert share_inside((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0       # apart
    assert share_inside((0, 0, 10, 10), (5, 0, 20, 10)) == pytest.approx(0.5)
    assert share_inside((3, 3, 3, 9), (0, 0, 10, 10)) == 0.0           # an empty box has nothing inside


def test_merge_keeps_the_window_and_the_glasses_and_drops_the_box_that_is_the_person():
    everyday = [Detection("person", 0.9, OWNER, 1)]
    shown, kept = merge_named(everyday, NamedFinds(1), [WINDOW, GLASSES, THE_OWNER, CURTAIN], 1, 10.0)
    assert [d.label for d in shown] == ["person", "window", "glasses", "curtain"]
    assert [d.track_id for d in shown[1:]] == [OFFSET + 3, OFFSET + 4, OFFSET + 6]
    assert [find.label for find, _seen in kept.finds] == ["window", "glasses", "curtain"]


# ---- Part A in the running pipeline: glasses on Sara ---------------------------------------

ON_SARA = Detection("glasses", 0.75, (150, 90, 210, 110), 21)          # inside PERSON (100, 60, 260, 300)
BEHIND_SARA = Detection("window", 0.6, (0, 0, 640, 340), 22)           # she is in front of it


def glasses_on_sara(make, **kwargs):
    sightings = FakeSightings()
    open_detector = FakeOpenDetector(finds=[ON_SARA, BEHIND_SARA])
    pipeline = make(source=FakeSource(frames=10_000), open_detector=open_detector, sightings=sightings,
                    config=memory_config(), state=RecordingState(), **kwargs)
    pipeline.search(["glasses", "window"])
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    pipeline.start()
    assert wait_until(lambda: {"glasses", "window"} <= {o["label"] for o in pipeline.status()["objects"]})
    return pipeline, sightings


def test_a_find_on_a_person_is_shown_listed_counted_and_marked(make):
    pipeline, _sightings = glasses_on_sara(make)
    objects = {o["label"]: o for o in pipeline.status()["objects"]}
    assert objects["glasses"]["found"] is True and objects["glasses"]["on_person"] is True
    assert "on_person" not in objects["window"] and objects["window"]["found"] is True
    assert "on_person" not in objects["person"]
    found = {entry["name"]: entry for entry in search_of(pipeline)["found"]}
    assert set(found) == {"glasses", "window"} and search_of(pipeline)["missing"] == []


def test_a_click_on_a_find_on_a_person_is_a_click_on_that_person(make):
    pipeline, _sightings = glasses_on_sara(make)
    for target in (pipeline.target(track_id=OFFSET + 21, label="glasses"),   # its pill
                   pipeline.target(x=180 / 640, y=100 / 360)):              # the video
        assert (target.kind, target.label, target.track_id, target.person) == ("person", "person", 1, SARA)
    window = pipeline.target(track_id=OFFSET + 22, label="window")
    assert (window.kind, window.label) == ("object", "window")              # behind her: an ordinary thing


def test_the_memory_is_never_told_about_a_find_on_a_person(make):
    pipeline, sightings = glasses_on_sara(make)
    assert wait_until(lambda: any(call["label"] == "window" for call in sightings.seen_calls))
    frames_later(pipeline)
    assert not any(call["label"] == "glasses" for call in sightings.seen_calls)
    people = [call for call in sightings.seen_calls if call["kind"] == "person"]
    assert people and all(call["picture"] is None for call in people)


class RecordingAssistant:
    """Fails the test the moment anything reaches the assistant."""

    def __getattr__(self, name):
        raise AssertionError(f"a click on a person reached assistant.{name}")


def test_no_photo_of_a_find_on_a_person_is_ever_sent(make):
    pipeline, _sightings = glasses_on_sara(make)
    app = create_app(pipeline, RecordingAssistant())
    app.testing = True
    client = app.test_client()
    reply = client.post("/api/inspect", json={"track_id": OFFSET + 21, "label": "glasses"})
    assert reply.status_code == 200
    body = reply.get_json()
    assert (body["kind"], body["describable"], body["person"]["name"]) == ("person", False, "Sara Ahmed")
    described = client.post("/api/describe", json={"id": body["id"]})
    assert described.status_code == 400                                   # people are not looked up


def test_a_find_that_is_the_person_is_dropped_and_the_person_keeps_their_box(make):
    is_sara = Detection("silhouette", 0.9, (104, 64, 256, 296), 23)
    open_detector = FakeOpenDetector(finds=[is_sara])
    pipeline = make(source=FakeSource(frames=10_000), open_detector=open_detector)
    pipeline.search(["silhouette"])
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    pipeline.start()
    assert wait_until(lambda: open_detector.count("detect") >= 2 and search_of(pipeline)["seen"])
    assert not any(o["label"] == "silhouette" for o in pipeline.status()["objects"])
    assert search_of(pipeline)["missing"] == ["silhouette"]
    assert pipeline.target(track_id=1).kind == "person"


# ---- Part C: a known person's name in Look for -----------------------------------------------

MOHAMED = Person("mo_adel", "Mohamed Adel", "Owner")
SARA_KHAN = Person("sara_khan", "Sara Khan", "Driver")
SECOND_PERSON = Detection("person", 0.85, (400, 60, 560, 300), 5)


class PeopleRecognizer(FakeRecognizer):
    """A recogniser that knows `people`; a face is named only while its person is enrolled.

    `faces` are (box, person id, track number). reload_known() changes who is enrolled, as
    enrolling again from the page does.
    """

    def __init__(self, faces, people, enrolled=None):
        super().__init__(faces=(), people=dict(people))
        self.seen = list(faces)
        self.enrolled = set(people) if enrolled is None else set(enrolled)
        self.face_model = None

    def recognize(self, frame, detections=()):
        self.calls += 1
        return [RecognizedFace(self.people[person_id].name if person_id in self.enrolled else "Unknown", 0.6, box,
                               person_id if person_id in self.enrolled else None, track)
                for box, person_id, track in self.seen]

    def check_faces(self, frame, detections=()):
        faces = self.recognize(frame, detections)
        return FaceCheck(faces, [face.box for face in faces])

    def enrolled_ids(self):
        return set(self.enrolled)

    def reload_known(self, people, matcher):
        self.people, self.enrolled = dict(people), set(matcher.person_ids)


MOHAMED_IN_VIEW = [(SARA_FACE.box, "mo_adel", 1)]          # his face in the PERSON box (track 1)


def people_pipeline(make, faces=MOHAMED_IN_VIEW, people=None, enrolled=None, detections=(PERSON,), **kwargs):
    recognizer = PeopleRecognizer(faces, people or {"mo_adel": MOHAMED}, enrolled)
    open_detector = kwargs.pop("open_detector", None) or FakeOpenDetector()
    pipeline = make(source=FakeSource(frames=10_000), detector=FakeDetector(detections=list(detections)),
                    recognizer=recognizer, open_detector=open_detector, **kwargs)
    return pipeline, recognizer, open_detector


@pytest.mark.parametrize("typed", ["Mohamed Adel", "mo adel", "mohamed", "MOHAMED adel", "mOhAmEd"])
def test_a_persons_name_folder_name_or_first_name_finds_them_by_their_face(make, typed):
    pipeline, _recognizer, open_detector = people_pipeline(make)
    assert pipeline.search([typed]) == ("Mohamed Adel",)                 # as the page will show it
    assert search_of(pipeline)["state"] == "ready"                        # at once: no open detector needed
    pipeline.start()
    assert wait_until(lambda: search_of(pipeline)["found"])
    search = search_of(pipeline)
    assert search["names"] == ["Mohamed Adel"] and search["people"] == ["Mohamed Adel"]
    assert search["found"] == [{"name": "Mohamed Adel", "count": 1, "place": "left", "person": True, "sure": True}]
    assert search["missing"] == [] and search["seen"] is True
    assert open_detector.calls == [] and pipeline._open_detector is open_detector   # never loaded, never asked


def test_a_person_not_in_view_is_missing_by_their_enrolled_name(make):
    pipeline, _recognizer, _open = people_pipeline(make, faces=[])
    pipeline.search(["mohamed"])
    pipeline.start()
    assert wait_until(lambda: search_of(pipeline)["seen"])
    assert search_of(pipeline)["missing"] == ["Mohamed Adel"] and search_of(pipeline)["found"] == []


def test_two_people_with_one_first_name_are_both_meant(make):
    people = {"sara": SARA, "sara_khan": SARA_KHAN}
    faces = [(SARA_FACE.box, "sara", 1), ((450, 70, 500, 130), "sara_khan", 5)]
    pipeline, recognizer, _open = people_pipeline(make, faces=faces, people=people,
                                                  detections=(PERSON, SECOND_PERSON))
    assert pipeline.search(["sara"]) == ("Sara",)
    pipeline.start()
    assert wait_until(lambda: search_of(pipeline)["found"] and search_of(pipeline)["found"][0]["count"] == 2)
    assert search_of(pipeline)["found"][0]["name"] == "Sara"
    assert {o["track_id"] for o in pipeline.status()["objects"] if o.get("found")} == {1, 5}
    recognizer.seen = faces[1:]                                           # Sara Ahmed leaves
    assert wait_until(lambda: search_of(pipeline)["found"] and search_of(pipeline)["found"][0]["count"] == 1)
    assert search_of(pipeline)["found"][0]["place"] == "right"


def test_somebody_who_is_not_enrolled_is_an_ordinary_name(make):
    # Mohamed has a folder but no saved faces: his name goes to the open detector like any other.
    pipeline, _recognizer, open_detector = people_pipeline(make, enrolled=())
    assert pipeline.search(["mohamed", "Lina"]) == ("mohamed", "Lina")
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    assert ("set_names", ("mohamed", "Lina")) in open_detector.calls
    assert "people" not in search_of(pipeline)


def test_with_recognition_off_every_name_is_an_ordinary_name(monkeypatch):
    monkeypatch.setattr(Pipeline, "_build_recognizer", lambda self, settings: None)
    open_detector = FakeOpenDetector()
    pipeline = Pipeline(CONFIG, source=FakeSource(), detector=FakeDetector(), open_detector=open_detector,
                        phone_fetch=NoPhone())
    try:
        assert pipeline.search(["Sara Ahmed"]) == ("Sara Ahmed",)
        assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
        assert ("set_names", ("Sara Ahmed",)) in open_detector.calls
    finally:
        pipeline.stop()


def test_the_open_detector_is_never_given_a_persons_name(make):
    pipeline, _recognizer, open_detector = people_pipeline(make)
    assert pipeline.search(["keys", "Mohamed"]) == ("keys", "Mohamed Adel")
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    pipeline.start()
    assert wait_until(lambda: len(search_of(pipeline)["found"]) == 2)
    calls = [call for call in open_detector.calls if isinstance(call, tuple) and call[0] == "set_names"]
    assert calls == [("set_names", ("keys",))]
    assert [entry["name"] for entry in search_of(pipeline)["found"]] == ["keys", "Mohamed Adel"]


def test_a_search_of_people_only_never_loads_the_open_detector(make, monkeypatch):
    built = []
    monkeypatch.setattr("car_vision.detector.OpenDetector", lambda settings: built.append(settings))
    recognizer = PeopleRecognizer(MOHAMED_IN_VIEW, {"mo_adel": MOHAMED})
    pipeline = make(source=FakeSource(frames=10_000), recognizer=recognizer)
    pipeline.search(["mohamed"])
    assert search_of(pipeline)["state"] == "ready"
    pipeline.start()
    assert wait_until(lambda: search_of(pipeline)["found"])
    frames_later(pipeline, 4)
    assert built == [] and pipeline._open_detector is None


def test_people_words_are_still_refused():
    for name in ("person", "man", "face"):
        with pytest.raises(ValueError, match="not supported"):
            pipeline_module.clean_names([name])


def enrolling(monkeypatch, people):
    """Enrolling again gives these people (person id -> Person), without reading a photo."""
    ids = list(people)
    enrolment = Enrolment(dict(people), {}, {person_id: 1 for person_id in ids}, ids,
                          np.ones((len(ids), 4), dtype=np.float32))
    monkeypatch.setattr(pipeline_module, "encode_people", lambda photos_dir, get_model: enrolment)
    monkeypatch.setattr(pipeline_module, "save_enrolment", lambda enrolment, photos_dir, path: [])


def test_people_added_or_removed_while_searching_count_from_the_next_frame(make, monkeypatch):
    pipeline, _recognizer, open_detector = people_pipeline(make, enrolled=())
    pipeline.search(["mohamed"])
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    pipeline.start()
    assert wait_until(lambda: search_of(pipeline)["missing"] == ["mohamed"])     # an ordinary name for now
    assert open_detector.names == ("mohamed",)

    enrolling(monkeypatch, {"mo_adel": MOHAMED})                                 # he is added from the page
    pipeline.reload_people()
    assert search_of(pipeline)["names"] == ["mohamed"]          # the text stays as first answered ...
    assert wait_until(lambda: search_of(pipeline)["found"])     # ... but his face answers it now
    assert search_of(pipeline)["found"][0] == {"name": "mohamed", "count": 1, "place": "left", "person": True,
                                               "sure": True}
    assert search_of(pipeline)["people"] == ["mohamed"]
    frames_later(pipeline)                                      # a frame under way may still have asked
    detects = open_detector.count("detect")
    frames_later(pipeline, 6)                                   # the finder would run on every 3rd frame
    assert open_detector.count("detect") == detects                              # no longer asked about him

    enrolling(monkeypatch, {})                                                   # ... and removed again
    pipeline.reload_people()
    assert search_of(pipeline)["names"] == ["mohamed"]
    assert wait_until(lambda: open_detector.count("detect") > detects)          # an ordinary name again
    assert wait_until(lambda: search_of(pipeline)["missing"] == ["mohamed"])


def test_a_persons_name_keeps_its_text_when_they_are_removed(make, monkeypatch):
    pipeline, _recognizer, _open = people_pipeline(make)
    assert pipeline.search(["mohamed"]) == ("Mohamed Adel",)
    pipeline.start()
    assert wait_until(lambda: search_of(pipeline)["found"])
    enrolling(monkeypatch, {})                                                   # removed while searched for
    pipeline.reload_people()
    assert search_of(pipeline)["names"] == ["Mohamed Adel"]                      # the page's search goes on
    assert wait_until(lambda: search_of(pipeline)["missing"] == ["Mohamed Adel"])
    assert "people" not in search_of(pipeline)


def test_a_curly_apostrophe_is_the_same_as_a_straight_one():
    enrolled = {"sean_obrien": "Sean O’Brien"}
    assert pipeline_module.people_among(["sean o'brien"], enrolled)[0].shown == "Sean O’Brien"
    assert pipeline_module.people_among(["Sean O’Brien"], {"x": "Sean O'Brien"})[0].person_ids == ("x",)


def test_the_found_persons_box_is_drawn_in_the_wanted_colour(make, monkeypatch):
    drawn = []
    real_draw = pipeline_module.draw.draw_detections
    faces_drawn = []
    real_faces = pipeline_module.draw.draw_faces

    def recording_draw(frame, detections, wanted=(), unsure=()):
        drawn.append(sorted(wanted))
        real_draw(frame, detections, wanted, unsure)

    def recording_faces(frame, faces):
        faces_drawn.append([(face.name, face.box) for face in faces])
        real_faces(frame, faces)

    monkeypatch.setattr(pipeline_module.draw, "draw_detections", recording_draw)
    monkeypatch.setattr(pipeline_module.draw, "draw_faces", recording_faces)
    pipeline, _recognizer, _open = people_pipeline(make)
    pipeline.search(["mohamed"])
    pipeline.start()
    assert wait_until(lambda: [1] in drawn)                       # his person box (track 1) stands out
    # His face box and name stay as they are.
    assert wait_until(lambda: any(("Mohamed Adel", SARA_FACE.box) in faces for faces in faces_drawn))
    # A frame is drawn before it is published: wait until the page sees it too.
    assert wait_until(lambda: any(o.get("found") for o in pipeline.status()["objects"]))
    objects = pipeline.status()["objects"]
    assert objects[0]["track_id"] == 1 and objects[0]["found"] is True
    assert pipeline.target(track_id=1).kind == "person"           # a click on him is a click on a person
    pipeline.search([])
    drawn.clear()
    assert wait_until(lambda: len(drawn) >= 2)
    assert drawn[-1] == []                                         # the search stopped: no highlight


def test_people_among_and_names_person_on_their_own():
    enrolled = {"mo_adel": "Mohamed Adel", "sara": "Sara Ahmed", "sara_khan": "Sara Khan"}
    people = pipeline_module.people_among(["keys", "mohamed  ADEL", "SARA", "sara khan"], enrolled)
    assert [(p.typed, p.shown, p.person_ids) for p in people] == [
        ("mohamed  ADEL", "Mohamed Adel", ("mo_adel",)),          # extra spaces and case do not matter
        ("SARA", "Sara", ("sara", "sara_khan")),                  # a shared first name means both
        ("sara khan", "Sara Khan", ("sara_khan",))]
    assert pipeline_module.people_among(["mohamed"], enrolled)[0].shown == "Mohamed Adel"
    assert pipeline_module.people_among(["adel"], enrolled) == ()          # not a first name: an ordinary name
    assert pipeline_module.people_among(["mohamed"], {}) == ()


def flask_client(pipeline, assistant=None):
    app = create_app(pipeline, assistant if assistant is not None else RecordingAssistant())
    app.testing = True
    return app.test_client()


def started_search(pipeline, names, label):
    pipeline.search(list(names))
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    pipeline.start()
    assert wait_until(lambda: any(o["label"] == label for o in pipeline.status()["objects"]))
    return pipeline


def told(sightings, label):
    return [call for call in sightings.seen_calls if call["label"] == label]


# ---- fix round A: a face is a person too, also without a person box -----------------------------

CLOSE_FACE = RecognizedFace("Unknown", 0.1, (250, 60, 370, 200), None, None)   # leaning close: no body
ON_THE_FACE = Detection("glasses", 0.75, (270, 100, 350, 130), 24)


def face_only(make):
    sightings = FakeSightings()
    pipeline = make(source=FakeSource(frames=10_000), detector=FakeDetector(detections=[]),
                    recognizer=FakeRecognizer(faces=[CLOSE_FACE], people={}),
                    open_detector=FakeOpenDetector(finds=[ON_THE_FACE]), sightings=sightings,
                    config=memory_config(), state=RecordingState())
    return started_search(pipeline, ["glasses"], "glasses"), sightings


def test_glasses_on_a_face_without_a_body_are_never_an_object(make):
    pipeline, sightings = face_only(make)
    glasses = next(o for o in pipeline.status()["objects"] if o["label"] == "glasses")
    assert glasses["on_person"] is True
    for target in (pipeline.target(track_id=OFFSET + 24, label="glasses"),   # its pill
                   pipeline.target(x=310 / 640, y=115 / 360)):              # the video
        assert (target.kind, target.label, target.person) == ("person", "face", None)   # as a click on the face
    frames_later(pipeline, 4)
    assert told(sightings, "glasses") == []


def test_nothing_of_a_face_without_a_body_reaches_the_assistant(make):
    pipeline, _sightings = face_only(make)
    client = flask_client(pipeline)
    for click in ({"track_id": OFFSET + 24, "label": "glasses"}, {"x": 310 / 640, "y": 115 / 360}):
        body = client.post("/api/inspect", json=click).get_json()
        assert (body["kind"], body["describable"], body["person"]) == ("person", False, None)
        assert client.post("/api/describe", json={"id": body["id"]}).status_code == 400
        asked = client.post("/api/follow_up", json={"id": body["id"], "question": "What is it?"})
        assert asked.status_code == 400                      # an unknown person: nothing is asked


# ---- fix round B: on a person, or the person, sticks to the track -------------------------------


class BlinkingDetector(FakeDetector):
    """Misses every person on every 4th frame, as a real detector now and then does."""

    def detect(self, frame):
        found = super().detect(frame)
        return found if self.calls % 4 else [d for d in found if d.label != "person"]


def blinking(make, monkeypatch, finds, names, record, frames=200):
    """Run `frames` frames with a person box that drops out on every 4th; record(pipeline) on each frame."""
    sightings = FakeSightings()
    pipeline = make(source=FakeSource(frames=frames), detector=BlinkingDetector(detections=[PERSON]),
                    recognizer=FakeRecognizer(faces=[]), open_detector=FakeOpenDetector(finds=list(finds)),
                    sightings=sightings, config=memory_config(), state=RecordingState())
    pipeline.search(list(names))
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    seen = []
    every_frame(pipeline, monkeypatch, lambda: seen.append(record(pipeline)))
    pipeline.start()
    assert wait_until(lambda: len(seen) == frames, timeout=30)
    return seen, sightings


def test_a_person_who_drops_out_every_fourth_frame_never_turns_a_find_into_an_object(make, monkeypatch):
    def clicks(pipeline):
        by_track = pipeline.target(track_id=OFFSET + 21, label="glasses")
        by_point = pipeline.target(x=180 / 640, y=100 / 360)
        return [target.kind for target in (by_track, by_point) if target is not None]

    kinds, sightings = blinking(make, monkeypatch, [ON_SARA], ["glasses"], clicks)
    every_kind = [kind for frame in kinds for kind in frame]
    assert "object" not in every_kind and every_kind.count("person") >= 380
    assert told(sightings, "glasses") == []
    assert all(call["picture"] is None for call in sightings.seen_calls)


def test_a_find_that_is_the_person_stays_dropped_on_a_miss_frame(make, monkeypatch):
    is_sara = Detection("silhouette", 0.9, (104, 64, 256, 296), 23)
    labels, sightings = blinking(make, monkeypatch, [is_sara], ["silhouette"],
                                 lambda pipeline: [o["label"] for o in pipeline.status()["objects"]])
    assert not any("silhouette" in frame for frame in labels)
    assert told(sightings, "silhouette") == []


def test_the_guard_keeps_a_track_on_a_person_until_it_ends_and_the_person_is_long_gone():
    guard = pipeline_module.PersonGuard()
    glasses = Detection("glasses", 0.7, (150, 90, 210, 110), OFFSET + 1)
    silhouette = Detection("silhouette", 0.9, (104, 64, 256, 296), OFFSET + 2)
    guard.see([PERSON.box], 10.0)
    assert guard.on_a_person(glasses, 10.0) and guard.is_the_person(silhouette, 10.0)
    guard.see([], 12.0)                                  # no person box for 2 seconds
    assert guard.on_a_person(glasses, 12.0) and guard.is_the_person(silhouette, 12.0)   # their tracks go on
    guard.forget_ended([silhouette], 12.0)               # the glasses' track ended
    assert not guard.on_a_person(glasses, 12.0) and guard.is_the_person(silhouette, 12.0)


def test_an_untracked_find_is_on_a_person_when_it_touches_one_seen_in_the_last_second():
    guard = pipeline_module.PersonGuard()
    guard.see([PERSON.box], 10.0)
    near = Detection("window", 0.6, (0, 0, 120, 100), None)      # touches her box, mostly outside it
    apart = Detection("curtain", 0.6, (400, 0, 600, 100), None)
    guard.see([], 10.5)                                          # she is missed on this frame
    assert guard.on_a_person(near, 10.5) and not guard.on_a_person(apart, 10.5)
    guard.see([], 11.6)
    assert not guard.on_a_person(near, 11.6)                     # her box is older than NAMED_HOLD_SECONDS


# ---- fix round D: the close person against the wall and the door ---------------------------------

CLOSE = Detection("person", 0.9, (70, 0, 570, 360), 1)
WALL_BEHIND = Detection("wall", 0.6, (0, 0, 640, 360), 50)      # overlaps her 0.78, but only 0.78 inside her
DOOR_BEHIND = Detection("door", 0.6, (250, 0, 420, 360), 51)    # wholly inside her box


def test_the_wall_and_the_door_behind_a_close_person_are_on_her_not_her():
    for find in (WALL_BEHIND, DOOR_BEHIND):
        assert not same_as_person(find, [CLOSE.box]) and on_person(find, [CLOSE.box])


def test_a_close_person_keeps_her_box_and_the_wall_and_door_on_her_are_never_objects(make):
    sightings = FakeSightings()
    pipeline = make(source=FakeSource(frames=10_000), detector=FakeDetector(detections=[CLOSE]),
                    recognizer=FakeRecognizer(faces=[]), open_detector=FakeOpenDetector(finds=[WALL_BEHIND, DOOR_BEHIND]),
                    sightings=sightings, config=memory_config(), state=RecordingState())
    started_search(pipeline, ["wall", "door"], "door")
    objects = {o["label"]: o for o in pipeline.status()["objects"]}
    assert objects["person"]["track_id"] == 1                    # the wall does not replace her box
    assert objects["wall"]["on_person"] is True and objects["door"]["on_person"] is True
    for track in (OFFSET + 50, OFFSET + 51):
        target = pipeline.target(track_id=track)
        assert (target.kind, target.track_id) == ("person", 1)
    frames_later(pipeline, 4)
    assert told(sightings, "wall") == [] and told(sightings, "door") == []


# ---- fix round E: a find inside two overlapping people ---------------------------------------------

STRANGER_BEHIND = Detection("person", 0.7, (80, 40, 300, 340), 5)      # Sara's box lies inside it
GLASSES_ON_SARA = Detection("glasses", 0.75, (160, 95, 200, 110), 25)  # centre in Sara's face box


def test_glasses_inside_two_people_belong_to_the_one_whose_face_holds_them(make):
    pipeline = make(source=FakeSource(frames=10_000), detector=FakeDetector(detections=[PERSON, STRANGER_BEHIND]),
                    open_detector=FakeOpenDetector(finds=[GLASSES_ON_SARA]))
    started_search(pipeline, ["glasses"], "glasses")
    assert wait_until(lambda: pipeline.target(track_id=1) is not None and pipeline.target(track_id=1).person == SARA)
    for target in (pipeline.target(track_id=OFFSET + 25), pipeline.target(x=180 / 640, y=102 / 360)):
        assert (target.kind, target.track_id, target.person) == ("person", 1, SARA)


def test_with_no_face_the_smallest_of_the_people_who_tie_is_meant():
    glasses = Detection("glasses", 0.75, (160, 95, 200, 110), OFFSET + 25)
    assert pipeline_module.wearer_of(glasses, [PERSON, STRANGER_BEHIND]) == 0
    assert pipeline_module.wearer_of(glasses, [STRANGER_BEHIND, PERSON]) == 1


# ---- a maybe, and a discovered find, on a person ----------------------------------------------------


def test_a_maybe_on_a_person_is_never_remembered_and_its_click_is_the_person(make):
    maybe_glasses = Detection("glasses", 0.22, (150, 90, 210, 110), 26)    # CONFIG: sure 0.25
    sightings = FakeSightings()
    pipeline = make(source=FakeSource(frames=10_000), open_detector=FakeOpenDetector(finds=[maybe_glasses]),
                    sightings=sightings, config=memory_config(), state=RecordingState())
    started_search(pipeline, ["glasses"], "glasses")
    glasses = next(o for o in pipeline.status()["objects"] if o["label"] == "glasses")
    assert glasses["maybe"] is True and glasses["on_person"] is True
    assert (pipeline.target(track_id=OFFSET + 26).kind, pipeline.target(track_id=OFFSET + 26).person) == ("person", SARA)
    frames_later(pipeline, 4)
    assert told(sightings, "glasses") == []


def test_a_discovered_find_on_a_person_is_shown_but_never_an_object(make):
    scarf = Detection("scarf", 0.8, (130, 150, 230, 200), 27)
    sightings = FakeSightings()
    pipeline = make(source=FakeSource(frames=10_000), open_detector=FakeOpenDetector(finds=[scarf]),
                    lookout=FakeLookout(["scarf"]), config=discover_config(0.02), sightings=sightings,
                    state=RecordingState()).start()
    pipeline.discover(True)
    assert wait_until(lambda: any(o["label"] == "scarf" for o in pipeline.status()["objects"]))
    entry = next(o for o in pipeline.status()["objects"] if o["label"] == "scarf")
    assert entry["discovered"] is True and entry["on_person"] is True
    assert pipeline.target(track_id=OFFSET + 27).kind == "person"
    frames_later(pipeline, 4)
    assert told(sightings, "scarf") == []


# ---- fix round C: no memory picture near a person, and faces covered in every cut-out -----------------


def test_a_window_that_holds_a_person_is_remembered_without_a_picture(make):
    window = Detection("window", 0.6, (0, 0, 640, 340), 60)       # Sara stands in front of it
    curtain = Detection("curtain", 0.6, (560, 0, 630, 150), 61)   # apart from her
    sightings = FakeSightings()
    pipeline = make(source=FakeSource(frames=10_000), open_detector=FakeOpenDetector(finds=[window, curtain]),
                    sightings=sightings, config=memory_config(), state=RecordingState())
    started_search(pipeline, ["window", "curtain"], "window")
    assert wait_until(lambda: any(call["picture"] for call in told(sightings, "curtain")))   # pictures are kept ...
    frames_later(pipeline)
    assert told(sightings, "window") and all(call["picture"] is None for call in told(sightings, "window"))
    assert pipeline.target(track_id=OFFSET + 60).kind == "object"                            # ... it is still a thing


RED = (0, 0, 255)                     # the face's pixels, so a face left in a picture is easy to find
CUP_BESIDE = Detection("cup", 0.6, (215, 90, 245, 130), 9)   # beside Sara's face: its photo holds the face


def red_face_picture():
    picture = np.full((360, 640, 3), 60, np.uint8)
    x1, y1, x2, y2 = SARA_FACE.box
    picture[y1:y2, x1:x2] = RED
    return picture


def red_and_grey(jpeg):
    """How many pixels of a JPEG are face red, and how many cover grey."""
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).astype(int)
    blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    face = (red > 150) & (green < 90) & (blue < 90)
    grey = (abs(blue - 128) < 12) & (abs(green - 128) < 12) & (abs(red - 128) < 12)
    return int(face.sum()), int(grey.sum())


def test_cover_faces_covers_each_face_grown_by_a_fifth_with_mid_grey():
    picture = red_face_picture()
    covered = pipeline_module.cover_faces(picture, [SARA_FACE.box])        # 60 x 70: grown 12 and 14
    assert (covered[56:154, 138:222] == 128).all()
    outside = covered.copy()
    outside[56:154, 138:222] = 60
    assert (outside == 60).all()                                           # nothing else changed
    assert (picture[70:140, 150:210] == RED).all()                         # the frame itself is untouched
    corner = pipeline_module.cover_faces(picture, [(0, 0, 20, 20)])
    assert (corner[0:24, 0:24] == 128).all() and (corner[24, 24] == 60).all()   # kept inside the frame


def covering_pipeline(make):
    pipeline = make(source=FakeSource(frames=10_000, picture=red_face_picture()),
                    detector=FakeDetector(detections=[PERSON, CUP_BESIDE])).start()
    assert wait_until(lambda: pipeline.status()["people"] and pipeline.target(track_id=9) is not None)
    return pipeline


def test_the_face_is_covered_in_object_and_area_cut_outs(make):
    pipeline = covering_pipeline(make)
    for target in (pipeline.target(track_id=9), pipeline.target(x=300 / 640, y=100 / 360)):
        assert target.kind in ("object", "area")
        red, grey = red_and_grey(target.photo)
        assert red == 0 and grey > 1000, target.kind
    red, _grey = red_and_grey(pipeline.target(track_id=1).photo)
    assert red > 1000                         # a person's own photo is never sent, so it is not covered


class AnsweringAssistant:
    """Keeps every picture a follow-up question would send."""

    def __init__(self):
        self.pictures = []

    def follow_up(self, photo, label, info, pairs, question, marked=None, language="en"):
        self.pictures += [photo, marked]
        return types.SimpleNamespace(text="A cup.", used_web=False, sources=[])


def test_the_face_is_covered_in_the_shown_photo_and_in_the_brushed_copy(make):
    import base64

    pipeline = covering_pipeline(make)
    assistant = AnsweringAssistant()
    client = flask_client(pipeline, assistant)
    body = client.post("/api/inspect", json={"track_id": 9, "label": "cup"}).get_json()
    shown = base64.b64decode(body["photo"].split(",", 1)[1])
    assert body["kind"] == "object" and red_and_grey(shown)[0] == 0      # the photo the page shows
    strokes = [{"points": [[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]]}]
    reply = client.post("/api/follow_up", json={"id": body["id"], "question": "What is it?", "strokes": strokes})
    assert reply.status_code == 200
    photo, marked = assistant.pictures
    assert marked is not None
    for picture in (photo, marked):
        assert red_and_grey(picture)[0] == 0
