"""The Flask server and the demo page, with a fake pipeline and a fake assistant.

No camera, model, network, graphics card or API key is needed. Flask's own
test client sends the requests; main() is run with Flask.run replaced, so no
port is opened.
"""

import ast
import base64
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
from car_vision import app as app_module
from car_vision.app import (AnswerCache, Inspection, InspectionStore, UnavailableAssistant, create_app,
                            jpeg_part, main, mjpeg_stream)
from car_vision.assistant import Answer, AssistantError
from car_vision.config import ConfigError
from car_vision.faces import Person
from car_vision.pipeline import Target
from car_vision.state import WorldState

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "car_vision" / "templates" / "index.html"

STATUS = {
    "source": "webcam", "connected": True, "fps": 21.4, "device": "cuda", "recognition": True,
    "notes": ["Nobody is enrolled yet, so every face is shown as Unknown."],
    "people": [{"id": "sara", "name": "Sara Ahmed", "role": "Team lead",
                "lines": ["Built the chassis"], "similarity": 0.61}],
    "objects": [{"label": "chair", "count": 2}],
    "reactions": [{"kind": "greeting", "text": "Hello, Sara Ahmed!"}],
    "error": None,
}
JPEG = cv2.imencode(".jpg", np.full((36, 64, 3), 90, np.uint8))[1].tobytes()
WEB_ANSWER = Answer("Rover follows a black line.", used_web=True,
                    sources=("https://example.org/a", "https://example.org/b"))


@dataclass(frozen=True)
class FakeInfo:
    """Shaped like the assistant's ObjectInfo."""

    name: str
    summary: str
    specs: tuple = ()
    used_web: bool = False
    sources: tuple = ()


PHOTO = cv2.imencode(".jpg", np.full((40, 30, 3), 120, np.uint8))[1].tobytes()
LAPTOP = Target("object", "laptop", 0.834, 7, PHOTO)
CUP = Target("object", "cup", 0.6, None, PHOTO)                  # the tracker gave it no number
AREA = Target("area", "", 0.0, None, PHOTO)
SARA = Person("sara", "Sara Ahmed", "Team lead", ("Built the chassis", "Fourth-year student"),
              story="Sara joined in 2024.\n\nShe likes motors.")
SARA_TARGET = Target("person", "person", 0.9, 1, PHOTO, SARA)
STRANGER = Target("person", "person", 0.9, 2, PHOTO)
LAPTOP_INFO = FakeInfo("Laptop", "A portable computer.", (("Screen", "15.6 in"), ("Weight", "about 2 kg")),
                       used_web=True, sources=("https://example.org/laptop",))


# ---- fakes ---------------------------------------------------------------------


class FakePipeline:
    instances = []

    def __init__(self, config=None, jpeg=JPEG, target=LAPTOP):
        self.config = config
        self.state = WorldState()
        self.jpeg = jpeg
        self.target_result = target
        self.clicks = []
        self.started = False
        self.stopped = False
        FakePipeline.instances.append(self)

    def target(self, x=None, y=None, track_id=None, label=None):
        self.clicks.append({"x": x, "y": y, "track_id": track_id, "label": label})
        if isinstance(self.target_result, Exception):
            raise self.target_result
        return self.target_result

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    def status(self):
        return dict(STATUS)

    def latest_jpeg(self):
        return self.jpeg


class FakeAssistant:
    def __init__(self, answer=WEB_ANSWER, error=None, reset_error=None, info=LAPTOP_INFO, describe_error=None):
        self.answer = answer
        self.error = error
        self.reset_error = reset_error
        self.info = info
        self.describe_error = describe_error
        self.questions = []
        self.looked_up = []                 # (image, label) of every describe call
        self.resets = 0

    def describe(self, image, label=None):
        self.looked_up.append((image, label))
        if self.describe_error is not None:
            raise self.describe_error
        return self.info

    def ask(self, question):
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return self.answer

    def reset(self):
        self.resets += 1
        if self.reset_error is not None:
            raise self.reset_error


def client_for(assistant=None, pipeline=None):
    app = create_app(pipeline or FakePipeline(), assistant or FakeAssistant())
    app.testing = True
    return app.test_client()


# ---- routes ------------------------------------------------------------------------


def test_page_loads():
    response = client_for().get("/")
    assert response.status_code == 200
    assert response.mimetype == "text/html"
    page = response.get_data(as_text=True)
    assert "<title>Robot car vision</title>" in page
    assert 'maxlength="2000"' in page
    assert 'src="/video"' in page


def test_status_is_the_pipeline_status_as_json():
    response = client_for().get("/api/status")
    assert response.status_code == 200
    assert response.get_json() == STATUS
    assert response.headers["Cache-Control"] == "no-store"


def test_ask_returns_the_answer():
    assistant = FakeAssistant()
    response = client_for(assistant).post("/api/ask", json={"question": "  How does Rover steer?  "})
    assert response.status_code == 200
    assert response.get_json() == {"text": "Rover follows a black line.", "used_web": True,
                                   "sources": ["https://example.org/a", "https://example.org/b"]}
    assert assistant.questions == ["How does Rover steer?"]


def test_ask_with_an_assistant_error_returns_200_and_the_message():
    assistant = FakeAssistant(error=AssistantError("No API key found. Copy .env.example to .env."))
    response = client_for(assistant).post("/api/ask", json={"question": "Hello?"})
    assert response.status_code == 200
    assert response.get_json() == {"error": "No API key found. Copy .env.example to .env."}


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_ask_with_an_empty_question_is_400(question):
    assistant = FakeAssistant()
    response = client_for(assistant).post("/api/ask", json={"question": question})
    assert response.status_code == 400
    assert "empty" in response.get_json()["error"]
    assert assistant.questions == []


def test_ask_with_an_over_long_question_is_400():
    assistant = FakeAssistant()
    client = client_for(assistant)
    response = client.post("/api/ask", json={"question": "a" * 2001})
    assert response.status_code == 400
    assert "too long" in response.get_json()["error"]
    assert assistant.questions == []
    assert client.post("/api/ask", json={"question": "a" * 2000}).status_code == 200


def test_ask_maps_the_assistants_value_error_to_400_with_its_message():
    assistant = FakeAssistant(error=ValueError("The question is empty."))
    response = client_for(assistant).post("/api/ask", json={"question": "﻿?"})
    assert response.status_code == 400
    assert response.get_json() == {"error": "The question is empty."}


@pytest.mark.parametrize("body, content_type", [
    ("question=hello", "application/x-www-form-urlencoded"),
    ("not json at all", "application/json"),
    ("hello", "text/plain"),
    ('["a list"]', "application/json"),
    ('{"text": "wrong key"}', "application/json"),
    ('{"question": 42}', "application/json"),
])
def test_ask_with_a_body_that_is_not_a_json_question_is_400(body, content_type):
    assistant = FakeAssistant()
    response = client_for(assistant).post("/api/ask", data=body, content_type=content_type)
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)
    assert assistant.questions == []


def test_ask_with_an_unexpected_error_gives_a_plain_message():
    assistant = FakeAssistant(error=KeyError("secret detail"))
    response = client_for(assistant).post("/api/ask", json={"question": "Hi"})
    assert response.status_code == 500
    message = response.get_json()["error"]
    assert "KeyError" in message and "secret detail" not in message and "Traceback" not in message


def test_reset_forgets_the_conversation():
    assistant = FakeAssistant()
    response = client_for(assistant).post("/api/reset")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert assistant.resets == 1


def test_reset_while_busy_returns_the_message():
    assistant = FakeAssistant(reset_error=AssistantError("The assistant is still busy."))
    response = client_for(assistant).post("/api/reset")
    assert response.status_code == 200
    assert response.get_json() == {"error": "The assistant is still busy."}


def test_page_works_and_ask_explains_when_the_assistant_could_not_be_built():
    client = client_for(UnavailableAssistant("The assistant is not available: the notes file was not found"))
    assert client.get("/").status_code == 200
    assert client.get("/api/status").status_code == 200
    response = client.post("/api/ask", json={"question": "Hi"})
    assert response.status_code == 200
    assert "notes file was not found" in response.get_json()["error"]
    assert client.post("/api/reset").get_json() == {"ok": True}


def test_only_the_listed_methods_are_accepted():
    client = client_for()
    assert client.get("/api/ask").status_code == 405
    assert client.post("/api/status").status_code == 405
    assert client.get("/api/inspect").status_code == 405
    assert client.get("/api/describe").status_code == 405


# ---- /api/inspect --------------------------------------------------------------------


def photo_from(address):
    assert address.startswith("data:image/jpeg;base64,")
    return base64.b64decode(address.split(",", 1)[1])


def test_inspect_a_click_on_the_video():
    pipeline = FakePipeline()
    response = client_for(pipeline=pipeline).post("/api/inspect", json={"x": 0.42, "y": 0.61})
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data["id"], str) and len(data["id"]) >= 12
    assert {k: v for k, v in data.items() if k not in ("id", "photo")} == {
        "kind": "object", "label": "laptop", "confidence": 0.83, "track_id": 7,
        "person": None, "describable": True, "info": None}
    assert photo_from(data["photo"]) == PHOTO
    assert pipeline.clicks == [{"x": 0.42, "y": 0.61, "track_id": None, "label": None}]


def test_inspect_from_the_list_passes_the_track_and_label():
    pipeline = FakePipeline()
    body = {"track_id": 7, "label": "laptop", "x": 0.5, "y": 0.25}
    assert client_for(pipeline=pipeline).post("/api/inspect", json=body).status_code == 200
    assert pipeline.clicks == [{"x": 0.5, "y": 0.25, "track_id": 7, "label": "laptop"}]


def test_inspect_with_a_track_id_needs_no_point():
    pipeline = FakePipeline()
    assert client_for(pipeline=pipeline).post("/api/inspect", json={"track_id": 7}).status_code == 200
    assert pipeline.clicks == [{"x": None, "y": None, "track_id": 7, "label": None}]


def test_inspect_points_on_the_edges_are_fine():
    client = client_for()
    for point in ({"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 1.0, "y": 0.0}):
        assert client.post("/api/inspect", json=point).status_code == 200


def test_inspect_a_recognised_person():
    response = client_for(pipeline=FakePipeline(target=SARA_TARGET)).post("/api/inspect", json={"x": 0.2, "y": 0.3})
    data = response.get_json()
    assert data["kind"] == "person" and data["describable"] is False and data["info"] is None
    assert data["person"] == {"id": "sara", "name": "Sara Ahmed", "role": "Team lead",
                              "lines": ["Built the chassis", "Fourth-year student"],
                              "story": "Sara joined in 2024.\n\nShe likes motors."}


def test_inspect_an_unrecognised_person_and_an_area():
    stranger = client_for(pipeline=FakePipeline(target=STRANGER)).post("/api/inspect", json={"x": 0.2, "y": 0.3})
    assert stranger.get_json()["person"] is None
    assert stranger.get_json()["describable"] is False
    area = client_for(pipeline=FakePipeline(target=AREA)).post("/api/inspect", json={"x": 0.2, "y": 0.3})
    assert {k: area.get_json()[k] for k in ("kind", "label", "confidence", "track_id", "describable")} == {
        "kind": "area", "label": "", "confidence": 0.0, "track_id": None, "describable": True}


@pytest.mark.parametrize("body", [
    {"x": -0.01, "y": 0.5},
    {"x": 0.5, "y": 1.2},
    {},                                         # no point and no track id
    {"label": "laptop"},
    {"x": 0.5},                                 # half a point
    {"track_id": 7, "y": 0.5},
    {"x": "0.5", "y": 0.5},
    {"x": True, "y": 0.5},
    {"x": None, "y": 0.5},
    {"track_id": "7"},
    {"track_id": 7.0},
    {"track_id": True},
    {"track_id": 7, "label": 3},
])
def test_inspect_with_a_bad_click_is_400(body):
    pipeline = FakePipeline()
    response = client_for(pipeline=pipeline).post("/api/inspect", json=body)
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)
    assert pipeline.clicks == []


@pytest.mark.parametrize("body, content_type", [
    ("x=0.5&y=0.5", "application/x-www-form-urlencoded"),
    ("not json", "application/json"),
    ("[0.5, 0.5]", "application/json"),
    ('{"x": NaN, "y": 0.5}', "application/json"),
    ('{"x": Infinity, "y": 0.5}', "application/json"),
])
def test_inspect_with_a_body_that_is_not_a_json_click_is_400(body, content_type):
    pipeline = FakePipeline()
    response = client_for(pipeline=pipeline).post("/api/inspect", data=body, content_type=content_type)
    assert response.status_code == 400
    assert pipeline.clicks == []


def test_inspect_something_no_longer_in_view_is_404():
    response = client_for(pipeline=FakePipeline(target=None)).post("/api/inspect", json={"track_id": 7})
    assert response.status_code == 404
    assert response.get_json() == {"error": "That is no longer in view."}


def test_inspect_with_an_unexpected_error_gives_a_plain_message():
    pipeline = FakePipeline(target=KeyError("secret detail"))
    response = client_for(pipeline=pipeline).post("/api/inspect", json={"x": 0.5, "y": 0.5})
    assert response.status_code == 500
    assert "secret detail" not in response.get_json()["error"]


# ---- /api/describe --------------------------------------------------------------------


def inspect_then_describe(client, body=None):
    inspection = client.post("/api/inspect", json=body or {"x": 0.5, "y": 0.5}).get_json()
    return inspection, client.post("/api/describe", json={"id": inspection["id"]})


def test_describe_an_object():
    assistant = FakeAssistant()
    _inspection, response = inspect_then_describe(client_for(assistant))
    assert response.status_code == 200
    assert response.get_json() == {
        "name": "Laptop", "summary": "A portable computer.",
        "specs": [{"name": "Screen", "value": "15.6 in"}, {"name": "Weight", "value": "about 2 kg"}],
        "used_web": True, "sources": ["https://example.org/laptop"]}
    assert assistant.looked_up == [(PHOTO, "laptop")]       # the server's own photo, never one from the page


def test_describe_an_area_sends_no_label():
    assistant = FakeAssistant(info=FakeInfo("Unknown object", "I cannot tell what this is."))
    _inspection, response = inspect_then_describe(client_for(assistant, FakePipeline(target=AREA)))
    assert response.get_json()["name"] == "Unknown object"
    assert response.get_json()["specs"] == [] and response.get_json()["used_web"] is False
    assert assistant.looked_up == [(PHOTO, None)]


@pytest.mark.parametrize("target", [SARA_TARGET, STRANGER])
def test_a_person_is_never_passed_to_describe(target):
    assistant = FakeAssistant()
    _inspection, response = inspect_then_describe(client_for(assistant, FakePipeline(target=target)))
    assert response.status_code == 400
    assert assistant.looked_up == []


def test_describe_with_an_unknown_id_is_404():
    assistant = FakeAssistant()
    response = client_for(assistant).post("/api/describe", json={"id": "not-a-real-id"})
    assert response.status_code == 404
    assert "click the object again" in response.get_json()["error"]
    assert assistant.looked_up == []


@pytest.mark.parametrize("body", [{}, {"id": 5}, {"id": None}, ["id"]])
def test_describe_with_a_bad_body_is_400(body):
    assistant = FakeAssistant()
    assert client_for(assistant).post("/api/describe", json=body).status_code == 400
    assert assistant.looked_up == []


def test_describe_an_expired_inspection_is_404():
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=CUP))
    first = client.post("/api/inspect", json={"x": 0.5, "y": 0.5}).get_json()["id"]
    for _ in range(20):
        client.post("/api/inspect", json={"x": 0.5, "y": 0.5})
    assert client.post("/api/describe", json={"id": first}).status_code == 404
    assert assistant.looked_up == []


def test_a_second_click_on_the_same_tracked_object_is_answered_from_the_cache():
    assistant = FakeAssistant()
    client = client_for(assistant)
    first, _answer = inspect_then_describe(client)
    second = client.post("/api/inspect", json={"track_id": 7, "label": "laptop"}).get_json()
    assert second["id"] != first["id"]
    assert second["info"]["name"] == "Laptop"                 # the page needs no second look-up
    assert client.post("/api/describe", json={"id": second["id"]}).get_json()["name"] == "Laptop"
    assert len(assistant.looked_up) == 1


@pytest.mark.parametrize("target", [CUP, AREA])
def test_untracked_objects_and_areas_are_not_cached(target):
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=target))
    _first, _answer = inspect_then_describe(client)
    second, _answer = inspect_then_describe(client)
    assert second["info"] is None
    assert len(assistant.looked_up) == 2


def test_describe_with_an_assistant_error_returns_200_and_the_message():
    assistant = FakeAssistant(describe_error=AssistantError("The assistant is still busy. Please try again."))
    client = client_for(assistant)
    _inspection, response = inspect_then_describe(client)
    assert response.status_code == 200
    assert response.get_json() == {"error": "The assistant is still busy. Please try again."}
    second, _answer = inspect_then_describe(client)
    assert second["info"] is None                               # an error is not cached


def test_describe_maps_the_assistants_value_error_to_400():
    assistant = FakeAssistant(describe_error=ValueError("The picture is too large."))
    _inspection, response = inspect_then_describe(client_for(assistant))
    assert response.status_code == 400
    assert response.get_json() == {"error": "The picture is too large."}


def test_describe_with_an_unexpected_error_gives_a_plain_message():
    assistant = FakeAssistant(describe_error=KeyError("secret detail"))
    _inspection, response = inspect_then_describe(client_for(assistant))
    assert response.status_code == 500
    assert "secret detail" not in response.get_json()["error"]


def test_describe_explains_when_the_assistant_could_not_be_built():
    client = client_for(UnavailableAssistant("The assistant is not available: no notes file"))
    _inspection, response = inspect_then_describe(client)
    assert response.status_code == 200
    assert "no notes file" in response.get_json()["error"]


def test_unavailable_assistant_describe_raises_the_same_error_as_ask():
    assistant = UnavailableAssistant("not available")
    with pytest.raises(AssistantError, match="not available"):
        assistant.describe(PHOTO, "laptop")
    with pytest.raises(AssistantError, match="not available"):
        assistant.ask("Hi")


def test_the_server_logs_no_photo_and_no_person_details(caplog):
    caplog.set_level(logging.DEBUG)
    client = client_for(FakeAssistant(), FakePipeline(target=SARA_TARGET))
    inspect_then_describe(client)
    inspect_then_describe(client_for(FakeAssistant()))
    text = caplog.text
    encoded = base64.b64encode(PHOTO).decode("ascii")
    for private in ("Sara", "Team lead", "chassis", "motors", encoded[:40]):
        assert private not in text


# ---- the stores behind the two routes --------------------------------------------------


def test_inspection_store_keeps_the_last_20():
    store = InspectionStore()
    ids = [store.add(Inspection("object", f"thing {n}", n, PHOTO)) for n in range(21)]
    assert len(set(ids)) == 21
    assert store.get(ids[0]) is None                           # the oldest went
    assert store.get(ids[1]).label == "thing 1"
    assert store.get(ids[20]).label == "thing 20"
    assert store.get("made-up") is None


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_answer_cache_forgets_after_ten_minutes():
    clock = Clock()
    cache = AnswerCache(clock=clock)
    cache.put((7, "laptop"), {"name": "Laptop"})
    clock.now += 599
    assert cache.get((7, "laptop")) == {"name": "Laptop"}
    assert cache.get((7, "cup")) is None and cache.get((8, "laptop")) is None
    clock.now += 1
    assert cache.get((7, "laptop")) is None


def test_answer_cache_keeps_at_most_50():
    clock = Clock()
    cache = AnswerCache(clock=clock)
    for n in range(51):
        clock.now += 1
        cache.put((n, "cup"), {"name": f"Cup {n}"})
    assert cache.get((0, "cup")) is None
    assert cache.get((1, "cup")) == {"name": "Cup 1"}
    assert cache.get((50, "cup")) == {"name": "Cup 50"}


def test_answer_cache_put_again_starts_the_ten_minutes_again():
    clock = Clock()
    cache = AnswerCache(clock=clock)
    cache.put((7, "laptop"), {"name": "Old"})
    clock.now += 500
    cache.put((7, "laptop"), {"name": "New"})
    clock.now += 500
    assert cache.get((7, "laptop")) == {"name": "New"}


# ---- the video stream ------------------------------------------------------------------


def test_video_has_the_mjpeg_content_type_and_sends_a_frame():
    response = client_for().get("/video", buffered=False)
    try:
        assert response.status_code == 200
        assert response.mimetype == "multipart/x-mixed-replace"
        assert response.mimetype_params["boundary"] == "frame"
        first = next(iter(response.response))
    finally:
        response.close()
    head, picture = first.split(b"\r\n\r\n", 1)
    assert head.startswith(b"--frame\r\nContent-Type: image/jpeg")
    assert f"Content-Length: {len(JPEG)}".encode() in head
    assert picture == JPEG + b"\r\n"
    decoded = cv2.imdecode(np.frombuffer(picture[:-2], np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (36, 64, 3)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ChangingPipeline:
    """A new picture on every call, as fast as it is asked."""

    def __init__(self):
        self.calls = 0

    def latest_jpeg(self):
        self.calls += 1
        return b"picture " + str(self.calls).encode()


def test_stream_sends_at_most_20_frames_a_second():
    clock = FakeClock()
    stream = mjpeg_stream(ChangingPipeline(), clock=clock, sleep=clock.sleep)
    sent_at = []
    while clock.now < 2.0:
        next(stream)
        sent_at.append(clock.now)
    gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
    assert min(gaps) >= 0.05 - 1e-9
    assert len(sent_at) <= 2.0 * 20 + 1


class StillPipeline:
    """No picture at first, then the same picture object for a while, then a new one."""

    def __init__(self):
        self.calls = 0
        self.first = b"first picture"
        self.second = b"second picture"

    def latest_jpeg(self):
        self.calls += 1
        if self.calls <= 3:
            return None
        return self.first if self.calls <= 30 else self.second


def test_stream_sends_a_frame_only_when_it_is_new():
    clock = FakeClock()
    pipeline = StillPipeline()
    stream = mjpeg_stream(pipeline, clock=clock, sleep=clock.sleep)
    assert next(stream) == jpeg_part(pipeline.first)
    assert next(stream) == jpeg_part(pipeline.second)
    assert pipeline.calls == 31                          # the first picture was not sent again


# ---- main() ----------------------------------------------------------------------------


TEST_CONFIG = {
    "stream": {"source": "webcam", "file_path": "data/samples/car_clip.mp4"},
    "assistant": {"notes_file": "data/does-not-exist/knowledge.md"},
    "app": {"host": "127.0.0.1", "port": 5000, "jpeg_quality": 80, "log_file": "logs/car_vision.log"},
}


@pytest.fixture
def fake_main(monkeypatch, tmp_path):
    """main() with the settings, the pipeline, the assistant and Flask.run replaced."""
    FakePipeline.instances = []
    calls = {"run": [], "assistants": [], "browser": []}
    config = {**TEST_CONFIG, "app": {**TEST_CONFIG["app"], "log_file": str(tmp_path / "logs" / "app.log")}}
    options = {"run_raises": None, "assistant_raises": None}

    def fake_run(self, **kwargs):
        calls["run"].append((self, kwargs))
        if options["run_raises"] is not None:
            raise options["run_raises"]

    def fake_assistant(settings, state=None):
        if options["assistant_raises"] is not None:
            raise options["assistant_raises"]
        calls["assistants"].append((settings, state))
        return FakeAssistant()

    monkeypatch.setattr(app_module, "load_config", lambda: config)
    monkeypatch.setattr(app_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(app_module, "Assistant", fake_assistant)
    monkeypatch.setattr(app_module.Flask, "run", fake_run)
    monkeypatch.setattr(app_module.webbrowser, "open", lambda url: calls["browser"].append(url))
    monkeypatch.setattr(app_module, "BROWSER_DELAY_SECONDS", 0.0)

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    werkzeug_level = logging.getLogger("werkzeug").level
    yield {"calls": calls, "options": options, "log_file": tmp_path / "logs" / "app.log"}
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    logging.getLogger("werkzeug").setLevel(werkzeug_level)


def test_main_serves_on_app_host_only_without_reloader_or_debug(fake_main, capsys):
    assert main([]) == 0
    (_app, kwargs), = fake_main["calls"]["run"]
    assert kwargs == {"host": "127.0.0.1", "port": 5000, "threaded": True, "debug": False,
                      "use_reloader": False, "load_dotenv": False}
    pipeline = FakePipeline.instances[0]
    assert pipeline.started and pipeline.stopped
    assert "http://127.0.0.1:5000/" in capsys.readouterr().out
    assert fake_main["log_file"].parent.is_dir()
    assert fake_main["calls"]["browser"] == []


def test_assistant_shares_the_pipelines_world_state(fake_main):
    main([])
    (settings, state), = fake_main["calls"]["assistants"]
    assert state is FakePipeline.instances[0].state
    assert settings == TEST_CONFIG["assistant"]


def test_source_argument_overrides_the_stream_source(fake_main):
    main(["--source", "file"])
    assert FakePipeline.instances[0].config["stream"]["source"] == "file"


def test_file_argument_implies_file_source_and_overrides_the_path(fake_main):
    main(["--file", "data/samples/people_clip.mp4"])
    stream = FakePipeline.instances[0].config["stream"]
    assert stream["source"] == "file"
    assert stream["file_path"] == "data/samples/people_clip.mp4"
    assert TEST_CONFIG["stream"]["file_path"] == "data/samples/car_clip.mp4"   # settings not changed


def test_file_argument_cannot_be_combined_with_another_source():
    with pytest.raises(SystemExit):
        main(["--file", "clip.mp4", "--source", "webcam"])


def test_ctrl_c_stops_the_pipeline_cleanly(fake_main):
    fake_main["options"]["run_raises"] = KeyboardInterrupt()
    assert main([]) == 0
    assert FakePipeline.instances[0].stopped


def test_port_in_use_gives_a_plain_message_and_stops_the_pipeline(fake_main, capsys):
    fake_main["options"]["run_raises"] = OSError("address already in use")
    assert main([]) == 1
    assert FakePipeline.instances[0].stopped
    assert "Cannot serve the page" in capsys.readouterr().out


def test_pipeline_config_error_gives_one_plain_message(fake_main, monkeypatch, capsys):
    def broken(config):
        raise ConfigError("stream.source is 'file' but the clip does not exist")

    monkeypatch.setattr(app_module, "Pipeline", broken)
    assert main([]) == 1
    assert "Cannot start" in capsys.readouterr().out
    assert fake_main["calls"]["run"] == []


def test_page_still_works_when_the_assistant_cannot_be_built(fake_main):
    fake_main["options"]["assistant_raises"] = AssistantError("The notes file was not found: knowledge.md")
    assert main([]) == 0
    (app, _kwargs), = fake_main["calls"]["run"]
    client = app.test_client()
    assert client.get("/").status_code == 200
    answer = client.post("/api/ask", json={"question": "Hi"})
    assert answer.status_code == 200
    assert "The notes file was not found" in answer.get_json()["error"]


def test_open_argument_opens_the_page_in_the_browser(fake_main):
    main(["--open"])
    deadline = time.monotonic() + 2.0
    while not fake_main["calls"]["browser"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake_main["calls"]["browser"] == ["http://127.0.0.1:5000/"]


def test_main_logs_to_the_log_file(fake_main):
    main([])
    logging.getLogger("car_vision.test").info("hello from the test")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello from the test" in fake_main["log_file"].read_text(encoding="utf-8")


# ---- the page template -------------------------------------------------------------------


def test_template_uses_no_inner_html_and_no_outside_resources():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "innerHTML" not in page and "outerHTML" not in page and "insertAdjacentHTML" not in page
    assert "http://" not in page and "https://" not in page
    assert not re.search(r"<link\b", page, re.IGNORECASE)
    assert not re.search(r"<script\b[^>]*\bsrc\s*=", page, re.IGNORECASE)
    assert "@import" not in page
    for address in re.findall(r"""\b(?:src|href)\s*=\s*["']([^"']*)["']""", page):
        in_page = address.startswith("#")                       # a link to a section of the page
        assert in_page or (address.startswith("/") and not address.startswith("//")), address


def test_template_only_makes_links_for_http_and_https():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert 'url.protocol === "http:" || url.protocol === "https:"' in page
    assert "textContent" in page


def test_template_has_the_list_the_panel_and_the_privacy_line():
    page = client_for().get("/").get_data(as_text=True)
    assert 'id="things"' in page and "In view" in page
    assert 'id="details"' in page and 'id="closeDetails"' in page and '"Escape"' in page
    assert ("A click on an object sends a small cut-out of the picture to the assistant's API. "
            "A click on a person sends nothing.") in page
    assert "Add a folder with photos and an about.md for them in data/people" in page
    assert "Something the detector has no name for" in page
    assert "/api/inspect" in page and "/api/describe" in page


def test_template_shows_a_photo_only_from_a_jpeg_data_address():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert 'PHOTO_PREFIX = "data:image/jpeg;base64,"' in page
    assert "address.startsWith(PHOTO_PREFIX)" in page
    assert page.count("detailPhoto.src =") == 1                     # set in one place, after that check


def test_template_measures_the_click_against_the_picture_not_the_element():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "naturalWidth" in page and "naturalHeight" in page and "getBoundingClientRect" in page
    assert "MAX_SOURCES = 3" in page and '" more"' in page


# ---- which files may use flask --------------------------------------------------------------


def flask_imports(source):
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if alias.name.split(".")[0] == "flask"]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] == "flask":
                found.append(node.module)
    return found


def test_flask_guard_finds_imports_anywhere():
    source = "import os\ndef f():\n    from flask import Flask\nimport flask.json as j\nimport flasky\n"
    assert sorted(flask_imports(source)) == ["flask", "flask.json"]     # flasky is another package


def test_only_app_imports_flask():
    files = [*ROOT.glob("*.py"), *(ROOT / "car_vision").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]
    offenders = {
        str(path.relative_to(ROOT)): found
        for path in files
        if path != ROOT / "car_vision" / "app.py"
        and (found := flask_imports(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}
    assert flask_imports((ROOT / "car_vision" / "app.py").read_text(encoding="utf-8"))
