"""The Flask server and the demo page, with a fake pipeline and a fake assistant.

No camera, model, network, graphics card or API key is needed. Flask's own
test client sends the requests; main() is run with Flask.run replaced, so no
port is opened.
"""

import ast
import base64
import json
import logging
import re
import socket
import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import pytest
from car_vision import app as app_module
from car_vision import phone as phone_module
from car_vision.app import (AnswerCache, Inspection, InspectionStore, UnavailableAssistant, create_app,
                            jpeg_part, main, mjpeg_stream)
from car_vision.assistant import Answer, AssistantError, ObjectInfo
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
OFFERED = ("640x360", "1280x720", "1920x1080")      # the sizes the fake phone offers
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
BIG_PHOTO = cv2.imencode(".jpg", np.full((240, 320, 3), 120, np.uint8))[1].tobytes()
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
        self.phones = []                    # every address use_phone() was given
        self.pictures = []                  # every choice set_phone_picture() took
        self.started = False
        self.stopped = False
        FakePipeline.instances.append(self)

    def use_phone(self, url):
        self.phones.append(url)

    def set_phone_picture(self, choice):
        """Like the real one: the two words, or a size this fake phone offers; ValueError otherwise."""
        if choice not in ("auto", "keep", *OFFERED):
            raise ValueError(f"The phone does not offer {choice}.")
        self.pictures.append(choice)

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
    def __init__(self, answer=WEB_ANSWER, error=None, reset_error=None, info=LAPTOP_INFO, describe_error=None,
                 follow_up_errors=()):
        self.answer = answer
        self.error = error
        self.reset_error = reset_error
        self.info = info
        self.describe_error = describe_error
        self.follow_up_errors = list(follow_up_errors)   # raised one per call, None for a good answer
        self.questions = []
        self.looked_up = []                 # (image, label) of every describe call
        self.follow_ups = []                # (image, label, info, earlier, question) of every follow_up call
        self.marked = []                    # the marked copy given with every follow_up call, or None
        self.resets = 0

    def follow_up(self, image, label, info, earlier, question, marked=None):
        self.follow_ups.append((image, label, info, earlier, question))
        self.marked.append(marked)
        error = self.follow_up_errors.pop(0) if self.follow_up_errors else None
        if error is not None:
            raise error
        return Answer(f"Answer {len(self.follow_ups)}", used_web=True, sources=("https://example.org/w",))

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


def test_only_the_listed_methods_are_accepted_and_the_answer_is_json():
    client = client_for()
    for method, path in (("get", "/api/ask"), ("post", "/api/status"), ("get", "/api/inspect"),
                         ("get", "/api/describe"), ("get", "/api/reset"), ("get", "/api/follow_up"),
                         ("get", "/api/camera"), ("get", "/api/camera/picture")):
        response = getattr(client, method)(path)
        assert response.status_code == 405
        assert response.is_json and isinstance(response.get_json()["error"], str)
        assert "Allow" in response.headers


def test_an_unknown_api_address_answers_json():
    response = client_for().post("/api/no-such-thing", json={})
    assert response.status_code == 404
    assert response.is_json and isinstance(response.get_json()["error"], str)


class BrokenPipeline(FakePipeline):
    def status(self):
        raise KeyError("secret detail")


def test_an_unexpected_error_in_any_api_route_answers_json_without_a_traceback():
    client = client_for(FakeAssistant(reset_error=RuntimeError("secret detail")), BrokenPipeline())
    for response in (client.get("/api/status"), client.post("/api/reset")):
        assert response.status_code == 500
        assert response.is_json
        message = response.get_json()["error"]
        assert "secret detail" not in message and "Traceback" not in message


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
    ('{"x": 1' + "0" * 400 + ', "y": 0.5}', "application/json"),       # too big for a float
    ('{"x": 0.5, "y": -1' + "0" * 400 + "}", "application/json"),
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


@dataclass(frozen=True)
class EpochTarget(Target):
    """A Target with the pipeline's epoch, whether or not pipeline.Target has it yet."""

    epoch: int = 0


def test_after_the_track_numbers_start_again_the_same_track_is_looked_up_again():
    assistant = FakeAssistant()
    pipeline = FakePipeline(target=EpochTarget("object", "laptop", 0.8, 7, PHOTO, epoch=0))
    client = client_for(assistant, pipeline)
    inspect_then_describe(client)
    pipeline.target_result = EpochTarget("object", "laptop", 0.8, 7, PHOTO, epoch=1)   # new numbers, same label
    second, described = inspect_then_describe(client)
    assert second["info"] is None                              # not the old object's description
    assert len(assistant.looked_up) == 2
    third, _described = inspect_then_describe(client)
    assert third["info"]["name"] == "Laptop" and len(assistant.looked_up) == 2   # same epoch: from memory


def test_the_cache_key_holds_the_epoch_and_a_target_without_one_is_epoch_0():
    assert app_module.cache_key(Inspection("object", "laptop", 7, PHOTO, epoch=3)) == (3, 7, "laptop")
    assert app_module.cache_key(Inspection("area", "", None, PHOTO, epoch=3)) is None
    client = client_for(pipeline=FakePipeline(target=LAPTOP))     # a Target from before epochs
    assert client.post("/api/inspect", json={"x": 0.5, "y": 0.5}).status_code == 200


class SlowAssistant(FakeAssistant):
    """describe() waits until the test lets it go, so two requests can overlap."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.go_on = threading.Event()

    def describe(self, image, label=None):
        self.started.set()
        assert self.go_on.wait(5)
        return super().describe(image, label)


def test_two_quick_clicks_on_the_same_object_make_one_api_call():
    assistant = SlowAssistant()
    app = create_app(FakePipeline(), assistant)
    app.testing = True
    ids = [app.test_client().post("/api/inspect", json={"track_id": 7}).get_json()["id"] for _ in range(2)]
    answers = {}

    def describe(inspection_id):
        answers[inspection_id] = app.test_client().post("/api/describe", json={"id": inspection_id}).get_json()

    first = threading.Thread(target=describe, args=(ids[0],))
    first.start()
    assert assistant.started.wait(5)                      # the first look-up is under way ...
    second = threading.Thread(target=describe, args=(ids[1],))
    second.start()                                        # ... when the second click arrives
    time.sleep(0.1)
    assistant.go_on.set()
    first.join(5)
    second.join(5)
    assert len(assistant.looked_up) == 1
    assert answers[ids[0]] == answers[ids[1]] and answers[ids[0]]["name"] == "Laptop"


def test_the_lookup_gate_forgets_objects_nobody_waits_for():
    gate = app_module.LookupGate()
    with gate.one_at_a_time((7, "laptop")):
        pass
    assert gate._gates == {}


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


# ---- /api/follow_up ----------------------------------------------------------------------

STORED_LAPTOP_INFO = ObjectInfo("Laptop", "A portable computer.", (("Screen", "15.6 in"), ("Weight", "about 2 kg")),
                                used_web=True, sources=("https://example.org/laptop",))


def ask_more(client, inspection_id, question="How heavy is it?"):
    return client.post("/api/follow_up", json={"id": inspection_id, "question": question})


def test_follow_up_about_an_object_sends_the_stored_photo_label_and_description():
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    response = ask_more(client, inspection["id"], "  How heavy is it?  ")
    assert response.status_code == 200
    assert response.get_json() == {"text": "Answer 1", "used_web": True, "sources": ["https://example.org/w"]}
    assert assistant.follow_ups == [(PHOTO, "laptop", STORED_LAPTOP_INFO, (), "How heavy is it?")]


def test_follow_up_sends_the_earlier_questions_and_answers():
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    ask_more(client, inspection["id"], "How heavy is it?")
    ask_more(client, inspection["id"], "And the battery?")
    assert assistant.follow_ups[1] == (PHOTO, "laptop", STORED_LAPTOP_INFO,
                                       (("How heavy is it?", "Answer 1"),), "And the battery?")


def test_follow_up_uses_the_description_from_the_cache():
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspect_then_describe(client)
    second = client.post("/api/inspect", json={"track_id": 7, "label": "laptop"}).get_json()
    assert second["info"] is not None                       # shown at once, no /api/describe call
    ask_more(client, second["id"])
    assert assistant.follow_ups[0][2] == STORED_LAPTOP_INFO
    assert len(assistant.looked_up) == 1


def test_follow_up_after_describe_answered_from_the_cache_has_the_description():
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspect_then_describe(client)
    _second, described = inspect_then_describe(client)       # /api/describe answers from the cache
    assert described.get_json()["name"] == "Laptop"
    ask_more(client, _second["id"])
    assert assistant.follow_ups[0][2] == STORED_LAPTOP_INFO


def test_follow_up_after_a_failed_description_sends_no_description():
    assistant = FakeAssistant(describe_error=AssistantError("No internet."))
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    assert ask_more(client, inspection["id"]).status_code == 200
    assert assistant.follow_ups[0][2] is None


def test_follow_up_about_an_area_sends_no_label():
    assistant = FakeAssistant(info=FakeInfo("Plant", "A green plant."))
    client = client_for(assistant, FakePipeline(target=AREA))
    inspection, _described = inspect_then_describe(client)
    ask_more(client, inspection["id"])
    image, label, info, earlier, _question = assistant.follow_ups[0]
    assert (image, label, info.name, earlier) == (PHOTO, None, "Plant", ())


def test_follow_up_about_a_recognised_person_goes_to_ask_with_a_naming_sentence():
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=SARA_TARGET))
    inspection = client.post("/api/inspect", json={"x": 0.2, "y": 0.3}).get_json()
    response = ask_more(client, inspection["id"], "What did she build?")
    assert response.status_code == 200
    assert response.get_json()["text"] == "Rover follows a black line."
    assert assistant.questions == ["This question is about Sara Ahmed, who the camera recognises. "
                                   "What did she build?"]
    assert assistant.follow_ups == [] and assistant.looked_up == []        # no picture of her anywhere


def test_follow_up_about_a_recognised_person_keeps_the_conversation_for_the_limit():
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=SARA_TARGET))
    inspection = client.post("/api/inspect", json={"x": 0.2, "y": 0.3}).get_json()
    for n in range(10):
        assert ask_more(client, inspection["id"], f"Question {n}").status_code == 200
    assert ask_more(client, inspection["id"]).status_code == 400
    assert len(assistant.questions) == 10


def test_a_person_question_is_checked_with_the_naming_sentence_added():
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=SARA_TARGET))
    inspection = client.post("/api/inspect", json={"x": 0.2, "y": 0.3}).get_json()
    sentence = "This question is about Sara Ahmed, who the camera recognises. "
    limit = 2000 - len(sentence)
    too_long = ask_more(client, inspection["id"], "a" * (limit + 1))
    assert too_long.status_code == 400
    assert too_long.get_json() == {"error": f"The question is too long ({limit + 1} characters). "
                                            f"Questions about a person can have at most {limit} characters."}
    assert assistant.questions == []                        # refused before the assistant saw it
    assert ask_more(client, inspection["id"], "a" * limit).status_code == 200
    assert len(assistant.questions[0]) == 2000


def test_an_unrecognised_person_has_no_follow_up():
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=STRANGER))
    inspection = client.post("/api/inspect", json={"x": 0.2, "y": 0.3}).get_json()
    response = ask_more(client, inspection["id"])
    assert response.status_code == 400
    assert response.get_json() == {"error": app_module.UNKNOWN_PERSON}
    assert assistant.questions == [] and assistant.follow_ups == []


@pytest.mark.parametrize("question, message", [("", "empty"), ("   ", "empty"), ("a" * 2001, "too long")])
def test_follow_up_with_an_empty_or_over_long_question_is_400(question, message):
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    response = ask_more(client, inspection["id"], question)
    assert response.status_code == 400
    assert message in response.get_json()["error"]
    assert assistant.follow_ups == []
    assert ask_more(client, inspection["id"], "a" * 2000).status_code == 200


def test_follow_up_with_an_unknown_id_is_404():
    assistant = FakeAssistant()
    response = ask_more(client_for(assistant), "not-a-real-id")
    assert response.status_code == 404
    assert "click" in response.get_json()["error"]
    assert assistant.follow_ups == []


@pytest.mark.parametrize("body, content_type", [
    ("id=x&question=y", "application/x-www-form-urlencoded"),
    ("not json", "application/json"),
    ('["x", "y"]', "application/json"),
    ('{"question": "How heavy?"}', "application/json"),
    ('{"id": "x"}', "application/json"),
    ('{"id": 5, "question": "How heavy?"}', "application/json"),
    ('{"id": "x", "question": 42}', "application/json"),
])
def test_follow_up_with_a_bad_body_is_400(body, content_type):
    assistant = FakeAssistant()
    response = client_for(assistant).post("/api/follow_up", data=body, content_type=content_type)
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)
    assert assistant.follow_ups == []


def test_follow_up_allows_ten_questions_then_400():
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    for n in range(10):
        assert ask_more(client, inspection["id"], f"Question {n}").status_code == 200
    response = ask_more(client, inspection["id"], "One more?")
    assert response.status_code == 400
    assert response.get_json() == {"error": app_module.TOO_MANY_FOLLOW_UPS}
    assert len(assistant.follow_ups) == 10
    assert len(assistant.follow_ups[-1][3]) == 9                # the tenth carried the nine before it


@pytest.mark.parametrize("error, status", [(AssistantError("The assistant is still busy."), 200),
                                           (ValueError("The question is too long."), 400),
                                           (KeyError("secret detail"), 500)])
def test_a_failed_follow_up_is_not_added_to_the_conversation(error, status):
    assistant = FakeAssistant(follow_up_errors=[None, error, None])
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    ask_more(client, inspection["id"], "First?")
    failed = ask_more(client, inspection["id"], "Second?")
    assert failed.status_code == status
    assert isinstance(failed.get_json()["error"], str) and "secret detail" not in failed.get_json()["error"]
    ask_more(client, inspection["id"], "Third?")
    assert assistant.follow_ups[2][3] == (("First?", "Answer 1"),)   # "Second?" is not in it


def test_follow_up_explains_when_the_assistant_could_not_be_built():
    client = client_for(UnavailableAssistant("The assistant is not available: no notes file"))
    inspection, _described = inspect_then_describe(client)
    response = ask_more(client, inspection["id"])
    assert response.status_code == 200
    assert "no notes file" in response.get_json()["error"]


def test_follow_up_logs_no_person_details(caplog):
    caplog.set_level(logging.DEBUG)
    client = client_for(FakeAssistant(), FakePipeline(target=SARA_TARGET))
    inspection = client.post("/api/inspect", json={"x": 0.2, "y": 0.3}).get_json()
    ask_more(client, inspection["id"], "What did she build?")
    assert "Sara" not in caplog.text and "Team lead" not in caplog.text


def test_inspection_store_keeps_the_description_and_the_pairs():
    store = InspectionStore()
    inspection_id = store.add(Inspection("object", "laptop", 7, PHOTO))
    store.set_info(inspection_id, STORED_LAPTOP_INFO)
    store.add_pair(inspection_id, "How heavy?", "About 2 kg.")
    kept = store.get(inspection_id)
    assert (kept.info, kept.pairs) == (STORED_LAPTOP_INFO, (("How heavy?", "About 2 kg."),))
    store.set_info("made-up", STORED_LAPTOP_INFO)               # an expired id changes nothing
    store.add_pair("made-up", "a", "b")
    assert store.get("made-up") is None


# ---- /api/follow_up with a marked part ------------------------------------------------------

STROKES = [{"points": [[0.2, 0.3], [0.25, 0.35], [0.3, 0.4]]}, {"points": [[1, 0]]}]


class FakeMarking:
    """Stands in for marking.mark_photo and remembers what it was given."""

    def __init__(self):
        self.calls = []
        self.limits = []                    # the max_bytes of every call

    def __call__(self, photo, strokes, quality=85, max_bytes=None):
        self.calls.append((photo, strokes))
        self.limits.append(max_bytes)
        return b"marked copy"


@pytest.fixture
def fake_marking(monkeypatch):
    marking = FakeMarking()
    monkeypatch.setattr(app_module, "mark_photo", marking)
    return marking


def ask_about_a_part(client, inspection_id, strokes=STROKES, question="What is this part?"):
    return client.post("/api/follow_up", json={"id": inspection_id, "question": question, "strokes": strokes})


def test_strokes_are_drawn_on_the_stored_photo_and_the_assistant_gets_the_marked_copy(fake_marking):
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    response = ask_about_a_part(client, inspection["id"])
    assert response.status_code == 200
    assert response.get_json()["text"] == "Answer 1"
    assert fake_marking.calls == [(PHOTO, [[(0.2, 0.3), (0.25, 0.35), (0.3, 0.4)], [(1.0, 0.0)]])]
    assert assistant.follow_ups == [(PHOTO, "laptop", STORED_LAPTOP_INFO, (), "What is this part?")]
    assert assistant.marked == [b"marked copy"]                 # the original photo still goes first
    assert fake_marking.limits == [app_module.MAX_IMAGE_BYTES]  # never larger than the assistant accepts


@pytest.mark.parametrize("strokes", [None, []])
def test_null_or_an_empty_list_of_strokes_is_an_ordinary_question(fake_marking, strokes):
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    assert ask_about_a_part(client, inspection["id"], strokes).status_code == 200
    ask_more(client, inspection["id"], "And then?")
    assert assistant.marked == [None, None] and fake_marking.calls == []
    assert assistant.follow_ups[1][3] == (("What is this part?", "Answer 1"),)   # no "marked" note


WHOLE_PHOTO = [{"points": [[x / 10, y / 25] for y in range(26)                  # back and forth, row by row
                           for x in (range(11) if y % 2 == 0 else range(10, -1, -1))]}]
ALL_FOUR_EDGES = [{"points": [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]}]


@pytest.mark.parametrize("strokes", [WHOLE_PHOTO, ALL_FOUR_EDGES])
def test_a_marking_over_nearly_the_whole_photo_is_an_ordinary_question(fake_marking, strokes):
    """Brushed all over, or along all four edges, there is no outline left to see."""
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=Target("object", "laptop", 0.8, 7, BIG_PHOTO)))
    inspection, _described = inspect_then_describe(client)
    assert ask_about_a_part(client, inspection["id"], strokes).status_code == 200
    ask_more(client, inspection["id"], "And then?")
    assert fake_marking.calls == [] and assistant.marked == [None, None]
    assert assistant.follow_ups[1][3] == (("What is this part?", "Answer 1"),)


def test_a_marking_of_a_part_of_the_same_photo_is_still_marked(fake_marking):
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=Target("object", "laptop", 0.8, 7, BIG_PHOTO)))
    inspection, _described = inspect_then_describe(client)
    ask_about_a_part(client, inspection["id"], [{"points": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.3, 0.3]]}])
    assert assistant.marked == [b"marked copy"]


def test_the_real_marking_gives_the_assistant_a_picture_of_the_same_size():
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=AREA))
    inspection, _described = inspect_then_describe(client)
    assert ask_about_a_part(client, inspection["id"]).status_code == 200
    marked = cv2.imdecode(np.frombuffer(assistant.marked[0], np.uint8), cv2.IMREAD_COLOR)
    assert marked.shape == cv2.imdecode(np.frombuffer(PHOTO, np.uint8), cv2.IMREAD_COLOR).shape
    assert assistant.marked[0] != PHOTO
    assert assistant.follow_ups[0][1] is None                   # an area still has no label


def test_without_strokes_no_marked_copy_is_sent(fake_marking):
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    ask_more(client, inspection["id"])
    assert assistant.marked == [None] and fake_marking.calls == []


def test_a_marked_question_is_kept_with_a_note_for_the_later_questions(fake_marking):
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    ask_about_a_part(client, inspection["id"], question="What is this part?")
    ask_more(client, inspection["id"], "And how heavy is it?")
    assert assistant.follow_ups[1][3] == (("(about a part the visitor marked) What is this part?", "Answer 1"),)
    assert assistant.marked == [b"marked copy", None]


def test_a_failed_marked_question_is_not_kept(fake_marking):
    assistant = FakeAssistant(follow_up_errors=[AssistantError("The assistant is still busy."), None])
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    assert "error" in ask_about_a_part(client, inspection["id"]).get_json()
    ask_more(client, inspection["id"], "Again?")
    assert assistant.follow_ups[1][3] == ()


def test_the_most_strokes_and_points_and_the_edges_are_accepted(fake_marking):
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    full = [{"points": [[0, 0], [1, 1]] * 200}] * 12             # 12 strokes of 400 points
    assert ask_about_a_part(client, inspection["id"], full).status_code == 200
    assert len(fake_marking.calls[0][1]) == 12 and len(fake_marking.calls[0][1][0]) == 400


@pytest.mark.parametrize("strokes", [
    "not a list",
    {"points": [[0.5, 0.5]]},
    5,
    [{"points": [[0.5, 0.5]]}] * 13,                             # more than 12 strokes
    [{"points": []}],                                             # a stroke with no point
    [{"points": [[0.5, 0.5]] * 401}],                             # more than 400 points
    [[[0.5, 0.5]]],                                               # a stroke that is not {"points": ...}
    [{"dots": [[0.5, 0.5]]}],
    [{"points": "0.5, 0.5"}],
    [{"points": [[0.5]]}],                                        # a point that is not two numbers
    [{"points": [[0.5, 0.5, 0.5]]}],
    [{"points": [{"x": 0.5, "y": 0.5}]}],
    [{"points": [["0.5", 0.5]]}],
    [{"points": [[True, 0.5]]}],
    [{"points": [[None, 0.5]]}],
    [{"points": [[1.01, 0.5]]}],                                  # outside 0 to 1
    [{"points": [[0.5, -0.01]]}],
    [{"points": [[0.5, 0.5]]}, {"points": [[2, 0.5]]}],          # one bad stroke among good ones
])
def test_bad_strokes_are_refused_with_a_plain_message(fake_marking, strokes):
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    response = ask_about_a_part(client, inspection["id"], strokes)
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)
    assert assistant.follow_ups == [] and fake_marking.calls == []


@pytest.mark.parametrize("raw", [
    '[{"points": [[NaN, 0.5]]}]',
    '[{"points": [[0.5, Infinity]]}]',
    '[{"points": [[1' + "0" * 400 + ', 0.5]]}]',                 # too big for a float
])
def test_strokes_that_are_not_finite_numbers_are_refused(fake_marking, raw):
    assistant = FakeAssistant()
    client = client_for(assistant)
    inspection, _described = inspect_then_describe(client)
    body = '{"id": "%s", "question": "What is this part?", "strokes": %s}' % (inspection["id"], raw)
    response = client.post("/api/follow_up", data=body, content_type="application/json")
    assert response.status_code == 400
    assert assistant.follow_ups == [] and fake_marking.calls == []


@pytest.mark.parametrize("target", [SARA_TARGET, STRANGER])
def test_strokes_for_a_person_are_refused_and_nothing_is_sent(fake_marking, target):
    assistant = FakeAssistant()
    client = client_for(assistant, FakePipeline(target=target))
    inspection = client.post("/api/inspect", json={"x": 0.2, "y": 0.3}).get_json()
    response = ask_about_a_part(client, inspection["id"])
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)
    assert assistant.questions == [] and assistant.follow_ups == [] and fake_marking.calls == []


def test_a_marking_with_an_unknown_id_is_404(fake_marking):
    response = ask_about_a_part(client_for(), "not-a-real-id")
    assert response.status_code == 404
    assert fake_marking.calls == []


def test_unavailable_assistant_follow_up_takes_a_marked_copy():
    with pytest.raises(AssistantError, match="not available"):
        UnavailableAssistant("not available").follow_up(PHOTO, "laptop", None, (), "What?", marked=PHOTO)


# ---- /api/camera -------------------------------------------------------------------------


def camera_client(tmp_path, pipeline=None):
    pipeline = pipeline or FakePipeline()
    app = create_app(pipeline, FakeAssistant(), camera_file=tmp_path / "data" / "camera.json")
    app.testing = True
    return app.test_client(), pipeline, tmp_path / "data" / "camera.json"


def test_camera_switches_to_the_phone_and_remembers_the_address(tmp_path):
    client, pipeline, camera_file = camera_client(tmp_path)
    response = client.post("/api/camera", json={"address": "  192.168.1.7:8080 "})
    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "url": "http://192.168.1.7:8080/video"}
    assert pipeline.phones == ["http://192.168.1.7:8080/video"]
    assert json.loads(camera_file.read_text(encoding="utf-8")) == {"phone_url": "http://192.168.1.7:8080/video"}


@pytest.mark.parametrize("address, message", [
    ("", phone_module.EMPTY),
    ("phone.local", phone_module.NOT_IPV4),
    ("::1", phone_module.NOT_IPV4),
    ("8.8.8.8", phone_module.NOT_LOCAL),
    ("192.168.1.7:70000", phone_module.BAD_PORT),
    ("https://192.168.1.7:8080", phone_module.NO_HTTPS),
    ("http://me:pw@192.168.1.7:8080", phone_module.NO_LOGIN),
    ("192.168.1.7:8080/shot.jpg", phone_module.BAD_PATH),
])
def test_camera_with_a_bad_address_is_400_with_its_message(tmp_path, address, message):
    client, pipeline, camera_file = camera_client(tmp_path)
    response = client.post("/api/camera", json={"address": address})
    assert response.status_code == 400
    assert response.get_json() == {"error": message}
    assert pipeline.phones == [] and not camera_file.exists()


@pytest.mark.parametrize("body, content_type", [
    ("address=192.168.1.7", "application/x-www-form-urlencoded"),
    ("not json", "application/json"),
    ('["192.168.1.7"]', "application/json"),
    ('{"url": "192.168.1.7"}', "application/json"),
    ('{"address": 192}', "application/json"),
    ('{"address": null}', "application/json"),
])
def test_camera_with_a_body_that_is_not_a_json_address_is_400(tmp_path, body, content_type):
    client, pipeline, camera_file = camera_client(tmp_path)
    response = client.post("/api/camera", data=body, content_type=content_type)
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)
    assert pipeline.phones == [] and not camera_file.exists()


def test_camera_without_a_file_switches_but_remembers_nothing():
    pipeline = FakePipeline()
    response = client_for(pipeline=pipeline).post("/api/camera", json={"address": "10.0.0.5"})
    assert response.get_json() == {"ok": True, "url": "http://10.0.0.5:8080/video"}
    assert pipeline.phones == ["http://10.0.0.5:8080/video"]


def test_camera_still_switches_when_the_address_cannot_be_written(tmp_path):
    pipeline = FakePipeline()
    folder = tmp_path / "camera.json"
    folder.mkdir()                                           # a folder where the file should go
    app = create_app(pipeline, FakeAssistant(), camera_file=folder)
    response = app.test_client().post("/api/camera", json={"address": "192.168.1.7"})
    assert response.status_code == 200 and response.get_json()["ok"] is True
    assert pipeline.phones == ["http://192.168.1.7:8080/video"]


# ---- /api/camera/picture ------------------------------------------------------------------


@pytest.mark.parametrize("choice", ["auto", "keep", "640x360", "1920x1080"])
def test_camera_picture_passes_the_choice_on_and_answers_at_once(choice):
    pipeline = FakePipeline()
    response = client_for(pipeline=pipeline).post("/api/camera/picture", json={"choice": choice})
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert pipeline.pictures == [choice]


@pytest.mark.parametrize("choice", ["800x600", "huge", "", "http://8.8.8.8/"])
def test_camera_picture_refuses_what_the_pipeline_refuses(choice):
    pipeline = FakePipeline()
    response = client_for(pipeline=pipeline).post("/api/camera/picture", json={"choice": choice})
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)
    assert pipeline.pictures == []


@pytest.mark.parametrize("body, content_type", [
    ("choice=auto", "application/x-www-form-urlencoded"),
    ("not json", "application/json"),
    ('["auto"]', "application/json"),
    ('{"size": "auto"}', "application/json"),
    ('{"choice": 640}', "application/json"),
    ('{"choice": null}', "application/json"),
    ('{"choice": ["auto"]}', "application/json"),
])
def test_camera_picture_with_a_body_that_is_not_a_json_choice_is_400(body, content_type):
    pipeline = FakePipeline()
    response = client_for(pipeline=pipeline).post("/api/camera/picture", data=body, content_type=content_type)
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)
    assert pipeline.pictures == []


def test_camera_picture_shows_the_pipelines_own_message():
    response = client_for().post("/api/camera/picture", json={"choice": "800x600"})
    assert response.get_json() == {"error": "The phone does not offer 800x600."}


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


def test_stream_sends_at_most_30_frames_a_second():
    assert app_module.VIDEO_MAX_FPS == 30                 # the phone sends 30 small pictures a second
    clock = FakeClock()
    stream = mjpeg_stream(ChangingPipeline(), clock=clock, sleep=clock.sleep)
    sent_at = []
    while clock.now < 2.0:
        next(stream)
        sent_at.append(clock.now)
    gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
    assert min(gaps) >= 1 / 30 - 1e-9
    assert len(sent_at) <= 2.0 * 30 + 1


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


class FrozenPipeline:
    """The same picture object forever: a stream that has stalled."""

    def __init__(self):
        self.picture = b"the last picture"

    def latest_jpeg(self):
        return self.picture


def test_a_stalled_stream_sends_the_last_frame_again_about_once_a_second():
    clock = FakeClock()
    pipeline = FrozenPipeline()
    stream = mjpeg_stream(pipeline, clock=clock, sleep=clock.sleep)
    sent_at = []
    for _ in range(4):
        assert next(stream) == jpeg_part(pipeline.picture)
        sent_at.append(clock.now)
    gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
    assert all(1.0 - 1e-9 <= gap <= 1.0 + 0.02 for gap in gaps), gaps


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
    monkeypatch.setattr(app_module, "CAMERA_FILE", str(tmp_path / "camera.json"))   # never the real one
    monkeypatch.setattr(app_module, "already_serving", lambda host, port: False)   # 5000 may really be in use
    monkeypatch.setattr(app_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(app_module, "Assistant", fake_assistant)
    monkeypatch.setattr(app_module.Flask, "run", fake_run)
    monkeypatch.setattr(app_module.webbrowser, "open", lambda url: calls["browser"].append(url))
    monkeypatch.setattr(app_module, "BROWSER_DELAY_SECONDS", 0.0)

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    werkzeug_level = logging.getLogger("werkzeug").level
    yield {"calls": calls, "options": options, "log_file": tmp_path / "logs" / "app.log",
           "camera_file": tmp_path / "camera.json"}
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


@pytest.mark.parametrize("argv", [[], ["--source", "file"], ["--source", "phone"], ["--file", "clip.mp4"]])
def test_a_remembered_phone_address_replaces_the_one_in_the_settings(fake_main, argv):
    fake_main["camera_file"].write_text('{"phone_url": "http://192.168.1.9:4747/video"}', encoding="utf-8")
    assert main(argv) == 0
    stream = FakePipeline.instances[0].config["stream"]
    assert stream["phone_url"] == "http://192.168.1.9:4747/video"
    assert "phone_url" not in TEST_CONFIG["stream"]                         # settings not changed


def test_without_a_remembered_address_the_settings_are_used(fake_main):
    assert main([]) == 0
    assert FakePipeline.instances[0].config["stream"] == TEST_CONFIG["stream"]


def test_a_broken_remembered_file_is_ignored(fake_main):
    fake_main["camera_file"].write_text("{broken", encoding="utf-8")
    assert main([]) == 0
    assert "phone_url" not in FakePipeline.instances[0].config["stream"]


LOGIN_CONFIG = {"stream": {"source": "webcam", "phone_url": "http://me:secret@192.168.1.9:4747/video"}}


@pytest.mark.parametrize("remembered, kept", [
    ("http://192.168.1.9:4747/video", "http://me:secret@192.168.1.9:4747/video"),   # same phone: keep the login
    ("http://192.168.1.9:8080/video", "http://192.168.1.9:8080/video"),             # another port: another phone
    ("http://192.168.1.10:4747/video", "http://192.168.1.10:4747/video"),
    (None, "http://me:secret@192.168.1.9:4747/video"),
])
def test_a_configured_login_is_kept_for_the_same_phone(remembered, kept):
    config = app_module.with_remembered_phone(LOGIN_CONFIG, remembered)
    assert config["stream"]["phone_url"] == kept
    assert LOGIN_CONFIG["stream"]["phone_url"] == "http://me:secret@192.168.1.9:4747/video"   # not changed


def test_main_keeps_the_configured_login_when_the_remembered_phone_is_the_same(fake_main, monkeypatch):
    config = {**TEST_CONFIG, "stream": {**TEST_CONFIG["stream"], **LOGIN_CONFIG["stream"]},
              "app": {**TEST_CONFIG["app"], "log_file": str(fake_main["log_file"])}}
    monkeypatch.setattr(app_module, "load_config", lambda: config)
    fake_main["camera_file"].write_text('{"phone_url": "http://192.168.1.9:4747/video"}', encoding="utf-8")
    assert main([]) == 0
    assert FakePipeline.instances[0].config["stream"]["phone_url"] == "http://me:secret@192.168.1.9:4747/video"


def test_main_gives_the_page_the_camera_file(fake_main):
    main([])
    (app, _kwargs), = fake_main["calls"]["run"]
    response = app.test_client().post("/api/camera", json={"address": "192.168.1.7:8080"})
    assert response.status_code == 200
    assert FakePipeline.instances[0].phones == ["http://192.168.1.7:8080/video"]
    assert "192.168.1.7:8080" in fake_main["camera_file"].read_text(encoding="utf-8")


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


REAL_ALREADY_SERVING = app_module.already_serving


def test_a_second_copy_stops_before_loading_anything_when_the_port_answers(fake_main, monkeypatch, capsys):
    with socket.socket() as listener:                     # a real server socket, as a running copy has
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        monkeypatch.setattr(app_module, "already_serving", REAL_ALREADY_SERVING)
        monkeypatch.setattr(app_module, "load_config",
                            lambda: {**TEST_CONFIG, "app": {**TEST_CONFIG["app"], "port": port}})
        assert main([]) == 1
    out = capsys.readouterr().out
    assert "already running" in out and f"http://127.0.0.1:{port}/" in out
    assert FakePipeline.instances == []                   # no pipeline, so no model was loaded
    assert fake_main["calls"]["run"] == []


def test_already_serving_is_false_when_nothing_listens():
    with socket.socket() as probe:                        # find a free port, then let it go
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert REAL_ALREADY_SERVING("127.0.0.1", port, timeout=0.2) is False


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


def test_template_has_no_animation_that_runs_all_the_time_behind_the_glass():
    """Shapes that drifted without end made the browser redraw every glass surface on
    every frame; on the demo laptop that cut the models from 22 to about 5 frames a second."""
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "animation-timeline: scroll(root)" in page        # the shapes follow the scroll position
    endless = [line.strip() for line in page.splitlines() if "infinite" in line and "animation" in line]
    # Only two small things may loop: the status dot, and the waiting dots while an answer is on its way.
    assert all("ping" in line or "blink" in line for line in endless), endless


def test_template_only_makes_links_for_http_and_https():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert 'url.protocol !== "http:" && url.protocol !== "https:"' in page
    assert "textContent" in page


def test_template_has_the_list_the_panel_and_the_privacy_line():
    page = client_for().get("/").get_data(as_text=True)
    assert 'id="things"' in page and "In view" in page
    assert 'id="details"' in page and 'id="closeDetails"' in page and '"Escape"' in page
    privacy = ("A click on an object or on an empty spot sends a small cut-out of the picture to the "
               "assistant's API. A click on a person sends nothing.")
    assert page.count(privacy) == 2                                 # under the list and in the footer
    assert "Add a folder with photos and an about.md for them in data/people" in page
    assert "Something the detector has no name for" in page
    assert "/api/inspect" in page and "/api/describe" in page


def test_template_shows_a_photo_only_from_a_jpeg_data_address():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert 'PHOTO_PREFIX = "data:image/jpeg;base64,"' in page
    assert "address.startsWith(PHOTO_PREFIX)" in page
    assert page.count("detailPhoto.src =") == 1                     # set in one place, after that check


def css_rule(page, selector):
    """The body of the first CSS rule that starts with exactly this selector."""
    match = re.search(r"^\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", page, re.MULTILINE)
    assert match, selector
    return match.group(1)


def test_template_keeps_blur_off_the_pills_and_off_the_video():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "backdrop-filter" not in css_rule(page, ".thing")        # many small pills: no blur
    assert 'make("button", "thing")' in page                       # and not the blurring .glass class
    assert "backdrop-filter: none" in css_rule(page, ".nav.over-video")
    narrow = page[page.index("@media (max-width: 1279.98px)"):]
    assert "backdrop-filter: none" in css_rule(narrow, ".details")  # the sheet over the video is solid


def test_template_status_pill_is_quiet_and_knows_every_state():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert '<div class="status-pill" id="status" aria-live="off">' in page
    for text in ('"Clip finished"', '"Stopped: "', '"Reconnecting"', '"Program not running"'):
        assert text in page


def test_template_clears_the_chat_only_when_the_server_said_ok():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "data.ok !== true" in page


def test_template_measures_the_click_against_the_picture_not_the_element():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "naturalWidth" in page and "naturalHeight" in page and "getBoundingClientRect" in page
    assert "MAX_SOURCES = 3" in page and '" more"' in page


VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class IdTree(HTMLParser):
    """For every element with an id, the ids of the elements around it."""

    def __init__(self):
        super().__init__()
        self.open = []                      # (tag, id) of the elements not closed yet
        self.around = {}

    def handle_starttag(self, tag, attrs):
        element_id = dict(attrs).get("id")
        if element_id:
            self.around[element_id] = [i for _tag, i in self.open if i]
        if tag not in VOID_TAGS:
            self.open.append((tag, element_id))

    def handle_endtag(self, tag):
        for index in range(len(self.open) - 1, -1, -1):
            if self.open[index][0] == tag:
                del self.open[index:]
                return


def ids_around(page):
    tree = IdTree()
    tree.feed(page)
    return tree.around


def rendered_page():
    return client_for().get("/").get_data(as_text=True)


def test_template_has_the_theme_button_and_sets_the_look_in_the_head():
    page = rendered_page()
    head = page[:page.index("<body")]
    assert 'id="themeButton"' in page and 'aria-label="Switch to the dark look"' in page
    script = head[head.index("<script>"):head.index("</script>")]
    assert head.index("<script>") < head.index("<style>")          # set before anything is drawn
    assert 'localStorage.getItem("theme")' in script and "try" in script and "catch" in script
    assert "prefers-color-scheme: dark" in script and "dataset.theme" in script
    assert ':root[data-theme="dark"]' in head and ':root[data-theme="light"]' in head
    assert "window.localStorage.setItem(THEME_KEY, theme)" in page   # the choice is kept


def test_template_theme_change_fades_only_colours():
    page = TEMPLATE.read_text(encoding="utf-8")
    fade = css_rule(page, ".theme-fade, .theme-fade *, .theme-fade *::before, .theme-fade *::after")
    assert "background-color" in fade and "color" in fade
    for line in page.splitlines():
        if "transition" in line:
            assert "filter" not in line and "blur" not in line, line     # never a blur animation


def test_template_has_the_camera_box_and_the_delay_switch():
    page = rendered_page()
    assert 'id="cameraButton"' in page and 'aria-controls="cameraBox"' in page
    assert 'id="cameraBox"' in page and '<label class="field-label" for="cameraAddress">Phone address</label>' in page
    assert 'id="cameraConnect" type="submit">Connect</button>' in page
    assert "The address IP Webcam shows on the phone's screen, like 192.168.1.7:8080." in page
    assert 'id="cameraResult"' in page and '"Connecting ..."' in page and '"Connected"' in page
    assert "No picture from " in page and "on the same Wi-Fi." in page and "CONNECT_WAIT_MS = 8000" in page
    assert '"/api/camera"' in page
    assert '<input type="checkbox" id="delaySwitch" role="switch">' in page and "Delay test" in page
    assert "backdrop-filter: none" in css_rule(page, ".glass.camera-box")   # it may lie over the video


def test_template_runs_the_clock_only_while_the_delay_test_is_on():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="delay" hidden' in page                              # off by default
    assert page.count("requestAnimationFrame(tickClock)") == 1      # the only loop, started by tickClock
    assert "cancelAnimationFrame(clockRequest)" in page
    assert "setInterval" not in page
    assert "font-variant-numeric: tabular-nums" in css_rule(page, ".delay-clock")


def test_template_full_screen_wraps_video_banners_pills_and_panel():
    page = rendered_page()
    around = ids_around(page)
    for inside in ("video", "warning", "greetings", "things", "details", "followQuestion", "fullButton", "delay"):
        assert "watch" in around[inside], inside
    assert 'id="fullButton" type="button" aria-label="Full screen"' in page
    assert "stage.requestFullscreen()" in page and 'stage = byId("watch")' in page
    assert "dblclick" not in page                                   # a click is already "inspect"
    assert 'event.key === "f"' in page and "isTyping(event.target)" in page


def test_template_full_screen_panel_has_its_own_column_and_no_blur():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "grid-template-columns" in css_rule(page, ".stage.is-full.has-details")
    panel = css_rule(page, ".stage.is-full .details")
    assert "backdrop-filter: none" in panel and "var(--glass-solid)" in panel
    assert "transform: none" in css_rule(page, ".stage.is-full .shot")    # never left half revealed
    assert 'if (stage.classList.contains("is-full")) return false;' in page   # bars: measure the click


def test_template_scrolls_only_the_panel_and_the_chat_list_never_the_page():
    """Seen live: sending a follow-up question scrolled the whole page and slid the video
    behind the bar. Focus never scrolls, and only the panel's or the list's own area moves."""
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "scrollIntoView" not in page and "scrollTo(" not in page and "scrollBy(" not in page
    calls = re.findall(r"\.focus\(([^)]*)\)", page)
    assert calls and all(call.strip() == "{ preventScroll: true }" for call in calls), calls
    assert "details.scrollTop = details.scrollHeight" in page
    assert "messages.scrollTop = messages.scrollHeight" in page
    # The panel beside the video is never taller than the window below its own top, so the
    # follow-up field is always in view and typing in it never makes the browser scroll the page.
    assert "function fitDetails()" in page
    assert page.count("fitDetails();") >= 3                        # on scroll, on opening, on full screen


def test_template_has_the_phone_picture_row():
    page = rendered_page()
    around = ids_around(page)
    assert "cameraBox" in around["pictureRow"] and "cameraBox" in around["pictureChoice"]
    assert '<div class="picture-row" id="pictureRow" hidden>' in page          # hidden until the phone says
    assert '<label class="field-label" for="pictureChoice">Picture from the phone</label>' in page
    assert '<option value="auto">Automatic (smallest that is sharp enough)</option>' in page
    assert '<option value="keep">Leave the phone as it is</option>' in page
    assert '<span id="pictureHint">Smaller pictures arrive sooner.</span>' in page and '" The page shows "' in page
    assert 'id="pictureNow"' in page and '"The phone sends "' in page
    assert '"/api/camera/picture", { choice }' in page                       # only the choice is sent
    assert 'pictureRow.hidden = picture === null;' in page
    assert 'pictureChoice.addEventListener("change"' in page                 # posts as soon as it changes
    assert "showCameraResult(errorText(data, response" in page                 # errors in the result line
    assert ".picture-line select:focus-visible" in page                        # a visible ring for the keyboard
    select = css_rule(page, ".picture-line select")
    assert "var(--surface)" in select and "var(--text)" in select              # follows the light and dark look
    assert "backdrop-filter" not in select


def test_template_has_the_follow_up_chat():
    page = rendered_page()
    assert '<label class="visually-hidden" for="followQuestion">Ask about this</label>' in page
    assert '<input id="followQuestion" type="text" maxlength="2000" placeholder="Ask about this">' in page
    assert "const body = { id, question };" in page                   # the id and the question ...
    assert "if (strokes) body.strokes = strokes;" in page              # ... and the points of a marking
    assert 'postJson("/api/follow_up", body)' in page
    assert "MAX_FOLLOW_UPS = 10" in page
    assert "Follow-up questions about an object send the same cut-out again." in page


def test_template_picture_help_says_what_the_choice_costs():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert '"Sharper, but a little later and slower."' in page
    assert '"Smaller pictures arrive sooner."' in page
    assert 'pictureChoice.value !== "auto" && pictureChoice.value !== "keep"' in page
    assert '" × " + video.naturalHeight + "."' in page                 # the real size of the video's pictures
    assert "either way" not in page


def test_template_full_view_never_depends_on_the_fullscreen_api():
    """The owner's browser pane neither grants nor refuses requestFullscreen(): the button
    must still fill the window, and it is never hidden."""
    page = rendered_page()
    assert 'id="fullButton" type="button" aria-label="Full screen" title="Full screen (F)">' in page   # not hidden
    script = page[page.index("function enterFullView()"):page.index("function leaveFullView()")]
    assert script.index('stage.classList.add("is-full")') < script.index("stage.requestFullscreen()")
    assert 'document.documentElement.classList.add("full-view")' in script
    assert "stage.requestFullscreen().catch(() => {})" in script     # refused: the view stays
    assert "fullButton.hidden" not in page
    assert "overflow: hidden" in css_rule(page, "html.full-view, html.full-view body")
    full = css_rule(page, ".stage.is-full")
    assert "position: fixed" in full and "inset: 0" in full and "background: #000" in full


def test_template_full_view_makes_the_page_behind_inert_and_puts_it_back():
    """Without real full screen, Tab could reach the Camera button behind the stage."""
    page = TEMPLATE.read_text(encoding="utf-8")
    inert = page[page.index("function setPageBehindInert()"):page.index("function enterFullView()")]
    assert "other.inert = true;" in inert and "madeInert.push(other)" in inert
    assert "other.inert = false;" in inert and "madeInert = [];" in inert
    assert "if (other === node || other.inert" in inert          # never the stage, never what was inert already
    enter = page[page.index("function enterFullView()"):page.index("function leaveFullView()")]
    leave = page[page.index("function leaveFullView()"):page.index("function toggleFullView()")]
    assert "setPageBehindInert();" in enter and "putPageBehindBack();" in leave
    # Every way out (button, F, Escape, the browser leaving full screen) goes through leaveFullView.
    assert page.count("putPageBehindBack();") == 1
    assert "\n  fullView = false;" in leave and page.count("\n  fullView = false;") == 1   # the only place


def test_template_sends_picture_choices_one_after_the_other():
    page = TEMPLATE.read_text(encoding="utf-8")
    change = page[page.index('pictureChoice.addEventListener("change"'):page.index("// ---- the delay test")]
    assert "if (pictureSending) pictureNext = pictureChoice.value;" in change   # only the newest waits
    assert "else sendPicture(pictureChoice.value);" in change
    assert "pictureSending = true;" in change and "pictureSending = false;" in change
    assert "sendPicture(next);" in change                                  # sent when the first is answered


def test_template_escape_closes_the_camera_box_then_the_full_view_then_the_panel():
    page = TEMPLATE.read_text(encoding="utf-8")
    body = page[page.index("function closeTopmost()"):page.index('document.addEventListener("keydown"')]
    order = [body.index("closeCameraBox(true)"), body.index("leaveFullView()"), body.index("closeDetails()")]
    assert order == sorted(order)
    change = page[page.index('document.addEventListener("fullscreenchange"'):]
    assert "!document.fullscreenElement && fullView" in change[:600]   # the browser's Escape leaves the view too


def test_template_has_the_brush_on_the_photo():
    page = rendered_page()
    around = ids_around(page)
    assert "photoBox" in around["brushCanvas"] and "details" in around["brushCanvas"]
    assert "details" in around["brushButton"] and "details" in around["brushClear"]
    assert 'id="brushButton" type="button" aria-pressed="false"' in page and "Mark a part" in page
    assert 'id="brushClear" type="button" hidden>Clear</button>' in page
    assert 'id="followPart" type="button" hidden>What is this part?</button>' in page
    assert "followForm" in around["followPart"]
    assert '"Ask about the marked part"' in page and '"marked part"' in page
    assert "A marked part is sent as the same cut-out with your outline drawn on it." in page
    assert page.count("A marked part is sent as the same cut-out") == 2   # under the list and in the footer


def test_template_brush_follows_the_rules_of_the_marking():
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "MAX_STROKES = 12" in page and "MAX_STROKE_POINTS = 400" in page
    assert "BRUSH_SHARE = 0.06" in page and "MIN_STEP_SHARE = 0.005" in page
    assert "setPointerCapture(event.pointerId)" in page               # a drag that leaves the photo ends cleanly
    for event in ("pointerdown", "pointermove", "pointerup", "pointercancel", "lostpointercapture"):
        assert f'photoBox.addEventListener("{event}"' in page
    brushing = css_rule(page, ".photo-box.is-brushing, .photo-box.is-brushing .photo")
    assert "touch-action: none" in brushing and "user-select: none" in brushing
    assert "pointer-events: none" in css_rule(page, ".brush-canvas")
    assert "new ResizeObserver(placeBrushCanvas).observe(detailPhoto)" in page   # stays over the picture
    assert "requestAnimationFrame" not in page[page.index("// ---- the brush"):page.index("// ---- asking more")]
    # Never for a person: the tools are shown only for something the API may see.
    assert 'allowBrush(data.describable === true && data.kind !== "person" && !data.person)' in page


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
