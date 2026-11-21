"""The demo page: live video, what is in view, details on a click, reactions and the chat.

Served by Flask; this is the only module that imports flask. A click on the
video goes to /api/inspect, which answers at once with a small photo of what
was clicked; for an object, /api/describe then asks the assistant what it is,
and /api/follow_up answers more questions about that one thing, also about a
part the visitor brushed over (the page sends only the brush strokes; marking.py
draws them on the server's own copy of the photo). The camera
box on the page sends a phone's address to /api/camera, which switches the
video to that phone and remembers the address for the next start, and the
picture size to ask the phone for to /api/camera/picture.
Start it with

    .venv\\Scripts\\python.exe -m car_vision.app
    .venv\\Scripts\\python.exe -m car_vision.app --source webcam --open
    .venv\\Scripts\\python.exe -m car_vision.app --file data/samples/people_clip.mp4

and open the address it prints. Ctrl+C stops it. The page is served on
app.host only (127.0.0.1 in config.yaml), so only this laptop can open it.
"""

from __future__ import annotations

import argparse
import base64
import logging
import math
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from car_vision.assistant import Assistant, AssistantError, ObjectInfo
from car_vision.config import ConfigError, load_config, require, resolve_path, section
from car_vision.marking import mark_photo
from car_vision.phone import load_remembered, phone_url, remember, same_phone
from car_vision.pipeline import Pipeline
from car_vision.stream import SOURCES

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
VIDEO_MAX_FPS = 30               # /video sends at most this many frames a second (the phone sends 30)
VIDEO_POLL_SECONDS = 0.01        # how often /video looks for a new frame
VIDEO_KEEPALIVE_SECONDS = 1.0    # with no new frame for this long, the last one is sent again, so a
                                 # closed browser tab is noticed and its server thread ends
PORT_CHECK_SECONDS = 0.5         # how long main() waits for an answer on the port before starting
BOUNDARY = "frame"               # separates the pictures in the MJPEG stream
BROWSER_DELAY_SECONDS = 1.0      # --open waits this long, so the server is up first
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

MAX_INSPECTIONS = 20             # /api/describe knows the last this many clicks
MAX_ANSWERS = 50                 # answers kept for repeated clicks on the same tracked object
ANSWER_SECONDS = 600.0           # ... for ten minutes
PHOTO_PREFIX = "data:image/jpeg;base64,"
GONE = "That is no longer in view."
CLICK_AGAIN = "That click is too old. Please click the object again."

MAX_STROKES = 12                 # brush strokes on one marked photo
MAX_STROKE_POINTS = 400          # points in one stroke
MARKED_PREFIX = "(about a part the visitor marked) "   # kept in front of a marked question
NO_MARKING_FOR_PEOPLE = "A person's picture is never marked or sent anywhere."

MAX_FOLLOW_UPS = 10              # questions about one clicked thing
TOO_MANY_FOLLOW_UPS = (f"That is {MAX_FOLLOW_UPS} questions about this one already. "
                       "Click it again to start a new conversation about it.")
UNKNOWN_PERSON = "There is nothing to ask about a person the camera does not recognise."
CAMERA_FILE = "data/camera.json" # the last phone address asked for (git-ignored)


class UnavailableAssistant:
    """Stands in when the real assistant could not be built, e.g. the notes file is missing.

    The page keeps working, and every question or look-up gets the reason as an error.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def ask(self, question: str) -> Any:
        raise AssistantError(self.reason)

    def describe(self, image: bytes, label: str | None = None) -> Any:
        raise AssistantError(self.reason)

    def follow_up(self, image: bytes, label: str | None, info: Any, earlier: Any, question: str,
                  marked: bytes | None = None) -> Any:
        raise AssistantError(self.reason)

    def reset(self) -> None:
        pass


# ---- what the server remembers about clicks -------------------------------------------


@dataclass(frozen=True)
class Inspection:
    """One click, kept so /api/describe and /api/follow_up need only its id.

    The browser never sends a picture, a label or a description: they all come from here.
    """

    kind: str                    # "object", "person" or "area"
    label: str
    track_id: int | None
    photo: bytes
    person_name: str | None = None           # the recognised person, for kind "person"
    info: ObjectInfo | None = None           # the description, once /api/describe has answered
    pairs: tuple[tuple[str, str], ...] = ()  # the follow-up questions and answers so far


class InspectionStore:
    """The last few inspections by their random id. Safe to use from many threads."""

    def __init__(self, size: int = MAX_INSPECTIONS) -> None:
        self.size = size
        self._items: OrderedDict[str, Inspection] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, inspection: Inspection) -> str:
        """Keep the inspection and return its new id; the oldest one goes when the store is full."""
        inspection_id = secrets.token_urlsafe(12)
        with self._lock:
            self._items[inspection_id] = inspection
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return inspection_id

    def get(self, inspection_id: str) -> Inspection | None:
        with self._lock:
            return self._items.get(inspection_id)

    def set_info(self, inspection_id: str, info: ObjectInfo) -> None:
        """Keep the description this inspection got. Nothing happens when it has expired."""
        with self._lock:
            inspection = self._items.get(inspection_id)
            if inspection is not None:
                self._items[inspection_id] = replace(inspection, info=info)

    def add_pair(self, inspection_id: str, question: str, answer: str) -> None:
        """Add one answered follow-up question to this inspection's conversation."""
        with self._lock:
            inspection = self._items.get(inspection_id)
            if inspection is not None:
                self._items[inspection_id] = replace(inspection, pairs=inspection.pairs + ((question, answer),))


class AnswerCache:
    """Answers by (track id, label) for a while, so a second click on the same object is free."""

    def __init__(self, size: int = MAX_ANSWERS, seconds: float = ANSWER_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.size = size
        self.seconds = seconds
        self._clock = clock
        self._items: OrderedDict[Hashable, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> dict[str, Any] | None:
        """The answer, or None when there is none or it is too old."""
        with self._lock:
            self._forget_old()
            item = self._items.get(key)
            return item[1] if item is not None else None

    def put(self, key: Hashable, answer: dict[str, Any]) -> None:
        with self._lock:
            self._items.pop(key, None)              # a new answer moves to the back of the line
            self._items[key] = (self._clock(), answer)
            self._forget_old()
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def _forget_old(self) -> None:
        now = self._clock()
        for key, (stored_at, _answer) in list(self._items.items()):
            if now - stored_at >= self.seconds:
                del self._items[key]


class LookupGate:
    """One look-up at a time for each object, so two quick clicks make only one paid API call.

    The second request waits until the first is done, then finds its answer in the cache.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gates: dict[Hashable, list] = {}       # key -> [its lock, how many requests use it]

    @contextmanager
    def one_at_a_time(self, key: Hashable) -> Iterator[None]:
        with self._lock:
            gate = self._gates.setdefault(key, [threading.Lock(), 0])
            gate[1] += 1
        try:
            with gate[0]:
                yield
        finally:
            with self._lock:
                gate[1] -= 1
                if gate[1] == 0:
                    del self._gates[key]             # nobody waits for this object any more


def cache_key(inspection: Inspection) -> tuple[int, str] | None:
    """Only tracked objects are cached: an area or an untracked box may be anything next time."""
    if inspection.kind != "object" or inspection.track_id is None:
        return None
    return inspection.track_id, inspection.label


# ---- turning answers into JSON -------------------------------------------------------------


def info_json(info: Any) -> dict[str, Any]:
    """An ObjectInfo from the assistant as plain JSON values."""
    return {
        "name": str(info.name),
        "summary": str(info.summary),
        "specs": [{"name": str(name), "value": str(value)} for name, value in info.specs],
        "used_web": bool(info.used_web),
        "sources": [str(source) for source in info.sources],
    }


def object_info(answer: dict[str, Any]) -> ObjectInfo:
    """The JSON from info_json back as an ObjectInfo, for the follow-up questions."""
    return ObjectInfo(
        name=answer["name"],
        summary=answer["summary"],
        specs=tuple((spec["name"], spec["value"]) for spec in answer["specs"]),
        used_web=answer["used_web"],
        sources=tuple(answer["sources"]),
    )


def answer_json(answer: Any) -> dict[str, Any]:
    """An Answer from the assistant as plain JSON values."""
    return {"text": str(answer.text), "used_web": bool(answer.used_web),
            "sources": [str(source) for source in answer.sources]}


def person_json(person: Any) -> dict[str, Any] | None:
    """What the panel shows about a recognised person, or None."""
    if person is None:
        return None
    return {
        "id": person.person_id,
        "name": person.name,
        "role": person.role,
        "lines": list(person.lines),
        "story": getattr(person, "story", ""),      # the newer Person has a story
    }


def photo_address(photo: bytes) -> str:
    return PHOTO_PREFIX + base64.b64encode(photo).decode("ascii")


# ---- reading the click ---------------------------------------------------------------------


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_click(body: Any) -> dict[str, Any] | str:
    """The arguments for pipeline.target from an /api/inspect body, or the reason it is wrong."""
    if not isinstance(body, dict):
        return 'Send the click as JSON, like {"x": 0.42, "y": 0.61}.'
    x, y = body.get("x"), body.get("y")
    track_id, label = body.get("track_id"), body.get("label")
    if track_id is not None and (not isinstance(track_id, int) or isinstance(track_id, bool)):
        return "track_id must be a whole number."
    if label is not None and not isinstance(label, str):
        return "label must be text."
    if (x is None) != (y is None):
        return "Send both x and y, or neither."
    if x is None and track_id is None:
        return "Send the point that was clicked (x and y), or a track_id."
    if x is not None:
        if not (is_number(x) and is_number(y)):
            return "x and y must be numbers."
        x, y = as_fraction(x), as_fraction(y)
        if x is None or y is None:
            return "x and y must be from 0 to 1."
    return {"x": x, "y": y, "track_id": track_id, "label": label}


def as_fraction(number: int | float) -> float | None:
    """The number as a float from 0 to 1, or None (also for a whole number too big for a float)."""
    try:
        value = float(number)
    except OverflowError:
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


# ---- reading the brush strokes ---------------------------------------------------------------


STROKES_SHAPE = 'Send the marking as a list of strokes, like [{"points": [[0.4, 0.5], [0.42, 0.52]]}].'


def read_strokes(strokes: Any) -> list[list[tuple[float, float]]] | str:
    """The brush strokes from a /api/follow_up body as lists of (x, y), or the reason they are wrong.

    Each point is two numbers from 0 to 1 across and down the photo. The
    page sends at most MAX_STROKES strokes of at most MAX_STROKE_POINTS points.
    """
    if not isinstance(strokes, list) or not strokes:
        return STROKES_SHAPE
    if len(strokes) > MAX_STROKES:
        return f"At most {MAX_STROKES} strokes can be marked, got {len(strokes)}."
    result = []
    for stroke in strokes:
        points = stroke.get("points") if isinstance(stroke, dict) else None
        if not isinstance(points, list):
            return STROKES_SHAPE
        if not points:
            return "A stroke has no point."
        if len(points) > MAX_STROKE_POINTS:
            return f"A stroke can have at most {MAX_STROKE_POINTS} points, got {len(points)}."
        read = [read_point(point) for point in points]
        if any(point is None for point in read):
            return "Each point must be two numbers from 0 to 1."
        result.append(read)
    return result


def read_point(point: Any) -> tuple[float, float] | None:
    """One point as (x, y) fractions of the photo, or None when it is not two numbers from 0 to 1."""
    if not isinstance(point, list) or len(point) != 2 or not all(is_number(value) for value in point):
        return None
    x, y = as_fraction(point[0]), as_fraction(point[1])
    return None if x is None or y is None else (x, y)


# ---- the web server ------------------------------------------------------------


def question_problem(question: str) -> str | None:
    """Why a typed question (spaces already stripped) cannot be sent, or None when it is fine."""
    if not question:
        return "The question is empty."
    if len(question) > MAX_QUESTION_CHARS:
        return (f"The question is too long ({len(question)} characters). "
                f"Please keep it under {MAX_QUESTION_CHARS} characters.")
    return None


def create_app(pipeline: Any, assistant: Any, camera_file: Path | None = None) -> Flask:
    """The Flask app for one pipeline and one assistant (real ones, or fakes in the tests).

    `camera_file` is where /api/camera remembers the phone's address; with None
    (as in most tests) the address is used but not remembered.
    """
    app = Flask(__name__)
    inspections = InspectionStore()
    answers = AnswerCache()
    gate = LookupGate()

    @app.get("/")
    def page() -> str:
        return render_template("index.html", max_question_chars=MAX_QUESTION_CHARS)

    @app.get("/video")
    def video() -> Response:
        return Response(mjpeg_stream(pipeline),
                        content_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status() -> Response:
        response = jsonify(pipeline.status())
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/ask")
    def ask() -> Any:
        body = request.get_json(silent=True)          # None when the body is not JSON
        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(question, str):
            return jsonify(error='Send the question as JSON, like {"question": "Who built the car?"}'), 400
        question = question.strip()
        problem = question_problem(question)
        if problem is not None:
            return jsonify(error=problem), 400
        try:
            answer = assistant.ask(question)
        except ValueError as error:                   # the assistant's own check of the question
            return jsonify(error=str(error)), 400
        except AssistantError as error:               # no key, no internet, busy, refusal ...
            return jsonify(error=str(error))
        except Exception as error:                    # never show a traceback on the page
            log.exception("Answering a question failed")
            return jsonify(error=f"Something went wrong while answering ({type(error).__name__}). "
                                 "Please ask again."), 500
        return jsonify(answer_json(answer))

    @app.post("/api/reset")
    def reset() -> Any:
        try:
            assistant.reset()
        except AssistantError as error:               # still busy with a question
            return jsonify(error=str(error))
        return jsonify(ok=True)

    @app.post("/api/inspect")
    def inspect_click() -> Any:
        click = read_click(request.get_json(silent=True))
        if isinstance(click, str):
            return jsonify(error=click), 400
        try:
            target = pipeline.target(**click)
        except Exception as error:                    # never show a traceback on the page
            log.exception("Finding what was clicked failed")
            return jsonify(error=f"Something went wrong with that click ({type(error).__name__})."), 500
        if target is None:
            return jsonify(error=GONE), 404
        inspection = Inspection(target.kind, target.label, target.track_id, target.photo,
                                person_name=target.person.name if target.person is not None else None)
        key = cache_key(inspection)
        cached = answers.get(key) if key is not None else None
        if cached is not None:                         # the page shows it at once: it is this click's description
            inspection = replace(inspection, info=object_info(cached))
        return jsonify(
            id=inspections.add(inspection),
            kind=target.kind, label=target.label,
            confidence=round(float(target.confidence), 2),
            track_id=int(target.track_id) if target.track_id is not None else None,
            photo=photo_address(target.photo),
            person=person_json(target.person),
            describable=target.kind != "person",       # a person's picture is never sent to the API
            info=cached,
        )

    @app.post("/api/describe")
    def describe_object() -> Any:
        body = request.get_json(silent=True)
        inspection_id = body.get("id") if isinstance(body, dict) else None
        if not isinstance(inspection_id, str):
            return jsonify(error='Send the inspection id as JSON, like {"id": "kq3..."}.'), 400
        inspection = inspections.get(inspection_id)
        if inspection is None:
            return jsonify(error=CLICK_AGAIN), 404
        if inspection.kind == "person":
            return jsonify(error="People are not looked up: their picture is never sent."), 400
        key = cache_key(inspection)
        if key is None:                               # an area or an untracked box: never cached
            reply, code = look_up(assistant, inspection)
        else:
            with gate.one_at_a_time(key):             # a second click on the same object waits here
                reply, code = answers.get(key), 200
                if reply is None:
                    reply, code = look_up(assistant, inspection)
                    if "error" not in reply:
                        answers.put(key, reply)
        if "error" not in reply:                      # kept for the follow-up questions
            inspections.set_info(inspection_id, object_info(reply))
        return jsonify(reply), code

    @app.post("/api/follow_up")
    def follow_up() -> Any:
        body = request.get_json(silent=True)
        inspection_id = body.get("id") if isinstance(body, dict) else None
        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(inspection_id, str) or not isinstance(question, str):
            return jsonify(error='Send the id and the question as JSON, like '
                                 '{"id": "kq3...", "question": "How heavy is it?"}.'), 400
        strokes = None                                # no "strokes": a question about the whole photo
        if "strokes" in body:
            strokes = read_strokes(body["strokes"])
            if isinstance(strokes, str):
                return jsonify(error=strokes), 400
        question = question.strip()
        problem = question_problem(question)
        if problem is not None:
            return jsonify(error=problem), 400
        # One question at a time for each clicked thing, so its conversation stays in order.
        with gate.one_at_a_time(("follow_up", inspection_id)):
            inspection = inspections.get(inspection_id)
            if inspection is None:
                return jsonify(error=CLICK_AGAIN), 404
            if inspection.kind == "person" and inspection.person_name is None:
                return jsonify(error=UNKNOWN_PERSON), 400
            if inspection.kind == "person" and strokes is not None:
                return jsonify(error=NO_MARKING_FOR_PEOPLE), 400
            if len(inspection.pairs) >= MAX_FOLLOW_UPS:
                return jsonify(error=TOO_MANY_FOLLOW_UPS), 400
            reply, code = answer_follow_up(assistant, inspection, question, strokes)
            if "error" not in reply:                  # a failed question is not kept
                kept = MARKED_PREFIX + question if strokes is not None else question
                inspections.add_pair(inspection_id, kept, reply["text"])
        return jsonify(reply), code

    @app.post("/api/camera")
    def camera() -> Any:
        body = request.get_json(silent=True)
        address = body.get("address") if isinstance(body, dict) else None
        if not isinstance(address, str):
            return jsonify(error='Send the address as JSON, like {"address": "192.168.1.7:8080"}.'), 400
        try:
            url = phone_url(address)
        except ValueError as error:                   # phone_url's own plain message
            return jsonify(error=str(error)), 400
        pipeline.use_phone(url)                       # returns at once; the page watches the status
        if camera_file is not None:
            try:
                remember(camera_file, url)
            except OSError as error:                  # the switch still happened
                log.warning("Could not remember the phone address in %s: %s", camera_file, error)
        return jsonify(ok=True, url=url)

    @app.post("/api/camera/picture")
    def camera_picture() -> Any:
        body = request.get_json(silent=True)
        choice = body.get("choice") if isinstance(body, dict) else None
        if not isinstance(choice, str):
            return jsonify(error='Send the choice as JSON, like {"choice": "auto"}.'), 400
        try:
            pipeline.set_phone_picture(choice)       # returns at once; the phone is asked in the background
        except ValueError as error:                  # not a choice, or a size the phone does not offer
            return jsonify(error=str(error)), 400
        return jsonify(ok=True)

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> Any:
        """An unknown /api/ address or a wrong method gets a JSON answer, not an HTML page."""
        if not request.path.startswith("/api/"):
            return error
        response = jsonify(error=error.description or error.name)
        response.status_code = error.code or 500
        allowed = error.get_response().headers.get("Allow")
        if allowed:
            response.headers["Allow"] = allowed
        return response

    @app.errorhandler(Exception)
    def unexpected_error(error: Exception) -> Any:
        """Anything nobody expected: logged, and a plain message instead of a traceback."""
        log.exception("Answering %s %s failed", request.method, request.path)
        message = f"Something went wrong ({type(error).__name__}). Please try again."
        if request.path.startswith("/api/"):
            return jsonify(error=message), 500
        return Response(message, status=500, mimetype="text/plain")

    return app


def look_up(assistant: Any, inspection: Inspection) -> tuple[dict[str, Any], int]:
    """Ask the assistant what the photo shows: (the JSON answer, the HTTP status)."""
    log.info("Looking up %s", f"a {inspection.label}" if inspection.label else "a clicked area")
    try:
        return info_json(assistant.describe(inspection.photo, inspection.label or None)), 200
    except ValueError as error:                   # the assistant's own check of the picture or label
        return {"error": str(error)}, 400
    except AssistantError as error:               # no key, no internet, busy, refusal ...
        return {"error": str(error)}, 200
    except Exception as error:
        log.exception("Looking up a clicked object failed")
        return {"error": f"Something went wrong while looking it up ({type(error).__name__}). "
                         "Please click it again."}, 500


def answer_follow_up(assistant: Any, inspection: Inspection, question: str,
                     strokes: list[list[tuple[float, float]]] | None = None) -> tuple[dict[str, Any], int]:
    """Ask one more question about a clicked thing: (the JSON answer, the HTTP status).

    An object or an area: the assistant gets the stored photo, label, description
    and earlier questions, and with brush strokes also a copy of the photo with
    the marked part outlined. A recognised person: only text goes to the chat,
    with a sentence naming them; their picture is never sent anywhere.
    """
    if inspection.kind == "person":
        named = f"This question is about {inspection.person_name}, who the camera recognises. {question}"
        if len(named) > MAX_QUESTION_CHARS:           # the text actually sent must fit the chat's limit
            limit = MAX_QUESTION_CHARS - (len(named) - len(question))
            return {"error": f"The question is too long ({len(question)} characters). "
                             f"Questions about a person can have at most {limit} characters."}, 400
    try:
        if inspection.kind == "person":
            log.info("Answering a follow-up question about a recognised person")
            answer = assistant.ask(named)
        else:
            log.info("Answering a follow-up question about %s%s",
                     f"a {inspection.label}" if inspection.label else "a clicked area",
                     " (a marked part)" if strokes is not None else "")
            answer = ask_about_object(assistant, inspection, question, strokes)
    except ValueError as error:                   # the assistant's own check of the question or picture
        return {"error": str(error)}, 400
    except AssistantError as error:               # no key, no internet, busy, refusal ...
        return {"error": str(error)}, 200
    except Exception as error:
        log.exception("Answering a follow-up question failed")
        return {"error": f"Something went wrong while answering ({type(error).__name__}). "
                         "Please ask again."}, 500
    return answer_json(answer), 200


def ask_about_object(assistant: Any, inspection: Inspection, question: str,
                     strokes: list[list[tuple[float, float]]] | None) -> Any:
    """The assistant's answer about an object or an area, with the marked copy when there are strokes.

    The marking is drawn here on the server's own photo; the browser sent only the points.
    """
    photo, label = inspection.photo, inspection.label or None
    if strokes is None:
        return assistant.follow_up(photo, label, inspection.info, inspection.pairs, question)
    marked = mark_photo(photo, strokes)
    return assistant.follow_up(photo, label, inspection.info, inspection.pairs, question, marked=marked)


def mjpeg_stream(pipeline: Any, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> Iterator[bytes]:
    """The parts of the MJPEG stream: each new frame once, at most VIDEO_MAX_FPS a second.

    The pipeline gives a new bytes object for every new picture, so `is`
    tells a new frame from the one already sent. When nothing new comes for
    VIDEO_KEEPALIVE_SECONDS (a stalled stream), the last picture is sent
    again: writing is the only way to notice that the browser tab was closed,
    and then this generator ends instead of waiting forever.
    """
    last: bytes | None = None
    next_send = 0.0
    sent_at = 0.0
    while True:
        jpeg = pipeline.latest_jpeg()
        now = clock()
        if jpeg is not None and jpeg is not last and now >= next_send:
            last = jpeg                                    # a new picture
        elif last is None or now - sent_at < VIDEO_KEEPALIVE_SECONDS:
            sleep(VIDEO_POLL_SECONDS)
            continue
        next_send = now + 1.0 / VIDEO_MAX_FPS
        sent_at = now
        yield jpeg_part(last)


def jpeg_part(jpeg: bytes) -> bytes:
    """One picture of the stream, with its own small header."""
    head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
    return head.encode("ascii") + jpeg + b"\r\n"


# ---- starting everything -----------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the demo page: live video, info cards, reactions and chat.")
    parser.add_argument("--source", choices=SOURCES, help="override stream.source from config.yaml for this run")
    parser.add_argument("--file", metavar="PATH",
                        help="play this clip instead of stream.file_path (implies --source file); "
                             "a relative path starts at the project folder")
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)
    if args.file and args.source not in (None, "file"):
        parser.error(f"--file plays a clip, so it cannot be used with --source {args.source}")
    return args


def with_remembered_phone(config: dict[str, Any], remembered: str | None) -> dict[str, Any]:
    """A copy of the settings with the phone address from the camera box, when one is remembered.

    It replaces stream.phone_url for every source choice: it only matters once the
    source is the phone, from the start or after a switch. When it is the same phone
    (host and port) as the one in config.yaml, the configured address is kept, because
    it may carry the phone's login, which the camera box never accepts. config.yaml
    is not changed.
    """
    if remembered is None:
        return config
    configured = section(config, "stream").get("phone_url")
    if isinstance(configured, str) and same_phone(configured, remembered):
        return config
    return {**config, "stream": {**section(config, "stream"), "phone_url": remembered}}


def with_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """A copy of the settings with the command-line choices applied. config.yaml is not changed."""
    stream = dict(section(config, "stream"))
    if args.source:
        stream["source"] = args.source
    if args.file:
        stream["source"] = "file"
        stream["file_path"] = args.file
    return {**config, "stream": stream}


def setup_logging(log_file: Path) -> None:
    """Log to the console and to log_file (its folder is created)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # no line for every status poll


def build_assistant(config: dict[str, Any], pipeline: Pipeline) -> Any:
    """The assistant, sharing what the pipeline sees; a stand-in that explains when it cannot be built."""
    try:
        return Assistant(section(config, "assistant"), state=pipeline.state)
    except (ConfigError, AssistantError) as error:
        log.warning("The assistant is not available: %s", error)
        return UnavailableAssistant(f"The assistant is not available: {error}")


def already_serving(host: str, port: int, timeout: float = PORT_CHECK_SECONDS) -> bool:
    """True when something already answers on host:port, e.g. a copy of this program.

    Windows lets two servers bind the same port, so the server itself would not
    notice; a second copy would quietly load the models again.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def open_browser_soon(url: str) -> None:
    timer = threading.Timer(BROWSER_DELAY_SECONDS, webbrowser.open, args=(url,))
    timer.daemon = True
    timer.start()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")      # an odd character in a name never crashes logging
    try:
        config = with_overrides(load_config(), args)
        settings = section(config, "app")
        host: str = require(settings, "host", str)
        port: int = require(settings, "port", int)
        log_file = resolve_path(require(settings, "log_file", str))
    except ConfigError as error:
        print(f"Cannot start: {error}")
        return 1
    url = f"http://{host}:{port}/"
    if already_serving(host, port):                   # checked before any model is loaded
        print(f"The demo page is already running at {url} (or another program uses port {port}). "
              "Open that address, or stop the other copy first.")
        return 1
    setup_logging(log_file)
    camera_file = resolve_path(CAMERA_FILE)
    config = with_remembered_phone(config, load_remembered(camera_file))   # after logging: it may log a line

    try:
        pipeline = Pipeline(config)
    except ConfigError as error:
        print(f"Cannot start: {error}")
        return 1
    assistant = build_assistant(config, pipeline)
    app = create_app(pipeline, assistant, camera_file)

    pipeline.start()
    print(f"Demo page: {url}   (Ctrl+C stops the program)")
    if args.open:
        open_browser_soon(url)
    try:
        # load_dotenv=False: Flask must not read .env; the assistant reads its key itself.
        app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False, load_dotenv=False)
    except KeyboardInterrupt:
        pass
    except OSError as error:
        print(f"Cannot serve the page on {host}:{port}: {error}. Is the program already running?")
        return 1
    finally:
        print("Stopping ...")
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
