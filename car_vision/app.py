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

The People section of the page adds, edits and removes the person folders
(/api/people, through people.PeopleBook) and takes a person's photos from the
live camera (/api/people/<id>/photos, then /api/people/photos for the
progress); after every such change the assistant reads its notes and briefs
again. /api/scene asks the assistant to describe the whole picture.
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

from car_vision.assistant import MAX_IMAGE_BYTES, Assistant, AssistantError, ObjectInfo
from car_vision.config import ConfigError, load_config, require, resolve_path, section
from car_vision.faces import PERSON_ID
from car_vision.marking import NEARLY_ALL, mark_photo, marked_share
from car_vision.people import MAX_LINE, MAX_LINES, MAX_NAME, MAX_ROLE, MAX_STORY, PeopleBook
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

MAX_ID_CHARS = 100               # a person id in an address is never longer
PHOTO_COUNT = 5                  # photos taken when the page does not say how many
NO_PEOPLE = ("Adding people from the page is not available: "
             "recognizer.photos_dir is not set in config.yaml.")
NO_SUCH_PERSON = "There is no such person. Reload the page to see who is there."
NO_PICTURE = "There is no picture from the camera yet. Please try again in a moment."
PERSON_SHAPE = ('Send the person as JSON, like {"name": "Sara Ahmed", "role": "Team lead", '
                '"lines": ["Built the chassis"], "story": ""}.')


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

    def describe_scene(self, image: bytes) -> Any:
        raise AssistantError(self.reason)

    def reset(self) -> None:
        pass

    def reload(self) -> None:
        raise AssistantError(f"{self.reason} Restart the program once the problem is fixed.")


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
    epoch: int = 0                           # goes up whenever the pipeline's track numbers start again


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


def cache_key(inspection: Inspection) -> tuple[int, int, str] | None:
    """Only tracked objects are cached: an area or an untracked box may be anything next time.

    The epoch is part of the key: track numbers start again when the picture size or
    the camera changes, and track 3 then may be another object with the same label.
    """
    if inspection.kind != "object" or inspection.track_id is None:
        return None
    return inspection.epoch, inspection.track_id, inspection.label


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


def read_strokes(strokes: Any) -> list[list[tuple[float, float]]] | str | None:
    """The brush strokes from a /api/follow_up body as lists of (x, y), or the reason they are wrong.

    Each point is two numbers from 0 to 1 across and down the photo. The
    page sends at most MAX_STROKES strokes of at most MAX_STROKE_POINTS points.
    No strokes (null, an empty list, or the key left out) is None: no marking.
    """
    if strokes is None or strokes == []:
        return None
    if not isinstance(strokes, list):
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


# ---- people, and the assistant reading its notes and briefs again ---------------------------


def is_person_id(person_id: str) -> bool:
    """True for an id that can be a person folder; checked before an id from an address is used."""
    return len(person_id) <= MAX_ID_CHARS and PERSON_ID.fullmatch(person_id) is not None


def entry_json(entry: Any, enrolled: set[str]) -> dict[str, Any]:
    """One person for the People section: details and a photo count, never a picture."""
    person = entry.person
    return {"id": person.person_id, "name": person.name, "role": person.role,
            "lines": list(person.lines), "story": person.story,
            "photos": int(entry.photos), "enrolled": person.person_id in enrolled}


def person_fields(body: Any) -> dict[str, Any] | str:
    """The arguments for PeopleBook.save from a request body, or the reason the shape is wrong.

    The kinds and lengths of the fields are checked by PeopleBook.save itself.
    """
    if not isinstance(body, dict):
        return PERSON_SHAPE
    lines = body.get("lines", [])
    if not isinstance(lines, list):
        return "The card lines must be a list of lines of text."
    return {"name": body.get("name", ""), "role": body.get("role", ""), "lines": lines,
            "story": body.get("story", "")}


def reload_assistant(assistant: Any) -> str | None:
    """Make the assistant read the notes and the person briefs again; a warning when it could not.

    The change itself has happened by then, so a failure here is a warning, not an error.
    """
    try:
        assistant.reload()
    except AssistantError as error:
        log.warning("The assistant could not read its notes again: %s", error)
        return f"The assistant still answers from the old notes: {error}"
    except Exception as error:
        log.exception("The assistant could not read its notes again")
        return f"The assistant still answers from the old notes ({type(error).__name__})."
    return None


def enrol_again(pipeline: Any) -> str | None:
    """Enrol the remaining people again after a removal; a warning when it failed."""
    try:
        pipeline.reload_people()
    except RuntimeError:                       # face recognition is off: nobody's face is in use
        return None
    except Exception as error:
        log.exception("Enrolling the people again failed")
        return (f"The faces could not be enrolled again ({type(error).__name__}); "
                "the camera may still recognise this person until the program is restarted.")
    return None


def with_warnings(payload: dict[str, Any], *warnings: str | None) -> dict[str, Any]:
    """The payload with "warning" added when there is one."""
    found = [warning for warning in warnings if warning]
    if found:
        payload["warning"] = " ".join(found)
    return payload


class PhotoWatch:
    """Remembers that the page started a photo session, so that its end reloads the assistant once.

    Several pages may poll the progress; only the first poll that sees the end gets True.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting = False

    def started(self) -> None:
        with self._lock:
            self._waiting = True

    def ended_done(self, state: Any) -> bool:
        """True exactly once after a started session ended "done"; "failed" just stops the waiting."""
        with self._lock:
            if not self._waiting or state not in ("done", "failed"):
                return False
            self._waiting = False
            return state == "done"


# ---- the web server ------------------------------------------------------------


def question_problem(question: str) -> str | None:
    """Why a typed question (spaces already stripped) cannot be sent, or None when it is fine."""
    if not question:
        return "The question is empty."
    if len(question) > MAX_QUESTION_CHARS:
        return (f"The question is too long ({len(question)} characters). "
                f"Please keep it under {MAX_QUESTION_CHARS} characters.")
    return None


def create_app(pipeline: Any, assistant: Any, camera_file: Path | None = None,
               people: PeopleBook | None = None) -> Flask:
    """The Flask app for one pipeline and one assistant (real ones, or fakes in the tests).

    `camera_file` is where /api/camera remembers the phone's address; with None
    (as in most tests) the address is used but not remembered. `people` is the
    book of person folders; with None the page hides its People section and those
    routes answer with a plain error.
    """
    app = Flask(__name__)
    inspections = InspectionStore()
    answers = AnswerCache()
    gate = LookupGate()
    photo_watch = PhotoWatch()

    @app.get("/")
    def page() -> str:
        return render_template("index.html", max_question_chars=MAX_QUESTION_CHARS,
                               has_people=people is not None, max_name=MAX_NAME, max_role=MAX_ROLE,
                               max_story=MAX_STORY, max_lines=MAX_LINES, max_line=MAX_LINE)

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
                                person_name=target.person.name if target.person is not None else None,
                                epoch=int(getattr(target, "epoch", 0)))   # older Targets have no epoch
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
        strokes = read_strokes(body.get("strokes"))   # None: a question about the whole photo
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
            strokes = marking_of_a_part(inspection, strokes)
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

    # ---- people: the order of effects is always files, then the vision side, then the assistant

    @app.get("/api/people")
    def people_list() -> Any:
        if people is None:
            return jsonify(error=NO_PEOPLE), 503
        enrolled = pipeline.enrolled_ids()
        recognition = bool(pipeline.status().get("recognition"))
        return jsonify(people=[entry_json(entry, enrolled) for entry in people.all()], recognition=recognition)

    @app.post("/api/people")
    def person_add() -> Any:
        if people is None:
            return jsonify(error=NO_PEOPLE), 503
        fields = person_fields(request.get_json(silent=True))
        if isinstance(fields, str):
            return jsonify(error=fields), 400
        try:
            person = people.save(**fields)
        except ValueError as error:                   # PeopleBook's own plain message
            return jsonify(error=str(error)), 400
        log.info("Added the person %s", person.person_id)
        return jsonify(with_warnings({"ok": True, "id": person.person_id}, reload_assistant(assistant)))

    @app.put("/api/people/<person_id>")
    def person_change(person_id: str) -> Any:
        if people is None:
            return jsonify(error=NO_PEOPLE), 503
        if not is_person_id(person_id):               # checked before the id is used for anything
            return jsonify(error=NO_SUCH_PERSON), 404
        fields = person_fields(request.get_json(silent=True))
        if isinstance(fields, str):
            return jsonify(error=fields), 400
        try:
            people.save(**fields, person_id=person_id)
        except KeyError:
            return jsonify(error=NO_SUCH_PERSON), 404
        except ValueError as error:
            return jsonify(error=str(error)), 400
        log.info("Changed the details of %s", person_id)
        # Enrolling again makes the camera side read the new name and role too, so the label on
        # the video and the person's panel change at once, not only after a restart.
        return jsonify(with_warnings({"ok": True, "id": person_id},
                                     enrol_again(pipeline), reload_assistant(assistant)))

    @app.delete("/api/people/<person_id>")
    def person_remove(person_id: str) -> Any:
        if people is None:
            return jsonify(error=NO_PEOPLE), 503
        if not is_person_id(person_id):
            return jsonify(error=NO_SUCH_PERSON), 404
        progress = pipeline.photo_progress()          # photos of this person stop before the folder moves
        if progress and progress.get("person_id") == person_id and progress.get("state") == "taking":
            pipeline.cancel_photos()
        try:
            people.remove(person_id)
        except KeyError:
            return jsonify(error=NO_SUCH_PERSON), 404
        except OSError as error:
            log.warning("Could not move the folder of %s aside: %s", person_id, error)
            return jsonify(error="The person's folder could not be moved aside. Is a file in it open "
                                 "in another program?"), 500
        log.info("Removed %s (the folder was moved aside)", person_id)
        return jsonify(with_warnings({"ok": True}, enrol_again(pipeline), reload_assistant(assistant)))

    @app.post("/api/people/<person_id>/photos")
    def photos_start(person_id: str) -> Any:
        if people is None:
            return jsonify(error=NO_PEOPLE), 503
        if not is_person_id(person_id):
            return jsonify(error=NO_SUCH_PERSON), 404
        body = request.get_json(silent=True)
        count = body.get("count", PHOTO_COUNT) if isinstance(body, dict) else PHOTO_COUNT
        try:
            pipeline.start_photos(person_id, count)
        except KeyError:
            return jsonify(error=NO_SUCH_PERSON), 404
        except ValueError as error:                   # recognition off, already running, a bad count
            return jsonify(error=str(error)), 400
        photo_watch.started()
        return jsonify(ok=True)

    @app.get("/api/people/photos")
    def photos_progress() -> Any:
        if people is None:
            return jsonify(error=NO_PEOPLE), 503
        progress = pipeline.photo_progress()
        payload: dict[str, Any] = {"progress": progress}
        if progress is not None and photo_watch.ended_done(progress.get("state")):
            with_warnings(payload, reload_assistant(assistant))   # once, however many pages poll
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.delete("/api/people/photos")
    def photos_cancel() -> Any:
        if people is None:
            return jsonify(error=NO_PEOPLE), 503
        pipeline.cancel_photos()
        return jsonify(ok=True)

    # ---- describe the scene --------------------------------------------------------------

    @app.post("/api/scene")
    def scene() -> Any:
        photo = pipeline.scene_photo()                # the server's own picture: none comes from the browser
        if photo is None:
            return jsonify(error=NO_PICTURE), 503
        try:
            answer = assistant.describe_scene(photo)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except AssistantError as error:               # no key, no internet, busy, refusal ...
            return jsonify(error=str(error))
        except Exception as error:
            log.exception("Describing the scene failed")
            return jsonify(error=f"Something went wrong while describing the scene ({type(error).__name__}). "
                                 "Please try again."), 500
        return jsonify(text=str(answer.text))

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


def marking_of_a_part(inspection: Inspection,
                      strokes: list[list[tuple[float, float]]] | None) -> list[list[tuple[float, float]]] | None:
    """The strokes, or None when they cover nearly the whole photo.

    Brushed all over, or along all four edges, the marking leaves no outline to see,
    so the question is asked about the whole photo, as an ordinary one.
    """
    if strokes is None:
        return None
    try:
        share = marked_share(inspection.photo, strokes)
    except ValueError:                            # not a picture: the assistant will say so
        return None
    return None if share > NEARLY_ALL else strokes


def ask_about_object(assistant: Any, inspection: Inspection, question: str,
                     strokes: list[list[tuple[float, float]]] | None) -> Any:
    """The assistant's answer about an object or an area, with the marked copy when there are strokes.

    The marking is drawn here on the server's own photo; the browser sent only the points.
    """
    photo, label = inspection.photo, inspection.label or None
    if strokes is None:
        return assistant.follow_up(photo, label, inspection.info, inspection.pairs, question)
    marked = mark_photo(photo, strokes, max_bytes=MAX_IMAGE_BYTES)   # never over what the assistant takes
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


def build_people_book(config: dict[str, Any]) -> PeopleBook | None:
    """The book of person folders in recognizer.photos_dir, with removed folders moved to
    <photos_dir>.removed beside it (data/people.removed); None when the setting is missing."""
    try:
        photos_dir = resolve_path(require(section(config, "recognizer"), "photos_dir", str))
    except ConfigError as error:
        log.warning("Adding people from the page is off: %s", error)
        return None
    return PeopleBook(photos_dir, photos_dir.with_name(photos_dir.name + ".removed"))


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
    app = create_app(pipeline, assistant, camera_file, build_people_book(config))

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
