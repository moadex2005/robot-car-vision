"""The demo page: live video, what is in view, details on a click, reactions and the chat.

Served by Flask; this is the only module that imports flask. A click on the
video goes to /api/inspect, which answers at once with a small photo of what
was clicked; for an object, /api/describe then asks the assistant what it is,
and /api/follow_up answers more questions about that one thing, also about a
part the visitor brushed over (the page sends only the brush strokes; marking.py
draws them on the server's own copy of the photo). The camera
box on the page sends a phone's address to /api/camera, which switches the
video to that phone and remembers the address for the next start, and the
picture size to ask the phone for to /api/camera/picture. /api/camera/power
turns the camera off and on (the round button on the video).

The People section of the page adds, edits and removes the person folders
(/api/people, through people.PeopleBook) and takes a person's photos from the
live camera (/api/people/<id>/photos, then /api/people/photos for the
progress); after every such change the assistant reads its notes and briefs
again. /api/scene asks the assistant to describe the whole picture.

/api/search starts looking for things by a typed name (the "Look for" field on
the video, or a sentence like "find my keys") and stops it; the page follows the
search in the status. /api/memory gives what the camera remembers (the Timeline),
a small picture of a remembered object, and empties the memory. /api/discover
switches auto-discover on and off (a small hosted model names what is in view).
/api/voice/say and /api/voice/hear are the hosted voice (voice.py): an answer's
text in, its sound out, and a recorded question in, its words out.

Only this page may change anything: every POST, PUT and DELETE under /api/ must
be sent as JSON, which a plain form on another website cannot do, and a request
that says it comes from another website (its Origin header) is refused.
Start it with

    .venv\\Scripts\\python.exe -m car_vision.app
    .venv\\Scripts\\python.exe -m car_vision.app --source webcam --open
    .venv\\Scripts\\python.exe -m car_vision.app --file data/samples/people_clip.mp4

and open the address it prints. Ctrl+C stops it. The page is served on
app.host only (127.0.0.1 in config.yaml), so only this laptop can open it.

The phone door: with phone_door.enabled in config.yaml, the same page is also
served over HTTPS on phone_door.port to the whole network, behind the password
PAGE_PASSWORD from .env (door.py has the parts). Only that second server marks its
requests as coming through the phone door (environ["car_vision.door"]); nothing a
browser sends can. Through it, every address but /door needs a signed-in session.
The laptop's own door stays exactly as it was: no password, and no /door.
"""

from __future__ import annotations

import argparse
import base64
import ipaddress
import logging
import math
import re
import secrets
import socket
import ssl
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

import cv2
from flask import Flask, Response, abort, jsonify, redirect, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.serving import WSGIRequestHandler, make_server

from car_vision.assistant import LANGUAGES, MAX_IMAGE_BYTES, Assistant, AssistantError, ObjectInfo
from car_vision.config import ConfigError, load_config, require, resolve_path, section
from car_vision.door import (LONGEST_PASSWORD, PASSWORD_VARIABLE, DoorError, PhoneDoor, can_be_typed,
                             ensure_certificate, is_loopback, lan_addresses, minutes_words, read_password,
                             read_settings)
from car_vision.faces import PERSON_ID
from car_vision.marking import NEARLY_ALL, mark_photo, marked_share
from car_vision.people import MAX_LINE, MAX_LINES, MAX_NAME, MAX_ROLE, MAX_STORY, PeopleBook
from car_vision.phone import load_remembered, phone_url, remember, same_phone
from car_vision.pipeline import Pipeline, clean_names, first_line
from car_vision.stream import SOURCES
from car_vision.voice import OFF, HostedVoice, VoiceError

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
VIDEO_MAX_FPS = 30               # /video sends at most this many frames a second (the phone sends 30)
VIDEO_POLL_SECONDS = 0.01        # how often /video looks for a new frame
VIDEO_KEEPALIVE_SECONDS = 1.0    # with no new frame for this long, the last one is sent again, so a
                                 # closed browser tab is noticed and its server thread ends
PORT_CHECK_SECONDS = 0.5         # how long main() waits for an answer on the port before starting
START_LOCK_FILE = "car_vision.lock"   # beside the log file; held by the copy of the program that runs
STILL_STARTING = "The demo is still starting. Wait for the page to open; it takes about a minute."
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

MAX_SEARCH_CHARS = 300           # a typed "Look for" text is never longer (6 names of 40 and the commas)
SEARCH_SHAPE = ('Send what to look for as JSON, like {"names": ["power bank", "keys"]} '
                'or {"text": "power bank, keys"}.')
NOTHING_TO_FIND = "Type what to look for, like power bank or keys."
NAME_SEPARATORS = re.compile(r",|\band\b", re.IGNORECASE)   # "power bank, keys and the remote"
MEMORY_LIMIT = 50                # sightings the Timeline asks for when it does not say
MOST_SIGHTINGS = 200             # ... and at most
MAX_FILTER_CHARS = 100           # the Timeline's filter text is never longer
SIGHTING_ID = re.compile(r"[0-9]{1,18}")   # an id in an address: digits only, and it fits SQLite's number
PICTURE_CACHE = "no-store"       # after Clear memory an id may belong to another row: never kept
NO_MEMORY_PICTURE = "There is no picture of that."
DISCOVER_SHAPE = 'Send the switch as JSON, like {"on": true} or {"on": false}.'
CAMERA_POWER_SHAPE = 'Send {"on": true} or {"on": false}.'

CHANGING_METHODS = frozenset({"POST", "PUT", "DELETE"})   # these must come from this page, as JSON
NOT_JSON = 'Send the request as JSON (Content-Type: application/json), with {} when there is nothing to send.'
OTHER_WEBSITE = "Requests from other websites are refused."

HEAR_PATH = "/api/voice/hear"    # the one changing request that is not JSON: its body is a WAV recording
WRONG_HOST = "Open this page by its own address, like http://127.0.0.1:5000/, not through another name."
MAX_RECORDING_BYTES = 2_000_000  # a recording above this is refused before it is read
SAY_SHAPE = 'Send the text to speak as JSON, like {"text": "Hello."}.'
HEAR_LANGUAGES = ("en", "ar")
NO_RECORDING_SIZE = "Send the recording with its size (Content-Length)."
RECORDING_TOO_LARGE = f"The recording is too large: at most {MAX_RECORDING_BYTES // 1_000_000} MB."


class UnavailableAssistant:
    """Stands in when the real assistant could not be built, e.g. the notes file is missing.

    The page keeps working, and every question or look-up gets the reason as an error.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def ask(self, question: str, language: str = "en") -> Any:
        raise AssistantError(self.reason)

    def describe(self, image: bytes, label: str | None = None, language: str = "en") -> Any:
        raise AssistantError(self.reason)

    def follow_up(self, image: bytes, label: str | None, info: Any, earlier: Any, question: str,
                  marked: bytes | None = None, language: str = "en") -> Any:
        raise AssistantError(self.reason)

    def describe_scene(self, image: bytes, language: str = "en") -> Any:
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


def cache_key(inspection: Inspection, language: str = "en") -> tuple[int, int, str, str] | None:
    """Only tracked objects are cached: an area or an untracked box may be anything next time.

    The epoch is part of the key: track numbers start again when the picture size or
    the camera changes, and track 3 then may be another object with the same label.
    The language is part of it too: a thing looked up in English is looked up again in Arabic.
    """
    if inspection.kind != "object" or inspection.track_id is None:
        return None
    return inspection.epoch, inspection.track_id, inspection.label, language


LANGUAGE_PROBLEM = 'The answer language must be "en" (English) or "ar" (Arabic), like {"language": "ar"}.'


def read_language(body: Any) -> str | None:
    """The answer language a request body asks for: "en" when it names none, None when it
    names something other than "en" or "ar"."""
    language = body.get("language", "en") if isinstance(body, dict) else "en"
    return language if isinstance(language, str) and language in LANGUAGES else None


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
        # Only the name and the role go to the panel. The card lines and the story stay on the
        # server: the assistant tells them when the visitor asks.
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


# ---- looking for things by name, and the camera's memory ---------------------------------


def split_names(text: str) -> list[str]:
    """The names in a typed text, split on commas and the word "and"; empty parts are dropped.

    "power bank, keys and the remote" gives ["power bank", "keys", "the remote"].
    """
    return [part.strip() for part in NAME_SEPARATORS.split(text) if part.strip()]


def search_names(body: Any) -> list[Any] | str:
    """The names from a POST /api/search body, not yet checked, or the reason the shape is wrong.

    The body has either "names" (a list) or "text" (what was typed, split here).
    """
    if not isinstance(body, dict) or ("names" in body) == ("text" in body):
        return SEARCH_SHAPE
    if "names" in body:
        names = body["names"]
        return names if isinstance(names, list) else SEARCH_SHAPE
    text = body["text"]
    if not isinstance(text, str):
        return SEARCH_SHAPE
    if len(text) > MAX_SEARCH_CHARS:
        return f"That is too long to look for. Please keep it under {MAX_SEARCH_CHARS} characters."
    return split_names(text)


def memory_of(pipeline: Any) -> Any:
    """The pipeline's memory (a SightingLog), or None when the memory is off."""
    sightings = getattr(pipeline, "sightings", None)   # older pipelines (and most fakes) have none
    return sightings() if callable(sightings) else None


def memory_query(args: Any) -> tuple[int, str | None] | str:
    """(limit, text) from the GET /api/memory address, or the reason it is wrong.

    The limit is held between 1 and MOST_SIGHTINGS; the filter text has a length limit.
    """
    limit_text = args.get("limit", str(MEMORY_LIMIT))
    if not re.fullmatch(r"[0-9]{1,6}", limit_text):
        return f"limit must be a whole number from 1 to {MOST_SIGHTINGS}."
    limit = min(max(int(limit_text), 1), MOST_SIGHTINGS)
    text = args.get("text", "").strip()
    if len(text) > MAX_FILTER_CHARS:
        return f"The filter can have at most {MAX_FILTER_CHARS} characters."
    return limit, text or None


def sighting_json(sighting: Any) -> dict[str, Any]:
    """One remembered sighting for the Timeline. A person never has a picture."""
    is_object = sighting.kind == "object"
    return {
        "id": int(sighting.id),
        "kind": str(sighting.kind),
        "label": str(sighting.label),
        "name": str(sighting.name) if sighting.name is not None else None,
        "first_seen": float(sighting.first_seen),
        "last_seen": float(sighting.last_seen),
        "place": str(sighting.place),
        "confidence": round(float(sighting.confidence), 2),
        "has_picture": bool(sighting.has_picture) and is_object,
    }


# ---- only this page may change anything ----------------------------------------------------


def own_origin(host_url: str) -> str:
    """This server's origin as a browser writes it in the Origin header: "http://127.0.0.1:5000"."""
    return host_url.rstrip("/")


def host_name(host: str) -> str:
    """The name in a Host header without its port: "127.0.0.1" for "127.0.0.1:5000",
    "::1" for "[::1]:5000", "" for a header that cannot be read."""
    host = host.strip()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end > 0 else ""
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def names_this_computer(host: str, laptop: bool) -> bool:
    """True when the Host header names this computer by its own address, not by a web name.

    A hostile website can point its own name at 127.0.0.1 (or at this laptop's
    network address) and so make a browser treat this server as "the same
    website": the Origin then matches the Host, and the same-origin rule in
    refusal() is fooled. Such a request always carries the website's name in
    the Host header. So the page answers only to `localhost` and to a plain
    address: this computer itself through the laptop's door, and also a
    private network address through the phone door.
    """
    name = host_name(host).lower()
    if name == "localhost":
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False                              # a web name: never this page's own address
    return address.is_loopback or (not laptop and address.is_private)


def refusal(method: str, path: str, origin: str | None, mimetype: str, host_url: str) -> tuple[str, int] | None:
    """Why a request must be refused, as (message, status), or None when it may go on.

    Another website open in the same browser can send a plain form post to this
    server (a form cannot send JSON), and a browser names that website in the
    Origin header. So: a request from another origin is 403, and a POST, PUT or
    DELETE under /api/ that is not JSON is 415. The one exception is the
    recording of a spoken question, posted to HEAR_PATH as audio/wav: no form
    can send that type, and another website's script that tries is stopped by
    the browser (and by the Origin check above).
    """
    if not path.startswith("/api/"):
        return None
    if origin is not None and origin != own_origin(host_url):
        return OTHER_WEBSITE, 403
    if method == "POST" and path == HEAR_PATH and mimetype == "audio/wav":
        return None                   # the recording of a question; a form cannot send audio/wav either
    if method in CHANGING_METHODS and mimetype != "application/json":
        return NOT_JSON, 415
    return None


def discover_interval(config: dict[str, Any]) -> float | None:
    """lookout.every_seconds, for the words beside the Auto-discover switch; None when it is not set."""
    lookout = config.get("lookout")
    every = lookout.get("every_seconds") if isinstance(lookout, dict) else None
    if not is_number(every) or not 0 < every < 86400:
        return None
    return float(every)


def voice_json(voice: Any) -> dict[str, Any]:
    """What GET /api/voice tells the page: whether there is a hosted voice, today's uses and cost, its limits."""
    if voice is None:
        return {"hosted": False}
    return {"hosted": True, "uses": int(voice.uses_today), "spent_usd": round(float(voice.spent_today), 6),
            "budget_usd": float(voice.budget), "max_chars": int(voice.max_chars),
            "max_seconds": int(voice.max_seconds)}


def interval_words(seconds: float | None) -> str:
    """"10" for 10.0, "2.5" for 2.5, "" when unknown: how the page writes the interval."""
    if seconds is None:
        return ""
    return str(int(seconds)) if float(seconds).is_integer() else f"{seconds:g}"


# ---- the phone door ------------------------------------------------------------

DOOR_ENVIRON = "car_vision.door"  # set to "phone" by the phone door's server only (through_phone_door)
PHONE = "phone"
DOOR_PAGE = "/door"
DOOR_PAGES = frozenset({DOOR_PAGE, "/door/leave"})   # the only addresses open without a session
DOOR_COOKIE = "car_vision_door"
MAX_PASSWORD_CHARS = LONGEST_PASSWORD   # a longer one is wrong without being compared
DOOR_BODY_BYTES = 4 * 1024        # the most a phone may send to /door and /door/leave
PHONE_BODY_BYTES = 3 * 1024 * 1024   # ... and to any other address
TOO_BIG = "That request is too big, or does not say its size."
SIGN_IN_FIRST = "Sign in first."
ONLY_ON_THE_LAPTOP = "This page opens only on the laptop itself."
WRONG_PASSWORD = "That is not the password."
PASSWORD_SHAPE = 'Send the password as JSON, like {"password": "..."}.'
DOOR_OFF = ("The phone door is off. Set phone_door.enabled to true in config.yaml and "
            f"{PASSWORD_VARIABLE} in .env, then start the program again.")
NOT_A_DOOR_ADDRESS = "That is not an address of the phone door."
NO_NETWORK = ("This laptop has no private network address. Join the phone's hotspot or "
              "the same Wi-Fi as the phone, then start the program again.")
DOOR_SOCKET_SECONDS = 10          # a phone-door connection waits at most this long for the client
DOOR_CLOSE_SECONDS = 3.0          # closing the door at the end never takes longer
NOT_LOCAL = ("app.host must be this laptop itself (127.0.0.1, localhost or ::1), got {host!r}. "
             "To open the page on a phone or another computer, use phone_door in config.yaml.")
QR_EDGE = 2                       # white modules added to the encoder's own 2, so 4 in all
QR_SCALE = 8                      # pixels for each module: about 260 pixels for an address


def from_the_laptop() -> bool:
    """True only for a request through the laptop's own door: not marked by the phone door's
    server (see through_phone_door), and from this computer itself. Anything else is a phone's."""
    return request.environ.get(DOOR_ENVIRON) != PHONE and is_loopback(request.remote_addr)


def body_too_big(method: str, path: str, length: int | None) -> bool:
    """True when a phone's request body must be refused before it is read: a change that does
    not say its size, or a body above the limit for this address."""
    if length is None:
        return method in CHANGING_METHODS
    return length > (DOOR_BODY_BYTES if path in DOOR_PAGES else PHONE_BODY_BYTES)


def answer(message: str, code: int) -> Any:
    """A JSON error that is never kept by the browser."""
    response = jsonify(error=message)
    response.status_code = code
    response.headers["Cache-Control"] = "no-store"
    return response


def door_refusal(method: str, origin: str | None, mimetype: str, host_url: str) -> tuple[str, int] | None:
    """refusal() for the sign-in addresses, which are not under /api/: the same two rules."""
    if origin is not None and origin != own_origin(host_url):
        return OTHER_WEBSITE, 403
    if method in CHANGING_METHODS and mimetype != "application/json":
        return NOT_JSON, 415
    return None


def sign_in_first(path: str) -> Any:
    """The answer to a phone without a session: 401 for data and video, the sign-in page otherwise."""
    if path.startswith("/api/") or path == "/video" or path.startswith("/video/"):
        return answer(SIGN_IN_FIRST, 401)
    return redirect(DOOR_PAGE)


def typed_password() -> str | None:
    """The password from a POST /door body {"password": "<text>"}, or None for anything else:
    no JSON, other JSON, JSON nested too deep to read, or text that cannot be typed."""
    try:
        body = request.get_json(silent=True)
    except RecursionError:                        # silent=True covers only ValueError
        return None
    given = body.get("password") if isinstance(body, dict) else None
    return given if isinstance(given, str) and can_be_typed(given) else None


def too_many_tries(door: PhoneDoor) -> str:
    return f"Too many tries. Wait {minutes_words(door.settings.pause_seconds)}."


def qr_png(text: str) -> bytes:
    """A PNG of a QR code for this text, black on white with a white edge."""
    code = cv2.QRCodeEncoder.create().encode(text)
    code = cv2.copyMakeBorder(code, QR_EDGE, QR_EDGE, QR_EDGE, QR_EDGE, cv2.BORDER_CONSTANT, value=255)
    code = cv2.resize(code, None, fx=QR_SCALE, fy=QR_SCALE, interpolation=cv2.INTER_NEAREST)
    ok, png = cv2.imencode(".png", code)
    if not ok:
        raise RuntimeError("The QR code could not be made into a PNG.")
    return png.tobytes()


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
               people: PeopleBook | None = None, discover_every: float | None = None,
               voice: Any = None, door: Any = None) -> Flask:
    """The Flask app for one pipeline and one assistant (real ones, or fakes in the tests).

    `camera_file` is where /api/camera remembers the phone's address; with None
    (as in most tests) the address is used but not remembered. `people` is the
    book of person folders; with None the page hides its People section and those
    routes answer with a plain error. `discover_every` is lookout.every_seconds,
    only for the words beside the Auto-discover switch ("looking every 10 s").
    `voice` is the HostedVoice behind /api/voice; with None the page offers only
    the browser's own voice and listening.
    `door` is the phone door (a door.PhoneDoor: the password, the sessions and the
    tries); with None there is no phone door, and nothing comes through one.
    """
    app = Flask(__name__)
    inspections = InspectionStore()
    answers = AnswerCache()
    gate = LookupGate()
    photo_watch = PhotoWatch()

    @app.before_request
    def only_this_page() -> Any:
        """Refuse what another website could send (see names_this_computer and refusal). An
        unknown address or a wrong method is left to Flask, so it still answers 404 or 405."""
        if not names_this_computer(request.host, from_the_laptop()):
            return answer(WRONG_HOST, 403)            # every address: the page and the video too
        if request.routing_exception is not None:
            return None
        refused = refusal(request.method, request.path, request.headers.get("Origin"),
                          request.mimetype, request.host_url)
        if refused is None:
            return None
        message, code = refused
        return jsonify(error=message), code

    @app.before_request
    def phone_door_guard() -> Any:
        """Through the phone door only a signed-in phone gets further; /door is open to sign in.
        Through the laptop's door /door is not there (404), and nothing else changes."""
        if from_the_laptop():
            if request.path in DOOR_PAGES:
                abort(404)
            return None
        if body_too_big(request.method, request.path, request.content_length):
            return answer(TOO_BIG, 413)               # refused before a byte of it is read
        if door is None:                              # no phone door: only the laptop opens the page
            return answer(ONLY_ON_THE_LAPTOP, 401)
        if request.path in DOOR_PAGES:
            refused = door_refusal(request.method, request.headers.get("Origin"), request.mimetype,
                                   request.host_url)
            return None if refused is None else (jsonify(error=refused[0]), refused[1])
        if door.sessions.valid(request.cookies.get(DOOR_COOKIE)):
            return None
        return sign_in_first(request.path)

    @app.get("/")
    def page() -> str:
        return render_template("index.html", max_question_chars=MAX_QUESTION_CHARS,
                               has_people=people is not None, max_name=MAX_NAME, max_role=MAX_ROLE,
                               max_story=MAX_STORY, max_lines=MAX_LINES, max_line=MAX_LINE,
                               has_memory=memory_of(pipeline) is not None,
                               max_search_chars=MAX_SEARCH_CHARS, max_filter_chars=MAX_FILTER_CHARS,
                               discover_every=interval_words(discover_every))

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
        language = read_language(body)
        if language is None:
            return jsonify(error=LANGUAGE_PROBLEM), 400
        try:
            answer = assistant.ask(question, language=language)
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
        body = request.get_json(silent=True)
        click = read_click(body)
        if isinstance(click, str):
            return jsonify(error=click), 400
        language = read_language(body)                 # a description kept in another language is not shown
        if language is None:
            return jsonify(error=LANGUAGE_PROBLEM), 400
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
        key = cache_key(inspection, language)
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
        language = read_language(body)
        if language is None:
            return jsonify(error=LANGUAGE_PROBLEM), 400
        key = cache_key(inspection, language)
        if key is None:                               # an area or an untracked box: never cached
            reply, code = look_up(assistant, inspection, language)
        else:
            with gate.one_at_a_time(key):             # a second click on the same object waits here
                reply, code = answers.get(key), 200
                if reply is None:
                    reply, code = look_up(assistant, inspection, language)
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
        language = read_language(body)
        if language is None:
            return jsonify(error=LANGUAGE_PROBLEM), 400
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
            reply, code = answer_follow_up(assistant, inspection, question, strokes, language)
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

    @app.post("/api/camera/power")
    def camera_power() -> Any:
        """Turn the camera off and on (the round button on the video)."""
        body = request.get_json(silent=True)
        on = body.get("on") if isinstance(body, dict) and set(body) == {"on"} else None
        if not isinstance(on, bool):
            return jsonify(error=CAMERA_POWER_SHAPE), 400
        try:
            pipeline.set_camera(on)                   # returns at once; the page follows the status
        except ConfigError as error:                  # e.g. the clip given by --file was deleted
            return jsonify(error=f"The camera could not be started: {first_line(error)}"), 400
        log.info("The camera was turned %s from the page", "on" if on else "off")
        return jsonify(ok=True, on=on)

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
        language = read_language(request.get_json(silent=True))
        if language is None:
            return jsonify(error=LANGUAGE_PROBLEM), 400
        photo = pipeline.scene_photo()                # the server's own picture: none comes from the browser
        if photo is None:
            return jsonify(error=NO_PICTURE), 503
        try:
            answer = assistant.describe_scene(photo, language=language)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except AssistantError as error:               # no key, no internet, busy, refusal ...
            return jsonify(error=str(error))
        except Exception as error:
            log.exception("Describing the scene failed")
            return jsonify(error=f"Something went wrong while describing the scene ({type(error).__name__}). "
                                 "Please try again."), 500
        return jsonify(text=str(answer.text))

    # ---- looking for things by name ------------------------------------------------------

    @app.post("/api/search")
    def search_start() -> Any:
        names = search_names(request.get_json(silent=True))
        if isinstance(names, str):
            return jsonify(error=names), 400
        try:
            names = list(clean_names(names))          # the pipeline's own rules and messages
            if not names:
                return jsonify(error=NOTHING_TO_FIND), 400
            shown = pipeline.search(names)            # returns at once; the status says how it goes
        except ValueError as error:
            return jsonify(error=str(error)), 400
        # The names as the status shows them: a known person's name as they were enrolled.
        return jsonify(ok=True, names=list(shown) if shown is not None else names)

    @app.delete("/api/search")
    def search_stop() -> Any:
        pipeline.search([])
        return jsonify(ok=True)

    # ---- auto-discover ----------------------------------------------------------------------

    @app.post("/api/discover")
    def discover() -> Any:
        body = request.get_json(silent=True)
        on = body.get("on") if isinstance(body, dict) and set(body) == {"on"} else None
        if not isinstance(on, bool):
            return jsonify(error=DISCOVER_SHAPE), 400
        pipeline.discover(on)                         # returns at once; the status says how it goes
        log.info("Auto-discover was switched %s from the page", "on" if on else "off")
        return jsonify(ok=True, on=on)

    # ---- the hosted voice: speech and listening through Hugging Face ---------------------
    # A VoiceError is a 200 with {"error", "kind"}, so the page quietly uses the browser's voice.

    @app.get("/api/voice")
    def voice_status() -> Any:
        payload = voice_json(voice)
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/voice/say")
    def voice_say() -> Any:
        body = request.get_json(silent=True)
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            return jsonify(error=SAY_SHAPE), 400
        if voice is None:
            return jsonify(error=OFF, kind="off")
        try:
            sound = voice.say(text)
        except VoiceError as error:
            return jsonify(error=str(error), kind=error.kind)
        return Response(sound, mimetype="audio/mpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/voice/hear")
    def voice_hear() -> Any:
        language = request.args.get("language", "en")
        if language not in HEAR_LANGUAGES:
            return jsonify(error="The language must be en or ar."), 400
        size = request.content_length
        if size is None:
            return jsonify(error=NO_RECORDING_SIZE), 411
        if size > MAX_RECORDING_BYTES:              # refused before the body is read into memory
            return jsonify(error=RECORDING_TOO_LARGE), 413
        if voice is None:
            return jsonify(error=OFF, kind="off")
        try:
            text = voice.hear(request.get_data(cache=False), language)
        except VoiceError as error:
            return jsonify(error=str(error), kind=error.kind)
        return jsonify(text=text)

    # ---- the camera's memory ---------------------------------------------------------------

    @app.get("/api/memory")
    def memory_list() -> Any:
        query = memory_query(request.args)
        if isinstance(query, str):
            return jsonify(error=query), 400
        memory = memory_of(pipeline)
        found = memory.recent(*query) if memory is not None else []
        response = jsonify(sightings=[sighting_json(sighting) for sighting in found])
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/memory/<sighting_id>/picture")
    def memory_picture(sighting_id: str) -> Any:
        memory = memory_of(pipeline)
        if memory is None or not SIGHTING_ID.fullmatch(sighting_id):   # checked before it is used
            return jsonify(error=NO_MEMORY_PICTURE), 404
        jpeg = memory.picture(int(sighting_id))       # only objects ever have one, never a person
        if not jpeg:
            return jsonify(error=NO_MEMORY_PICTURE), 404
        return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": PICTURE_CACHE})

    @app.delete("/api/memory")
    def memory_clear() -> Any:
        memory = memory_of(pipeline)
        if memory is not None:
            memory.clear()
            log.info("The camera's memory was emptied from the page")
        return jsonify(ok=True)

    # ---- the phone door: signing in (phone door only; phone_door_guard sends the laptop a 404) --

    @app.get("/door")
    def door_page() -> Any:
        if door.sessions.valid(request.cookies.get(DOOR_COOKIE)):
            return redirect("/")                      # already signed in
        response = Response(render_template("door.html", max_password_chars=MAX_PASSWORD_CHARS))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/door")
    def door_sign_in() -> Any:
        """The password, compared only by door.matches. Too many wrong ones: a pause.

        Every try counts as wrong (door.throttle.take) until the password is right, also a
        body that is not a password at all.
        """
        who = request.remote_addr or "unknown"        # from the connection, never from a header
        if not door.throttle.take(who):               # also when this one would be right
            return jsonify(error=too_many_tries(door)), 429
        given = typed_password()
        if given is None:
            log.warning("A sign-in at the phone door sent no password (from %s)", who)
            return jsonify(error=PASSWORD_SHAPE), 400
        if len(given) > MAX_PASSWORD_CHARS or not door.matches(given):
            log.warning("A wrong password was typed at the phone door (from %s)", who)
            return jsonify(error=WRONG_PASSWORD), 403
        door.throttle.passed(who)
        response = jsonify(ok=True)
        response.set_cookie(DOOR_COOKIE, door.sessions.open(), max_age=int(door.settings.session_seconds),
                            path="/", secure=True, httponly=True, samesite="Strict")
        log.info("A phone signed in at the phone door (from %s)", who)
        return response

    @app.post("/door/leave")
    def door_leave() -> Any:
        door.sessions.close(request.cookies.get(DOOR_COOKIE))
        response = jsonify(ok=True)
        response.delete_cookie(DOOR_COOKIE, path="/", secure=True, httponly=True, samesite="Strict")
        return response

    # ---- the phone door as the laptop sees it (laptop's door only) -------------------------

    @app.get("/api/door")
    def door_status() -> Any:
        """Whether a phone can open the page, and at which addresses; or why not."""
        if not from_the_laptop():
            abort(404)
        if door is None:
            payload: dict[str, Any] = {"open": False, "why": DOOR_OFF}
        elif door.addresses:
            payload = {"open": True, "addresses": list(door.addresses)}
        else:
            payload = {"open": False, "why": door.why}
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/door/qr")
    def door_qr() -> Any:
        """A QR code of one of the phone door's own addresses, never of anything else."""
        if not from_the_laptop():
            abort(404)
        address = request.args.get("address", "")
        if door is None or address not in door.addresses:
            return jsonify(error=NOT_A_DOOR_ADDRESS), 400
        return Response(qr_png(address), mimetype="image/png", headers={"Cache-Control": "no-store"})

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


def look_up(assistant: Any, inspection: Inspection, language: str = "en") -> tuple[dict[str, Any], int]:
    """Ask the assistant what the photo shows, in this language: (the JSON answer, the HTTP status)."""
    log.info("Looking up %s", f"a {inspection.label}" if inspection.label else "a clicked area")
    try:
        return info_json(assistant.describe(inspection.photo, inspection.label or None, language=language)), 200
    except ValueError as error:                   # the assistant's own check of the picture or label
        return {"error": str(error)}, 400
    except AssistantError as error:               # no key, no internet, busy, refusal ...
        return {"error": str(error)}, 200
    except Exception as error:
        log.exception("Looking up a clicked object failed")
        return {"error": f"Something went wrong while looking it up ({type(error).__name__}). "
                         "Please click it again."}, 500


def answer_follow_up(assistant: Any, inspection: Inspection, question: str,
                     strokes: list[list[tuple[float, float]]] | None = None,
                     language: str = "en") -> tuple[dict[str, Any], int]:
    """Ask one more question about a clicked thing: (the JSON answer, the HTTP status).

    An object or an area: the assistant gets the stored photo, label, description
    and earlier questions, and with brush strokes also a copy of the photo with
    the marked part outlined. A recognised person: only text goes to the chat,
    with a sentence naming them; their picture is never sent anywhere. The answer
    is in `language` ("en" or "ar").
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
            answer = assistant.ask(named, language=language)
        else:
            log.info("Answering a follow-up question about %s%s",
                     f"a {inspection.label}" if inspection.label else "a clicked area",
                     " (a marked part)" if strokes is not None else "")
            answer = ask_about_object(assistant, inspection, question, strokes, language)
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
                     strokes: list[list[tuple[float, float]]] | None, language: str = "en") -> Any:
    """The assistant's answer about an object or an area, with the marked copy when there are strokes.

    The marking is drawn here on the server's own photo; the browser sent only the points.
    """
    photo, label = inspection.photo, inspection.label or None
    if strokes is None:
        return assistant.follow_up(photo, label, inspection.info, inspection.pairs, question,
                                   language=language)
    marked = mark_photo(photo, strokes, max_bytes=MAX_IMAGE_BYTES)   # never over what the assistant takes
    return assistant.follow_up(photo, label, inspection.info, inspection.pairs, question, marked=marked,
                               language=language)


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


def build_hosted_voice(config: dict[str, Any]) -> HostedVoice | None:
    """The hosted voice when voice.hosted is true; None when it is off or a setting is wrong (one line says which)."""
    settings = config.get("voice")
    if not isinstance(settings, dict) or settings.get("hosted") is not True:
        return None
    try:
        return HostedVoice(settings)
    except ConfigError as error:
        print(f"The hosted voice is off: {error}")
        return None


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


def take_start_lock(path: Path) -> Any | None:
    """Hold a lock on this file for as long as the program runs. None: another copy holds it.

    The port is only taken once the models are loaded, so during the first minute a
    second copy cannot see the first one on the port; it sees this lock instead.
    Windows lets go of the lock by itself when the program ends, however it ends,
    so a crash never leaves a lock behind. Where msvcrt is missing (not Windows)
    the file is opened but not locked, so the program still starts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")                    # for writing, without emptying it
    try:
        import msvcrt
    except ImportError:
        return handle
    try:
        handle.seek(0)                            # every copy locks the same first byte
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:                               # another copy holds it
        handle.close()
        return None
    return handle


def show_running_copy(url: str, serving: bool, open_page: bool) -> int:
    """What a second start does while another copy holds the start lock: the exit code."""
    if not serving:                               # the other copy is still loading its models
        print(STILL_STARTING)
        return 1
    print(f"The demo page is already running at {url}.")
    if not open_page:
        return 1
    print("Opening it in the browser.")
    webbrowser.open(url)
    return 0


def open_browser_soon(url: str) -> None:
    timer = threading.Timer(BROWSER_DELAY_SECONDS, webbrowser.open, args=(url,))
    timer.daemon = True
    timer.start()


def through_phone_door(app: Any) -> Callable[[dict, Any], Any]:
    """The app as the phone door's server runs it: every request is marked as one through the
    phone door. Only this wrapper sets the mark; the browser has no way to set it."""
    def marked(environ: dict, start_response: Any) -> Any:
        environ[DOOR_ENVIRON] = PHONE
        return app(environ, start_response)
    return marked


def open_phone_door(app: Flask, door: PhoneDoor | None) -> Any | None:
    """Start the phone door's HTTPS server in a thread of its own and print its one line.

    None when there is no door, or when it could not open: then one plain line says
    why (the Camera box shows the same), and the program runs on without the door.
    """
    if door is None:
        return None
    try:
        server = start_door_server(app, door)
    except DoorError as error:                        # a plain message that says what to do
        line = f"Phone door closed: {error}"
    except Exception as error:                        # anything else: its kind only, never its text
        line = f"Phone door closed: {type(error).__name__} while opening it. The page runs on without it."
    else:
        print(f"Phone door: {'  '.join(door.addresses)}  (password: {PASSWORD_VARIABLE} in .env)")
        return server
    door.shut(line)
    print(line)
    return None


def start_door_server(app: Flask, door: PhoneDoor) -> Any:
    """Read the password, find the addresses, make the certificate, and serve on 0.0.0.0:<port>."""
    port = door.settings.port
    door.set_password(read_password())
    addresses = lan_addresses()
    if not addresses:
        raise DoorError(NO_NETWORK)
    certificate, key = ensure_certificate(door.settings.certificate_dir, addresses)
    context = tls_context(certificate, key)
    if already_serving("127.0.0.1", port):            # Windows would let a second server bind the port
        raise DoorError(f"Port {port} is taken by another program. Stop it, or choose another "
                        "phone_door.port in config.yaml.")
    try:
        server = make_door_server(app, "0.0.0.0", port, context)
    except OSError as error:
        raise DoorError(f"The phone door cannot use port {port} ({type(error).__name__}). Stop the program "
                        "that uses it, or choose another phone_door.port in config.yaml.") from None
    threading.Thread(target=server.serve_forever, name="phone door", daemon=True).start()
    door.opened([f"https://{address}:{port}/" for address in addresses])
    return server


class PhoneDoorHandler(WSGIRequestHandler):
    """werkzeug's request handler for one phone-door connection, in that connection's own thread.

    The TLS handshake happens here, not in the accept loop that every phone shares, so a
    client that connects and sends nothing holds up only its own thread. Every wait on
    the connection (the handshake, reading a request, an idle keep-alive) ends after
    DOOR_SOCKET_SECONDS; socketserver sets that time limit from `timeout`.
    """

    timeout = DOOR_SOCKET_SECONDS

    def handle(self) -> None:
        try:
            self.connection.do_handshake()
        except (OSError, ValueError) as error:        # too slow, the certificate refused, not TLS at all
            self.log_error("The TLS handshake did not finish (%s)", type(error).__name__)
            return
        super().handle()

    def log_error(self, format: str, *args: Any) -> None:
        """Slow or dropped connections are common on a network: logged quietly, not as errors."""
        self.log("info", format, *args)


def make_door_server(app: Flask, host: str, port: int, context: ssl.SSLContext) -> Any:
    """werkzeug's threaded server for the phone door, with TLS done by PhoneDoorHandler."""
    server = make_server(host, port, through_phone_door(app), threaded=True, request_handler=PhoneDoorHandler)
    server.socket = wrap_listener(server.socket, context)
    server.ssl_context = context                      # so each request's environ says https
    return server


def wrap_listener(listener: socket.socket, context: ssl.SSLContext) -> ssl.SSLSocket:
    """TLS on the listening socket, without a handshake inside accept()."""
    return context.wrap_socket(listener, server_side=True, do_handshake_on_connect=False)


def tls_context(certificate: Path, key: Path) -> ssl.SSLContext:
    """The server's TLS settings with this certificate; a DoorError when the files do not load."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain(certificate, key)
    except OSError as error:                          # ssl.SSLError is an OSError too
        raise DoorError(f"The certificate in {certificate.parent} could not be loaded "
                        f"({type(error).__name__}). Delete the files in that folder and start again.") from None
    return context


def close_phone_door(server: Any | None) -> bool:
    """Stop the phone door's server, waiting at most DOOR_CLOSE_SECONDS; never hangs the program.

    False when it did not stop in time: its threads are daemon threads, so it ends with
    the program anyway.
    """
    if server is None:
        return True
    closer = threading.Thread(target=shut_down, args=(server,), name="phone door closing", daemon=True)
    closer.start()
    closer.join(DOOR_CLOSE_SECONDS)
    if closer.is_alive():
        print("The phone door did not close in time; it stops with the program.")
        return False
    return True


def shut_down(server: Any) -> None:
    try:
        server.shutdown()
        server.server_close()
    except OSError:                                   # the program is stopping anyway
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")      # an odd character in a name never crashes logging
    try:
        config = with_overrides(load_config(), args)
        settings = section(config, "app")
        host: str = require(settings, "host", str)
        if not is_loopback(host):                     # the laptop's door stays on the laptop
            raise ConfigError(NOT_LOCAL.format(host=host))
        port: int = require(settings, "port", int)
        log_file = resolve_path(require(settings, "log_file", str))
        door_settings = read_settings(config)         # None when phone_door.enabled is false
    except ConfigError as error:
        print(f"Cannot start: {error}")
        return 1
    url = f"http://{host}:{port}/"
    # Both checks come before any model is loaded. The lock stays held while this local
    # variable lives, that is until main() returns (or the program dies).
    start_lock = take_start_lock(log_file.parent / START_LOCK_FILE)
    serving = already_serving(host, port)
    if start_lock is None:                            # another copy of this program runs
        return show_running_copy(url, serving, args.open)
    if serving:                                       # another program answers on the port
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
    voice = build_hosted_voice(config)               # None: the page uses only the browser's own voice
    door = PhoneDoor(door_settings) if door_settings is not None else None
    app = create_app(pipeline, assistant, camera_file, build_people_book(config),
                     discover_every=discover_interval(config), voice=voice, door=door)
    door_server = open_phone_door(app, door)          # None when the door is off or could not open

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
        try:
            close_phone_door(door_server)             # at most DOOR_CLOSE_SECONDS
        finally:
            pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
