"""The demo page: live video, what is in view, details on a click, reactions and the chat.

Served by Flask; this is the only module that imports flask. A click on the
video goes to /api/inspect, which answers at once with a small photo of what
was clicked; for an object, /api/describe then asks the assistant what it is.
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
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

from car_vision.assistant import Assistant, AssistantError
from car_vision.config import ConfigError, load_config, require, resolve_path, section
from car_vision.pipeline import Pipeline
from car_vision.stream import SOURCES

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
VIDEO_MAX_FPS = 20               # /video sends at most this many frames a second
VIDEO_POLL_SECONDS = 0.01        # how often /video looks for a new frame
BOUNDARY = "frame"               # separates the pictures in the MJPEG stream
BROWSER_DELAY_SECONDS = 1.0      # --open waits this long, so the server is up first
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

MAX_INSPECTIONS = 20             # /api/describe knows the last this many clicks
MAX_ANSWERS = 50                 # answers kept for repeated clicks on the same tracked object
ANSWER_SECONDS = 600.0           # ... for ten minutes
PHOTO_PREFIX = "data:image/jpeg;base64,"
GONE = "That is no longer in view."
CLICK_AGAIN = "That click is too old. Please click the object again."


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

    def reset(self) -> None:
        pass


# ---- what the server remembers about clicks -------------------------------------------


@dataclass(frozen=True)
class Inspection:
    """One click, kept so /api/describe needs only its id: the browser never sends a picture."""

    kind: str                    # "object", "person" or "area"
    label: str
    track_id: int | None
    photo: bytes


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
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (x, y)):
            return "x and y must be from 0 to 1."
    return {"x": None if x is None else float(x), "y": None if y is None else float(y),
            "track_id": track_id, "label": label}


# ---- the web server ------------------------------------------------------------


def create_app(pipeline: Any, assistant: Any) -> Flask:
    """The Flask app for one pipeline and one assistant (real ones, or fakes in the tests)."""
    app = Flask(__name__)
    inspections = InspectionStore()
    answers = AnswerCache()

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
        if not question:
            return jsonify(error="The question is empty."), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify(error=f"The question is too long ({len(question)} characters). "
                                 f"Please keep it under {MAX_QUESTION_CHARS} characters."), 400
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
        return jsonify(text=answer.text, used_web=bool(answer.used_web), sources=list(answer.sources))

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
        inspection = Inspection(target.kind, target.label, target.track_id, target.photo)
        key = cache_key(inspection)
        return jsonify(
            id=inspections.add(inspection),
            kind=target.kind, label=target.label,
            confidence=round(float(target.confidence), 2),
            track_id=int(target.track_id) if target.track_id is not None else None,
            photo=photo_address(target.photo),
            person=person_json(target.person),
            describable=target.kind != "person",       # a person's picture is never sent to the API
            info=answers.get(key) if key is not None else None,
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
        cached = answers.get(key) if key is not None else None
        if cached is not None:
            return jsonify(cached)
        log.info("Looking up %s", f"a {inspection.label}" if inspection.label else "a clicked area")
        try:
            info = info_json(assistant.describe(inspection.photo, inspection.label or None))
        except ValueError as error:                   # the assistant's own check of the picture or label
            return jsonify(error=str(error)), 400
        except AssistantError as error:               # no key, no internet, busy, refusal ...
            return jsonify(error=str(error))
        except Exception as error:
            log.exception("Looking up a clicked object failed")
            return jsonify(error=f"Something went wrong while looking it up ({type(error).__name__}). "
                                 "Please click it again."), 500
        if key is not None:
            answers.put(key, info)
        return jsonify(info)

    return app


def mjpeg_stream(pipeline: Any, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> Iterator[bytes]:
    """The parts of the MJPEG stream: each new frame once, at most VIDEO_MAX_FPS a second.

    The pipeline gives a new bytes object for every new picture, so `is`
    tells a new frame from the one already sent.
    """
    last: bytes | None = None
    next_send = 0.0
    while True:
        jpeg = pipeline.latest_jpeg()
        now = clock()
        if jpeg is None or jpeg is last or now < next_send:
            sleep(VIDEO_POLL_SECONDS)
            continue
        last = jpeg
        next_send = now + 1.0 / VIDEO_MAX_FPS
        yield jpeg_part(jpeg)


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
    setup_logging(log_file)

    try:
        pipeline = Pipeline(config)
    except ConfigError as error:
        print(f"Cannot start: {error}")
        return 1
    assistant = build_assistant(config, pipeline)
    app = create_app(pipeline, assistant)

    url = f"http://{host}:{port}/"
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
