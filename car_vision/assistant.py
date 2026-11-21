"""The assistant: answers questions from the team's notes, and says what an object is.

Every question becomes one Messages API call to Claude. The whole notes file
and what the team wrote about each person (data/people/<id>/about.md) sit in
the system prompt, which never changes during a run, so it is cached. Who the
camera recognises right now, and which objects the detector reports, are added
as lines in front of the question. When the notes do not cover a question,
Claude may use Anthropic's web search tool, which runs on Anthropic's servers.

`Assistant.describe` is separate from the chat: it sends one small picture of
an object the visitor clicked, with the detector's label, and reads back a
name, a short description and a few specifications. It has no memory and its
own system prompt.

This is the only module that imports the `anthropic` library. `Assistant`
talks to an `LLMClient`, so a fake (in the tests) or another provider can
stand in for Claude.

Run it on its own for a typed chat:

    .venv\\Scripts\\python.exe -m car_vision.assistant

Type /reset to forget the conversation, /quit or an empty line to stop.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import anthropic
import dotenv

from car_vision.config import PROJECT_ROOT, ConfigError, load_config, require, resolve_path, section
from car_vision.state import WorldState

log = logging.getLogger(__name__)

ENV_FILE = PROJECT_ROOT / ".env"
API_KEY_VARIABLE = "ANTHROPIC_API_KEY"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Request details that come from the Claude API reference, not from config.yaml.
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"     # newest web search; Sonnet 5.5 and Opus 5.5 take it
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # needed for fallbacks="default" (on refusal,
                                                   # Anthropic picks the model to retry on)
THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"  # needed for block_binding
MAX_CONTINUATIONS = 5     # how often a paused web search turn is resumed before giving up

# Limits that keep a demo from hanging or filling up.
REQUEST_TIMEOUT_SECONDS = 60.0   # one HTTP request to the API
MAX_RETRIES = 1                  # the SDK tries a failed request once more
LOCK_WAIT_SECONDS = 90.0         # how long a second question waits for the first
DESCRIBE_WAIT_SECONDS = 30.0     # how long a second look-up waits for the first
MAX_QUESTION_CHARS = 2000
MAX_LABEL_CHARS = 60             # a detector label is a word or two
MAX_SPECS = 6
MAX_IMAGE_BYTES = 1_500_000      # one camera picture (1.5 MB)
JPEG_MARKER = bytes([0xFF, 0xD8, 0xFF])   # every JPEG file starts with these bytes

PERSON_ID = re.compile(r"[a-z0-9_]+")    # a person folder name: small letters, digits, underscores
ABOUT_FILE = "about.md"

PLACEHOLDER = "[REPLACE]"
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")
# A UTF-8 byte-order mark as it arrives when piped input is read as UTF-8 or as cp1252.
BOM = chr(0xFEFF)                                    # the mark, read as UTF-8
CP1252_BOM = chr(0xEF) + chr(0xBB) + chr(0xBF)       # the same three bytes, read as cp1252
BOM_FORMS = (BOM, CP1252_BOM)

Turn = dict[str, Any]     # {"role": "user" | "assistant", "content": ...}


class AssistantError(Exception):
    """Something the user can act on: no key, no internet, rate limit, refusal."""


@dataclass(frozen=True)
class Answer:
    text: str
    used_web: bool = False
    sources: tuple[str, ...] = ()        # page URLs, in the order cited, no repeats


@dataclass(frozen=True)
class ObjectInfo:
    """What `Assistant.describe` found out about one object in a picture."""

    name: str                              # what the thing is, e.g. "Laptop" or "Office chair"
    summary: str                           # two or three sentences
    specs: tuple[tuple[str, str], ...]     # (name, value) pairs, at most six
    used_web: bool = False
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMReply:
    """What a client hands back for one question.

    `new_turns` are the assistant turns to add to the conversation, exactly as
    the provider returned them. There is usually one; a paused web search adds
    one more for each time the turn was resumed.
    """

    text: str
    new_turns: tuple[Turn, ...]
    used_web: bool = False
    sources: tuple[str, ...] = ()


class LLMClient(Protocol):
    """The seam that lets another provider or a local model replace Claude later.

    `turns` is the whole conversation, oldest first. A user turn's content is
    the text, or, for `Assistant.describe`, a list of two blocks: the picture
    (see `user_turn`) and then the text.
    """

    def complete(self, system: str, turns: list, allow_web: bool) -> "LLMReply": ...


@dataclass(frozen=True)
class Brief:
    """What the team wrote about one person the camera can recognise."""

    person_id: str      # the folder name
    name: str           # display name: the first "# " heading, else the folder name
    text: str           # the whole about.md, without HTML comments


# ---- notes, briefs and prompts -------------------------------------------------


def shown_path(path: Path) -> str:
    """A path as the user may see it: relative to the project folder, such as
    data/knowledge.md. Outside the project only its last folder and name show."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return f"{path.parent.name}/{path.name}"


def read_utf8(path: Path, what: str) -> str:
    """Read a hand-written text file as UTF-8, accepting a byte-order mark.

    Any problem becomes an AssistantError with a plain sentence.
    """
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        problem = AssistantError(
            f"The {what} {shown_path(path)} is not saved as UTF-8. "
            "Open it in a text editor and save it again with the UTF-8 encoding."
        )
    except OSError as error:                 # e.g. PermissionError while another program holds it
        problem = AssistantError(
            f"The {what} {shown_path(path)} could not be read ({type(error).__name__}). "
            "Close any program that has it open and try again."
        )
    raise problem from None


def load_notes(path: Path) -> str:
    """Read the team's notes file. An empty file is allowed."""
    if not path.is_file():
        raise AssistantError(
            f"The notes file {shown_path(path)} was not found. "
            "Create it, or point assistant.notes_file in config.yaml at the right file."
        )
    return read_utf8(path, "notes file")


def load_briefs(people_dir: Path) -> list[Brief]:
    """Every <people_dir>/<id>/about.md, in sorted id order.

    A missing folder, a person without about.md, or an empty about.md adds
    nothing. A folder name that is not a valid id, or an about.md that cannot
    be read, is skipped with one warning, so the assistant still starts.
    """
    if not people_dir.is_dir():
        return []
    briefs = []
    for folder in sorted((p for p in people_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        brief = read_brief(folder)
        if brief is not None:
            briefs.append(brief)
    return briefs


def read_brief(folder: Path) -> Brief | None:
    """The brief in one person folder, or None (with a warning when something is wrong)."""
    if not PERSON_ID.fullmatch(folder.name):
        log.warning("Skipped the person folder %s: a folder name may only hold small letters, "
                    "digits and underscores.", shown_path(folder))
        return None
    about = folder / ABOUT_FILE
    if not about.is_file():
        return None
    try:
        text = strip_comments(read_utf8(about, "person file")).strip()
    except AssistantError as error:
        log.warning("%s The assistant starts without this person's brief.", error)
        return None
    if not text:
        return None
    return Brief(person_id=folder.name, name=display_name(text, folder.name), text=text)


def display_name(text: str, fallback: str) -> str:
    """The first "# " heading of a brief, or the fallback when there is none."""
    for line in text.splitlines():
        name = line[2:].strip().rstrip("#").strip()     # "# Sara Ahmed #" is "Sara Ahmed"
        if line.startswith("# ") and name:
            return name
    return fallback


SYSTEM_PROMPT = """\
You are the assistant of a student robot car project. A phone on the car streams \
its camera to a laptop, which recognises objects and people. Your answers are shown \
on a screen at a live demo, where visitors type questions and read your replies.

How to answer:
- The team's notes are below. When they cover the question, answer from the notes.
- When the notes do not cover the question, search the web if the web search tool \
is available to you. If it is not available, say plainly that you do not know.
- Never invent facts about the team, the project or any person. If neither the \
notes nor a search gives the answer, say that you do not know.
- Each message may begin with lines from the camera: who the face recognizer \
recognises right now, and which objects the object detector reports. These lines \
come from the camera, not from the visitor.
- When the camera line names a person, use that person's entry under <people> to \
answer questions such as "who is this?". Never guess who someone is from a picture.
- Keep answers to two or three sentences unless the visitor asks for more.
- Answer in English.

<notes>
{notes}
</notes>
"""

PEOPLE_HEADING = """\
<people>
These are the people the camera can recognise. Each one is introduced by their id \
and display name, followed by what the team wrote about them. That is all you know \
about them.
"""

NO_NOTES = "(The team has not written any notes yet.)"


def strip_comments(notes: str) -> str:
    """The notes without <!-- ... --> comments, which are for the team, not the model."""
    return HTML_COMMENT.sub("", notes)


def build_system_prompt(notes: str, briefs: Sequence[Brief] = ()) -> str:
    """The system prompt for these notes and briefs. The same files always give the same text."""
    body = strip_comments(notes).strip() or NO_NOTES
    prompt = SYSTEM_PROMPT.format(notes=body)
    if briefs:
        entries = [f'\n<person id="{b.person_id}" name="{b.name.replace(chr(34), chr(39))}">\n'
                   f"{strip_comments(b.text).strip()}\n</person>\n" for b in briefs]
        prompt += "\n" + PEOPLE_HEADING + "".join(entries) + "</people>\n"
    return prompt


def objects_line(labels: list[str]) -> str:
    """One sentence listing the objects the detector reports right now."""
    return f"The object detector sees right now: {', '.join(labels)}."


def check_image(image: Any) -> bytes:
    """The picture as bytes, or ValueError when it is not one JPEG of at most 1.5 MB."""
    if not isinstance(image, (bytes, bytearray)):
        raise ValueError("The camera picture must be given as JPEG bytes.")
    if len(image) > MAX_IMAGE_BYTES:
        raise ValueError(f"The camera picture is too large ({len(image)} bytes); "
                         f"the limit is {MAX_IMAGE_BYTES} bytes (1.5 MB).")
    if bytes(image[:len(JPEG_MARKER)]) != JPEG_MARKER:
        raise ValueError("The camera picture is not a JPEG image.")
    return bytes(image)


def user_turn(text: str, image: bytes | None = None) -> Turn:
    """The user turn: the text alone, or the picture followed by the text,
    in the image block shape of the Claude API reference (base64 source)."""
    if image is None:
        return {"role": "user", "content": text}
    picture = {"type": "image",
               "source": {"type": "base64", "media_type": "image/jpeg",
                          "data": base64.standard_b64encode(image).decode("ascii")}}
    return {"role": "user", "content": [picture, {"type": "text", "text": text}]}


def people_line(names: list[str]) -> str:
    """One sentence saying who the camera recognises right now."""
    if not names:
        return "Nobody recognised is in front of the camera right now."
    return f"In front of the camera right now: {', '.join(names)}."


def labels_in_view(state: WorldState, max_age_seconds: float) -> list[str]:
    """The distinct object labels the detector reports right now, in sorted
    order, without "person". Empty when the detections are older than max_age_seconds."""
    snap = state.snapshot()
    if time.monotonic() - snap.detections_at > max_age_seconds:
        return []
    return sorted({d.label for d in snap.detections if d.label != "person"})


# ---- describing one object ---------------------------------------------------


DESCRIBE_PROMPT = """\
You look at pictures for a student robot car project. A phone on the car streams \
its camera to a laptop, which detects objects. A visitor at a live demo clicked \
something in the video. The picture you get is a small cut-out of the camera \
picture around it.

How to answer:
- Say what the thing in the middle of the picture is, as exactly as the picture allows.
- The object detector's label is a hint, not the truth.
- Name a brand or model only when you can read it in the picture or it is unmistakable.
- Give up to six specifications a curious visitor would want, such as size, weight, \
material, power and typical price. When you do not know the exact model, give \
typical values and say that they are typical.
- When you cannot tell what the thing is, or you can see a brand or model whose \
details you do not know, search the web if the web search tool is available to you.
- Never invent facts. If you still cannot tell what the thing is, use the name \
Unknown object and say that you cannot tell.
- If the thing in the middle is a person, use the name Person and say nothing about \
who they are or how they look.
- Say nothing about any people elsewhere in the picture.
- Answer in English, in exactly this shape and nothing else:

NAME: <a few words>
ABOUT: <two or three sentences>
SPECS:
- <name>: <value>
- <name>: <value>
"""

UNKNOWN_OBJECT = "Unknown object"
DESCRIBE_BUSY_MESSAGE = ("The assistant is still busy looking up the previous object. "
                         "Please wait and try again.")

# A key line of the answer: NAME, ABOUT or SPECS in any letter case, perhaps in
# Markdown bold or after a heading mark, then a colon. It is never a bullet.
KEY_LINE = re.compile(r"^\s*#*\s*(?:\*\*)?\s*(name|about|specifications|specs)\s*(?:\*\*)?\s*:"
                      r"\s*(?:\*\*)?(.*)$", re.IGNORECASE)
# The specs heading alone on its line, without a colon: "## Specs", "**SPECS**", "Specifications".
SPECS_HEADING = re.compile(r"^\s*#*\s*(?:\*\*)?\s*(?:specifications|specs)\s*(?:\*\*)?\s*$",
                           re.IGNORECASE)
# "Sources:" after the specs: the end of the answer's parts.
SOURCES_LINE = re.compile(r"^\s*#*\s*(?:\*\*)?\s*sources?\s*(?:\*\*)?\s*:", re.IGNORECASE)
# A bullet: "-", "*", "•", an en or em dash, or a number such as "1." or "1)".
BULLET = re.compile(r"^\s*(?:[-*•–—]|\d+[.)])\s+")


def label_text(label: str | None) -> str:
    """The sentence that tells the model what the detector thinks the thing is."""
    if label is not None and not isinstance(label, str):
        raise ValueError("The detector label must be text.")
    if label is not None and len(label) > MAX_LABEL_CHARS:
        raise ValueError(f"The detector label is too long ({len(label)} characters); "
                         f"the limit is {MAX_LABEL_CHARS}.")
    words = " ".join((label or "").split())
    if not words:
        return ("The object detector has no label for this. "
                "The visitor clicked the middle of the picture.")
    return f'The object detector labelled this "{words}".'


def parse_object_info(text: str, label: str | None) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Read (name, summary, specs) out of the model's NAME / ABOUT / SPECS answer.

    Forgiving: keys in any letter case and in Markdown bold, a SPECS heading
    without a colon, an ABOUT over several lines, dash, star or numbered
    bullets. Spec lines without a colon, or named like a link, are skipped;
    a "Sources:" line ends the specs; only the first six specs are kept.
    An empty or missing NAME gives the label's name. An answer with no key
    line at all becomes the summary as a whole.
    """
    name, about, specs = "", [], []
    part = ""                     # where we are: "name", "about", "specs", or "end" after Sources
    found_a_key = False
    for line in text.splitlines():
        key, rest = answer_key(line)
        if key:
            found_a_key, part = True, key
            if key == "name" and not name:
                name = rest
            elif key == "about" and rest:
                about.append(rest)
            elif key == "specs":
                add_spec(specs, rest)          # "SPECS: - Weight: 1.5 kg" on one line
        elif SOURCES_LINE.match(line):
            part = "end"
        elif part == "about" and line.strip():
            about.append(line.strip())
        elif part == "specs":
            add_spec(specs, line)
    if not found_a_key:
        return fallback_name(label), text.strip(), ()
    return name or fallback_name(label), " ".join(about), tuple(specs[:MAX_SPECS])


def answer_key(line: str) -> tuple[str, str]:
    """("name" | "about" | "specs", the rest of the line) for a key line, else ("", "")."""
    match = KEY_LINE.match(line)
    if match:
        key = match.group(1).lower()
        return ("specs" if key == "specifications" else key), without_bold(match.group(2))
    if SPECS_HEADING.match(line):
        return "specs", ""
    return "", ""


def without_bold(text: str) -> str:
    """Text without Markdown bold marks and without spaces around it."""
    return text.replace("**", "").strip()


def add_spec(specs: list[tuple[str, str]], line: str) -> None:
    """Add the line to specs when it is a "- name: value" line."""
    pair = spec_pair(line)
    if pair:
        specs.append(pair)


def spec_pair(line: str) -> tuple[str, str] | None:
    """One "- name: value" line as (name, value), or None when it is not one."""
    body = without_bold(BULLET.sub("", line))
    if ":" not in body:
        return None
    name, value = (part.strip() for part in body.split(":", 1))
    if not name or not value or looks_like_link(name):
        return None
    return name, value


def looks_like_link(name: str) -> bool:
    """True for a "name" that is really a web address or a citation mark like [1]."""
    return "/" in name or name.lower().startswith("http") or name.startswith("[")


def fallback_name(label: str | None) -> str:
    """The label with a capital first letter, or "Unknown object"."""
    words = " ".join((label or "").split())
    return words[:1].upper() + words[1:] if words else UNKNOWN_OBJECT


# ---- the assistant -----------------------------------------------------------


BUSY_MESSAGE = "The assistant is still busy with the previous question. Please wait and try again."


class Assistant:
    """Keeps one conversation and asks the client one question at a time.

    `describe` runs beside the chat with its own lock, so a slow look-up does
    not hold up a question.
    """

    def __init__(
        self,
        settings: dict,
        state: WorldState | None = None,
        client: LLMClient | None = None,
    ) -> None:
        provider: str = require(settings, "provider", str)
        require(settings, "model", str)
        require(settings, "effort", str)
        require(settings, "max_tokens", int)
        notes_file: str = require(settings, "notes_file", str)
        people_dir: str = require(settings, "people_dir", str)
        self.web_search: bool = require(settings, "web_search", bool)
        require(settings, "web_search_max_uses", int)
        self.history_turns: int = require(settings, "history_turns", int)
        self.seen_within_seconds = float(require(settings, "seen_within_seconds", (int, float)))
        if provider != "anthropic":
            raise ConfigError(f"assistant.provider must be 'anthropic', got {provider!r}")

        notes_path = resolve_path(notes_file)
        notes = load_notes(notes_path)
        if PLACEHOLDER in strip_comments(notes):
            log.warning("The notes file %s still has placeholder lines (%s). "
                        "Replace them with real facts before the demo.", notes_path, PLACEHOLDER)
        briefs = load_briefs(resolve_path(people_dir))
        self._system = build_system_prompt(notes, briefs)
        self._state = state
        self._client: LLMClient = client if client is not None else AnthropicClient(settings)
        self._lock = threading.Lock()              # the chat
        self._describe_lock = threading.Lock()     # describe, so it never waits for the chat
        self._history: list[Turn] = []
        self._exchanges = 0      # question-answer pairs in the current conversation

    def ask(self, question: str) -> Answer:
        """Answer one question. Waits a while if another question is still being answered."""
        text = clean(question)
        if not text:
            raise ValueError("The question is empty.")
        if len(text) > MAX_QUESTION_CHARS:
            raise ValueError(f"The question is too long ({len(text)} characters). "
                             f"Please keep it under {MAX_QUESTION_CHARS} characters.")
        self._acquire()
        try:
            # The history is only ever added to. Once it holds history_turns
            # exchanges, the next question starts a fresh conversation instead
            # of cutting old turns out (that would change earlier turns).
            fresh = self._exchanges >= self.history_turns
            earlier = [] if fresh else self._history
            turns = earlier + [{"role": "user", "content": self._user_content(text)}]
            reply = self._client.complete(self._system, turns, self.web_search)
            # Only a finished answer is kept; a failed question leaves the history as it was.
            self._history = turns + list(reply.new_turns)
            self._exchanges = 1 if fresh else self._exchanges + 1
            return Answer(text=reply.text, used_web=reply.used_web,
                          sources=unique(reply.sources))
        finally:
            self._lock.release()

    def reset(self) -> None:
        """Forget the conversation."""
        self._acquire()
        try:
            self._history = []
            self._exchanges = 0
        finally:
            self._lock.release()

    def describe(self, image: bytes, label: str | None = None) -> ObjectInfo:
        """Say what the object in one JPEG picture is. `label` is the detector's
        label, or None when the visitor clicked where the detector found nothing.

        One call with no memory: the chat history is neither sent nor changed.
        """
        picture = check_image(image)
        text = label_text(label)
        if not self._describe_lock.acquire(timeout=DESCRIBE_WAIT_SECONDS):
            raise AssistantError(DESCRIBE_BUSY_MESSAGE)
        try:
            reply = self._client.complete(DESCRIBE_PROMPT, [user_turn(text, picture)], self.web_search)
        finally:
            self._describe_lock.release()
        name, summary, specs = parse_object_info(reply.text, label)
        return ObjectInfo(name=name, summary=summary, specs=specs,
                          used_web=reply.used_web, sources=unique(reply.sources))

    def _acquire(self) -> None:
        if not self._lock.acquire(timeout=LOCK_WAIT_SECONDS):
            raise AssistantError(BUSY_MESSAGE)

    def _user_content(self, question: str) -> str:
        """The question after the camera lines: who is recognised, then which objects."""
        if self._state is None:
            return question
        lines = [people_line(self._state.current_people(self.seen_within_seconds))]
        labels = labels_in_view(self._state, self.seen_within_seconds)
        if labels:
            lines.append(objects_line(labels))
        return "\n".join(lines + [question])


def clean(text: str) -> str:
    """Typed text without spaces around it or the byte-order mark that
    Windows PowerShell puts in front of piped input."""
    text = text.strip()
    while text.startswith(BOM_FORMS):
        for form in BOM_FORMS:
            if text.startswith(form):
                text = text[len(form):]
        text = text.strip()
    return text


def unique(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """The items in their first order, each only once."""
    return tuple(dict.fromkeys(items))


# ---- Claude ----------------------------------------------------------------


def read_api_key() -> str:
    """The API key from .env or the environment. Never printed or logged."""
    dotenv.load_dotenv(ENV_FILE)
    key = os.environ.get(API_KEY_VARIABLE, "").strip()
    if not key:
        raise AssistantError(
            "No API key found. Copy .env.example to .env in the project folder "
            f"and put your Anthropic API key after {API_KEY_VARIABLE}=."
        )
    return key


def without_key(text: str, key: str = "") -> str:
    """Text with the API key, and anything shaped like one, cut out."""
    if key:
        text = text.replace(key, "[key removed]")
    return KEY_PATTERN.sub("[key removed]", text)


def api_message(error: Any, key: str = "") -> str:
    """The API's own explanation of an error, short and without the key."""
    body = getattr(error, "body", None)
    message = ""
    if isinstance(body, dict):
        detail = body.get("error")
        if isinstance(detail, dict) and isinstance(detail.get("message"), str):
            message = detail["message"]
    if not message:
        message = str(getattr(error, "message", "") or "")
    message = " ".join(without_key(message, key).split())
    return message[:300]


@dataclass
class _ReplyParts:
    texts: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    used_web: bool = False


class AnthropicClient:
    """Asks Claude through the Anthropic Messages API (implements LLMClient)."""

    def __init__(self, settings: dict) -> None:
        self.model: str = require(settings, "model", str)
        self.effort: str = require(settings, "effort", str)
        self.max_tokens: int = require(settings, "max_tokens", int)
        self.web_search_max_uses: int = require(settings, "web_search_max_uses", int)
        if self.effort not in EFFORT_LEVELS:
            raise ConfigError(f"assistant.effort must be one of {EFFORT_LEVELS}, got {self.effort!r}")
        self._sdk: Any = None      # created on the first question
        self._key = ""

    def complete(self, system: str, turns: list, allow_web: bool) -> LLMReply:
        messages = list(turns)               # never change the caller's list
        new_turns: list[Turn] = []
        responses: list[Any] = []
        for _ in range(MAX_CONTINUATIONS + 1):
            response = self._send(system, messages, allow_web)
            self._check_stop(response)
            turn = {"role": "assistant", "content": response.content}   # exactly as returned
            messages.append(turn)
            new_turns.append(turn)
            responses.append(response)
            if response.stop_reason != "pause_turn":
                return self._reply(responses, new_turns)
            # pause_turn: the server paused a long web search. Sending the paused
            # turn back, with no extra user message, lets it carry on.
        raise AssistantError("The web search took too long. Please ask again, perhaps more simply.")

    def request(self, system: str, messages: list, allow_web: bool) -> dict[str, Any]:
        """The keyword arguments for one Messages API call."""
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "thinking": {"type": "adaptive",
                         "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
            "output_config": {"effort": self.effort},
            "fallbacks": "default",
            "betas": [FALLBACK_BETA, THINKING_BINDING_BETA],
        }
        if allow_web:
            request["tools"] = [{"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search",
                                 "max_uses": self.web_search_max_uses}]
        return request

    def _client(self) -> Any:
        if self._sdk is None:
            self._key = read_api_key()
            self._sdk = anthropic.Anthropic(api_key=self._key, timeout=REQUEST_TIMEOUT_SECONDS,
                                            max_retries=MAX_RETRIES)
        return self._sdk

    def _send(self, system: str, messages: list, allow_web: bool) -> Any:
        try:
            sdk = self._client()
            response = sdk.beta.messages.create(**self.request(system, messages, allow_web))
        except AssistantError:
            raise                                # no key: already a plain message
        except anthropic.AnthropicError as error:
            problem = self._explain(error)
        except Exception as error:               # e.g. a ValueError the SDK raises before sending
            log.warning("The request to Claude failed before it was sent: %s", type(error).__name__)
            problem = AssistantError(
                f"The request to Claude could not be made ({type(error).__name__}). "
                "Check the assistant settings in config.yaml, such as max_tokens."
            )
        else:
            _log_dropped_thinking(response)
            return response
        # Raised outside the except blocks, so the SDK's exception (and the request
        # it carries, with the key in its headers) is not attached as __context__.
        raise problem from None

    def _explain(self, error: anthropic.AnthropicError) -> AssistantError:
        """One plain sentence for an SDK error, most specific case first."""
        if isinstance(error, anthropic.AuthenticationError):
            return AssistantError(f"The API key was not accepted. Check {API_KEY_VARIABLE} in the .env file.")
        if isinstance(error, anthropic.PermissionDeniedError):
            return AssistantError(
                f"This API key has no access to the model '{self.model}' (status 403): "
                f"{api_message(error, self._key)} Check assistant.model in config.yaml."
            )
        if isinstance(error, anthropic.NotFoundError):
            return AssistantError(
                f"The model name '{self.model}' was not found (status 404): "
                f"{api_message(error, self._key)} Check assistant.model in config.yaml."
            )
        if isinstance(error, anthropic.BadRequestError):
            return AssistantError(
                f"Claude rejected the request (status 400): {api_message(error, self._key)} "
                "Check the assistant settings in config.yaml."
            )
        if isinstance(error, anthropic.RateLimitError):
            return AssistantError("Too many questions at once (rate limit). Wait a minute and ask again.")
        if isinstance(error, (anthropic.APITimeoutError, anthropic.RequestTimeoutError,
                              anthropic.GatewayTimeoutError)):
            return AssistantError("Claude took too long to answer. Please ask again.")
        if isinstance(error, anthropic.APIConnectionError):
            return AssistantError("Could not reach the Anthropic API. Check the laptop's internet connection.")
        if isinstance(error, anthropic.APIStatusError):
            return AssistantError(
                f"The Anthropic API returned an error (status {error.status_code}). Try again in a moment."
            )
        return AssistantError("The request to the Anthropic API failed. Try again in a moment.")

    @staticmethod
    def _check_stop(response: Any) -> None:
        reason = response.stop_reason
        if reason == "refusal":
            # Fallbacks already ran on the server; a refusal here means every model declined.
            raise AssistantError("The assistant declined to answer that. Try asking another way.")
        if reason == "max_tokens":
            raise AssistantError(
                "The answer was cut off because it grew longer than assistant.max_tokens allows. "
                "Ask for a shorter answer, or raise max_tokens in config.yaml."
            )
        if reason == "model_context_window_exceeded":
            raise AssistantError("The conversation has grown too long. Type /reset and ask again.")

    @staticmethod
    def _reply(responses: list[Any], new_turns: list[Turn]) -> LLMReply:
        parts = _ReplyParts()
        for response in responses:
            for block in response.content:
                _read_block(block, parts)
        text = final_text(responses[-1].content)
        if not text:
            raise AssistantError("The assistant gave no answer. Please ask again.")
        sources = parts.cited if parts.cited else parts.found
        return LLMReply(text=text, new_turns=tuple(new_turns),
                        used_web=parts.used_web, sources=unique(sources))


def final_text(content: list[Any]) -> str:
    """The answer itself: the text after the last web search result, so the
    model's "let me search" narration before it is left out."""
    start = 0
    for index, block in enumerate(content):
        if getattr(block, "type", None) == "web_search_tool_result":
            start = index + 1
    texts = [block.text for block in content[start:] if getattr(block, "type", None) == "text"]
    return "".join(texts).strip()


def _read_block(block: Any, parts: _ReplyParts) -> None:
    """Collect web sources from any block of the exchange."""
    kind = getattr(block, "type", None)
    if kind == "text":
        for citation in getattr(block, "citations", None) or []:
            if getattr(citation, "type", None) == "web_search_result_location":
                parts.cited.append(citation.url)
    elif kind == "server_tool_use" and getattr(block, "name", None) == "web_search":
        parts.used_web = True
    elif kind == "web_search_tool_result":
        parts.used_web = True
        results = getattr(block, "content", None)
        if isinstance(results, list):            # an error comes back as one object instead
            parts.found.extend(r.url for r in results if getattr(r, "type", None) == "web_search_result")


def _log_dropped_thinking(response: Any) -> None:
    """Warn when the API had to drop earlier reasoning: a sign the history changed."""
    changes = getattr(response, "input_transformations", None) or []
    dropped = [c for c in changes if getattr(c, "type", None) == "thinking_dropped"]
    if dropped:
        reasons = sorted({str(getattr(c, "reason", "?")) for c in dropped})
        log.warning("The API dropped %d earlier thinking block(s): %s", len(dropped), ", ".join(reasons))


# ---- running this file on its own ----------------------------------------


def read_piped_input_as_utf8() -> None:
    """PowerShell pipes UTF-8 with a byte-order mark, but Python reads piped
    input with the Windows code page. Read it as UTF-8 and drop the mark."""
    stdin = sys.stdin
    if stdin is None or not hasattr(stdin, "reconfigure"):
        return
    try:
        if not stdin.isatty():
            stdin.reconfigure(encoding="utf-8-sig")
    except (OSError, ValueError):
        pass                                     # keep the default reading


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the project assistant in the terminal.")
    parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # an odd character never crashes the console
    read_piped_input_as_utf8()
    try:
        return chat()
    except KeyboardInterrupt:                      # Ctrl+C: leave quietly
        print()
        return 0


def chat() -> int:
    """The typed chat loop behind main()."""
    try:
        assistant = Assistant(section(load_config(), "assistant"))
    except (ConfigError, AssistantError) as error:
        print(error)
        return 1

    print("Ask a question. /reset forgets the conversation; /quit or an empty line stops.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        question = clean(line)
        if not question or question == "/quit":
            return 0
        try:
            if question == "/reset":
                assistant.reset()
                print("(conversation forgotten)")
                continue
            answer = assistant.ask(question)
        except (AssistantError, ValueError) as error:
            print(error)
            continue
        except Exception as error:                 # never let one question end the chat
            print(f"Something went wrong while answering ({type(error).__name__}). Please ask again.")
            continue
        print(answer.text)
        if answer.used_web:
            print("(from the web)")
            for url in answer.sources:
                print(url)


if __name__ == "__main__":
    raise SystemExit(main())
