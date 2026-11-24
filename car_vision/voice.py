"""The hosted voice: natural speech and listening in every browser, through Hugging Face.

The page can speak and listen with the browser's own speech, but only Microsoft
Edge does that well. This module offers the same natural voice and the same
listening in every browser, with two hosted models reached through Hugging
Face's Inference Providers (`voice.provider`, deepinfra):

- `say(text)` sends the text of one piece of an answer to a text-to-speech model
  (`voice.speech_model`) and returns the MP3 sound. It speaks English only, so
  text with Arabic letters is refused; the page then uses the browser's voice.
- `hear(wav, language)` sends one recorded question, as a WAV file, to a
  speech-to-text model (`voice.listen_model`) and returns the words.

What it costs. Speech is about a dollar per million characters (a 300-character
answer: 0.0003 dollars), listening 0.0002 dollars per minute. Before each
request its estimated cost is added to a daily total kept in `voice.spend_file`,
and a request is refused once that total has reached `voice.daily_budget_usd`.
So every request counts before it is sent, also one that then fails. The total
only goes up during a day and starts from zero on the next local date.

What it sends. The text of an answer, or the recording of a question, goes to
router.huggingface.co only, with the key `HF_TOKEN` from .env (the same key as
auto-discover). The key is never printed, logged, written to the spend file or
put in an error message. Nothing about the text, the words or the sound is
logged: counts and errors only.

One request of each kind at a time: a second `say` waits for the first, at most
WAIT_TURN_SECONDS. Plain HTTPS with the standard library, the multipart body
built by hand; `dotenv` reads .env. Built like `lookout.py`, which it borrows a
few small helpers from.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import struct
import threading
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import dotenv

from car_vision.config import PROJECT_ROOT, ConfigError, require, resolve_path
from car_vision.lookout import MODEL_PATTERN, ROUTER_HOST, RefuseRedirects, is_timeout, without_key

log = logging.getLogger(__name__)

# ---- the service ---------------------------------------------------------------

ENV_FILE = PROJECT_ROOT / ".env"
KEY_VARIABLE = "HF_TOKEN"
REQUEST_SECONDS = 30          # the real fetch waits at most this long for an answer
WAIT_TURN_SECONDS = 20        # a second call of the same kind waits at most this long for the first
LONGEST_SOUND = 4_000_000     # bytes of sound read from the server at most (600 characters: about 300 kB)
LONGEST_REPLY = 64_000        # bytes of a JSON answer read at most
LONGEST_WAV = 2_000_000       # bytes; the page sends 16-bit mono at 16000 Hz, 20 seconds is 640 kB
WAV_RATE = 16000              # the only WAV accepted: plain PCM, 1 channel, 16000 Hz, 16 bits ...
WAV_BYTES_PER_SECOND = 32000  # ... which is 32000 bytes of sound a second
WAV_HEADER_BYTES = 44
LONGEST_WORDS = 500           # characters of the heard words that are given back
LANGUAGES = ("en", "ar")
WAV_NAME = "question.wav"
MOST_PRICE = 1000.0           # dollars; a setting above this is surely a typing mistake
LONGEST_SPEND_FILE = 4_000    # bytes; the spend file holds three small values

# Arabic letters, in all their Unicode blocks: the speech model has no Arabic voice.
ARABIC = re.compile("[\u0600-\u06ff\u0750-\u077f\u0870-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
PROVIDER_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
VOICE_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,40}")


def speech_url(provider: str) -> str:
    return f"https://{ROUTER_HOST}/{provider}/v1/openai/audio/speech"


def listen_url(provider: str) -> str:
    return f"https://{ROUTER_HOST}/{provider}/v1/openai/audio/transcriptions"


# ---- errors ----------------------------------------------------------------------

OFF = "The hosted voice is off (voice.hosted in config.yaml)."
NO_KEY = ("No Hugging Face key found. Put your key after HF_TOKEN= in the .env file "
          "in the project folder, then start the program again.")
ODD_KEY = "HF_TOKEN in .env has spaces or unusual characters, so it cannot be a Hugging Face key."
KEY_REFUSED = ("Hugging Face refused the key (HF_TOKEN in .env). Check the key and that it "
               "may use Inference Providers, then start the program again.")
NO_CREDIT = "The Hugging Face account has no credit left for Inference Providers."
NOT_AVAILABLE = "The hosted voice's model or provider ({model}) is not available on Hugging Face."
SERVICE_BUSY = "Hugging Face is busy right now. Try again later."
STILL_BUSY = "The hosted voice is still busy with the previous request."
NO_ANSWER_IN_TIME = "Hugging Face did not answer in time."
NO_NETWORK = "Could not reach Hugging Face. Is the laptop online?"
BAD_ANSWER = "Hugging Face sent an answer the hosted voice could not use."
ODD_STATUS = "Hugging Face sent an unexpected answer (HTTP {status})."
BUDGET_USED_UP = ("Today's budget for the hosted voice is used up (about ${spent:.2f} of ${budget:.2f}). "
                  "It starts again tomorrow; until then the browser's own voice is used.")
NO_TEXT = "There is no text to speak."
TOO_LONG = "The text is too long for the hosted voice: at most {limit} characters at a time."
ARABIC_TEXT = "The hosted voice speaks English only, so the browser's own voice speaks Arabic text."
NOT_A_WAV = "The recording must be the page's own WAV file: 16-bit mono sound at 16000 Hz."
WAV_TOO_LARGE = f"The recording is too large: at most {LONGEST_WAV // 1_000_000} MB."
WAV_TOO_LONG = "The recording is too long: at most {limit:g} seconds."
WAV_EMPTY = "The recording is empty."
BAD_LANGUAGE = "The language must be 'en' or 'ar'."


class VoiceError(Exception):
    """A plain message for the page, never holding the key.

    `kind` says what went wrong: "off", "no_key", "key_refused", "no_credit",
    "not_available", "busy", "budget", "bad_input" or "failed". The page stops
    using the hosted voice for the visit after "budget", "no_key",
    "key_refused" and "no_credit"; after the others it uses the browser's voice
    for the rest of that one answer.
    """

    def __init__(self, message: str, kind: str = "failed") -> None:
        super().__init__(message)
        self.kind = kind


# ---- the key -----------------------------------------------------------------------


def read_key() -> str:
    """HF_TOKEN from .env or the environment. Never printed or logged."""
    dotenv.load_dotenv(ENV_FILE)
    key = os.environ.get(KEY_VARIABLE, "").strip()
    if not key:
        raise VoiceError(NO_KEY, "no_key")
    if not (key.isascii() and key.isprintable()) or " " in key:
        raise VoiceError(ODD_KEY, "no_key")
    return key


# ---- checking what is sent ---------------------------------------------------------


def checked_text(text: Any, limit: int) -> str:
    """The text to speak, trimmed; VoiceError("bad_input") when it cannot be spoken."""
    if not isinstance(text, str) or not text.strip():
        raise VoiceError(NO_TEXT, "bad_input")
    text = text.strip()
    if len(text) > limit:
        raise VoiceError(TOO_LONG.format(limit=limit), "bad_input")
    if ARABIC.search(text):
        raise VoiceError(ARABIC_TEXT, "bad_input")
    return text


def wav_seconds(wav: Any) -> float:
    """How long a recording is, from its real number of sound bytes; VoiceError("bad_input") otherwise.

    Only the WAV the page itself writes is accepted: the 44-byte header
    "RIFF", the size of the rest, "WAVE", a 16-byte "fmt " chunk saying plain
    PCM, 1 channel, 16000 Hz, 16 bits (so 2 bytes a sample and 32000 a
    second), then the "data" chunk, whose size is exactly the number of bytes
    after it, and nothing more. No header field is trusted for the length:
    the seconds are the sound bytes divided by WAV_BYTES_PER_SECOND, so a
    forged header cannot make a long recording look short (and cheap).
    """
    if not isinstance(wav, (bytes, bytearray)):
        raise VoiceError(NOT_A_WAV, "bad_input")
    if len(wav) > LONGEST_WAV:
        raise VoiceError(WAV_TOO_LARGE, "bad_input")
    if len(wav) < WAV_HEADER_BYTES or bytes(wav[:WAV_HEADER_BYTES - 4]) != expected_header(len(wav)):
        raise VoiceError(NOT_A_WAV, "bad_input")
    sound_bytes = len(wav) - WAV_HEADER_BYTES
    if struct.unpack("<I", wav[40:44])[0] != sound_bytes:   # the data size must be what really follows
        raise VoiceError(NOT_A_WAV, "bad_input")
    if sound_bytes == 0:
        raise VoiceError(WAV_EMPTY, "bad_input")
    return sound_bytes / WAV_BYTES_PER_SECOND


def expected_header(total_bytes: int) -> bytes:
    """The first 40 bytes of the page's WAV of this total length (the data size comes after them)."""
    return (b"RIFF" + struct.pack("<I", total_bytes - 8) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, WAV_RATE, WAV_BYTES_PER_SECOND, 2, 16)
            + b"data")


def checked_language(language: Any) -> str:
    if language not in LANGUAGES:
        raise VoiceError(BAD_LANGUAGE, "bad_input")
    return language


def is_mp3(content_type: str, sound: bytes) -> bool:
    """True for an audio answer that starts like an MP3: an ID3 tag or a frame's sync byte."""
    return content_type.lower().startswith("audio/") and (sound.startswith(b"ID3") or sound[:1] == b"\xff")


# ---- the requests and the answers ------------------------------------------------------


def speech_body(model: str, voice: str, text: str) -> bytes:
    """The text-to-speech request, as JSON bytes."""
    request = {"model": model, "input": text, "voice": voice, "response_format": "mp3"}
    return json.dumps(request).encode("utf-8")


def speech_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "audio/mpeg"}


def listen_body(model: str, wav: bytes, language: str, boundary: str) -> bytes:
    """The speech-to-text request: a multipart form with the WAV file, the model and the language.

    Each part is the boundary line, its headers, an empty line and its value;
    the last boundary line ends with "--".
    """
    def field(name: str, value: str) -> bytes:
        return (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n").encode("utf-8")

    file_head = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{WAV_NAME}"\r\n'
                 "Content-Type: audio/wav\r\n\r\n").encode("ascii")
    return (file_head + bytes(wav) + b"\r\n" + field("model", model) + field("language", language)
            + f"--{boundary}--\r\n".encode("ascii"))


def listen_headers(key: str, boundary: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json"}


def new_boundary() -> str:
    """A boundary that cannot appear in the recording by chance."""
    return "----car-vision-" + secrets.token_hex(16)


def heard_words(body: bytes, key: str) -> str:
    """The words in a speech-to-text answer, trimmed and cut at LONGEST_WORDS; "" when there are none."""
    try:
        reply = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise VoiceError(BAD_ANSWER, "failed") from None
    if not isinstance(reply, dict):
        raise VoiceError(BAD_ANSWER, "failed")
    text = reply.get("text")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise VoiceError(BAD_ANSWER, "failed")
    return without_key(text, key).strip()[:LONGEST_WORDS].strip()


def error_for_status(status: int, model: str) -> VoiceError:
    """The plain error for an HTTP status other than 200."""
    if status in (401, 403):
        return VoiceError(KEY_REFUSED, "key_refused")
    if status == 402:
        return VoiceError(NO_CREDIT, "no_credit")
    if status == 404:
        return VoiceError(NOT_AVAILABLE.format(model=model), "not_available")
    if status == 429 or 500 <= status <= 599:
        return VoiceError(SERVICE_BUSY, "busy")
    return VoiceError(ODD_STATUS.format(status=status), "failed")


# ---- the real network call -------------------------------------------------------------


def real_fetch(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, str, bytes]:
    """POST body to url over HTTPS: (status, content type, body bytes); only router.huggingface.co."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != ROUTER_HOST:
        raise ValueError(f"The hosted voice only talks to {ROUTER_HOST}.")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(RefuseRedirects)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Type", ""), read_limited(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, error.headers.get("Content-Type", "") if error.headers else "", read_limited(error)
        except Exception:      # an error answer whose body cannot be read is still its status
            return error.code, "", b""
        finally:
            error.close()


def read_limited(stream: Any) -> bytes:
    """At most LONGEST_SOUND bytes of a response."""
    raw = stream.read(LONGEST_SOUND)
    return raw if isinstance(raw, bytes) else b""


# ---- the spend file (as the lookout keeps its own) ----------------------------------------


def load_spend(path: Path, today: date) -> tuple[date, int, float]:
    """The day, uses and estimated dollars from the spend file.

    A missing file, or one from an earlier day, starts from zero. An
    unreadable one also starts from zero, with one log line. A file dated
    tomorrow (the clock went back a little) keeps its totals, so the estimate
    never goes down; one dated further ahead is taken as broken, since it
    would otherwise block the hosted voice for days.
    """
    if not path.exists():
        return today, 0, 0.0
    try:
        with open(path, "r", encoding="utf-8") as file:
            text = file.read(LONGEST_SPEND_FILE + 1)
        day, uses, spent = parse_spend(text)
        if day > today + timedelta(days=1):
            raise ValueError("dated more than a day ahead")
    except (OSError, ValueError, RecursionError):
        log.warning("Hosted voice: the spend file %s could not be read; today's count starts from zero.", path.name)
        return today, 0, 0.0
    if day < today:
        return today, 0, 0.0
    if day > today:
        log.warning("Hosted voice: the spend file is dated tomorrow; its totals are kept.")
    return day, uses, spent


def parse_spend(text: str) -> tuple[date, int, float]:
    """The three values of the spend file's text; ValueError when anything is off."""
    if len(text) > LONGEST_SPEND_FILE:
        raise ValueError("spend file too long")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("not an object")
    day = date.fromisoformat(data["day"]) if isinstance(data.get("day"), str) else None
    uses = data.get("uses")
    spent = data.get("spent_usd")
    if day is None or isinstance(uses, bool) or not isinstance(uses, int) or uses < 0:
        raise ValueError("bad day or uses")
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not math.isfinite(spent) or spent < 0:
        raise ValueError("bad spent_usd")
    return day, uses, float(spent)


def save_spend(path: Path, day: date, uses: int, spent: float) -> None:
    """Write the spend file through a temporary file, so it is never half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        json.dump({"day": day.isoformat(), "uses": uses, "spent_usd": spent}, file)
    os.replace(temporary, path)


# ---- settings ------------------------------------------------------------------------


def read_pattern(settings: dict, key: str, pattern: re.Pattern, example: str) -> str:
    """A name setting that must look like the example."""
    value = require(settings, key, str)
    if not pattern.fullmatch(value):
        raise ConfigError(f"voice.{key} must look like {example!r}, got {value!r}")
    return value


def read_dollars(settings: dict, key: str) -> float:
    """A price or budget setting: a number from 0 to MOST_PRICE."""
    value = require(settings, key, (int, float))
    if not math.isfinite(value) or not 0 <= value <= MOST_PRICE:
        raise ConfigError(f"voice.{key} must be a number from 0 to {MOST_PRICE:g}, got {value!r}")
    return float(value)


def read_whole(settings: dict, key: str, low: int, high: int) -> int:
    """A whole-number setting from low to high."""
    value = require(settings, key, int)
    if not low <= value <= high:
        raise ConfigError(f"voice.{key} must be a whole number from {low} to {high}, got {value}")
    return value


# ---- the hosted voice ----------------------------------------------------------------


Fetch = Callable[[str, dict, bytes, float], tuple[int, str, bytes]]


class HostedVoice:
    """Speaks text and hears recordings with two hosted models, within a daily budget.

    `fetch(url, headers, body_bytes, timeout) -> (status, content_type, body_bytes)`
    and `clock() -> datetime` (local time) can be replaced in tests. Web threads
    call it at the same time: one lock per kind of request lets one through at a
    time, and a small lock keeps the day's counters consistent.
    """

    def __init__(self, settings: dict, fetch: Fetch | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.provider = read_pattern(settings, "provider", PROVIDER_PATTERN, "deepinfra")
        self.speech_model = read_pattern(settings, "speech_model", MODEL_PATTERN, "hexgrad/Kokoro-82M")
        self.speech_voice = read_pattern(settings, "speech_voice", VOICE_PATTERN, "af_heart")
        self.listen_model = read_pattern(settings, "listen_model", MODEL_PATTERN, "openai/whisper-large-v3-turbo")
        self.max_chars = read_whole(settings, "max_spoken_chars", 1, 5000)
        self.max_seconds = read_whole(settings, "max_listen_seconds", 1, 60)
        self._budget = read_dollars(settings, "daily_budget_usd")
        self.speech_price = read_dollars(settings, "speech_price_per_million_chars")
        self.listen_price = read_dollars(settings, "listen_price_per_minute")
        self.spend_file = resolve_path(require(settings, "spend_file", str))
        self._fetch = fetch if fetch is not None else real_fetch
        self._clock = clock if clock is not None else datetime.now
        self._lock = threading.Lock()           # the counters and the key
        self._speaking = threading.Lock()       # one say() at a time
        self._hearing = threading.Lock()        # one hear() at a time
        self._key = ""                          # read at the first request
        self._save_failed = False               # log a failed save only once
        self._day, self._uses, self._spent = load_spend(self.spend_file, self._today())

    # -- what the page shows

    @property
    def uses_today(self) -> int:
        """Requests sent today, both kinds."""
        with self._lock:
            self._roll_day()
            return self._uses

    @property
    def spent_today(self) -> float:
        """Estimated dollars spent today."""
        with self._lock:
            self._roll_day()
            return self._spent

    @property
    def budget(self) -> float:
        """voice.daily_budget_usd: the hosted voice stops for the day when spent_today reaches it."""
        return self._budget

    # -- speaking and hearing

    def say(self, text: str) -> bytes:
        """The MP3 sound of one piece of text; VoiceError when there is none."""
        text = checked_text(text, self.max_chars)
        with Turn(self._speaking):
            key = self._key_for_request()
            self._count_before_sending(len(text) * self.speech_price / 1_000_000)
            body = speech_body(self.speech_model, self.speech_voice, text)
            _status, content_type, sound = self._send(speech_url(self.provider), speech_headers(key), body,
                                                     self.speech_model)
        if not is_mp3(content_type, sound):
            log.warning("Hosted voice: the speech answer was not MP3 sound.")
            raise VoiceError(BAD_ANSWER, "failed")
        return sound

    def hear(self, wav: bytes, language: str = "en") -> str:
        """The words in one WAV recording ("" when nothing was said); VoiceError when there are none."""
        seconds = wav_seconds(wav)
        if seconds > self.max_seconds:
            raise VoiceError(WAV_TOO_LONG.format(limit=self.max_seconds), "bad_input")
        language = checked_language(language)
        with Turn(self._hearing):
            key = self._key_for_request()
            self._count_before_sending(seconds / 60 * self.listen_price)
            boundary = new_boundary()
            body = listen_body(self.listen_model, wav, language, boundary)
            _status, _content_type, answer = self._send(listen_url(self.provider), listen_headers(key, boundary),
                                                       body, self.listen_model)
        return heard_words(answer[:LONGEST_REPLY], key)

    def _key_for_request(self) -> str:
        with self._lock:
            key = self._key
        if not key:
            key = read_key()
            with self._lock:
                self._key = key
        return key

    def _forget_key(self) -> None:
        with self._lock:
            self._key = ""

    def _send(self, url: str, headers: dict, body: bytes, model: str) -> tuple[int, str, bytes]:
        """One request: (200, content type, body), or the plain VoiceError for what went wrong."""
        try:
            answer = self._fetch(url, headers, body, REQUEST_SECONDS)
        except Exception as error:      # every failure becomes a plain VoiceError
            if is_timeout(error):
                log.warning("Hosted voice: no answer within %d seconds.", REQUEST_SECONDS)
                raise VoiceError(NO_ANSWER_IN_TIME, "failed") from None
            log.warning("Hosted voice: the request failed (%s).", type(error).__name__)
            raise VoiceError(NO_NETWORK, "failed") from None
        status, content_type, reply = checked_answer(answer)
        if status != 200:
            log.warning("Hosted voice: Hugging Face answered HTTP %d.", status)
            if status in (401, 403):
                self._forget_key()
            raise error_for_status(status, model)
        return status, content_type, reply

    # -- the day's count and cost

    def _today(self) -> date:
        now = self._clock()
        return now.date() if isinstance(now, datetime) else now

    def _roll_day(self) -> None:
        """Start from zero on a later local date. Call with the lock held.

        An earlier date (the clock went back) keeps the totals.
        """
        today = self._today()
        if today > self._day:
            self._day, self._uses, self._spent = today, 0, 0.0

    def _count_before_sending(self, cost: float) -> None:
        """Refuse when the budget is used up; else add one use and its cost, before it is sent."""
        with self._lock:
            self._roll_day()
            if self._spent >= self._budget:
                raise VoiceError(BUDGET_USED_UP.format(spent=self._spent, budget=self._budget), "budget")
            self._uses += 1
            self._spent += cost
            self._save()
            uses, spent = self._uses, self._spent
        log.debug("Hosted voice: %d uses today, about $%.6f.", uses, spent)

    def _save(self) -> None:
        """Write the spend file. Call with the lock held, so writes keep their order."""
        try:
            save_spend(self.spend_file, self._day, self._uses, self._spent)
        except OSError as error:
            if not self._save_failed:
                log.warning("Hosted voice: could not write the spend file (%s).", type(error).__name__)
            self._save_failed = True


class Turn:
    """`with Turn(lock):` waits at most WAIT_TURN_SECONDS for the lock, else VoiceError("busy")."""

    def __init__(self, lock: threading.Lock) -> None:
        self.lock = lock

    def __enter__(self) -> None:
        if not self.lock.acquire(timeout=WAIT_TURN_SECONDS):
            raise VoiceError(STILL_BUSY, "busy")

    def __exit__(self, *_exc: Any) -> None:
        self.lock.release()


def checked_answer(answer: Any) -> tuple[int, str, bytes]:
    """fetch's (status, content type, body), or VoiceError when it is anything else."""
    if (isinstance(answer, tuple) and len(answer) == 3 and isinstance(answer[0], int)
            and not isinstance(answer[0], bool) and isinstance(answer[1], str)
            and isinstance(answer[2], (bytes, bytearray))):
        return answer[0], answer[1], bytes(answer[2])
    raise VoiceError(BAD_ANSWER, "failed")
