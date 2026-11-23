"""The lookout: a small hosted model that names what the camera sees (auto-discover).

The everyday detector knows 80 kinds of object, and the find-by-name detector
finds anything, but only when it is given a name. The lookout supplies those
names: every few seconds the pipeline hands it one small camera picture, the
lookout sends it to a small vision-language model on Hugging Face's Inference
Providers, and returns the plain names of the objects in the answer, such as
["router", "cushion", "curtains"]. A local detector then looks for them.

What it costs. One look is about 312 tokens in and 17 out, roughly 0.000013
dollars at the listed prices; one look every 10 seconds is about half a cent
an hour. Every look adds its estimated cost to a daily total kept in
`lookout.spend_file`, and before each request the lookout refuses to send when
that total has reached `lookout.daily_budget_usd`. The total only goes up
during a day and starts from zero on the next local date.

What it sends. The picture shows the whole view, including anyone in it, so it
goes to a third party; that is why auto-discover is off until switched on. The
request goes to router.huggingface.co only, with the key `HF_TOKEN` from .env.
The key is never printed, logged, written to the spend file or put in an
error message. Nothing about the picture or the names is logged.

The model's answer is untrusted text, so `clean_names` lets only short plain
names through: letters, digits, spaces, hyphens and apostrophes, and never
words for people.

Plain HTTPS and JSON with the standard library; `dotenv` reads .env.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import threading
import unicodedata
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import dotenv

from car_vision.config import PROJECT_ROOT, ConfigError, require, resolve_path

log = logging.getLogger(__name__)

# ---- the service ---------------------------------------------------------------

ROUTER_HOST = "router.huggingface.co"
CHAT_URL = f"https://{ROUTER_HOST}/v1/chat/completions"
ENV_FILE = PROJECT_ROOT / ".env"
KEY_VARIABLE = "HF_TOKEN"
KEY_PATTERN = re.compile(r"hf_[A-Za-z0-9]{6,}", re.IGNORECASE)   # anything shaped like a Hugging Face key
REQUEST_SECONDS = 20          # the caller waits at most this long for an answer
MAX_TOKENS = 80               # the longest answer the model may write
LONGEST_JPEG = 200_000        # bytes; a 512-wide picture is far smaller
LONGEST_REPLY = 64_000        # bytes read from the server at most
LONGEST_ANSWER = 4_000        # characters of the model's answer looked at
GUESSED_TOKENS_IN = 400       # the cost of a look whose answer does not say
GUESSED_TOKENS_OUT = 30
MOST_TOKENS = 1_000_000_000   # a reported token count is never taken as more than this
MOST_PRICE = 1000.0           # dollars; a setting above this is surely a typing mistake
LONGEST_SPEND_FILE = 4_000    # bytes; the spend file holds three small values

QUESTION = (
    "List the distinct physical objects you can see in this picture, the most noticeable first. "
    "Answer only with their short names, separated by commas, and nothing else. "
    "Do not list people or any part of a person."
)

# ---- what a name may be --------------------------------------------------------

LONGEST_NAME = 40
MOST_WORDS = 3                # "remote control car" is a name; "the picture shows a desk" is not
NAME_MARKS = " -'"            # besides letters and digits
SEPARATORS = re.compile(r"[,、،\r\n\v\f\u0085  ]")   # commas (also 、 and ،) and line breaks
NUMBERING = re.compile(r"^\d{1,3}(?:\s*[.)\]]|\s*-\s)\s*")              # "1. ", "2) ", "3 - "
LEADING_WORDS = re.compile(r"^(?:(?:and|or|a|an|the)\s+)+")             # "and a red cup" -> "red cup"
EDGE_MARKS = " -'\"`*_#>+~.!?;()[]{}•·‣▪–—“”«»"
FILLER = frozenset({"etc", "and so on", "and more", "more", "others", "and others",
                    "none", "nothing", "unknown", "objects", "object"})

# Words for people or parts of people. A name that contains one of them is
# left out, so "man's jacket" and "hand" never become something to look for.
PEOPLE_WORDS = frozenset({
    "person", "persons", "people", "human", "humans", "individual", "individuals",
    "someone", "somebody", "anyone", "everyone", "crowd", "family", "couple",
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "guy", "guys",
    "lady", "ladies", "gentleman", "gentlemen", "adult", "adults", "teen", "teens",
    "teenager", "teenagers", "child", "children", "kid", "kids", "baby", "babies",
    "toddler", "toddlers", "mother", "father", "mom", "mum", "dad",
    "face", "faces", "head", "heads", "hair", "beard", "eye", "eyes", "ear", "ears",
    "nose", "mouth", "lips", "hand", "hands", "finger", "fingers", "arm", "arms",
    "leg", "legs", "foot", "feet", "body", "selfie",
})

# ---- errors ----------------------------------------------------------------------

NO_KEY = ("No Hugging Face key found. Put your key after HF_TOKEN= in the .env file "
          "in the project folder, then start the program again.")
ODD_KEY = "HF_TOKEN in .env has spaces or unusual characters, so it cannot be a Hugging Face key."
KEY_REFUSED = ("Hugging Face refused the key (HF_TOKEN in .env). Check the key and that it "
               "may use Inference Providers, then start the program again.")
NO_CREDIT = "The Hugging Face account has no credit left for Inference Providers."
NOT_AVAILABLE = "The model or provider in lookout.model ({model}) is not available on Hugging Face."
SERVICE_BUSY = "Hugging Face is busy right now. Try again later."
NO_ANSWER_IN_TIME = f"Hugging Face did not answer within {REQUEST_SECONDS} seconds."
NO_NETWORK = "Could not reach Hugging Face. Is the laptop online?"
BAD_ANSWER = "Hugging Face sent an answer that auto-discover could not read."
ODD_STATUS = "Hugging Face sent an unexpected answer (HTTP {status})."
STILL_LOOKING = "The previous look has not finished yet."
BUDGET_USED_UP = ("Today's budget for auto-discover is used up (about ${spent:.2f} of ${budget:.2f}). "
                  "Looking starts again tomorrow.")


class LookoutError(Exception):
    """A plain message for the page, never holding the key.

    `kind` says what went wrong, so the pipeline can tell a lasting problem
    from a passing one: "no_key", "key_refused", "budget" and "no_credit" last;
    "busy", "timeout", "network", "bad_answer", "not_available" and
    "still_looking" may pass.
    """

    def __init__(self, message: str, kind: str = "other") -> None:
        super().__init__(message)
        self.kind = kind


# ---- cleaning the model's answer -----------------------------------------------------


def clean_names(answer: str, max_names: int) -> list[str]:
    """The object names in the model's answer: plain, lower case, no people, no repeats.

    Splits on commas and line breaks, removes bullets, numbers, quotes and a
    leading "a" or "the", and keeps only names of 1 to 40 letters, digits,
    spaces, hyphens and apostrophes with at most three words. Anything else
    is dropped, so an answer with no usable name gives [].
    """
    if not isinstance(answer, str):
        return []
    text = unicodedata.normalize("NFKC", answer[:LONGEST_ANSWER])
    kept: list[str] = []
    for piece in SEPARATORS.split(text):
        name = clean_name(piece)
        if name and name not in kept and not about_people(name):
            kept.append(name)
            if len(kept) >= max_names:
                break
    return kept


def clean_name(piece: str) -> str:
    """One piece of the answer as a plain name, or "" when it is not one."""
    text = piece.replace("’", "'").replace("‘", "'")
    text = text.rsplit(":", 1)[-1]                 # "Objects: TV" -> "TV"
    text = " ".join(text.split()).lower()
    for _ in range(3):                             # "- 1. **TV**" needs a few rounds
        text = NUMBERING.sub("", text.strip(EDGE_MARKS))
    text = LEADING_WORDS.sub("", text).strip(EDGE_MARKS)
    return text if is_plain_name(text) else ""


def is_plain_name(text: str) -> bool:
    """True for 1 to 40 allowed characters, at least one letter, at most three words."""
    if not 1 <= len(text) <= LONGEST_NAME or text in FILLER:
        return False
    if len(text.split(" ")) > MOST_WORDS:
        return False
    if not all(allowed_in_name(letter) for letter in text):
        return False
    return any(letter.isalpha() for letter in text)


def allowed_in_name(letter: str) -> bool:
    """Letters and their accents in any script, the digits 0 to 9, and the marks in NAME_MARKS."""
    if letter in NAME_MARKS or "0" <= letter <= "9":
        return True
    return unicodedata.category(letter)[0] in "LM"


def about_people(name: str) -> bool:
    """True when a word of the name is a word for people or a part of a person."""
    for word in name.replace("-", " ").split():
        word = word.removesuffix("'s").removesuffix("'")
        if word in PEOPLE_WORDS:
            return True
    return False


# ---- the key -----------------------------------------------------------------------


def read_key() -> str:
    """HF_TOKEN from .env or the environment. Never printed or logged."""
    dotenv.load_dotenv(ENV_FILE)
    key = os.environ.get(KEY_VARIABLE, "").strip()
    if not key:
        raise LookoutError(NO_KEY, "no_key")
    if not (key.isascii() and key.isprintable()) or " " in key:
        raise LookoutError(ODD_KEY, "no_key")
    return key


KEY_REMOVED = "[key=removed]"     # the "=" makes sure it can never pass as a name


def without_key(text: str, key: str = "") -> str:
    """Text with the key, and anything shaped like a Hugging Face key, cut out."""
    if key:
        text = text.replace(key, KEY_REMOVED)
    return KEY_PATTERN.sub(KEY_REMOVED, text)


# ---- the request and the answer ------------------------------------------------------


def request_body(model: str, jpeg: bytes) -> bytes:
    """The chat request: one picture and the question, as JSON bytes."""
    picture = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    request = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": picture}},
            {"type": "text", "text": QUESTION},
        ]}],
    }
    return json.dumps(request).encode("utf-8")


def request_headers(key: str) -> dict[str, str]:
    """The request's headers, with the key."""
    return {"Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json"}


def read_answer(body_text: str) -> tuple[str, Any]:
    """The answer's text and its "usage" part; LookoutError when the JSON is not as expected."""
    try:
        reply = json.loads(body_text)
        message = reply["choices"][0]["message"]
        content = message.get("content")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError):
        raise LookoutError(BAD_ANSWER, "bad_answer") from None
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise LookoutError(BAD_ANSWER, "bad_answer")
    return content, reply.get("usage") if isinstance(reply, dict) else None


def tokens_used(usage: Any) -> tuple[int, int]:
    """Tokens in and out as the answer reports them; the guessed amounts where it does not."""
    if not isinstance(usage, dict):
        usage = {}
    return (token_count(usage.get("prompt_tokens"), GUESSED_TOKENS_IN),
            token_count(usage.get("completion_tokens"), GUESSED_TOKENS_OUT))


def token_count(value: Any, guess: int) -> int:
    """A reported token count when it is a whole number of zero or more, else the guess."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return guess
    return min(value, MOST_TOKENS)


def error_for_status(status: int, model: str) -> LookoutError:
    """The plain error for an HTTP status other than 200."""
    if status in (401, 403):
        return LookoutError(KEY_REFUSED, "key_refused")
    if status == 402:
        return LookoutError(NO_CREDIT, "no_credit")
    if status == 404:
        return LookoutError(NOT_AVAILABLE.format(model=model), "not_available")
    if status == 429 or 500 <= status <= 599:
        return LookoutError(SERVICE_BUSY, "busy")
    return LookoutError(ODD_STATUS.format(status=status), "bad_answer")


def is_timeout(error: BaseException) -> bool:
    """True when a network error means that no answer came in time."""
    if isinstance(error, TimeoutError):
        return True
    return isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError)


# ---- the real network call -------------------------------------------------------------


class RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Follows no redirect: the key must never travel to another address.

    (A redirected POST would also lose its body, so following one never helps.)
    The 3xx answer then arrives as an ordinary HTTP status.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """urllib asks where to go next; None means nowhere."""
        return None


def real_fetch(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, str]:
    """POST body to url over HTTPS and return (status, text); only router.huggingface.co."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != ROUTER_HOST:
        raise ValueError(f"The lookout only talks to {ROUTER_HOST}.")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(RefuseRedirects)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, read_limited(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, read_limited(error)
        except Exception:      # an error answer whose body cannot be read is still its status
            return error.code, ""
        finally:
            error.close()


def read_limited(stream: Any) -> str:
    """At most LONGEST_REPLY bytes of a response, as text."""
    raw = stream.read(LONGEST_REPLY)
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else ""


# ---- the spend file ------------------------------------------------------------------


def load_spend(path: Path, today: date) -> tuple[date, int, float]:
    """The day, looks and estimated dollars from the spend file.

    A missing file, or one from an earlier day, starts from zero. An
    unreadable one also starts from zero, with one log line. A file from a
    later day than today (the clock went back) keeps its totals, so the
    estimate never goes down.
    """
    if not path.exists():
        return today, 0, 0.0
    try:
        with open(path, "r", encoding="utf-8") as file:
            text = file.read(LONGEST_SPEND_FILE + 1)
        day, looks, spent = parse_spend(text)
    except (OSError, ValueError, RecursionError):
        log.warning("Lookout: the spend file %s could not be read; today's count starts from zero.", path.name)
        return today, 0, 0.0
    if day < today:
        return today, 0, 0.0
    if day > today:
        log.warning("Lookout: the spend file is dated after today; its totals are kept.")
    return day, looks, spent


def parse_spend(text: str) -> tuple[date, int, float]:
    """The three values of the spend file's text; ValueError when anything is off."""
    if len(text) > LONGEST_SPEND_FILE:
        raise ValueError("spend file too long")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("not an object")
    day = date.fromisoformat(data["day"]) if isinstance(data.get("day"), str) else None
    looks = data.get("looks")
    spent = data.get("spent_usd")
    if day is None or isinstance(looks, bool) or not isinstance(looks, int) or looks < 0:
        raise ValueError("bad day or looks")
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not math.isfinite(spent) or spent < 0:
        raise ValueError("bad spent_usd")
    return day, looks, float(spent)


def save_spend(path: Path, day: date, looks: int, spent: float) -> None:
    """Write the spend file through a temporary file, so it is never half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        json.dump({"day": day.isoformat(), "looks": looks, "spent_usd": spent}, file)
    os.replace(temporary, path)


# ---- settings ------------------------------------------------------------------------


MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-/:]{0,199}")


def read_model(settings: dict) -> str:
    """lookout.model, like "google/gemma-3-4b-it:parasail"."""
    model = require(settings, "model", str)
    if not MODEL_PATTERN.fullmatch(model):
        raise ConfigError(f"lookout.model must look like 'google/gemma-3-4b-it:parasail', got {model!r}")
    return model


def read_dollars(settings: dict, key: str) -> float:
    """A price or budget setting: a number from 0 to MOST_PRICE."""
    value = require(settings, key, (int, float))
    if not math.isfinite(value) or not 0 <= value <= MOST_PRICE:
        raise ConfigError(f"lookout.{key} must be a number from 0 to {MOST_PRICE:g}, got {value!r}")
    return float(value)


def read_max_names(settings: dict) -> int:
    """lookout.max_names: how many names one look gives at most."""
    value = require(settings, "max_names", int)
    if value < 1:
        raise ConfigError(f"lookout.max_names must be 1 or more, got {value}")
    return value


# ---- the lookout ---------------------------------------------------------------------


Fetch = Callable[[str, dict, bytes, float], tuple[int, str]]


class Lookout:
    """Asks the hosted model to name what is in one picture, within a daily budget.

    `fetch(url, headers, body_bytes, timeout) -> (status, body_text)` and
    `clock() -> datetime` (local time) can be replaced in tests. `names` may
    run on a background thread while web threads read the counters; one small
    lock keeps them consistent, and only one request is ever under way.
    """

    def __init__(self, settings: dict, fetch: Fetch | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.model = read_model(settings)
        self.max_names = read_max_names(settings)
        self._budget = read_dollars(settings, "daily_budget_usd")
        self.input_price = read_dollars(settings, "input_price_per_million")
        self.output_price = read_dollars(settings, "output_price_per_million")
        self.spend_file = resolve_path(require(settings, "spend_file", str))
        self._fetch = fetch if fetch is not None else real_fetch
        self._clock = clock if clock is not None else datetime.now
        self._lock = threading.Lock()
        self._key = ""                 # read at the first look
        self._looking = False          # a request is under way
        self._save_failed = False      # log a failed save only once
        self._day, self._looks, self._spent = load_spend(self.spend_file, self._today())

    # -- what the page shows

    @property
    def looks_today(self) -> int:
        with self._lock:
            self._roll_day()
            return self._looks

    @property
    def spent_today(self) -> float:
        """Estimated dollars spent today."""
        with self._lock:
            self._roll_day()
            return self._spent

    @property
    def budget(self) -> float:
        """lookout.daily_budget_usd: looking stops for the day when spent_today reaches it."""
        return self._budget

    # -- one look

    def names(self, jpeg: bytes) -> list[str]:
        """The names of the objects in one JPEG picture; LookoutError when there is no answer."""
        check_jpeg(jpeg)
        key = self._key_for_look()
        body = request_body(self.model, jpeg)
        self._start_look()
        status, text = self._ask(key, body)
        if status != 200:
            log.warning("Lookout: Hugging Face answered HTTP %d.", status)
            if status in (401, 403):
                self._forget_key()
            raise error_for_status(status, self.model)
        try:
            content, usage = read_answer(text)
        except LookoutError:
            self._charge(GUESSED_TOKENS_IN, GUESSED_TOKENS_OUT)
            log.warning("Lookout: the answer was not the expected JSON.")
            raise
        self._charge(*tokens_used(usage))
        return clean_names(without_key(content, key), self.max_names)

    def _key_for_look(self) -> str:
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

    def _start_look(self) -> None:
        """Refuse when the budget is used up or a look is under way; else mark one as started."""
        with self._lock:
            self._roll_day()
            if self._spent >= self._budget:
                raise LookoutError(BUDGET_USED_UP.format(spent=self._spent, budget=self._budget), "budget")
            if self._looking:
                raise LookoutError(STILL_LOOKING, "still_looking")
            self._looking = True

    def _finish_look(self) -> None:
        with self._lock:
            self._looking = False

    def _ask(self, key: str, body: bytes) -> tuple[int, str]:
        """Run fetch on a helper thread and wait at most REQUEST_SECONDS for (status, text).

        A look that timed out is charged the guessed cost, since the model may
        have run. It stays "under way" until its thread really ends.
        """
        result: dict[str, Any] = {}
        finished = threading.Event()

        def work() -> None:
            try:
                result["answer"] = self._fetch(CHAT_URL, request_headers(key), body, REQUEST_SECONDS)
            except Exception as error:  # every failure becomes a plain LookoutError below
                result["error"] = error
            finally:
                self._finish_look()
                finished.set()

        try:
            threading.Thread(target=work, name="lookout-request", daemon=True).start()
        except RuntimeError:
            self._finish_look()
            raise LookoutError(NO_NETWORK, "network") from None
        if not finished.wait(REQUEST_SECONDS):
            return self._no_answer_in_time()
        if "error" in result:
            return self._failed(result["error"])
        return checked_answer(result.get("answer"))

    def _no_answer_in_time(self) -> tuple[int, str]:
        self._charge(GUESSED_TOKENS_IN, GUESSED_TOKENS_OUT)
        log.warning("Lookout: no answer within %d seconds.", REQUEST_SECONDS)
        raise LookoutError(NO_ANSWER_IN_TIME, "timeout")

    def _failed(self, error: BaseException) -> tuple[int, str]:
        if is_timeout(error):
            return self._no_answer_in_time()
        log.warning("Lookout: the request failed (%s).", type(error).__name__)
        raise LookoutError(NO_NETWORK, "network")

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
            self._day, self._looks, self._spent = today, 0, 0.0

    def _charge(self, tokens_in: int, tokens_out: int) -> None:
        """Add one look and its estimated cost, and keep them in the spend file."""
        cost = (tokens_in * self.input_price + tokens_out * self.output_price) / 1_000_000
        with self._lock:
            self._roll_day()
            self._looks += 1
            self._spent += cost
            self._save()
            looks, spent = self._looks, self._spent
        log.debug("Lookout: %d looks today, about $%.6f.", looks, spent)

    def _save(self) -> None:
        """Write the spend file. Call with the lock held, so writes keep their order."""
        try:
            save_spend(self.spend_file, self._day, self._looks, self._spent)
        except OSError as error:
            if not self._save_failed:
                log.warning("Lookout: could not write the spend file (%s).", type(error).__name__)
            self._save_failed = True


def check_jpeg(jpeg: Any) -> None:
    """ValueError unless jpeg is the bytes of a JPEG picture of at most LONGEST_JPEG bytes."""
    if not isinstance(jpeg, (bytes, bytearray)) or not jpeg.startswith(b"\xff\xd8\xff"):
        raise ValueError("The lookout needs a JPEG picture.")
    if len(jpeg) > LONGEST_JPEG:
        raise ValueError(f"The picture for the lookout must be at most {LONGEST_JPEG // 1000} kB.")


def checked_answer(answer: Any) -> tuple[int, str]:
    """fetch's (status, text), or LookoutError when it is anything else."""
    if (isinstance(answer, tuple) and len(answer) == 2 and isinstance(answer[0], int)
            and not isinstance(answer[0], bool) and isinstance(answer[1], str)):
        return answer
    raise LookoutError(BAD_ANSWER, "bad_answer")
