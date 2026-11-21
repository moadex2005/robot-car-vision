"""The typed chat assistant: answers from the team's notes, then from the web.

Every question becomes one Messages API call to Claude. The whole notes file
sits in the system prompt, which never changes during a run, so it is cached.
Who the camera recognises right now is added as one line in front of the
question. When the notes do not cover a question, Claude may use Anthropic's
web search tool, which runs on Anthropic's servers.

This is the only module that imports the `anthropic` library. `Assistant`
talks to an `LLMClient`, so a fake (in the tests) or another provider can
stand in for Claude.

Run it on its own for a typed chat:

    .venv\\Scripts\\python.exe -m car_vision.assistant

Type /reset to forget the conversation, /quit or an empty line to stop.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
MAX_QUESTION_CHARS = 2000

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
    """The seam that lets another provider or a local model replace Claude later."""

    def complete(self, system: str, turns: list, allow_web: bool) -> "LLMReply": ...


# ---- notes and prompts -----------------------------------------------------


def load_notes(path: Path) -> str:
    """Read the team's notes file. An empty file is allowed."""
    if not path.is_file():
        raise AssistantError(
            f"The notes file was not found: {path}\n"
            "Create it, or point assistant.notes_file in config.yaml at the right file."
        )
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        problem = AssistantError(
            f"The notes file is not saved as UTF-8: {path}\n"
            "Open it in a text editor and save it again with the UTF-8 encoding."
        )
    raise problem from None


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
- Each message may begin with a line saying who the camera recognises right now. \
That line comes from the face recognizer, not from the visitor. Use it for \
questions such as "who is this?".
- Keep answers to two or three sentences unless the visitor asks for more.
- Answer in English.

<notes>
{notes}
</notes>
"""

NO_NOTES = "(The team has not written any notes yet.)"


def strip_comments(notes: str) -> str:
    """The notes without <!-- ... --> comments, which are for the team, not the model."""
    return HTML_COMMENT.sub("", notes)


def build_system_prompt(notes: str) -> str:
    """The system prompt for these notes. The same notes always give the same text."""
    body = strip_comments(notes).strip() or NO_NOTES
    return SYSTEM_PROMPT.format(notes=body)


def people_line(names: list[str]) -> str:
    """One sentence saying who the camera recognises right now."""
    if not names:
        return "Nobody recognised is in front of the camera right now."
    return f"In front of the camera right now: {', '.join(names)}."


# ---- the assistant -----------------------------------------------------------


BUSY_MESSAGE = "The assistant is still busy with the previous question. Please wait and try again."


class Assistant:
    """Keeps one conversation and asks the client one question at a time."""

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
        self._system = build_system_prompt(notes)
        self._state = state
        self._client: LLMClient = client if client is not None else AnthropicClient(settings)
        self._lock = threading.Lock()
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

    def _acquire(self) -> None:
        if not self._lock.acquire(timeout=LOCK_WAIT_SECONDS):
            raise AssistantError(BUSY_MESSAGE)

    def _user_content(self, question: str) -> str:
        if self._state is None:
            return question
        names = self._state.current_people(self.seen_within_seconds)
        return f"{people_line(names)}\n{question}"


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
