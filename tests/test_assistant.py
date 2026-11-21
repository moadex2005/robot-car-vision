"""Tests for car_vision.assistant. No network, no key, no graphics card.

The Assistant is tested with a fake LLMClient. AnthropicClient is tested with
the SDK object replaced by a stub that records each request and plays back
prepared responses. Every test runs with ANTHROPIC_API_KEY cleared and the
.env lookup pointed at an empty temporary folder, so the project's real .env
is never read.
"""

from __future__ import annotations

import base64
import re
import threading
import time
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from car_vision import assistant
from car_vision.assistant import (
    Answer,
    AnthropicClient,
    Assistant,
    AssistantError,
    LLMReply,
    ObjectInfo,
    build_system_prompt,
    load_briefs,
    load_notes,
    parse_object_info,
    people_line,
)
from car_vision.config import PROJECT_ROOT, ConfigError, load_config, section
from car_vision.state import Detection, RecognizedFace, WorldState

KEY_VARIABLE = "ANTHROPIC_API_KEY"
FAKE_KEY = "sk-ant-test-RECOGNISABLE-FAKE-KEY-7f3a9c"
NOTES = "# Project\nThe car is called Rover and it follows a black line.\n"
BOM = assistant.BOM
CP1252_BOM = assistant.CP1252_BOM


# ---- fixtures and fakes --------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_environment(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
                         tmp_path: Path) -> None:
    """No test sees the real key, and no test can build a real SDK client."""
    if request.node.get_closest_marker("hardware"):
        return
    # setenv first so that teardown removes whatever load_dotenv may set later.
    monkeypatch.setenv(KEY_VARIABLE, "placeholder")
    monkeypatch.delenv(KEY_VARIABLE)
    empty_folder = tmp_path / "no_env_here"
    empty_folder.mkdir()
    monkeypatch.setattr(assistant, "ENV_FILE", empty_folder / ".env")

    def no_real_client(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a test tried to create a real Anthropic client")

    monkeypatch.setattr(assistant.anthropic, "Anthropic", no_real_client)


@pytest.fixture
def notes_path(tmp_path: Path) -> Path:
    path = tmp_path / "knowledge.md"
    path.write_text(NOTES, encoding="utf-8")
    return path


@pytest.fixture
def people_dir(tmp_path: Path) -> Path:
    """A people folder that does not exist yet; tests add persons with add_person."""
    return tmp_path / "people"


@pytest.fixture
def settings(notes_path: Path, people_dir: Path) -> dict[str, Any]:
    return {
        "provider": "anthropic",
        "model": "test-model-from-config",
        "effort": "low",
        "max_tokens": 16000,
        "notes_file": str(notes_path),
        "people_dir": str(people_dir),
        "web_search": True,
        "web_search_max_uses": 3,
        "history_turns": 10,
        "seen_within_seconds": 3.0,
    }


class FakeClient:
    """Records every call; answers with a fresh assistant turn each time."""

    def __init__(self, replies: list[LLMReply] | None = None) -> None:
        self.calls: list[tuple[str, list, bool]] = []
        self.replies = list(replies or [])
        self.error: Exception | None = None

    def complete(self, system: str, turns: list, allow_web: bool) -> LLMReply:
        self.calls.append((system, list(turns), allow_web))
        if self.error is not None:
            raise self.error
        if self.replies:
            return self.replies.pop(0)
        number = len(self.calls)
        turn = {"role": "assistant", "content": [f"block-{number}"]}
        return LLMReply(text=f"answer {number}", new_turns=(turn,))


def state_with(*names: str) -> WorldState:
    state = WorldState()
    faces = [RecognizedFace(name=n, similarity=0.9, box=(0, 0, 10, 10), person_id=n.lower())
             for n in names]
    state.update_faces(faces)
    return state


class FakeMessages:
    def __init__(self, sdk: "FakeSDK") -> None:
        self._sdk = sdk

    def create(self, **request: Any) -> Any:
        # Copy the message list: the client appends to it after the call.
        self._sdk.requests.append(dict(request, messages=list(request["messages"])))
        if self._sdk.error is not None:
            raise self._sdk.error
        return self._sdk.responses.pop(0)


class FakeSDK:
    """Stands in for anthropic.Anthropic: records requests, plays back responses."""

    def __init__(self, responses: list[Any] | None = None, error: Exception | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses = list(responses or [])
        self.error = error
        self.beta = type("Beta", (), {})()
        self.beta.messages = FakeMessages(self)


def message(content: list[dict[str, Any]], stop_reason: str = "end_turn", **extra: Any) -> BetaMessage:
    """A real SDK response object built from the API's JSON shape."""
    data = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model-from-config",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    data.update(extra)
    return BetaMessage.model_validate(data)


def text_block(text: str, urls: tuple[str, ...] = ()) -> dict[str, Any]:
    citations = [{"type": "web_search_result_location", "url": url, "title": "Page",
                  "cited_text": "quoted", "encrypted_index": "idx"} for url in urls]
    return {"type": "text", "text": text, "citations": citations or None}


THINKING = {"type": "thinking", "thinking": "", "signature": "sig"}
SEARCH_CALL = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
               "input": {"query": "robot car"}}


def search_result(*urls: str) -> dict[str, Any]:
    return {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
            "content": [{"type": "web_search_result", "url": u, "title": "T",
                         "encrypted_content": "enc", "page_age": None} for u in urls]}


def client_with(settings: dict[str, Any], sdk: FakeSDK) -> AnthropicClient:
    client = AnthropicClient(settings)
    client._sdk = sdk            # skip the real SDK client and the key lookup
    return client


def keys_anywhere(value: Any) -> set[str]:
    """Every dictionary key at any depth of a request."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, inner in value.items():
            found.add(key)
            found |= keys_anywhere(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            found |= keys_anywhere(inner)
    return found


# ---- prompts -----------------------------------------------------------------


def test_system_prompt_holds_notes_and_is_the_same_for_every_question(settings: dict) -> None:
    fake = FakeClient()
    bot = Assistant(settings, state=state_with("Sara", "Omar"), client=fake)
    bot.ask("What is the car called?")
    bot.ask("Who is this?")
    first, second = fake.calls[0][0], fake.calls[1][0]
    assert first == second
    assert NOTES.strip() in first
    assert "Sara" not in first and "Omar" not in first
    assert build_system_prompt(NOTES) == build_system_prompt(NOTES)


def test_system_prompt_with_empty_notes_still_works() -> None:
    prompt = build_system_prompt("   \n")
    assert "<notes>" in prompt and "not written any notes" in prompt


def test_people_line_sentences() -> None:
    assert people_line(["Sara", "Omar"]) == "In front of the camera right now: Sara, Omar."
    assert "Nobody recognised" in people_line([])


def test_people_line_goes_into_the_user_turn(settings: dict) -> None:
    fake = FakeClient()
    Assistant(settings, state=state_with("Sara", "Omar"), client=fake).ask("Who is this?")
    user_turn = fake.calls[0][1][-1]
    assert user_turn["role"] == "user"
    assert user_turn["content"] == "In front of the camera right now: Sara, Omar.\nWho is this?"


def test_people_line_when_nobody_is_recognised(settings: dict) -> None:
    fake = FakeClient()
    Assistant(settings, state=WorldState(), client=fake).ask("Who is this?")
    content = fake.calls[0][1][-1]["content"]
    assert content.startswith("Nobody recognised")
    assert content.endswith("\nWho is this?")


def test_without_a_state_the_question_goes_alone(settings: dict) -> None:
    fake = FakeClient()
    Assistant(settings, client=fake).ask("  What is the car called?  ")
    assert fake.calls[0][1] == [{"role": "user", "content": "What is the car called?"}]


# ---- conversation history ----------------------------------------------------


def test_history_grows_and_assistant_turns_come_back_unchanged(settings: dict) -> None:
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.ask("one")
    bot.ask("two")
    bot.ask("three")
    turns = fake.calls[2][1]
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant", "user"]
    assert turns[1]["content"] == ["block-1"] and turns[3]["content"] == ["block-2"]
    # The very same objects the client returned, not copies or rebuilt text.
    assert turns[1] is fake.calls[1][1][1]
    assert turns[1]["content"] is fake.calls[1][1][1]["content"]


def test_reset_forgets_the_conversation(settings: dict) -> None:
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.ask("one")
    bot.reset()
    bot.ask("two")
    assert fake.calls[1][1] == [{"role": "user", "content": "two"}]


def test_after_history_turns_exchanges_a_fresh_conversation_starts(settings: dict) -> None:
    settings["history_turns"] = 2
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    for question in ("one", "two", "three", "four"):
        bot.ask(question)
    sent = [len(call[1]) for call in fake.calls]
    # one: alone; two: after 1 pair; three: 2 pairs reached, so fresh; four: after 1 pair.
    assert sent == [1, 3, 1, 3]
    # Nothing is cut out of the middle: the fresh conversation holds only new turns.
    assert fake.calls[3][1][0] == {"role": "user", "content": "three"}


def test_history_turns_zero_sends_every_question_alone(settings: dict) -> None:
    settings["history_turns"] = 0
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.ask("one")
    bot.ask("two")
    assert [len(call[1]) for call in fake.calls] == [1, 1]


def test_every_assistant_turn_of_a_resumed_reply_is_kept(settings: dict) -> None:
    paused = {"role": "assistant", "content": ["paused-search"]}
    final = {"role": "assistant", "content": ["final-answer"]}
    fake = FakeClient([LLMReply(text="done", new_turns=(paused, final))])
    bot = Assistant(settings, client=fake)
    bot.ask("one")
    bot.ask("two")
    assert fake.calls[1][1][1:3] == [paused, final]
    assert fake.calls[1][1][1] is paused and fake.calls[1][1][2] is final


def test_a_failed_question_leaves_the_history_as_it_was(settings: dict) -> None:
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.ask("one")
    fake.error = AssistantError("no internet")
    with pytest.raises(AssistantError):
        bot.ask("two")
    fake.error = None
    answer = bot.ask("three")
    assert answer.text == "answer 3"
    contents = [t["content"] for t in fake.calls[2][1]]
    assert contents == ["one", ["block-1"], "three"]


def test_a_failed_question_at_the_limit_keeps_the_old_history(settings: dict) -> None:
    settings["history_turns"] = 1
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.ask("one")
    fake.error = AssistantError("timeout")
    with pytest.raises(AssistantError):
        bot.ask("two")                     # would have started fresh
    fake.error = None
    bot.ask("three")
    assert [len(call[1]) for call in fake.calls] == [1, 1, 1]


def test_a_second_ask_waits_for_the_first(settings: dict) -> None:
    class SlowClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.release = threading.Event()
            self.active = 0
            self.most_active = 0
            self.count_lock = threading.Lock()

        def complete(self, system: str, turns: list, allow_web: bool) -> LLMReply:
            with self.count_lock:
                self.active += 1
                self.most_active = max(self.most_active, self.active)
            self.release.wait(5)
            with self.count_lock:
                self.active -= 1
            return super().complete(system, turns, allow_web)

    slow = SlowClient()
    bot = Assistant(settings, client=slow)
    threads = [threading.Thread(target=bot.ask, args=(q,)) for q in ("one", "two")]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    assert slow.active == 1
    slow.release.set()
    for thread in threads:
        thread.join(5)
    assert slow.most_active == 1
    assert len(slow.calls) == 2


# ---- settings and replies ------------------------------------------------------


@pytest.mark.parametrize("web_search", [True, False])
def test_allow_web_follows_the_setting(settings: dict, web_search: bool) -> None:
    settings["web_search"] = web_search
    fake = FakeClient()
    Assistant(settings, client=fake).ask("question")
    assert fake.calls[0][2] is web_search


def test_used_web_and_sources_come_through_without_repeats(settings: dict) -> None:
    reply = LLMReply(text="From the web.", new_turns=({"role": "assistant", "content": []},),
                     used_web=True, sources=("https://a.example", "https://b.example",
                                             "https://a.example"))
    answer = Assistant(settings, client=FakeClient([reply])).ask("question")
    assert answer.text == "From the web."
    assert answer.used_web is True
    assert answer.sources == ("https://a.example", "https://b.example")


# ---- errors ------------------------------------------------------------------


def test_empty_question_raises_value_error(settings: dict) -> None:
    with pytest.raises(ValueError):
        Assistant(settings, client=FakeClient()).ask("   ")


def test_unknown_provider_raises_config_error(settings: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_VARIABLE, FAKE_KEY)
    settings["provider"] = "someone-else"
    with pytest.raises(ConfigError) as caught:
        Assistant(settings, client=FakeClient())
    assert FAKE_KEY not in str(caught.value)


def test_missing_setting_raises_config_error(settings: dict) -> None:
    del settings["history_turns"]
    with pytest.raises(ConfigError):
        Assistant(settings, client=FakeClient())


def test_missing_notes_file_names_the_path(settings: dict, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_VARIABLE, FAKE_KEY)
    missing = tmp_path / "nowhere" / "knowledge.md"
    settings["notes_file"] = str(missing)
    with pytest.raises(AssistantError) as caught:
        Assistant(settings, client=FakeClient())
    assert "nowhere/knowledge.md" in str(caught.value)
    assert str(tmp_path) not in str(caught.value)          # no full local path on screen
    assert FAKE_KEY not in str(caught.value)


def test_empty_notes_file_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")
    assert load_notes(path) == ""


def test_missing_key_says_to_copy_env_example(settings: dict) -> None:
    client = AnthropicClient(settings)
    with pytest.raises(AssistantError) as caught:
        client.complete("system", [{"role": "user", "content": "hi"}], allow_web=True)
    assert ".env.example" in str(caught.value)


def test_missing_key_message_in_the_chat_loop(settings: dict, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    """The real CLI with no key: the message appears and the loop carries on."""
    monkeypatch.setattr(assistant, "load_config",
                        lambda: {"assistant": dict(settings)})
    answers = iter(["What is the car called?", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert assistant.main([]) == 0
    printed = capsys.readouterr().out
    assert ".env.example" in printed
    assert "Traceback" not in printed


@pytest.mark.parametrize("lines", [
    [BOM],                                   # PowerShell 5.1 piping an empty line
    [BOM + "/quit"],
    ["/reset", "  /quit  "],
])
def test_chat_loop_commands_ask_nothing(settings: dict, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str], lines: list[str]) -> None:
    fake = FakeClient()
    real_assistant = assistant.Assistant
    monkeypatch.setattr(assistant, "load_config", lambda: {"assistant": dict(settings)})
    monkeypatch.setattr(assistant, "Assistant",
                        lambda section_settings: real_assistant(section_settings, client=fake))
    answers = iter(lines)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert assistant.main([]) == 0
    assert fake.calls == []


def test_chat_loop_prints_web_sources(settings: dict, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    reply = LLMReply(text="It is sunny.", new_turns=({"role": "assistant", "content": []},),
                     used_web=True, sources=("https://a.example", "https://a.example",
                                             "https://b.example"))
    fake = FakeClient([reply])
    real_assistant = assistant.Assistant
    monkeypatch.setattr(assistant, "load_config", lambda: {"assistant": dict(settings)})
    monkeypatch.setattr(assistant, "Assistant",
                        lambda section_settings: real_assistant(section_settings, client=fake))
    answers = iter([BOM + "Weather?", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert assistant.main([]) == 0
    printed = capsys.readouterr().out.splitlines()
    tail = printed[-4:]
    assert tail == ["It is sunny.", "(from the web)", "https://a.example", "https://b.example"]
    assert fake.calls[0][1][-1]["content"] == "Weather?"


def test_key_is_read_from_the_env_file(settings: dict, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "with_key" / ".env"
    env_file.parent.mkdir()
    env_file.write_text(f"{KEY_VARIABLE}={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(assistant, "ENV_FILE", env_file)
    made: dict[str, Any] = {}
    sdk = FakeSDK([message([text_block("Hi.")])])

    def fake_anthropic(**kwargs: Any) -> FakeSDK:
        made.update(kwargs)
        return sdk

    monkeypatch.setattr(assistant.anthropic, "Anthropic", fake_anthropic)
    client = AnthropicClient(settings)
    assert made == {}                      # nothing is created before the first question
    client.complete("system", [{"role": "user", "content": "hi"}], allow_web=False)
    # F2: every wait on the network is bounded.
    assert made == {"api_key": FAKE_KEY, "timeout": 60.0, "max_retries": 1}


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(kind: type, status: int) -> Exception:
    return kind(f"error mentioning {FAKE_KEY}", response=httpx2.Response(status, request=REQUEST),
                body=None)


def body_error(kind: type, status: int, api_text: str) -> Exception:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": api_text}}
    return kind(f"Error code: {status}", response=httpx2.Response(status, request=REQUEST), body=body)


@pytest.mark.parametrize(("error", "expected"), [
    (status_error(anthropic.AuthenticationError, 401), "key was not accepted"),
    (status_error(anthropic.RateLimitError, 429), "rate limit"),
    (anthropic.APITimeoutError(request=REQUEST), "took too long"),
    (anthropic.APIConnectionError(message=FAKE_KEY, request=REQUEST), "internet connection"),
    (status_error(anthropic.InternalServerError, 500), "status 500"),
    (status_error(anthropic.BadRequestError, 400), "rejected the request (status 400)"),
    (status_error(anthropic.PermissionDeniedError, 403), "no access to the model"),
    (status_error(anthropic.NotFoundError, 404), "was not found (status 404)"),
    (ValueError("Streaming is required for operations that may take longer than 10 minutes."),
     "could not be made (ValueError)"),
])
def test_sdk_errors_become_plain_sentences_without_the_key(
        settings: dict, monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str) -> None:
    monkeypatch.setenv(KEY_VARIABLE, FAKE_KEY)
    sdk = FakeSDK(error=error)
    monkeypatch.setattr(assistant.anthropic, "Anthropic", lambda **_kwargs: sdk)
    bot = Assistant(settings)              # the real AnthropicClient, with the SDK stubbed
    with pytest.raises(AssistantError) as caught:
        bot.ask("question")
    assert expected in str(caught.value)
    assert FAKE_KEY not in str(caught.value)
    assert FAKE_KEY not in repr(caught.value)
    # F6: the SDK exception (whose request carries the key) is reachable neither way.
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert caught.value.__context__ is None


@pytest.mark.parametrize(("kind", "status", "says"), [
    (anthropic.BadRequestError, 400, "Claude rejected the request"),
    (anthropic.PermissionDeniedError, 403, "no access to the model 'test-model-from-config'"),
    (anthropic.NotFoundError, 404, "model name 'test-model-from-config' was not found"),
])
def test_errors_that_will_not_fix_themselves_show_the_api_text(
        settings: dict, monkeypatch: pytest.MonkeyPatch, kind: type, status: int, says: str) -> None:
    monkeypatch.setenv(KEY_VARIABLE, FAKE_KEY)
    error = body_error(kind, status, f"max_tokens: too large for this model (sent with {FAKE_KEY})")
    sdk = FakeSDK(error=error)
    monkeypatch.setattr(assistant.anthropic, "Anthropic", lambda **_kwargs: sdk)
    with pytest.raises(AssistantError) as caught:
        Assistant(settings).ask("question")
    message = str(caught.value)
    assert says in message
    assert "max_tokens: too large for this model" in message      # the API's own words
    assert FAKE_KEY not in message and "[key removed]" in message
    assert "Try again in a moment" not in message
    if status in (403, 404):
        assert "assistant.model in config.yaml" in message


def test_anything_shaped_like_a_key_is_cut_from_api_text() -> None:
    error = body_error(anthropic.BadRequestError, 400, "bad header sk-ant-api03-Zz9_x-1")
    assert assistant.api_message(error) == "bad header [key removed]"


# ---- AnthropicClient with the SDK call stubbed --------------------------------


def test_request_shape(settings: dict) -> None:
    sdk = FakeSDK([message([THINKING, text_block("Rover.")])] * 2)
    client = client_with(settings, sdk)
    client.complete("the system prompt", [{"role": "user", "content": "hi"}], allow_web=True)
    client.complete("the system prompt", [{"role": "user", "content": "hi"}], allow_web=False)
    with_web, without_web = sdk.requests

    assert with_web["model"] == settings["model"]
    assert with_web["max_tokens"] == settings["max_tokens"]
    assert with_web["output_config"] == {"effort": settings["effort"]}
    assert with_web["thinking"]["type"] == "adaptive"
    assert with_web["system"] == [{"type": "text", "text": "the system prompt",
                                   "cache_control": {"type": "ephemeral"}}]
    assert with_web["fallbacks"] == "default"
    assert assistant.FALLBACK_BETA in with_web["betas"]
    assert with_web["tools"] == [{"type": "web_search_20260209", "name": "web_search",
                                  "max_uses": settings["web_search_max_uses"]}]
    assert "tools" not in without_web
    for request in (with_web, without_web):
        keys = keys_anywhere(request)
        assert not keys & {"budget_tokens", "temperature", "top_p", "top_k", "tool_choice"}
        assert "cache_control" not in keys_anywhere(request["messages"])


def test_model_comes_only_from_settings(settings: dict) -> None:
    settings["model"] = "some-other-model"
    sdk = FakeSDK([message([text_block("Hi.")])])
    client_with(settings, sdk).complete("s", [{"role": "user", "content": "hi"}], allow_web=False)
    assert sdk.requests[0]["model"] == "some-other-model"


def test_unknown_effort_raises_config_error(settings: dict) -> None:
    settings["effort"] = "extreme"
    with pytest.raises(ConfigError):
        AnthropicClient(settings)


def test_reply_text_leaves_out_thinking_and_tools_and_collects_sources(settings: dict) -> None:
    response = message([
        THINKING,
        SEARCH_CALL,
        search_result("https://found.example/1", "https://found.example/2"),
        text_block("The car "),
        text_block("is fast.", urls=("https://cited.example/a", "https://cited.example/b")),
        text_block("", urls=("https://cited.example/a",)),
    ])
    reply = client_with(settings, FakeSDK([response])).complete(
        "s", [{"role": "user", "content": "q"}], allow_web=True)
    assert reply.text == "The car is fast."
    assert reply.used_web is True
    assert reply.sources == ("https://cited.example/a", "https://cited.example/b")
    assert reply.new_turns == ({"role": "assistant", "content": response.content},)
    assert reply.new_turns[0]["content"] is response.content


def test_search_results_are_the_sources_when_nothing_is_cited(settings: dict) -> None:
    response = message([SEARCH_CALL, search_result("https://found.example/1"),
                        text_block("Found it.")])
    reply = client_with(settings, FakeSDK([response])).complete(
        "s", [{"role": "user", "content": "q"}], allow_web=True)
    assert reply.sources == ("https://found.example/1",)


def test_answer_from_notes_is_not_marked_as_web(settings: dict) -> None:
    reply = client_with(settings, FakeSDK([message([THINKING, text_block("Rover.")])])).complete(
        "s", [{"role": "user", "content": "q"}], allow_web=True)
    assert reply.used_web is False and reply.sources == ()


def test_pause_turn_is_continued_with_the_paused_turn_and_no_new_user_message(settings: dict) -> None:
    paused = message([THINKING, SEARCH_CALL], stop_reason="pause_turn")
    final = message([search_result("https://found.example/1"),
                     text_block("Done.", urls=("https://found.example/1",))])
    sdk = FakeSDK([paused, final])
    turns = [{"role": "user", "content": "q"}]
    reply = client_with(settings, sdk).complete("s", turns, allow_web=True)

    assert len(sdk.requests) == 2
    second = sdk.requests[1]["messages"]
    assert second[0] is turns[0]
    assert second[1] == {"role": "assistant", "content": paused.content}
    assert second[1]["content"] is paused.content
    assert len(second) == 2                          # no "continue" message added
    assert sdk.requests[1]["system"] == sdk.requests[0]["system"]
    assert sdk.requests[1].get("tools") == sdk.requests[0].get("tools")
    assert turns == [{"role": "user", "content": "q"}]   # the caller's list is untouched
    assert [t["content"] for t in reply.new_turns] == [paused.content, final.content]
    assert reply.text == "Done." and reply.used_web


def test_endless_pause_turn_gives_up(settings: dict) -> None:
    paused = message([SEARCH_CALL], stop_reason="pause_turn")
    sdk = FakeSDK([paused] * (assistant.MAX_CONTINUATIONS + 1))
    with pytest.raises(AssistantError):
        client_with(settings, sdk).complete("s", [{"role": "user", "content": "q"}], allow_web=True)
    assert len(sdk.requests) == assistant.MAX_CONTINUATIONS + 1


def test_refusal_raises_and_keeps_the_history(settings: dict) -> None:
    refused = message([], stop_reason="refusal",
                      stop_details={"type": "refusal", "category": "cyber", "explanation": None})
    sdk = FakeSDK([message([text_block("First.")]), refused, message([text_block("Third.")])])
    bot = Assistant(settings, client=client_with(settings, sdk))
    bot.ask("one")
    with pytest.raises(AssistantError) as caught:
        bot.ask("two")
    assert "declined" in str(caught.value)
    bot.ask("three")
    sent = sdk.requests[2]["messages"]
    assert [t["role"] for t in sent] == ["user", "assistant", "user"]
    assert sent[2]["content"] == "three"


def test_max_tokens_stop_raises(settings: dict) -> None:
    cut_off = message([THINKING, text_block("The car is")], stop_reason="max_tokens")
    with pytest.raises(AssistantError) as caught:
        client_with(settings, FakeSDK([cut_off])).complete(
            "s", [{"role": "user", "content": "q"}], allow_web=True)
    assert "max_tokens" in str(caught.value)


# ---- fix round 1 ---------------------------------------------------------------


def patch_chat(monkeypatch: pytest.MonkeyPatch, settings: dict, fake: FakeClient) -> None:
    """Make main() build its Assistant from these settings with this fake client."""
    real_assistant = assistant.Assistant
    monkeypatch.setattr(assistant, "load_config", lambda: {"assistant": dict(settings)})
    monkeypatch.setattr(assistant, "Assistant",
                        lambda section_settings: real_assistant(section_settings, client=fake))


@pytest.mark.parametrize("typed", [BOM + "/quit", CP1252_BOM + "/quit", "  " + CP1252_BOM + " /quit "])
def test_f1_clean_strips_both_forms_of_the_byte_order_mark(typed: str) -> None:
    assert assistant.clean(typed) == "/quit"


def test_f1_clean_keeps_non_ascii_text() -> None:
    assert assistant.clean("Qu'est-ce que c'est, Rover ? éè 你好") == \
        "Qu'est-ce que c'est, Rover ? éè 你好"


def test_f1_piped_input_is_read_as_utf8_and_the_mark_is_dropped(
        settings: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """What PowerShell 5.1 sends: UTF-8 bytes with a mark, read by Python as cp1252."""
    import io
    piped = (BOM + "Café, naïve, 你好?\n/quit\n").encode("utf-8")
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(piped), encoding="cp1252"))
    fake = FakeClient()
    patch_chat(monkeypatch, settings, fake)
    assert assistant.main([]) == 0
    assert [call[1][-1]["content"] for call in fake.calls] == ["Café, naïve, 你好?"]


def test_f2_a_question_does_not_wait_forever_for_the_previous_one(
        settings: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assistant, "LOCK_WAIT_SECONDS", 0.2)
    release = threading.Event()

    class StuckClient(FakeClient):
        def complete(self, system: str, turns: list, allow_web: bool) -> LLMReply:
            release.wait(5)
            return super().complete(system, turns, allow_web)

    bot = Assistant(settings, client=StuckClient())
    first = threading.Thread(target=bot.ask, args=("one",))
    first.start()
    time.sleep(0.05)
    try:
        with pytest.raises(AssistantError, match="still busy with the previous question"):
            bot.ask("two")
        with pytest.raises(AssistantError, match="still busy with the previous question"):
            bot.reset()
    finally:
        release.set()
        first.join(5)
    bot.reset()                               # free again once the first answer is in
    assert bot.ask("three").text


def test_f3_notes_saved_as_ansi_give_a_plain_error(tmp_path: Path) -> None:
    path = tmp_path / "ansi.md"
    path.write_bytes("Café notes".encode("cp1252"))
    with pytest.raises(AssistantError, match="UTF-8") as caught:
        load_notes(path)
    assert f"{tmp_path.name}/ansi.md" in str(caught.value)
    assert str(path) not in str(caught.value)
    assert caught.value.__context__ is None


def test_f3_notes_with_a_byte_order_mark_are_read_without_it(tmp_path: Path) -> None:
    path = tmp_path / "bom.md"
    path.write_bytes((BOM + "# Project\nRover").encode("utf-8"))
    assert load_notes(path) == "# Project\nRover"


def test_f3_ctrl_c_while_answering_exits_quietly(
        settings: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeClient()
    fake.error = KeyboardInterrupt()
    patch_chat(monkeypatch, settings, fake)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "question")
    assert assistant.main([]) == 0
    assert "Traceback" not in capsys.readouterr().out


def test_f3_ctrl_c_at_the_prompt_exits_quietly(
        settings: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_chat(monkeypatch, settings, FakeClient())

    def interrupted(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)
    assert assistant.main([]) == 0


def test_f3_an_unexpected_error_prints_one_line_and_the_chat_goes_on(
        settings: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class BrokenOnce(FakeClient):
        def complete(self, system: str, turns: list, allow_web: bool) -> LLMReply:
            if not self.calls:
                self.calls.append((system, list(turns), allow_web))
                raise RuntimeError(f"internal detail {FAKE_KEY}")
            return super().complete(system, turns, allow_web)

    fake = BrokenOnce()
    patch_chat(monkeypatch, settings, fake)
    answers = iter(["first", "second", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert assistant.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "Something went wrong while answering (RuntimeError). Please ask again." in lines
    assert lines[-1] == "answer 2"
    assert not any(FAKE_KEY in line for line in lines)


def test_f3_a_too_long_question_in_the_chat_prints_and_goes_on(
        settings: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeClient()
    patch_chat(monkeypatch, settings, fake)
    answers = iter(["x" * 2001, "short", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert assistant.main([]) == 0
    printed = capsys.readouterr().out
    assert "too long" in printed
    assert len(fake.calls) == 1


def test_f4_narration_before_the_search_is_left_out(settings: dict) -> None:
    response = message([
        THINKING,
        text_block("Let me search for that. "),
        SEARCH_CALL,
        search_result("https://found.example/1"),
        text_block("The answer", urls=("https://cited.example/a",)),
        text_block(" is 42."),
    ])
    reply = client_with(settings, FakeSDK([response])).complete(
        "s", [{"role": "user", "content": "q"}], allow_web=True)
    assert reply.text == "The answer is 42."
    assert reply.sources == ("https://cited.example/a",)


def test_f4_text_from_a_paused_turn_is_left_out_but_its_citations_count(settings: dict) -> None:
    paused = message([text_block("Searching now.", urls=("https://early.example",)), SEARCH_CALL],
                     stop_reason="pause_turn")
    final = message([search_result("https://found.example/1"),
                     text_block("Final answer.", urls=("https://late.example",))])
    reply = client_with(settings, FakeSDK([paused, final])).complete(
        "s", [{"role": "user", "content": "q"}], allow_web=True)
    assert reply.text == "Final answer."
    assert reply.sources == ("https://early.example", "https://late.example")


def test_f4_without_a_search_all_text_blocks_are_the_answer(settings: dict) -> None:
    response = message([THINKING, text_block("Rover "), text_block("follows lines.")])
    reply = client_with(settings, FakeSDK([response])).complete(
        "s", [{"role": "user", "content": "q"}], allow_web=True)
    assert reply.text == "Rover follows lines."


def test_f5_a_question_over_2000_characters_is_refused_before_sending(settings: dict) -> None:
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.ask("first")
    with pytest.raises(ValueError, match="too long"):
        bot.ask("x" * 2001)
    assert len(fake.calls) == 1
    bot.ask("y" * 2000)                     # exactly the limit is fine
    assert [t["content"] for t in fake.calls[1][1]] == ["first", ["block-1"], "y" * 2000]


def test_f7_placeholder_notes_give_one_warning(settings: dict, notes_path: Path,
                                               caplog: pytest.LogCaptureFixture) -> None:
    notes_path.write_text("# Team\n[REPLACE] names\n# Demo\n[REPLACE] where\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="car_vision.assistant"):
        Assistant(settings, client=FakeClient())
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "placeholder" in warnings[0].getMessage()


def test_f7_real_notes_give_no_warning(settings: dict, notes_path: Path,
                                       caplog: pytest.LogCaptureFixture) -> None:
    notes_path.write_text("<!-- replace lines starting with [REPLACE] -->\n# Project\nRover.\n",
                          encoding="utf-8")
    with caplog.at_level("WARNING", logger="car_vision.assistant"):
        Assistant(settings, client=FakeClient())
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_f7_the_shipped_template_warns(settings: dict, caplog: pytest.LogCaptureFixture) -> None:
    settings["notes_file"] = "data/knowledge.md"
    with caplog.at_level("WARNING", logger="car_vision.assistant"):
        Assistant(settings, client=FakeClient())
    assert any("placeholder" in r.getMessage() for r in caplog.records)


def test_f7_html_comments_do_not_reach_the_model() -> None:
    notes = "<!--\nTEMPLATE note for the team\n-->\n# Project\nRover.<!-- inline -->\n"
    prompt = build_system_prompt(notes)
    assert "TEMPLATE note" not in prompt and "inline" not in prompt and "<!--" not in prompt
    assert "# Project\nRover." in prompt
    assert "not written any notes" in build_system_prompt("<!-- only a comment -->")


# ---- M6: person briefs ---------------------------------------------------------


def add_person(people_dir: Path, person_id: str, text: str | bytes) -> Path:
    """Make data/people/<id>/about.md in a temporary folder."""
    folder = people_dir / person_id
    folder.mkdir(parents=True, exist_ok=True)
    about = folder / "about.md"
    if isinstance(text, bytes):
        about.write_bytes(text)
    else:
        about.write_text(text, encoding="utf-8")
    return about


def test_briefs_reach_the_system_prompt_in_sorted_order(settings: dict, people_dir: Path) -> None:
    add_person(people_dir, "sara", "# Sara Ahmed\nRole: Team lead\n- Built the chassis\n")
    add_person(people_dir, "omar", "# Omar\n<!-- team note -->\nWrote the vision code.\n")
    fake = FakeClient()
    bot = Assistant(settings, state=state_with("Sara Ahmed"), client=fake)
    bot.ask("Who is this?")
    bot.ask("And what did they build?")
    first, second = fake.calls[0][0], fake.calls[1][0]
    assert first == second
    assert first.index('<person id="omar" name="Omar">') < \
        first.index('<person id="sara" name="Sara Ahmed">')
    assert "Built the chassis" in first and "Wrote the vision code." in first
    assert "team note" not in first and "<!--" not in first
    assert first.index("</notes>") < first.index("</people>")
    # Nothing per-question goes into the system prompt.
    assert "Who is this?" not in first


def test_brief_without_a_heading_is_named_by_its_folder(people_dir: Path) -> None:
    add_person(people_dir, "lina", "Lina tests the car.\n")
    assert [(b.person_id, b.name) for b in load_briefs(people_dir)] == [("lina", "lina")]


def test_folders_that_add_no_brief_are_left_out(people_dir: Path) -> None:
    add_person(people_dir, "Not-An-Id", "# Bad\n")
    add_person(people_dir, "empty", "<!-- only a comment -->\n  \n")
    (people_dir / "no_about").mkdir()
    add_person(people_dir, "zoe", "# Zoe\n")
    assert [b.person_id for b in load_briefs(people_dir)] == ["zoe"]


def test_missing_people_folder_adds_nothing(settings: dict, tmp_path: Path) -> None:
    settings["people_dir"] = str(tmp_path / "nowhere")
    fake = FakeClient()
    Assistant(settings, client=fake).ask("hi")
    assert "</people>" not in fake.calls[0][0]
    assert fake.calls[0][0] == build_system_prompt(NOTES)


def test_empty_people_folder_adds_nothing(settings: dict, people_dir: Path) -> None:
    people_dir.mkdir()
    fake = FakeClient()
    Assistant(settings, client=fake).ask("hi")
    assert "</people>" not in fake.calls[0][0]


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_brief_in_a_wrong_encoding_is_skipped_with_one_warning(
        settings: dict, people_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    add_person(people_dir, "sara", "# Sára\nCafé".encode("cp1252"))
    add_person(people_dir, "omar", "# Omar\n")
    fake = FakeClient()
    with caplog.at_level("WARNING", logger="car_vision.assistant"):
        Assistant(settings, client=fake).ask("hi")         # the assistant still starts
    assert '<person id="omar"' in fake.calls[0][0] and 'id="sara"' not in fake.calls[0][0]
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert "sara/about.md" in warnings[0] and "UTF-8" in warnings[0]


def test_an_about_file_that_cannot_be_read_is_skipped(
        people_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    locked = add_person(people_dir, "sara", "# Sara\n")
    add_person(people_dir, "omar", "# Omar\n")
    real_read_text = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == locked:
            raise PermissionError("held by an editor")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    with caplog.at_level("WARNING", logger="car_vision.assistant"):
        assert [b.person_id for b in load_briefs(people_dir)] == ["omar"]
    warnings = warnings_in(caplog)
    assert len(warnings) == 1 and "PermissionError" in warnings[0] and "sara/about.md" in warnings[0]


def test_a_notes_file_that_cannot_be_read_gives_a_plain_error(
        notes_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def read_text(_path: Path, *_args: Any, **_kwargs: Any) -> str:
        raise PermissionError(f"held by an editor: {notes_path}")

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(AssistantError, match="could not be read") as caught:
        load_notes(notes_path)
    assert "PermissionError" in str(caught.value)
    assert str(notes_path) not in str(caught.value)
    assert caught.value.__context__ is None


def test_each_invalid_folder_name_gives_one_warning(
        people_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    add_person(people_dir, "Sara", "# Sara\n")
    add_person(people_dir, "omar-k", "# Omar\n")
    add_person(people_dir, "lina", "# Lina\n")
    with caplog.at_level("WARNING", logger="car_vision.assistant"):
        assert [b.person_id for b in load_briefs(people_dir)] == ["lina"]
    warnings = warnings_in(caplog)
    assert len(warnings) == 2
    assert any("people/Sara" in w for w in warnings) and any("people/omar-k" in w for w in warnings)


def test_paths_inside_the_project_are_shown_relative_to_it() -> None:
    assert assistant.shown_path(PROJECT_ROOT / "data" / "knowledge.md") == "data/knowledge.md"
    about = PROJECT_ROOT / "data" / "people" / "sara" / "about.md"
    assert assistant.shown_path(about) == "data/people/sara/about.md"


def test_a_missing_notes_file_in_the_project_shows_the_short_path(settings: dict) -> None:
    settings["notes_file"] = "data/no_such_notes_file.md"
    with pytest.raises(AssistantError) as caught:
        Assistant(settings, client=FakeClient())
    assert "data/no_such_notes_file.md" in str(caught.value)
    assert str(PROJECT_ROOT) not in str(caught.value)


def test_brief_with_a_byte_order_mark(people_dir: Path) -> None:
    add_person(people_dir, "sara", (BOM + "# Sara\n").encode("utf-8"))
    assert load_briefs(people_dir)[0].name == "Sara"


def test_missing_people_dir_setting_raises_config_error(settings: dict) -> None:
    del settings["people_dir"]
    with pytest.raises(ConfigError, match="people_dir"):
        Assistant(settings, client=FakeClient())


def test_chat_prompt_says_nothing_about_attached_pictures() -> None:
    prompt = build_system_prompt(NOTES).lower()
    assert "attached" not in prompt
    assert "never guess who someone is from a picture" in prompt


# ---- M6: the objects line --------------------------------------------------------


def state_seeing(labels: list[str], age_seconds: float = 0.0) -> WorldState:
    state = state_with("Sara")
    detections = [Detection(label=label, confidence=0.8, box=(0, 0, 10, 10)) for label in labels]
    state.update_detections(detections, (640, 480), now=time.monotonic() - age_seconds)
    return state


def test_objects_line_follows_the_people_line(settings: dict) -> None:
    fake = FakeClient()
    state = state_seeing(["laptop", "person", "cup", "laptop"])
    Assistant(settings, state=state, client=fake).ask("What is on the table?")
    assert fake.calls[0][1][-1]["content"] == (
        "In front of the camera right now: Sara.\n"
        "The object detector sees right now: cup, laptop.\n"
        "What is on the table?")


def test_no_objects_line_for_stale_detections(settings: dict) -> None:
    fake = FakeClient()
    state = state_seeing(["laptop"], age_seconds=settings["seen_within_seconds"] + 5)
    Assistant(settings, state=state, client=fake).ask("Anything?")
    assert "object detector" not in fake.calls[0][1][-1]["content"]


def test_no_objects_line_with_only_people_in_view(settings: dict) -> None:
    fake = FakeClient()
    Assistant(settings, state=state_seeing(["person", "person"]), client=fake).ask("Anything?")
    assert fake.calls[0][1][-1]["content"] == "In front of the camera right now: Sara.\nAnything?"


# ---- M6: pictures --------------------------------------------------------------


PICTURE = assistant.JPEG_MARKER + b"\xe0 a small fake jpeg"


def test_user_turn_with_a_picture_has_the_documented_image_block() -> None:
    turn = assistant.user_turn("What is it?", PICTURE)
    assert turn["role"] == "user"
    picture, text = turn["content"]
    assert picture == {"type": "image",
                       "source": {"type": "base64", "media_type": "image/jpeg",
                                  "data": base64.standard_b64encode(PICTURE).decode("ascii")}}
    assert text == {"type": "text", "text": "What is it?"}
    assert assistant.user_turn("hi") == {"role": "user", "content": "hi"}


@pytest.mark.parametrize("image", [
    "not bytes",
    b"\x89PNG\r\n\x1a\n not a jpeg",
    b"",
    assistant.JPEG_MARKER + b"x" * assistant.MAX_IMAGE_BYTES,
])
def test_check_image_refuses_anything_but_one_small_jpeg(image: Any) -> None:
    with pytest.raises(ValueError):
        assistant.check_image(image)


def test_check_image_accepts_exactly_the_limit() -> None:
    image = assistant.JPEG_MARKER + b"x" * (assistant.MAX_IMAGE_BYTES - 3)
    assert assistant.check_image(bytearray(image)) == image


# ---- M7: describe ----------------------------------------------------------------


GOOD_ANSWER = """\
NAME: Laptop
ABOUT: A portable computer with a screen and keyboard.
SPECS:
- Screen: 14 to 16 inches (typical)
- Weight: about 1.5 kg (typical)
"""


def described(settings: dict, label: str | None = "laptop", text: str = GOOD_ANSWER,
              **reply: Any) -> tuple[ObjectInfo, FakeClient]:
    fake = FakeClient([LLMReply(text=text, new_turns=({"role": "assistant", "content": []},),
                                **reply)])
    info = Assistant(settings, client=fake).describe(PICTURE, label)
    return info, fake


def test_describe_reads_the_answer(settings: dict) -> None:
    info, _ = described(settings)
    assert info == ObjectInfo(name="Laptop", summary="A portable computer with a screen and keyboard.",
                              specs=(("Screen", "14 to 16 inches (typical)"),
                                     ("Weight", "about 1.5 kg (typical)")))


def test_describe_system_prompt_is_the_same_for_every_call(settings: dict, people_dir: Path) -> None:
    add_person(people_dir, "sara", "# Sara\n")
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.describe(PICTURE, "laptop")
    bot.describe(PICTURE, None)
    assert fake.calls[0][0] == fake.calls[1][0] == assistant.DESCRIBE_PROMPT
    # Its own prompt: neither the notes nor the people go with a picture.
    assert NOTES.strip() not in assistant.DESCRIBE_PROMPT and "Sara" not in fake.calls[0][0]


@pytest.mark.parametrize(("label", "sentence"), [
    ("laptop", 'The object detector labelled this "laptop".'),
    ("cell  phone ", 'The object detector labelled this "cell phone".'),
    (None, "The object detector has no label for this. The visitor clicked the middle of the picture."),
])
def test_describe_sends_the_picture_and_the_label_line(settings: dict, label: str | None,
                                                       sentence: str) -> None:
    _, fake = described(settings, label)
    turns = fake.calls[0][1]
    assert turns == [assistant.user_turn(sentence, PICTURE)]
    picture, text = turns[0]["content"]
    assert base64.standard_b64decode(picture["source"]["data"]) == PICTURE
    assert text == {"type": "text", "text": sentence}


@pytest.mark.parametrize("web_search", [True, False])
def test_describe_web_search_follows_the_setting(settings: dict, web_search: bool) -> None:
    settings["web_search"] = web_search
    _, fake = described(settings)
    assert fake.calls[0][2] is web_search


def test_describe_neither_sends_nor_changes_the_chat_history(settings: dict) -> None:
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.ask("one")
    bot.describe(PICTURE, "laptop")
    bot.ask("two")
    assert len(fake.calls[1][1]) == 1                    # describe: only its own user turn
    assert [t["content"] for t in fake.calls[2][1]] == ["one", ["block-1"], "two"]


def test_describe_passes_sources_and_used_web_through(settings: dict) -> None:
    info, _ = described(settings, used_web=True,
                        sources=("https://a.example", "https://b.example", "https://a.example"))
    assert info.used_web is True
    assert info.sources == ("https://a.example", "https://b.example")


@pytest.mark.parametrize("image", ["text", b"\x89PNG not a jpeg",
                                   assistant.JPEG_MARKER + b"x" * assistant.MAX_IMAGE_BYTES])
def test_describe_refuses_a_bad_picture_before_sending(settings: dict, image: Any) -> None:
    fake = FakeClient()
    with pytest.raises(ValueError):
        Assistant(settings, client=fake).describe(image, "laptop")
    assert fake.calls == []


@pytest.mark.parametrize("label", ["x" * 61, 42, b"laptop", ["laptop"]])
def test_describe_refuses_a_bad_label_before_sending(settings: dict, label: Any) -> None:
    fake = FakeClient()
    with pytest.raises(ValueError):
        Assistant(settings, client=fake).describe(PICTURE, label)
    assert fake.calls == []


def test_describe_accepts_a_label_of_exactly_sixty_characters(settings: dict) -> None:
    _, fake = described(settings, label="x" * 60)
    assert len(fake.calls) == 1


def test_a_second_describe_gives_up_when_busy_and_the_chat_still_works(
        settings: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assistant, "DESCRIBE_WAIT_SECONDS", 0.2)
    release = threading.Event()

    class StuckOnPictures(FakeClient):
        def complete(self, system: str, turns: list, allow_web: bool) -> LLMReply:
            if isinstance(turns[-1]["content"], list):     # a describe call
                release.wait(5)
            return super().complete(system, turns, allow_web)

    bot = Assistant(settings, client=StuckOnPictures())
    first = threading.Thread(target=bot.describe, args=(PICTURE, "laptop"))
    first.start()
    time.sleep(0.05)
    try:
        with pytest.raises(AssistantError, match="still busy"):
            bot.describe(PICTURE, "cup")
        assert bot.ask("Is the chat free?").text       # its own lock: the chat does not wait
    finally:
        release.set()
        first.join(5)
    assert bot.describe(PICTURE, "cup").name           # free again once the first is done


@pytest.mark.parametrize(("stop_reason", "says"), [("refusal", "declined"),
                                                   ("max_tokens", "max_tokens")])
def test_describe_refusal_and_max_tokens_raise(settings: dict, stop_reason: str, says: str) -> None:
    response = message([text_block("NAME: Lap")], stop_reason=stop_reason)
    bot = Assistant(settings, client=client_with(settings, FakeSDK([response])))
    with pytest.raises(AssistantError, match=says):
        bot.describe(PICTURE, "laptop")


# ---- M7: reading the answer --------------------------------------------------------


def test_parse_keys_in_any_case_and_in_bold() -> None:
    text = ("**name:** Office chair\n**About**: A chair on wheels.\n"
            "**SPECS:**\n* **Height:** 90 to 110 cm\n* Material: mesh and plastic\n")
    assert parse_object_info(text, "chair") == (
        "Office chair", "A chair on wheels.",
        (("Height", "90 to 110 cm"), ("Material", "mesh and plastic")))


def test_parse_about_over_several_lines() -> None:
    text = "NAME: Cup\nABOUT: A cup for drinks.\nIt holds about 250 ml.\n\nIt is often ceramic.\nSPECS:\n"
    name, summary, specs = parse_object_info(text, "cup")
    assert summary == "A cup for drinks. It holds about 250 ml. It is often ceramic."
    assert specs == ()


def test_parse_keeps_six_specs_and_skips_lines_without_a_colon() -> None:
    spec_lines = "".join(f"- Spec {n}: value {n}\n" for n in range(1, 9))
    text = f"NAME: Robot\nABOUT: A robot.\nSPECS:\n- no colon here\n{spec_lines}- Empty:\n"
    _, _, specs = parse_object_info(text, None)
    assert specs == tuple((f"Spec {n}", f"value {n}") for n in range(1, 7))


def test_parse_keeps_colons_inside_a_value() -> None:
    _, _, specs = parse_object_info("NAME: Clock\nSPECS:\n- Shows: 12:45 style time\n", None)
    assert specs == (("Shows", "12:45 style time"),)


def test_parse_a_spec_named_like_a_key_stays_a_spec() -> None:
    _, _, specs = parse_object_info("NAME: Laptop\nSPECS:\n- Name: ThinkPad\n- About: 14 inch\n", None)
    assert specs == (("Name", "ThinkPad"), ("About", "14 inch"))


@pytest.mark.parametrize(("label", "name"), [("cell phone", "Cell phone"), (None, "Unknown object"),
                                             ("", "Unknown object")])
def test_parse_without_a_name_line_uses_the_label_and_the_whole_text(label: str | None,
                                                                    name: str) -> None:
    text = "  I think this is a phone lying on a desk.\nIt has a dark case.  \n"
    assert parse_object_info(text, label) == (
        name, "I think this is a phone lying on a desk.\nIt has a dark case.", ())


def test_parse_passes_the_unknown_and_person_names_through() -> None:
    assert parse_object_info("NAME: Unknown object\nABOUT: I cannot tell.", None)[0] == "Unknown object"
    assert parse_object_info("NAME: Person\nABOUT: A person.", "person")[0] == "Person"


@pytest.mark.parametrize("sources", ["Sources:", "**Sources:**", "source: the web", "SOURCES:"])
def test_parse_stops_the_specs_at_a_sources_line(sources: str) -> None:
    text = (f"NAME: Laptop\nABOUT: A computer.\nSPECS:\n- Weight: 1.5 kg\n{sources}\n"
            "- https://example.com/page\n- Shop: example.com\n")
    assert parse_object_info(text, None)[2] == (("Weight", "1.5 kg"),)


@pytest.mark.parametrize("line", ["- https://example.com/page", "- [1]: from the maker's page",
                                  "- example.com/specs: 2 kg", "- HTTP://EXAMPLE.COM: x"])
def test_parse_refuses_a_spec_named_like_a_link(line: str) -> None:
    text = f"NAME: Laptop\nSPECS:\n- Weight: 1.5 kg\n{line}\n"
    assert parse_object_info(text, None)[2] == (("Weight", "1.5 kg"),)


@pytest.mark.parametrize("heading", ["## Specs", "**SPECS**", "Specs", "Specifications",
                                     "SPECIFICATIONS:", "### **Specifications:**"])
def test_parse_a_specs_heading_without_a_colon_starts_the_specs(heading: str) -> None:
    text = f"NAME: Laptop\nABOUT: A computer.\n{heading}\n- Weight: 1.5 kg\n"
    assert parse_object_info(text, None) == ("Laptop", "A computer.", (("Weight", "1.5 kg"),))


@pytest.mark.parametrize("line", ["SPECS: - Weight: 1.5 kg", "**Specs:** Weight: 1.5 kg"])
def test_parse_a_spec_on_the_specs_line_is_kept(line: str) -> None:
    text = f"NAME: Laptop\n{line}\n- Screen: 14 inches\n"
    assert parse_object_info(text, None)[2] == (("Weight", "1.5 kg"), ("Screen", "14 inches"))


def test_parse_numbered_and_dash_bullets() -> None:
    text = ("NAME: Laptop\nSPECS:\n1. Weight: 1.5 kg\n2) Screen: 14 inches\n"
            "– Battery: 8 hours\n— **Price:** about 800 euros\n")
    assert parse_object_info(text, None)[2] == (
        ("Weight", "1.5 kg"), ("Screen", "14 inches"), ("Battery", "8 hours"),
        ("Price", "about 800 euros"))


@pytest.mark.parametrize(("label", "name"), [("laptop", "Laptop"), (None, "Unknown object")])
def test_parse_an_empty_name_keeps_about_and_specs(label: str | None, name: str) -> None:
    text = "NAME:\nABOUT: A computer.\nSPECS:\n- Weight: 1.5 kg\n"
    assert parse_object_info(text, label) == (name, "A computer.", (("Weight", "1.5 kg"),))


def test_parse_no_name_line_but_other_keys_keeps_them() -> None:
    text = "ABOUT: A computer.\nSPECS:\n- Weight: 1.5 kg\n"
    assert parse_object_info(text, "laptop") == ("Laptop", "A computer.", (("Weight", "1.5 kg"),))


# ---- M8: follow-up questions -------------------------------------------------------


INFO = ObjectInfo(name="Laptop", summary="A portable computer.",
                  specs=(("Weight", "about 1.5 kg"), ("Battery", "8 hours")))
LAPTOP_LINE = 'The object detector labelled this "laptop".'
PAIRS = [("How heavy is it?", "About 1.5 kg."), ("Is it new?", "It looks a few years old.")]


def picture_block() -> dict[str, Any]:
    return assistant.user_turn("x", PICTURE)["content"][0]


def followed_up(settings: dict, info: ObjectInfo | None = INFO, earlier: Any = (),
                question: str = "What is it made of?", label: Any = "laptop",
                **reply: Any) -> tuple[Answer, FakeClient]:
    fake = FakeClient([LLMReply(text="Aluminium and plastic.",
                                new_turns=({"role": "assistant", "content": []},), **reply)])
    answer = Assistant(settings, client=fake).follow_up(PICTURE, label, info, earlier, question)
    return answer, fake


def test_follow_up_turns_without_a_description(settings: dict) -> None:
    _, fake = followed_up(settings, info=None, label=None)
    no_label = assistant.label_text(None)
    # The question joins the picture turn, so two user turns never follow each other.
    assert fake.calls[0][1] == [{"role": "user", "content": [
        picture_block(), {"type": "text", "text": no_label},
        {"type": "text", "text": "What is it made of?"}]}]


def test_follow_up_turns_without_a_description_but_with_pairs(settings: dict) -> None:
    _, fake = followed_up(settings, info=None, earlier=PAIRS[:1])
    assert fake.calls[0][1] == [
        {"role": "user", "content": [picture_block(), {"type": "text", "text": LAPTOP_LINE},
                                     {"type": "text", "text": "How heavy is it?"}]},
        {"role": "assistant", "content": "About 1.5 kg."},
        {"role": "user", "content": "What is it made of?"},
    ]


def test_follow_up_turns_with_a_description_and_no_pairs(settings: dict) -> None:
    _, fake = followed_up(settings)
    assert fake.calls[0][1] == [
        assistant.user_turn(LAPTOP_LINE, PICTURE),          # the same turn describe sends
        {"role": "assistant", "content": assistant.info_text(INFO)},
        {"role": "user", "content": "What is it made of?"},
    ]


def test_follow_up_turns_with_a_description_and_two_pairs(settings: dict) -> None:
    _, fake = followed_up(settings, earlier=PAIRS, question="  Who makes it?  ")
    assert fake.calls[0][1] == [
        assistant.user_turn(LAPTOP_LINE, PICTURE),
        {"role": "assistant", "content": assistant.info_text(INFO)},
        {"role": "user", "content": "How heavy is it?"},
        {"role": "assistant", "content": "About 1.5 kg."},
        {"role": "user", "content": "Is it new?"},
        {"role": "assistant", "content": "It looks a few years old."},
        {"role": "user", "content": "Who makes it?"},
    ]


def test_follow_up_turns_match_what_describe_sent(settings: dict) -> None:
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.describe(PICTURE, "laptop")
    bot.follow_up(PICTURE, "laptop", INFO, [], "More?")
    assert fake.calls[1][1][0] == fake.calls[0][1][0]


def test_follow_up_system_prompt_is_its_own_and_the_same_every_call(settings: dict) -> None:
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.follow_up(PICTURE, "laptop", INFO, [], "One?")
    bot.follow_up(PICTURE, None, None, PAIRS, "Two?")
    assert fake.calls[0][0] == fake.calls[1][0] == assistant.FOLLOW_UP_PROMPT
    assert assistant.FOLLOW_UP_PROMPT not in (assistant.DESCRIBE_PROMPT, build_system_prompt(NOTES))


def test_follow_up_answer_and_sources(settings: dict) -> None:
    answer, _ = followed_up(settings, used_web=True,
                            sources=("https://a.example", "https://a.example", "https://b.example"))
    assert answer == Answer(text="Aluminium and plastic.", used_web=True,
                            sources=("https://a.example", "https://b.example"))


@pytest.mark.parametrize("web_search", [True, False])
def test_follow_up_web_search_follows_the_setting(settings: dict, web_search: bool) -> None:
    settings["web_search"] = web_search
    _, fake = followed_up(settings)
    assert fake.calls[0][2] is web_search


def test_follow_up_neither_sends_nor_changes_the_chat_history(settings: dict) -> None:
    fake = FakeClient()
    bot = Assistant(settings, client=fake)
    bot.ask("one")
    bot.follow_up(PICTURE, "laptop", INFO, PAIRS, "What is it made of?")
    bot.ask("two")
    assert "one" not in [t["content"] for t in fake.calls[1][1]]
    assert [t["content"] for t in fake.calls[2][1]] == ["one", ["block-1"], "two"]


@pytest.mark.parametrize(("change", "says"), [
    ({"image": b"\x89PNG not a jpeg"}, "JPEG"),
    ({"label": "x" * 61}, "too long"),
    ({"label": 7}, "must be text"),
    ({"question": "   "}, "empty"),
    ({"question": "x" * 2001}, "too long"),
    ({"question": None}, "must be text"),
    ({"earlier": [("q", "a")] * 11}, "At most 10"),
    ({"earlier": [("q", "")]}, "pair of texts"),
    ({"earlier": [("q", "a", "extra")]}, "pair of texts"),
    ({"earlier": ["qa"]}, "pair of texts"),
])
def test_follow_up_refuses_bad_input_before_sending(settings: dict, change: dict, says: str) -> None:
    call = {"image": PICTURE, "label": "laptop", "info": INFO, "earlier": [], "question": "Why?"}
    call.update(change)
    fake = FakeClient()
    with pytest.raises(ValueError, match=says):
        Assistant(settings, client=fake).follow_up(**call)
    assert fake.calls == []


def test_follow_up_accepts_ten_earlier_pairs_given_as_lists(settings: dict) -> None:
    _, fake = followed_up(settings, earlier=[["q", "a"]] * 10)
    assert len(fake.calls[0][1]) == 2 + 20 + 1


@pytest.mark.parametrize(("stop_reason", "says"), [("refusal", "declined"),
                                                   ("max_tokens", "max_tokens")])
def test_follow_up_refusal_and_max_tokens_raise(settings: dict, stop_reason: str, says: str) -> None:
    response = message([text_block("It is")], stop_reason=stop_reason)
    bot = Assistant(settings, client=client_with(settings, FakeSDK([response])))
    with pytest.raises(AssistantError, match=says):
        bot.follow_up(PICTURE, "laptop", INFO, [], "Why?")


def test_follow_up_waits_on_the_describe_lock(settings: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assistant, "DESCRIBE_WAIT_SECONDS", 0.2)
    release = threading.Event()

    class StuckOnPictures(FakeClient):
        def complete(self, system: str, turns: list, allow_web: bool) -> LLMReply:
            if isinstance(turns[0]["content"], list):        # describe or follow_up
                release.wait(5)
            return super().complete(system, turns, allow_web)

    bot = Assistant(settings, client=StuckOnPictures())
    first = threading.Thread(target=bot.describe, args=(PICTURE, "laptop"))
    first.start()
    time.sleep(0.05)
    try:
        with pytest.raises(AssistantError) as caught:
            bot.follow_up(PICTURE, "laptop", INFO, [], "Why?")
        assert str(caught.value) == assistant.DESCRIBE_BUSY_MESSAGE
        assert bot.ask("Is the chat free?").text
    finally:
        release.set()
        first.join(5)
    assert bot.follow_up(PICTURE, "laptop", INFO, [], "Why?").text


@pytest.mark.parametrize("info", [
    INFO,
    ObjectInfo(name="Clock", summary="A wall clock. It ticks.", specs=(("Shows", "12:45 style time"),)),
    ObjectInfo(name="Unknown object", summary="I cannot tell what this is.", specs=()),
    ObjectInfo(name="Office chair", summary="",
               specs=tuple((f"Spec {n}", f"value {n}") for n in range(1, 7))),
    ObjectInfo(name="Robot", summary="A robot.", specs=(("Name", "Rover"), ("About", "2 kg"))),
])
def test_info_text_round_trips_through_the_parser(info: ObjectInfo) -> None:
    assert parse_object_info(assistant.info_text(info), None) == (info.name, info.summary, info.specs)


def test_info_text_shape() -> None:
    assert assistant.info_text(INFO) == ("NAME: Laptop\nABOUT: A portable computer.\nSPECS:\n"
                                         "- Weight: about 1.5 kg\n- Battery: 8 hours")


def test_info_text_puts_a_summary_on_one_line() -> None:
    info = ObjectInfo(name="Cup", summary="A cup.\nIt holds tea.", specs=())
    assert parse_object_info(assistant.info_text(info), None)[1] == "A cup. It holds tea."


# ---- the one-module rule and the live test -------------------------------------


def test_only_assistant_py_imports_anthropic() -> None:
    pattern = re.compile(r"^\s*(import\s+anthropic\b|from\s+anthropic\b)", re.MULTILINE)
    offenders = [path.name for path in (PROJECT_ROOT / "car_vision").rglob("*.py")
                 if path.name != "assistant.py"
                 and pattern.search(path.read_text(encoding="utf-8"))]
    assert offenders == []


@pytest.mark.hardware
def test_live_question_with_the_real_key() -> None:
    """Asks Claude one real question. Needs the user's .env and the internet."""
    try:
        assistant.read_api_key()
    except AssistantError:
        pytest.skip("no API key set")
    answer = Assistant(section(load_config(), "assistant")).ask(
        "In one sentence, what is this project about?")
    assert answer.text.strip()
