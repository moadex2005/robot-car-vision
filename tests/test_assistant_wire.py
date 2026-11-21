"""Reviewer tests for car_vision.assistant that go through the real SDK.

test_assistant.py replaces the SDK object with a stub, so it never checks that the
SDK can turn a real response back into a request. These tests keep the real
`anthropic` client but give it an in-memory transport, so no network is used.
The key is a fake one; the project's .env is never read.

Tests marked xfail record known problems found in review; they turn into
failures-to-be-removed (XPASS) once the problem is fixed.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from car_vision import assistant

FAKE_KEY = "sk-ant-wire-test-FAKE-KEY-0000"

FULL_CONTENT = [
    {"type": "fallback", "from": {"model": "claude-sonnet-5-5"}, "to": {"model": "claude-sonnet-5"}},
    {"type": "thinking", "thinking": "", "signature": "SIG1"},
    {"type": "text", "text": "Let me search. "},
    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
     "input": {"query": "x"}, "caller": {"type": "direct"}},
    {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
        {"type": "web_search_result", "url": "https://a.example", "title": "A",
         "encrypted_content": "ENC", "page_age": None}]},
    {"type": "text", "text": "Answer.", "citations": [
        {"type": "web_search_result_location", "url": "https://a.example", "title": "A",
         "cited_text": "c", "encrypted_index": "ix"}]},
]


def reply_json(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> dict[str, Any]:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


class Wire:
    """An Anthropic SDK client whose HTTP layer is a list of prepared replies."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.sdk = anthropic.Anthropic(
            api_key=FAKE_KEY, max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self._handle)))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        self.headers.append({k.lower(): v for k, v in request.headers.items() if k.lower() != "x-api-key"})
        return httpx2.Response(200, json=self.replies.pop(0))


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr(assistant, "ENV_FILE", tmp_path / "no_env_here" / ".env")


@pytest.fixture
def settings(tmp_path: Path) -> dict[str, Any]:
    notes = tmp_path / "knowledge.md"
    notes.write_text("# Project\nThe car is called Rover.\n", encoding="utf-8")
    return {"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "low",
            "max_tokens": 16000, "notes_file": str(notes),
            "people_dir": str(tmp_path / "people"), "web_search": True,
            "web_search_max_uses": 3, "history_turns": 10, "seen_within_seconds": 3.0}


def bot_on(settings: dict[str, Any], wire: Wire) -> assistant.Assistant:
    bot = assistant.Assistant(settings)
    bot._client._sdk = wire.sdk          # the real SDK client, minus the key lookup
    return bot


def test_wire_request_has_the_documented_shape(settings: dict[str, Any]) -> None:
    wire = Wire([reply_json([{"type": "text", "text": "Rover."}])])
    bot_on(settings, wire).ask("What is the car called?")
    body, headers = wire.bodies[0], wire.headers[0]
    assert body["model"] == "claude-sonnet-5-5"
    assert body["output_config"] == {"effort": "low"}
    assert body["thinking"]["type"] == "adaptive"
    assert body["fallbacks"] == "default"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"] == [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
    assert set(headers["anthropic-beta"].split(",")) == {
        "server-side-fallback-2026-07-01", "thinking-binding-controls-2026-08-01"}
    assert not {"temperature", "top_p", "top_k", "tool_choice"} & set(body)
    assert "budget_tokens" not in json.dumps(body)


def test_wire_replayed_assistant_turn_is_identical_to_what_the_api_returned(
        settings: dict[str, Any]) -> None:
    wire = Wire([reply_json(FULL_CONTENT), reply_json([{"type": "text", "text": "Two."}])])
    bot = bot_on(settings, wire)
    bot.ask("one")
    bot.ask("two")
    second = wire.bodies[1]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][1]["content"] == FULL_CONTENT
    assert second["system"] == wire.bodies[0]["system"]
    assert second["tools"] == wire.bodies[0]["tools"]


def test_wire_paused_turn_is_resumed_and_kept(settings: dict[str, Any]) -> None:
    paused = [{"type": "thinking", "thinking": "", "signature": "S2"},
              {"type": "server_tool_use", "id": "srvtoolu_2", "name": "web_search",
               "input": {"query": "y"}, "caller": {"type": "direct"}}]
    wire = Wire([reply_json(paused, "pause_turn"),
                 reply_json([{"type": "text", "text": "Final."}]),
                 reply_json([{"type": "text", "text": "Next."}])])
    bot = bot_on(settings, wire)
    assert bot.ask("web question").text == "Final."
    assert [m["role"] for m in wire.bodies[1]["messages"]] == ["user", "assistant"]
    assert wire.bodies[1]["messages"][1]["content"] == paused
    bot.ask("follow up")
    assert [m["role"] for m in wire.bodies[2]["messages"]] == ["user", "assistant", "assistant", "user"]


def test_wire_describe_sends_one_user_turn_with_the_picture(settings: dict[str, Any]) -> None:
    picture = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"fake jpeg body"
    answer = "NAME: Laptop\nABOUT: A portable computer.\nSPECS:\n- Weight: about 1.5 kg\n"
    wire = Wire([reply_json([{"type": "text", "text": "Hi."}]),
                 reply_json([{"type": "text", "text": answer}])])
    bot = bot_on(settings, wire)
    bot.ask("one")                                   # some chat history that must not go along
    info = bot.describe(picture, "laptop")
    body, headers = wire.bodies[1], wire.headers[1]
    assert body["messages"] == [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": base64.standard_b64encode(picture).decode("ascii")}},
        {"type": "text", "text": 'The object detector labelled this "laptop".'},
    ]}]
    assert body["system"] == [{"type": "text", "text": assistant.DESCRIBE_PROMPT,
                               "cache_control": {"type": "ephemeral"}}]
    assert body["tools"] == [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
    assert body["model"] == "claude-sonnet-5-5" and body["output_config"] == {"effort": "low"}
    assert "format" not in body["output_config"]     # no JSON mode: it cannot go with citations
    assert set(headers["anthropic-beta"].split(",")) == {
        "server-side-fallback-2026-07-01", "thinking-binding-controls-2026-08-01"}
    assert info.name == "Laptop" and info.specs == (("Weight", "about 1.5 kg"),)


def test_wire_follow_up_sends_the_whole_short_conversation(settings: dict[str, Any]) -> None:
    picture = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"fake jpeg body"
    info = assistant.ObjectInfo(name="Laptop", summary="A portable computer.",
                                specs=(("Weight", "about 1.5 kg"),))
    wire = Wire([reply_json([{"type": "text", "text": "Hi."}]),
                 reply_json([{"type": "text", "text": "Aluminium."}])])
    bot = bot_on(settings, wire)
    bot.ask("one")                                   # chat history that must not go along
    answer = bot.follow_up(picture, "laptop", info, [("How heavy?", "About 1.5 kg.")],
                           "What is it made of?")
    body = wire.bodies[1]
    assert body["messages"] == [
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(picture).decode("ascii")}},
            {"type": "text", "text": 'The object detector labelled this "laptop".'}]},
        {"role": "assistant",
         "content": "NAME: Laptop\nABOUT: A portable computer.\nSPECS:\n- Weight: about 1.5 kg"},
        {"role": "user", "content": "How heavy?"},
        {"role": "assistant", "content": "About 1.5 kg."},
        {"role": "user", "content": "What is it made of?"},
    ]
    assert body["system"] == [{"type": "text", "text": assistant.FOLLOW_UP_PROMPT,
                               "cache_control": {"type": "ephemeral"}}]
    assert body["tools"] == [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
    assert answer.text == "Aluminium." and answer.used_web is False


def test_wire_follow_up_with_a_marked_picture_sends_two_pictures(settings: dict[str, Any]) -> None:
    picture = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"plain cut-out"
    marked = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"cut-out with a magenta outline"
    info = assistant.ObjectInfo(name="Laptop", summary="A portable computer.", specs=())
    wire = Wire([reply_json([{"type": "text", "text": "That part is the hinge."}])])
    answer = bot_on(settings, wire).follow_up(picture, "laptop", info, [("How heavy?", "About 1.5 kg.")],
                                              "What is this part?", marked=marked)

    def image(data: bytes) -> dict[str, Any]:
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.standard_b64encode(data).decode("ascii")}}

    messages = wire.bodies[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"][0] == image(picture)
    assert messages[2] == {"role": "user", "content": "How heavy?"}
    assert messages[-1] == {"role": "user", "content": [
        image(marked),
        {"type": "text", "text": "The visitor brushed over part of the picture. In this second "
                                 "copy that part is outlined in magenta. The question is about "
                                 "the outlined part: What is this part?"}]}
    assert "magenta outline" in wire.bodies[0]["system"][0]["text"]
    assert answer.text == "That part is the hinge."


def test_wire_describe_scene_sends_the_picture_without_the_web_tool(settings: dict[str, Any]) -> None:
    picture = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"whole camera picture"
    wire = Wire([reply_json([{"type": "text", "text": "Hi."}]),
                 reply_json([{"type": "thinking", "thinking": "", "signature": "S3"},
                             {"type": "text", "text": "A desk with a laptop on it."}])])
    bot = bot_on(settings, wire)
    bot.ask("one")                                   # chat history that must not go along
    answer = bot.describe_scene(picture)
    body = wire.bodies[1]
    assert body["messages"] == [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": base64.standard_b64encode(picture).decode("ascii")}},
        {"type": "text", "text": "Describe the scene."},
    ]}]
    assert body["system"] == [{"type": "text", "text": assistant.SCENE_PROMPT,
                               "cache_control": {"type": "ephemeral"}}]
    assert "tools" not in body
    assert answer == assistant.Answer(text="A desk with a laptop on it.")


def test_bom_from_a_piped_powershell_line_is_stripped_under_cp1252() -> None:
    assert assistant.clean(b"\xef\xbb\xbf/quit".decode("cp1252")) == "/quit"


def test_large_max_tokens_setting_does_not_escape_as_a_raw_value_error(settings: dict[str, Any]) -> None:
    settings["max_tokens"] = 32000
    wire = Wire([reply_json([{"type": "text", "text": "ok"}])])
    try:
        bot_on(settings, wire).ask("hi")
    except assistant.AssistantError:
        return                       # a plain message the user can act on is fine
    # Also fine: the call went through (for example after the client got an explicit timeout).


def test_notes_saved_as_ansi_give_a_plain_error(tmp_path: Path) -> None:
    path = tmp_path / "ansi.md"
    path.write_bytes("caf\xe9".encode("cp1252"))
    try:
        assistant.load_notes(path)
    except assistant.AssistantError:
        return
    # Reading it successfully is fine too.
