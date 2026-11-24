"""Tests for car_vision/voice.py with a fake fetch and clock: no network, no real key."""

from __future__ import annotations

import io
import json
import logging
import struct
import threading
import urllib.error
import urllib.request
import wave
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from car_vision import voice as voice_module
from car_vision.config import ConfigError
from car_vision.voice import HostedVoice, VoiceError, wav_seconds

FAKE_KEY = "hf_FAKEtestKEYnot4real0123456789"
OTHER_FAKE_KEY = "hf_ANOTHERfakeKEY9876543210zyxw"
TODAY = date(2026, 11, 24)
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(range(256)) * 8
SPEECH_URL = "https://router.huggingface.co/deepinfra/v1/openai/audio/speech"
LISTEN_URL = "https://router.huggingface.co/deepinfra/v1/openai/audio/transcriptions"


# ---- fixtures and fakes --------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test sees the real .env or key, and no test can reach the network."""
    monkeypatch.setenv(voice_module.KEY_VARIABLE, "placeholder")   # teardown then removes what tests set
    monkeypatch.delenv(voice_module.KEY_VARIABLE)
    empty_folder = tmp_path / "no_env_here"
    empty_folder.mkdir()
    monkeypatch.setattr(voice_module, "ENV_FILE", empty_folder / ".env")

    def no_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", no_network)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)


@pytest.fixture
def with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(voice_module.KEY_VARIABLE, FAKE_KEY)


@pytest.fixture
def settings(tmp_path: Path) -> dict:
    return {
        "hosted": True,
        "provider": "deepinfra",
        "speech_model": "hexgrad/Kokoro-82M",
        "speech_voice": "af_heart",
        "listen_model": "openai/whisper-large-v3-turbo",
        "max_spoken_chars": 600,
        "max_listen_seconds": 20,
        "daily_budget_usd": 0.25,
        "speech_price_per_million_chars": 0.99,
        "listen_price_per_minute": 0.0002,
        "spend_file": str(tmp_path / "data" / "voice.json"),
    }


class FakeClock:
    """A clock whose local date the test sets."""

    def __init__(self, day: date = TODAY) -> None:
        self.day = day

    def __call__(self) -> datetime:
        return datetime(self.day.year, self.day.month, self.day.day, 12, 0, 0)


class FakeFetch:
    """Records each request and gives back the answers it was handed, in turn (the last one stays)."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls: list[dict] = []

    def __call__(self, url: str, headers: dict, body: bytes, timeout: float) -> Any:
        self.calls.append({"url": url, "headers": dict(headers), "body": body, "timeout": timeout})
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def sound(body: bytes = MP3, content_type: str = "audio/mpeg") -> tuple[int, str, bytes]:
    return 200, content_type, body


def words(text: Any = " Hello, this is the Robot Car. ") -> tuple[int, str, bytes]:
    return 200, "application/json", json.dumps({"text": text}).encode("utf-8")


def make_wav(seconds: float, rate: int = 16000, channels: int = 1) -> bytes:
    """A silent 16-bit WAV of this length, made with the wave module."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x00\x00" * channels * int(seconds * rate))
    return buffer.getvalue()


def make(settings: dict, fetch: Any, clock: Any = None) -> HostedVoice:
    return HostedVoice(settings, fetch=fetch, clock=clock or FakeClock())


def multipart_parts(body: bytes, boundary: str) -> dict[str, tuple[dict[str, str], bytes]]:
    """The multipart body parsed back: {name: (headers, value)}."""
    marker = b"--" + boundary.encode("ascii")
    closing = b"\r\n" + marker + b"--\r\n"
    assert body.startswith(marker + b"\r\n") and body.endswith(closing)
    parts = {}
    for chunk in body[len(marker) + 2:-len(closing)].split(b"\r\n" + marker + b"\r\n"):
        head, value = chunk.split(b"\r\n\r\n", 1)
        headers = dict(line.split(": ", 1) for line in head.decode("utf-8").split("\r\n"))
        name = headers["Content-Disposition"].split('name="')[1].split('"')[0]
        parts[name] = (headers, value)
    return parts


# ---- say: the request ------------------------------------------------------


def test_say_sends_one_request_as_measured(settings, with_key):
    fetch = FakeFetch(sound())
    assert make(settings, fetch).say("  Hello, this is the robot car.  ") == MP3
    (call,) = fetch.calls
    assert call["url"] == SPEECH_URL and call["timeout"] == 30
    assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert call["headers"]["Content-Type"] == "application/json"
    assert json.loads(call["body"]) == {"model": "hexgrad/Kokoro-82M", "input": "Hello, this is the robot car.",
                                        "voice": "af_heart", "response_format": "mp3"}


def test_say_takes_an_mp3_that_starts_with_a_frame_sync_byte(settings, with_key):
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 100
    assert make(settings, FakeFetch(sound(frame))).say("Hi.") == frame


@pytest.mark.parametrize("answer", [
    sound(content_type="application/json"),
    sound(content_type="text/html"),
    sound(b"<html>not sound</html>"),
    sound(b""),
    sound(b"RIFF....WAVE", "audio/wav"),
])
def test_say_refuses_an_answer_that_is_not_mp3_sound(settings, with_key, answer):
    with pytest.raises(VoiceError) as caught:
        make(settings, FakeFetch(answer)).say("Hi.")
    assert caught.value.kind == "failed"


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None, 42, b"bytes"])
def test_say_refuses_no_text(settings, with_key, text):
    fetch = FakeFetch(sound())
    with pytest.raises(VoiceError) as caught:
        make(settings, fetch).say(text)
    assert caught.value.kind == "bad_input" and fetch.calls == []


def test_say_refuses_text_longer_than_the_setting(settings, with_key):
    fetch = FakeFetch(sound())
    voice = make(settings, fetch)
    assert voice.say("a" * 600) == MP3                        # exactly the limit is fine
    with pytest.raises(VoiceError) as caught:
        voice.say("a" * 601)
    assert caught.value.kind == "bad_input" and "600" in str(caught.value)
    assert len(fetch.calls) == 1


@pytest.mark.parametrize("text", ["مرحبا، هذه سيارة الروبوت.", "The word سيارة is Arabic.", "ﷲ", "ﻻ"])
def test_say_refuses_arabic_text(settings, with_key, text):
    fetch = FakeFetch(sound())
    with pytest.raises(VoiceError) as caught:
        make(settings, fetch).say(text)
    assert caught.value.kind == "bad_input" and "English only" in str(caught.value)
    assert fetch.calls == []


def test_say_speaks_other_languages_that_are_not_arabic(settings, with_key):
    fetch = FakeFetch(sound())
    assert make(settings, fetch).say("Café, naïve, 3 €.") == MP3


# ---- hear: the request -------------------------------------------------------


def test_hear_sends_one_multipart_request_as_measured(settings, with_key):
    fetch = FakeFetch(words())
    wav = make_wav(2.0)
    assert make(settings, fetch).hear(wav, "en") == "Hello, this is the Robot Car."
    (call,) = fetch.calls
    assert call["url"] == LISTEN_URL and call["timeout"] == 30
    assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    content_type = call["headers"]["Content-Type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=", 1)[1]
    parts = multipart_parts(call["body"], boundary)
    assert set(parts) == {"file", "model", "language"}
    file_headers, file_value = parts["file"]
    assert file_headers["Content-Disposition"] == 'form-data; name="file"; filename="question.wav"'
    assert file_headers["Content-Type"] == "audio/wav"
    assert file_value == wav
    assert parts["model"][1] == b"openai/whisper-large-v3-turbo"
    assert parts["language"][1] == b"en"


def test_hear_sends_arabic_as_its_language(settings, with_key):
    fetch = FakeFetch(words("مرحبا"))
    assert make(settings, fetch).hear(make_wav(1.0), "ar") == "مرحبا"
    boundary = fetch.calls[0]["headers"]["Content-Type"].split("boundary=", 1)[1]
    assert multipart_parts(fetch.calls[0]["body"], boundary)["language"][1] == b"ar"


def test_each_request_has_its_own_boundary(settings, with_key):
    fetch = FakeFetch(words())
    voice = make(settings, fetch)
    voice.hear(make_wav(1.0))
    voice.hear(make_wav(1.0))
    first, second = (call["headers"]["Content-Type"] for call in fetch.calls)
    assert first != second


@pytest.mark.parametrize("language", ["fr", "EN", "", None, "english"])
def test_hear_refuses_another_language(settings, with_key, language):
    fetch = FakeFetch(words())
    with pytest.raises(VoiceError) as caught:
        make(settings, fetch).hear(make_wav(1.0), language)
    assert caught.value.kind == "bad_input" and fetch.calls == []


@pytest.mark.parametrize("text, heard", [
    ("  Hello.  ", "Hello."),
    ("", ""),
    ("   ", ""),
    (None, ""),
    ("a" * 900, "a" * 500),
])
def test_hear_gives_the_words_trimmed_and_cut(settings, with_key, text, heard):
    assert make(settings, FakeFetch(words(text))).hear(make_wav(1.0)) == heard


def test_hear_without_text_in_the_answer_gives_nothing(settings, with_key):
    fetch = FakeFetch((200, "application/json", b'{"task": "transcribe"}'))
    assert make(settings, fetch).hear(make_wav(1.0)) == ""


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b'{"text": 5}', b'{"text": ["a"]}', b"\xff\xfe"])
def test_hear_refuses_an_answer_it_cannot_read(settings, with_key, body):
    with pytest.raises(VoiceError) as caught:
        make(settings, FakeFetch((200, "application/json", body))).hear(make_wav(1.0))
    assert caught.value.kind == "failed"


# ---- the WAV checks ----------------------------------------------------------


def test_wav_seconds_counts_the_real_sound_bytes():
    assert wav_seconds(make_wav(2.5)) == 2.5
    assert len(make_wav(1.0)) == 44 + 32000                    # the wave module writes the page's layout


def forged(wav: bytes, at: int, fmt: str, value: int) -> bytes:
    """The WAV with one header field replaced."""
    size = struct.calcsize(fmt)
    return wav[:at] + struct.pack(fmt, value) + wav[at + size:]


def page_header(sound_bytes: int, channels: int = 1, rate: int = 16000, bits: int = 16, kind: int = 1,
                byte_rate: int | None = None, align: int | None = None) -> bytes:
    """A 44-byte WAV header with these fields (by default the page's own)."""
    align = channels * bits // 8 if align is None else align
    byte_rate = rate * align if byte_rate is None else byte_rate
    return (b"RIFF" + struct.pack("<I", 36 + sound_bytes) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, kind, channels, rate, byte_rate, align, bits)
            + b"data" + struct.pack("<I", sound_bytes))


def test_the_page_header_helper_matches_the_wave_module():
    assert make_wav(1.0)[:44] == page_header(32000)


def test_the_reviewers_forged_byte_rate_is_refused(settings, with_key):
    """2 MB of sound (62 seconds at the real rate) that claims to be 0.0005 seconds long."""
    sound = 2_000_000 - 44
    wav = page_header(sound, byte_rate=0xFFFFFFFF) + b"\x00" * sound
    fetch = FakeFetch(words())
    voice = make(settings, fetch)
    with pytest.raises(VoiceError) as caught:
        voice.hear(wav)
    assert caught.value.kind == "bad_input" and str(caught.value) == voice_module.NOT_A_WAV
    assert fetch.calls == [] and voice.uses_today == 0 and voice.spent_today == 0


def test_two_mb_of_real_sound_is_too_long_by_its_bytes(settings, with_key):
    sound = 2_000_000 - 44
    fetch = FakeFetch(words())
    with pytest.raises(VoiceError) as caught:
        make(settings, fetch).hear(page_header(sound) + b"\x00" * sound)
    assert caught.value.kind == "bad_input" and "20" in str(caught.value) and fetch.calls == []


SOUND = b"\x01\x02" * 16000                                     # one second of the page's sound


@pytest.mark.parametrize("wav", [
    page_header(len(SOUND) - 2) + SOUND,                        # a data size smaller than what follows
    page_header(len(SOUND) + 2) + SOUND,                        # ... and larger
    page_header(len(SOUND)) + SOUND + b"\x00\x00",              # bytes after the sound
    page_header(len(SOUND)) + SOUND + b"LIST\x04\x00\x00\x00abcd",
    page_header(len(SOUND), rate=8000) + SOUND,                 # 8 kHz
    page_header(len(SOUND), bits=8) + SOUND,                    # 8 bit
    page_header(len(SOUND), channels=2) + SOUND,                # two channels
    page_header(len(SOUND), kind=3, bits=32) + SOUND,           # float
    page_header(len(SOUND), kind=3) + SOUND,
    page_header(len(SOUND), align=4) + SOUND,                   # a wrong block align
    page_header(len(SOUND), byte_rate=64000) + SOUND,           # a wrong byte rate
    forged(page_header(len(SOUND)) + SOUND, 4, "<I", 0),        # a wrong RIFF size
    forged(page_header(len(SOUND)) + SOUND, 16, "<I", 18),      # a fmt chunk that is not 16 bytes
    # an extra chunk before data
    (b"RIFF" + struct.pack("<I", 36 + 12 + len(SOUND)) + b"WAVE" + page_header(0)[12:36]
     + b"LIST\x04\x00\x00\x00abcd" + b"data" + struct.pack("<I", len(SOUND)) + SOUND),
])
def test_only_the_pages_own_wav_is_accepted(settings, with_key, wav):
    fetch = FakeFetch(words())
    voice = make(settings, fetch)
    with pytest.raises(VoiceError) as caught:
        voice.hear(wav)
    assert caught.value.kind == "bad_input" and str(caught.value) == voice_module.NOT_A_WAV
    assert fetch.calls == [] and voice.uses_today == 0


def test_the_pages_own_wav_is_sent_exactly_as_it_came(settings, with_key):
    wav = page_header(len(SOUND)) + SOUND
    fetch = FakeFetch(words())
    make(settings, fetch).hear(wav)
    boundary = fetch.calls[0]["headers"]["Content-Type"].split("boundary=", 1)[1]
    assert multipart_parts(fetch.calls[0]["body"], boundary)["file"][1] == wav


def test_a_hundred_recordings_of_the_longest_length_cost_a_hundred_times_one(settings, with_key):
    settings["daily_budget_usd"] = 1.0
    voice = make(settings, FakeFetch(words()))
    longest = make_wav(20.0)
    voice.hear(longest)
    one = voice.spent_today
    assert one == pytest.approx(20 / 60 * 0.0002)
    for _ in range(99):
        voice.hear(longest)
    assert voice.uses_today == 100 and voice.spent_today == pytest.approx(100 * one)


@pytest.mark.parametrize("wav", [
    b"", b"RIFF", b"RIFF\x00\x00\x00\x00WAVX", b"ID3 an mp3", "RIFF....WAVE", None,
    b"RIFF\x24\x00\x00\x00WAVE",                                      # no fmt and no data
    b"RIFF\x24\x00\x00\x00WAVEdata\x04\x00\x00\x00\x00\x00\x00\x00",  # data without fmt
])
def test_hear_refuses_what_is_not_a_wav(settings, with_key, wav):
    fetch = FakeFetch(words())
    with pytest.raises(VoiceError) as caught:
        make(settings, fetch).hear(wav)
    assert caught.value.kind == "bad_input" and fetch.calls == []


def test_hear_refuses_an_empty_recording(settings, with_key):
    with pytest.raises(VoiceError) as caught:
        make(settings, FakeFetch(words())).hear(make_wav(0))
    assert caught.value.kind == "bad_input"


def test_hear_refuses_a_recording_longer_than_the_setting(settings, with_key):
    fetch = FakeFetch(words())
    voice = make(settings, fetch)
    assert voice.hear(make_wav(20.0)) == "Hello, this is the Robot Car."   # exactly the limit is fine
    with pytest.raises(VoiceError) as caught:
        voice.hear(make_wav(20.5))
    assert caught.value.kind == "bad_input" and "20" in str(caught.value)
    assert len(fetch.calls) == 1


def test_hear_refuses_a_recording_larger_than_2_mb(settings, with_key):
    wav = make_wav(1.0)
    too_large = wav + b"\x00" * (2_000_001 - len(wav))
    fetch = FakeFetch(words())
    with pytest.raises(VoiceError) as caught:
        make(settings, fetch).hear(too_large)
    assert caught.value.kind == "bad_input" and "2 MB" in str(caught.value) and fetch.calls == []


# ---- errors ------------------------------------------------------------------------


def test_no_key_is_a_plain_error_and_sends_nothing(settings):
    fetch = FakeFetch(sound())
    voice = make(settings, fetch)
    for call in (lambda: voice.say("Hi."), lambda: voice.hear(make_wav(1.0))):
        with pytest.raises(VoiceError) as caught:
            call()
        assert caught.value.kind == "no_key" and "HF_TOKEN" in str(caught.value) and ".env" in str(caught.value)
    assert fetch.calls == [] and voice.uses_today == 0


def test_a_key_with_spaces_is_refused(settings, monkeypatch):
    monkeypatch.setenv(voice_module.KEY_VARIABLE, "hf_two words")
    with pytest.raises(VoiceError) as caught:
        make(settings, FakeFetch(sound())).say("Hi.")
    assert caught.value.kind == "no_key" and "hf_two" not in str(caught.value)


@pytest.mark.parametrize("status, kind", [
    (401, "key_refused"), (403, "key_refused"), (402, "no_credit"), (404, "not_available"),
    (429, "busy"), (500, "busy"), (503, "busy"), (418, "failed"), (301, "failed"),
])
def test_each_http_status_has_its_kind(settings, with_key, status, kind):
    answer = (status, "application/json", json.dumps({"error": f"bad {FAKE_KEY}"}).encode())
    voice = make(settings, FakeFetch(answer))
    for call in (lambda: voice.say("Hi."), lambda: voice.hear(make_wav(1.0))):
        with pytest.raises(VoiceError) as caught:
            call()
        assert caught.value.kind == kind
        assert FAKE_KEY not in str(caught.value)


def test_not_available_names_the_model(settings, with_key):
    voice = make(settings, FakeFetch((404, "application/json", b"{}")))
    with pytest.raises(VoiceError) as caught:
        voice.say("Hi.")
    assert "hexgrad/Kokoro-82M" in str(caught.value)
    with pytest.raises(VoiceError) as caught:
        voice.hear(make_wav(1.0))
    assert "openai/whisper-large-v3-turbo" in str(caught.value)


@pytest.mark.parametrize("error", [
    TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out")),
    urllib.error.URLError("no route to host"), ConnectionResetError("reset"), OSError("down"),
    ValueError("odd"),
])
def test_a_timeout_or_no_network_is_a_plain_failure(settings, with_key, error):
    voice = make(settings, FakeFetch(error))
    with pytest.raises(VoiceError) as caught:
        voice.say("Hi.")
    assert caught.value.kind == "failed"
    assert str(caught.value) in (voice_module.NO_ANSWER_IN_TIME, voice_module.NO_NETWORK)


def test_a_timeout_says_so(settings, with_key):
    with pytest.raises(VoiceError) as caught:
        make(settings, FakeFetch(TimeoutError())).hear(make_wav(1.0))
    assert str(caught.value) == voice_module.NO_ANSWER_IN_TIME


@pytest.mark.parametrize("answer", [None, (200, b"x"), (200, "audio/mpeg", "text"), ("200", "audio/mpeg", MP3),
                                    (True, "audio/mpeg", MP3), (200, None, MP3)])
def test_a_fetch_that_answers_oddly_is_a_plain_failure(settings, with_key, answer):
    with pytest.raises(VoiceError) as caught:
        make(settings, FakeFetch(answer)).say("Hi.")
    assert caught.value.kind == "failed"


def test_a_refused_key_is_read_again_for_the_next_request(settings, with_key, monkeypatch):
    fetch = FakeFetch((401, "application/json", b"{}"), sound())
    voice = make(settings, fetch)
    with pytest.raises(VoiceError):
        voice.say("Hi.")
    monkeypatch.setenv(voice_module.KEY_VARIABLE, OTHER_FAKE_KEY)
    assert voice.say("Hi.") == MP3
    assert fetch.calls[1]["headers"]["Authorization"] == f"Bearer {OTHER_FAKE_KEY}"


def test_the_key_is_never_in_an_error_or_the_log(settings, with_key, caplog):
    caplog.set_level(logging.DEBUG)
    echoed = json.dumps({"error": f"Invalid key {FAKE_KEY}", "text": f"say {FAKE_KEY}"}).encode()
    answers = [(401, "application/json", echoed), (500, "text/plain", echoed), (200, "application/json", echoed),
               urllib.error.URLError(f"failed with {FAKE_KEY}"), (200, "text/plain", echoed)]
    messages = []
    for answer in answers:
        voice = make(settings, FakeFetch(answer))
        for call in (lambda: voice.say("Hi."), lambda: voice.hear(make_wav(1.0))):
            try:
                messages.append(call())
            except VoiceError as error:
                messages.append(str(error))
    assert messages and not [m for m in messages if isinstance(m, str) and FAKE_KEY in m]
    assert FAKE_KEY not in caplog.text and "hf_" not in caplog.text


def test_nothing_about_the_text_the_words_or_the_sound_is_logged(settings, with_key, caplog):
    caplog.set_level(logging.DEBUG)
    voice = make(settings, FakeFetch(sound(), words("the secret words heard")))
    voice.say("a private answer text")
    voice.hear(make_wav(1.0))
    assert "private answer" not in caplog.text and "secret words" not in caplog.text
    assert "ID3" not in caplog.text


# ---- the cost estimate and the budget ---------------------------------------------


def test_the_estimate_counts_characters_spoken_and_seconds_heard(settings, with_key):
    voice = make(settings, FakeFetch(sound(), words()))
    voice.say("a" * 300)
    assert voice.uses_today == 1 and voice.spent_today == pytest.approx(300 * 0.99 / 1_000_000)
    voice.hear(make_wav(6.0))
    assert voice.uses_today == 2
    assert voice.spent_today == pytest.approx(300 * 0.99 / 1_000_000 + 6 / 60 * 0.0002)


def test_the_estimate_counts_the_trimmed_text(settings, with_key):
    voice = make(settings, FakeFetch(sound()))
    voice.say("   Hi.   ")
    assert voice.spent_today == pytest.approx(3 * 0.99 / 1_000_000)


def test_every_request_counts_before_it_is_sent_also_one_that_fails(settings, with_key):
    seen = []

    def fetch(url, headers, body, timeout):
        seen.append(voice.uses_today)                   # already counted while it is being sent
        raise urllib.error.URLError("down")

    voice = make(settings, fetch)
    with pytest.raises(VoiceError):
        voice.say("Hello.")
    assert seen == [1] and voice.uses_today == 1 and voice.spent_today > 0


def test_a_refused_input_costs_nothing(settings, with_key):
    voice = make(settings, FakeFetch(sound()))
    for call in (lambda: voice.say(""), lambda: voice.say("مرحبا"), lambda: voice.hear(b"nope"),
                 lambda: voice.hear(make_wav(30.0))):
        with pytest.raises(VoiceError):
            call()
    assert voice.uses_today == 0 and voice.spent_today == 0


def test_the_budget_stops_both_calls_before_sending(settings, with_key):
    settings["daily_budget_usd"] = 0.0001
    fetch = FakeFetch(sound(), words())
    voice = make(settings, fetch)
    voice.say("a" * 102)                                # just over 0.0001 dollars: the budget is reached
    assert voice.spent_today >= voice.budget
    for call in (lambda: voice.say("Hi."), lambda: voice.hear(make_wav(1.0))):
        with pytest.raises(VoiceError) as caught:
            call()
        assert caught.value.kind == "budget" and "budget" in str(caught.value)
    assert len(fetch.calls) == 1 and voice.uses_today == 1


def test_a_budget_of_zero_sends_nothing(settings, with_key):
    settings["daily_budget_usd"] = 0
    fetch = FakeFetch(sound())
    with pytest.raises(VoiceError) as caught:
        make(settings, fetch).say("Hi.")
    assert caught.value.kind == "budget" and fetch.calls == []


def test_the_budget_starts_again_on_a_new_day(settings, with_key):
    settings["daily_budget_usd"] = 0.000001
    clock = FakeClock()
    fetch = FakeFetch(sound())
    voice = make(settings, fetch, clock)
    voice.say("a" * 10)
    with pytest.raises(VoiceError):
        voice.say("Hi.")
    clock.day = TODAY + timedelta(days=1)
    assert voice.uses_today == 0 and voice.spent_today == 0
    assert voice.say("Hi.") == MP3


# ---- the spend file ------------------------------------------------------------------


def spend_path(settings: dict) -> Path:
    return Path(settings["spend_file"])


def test_the_spend_file_is_written_and_read_after_a_restart(settings, with_key):
    make(settings, FakeFetch(sound())).say("a" * 100)
    saved = json.loads(spend_path(settings).read_text(encoding="utf-8"))
    assert saved == {"day": "2026-11-24", "uses": 1, "spent_usd": pytest.approx(100 * 0.99 / 1_000_000)}
    again = make(settings, FakeFetch(sound()))
    assert again.uses_today == 1 and again.spent_today == pytest.approx(100 * 0.99 / 1_000_000)
    assert not spend_path(settings).with_name("voice.json.tmp").exists()   # written through a temporary file


def test_a_spend_file_from_an_earlier_day_starts_from_zero(settings, with_key):
    spend_path(settings).parent.mkdir(parents=True)
    spend_path(settings).write_text('{"day": "2026-11-23", "uses": 40, "spent_usd": 0.2}', encoding="utf-8")
    voice = make(settings, FakeFetch(sound()))
    assert voice.uses_today == 0 and voice.spent_today == 0


def test_a_spend_file_dated_tomorrow_keeps_its_totals(settings, with_key):
    spend_path(settings).parent.mkdir(parents=True)
    spend_path(settings).write_text('{"day": "2026-11-25", "uses": 3, "spent_usd": 0.01}', encoding="utf-8")
    voice = make(settings, FakeFetch(sound()))
    assert voice.uses_today == 3 and voice.spent_today == 0.01


@pytest.mark.parametrize("day", ["2026-11-26", "2027-01-01", "9999-12-31"])
def test_a_spend_file_dated_further_ahead_is_taken_as_broken(settings, with_key, caplog, day):
    spend_path(settings).parent.mkdir(parents=True)
    spend_path(settings).write_text(json.dumps({"day": day, "uses": 99, "spent_usd": 5.0}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="car_vision.voice"):
        voice = make(settings, FakeFetch(sound()))
    assert voice.uses_today == 0 and voice.spent_today == 0
    assert len([r for r in caplog.records if r.name == "car_vision.voice"]) == 1
    assert voice.say("Hi.") == MP3                              # not blocked


@pytest.mark.parametrize("text", ["{broken", "[]", '{"day": "2026-11-24"}', '{"day": 5, "uses": 1, "spent_usd": 0}',
                                  '{"day": "2026-11-24", "uses": -1, "spent_usd": 0}',
                                  '{"day": "2026-11-24", "uses": true, "spent_usd": 0}',
                                  '{"day": "2026-11-24", "uses": 1, "spent_usd": -0.5}',
                                  '{"day": "2026-11-24", "uses": 1, "spent_usd": "0.1"}',
                                  '{"day": "2026-11-24", "uses": 1, "spent_usd": NaN}',
                                  '{"day": "2026-11-24", "looks": 1, "spent_usd": 0.1}',
                                  " " * 5000])
def test_a_broken_spend_file_starts_from_zero_with_one_log_line(settings, with_key, caplog, text):
    spend_path(settings).parent.mkdir(parents=True)
    spend_path(settings).write_text(text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="car_vision.voice"):
        voice = make(settings, FakeFetch(sound()))
    assert voice.uses_today == 0 and voice.spent_today == 0
    assert len([r for r in caplog.records if r.name == "car_vision.voice"]) == 1
    voice.say("Hi.")
    assert json.loads(spend_path(settings).read_text(encoding="utf-8"))["uses"] == 1


def test_a_spend_file_that_cannot_be_written_is_logged_once(settings, with_key, caplog, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(voice_module, "save_spend", refuse)
    voice = make(settings, FakeFetch(sound()))
    with caplog.at_level(logging.WARNING, logger="car_vision.voice"):
        voice.say("Hi.")
        voice.say("Hi.")
    assert voice.uses_today == 2
    assert len([r for r in caplog.records if "spend file" in r.getMessage()]) == 1


def test_the_key_is_never_written_to_the_spend_file(settings, with_key):
    make(settings, FakeFetch(sound())).say("Hi.")
    assert FAKE_KEY not in spend_path(settings).read_text(encoding="utf-8")


# ---- settings ----------------------------------------------------------------------------


def test_good_settings_are_read(settings):
    voice = make(settings, FakeFetch(sound()))
    assert voice.budget == 0.25 and voice.max_chars == 600 and voice.max_seconds == 20
    assert voice.uses_today == 0 and voice.spent_today == 0


@pytest.mark.parametrize("key, value", [
    ("provider", "Deep Infra"), ("provider", ""), ("provider", "deepinfra/../x"), ("provider", 5),
    ("speech_model", "hexgrad/Kokoro 82M"), ("speech_model", ""), ("listen_model", "../whisper"),
    ("speech_voice", "af heart"), ("speech_voice", ""), ("speech_voice", None),
    ("max_spoken_chars", 0), ("max_spoken_chars", "600"), ("max_spoken_chars", 600.0), ("max_spoken_chars", True),
    ("max_spoken_chars", 6000), ("max_listen_seconds", 0), ("max_listen_seconds", 61), ("max_listen_seconds", 2.5),
    ("daily_budget_usd", -1), ("daily_budget_usd", float("nan")), ("daily_budget_usd", float("inf")),
    ("daily_budget_usd", "0.25"), ("daily_budget_usd", 5000), ("speech_price_per_million_chars", -0.1),
    ("listen_price_per_minute", True), ("spend_file", None), ("spend_file", 5),
])
def test_wrong_settings_are_refused(settings, key, value):
    settings[key] = value
    with pytest.raises(ConfigError):
        make(settings, FakeFetch(sound()))


@pytest.mark.parametrize("key", ["provider", "speech_model", "speech_voice", "listen_model", "max_spoken_chars",
                                 "max_listen_seconds", "daily_budget_usd", "speech_price_per_million_chars",
                                 "listen_price_per_minute", "spend_file"])
def test_a_missing_setting_is_refused(settings, key):
    del settings[key]
    with pytest.raises(ConfigError):
        make(settings, FakeFetch(sound()))


def test_the_provider_goes_into_the_address(settings, with_key):
    settings["provider"] = "hf-inference"
    fetch = FakeFetch(sound())
    make(settings, fetch).say("Hi.")
    assert fetch.calls[0]["url"] == "https://router.huggingface.co/hf-inference/v1/openai/audio/speech"


# ---- one request of each kind at a time ---------------------------------------------------


class BlockingFetch:
    """Holds every request until the test lets it go."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.inside = 0
        self.most_inside = 0
        self.lock = threading.Lock()

    def __call__(self, url, headers, body, timeout):
        with self.lock:
            self.calls += 1
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
        self.started.set()
        self.release.wait(5)
        with self.lock:
            self.inside -= 1
        return self.answer(url) if callable(self.answer) else self.answer


def test_a_second_say_waits_for_the_first(settings, with_key):
    fetch = BlockingFetch(sound())
    voice = make(settings, fetch)
    results = []
    threads = [threading.Thread(target=lambda: results.append(voice.say("Hi."))) for _ in range(3)]
    for thread in threads:
        thread.start()
    assert fetch.started.wait(5)
    fetch.release.set()
    for thread in threads:
        thread.join(5)
    assert results == [MP3, MP3, MP3] and fetch.calls == 3 and fetch.most_inside == 1


def test_a_say_that_waits_too_long_is_busy(settings, with_key, monkeypatch):
    monkeypatch.setattr(voice_module, "WAIT_TURN_SECONDS", 0.1)
    fetch = BlockingFetch(sound())
    voice = make(settings, fetch)
    first = threading.Thread(target=lambda: voice.say("Hi."))
    first.start()
    assert fetch.started.wait(5)
    with pytest.raises(VoiceError) as caught:
        voice.say("Hi.")
    assert caught.value.kind == "busy"
    fetch.release.set()
    first.join(5)
    assert fetch.calls == 1 and voice.uses_today == 1     # the one that waited was not counted


def test_a_hear_that_waits_too_long_is_busy(settings, with_key, monkeypatch):
    monkeypatch.setattr(voice_module, "WAIT_TURN_SECONDS", 0.1)
    fetch = BlockingFetch(words())
    voice = make(settings, fetch)
    first = threading.Thread(target=lambda: voice.hear(make_wav(1.0)))
    first.start()
    assert fetch.started.wait(5)
    with pytest.raises(VoiceError) as caught:
        voice.hear(make_wav(1.0))
    assert caught.value.kind == "busy"
    fetch.release.set()
    first.join(5)


def test_speaking_and_hearing_have_a_lock_each(settings, with_key):
    fetch = BlockingFetch(lambda url: sound() if url == SPEECH_URL else words())
    voice = make(settings, fetch)
    results = []
    threads = [threading.Thread(target=lambda: results.append(voice.say("Hi."))),
               threading.Thread(target=lambda: results.append(voice.hear(make_wav(1.0))))]
    for thread in threads:
        thread.start()
    for _ in range(100):                                 # both are inside fetch at the same time
        if fetch.most_inside == 2:
            break
        threading.Event().wait(0.02)
    fetch.release.set()
    for thread in threads:
        thread.join(5)
    assert fetch.most_inside == 2 and sorted(map(str, results)) == sorted([str(MP3), "Hello, this is the Robot Car."])


# ---- the real fetch ---------------------------------------------------------------------


@pytest.mark.parametrize("url", ["http://router.huggingface.co/deepinfra/v1/openai/audio/speech",
                                 "https://example.com/v1/openai/audio/speech",
                                 "https://router.huggingface.co.evil.example/x"])
def test_the_real_fetch_talks_only_to_the_router(url):
    with pytest.raises(ValueError):
        voice_module.real_fetch(url, {}, b"", 1)


def test_the_real_fetch_follows_no_redirect():
    handler = voice_module.RefuseRedirects()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://elsewhere.example/") is None


def test_voice_uses_only_the_standard_library_and_dotenv():
    source = Path(voice_module.__file__).read_text(encoding="utf-8")
    for library in ("requests", "httpx", "flask", "anthropic", "ultralytics", "insightface", "aiohttp"):
        assert f"import {library}" not in source and f"from {library}" not in source, library
