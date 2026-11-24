"""M23 Parts B and C: the hosted voice's routes with a fake voice, the one exception in the
same-origin rule (a WAV recording to /api/voice/hear), main() building the voice, and the
page's voice code checked by the shape of the script (there is no JavaScript engine here).
"""

from __future__ import annotations

import io

import pytest

from car_vision import app as app_module
from car_vision.app import (HEAR_PATH, NOT_JSON, OTHER_WEBSITE, SAY_SHAPE, build_hosted_voice, create_app,
                            refusal)
from car_vision.voice import OFF, HostedVoice, VoiceError
from test_app import TEMPLATE, between, fake_main, ids_around, rendered_page  # noqa: F401 (fake_main: a fixture)
from test_app_discover import FakeAssistant, FakePipeline, changing_routes

MP3 = b"ID3\x04\x00\x00\x00" + b"\x00" * 64
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 40
KINDS = ["off", "no_key", "key_refused", "no_credit", "not_available", "busy", "budget", "bad_input", "failed"]


class FakeVoice:
    """Records what it was asked; answers with the sound, the words or the error it was given."""

    def __init__(self, error: VoiceError | None = None, words: str = "Who built the car?") -> None:
        self.error = error
        self.words = words
        self.said: list[str] = []
        self.heard: list[tuple[bytes, str]] = []
        self.uses_today, self.spent_today, self.budget = 12, 0.0031, 0.25
        self.max_chars, self.max_seconds = 600, 20

    def say(self, text: str) -> bytes:
        self.said.append(text)
        if self.error is not None:
            raise self.error
        return MP3

    def hear(self, wav: bytes, language: str = "en") -> str:
        self.heard.append((wav, language))
        if self.error is not None:
            raise self.error
        return self.words


def client_for(voice):
    app = create_app(FakePipeline(), FakeAssistant(), voice=voice)
    app.testing = True
    return app.test_client()


def post_wav(client, body=WAV, address=HEAR_PATH + "?language=en", **kwargs):
    return client.post(address, data=body, content_type="audio/wav", **kwargs)


# ---- GET /api/voice ----------------------------------------------------------------------


def test_voice_status_says_what_the_hosted_voice_can_do():
    reply = client_for(FakeVoice()).get("/api/voice")
    assert reply.status_code == 200 and reply.headers["Cache-Control"] == "no-store"
    assert reply.get_json() == {"hosted": True, "uses": 12, "spent_usd": 0.0031, "budget_usd": 0.25,
                                "max_chars": 600, "max_seconds": 20}


def test_voice_status_without_a_hosted_voice():
    reply = client_for(None).get("/api/voice")
    assert reply.status_code == 200 and reply.get_json() == {"hosted": False}


# ---- POST /api/voice/say ------------------------------------------------------------------


def test_say_answers_with_the_sound():
    voice = FakeVoice()
    reply = client_for(voice).post("/api/voice/say", json={"text": "Hello, this is the robot car."})
    assert reply.status_code == 200 and reply.data == MP3
    assert reply.mimetype == "audio/mpeg" and reply.headers["Cache-Control"] == "no-store"
    assert voice.said == ["Hello, this is the robot car."]


@pytest.mark.parametrize("kind", KINDS)
def test_say_answers_each_voice_error_as_200_json(kind):
    voice = FakeVoice(VoiceError("A plain message.", kind))
    reply = client_for(voice).post("/api/voice/say", json={"text": "Hi."})
    assert reply.status_code == 200 and reply.get_json() == {"error": "A plain message.", "kind": kind}


@pytest.mark.parametrize("body", [{}, [], "Hi.", {"text": 5}, {"text": None}, {"words": "Hi."}, {"text": ["Hi."]}])
def test_say_with_the_wrong_shape_is_400(body):
    voice = FakeVoice()
    reply = client_for(voice).post("/api/voice/say", json=body)
    assert reply.status_code == 400 and reply.get_json() == {"error": SAY_SHAPE}
    assert voice.said == []


def test_say_without_a_hosted_voice_is_off():
    reply = client_for(None).post("/api/voice/say", json={"text": "Hi."})
    assert reply.status_code == 200 and reply.get_json() == {"error": OFF, "kind": "off"}


@pytest.mark.parametrize("content_type", ["text/plain", "audio/wav", "audio/mpeg", "application/x-www-form-urlencoded"])
def test_say_must_be_json(content_type):
    voice = FakeVoice()
    reply = client_for(voice).post("/api/voice/say", data='{"text": "Hi."}', content_type=content_type)
    assert reply.status_code == 415 and reply.get_json() == {"error": NOT_JSON} and voice.said == []


# ---- POST /api/voice/hear -----------------------------------------------------------------


@pytest.mark.parametrize("address, language", [(HEAR_PATH + "?language=en", "en"), (HEAR_PATH + "?language=ar", "ar"),
                                               (HEAR_PATH, "en")])
def test_hear_answers_with_the_words(address, language):
    voice = FakeVoice()
    reply = post_wav(client_for(voice), address=address)
    assert reply.status_code == 200 and reply.get_json() == {"text": "Who built the car?"}
    assert voice.heard == [(WAV, language)]


def test_hear_gives_empty_words_as_they_are():
    reply = post_wav(client_for(FakeVoice(words="")))
    assert reply.get_json() == {"text": ""}


@pytest.mark.parametrize("kind", KINDS)
def test_hear_answers_each_voice_error_as_200_json(kind):
    reply = post_wav(client_for(FakeVoice(VoiceError("A plain message.", kind))))
    assert reply.status_code == 200 and reply.get_json() == {"error": "A plain message.", "kind": kind}


@pytest.mark.parametrize("language", ["fr", "EN", "", "en,ar"])
def test_hear_in_another_language_is_400(language):
    voice = FakeVoice()
    reply = post_wav(client_for(voice), address=f"{HEAR_PATH}?language={language}")
    assert reply.status_code == 400 and voice.heard == []


def test_hear_without_a_hosted_voice_is_off():
    reply = post_wav(client_for(None))
    assert reply.status_code == 200 and reply.get_json() == {"error": OFF, "kind": "off"}


class UnreadableBody(io.RawIOBase):
    """A request body that fails the test if anything reads it."""

    def readable(self):
        return True

    def readinto(self, buffer):
        raise AssertionError("the body was read")


def post_unreadable(client, content_length):
    """A WAV post whose body cannot be read, with this Content-Length (None: sent in chunks, no size)."""
    environ = {"wsgi.input": UnreadableBody(), "CONTENT_LENGTH": content_length or "0"}
    if content_length is None:
        environ["HTTP_TRANSFER_ENCODING"] = "chunked"
    return client.post(HEAR_PATH, data=b"", content_type="audio/wav", environ_overrides=environ)


def test_a_recording_above_2_mb_is_refused_without_reading_it():
    voice = FakeVoice()
    reply = post_unreadable(client_for(voice), "2000001")
    assert reply.status_code == 413 and "2 MB" in reply.get_json()["error"]
    assert voice.heard == []
    reply = post_unreadable(client_for(voice), "900000000")
    assert reply.status_code == 413 and voice.heard == []


def test_a_recording_of_exactly_2_mb_is_read():
    voice = FakeVoice()
    body = WAV + b"\x00" * (2_000_000 - len(WAV))
    assert post_wav(client_for(voice), body=body).status_code == 200
    assert voice.heard[0][0] == body


def test_a_recording_without_its_size_is_refused():
    voice = FakeVoice()
    reply = post_unreadable(client_for(voice), None)
    assert reply.status_code == 411 and voice.heard == []


@pytest.mark.parametrize("content_type", ["text/plain", "application/octet-stream", "audio/mpeg", "audio/webm",
                                          "multipart/form-data; boundary=x", "application/x-www-form-urlencoded"])
def test_hear_accepts_only_a_wav(content_type):
    voice = FakeVoice()
    reply = client_for(voice).post(HEAR_PATH, data=WAV, content_type=content_type)
    assert reply.status_code == 415 and reply.get_json() == {"error": NOT_JSON} and voice.heard == []


def test_hear_also_takes_json_like_every_other_route_but_reads_its_body_as_the_recording():
    voice = FakeVoice(VoiceError("The recording must be a WAV file.", "bad_input"))
    reply = client_for(voice).post(HEAR_PATH, json={})
    assert reply.status_code == 200 and reply.get_json()["kind"] == "bad_input"


@pytest.mark.parametrize("origin", ["http://evil.example", "null", "http://127.0.0.1", "https://localhost"])
def test_a_recording_from_another_website_is_refused(origin):
    voice = FakeVoice()
    reply = post_wav(client_for(voice), headers={"Origin": origin})
    assert reply.status_code == 403 and reply.get_json() == {"error": OTHER_WEBSITE} and voice.heard == []


def test_a_recording_from_this_page_goes_through():
    voice = FakeVoice()
    assert post_wav(client_for(voice), headers={"Origin": "http://localhost"}).status_code == 200
    assert len(voice.heard) == 1


# ---- refusal(): exactly one exception -------------------------------------------------------


HOST = "http://127.0.0.1:5000/"


def test_refusal_lets_only_a_wav_to_hear_through():
    assert refusal("POST", HEAR_PATH, None, "audio/wav", HOST) is None
    assert refusal("POST", HEAR_PATH, "http://127.0.0.1:5000", "audio/wav", HOST) is None
    assert refusal("POST", HEAR_PATH, "http://evil.example", "audio/wav", HOST) == (OTHER_WEBSITE, 403)
    for method, path, mimetype in [("PUT", HEAR_PATH, "audio/wav"), ("DELETE", HEAR_PATH, "audio/wav"),
                                   ("POST", HEAR_PATH + "/", "audio/wav"), ("POST", "/api/voice/say", "audio/wav"),
                                   ("POST", "/api/ask", "audio/wav"), ("POST", HEAR_PATH, "audio/wave"),
                                   ("POST", HEAR_PATH, "audio/x-wav"), ("POST", HEAR_PATH, "text/plain")]:
        assert refusal(method, path, None, mimetype, HOST) == (NOT_JSON, 415), (method, path, mimetype)


def test_every_other_change_sent_as_a_wav_is_still_415():
    app = create_app(FakePipeline(), FakeAssistant(), voice=FakeVoice())
    client = app.test_client()
    routes = [(method, address) for method, address in changing_routes(app) if (method, address) != ("POST", HEAR_PATH)]
    assert ("POST", "/api/voice/say") in routes and len(routes) > 15
    for method, address in routes:
        reply = client.open(address, method=method, data=WAV, content_type="audio/wav")
        assert reply.status_code == 415, (method, address, reply.status_code)


# ---- main() builds the hosted voice ------------------------------------------------------------


GOOD_VOICE = {"hosted": True, "provider": "deepinfra", "speech_model": "hexgrad/Kokoro-82M",
              "speech_voice": "af_heart", "listen_model": "openai/whisper-large-v3-turbo",
              "max_spoken_chars": 600, "max_listen_seconds": 20, "daily_budget_usd": 0.25,
              "speech_price_per_million_chars": 0.99, "listen_price_per_minute": 0.0002,
              "spend_file": "data/voice.json"}


def test_build_hosted_voice_only_when_it_is_switched_on(tmp_path):
    settings = {**GOOD_VOICE, "spend_file": str(tmp_path / "voice.json")}
    assert isinstance(build_hosted_voice({"voice": settings}), HostedVoice)
    assert build_hosted_voice({}) is None
    assert build_hosted_voice({"voice": None}) is None
    assert build_hosted_voice({"voice": {**settings, "hosted": False}}) is None
    assert build_hosted_voice({"voice": {**settings, "hosted": "yes"}}) is None


def test_a_wrong_voice_setting_prints_one_line_and_the_program_runs_without(capsys, tmp_path):
    settings = {**GOOD_VOICE, "spend_file": str(tmp_path / "voice.json"), "max_spoken_chars": "many"}
    assert build_hosted_voice({"voice": settings}) is None
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 1 and printed[0].startswith("The hosted voice is off:")


def served_voice_status(fake_main) -> dict:
    (app, _kwargs) = fake_main["calls"]["run"][-1]
    return app.test_client().get("/api/voice").get_json()


def test_main_gives_the_page_the_hosted_voice(fake_main, monkeypatch, tmp_path):
    assert app_module.main([]) == 0                            # TEST_CONFIG has no voice section
    assert served_voice_status(fake_main) == {"hosted": False}
    config = {**app_module.load_config(), "voice": {**GOOD_VOICE, "spend_file": str(tmp_path / "voice.json")}}
    monkeypatch.setattr(app_module, "load_config", lambda: config)
    assert app_module.main([]) == 0
    status = served_voice_status(fake_main)
    assert status["hosted"] is True and status["max_chars"] == 600 and status["max_seconds"] == 20


def test_main_runs_without_the_hosted_voice_when_a_setting_is_wrong(fake_main, monkeypatch, tmp_path, capsys):
    voice = {**GOOD_VOICE, "spend_file": str(tmp_path / "voice.json"), "provider": "Deep Infra"}
    config = {**app_module.load_config(), "voice": voice}
    monkeypatch.setattr(app_module, "load_config", lambda: config)
    assert app_module.main([]) == 0
    assert served_voice_status(fake_main) == {"hosted": False}
    assert len([line for line in capsys.readouterr().out.splitlines() if "hosted voice is off" in line]) == 1


# ---- the page: the Voice box --------------------------------------------------------------------


def page_text():
    return TEMPLATE.read_text(encoding="utf-8")


def test_the_voice_box_has_the_two_hosted_switches_above_the_voices():
    page = rendered_page()
    around = ids_around(page)
    for inside in ("hostedRow", "hostedSpeakSwitch", "hostedListenSwitch", "hostedNote"):
        assert "voiceBox" in around[inside], inside
    assert '<div class="hosted-voice" id="hostedRow" hidden>' in page
    assert '<input type="checkbox" id="hostedSpeakSwitch" role="switch"' in page
    assert '<input type="checkbox" id="hostedListenSwitch" role="switch"' in page
    assert "Hosted voice: the same in every browser" in page
    assert "Hosted listening: works in every browser" in page
    assert '<label class="switch" id="hostedListenRow" hidden>' in page
    assert page.index('id="hostedRow"') < page.index('for="voiceChoice"')     # above the list of voices


def test_the_hosted_switches_follow_the_server_and_the_browser():
    page = page_text()
    assert 'let hostedSpeakWanted = readStored(HOSTED_SPEAK_KEY) === "on";' in page     # off until switched on
    assert 'store(HOSTED_SPEAK_KEY, hostedSpeakWanted ? "on" : "off");' in page
    assert "store(HOSTED_LISTEN_KEY, hostedListenChoice);" in page
    # Hosted listening: on by default only where the browser has no recognition of its own.
    wanted = between(page, "function hostedListenWanted()", "\n}\n")
    assert 'hostedListenChoice === null ? Recognition === null : hostedListenChoice === "on"' in wanted
    record = between(page, "function canRecord()", "\n}\n")
    assert "navigator.mediaDevices.getUserMedia" in record and 'typeof window.MediaRecorder === "function"' in record
    draw = between(page, "function drawHosted()", "\n}\n")
    assert "hostedRow.hidden = hosted === null;" in draw and "hostedListenRow.hidden = !canRecord();" in draw
    assert 'readHosted(data)' in between(page, "async function loadHosted()", "\n}\n")
    assert 'data.hosted !== true' in between(page, "function readHosted(", "\n}\n")
    assert "loadHosted();" in between(page, "function setUpVoice()", "setUpVoice();\n")


def test_the_small_note_says_the_uses_and_cost_or_why_it_paused():
    page = page_text()
    words = between(page, "function usesWords(", "\n}\n")
    assert '" uses today"' in words and '"1 use today"' in words and 'aboutDollars(info.spent)' in words
    assert "setText(hostedNote, hostedStopped || usesWords(hosted));" in page
    assert '.hosted-note {' in page and "font-size: 0.8rem" in page


# ---- the page: speaking with either voice through one set of functions --------------------------


def test_both_voices_go_through_the_same_speaking_functions():
    page = page_text()
    speak = between(page, "function speak(", "function startTalking(")
    assert "hosted: speaksHosted() ? hostedParts() : null" in speak          # chosen once per answer
    start = between(page, "function startTalking(", "function speakPiece(")
    assert "if (talk.hosted) {" in start and "speakPiece(talk);" in start
    piece = between(page, "function speakPiece(", "function utterance(")
    assert "if (talk.hosted) {" in piece and "playHostedPiece(talk);" in piece
    # One "piece ended", one "piece started", one "talk ended", one hush for both voices.
    assert page.count("function pieceEnded(") == 1 and page.count("function pieceStarted(") == 1
    assert page.count("showSpeaking(true);") == 1
    clip = between(page, "function playClip(", "\n}\n")
    assert "pieceStarted(talk, talk.pieces[index]);" in clip and "pieceEnded(talk);" in clip
    words = between(page, "function utterance(", "function pieceStarted(")
    assert "pieceEnded(talk);" in words
    assert "stopHostedSound(talking);" in between(page, "function hush(", "\n}\n")
    assert "stopHostedSound(talk);" in between(page, "function talkEnded(", "\n}\n")
    # The page never speaks by itself: no hosted sound outside a talk that say() or speak() began.
    assert page.count("new Audio(") == 1 and "new Audio(url)" in clip
    assert "playClip(talk, url, index)" in between(page, "function playHostedPiece(", "\n}\n")


def test_the_hosted_voice_fetches_the_next_piece_while_one_plays_and_revokes_every_address():
    page = page_text()
    clip = between(page, "function playClip(", "\n}\n")
    assert "if (index + 1 < talk.pieces.length) hostedClip(talk, index + 1);" in clip
    cached = between(page, "function hostedClip(", "\n}\n")
    assert "if (!parts.clips[index]) parts.clips[index] = fetchClip(parts, talk.pieces[index]);" in cached
    assert "URL.createObjectURL(sound)" in page
    assert "URL.revokeObjectURL(url)" in between(page, "function forgetClip(", "\n}\n")
    stop = between(page, "function stopHostedSound(", "\n}\n")
    assert "for (const controller of parts.fetches) controller.abort();" in stop
    assert "for (const url of parts.urls) URL.revokeObjectURL(url);" in stop
    assert "sound.pause();" in stop


def test_the_hosted_voice_sends_json_and_falls_back_quietly():
    page = page_text()
    fetch = between(page, "async function fetchClip(", "\n}\n")
    assert 'fetch("/api/voice/say", {' in fetch
    assert 'headers: { "Content-Type": "application/json" }' in fetch and "JSON.stringify({ text: piece })" in fetch
    assert "ARABIC_LETTERS.test(piece) || piece.length > hosted.maxChars" in fetch    # never sent
    assert "setTimeout(() => controller.abort(), HOSTED_FETCH_MS)" in fetch
    assert "hostedRefused(await response.json());" in fetch and "return null;" in fetch
    # A failed piece: it and the rest of the answer in the browser's voice; no message, no retry.
    play = between(page, "function playHostedPiece(", "\n}\n")
    assert "else browserTakesOver(talk);" in play
    takes = between(page, "function browserTakesOver(", "\n}\n")
    assert "talk.hosted = null;" in takes and "startTalking(talk);" in takes
    assert page.count("fetchClip(") == 2          # its definition and the one cached call in hostedClip
    assert "addMessage" not in takes and "addMessage" not in fetch


def test_a_lasting_error_stops_the_hosted_voice_for_the_visit():
    page = page_text()
    assert 'const LASTING_KINDS = ["budget", "no_key", "key_refused", "no_credit", "off"];' in page
    refused = between(page, "function hostedRefused(", "\n}\n")
    assert "LASTING_KINDS.includes(data.kind)" in refused and "stopHosted(" in refused
    assert "!hostedStopped" in between(page, "function speaksHosted()", "\n}\n")
    assert "!hostedStopped" in between(page, "function listensHosted()", "\n}\n")


def test_the_test_button_speaks_with_whichever_voice_is_chosen():
    page = page_text()
    assert 'voiceTest.addEventListener("click", () => speak(testSentence(), "test"));' in page   # M26: or Arabic


# ---- the page: hosted listening --------------------------------------------------------------


def test_a_press_records_when_hosted_listening_is_on():
    page = page_text()
    start = between(page, "function startListening(", "function beginHearing(")
    assert start.index("if (listensHosted()) {") < start.index("new Recognition()")
    assert "startRecording(button, target, conv);" in start
    record = between(page, "function startRecording(", "\n}\n")
    assert "listeningSession(button, target, conv, null)" in record and "const wasTalking = hush();" in record
    assert "showListening(session, true);" in record
    opened = between(page, "async function openMicrophone(", "\n}\n")
    assert "navigator.mediaDevices.getUserMedia({ audio: true })" in opened
    assert "new MediaRecorder(stream)" in opened
    assert '"not-allowed"' in opened                                   # the same plain line as today's refusal
    assert "rec.maxSeconds * 1000 - RECORD_MARGIN_MS" in opened
    press = between(page, "function pressMic(", "\n}\n")
    assert "if (isRecording(hearing)) finishRecording(hearing);" in press


def test_recording_stops_after_quiet_following_speech_measured_by_an_analyser():
    page = page_text()
    assert "QUIET_END_MS = 1200" in page
    watch = between(page, "function watchLevel(", "\n}\n")
    assert "createAnalyser()" in watch and "createMediaStreamSource(rec.stream)" in watch
    check = between(page, "function checkLevel(", "\n}\n")
    assert "getFloatTimeDomainData" in check and "rec.spoke && now - rec.quietSince >= QUIET_END_MS" in check
    assert "setInterval" not in page                                       # a short chain of timeouts instead


def test_the_microphone_is_released_as_soon_as_recording_stops():
    page = page_text()
    release = between(page, "function releaseMicrophone(", "\n}\n")
    assert "for (const track of rec.stream.getTracks()) track.stop();" in release
    for function in ("function finishRecording(", "async function recordingStopped(", "function stopRecording(",
                     "async function openMicrophone("):
        assert "releaseMicrophone(session);" in between(page, function, "\n}\n"), function
    end = between(page, "function endListening(", "function sendHeard(")
    assert "if (session.recording) stopRecording(session);" in end
    assert "stopRecording(session);" in between(page, "function stopRecognizer(", "\n}\n")


def test_the_recording_becomes_a_16_khz_mono_wav():
    page = page_text()
    assert "WAV_RATE = 16000" in page
    assert "context.decodeAudioData(await blob.arrayBuffer())" in between(page, "async function recordingAsWav(", "\n}\n")
    wav = between(page, "async function speechWav(", "\n}\n")
    assert "new OfflineAudioContext(1, frames, WAV_RATE)" in wav and "Math.floor(maxSeconds * WAV_RATE)" in wav
    writer = between(page, "function wavBytes(", "\n}\n")
    for part in ('word(0, "RIFF")', 'word(8, "WAVE")', 'word(12, "fmt ")', 'word(36, "data")',
                 "view.setUint16(20, 1, true)", "view.setUint16(22, 1, true)", "view.setUint16(34, 16, true)",
                 "view.setInt16(44 + i * 2"):
        assert part in writer, part


def test_the_words_go_where_recognised_words_go():
    page = page_text()
    send = between(page, "async function sendRecording(", "\n}\n")
    assert '"/api/voice/hear?language=" + encodeURIComponent(answerLanguage())' in send
    assert 'headers: { "Content-Type": "audio/wav" }' in send and "body: wav" in send
    assert 'takeWords(session, data.text, data.text.trim() !== "");' in send
    assert "endListening(session);" in send and 'session.error = "hosted";' in send
    assert "takeWords(session, words, final);" in between(page, "function heardWords(", "\n}\n")
    # Nothing said, nothing sent: "nothing was heard" as today.
    stopped = between(page, "async function recordingStopped(", "\n}\n")
    assert 'session.error = "no-speech";' in stopped
    assert stopped.index('session.error = "no-speech";') < stopped.index("sendRecording(session, wav)")
    assert page.count("sendRecording(session, wav);") == 1 and "if (session.done || rec.sent) return;" in stopped


def test_the_microphones_show_when_either_way_of_listening_is_possible():
    page = page_text()
    draw = between(page, "function drawListening()", "\n}\n")
    assert "const canListen = Recognition !== null || listensHosted();" in draw
    assert "button.hidden = !canListen;" in draw
    for function in ("async function loadHosted()", "function stopHosted("):
        assert "drawListening();" in between(page, function, "\n}\n"), function


def test_the_privacy_line_names_the_hosted_service():
    page = rendered_page()
    listen = ('<span class="privacy-hosted-listen" hidden>With hosted listening, the recording of your question '
              "is sent to a Hugging Face inference provider to turn it into text.</span>")
    speak = ('<span class="privacy-hosted-speak" hidden>With the hosted voice, the text of the answer is sent to '
             "a Hugging Face inference provider to be spoken.</span>")
    assert page.count(listen) == 2 and page.count(speak) == 2
    draw = between(page_text(), "function drawVoicePrivacy()", "\n}\n")
    assert 'querySelectorAll(".privacy-hosted-listen")) part.hidden = !hostedListen;' in draw
    assert 'querySelectorAll(".privacy-hosted-speak")) part.hidden = !hostedSpeak;' in draw


# ---- fix round: conversations end, the microphone question has a limit, the hosted voice alone --------


def test_a_conversation_pauses_after_max_turns_or_max_full_turns():
    page = page_text()
    assert "MAX_TURNS = 20" in page and "MAX_FULL_TURNS = 3" in page
    assert 'CONVERSATION_PAUSED = "The conversation was paused. Press the microphone to go on.";' in page
    too_long = between(page, "function conversationTooLong(", "\n}\n")
    assert "conv.turns >= MAX_TURNS || conv.fullTurns >= MAX_FULL_TURNS" in too_long
    listen = between(page, "function toListening(", "\n}\n")
    assert listen.index("if (conversationTooLong(conv)) {") < listen.index("conv.turns += 1;") < listen.index("startListening(")
    paused = listen[listen.index("if (conversationTooLong(conv)) {"):listen.index("conv.turns += 1;")]
    assert "endConversation(conv);" in paused and "target.reply(CONVERSATION_PAUSED);" in paused
    assert "speak(" not in paused and "say(" not in paused                     # nothing is spoken about it
    assert "conv.fullTurns = session.full ? conv.fullTurns + 1 : 0;" in between(page, "function turnEnded(", "\n}\n")
    assert "turns: 0, fullTurns: 0" in between(page, "function startConversation(", "\n}\n")


def test_both_listeners_mark_a_turn_that_ran_to_the_time_limit():
    page = page_text()
    assert "full: false" in between(page, "function listeningSession(", "\n}\n")
    browser = between(page, "function listeningTimedOut(", "\n}\n")         # the browser's recogniser
    assert "session.full = true;" in browser
    opened = between(page, "async function openMicrophone(", "\nfunction microphoneNotGiven(")
    limit = opened[opened.index("rec.maxSeconds * 1000 - RECORD_MARGIN_MS") - 140:]
    assert "session.full = true;" in limit and "finishRecording(session);" in limit
    assert "session.full" not in between(page, "function checkLevel(", "\n}\n")   # quiet is not full


def test_the_microphone_question_is_given_up_after_30_seconds():
    page = page_text()
    assert "MICROPHONE_WAIT_MS = 30000" in page
    opened = between(page, "async function openMicrophone(", "\nfunction microphoneNotGiven(")
    assert opened.index("setTimeout(() => microphoneNotGiven(session), MICROPHONE_WAIT_MS)") < opened.index("getUserMedia")
    assert "if (session.done) {" in opened and "releaseMicrophone(session);" in opened   # a late microphone is let go
    given_up = between(page, "function microphoneNotGiven(", "\n}\n")
    assert 'session.error = "not-allowed";' in given_up and "endListening(session);" in given_up


def test_the_browsers_audio_capture_keeps_its_old_line():
    page = page_text()
    problems = between(page, "const LISTEN_PROBLEMS = {", "};")
    assert '"audio-capture"' not in problems                                     # as before M23
    assert '"hosted-no-microphone": "No microphone was found.' in problems
    assert '"hosted-no-microphone"' in between(page, "async function openMicrophone(", "\nfunction microphoneNotGiven(")


def test_the_hosted_voice_works_without_the_browsers_own_speech():
    page = page_text()
    assert "return hosted !== null && hostedSpeakWanted && !hostedStopped && canPlayHosted();" in page
    controls = between(page, "function drawVoiceControls()", "\n}\n")
    assert "voiceButton.hidden = speech === null && !offered;" in controls
    assert "for (const part of [voiceChoiceLabel, voiceChoice, voiceHelp]) part.hidden = speech === null;" in controls
    assert "voiceTest.hidden = speech === null && !speaksHosted();" in controls
    assert "drawVoiceControls();" in between(page, "function drawHosted()", "\n}\n")
    assert "drawVoiceControls();" in between(page, "function loseSpeech()", "\n}\n")
    takes = between(page, "function browserTakesOver(", "\n}\n")
    assert takes.index("if (!speech) {") < takes.index("startTalking(talk);")   # no browser voice: the piece is left out
    assert "pieceEnded(talk);" in takes
    assert "if (hostedStopped) return null;" in between(page, "async function fetchClip(", "\n}\n")
