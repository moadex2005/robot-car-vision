# M23 spec: a hosted voice and hosted listening

Today the page speaks and listens with the browser's own speech. That is only good in Microsoft Edge: other browsers have robotic voices, and Firefox and many phones cannot listen at all. After this milestone the page can speak with the same natural voice in every browser and understand a spoken question in every browser, through two hosted models reached with the Hugging Face token the project already has. It has a hard daily cost limit, like auto-discover.

Read `CLAUDE.md`, then `specs/M14-auto-discover.md` and `car_vision/lookout.py` (the model for this module: one small file, a replaceable `fetch`, the spend file, plain errors, the key never in a message), then the sections `// ---- voice` of `car_vision/templates/index.html` (speaking: `say`, `speak`, `startTalking`, `speakPiece`, `hush`; listening: `pressMic`, `startListening`, `heardWords`, `endListening`; conversation mode).

## What was measured before this was written (the owner's key, synthetic text only)

```
POST https://router.huggingface.co/deepinfra/v1/openai/audio/speech
Authorization: Bearer <HF_TOKEN>      Content-Type: application/json
{"model": "hexgrad/Kokoro-82M", "input": "Hello, this is the robot car.", "voice": "af_heart", "response_format": "mp3"}
-> 200, Content-Type: audio/mpeg, the raw MP3 bytes (30 kB for 61 characters; 203 kB for 420 characters)
   1.4 to 2.9 seconds. "response_format": "wav" gives audio/wav (215 kB for the same sentence).

POST https://router.huggingface.co/deepinfra/v1/openai/audio/transcriptions
Authorization: Bearer <HF_TOKEN>      Content-Type: multipart/form-data; boundary=...
  file = <the WAV bytes, filename question.wav, audio/wav>     model = openai/whisper-large-v3-turbo     language = en
-> 200, application/json: {"text": " Hello, this is the Robot Car.  I can see a bottle on the left."}
   1.2 to 1.8 seconds. The language field is optional. An MP3 of the same speech lost its second sentence: send WAV.
```

Listed prices: speech 0.99 dollars per million characters (a 300-character answer: 0.0003 dollars); listening 0.0002 dollars per minute (a 6-second question: 0.00002 dollars). The speech model speaks English only: it has no Arabic voice. The listening model understands Arabic.

## Settings (already in `config.yaml`, added by the orchestrator)

```yaml
voice:
  hosted: true
  provider: deepinfra
  speech_model: "hexgrad/Kokoro-82M"
  speech_voice: af_heart
  listen_model: "openai/whisper-large-v3-turbo"
  max_spoken_chars: 600
  max_listen_seconds: 20
  daily_budget_usd: 0.25
  speech_price_per_million_chars: 0.99
  listen_price_per_minute: 0.0002
  spend_file: data/voice.json
```

The key is `HF_TOKEN` in `.env`, as for auto-discover. Never printed, logged, committed or put in a test, also when the server echoes it.

## Part A: `car_vision/voice.py` (new; standard library, plus `dotenv` as `lookout.py` uses it)

```python
class VoiceError(Exception):
    kind: str            # "off", "no_key", "key_refused", "no_credit", "not_available", "busy", "budget", "bad_input", "failed"

class HostedVoice:
    def __init__(self, settings: dict, fetch=None, clock=None) -> None: ...
    def say(self, text: str) -> bytes: ...                       # MP3 bytes
    def hear(self, wav: bytes, language: str = "en") -> str: ... # the words
    @property
    def spent_today(self) -> float: ...
    @property
    def budget(self) -> float: ...
    @property
    def uses_today(self) -> int: ...
```

- `say`: the text trimmed; empty, longer than `max_spoken_chars`, or holding Arabic letters is a `VoiceError("bad_input")` with a plain message (the model cannot speak Arabic; the page then uses the browser's voice). One request as measured; the answer must be `audio/*` and start like an MP3 (`ID3` or a frame sync byte `0xFF`), else `VoiceError("failed")`.
- `hear`: `wav` must start with `RIFF....WAVE`, be at most 2 MB and, by its header, at most `max_listen_seconds` long, else `VoiceError("bad_input")`. `language` is `"en"` or `"ar"`. One multipart request as measured, built by hand (no library). The answer's `text`, trimmed and cut at 500 characters; no text is `""`.
- The cost estimate grows by the characters spoken times the price, and by the recording's seconds (from the WAV header) times the price. The day's count and estimate live in `spend_file`, exactly as `lookout.py` keeps its own (atomic write, a new day starts from zero, a broken file starts from zero with one log line). Before sending, both calls raise `VoiceError("budget")` when the estimate has reached `daily_budget_usd`.
- Errors as in `lookout.py`: no key, 401 or 403, 402, 404, 429 and 5xx, a timeout or no network, an unexpected answer. Each with a plain message; nothing leaked; the key never in a message or a log line.
- Nothing about the text, the words heard or the audio is logged: counts and errors only.
- `fetch(url, headers, body_bytes, timeout) -> (status, content_type, body_bytes)` is replaceable; the real one uses `urllib` with a 30 second timeout and follows no redirect to another host.
- One call at a time per kind (a lock each); a second `say` waits its turn, at most 20 seconds.

## Part B: `car_vision/app.py`

`create_app(...)` gains a parameter `voice: Any = None` (a `HostedVoice`, or `None` when `voice.hosted` is false; tests pass a fake). Three routes, placed directly after the `discover` route:

- `GET /api/voice`: `{"hosted": true, "uses": 12, "spent_usd": 0.0031, "budget_usd": 0.25, "max_chars": 600, "max_seconds": 20}` or `{"hosted": false}`.
- `POST /api/voice/say` with JSON `{"text": "..."}`: 200 with `Content-Type: audio/mpeg`, `Cache-Control: no-store` and the bytes. A `VoiceError`: 200 with JSON `{"error": "<plain message>", "kind": "<kind>"}` (as `/api/ask` answers an `AssistantError`), so the page can fall back quietly. Wrong input: 400.
- `POST /api/voice/hear?language=en` with the WAV as the request body and `Content-Type: audio/wav`: `{"text": "..."}`, or `{"error", "kind"}` as above. The body is refused above 2 MB before it is read into memory (`request.content_length`). This one changing route does not send JSON, so `refusal()` must let exactly this path through with `audio/wav` (and nothing else); the Origin check still applies. Change `refusal` for that and nothing more.
- `main()` builds the `HostedVoice` when `voice.hosted` is true; a wrong setting prints one plain line and the program runs without the hosted voice.

## Part C: `templates/index.html`, the voice code and the Voice box only

**Speaking.** In the Voice box, above the list of browser voices, a switch `Hosted voice: the same in every browser` (shown only when `GET /api/voice` says `hosted: true`; off until switched on; remembered in this browser like the other voice choices). Beside it, small and quiet: `12 uses today · about $0.003` or the reason it paused.

- With it on, `speak(text, kind)` gets its audio from `/api/voice/say`, piece by piece: the text is cleaned as today (`earText`) and cut into pieces as today (`speechPieces`, with each piece at most `max_chars`); the first piece is fetched and played with an `Audio` element from a blob address; the next piece is fetched while the one before plays, so there is no gap; each blob address is revoked after use. `hush()` stops the sound at once and drops what is still being fetched (an `AbortController`), exactly as it stops the browser's voice today. Everything that depends on "the page is speaking" (the speaker button's waves, conversation mode waiting for the end of the answer, a new answer replacing an old one) works the same with either voice: keep one small set of functions that both voices go through, do not copy the logic.
- A piece that holds Arabic letters, any `{error}` answer, a failed request or audio that will not play: that piece and the rest of this answer are spoken by the browser's voice as today, without a message on the page. After a `budget`, `no_key`, `key_refused` or `no_credit` error the hosted voice is not tried again in this visit, and the small text says why.
- The rule that the page never speaks by itself is untouched: only what `say()` is called for today is spoken. Browsers block sound until the visitor has touched the page; the existing `notePageTouched` logic stays the one judge of that.
- The Test button speaks its sentence with whichever voice is chosen.

**Listening.** A second switch in the Voice box: `Hosted listening: works in every browser` (shown when hosted is true and the browser can record: `navigator.mediaDevices.getUserMedia` and `MediaRecorder` exist; on by default in a browser that has no speech recognition of its own, off by default where it has).

- With it on, pressing a microphone button records instead of using the browser's recognition: `getUserMedia({audio: true})`, a `MediaRecorder`, a visible "listening" state as today; it stops when the visitor presses again, after 1.2 seconds of quiet following speech (measure the level with an `AnalyserNode`), or at `max_seconds`. The recording is then decoded with `decodeAudioData`, mixed to one channel, resampled to 16000 Hz with an `OfflineAudioContext`, written as 16-bit WAV, and posted to `/api/voice/hear?language=<answerLanguage()>`. The words that come back go exactly where recognised words go today (`heardWords` and what follows it: the chat, the follow-up, the stage, find by words, conversation mode). An empty text is "I did not hear anything", as today.
- The microphone is released as soon as the recording stops (every track stopped), so the browser's recording sign goes away. A refused permission shows the same plain line as today's refusal.
- The microphone buttons are shown when either way of listening is possible.
- The privacy line of the voice box (`drawVoicePrivacy`) says, when a hosted switch is on, that the answer's text or the recording of the question is sent to a Hugging Face inference provider.

`answerLanguage()` already exists near the top of the script and returns `"en"`; another milestone will make it return `"ar"`. Call it; do not change it. Other work is going on in this file in parallel: touch only the voice sections and the Voice box's markup and CSS.

## Tests (new files)

- `tests/test_voice.py`, with a fake `fetch` and clock (no network, no real key): both request shapes exactly (address, headers, JSON body, the multipart body parsed back into its parts); the checks on text and on WAV headers (make WAVs with the `wave` module: too long, not a WAV, too large); Arabic text refused; each error kind; the key never in an error or the log; the estimate for both calls; the budget stop; the spend file across a restart, a new day and a broken file; the settings checks; the two locks.
- `tests/test_app_voice.py`: the three routes with a fake voice (the audio answer's headers and bytes, each error as 200 JSON, 400s, the 2 MB refusal without reading the body, `hosted: false`), the same-origin rule on `/api/voice/hear`, that `refusal()` still refuses every other non-JSON changing request, and string checks on the template (the two switches, one shared set of speaking functions, `URL.revokeObjectURL`, every track stopped, `answerLanguage()` used for the language).

## Checking the JavaScript for real

The WAV writer is the part most likely to be wrong. Before you report, run it once in headless Edge on a scratch page outside the project: synthesize one second of a 440 Hz tone at 48000 Hz in two channels as an `AudioBuffer`, pass it through your own functions copied from the template, and print the first 44 header bytes, the sample rate, the channel count and the length. Put the result in your report. Do not use the Claude browser pane.

## Exit test

The whole test suite passes. You make no real request: the orchestrator checks live, with the real key, that the hosted voice speaks an answer and that a spoken question typed by voice arrives in the chat, and reads the spend file.
