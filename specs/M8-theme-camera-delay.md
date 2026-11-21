# M8 spec: light and dark, the camera address on the page, less delay

Three requests from the project owner after the first live run on the phone:

1. A switch on the page between a light and a dark look.
2. A box on the page for the phone's address, so another phone can be used without editing a file or asking anyone.
3. Less delay between moving the camera and seeing it on the page.
4. The video in full screen, with the details panel working there too.
5. A small chat inside the details panel, to ask more about the thing that was clicked.

Read `CLAUDE.md` first, then `specs/M7-click-to-inspect.md` (its "Look and feel" section still holds). The work is three parts for three people working at the same time. The interfaces between them are fixed: the `Capture` protocol in `stream.py` and the settings dictionary of `FrameSource` (unchanged), and `Assistant.follow_up` as written in Part C.

---

## Part A: the phone stream reader (`car_vision/stream.py`, `tests/test_stream.py`)

> **Outcome (measured on the real phone, 2026-11-21): built, reviewed, then taken out again.** The reader below worked, but it is Python, and Python threads share one lock with the detector's thread. Reader alone, the stream delay was the same as FFmpeg's (59 to 79 ms against 66 ms at 640 x 360). Inside the full pipeline it lost badly: FFmpeg let the pipeline process 25.8 pictures a second, our reader 14.5, and in one run 3. FFmpeg reads and decodes in C, outside that lock. The phone is therefore read with FFmpeg and its low-delay options, and the delay was cut by asking the phone for a smaller picture instead (`specs/M9-phone-picture-size.md`). The text below is kept as the record of what was tried.

Today the phone is opened with `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`. Every 1920 x 1080 picture is decoded in full and then shrunk to 640 x 360, and the laptop has no say over buffering. Replace that with a reader of our own for the phone's MJPEG stream.

```python
class MjpegCapture:                      # has read(), release(), isOpened(): the Capture protocol
    def __init__(self, url: str, max_width: int, max_height: int,
                 open_timeout: float, read_timeout: float, opener=None) -> None: ...
```

- **Standard library only** for the network side (`urllib.request` or `http.client`). The stream is `multipart/x-mixed-replace`: parts with their own headers, each holding one JPEG. Use `Content-Length` when a part has it; otherwise find the picture by its JPEG start and end markers. A boundary given with or without quotes or leading dashes must work.
- **The network never waits for decoding.** A small thread reads parts and keeps only the newest undecoded JPEG. `read()` hands back the newest picture, decoded; when a newer one arrived while the last was being decoded, the older one is dropped, never queued. `read()` waits up to `read_timeout` for a picture it has not returned before, then answers `(False, None)`, which makes `FrameSource` reconnect as it does today.
- **Decode small.** Decode with OpenCV's reduced modes (`cv2.IMREAD_REDUCED_COLOR_2`, `_4`, `_8`): choose the largest reduction whose result is still at least as large as the size `fitted_size` will ask for, so nothing visible is lost and `FrameSource._fit` finishes the job. A pure function `reduction_for(width, height, max_width, max_height) -> int` (1, 2, 4 or 8) decides it from the JPEG's own size, read from its header without decoding.
- **Opening.** `isOpened()` is true once the server answered with status 200 and a multipart content type. Anything else (refused, timed out, another content type) is false, and `release()` always closes the socket and ends the thread within a second.
- **Fallback.** `FrameSource`'s phone opener tries `MjpegCapture` first. When it does not open because the server is there but speaks something else (a 200 answer that is not multipart), it falls back to the old FFmpeg capture, opened with the low-delay options `fflags;nobuffer|flags;low_delay` through `OPENCV_FFMPEG_CAPTURE_OPTIONS`. A phone that is simply not reachable does not go to the fallback (it would only double the waiting time).
- A broken picture (cut off, not a JPEG) is skipped and logged once, not raised.
- Nothing else in `FrameSource` changes: the reconnect loop, `read()`, the sequence numbers, the webcam and the clip.
- **Tests** (no sockets, no camera): the parser on in-memory byte streams built in the test (with and without `Content-Length`, a quoted boundary, two pictures arriving in one read, a picture split across many reads, junk between parts, a cut-off stream); `reduction_for` on 1920 x 1080 into 640 x 480 (4 would give 480 x 270, too small, so 2), 1280 x 720, 640 x 480, and a tiny picture; newest-wins when three pictures are waiting; `read()` timing out; `release()` ending the thread; the fallback rule with fake openers.
- Add `--seconds`-style output to `python -m car_vision.stream` only if it is already there; do not add new command-line options.

## Part B: the page and the server (`car_vision/pipeline.py`, `car_vision/app.py`, new `car_vision/phone.py`, `car_vision/templates/index.html`, their tests)

### B1. Light and dark

- A small button in the navigation bar switches between the light and the dark look. It shows which one is active (a sun or moon drawn in inline SVG or with CSS, no emoji, no outside file) and has a proper label for screen readers.
- Until the visitor chooses, the page follows the system (`prefers-color-scheme`) as it does now. The choice is kept in `localStorage` (wrapped in `try`/`catch`; the page must work without it) and applied before the first paint by a few lines of script in the `<head>`, so there is no flash of the wrong look.
- Implementation: the colour variables are defined once for light and once for dark; `:root[data-theme="light"]` and `:root[data-theme="dark"]` win over the media query. Changing the look cross-fades colours briefly; no layout moves.
- Both looks must stay readable: check the status pill, the pills, the panel, the chat, the banners and the camera box in both.

### B2. The camera address

New module `car_vision/phone.py` (standard library only), which owns everything about the phone's address:

```python
def phone_url(text: str) -> str: ...        # the full stream address, or ValueError with a plain message
def load_remembered(path: Path) -> str | None: ...
def remember(path: Path, url: str) -> None: ...
```

- `phone_url` accepts what a person types from the phone's screen: `192.168.1.7`, `192.168.1.7:8080`, `http://192.168.1.7:8080`, `http://192.168.1.7:8080/`, `http://192.168.1.7:8080/video`, with spaces around it. It returns `http://<ip>:<port>/video` (port 8080 when none is given). It refuses, each with its own short message: empty text; anything that is not an IPv4 address (names, IPv6); an address outside the private and loopback ranges (the phone is always on the same local network; this also keeps the server from being pointed at the internet); a port outside 1 to 65535; `https`; a user name or password in the address; any path other than none, `/` or `/video`.
- The last address that was asked for is kept in `data/camera.json` (git-ignored; the orchestrator adds the ignore line). At start-up, a remembered address replaces `stream.phone_url` from `config.yaml` for that run. `config.yaml` is never written. A missing, unreadable or invalid file is ignored with one log line.

`pipeline.py`:

```python
class Pipeline:
    def use_phone(self, url: str) -> None: ...     # switch the live source to this phone, now
```

- Builds a new `FrameSource` for the phone source at that address (through a factory that tests can replace), swaps it in, and stops the old one. The loop thread must never read a half-swapped source, and must start again if it had stopped (after ten failures, or a finished clip). Everything tied to the old picture is cleared: the frame sequence it last saw, the remembered faces, the failure count, the error, the "stream lost" line, the view (so a click cannot hit the old camera's picture).
- `status()` gains `"phone_url"`: the address in use or remembered, so the page can show it.
- Switching while a switch is still running is safe (one at a time; the last one wins).

`app.py`:

| Route | Behaviour |
| --- | --- |
| `POST /api/camera` | Body `{"address": "192.168.1.7:8080"}`. Bad input: 400 with `phone_url`'s message. Otherwise calls `pipeline.use_phone(url)`, remembers the address, and answers `{"ok": true, "url": ...}` at once; it does not wait for the phone. |

The page: a **Camera** control in the navigation bar (the status pill itself may open it). It opens a small glass box with: the text field `Phone address` (showing the current address without `http://` and `/video`), a **Connect** button (Enter works), one line of help (`The address IP Webcam shows on the phone's screen, like 192.168.1.7:8080.`), and a result line. After Connect the result line says `Connecting ...`, then `Connected` as soon as the status reports a live phone picture, or after 8 seconds `No picture from <address> yet. Check that IP Webcam is started and that the phone and this laptop are on the same Wi-Fi.` A 400 shows the server's message in the error colour. Escape and a click outside close the box.

### B3. A delay meter

So that delay can be measured through the whole path (phone, network, models, browser) instead of guessed:

- In the camera box, a switch **Delay test**. While it is on, a large clock is shown beside or under the video: the time of day with milliseconds, in tabular numerals, updated on every animation frame. Off by default; when off, no timer or animation-frame loop runs.
- How it is used: point the phone at the laptop screen. The clock shows now, the clock inside the video shows then, the difference is the delay. One sentence under the clock says so.
- This is the only animation-frame loop on the page, and it exists only while the switch is on.

### B4. Full screen

The owner sometimes wants the video to fill the whole screen, with everything still working there.

- A button on the video frame (an expand mark drawn inline, with a label for screen readers) and the `F` key switch full screen on and off, using the browser's Fullscreen API on one wrapper element. No double-click: a click on the video is already "inspect", and it can cost an API call.
- That wrapper holds everything that must stay usable in full screen: the video, the greeting and warning banners, the In view pills and the details panel. A click on the video or on a pill opens the details panel inside full screen exactly as on the page, with its photo, text, sources and the follow-up chat of B5.
- Layout in full screen: a black stage; the video as large as possible, keeping its shape; the pills in one row over the bottom edge of the stage (they may scroll sideways); when the details panel is open it takes its own column at the right and the video makes room for it, so no large blurred surface lies over the moving picture. The click maths must stay right in every one of these layouts.
- Leaving full screen (the button, `F`, or the browser's own Escape) returns to the page as it was, with the same panel still open. Where the Fullscreen API is missing or refused, the button is hidden.

### B5. Asking more about what was clicked

Under the description in the details panel there is a small chat about that one thing.

- A text field (`Ask about this`) and a send button; Enter sends. Questions and answers appear above it, newest at the bottom, in the panel's own scroll area. While waiting, the field is disabled and a waiting line shows. Web answers show their sources as in the main chat. Errors show in the error colour.
- It appears once the description has arrived (or failed), and for a recognised person at once. An unrecognised person has no chat.
- Each clicked thing has its own short conversation, kept by the server with the inspection (at most 10 questions each; after that the field is replaced by one quiet line). Clicking another thing shows that thing's conversation; a late answer for a thing that is no longer shown is ignored.

| Route | Behaviour |
| --- | --- |
| `POST /api/follow_up` | Body `{"id": "<inspection id>", "question": "..."}`. Empty or over-long question (the chat's 2000-character limit), unknown id, an unrecognised person, or more than 10 questions: 400 or 404 with a plain message. For an object or an area: `assistant.follow_up(photo, label or None, info, earlier, question)`, where `info` is the description this inspection got (or `None` when it failed) and `earlier` the question-answer pairs so far. For a recognised person: `assistant.ask(...)` with the question prefixed by one sentence naming the person (`This question is about <name>, who the camera recognises.`); no picture is involved. Answers `{"text", "used_web", "sources"}`; `AssistantError` gives status 200 with `{"error": ...}`. A failed question is not added to the conversation. |

- The inspection store keeps, for each inspection, the description once `/api/describe` has answered (also when it came from the cache) and the follow-up pairs. It stays bounded as before.
- The privacy line gains: follow-up questions about an object send the same cut-out again.

### Tests

- Routes for `/api/follow_up`: an object with and without earlier pairs (the assistant receives exactly the stored photo, label, description and pairs); a recognised person goes to `ask` and never to `follow_up`; each refusal above; the limit of 10; a failed question leaves the conversation unchanged.
- The template: the full-screen button, the wrapper that holds video, pills and panel, the follow-up field.
- `phone_url`: every accepted form and every refusal above; remember and load, including a broken file.
- `Pipeline.use_phone` with a fake source factory: the new source is used and the old one stopped; frames from the new source reach the state; the view is cleared; the loop restarts after it had stopped; `status()["phone_url"]`.
- Routes: the good case (pipeline called with the normalised address, file written to a temporary path), each kind of bad input, a non-JSON body.
- The template: the theme button, the camera box, the delay switch are present; the theme script in the head; still no `innerHTML`, no outside resources, no endless animation (the existing test).

---

## Part C: follow-up questions (`car_vision/assistant.py`, its tests)

```python
class Assistant:
    def follow_up(self, image: bytes, label: str | None, info: ObjectInfo | None,
                  earlier: Sequence[tuple[str, str]], question: str) -> Answer: ...
```

- One call that carries its whole conversation with it; the assistant keeps nothing between calls and the main chat's history is neither read nor changed. The turns sent are: the same first user turn `describe` sends (the picture and the label sentence); when `info` is given, an assistant turn holding that description written back in the `NAME` / `ABOUT` / `SPECS` shape; then each earlier question and answer as plain text turns; then the new question.
- Its own system prompt, a module constant: the visitor is asking more about the thing in the picture; answer in two or three sentences of plain text (no `NAME`/`SPECS` shape); use the picture and the description; search the web when the answer needs facts you do not have and the tool is available; never invent; the same rules about people as `describe`; answer in English.
- The same checks as elsewhere, before anything is sent: `check_image`, `label_text`, the question cleaned and limited like `ask` (`ValueError`), at most 10 earlier pairs (`ValueError`).
- It waits on the same lock as `describe`, with the same busy message. Web search follows `assistant.web_search`. `Answer` is filled as in `ask` (final text, `used_web`, de-duplicated sources).
- A pure function `info_text(info: ObjectInfo) -> str` writes a description back in the shape `parse_object_info` reads; test that parsing its output gives the same name, summary and specs.
- Tests with the fake client: the exact turns for no info, info and no pairs, info and two pairs; the chat history untouched; refusals; the busy case; one request-shape test in `tests/test_assistant_wire.py`.

---

## Not in this spec

Asking the phone itself for a smaller picture (IP Webcam can change its video size on request) is probably the biggest gain, because a 640 x 360 picture is about a ninth of the data of 1920 x 1080. It is being looked up and will be checked against the real phone before anything is built.
