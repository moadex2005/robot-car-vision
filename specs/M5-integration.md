# M5 spec: integration, reactions and the demo page

Join the camera, the detector, the recogniser and the assistant into one program with one demo page in the browser, and add two simple reactions. No new models. Nothing is sent to the car.

Read `CLAUDE.md` first, then the modules already built: `state.py`, `config.py`, `stream.py`, `detector.py`, `draw.py`, `faces.py`, `recognizer.py`, `assistant.py`. They are fixed; if one of them makes this spec impossible, stop and report it.

## Files to create

| File | What it holds |
| --- | --- |
| `car_vision/reactions.py` | Pure logic: decides greetings and the close warning from a `WorldSnapshot`. |
| `car_vision/pipeline.py` | The see-decide-show loop in its own thread. Imports no third-party library except `cv2` and `numpy`. |
| `car_vision/app.py` | The Flask server. The only module that imports `flask`. |
| `car_vision/templates/index.html` | The demo page: one file, its CSS and JavaScript inline, no outside links. |
| `tests/test_reactions.py`, `tests/test_pipeline.py`, `tests/test_app.py` | Unit tests with fakes. |

## `car_vision/reactions.py`

```python
@dataclass(frozen=True)
class Reaction:
    kind: str                 # "greeting" or "warning"
    text: str
    person_id: str | None = None

class ReactionEngine:
    def __init__(self, settings: dict) -> None: ...
    def update(self, snapshot: WorldSnapshot, now: float) -> list[Reaction]: ...
```

`settings` is the `reactions:` section: `greet_cooldown_seconds`, `close_box_height_ratio`, `close_hold_seconds`, `message_seconds`. `update` is called once per processed frame and returns the reactions that should be on screen at `now`.

- **Greeting.** When a known face appears and that person has not been greeted within `greet_cooldown_seconds`, start a greeting `Hello, <name>!` that stays for `message_seconds`. One greeting per person per cooldown, however many frames they stay in view. Two known people in view get one greeting each.
- **Close warning.** A person is close when the height of their `person` box is at least `close_box_height_ratio` of the frame height. When any person has been close without a break for `close_hold_seconds`, show the warning `Person close in front of the car`. It disappears on the first frame in which nobody is close. A frame size of zero means no warning.
- The engine has no clock of its own; everything is driven by `now`. Log each start and end of a reaction once, at INFO level, with the reason (the name, or the box height ratio).

## `car_vision/pipeline.py`

```python
class Pipeline:
    def __init__(self, config: dict, source=None, detector=None, recognizer=None,
                 reactions=None, state: WorldState | None = None) -> None: ...
    def start(self) -> "Pipeline": ...
    def stop(self) -> None: ...
    def latest_jpeg(self) -> bytes | None: ...     # the newest annotated frame
    def status(self) -> dict: ...
    @property
    def state(self) -> WorldState: ...
```

The injected parts are for tests. When they are `None`, the pipeline builds the real ones from `config`.

The worker thread, for every new frame from the `FrameSource` (skip frames already seen, by sequence number): run the detector; on every `recognizer.every_n_frames`-th frame run the recogniser and `FaceMemory.remember`; take the faces to show from `FaceMemory.current`; write detections and faces into the `WorldState`; ask the `ReactionEngine`; draw the detections and the face boxes with names (no info card on the video: the page shows the cards); encode the frame as JPEG with `app.jpeg_quality`.

`status()` returns a plain dictionary that can be turned into JSON as is:

```python
{
  "source": "webcam", "connected": True,
  "fps": 21.4,                          # processed frames per second
  "device": "cuda",
  "recognition": True,                  # False when it could not be started
  "notes": ["..."],                     # start-up problems worth showing, e.g. nobody enrolled yet
  "people": [{"id": "sara", "name": "Sara Ahmed", "role": "Team lead",
              "lines": ["...", "..."], "similarity": 0.61}],
  "objects": [{"label": "chair", "count": 2}],
  "reactions": [{"kind": "greeting", "text": "Hello, Sara Ahmed!"}],
}
```

Start-up must degrade, not die: with no embeddings file or nobody enrolled, run without recognition and say so in `notes`; a detector that cannot load is a real failure and raises. While the stream is down, keep serving the last frame with a `reconnecting` status line drawn on it. A frame that makes the detector or recogniser raise is logged and skipped; the loop carries on, and after 10 failures in a row it stops and `status()` reports the error. `stop()` ends the thread within two seconds.

## `car_vision/app.py`

```python
def create_app(pipeline, assistant) -> Flask: ...
def main(argv: list[str] | None = None) -> int: ...
```

| Route | Behaviour |
| --- | --- |
| `GET /` | The demo page. |
| `GET /video` | MJPEG stream (`multipart/x-mixed-replace`) of `pipeline.latest_jpeg()`, at most 20 frames a second, sending a frame only when it is new. |
| `GET /api/status` | `pipeline.status()` as JSON. |
| `POST /api/ask` | Body `{"question": "..."}`. Returns `{"text", "used_web", "sources"}`. An `AssistantError` returns status 200 with `{"error": "<its message>"}`; an empty or over-long (more than 2000 characters) question returns 400. |
| `POST /api/reset` | Forgets the conversation. |

`main` loads the config, sets up logging to the console and to `app.log_file` (create the folder), builds the `Pipeline` and the `Assistant` (sharing the pipeline's `WorldState`), starts the pipeline, prints the address of the page, and serves on `app.host` and `app.port` with threading on and the reloader and debug mode off. Arguments: `--source` overrides `stream.source`; `--file PATH` plays that clip (it implies `--source file` and overrides `stream.file_path` for this run); `--open` opens the page in the default browser. Ctrl+C stops the pipeline cleanly. The server binds to `app.host` only; do not bind to all interfaces.

If the assistant cannot be built (missing notes file, for example), the page still works and `/api/ask` returns that error.

## The demo page

One screen, readable from the back of a room, working at 1280 x 720 and larger, in light text on a dark background.

- Left: the live video, as large as the space allows, keeping the frame's own shape (the phone sends a wide 16:9 picture, the laptop webcam 4:3).
- Over the video, at the top: the warning as a red banner while it is active, and greetings as a calmer banner.
- Right column, top: an info card for each recognised person (name, role, lines). When nobody is recognised, a quiet line saying so.
- Right column, bottom: the chat. A scrolling list of questions and answers, a text box and a Send button; Enter sends. While waiting, the button is disabled and a "thinking" line shows. A web answer is marked `From the web` and lists its sources as links that open in a new tab. Errors show in the chat in a distinct colour.
- A small status line: source, FPS, device, and `reconnecting` when the stream is down. Start-up notes from `status()` show here too.
- The page polls `/api/status` twice a second and updates only what changed.
- Every piece of text that comes from the server (names, card lines, answers, notes, source URLs) is inserted with `textContent`, never as HTML. Source links are created only for `http:` and `https:` URLs.
- If the status request fails, show `Program not running` instead of stale data.

## Tests

No camera, network, graphics card or key needed.

- `ReactionEngine`: one greeting per person per cooldown; greeting again after the cooldown; two people; greeting expires after `message_seconds`; warning only after the hold time; warning clears at once; a short gap resets the hold; zero frame size; unknown faces are never greeted.
- `Pipeline` with fake source, detector, recogniser: detections and faces reach the `WorldState`; the recogniser runs only on every n-th frame; `latest_jpeg` returns decodable JPEG bytes; `status()` has every key and is JSON-serialisable; a raising detector frame is skipped; ten in a row stop the loop with an error in the status; start with a recogniser that fails to build gives `recognition: False` and a note; `stop()` returns within two seconds.
- `app` with Flask's test client and fake pipeline and assistant: each route's good case; `/api/ask` with an `AssistantError`, an empty question, an over-long question, and a body that is not JSON; `/video` returns the right content type and at least one frame.
- The page template contains no `innerHTML` and no `http://` or `https://` references to outside resources.
- One test that fails if any file other than `app.py` imports `flask`.

## Exit test on the laptop

1. `.venv\Scripts\python.exe -m pytest` passes.
2. `.venv\Scripts\python.exe -m car_vision.app --source webcam` starts, the page loads, the video shows labelled objects, `/api/status` reports `cuda` and at least 12 FPS, and a question typed without an API key shows the missing-key message in the chat.
3. Covering the laptop camera or pulling the stream does not crash the program.

Three full demo runs in a row with the phone, in the demo room, are done later by the user.
