# M13 spec: find anything by name, and remember what was seen

Three features the project owner chose:

1. **Detect anything by typing its name.** Today the detector knows 80 kinds of object. The visitor types `power bank` and the camera finds it live.
2. **Find it.** "Find my keys", typed or spoken: the page highlights the thing and says where it is.
3. **Memory.** The program remembers what it saw and when: "Where did you last see the bottle?", "Who came by today?", and a timeline on the page.

Read `CLAUDE.md` first, then `specs/M7-click-to-inspect.md` (page rules), `specs/M10-full-view-size-brush.md` and `specs/M11-voice.md`.

## What was measured before this was written (owner's laptop, GTX 1660 Ti, real clip)

The open-vocabulary model is **YOLOE** from the Ultralytics package already installed (8.4.216). Its classes are set from text at run time.

| | `yoloe-26n-seg.pt` | `yoloe-26s-seg.pt` | today's `yolo26n.pt` |
| --- | --- | --- | --- |
| One frame, `predict` | 57 to 69 ms | 62 ms | about 30 ms |
| One frame, `track` (ByteTrack) | 85 to 89 ms | 82 ms | about 30 ms |
| Found in the room clip, asked for television, router, couch, curtain, table, cable | couch, television | couch, 2 televisions, curtain, router | (80 fixed classes) |
| Asked for `wifi router` | nothing | found | cannot |
| Changing the names (`set_classes`) | about 1 s | about 1 s | |
| Peak video memory | about 115 to 270 MB | about 265 MB | |

So: the small model costs the same time as the nano one and finds much more; it is the one to use. It is two to three times slower than the everyday detector, so it does **not** replace it: it runs beside it, only while the visitor is looking for something, and only on every few frames. Changing the names while tracking worked and kept the track numbers going.

Facts about the calls, checked on this machine:

```python
from ultralytics import YOLOE
model = YOLOE("yoloe-26s-seg.pt")                      # 11 s the first time it is loaded
model.set_classes(names, model.get_text_pe(names))     # first call about 7 s: loads the text encoder
results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, conf=0.25,
                      half=False, device=0, verbose=False)
# results[0].boxes (xyxy, conf, cls, id), results[0].names: {index: typed name}, results[0].masks
```

The first `set_classes` downloads the text encoder `mobileclip2_b.ts` (242 MB) **into the current working directory** and the package installs `clip` and `ftfy` by itself. Both are already on the owner's laptop, in `models/`. The code must make the weights and the encoder land in `detector.models_dir` whatever the working directory is, and `requirements.txt` must name the two packages.

## The parts

Five parts. A, B and C are independent; D is built against A's and B's interfaces at the same time, with fakes; E (routes and page) follows. **The interfaces below are fixed.**

### New settings (already added to `config.yaml` by the orchestrator)

```yaml
detector:
  open_model: yoloe-26s-seg.pt   # finds things by a typed name
  open_confidence: 0.25
  open_every_n_frames: 3         # the named search runs on every 3rd frame
memory:
  file: data/memory.sqlite3      # what the camera saw and when (stays on this laptop)
  keep_days: 7
  min_seconds: 1.0               # a thing must stay in view this long to be remembered
  pictures: true                 # keep a small picture of remembered objects (never of people)
```

---

## Part A: `car_vision/detector.py` (`ultralytics` stays in this module only)

```python
class OpenDetector:
    """Finds things by a typed name. Heavy parts are loaded on demand, never at start-up."""
    def __init__(self, settings: dict, model=None) -> None: ...       # reads settings, loads nothing
    @property
    def loaded(self) -> bool: ...
    def load(self) -> None: ...                                        # the model; slow, call off the video loop
    @property
    def names(self) -> tuple[str, ...]: ...
    def set_names(self, names: Sequence[str]) -> None: ...             # about 1 s (7 s the first time)
    def detect(self, frame: np.ndarray) -> list[Detection]: ...       # label = the typed name
    def reset_tracks(self) -> None: ...
```

- `detect` returns `Detection(label=<typed name>, confidence, box, track_id)` using the model's own ByteTrack state. With no names set it returns `[]` without touching the model. Never half precision. Device from `detector.device` like `Detector`.
- `set_names` with the same names as now does nothing. With new names it sets them and starts the tracks again. An empty list clears the names.
- `load`, `set_names` and `detect` are never called at the same time by the pipeline; `OpenDetector` does not need its own lock, but say so in its docstring.
- Files: the weights and `mobileclip2_b.ts` are found in and downloaded to `detector.models_dir`, wherever the program was started from, and the working directory is the same after the call as before it.
- Masks are not used; if the result carries them they are ignored. If you find a supported way to skip computing them that is measurably faster, use it and say what you measured (with a fake you cannot measure; then leave it).
- Errors: a model or encoder that cannot be loaded (no internet on the first run) raises a `DetectorError` (new, a plain message naming what could not be loaded and that the first use needs the internet); nothing else leaks out.
- `requirements.txt`: add the `clip` package the way Ultralytics installs it and `ftfy`, with a comment.
- Tests with a fake model (no weights, no network): settings validation; nothing loaded at construction; `detect` with no names; names set and changed; labels are the typed names; track ids passed through; tracks start again on new names; the working directory restored; load failure gives `DetectorError`.

## Part B: `car_vision/memory.py` (new; standard library only) and two methods in `state.py`

```python
@dataclass(frozen=True)
class Sighting:
    id: int
    kind: str              # "object" or "person"
    label: str             # the detector's label or the typed name; "person" for people
    name: str | None       # the recognised person's display name, else None
    first_seen: float      # seconds since 1970, as time.time() gives
    last_seen: float
    place: str             # "left", "middle" or "right" of the picture when last seen
    confidence: float
    has_picture: bool

class SightingLog:
    def __init__(self, path: Path, keep_days: float = 7, min_seconds: float = 1.0) -> None: ...
    def seen(self, key: Hashable, kind: str, label: str, name: str | None, place: str,
             confidence: float, now: float, picture: bytes | None = None) -> None: ...
    def recent(self, limit: int = 50, text: str | None = None) -> list[Sighting]: ...
    def picture(self, sighting_id: int) -> bytes | None: ...
    def summary(self, now: float, limit: int = 30) -> list[str]: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...
```

- Stored in one SQLite file (`sqlite3` from the standard library), created with its folder when missing. Safe to use from the video loop thread and the web threads at once.
- `seen` is called again and again while a thing stays in view, with the same `key` (the pipeline uses the track epoch, the track number and the label). The first call starts a candidate; once the thing has been in view for `min_seconds` it becomes a row; later calls extend `last_seen`, the place and the best confidence. Writing to the file happens at most every 2 seconds per thing, and when the thing has not been seen for 3 seconds its row is final (a later sighting with a new key is a new row). A candidate that vanishes before `min_seconds` leaves nothing behind: that is what keeps a flickering wrong label out of the memory.
- A person who is recognised part-way through a sighting gets their name on the row from then on. `picture` is stored only for `kind == "object"` (a small JPEG the pipeline hands over, at most 40 kB; larger ones are refused quietly); a picture given for a person is ignored, always.
- Rows older than `keep_days` are deleted at start-up and once an hour. `clear` empties everything.
- `recent`: newest `last_seen` first; `text` filters on label and name, case-insensitively.
- `summary` gives plain lines for the assistant, newest first, one per distinct thing (the latest sighting of each label or person): `laptop: last seen 3 minutes ago on the left, in view for 2 minutes` / `Sara Ahmed (person): last seen 1 hour ago in the middle, in view for 40 seconds`. Times in words (`just now`, `N seconds`, `N minutes`, `N hours`, `yesterday`, `N days ago`). An unnamed person is `an unrecognised person`.
- A broken or locked file must never stop the video: every method logs once and carries on (reads give empty results).
- `state.py`: `WorldState.set_memory(lines: Sequence[str]) -> None` and `WorldState.memory() -> tuple[str, ...]`, thread-safe like the rest. Nothing else in `state.py` changes.
- Tests with a temporary file and injected times: the `min_seconds` rule; extending; a gap makes a new row; the person name arriving late; no picture for people; the 40 kB limit; `recent` order and filter; `summary` wording for each time range; `keep_days`; `clear`; two threads writing and reading; a file that cannot be opened.

## Part C: `car_vision/assistant.py`

- The lines the camera puts in front of each chat question (people in view, objects in view) gain the memory: when the `WorldState` holds memory lines, they are added under `What the camera remembers (newest first):`. At most 30 lines. Nothing changes in the system prompt except one added sentence in the chat prompt: the memory lines come from the camera's own log, they may be used to answer questions about what was seen and when, and anything not in them was not recorded.
- `describe`, `follow_up` and `describe_scene` do not get the memory.
- Tests: the lines appear in order and capped; absent with no memory or no state; the system prompt is still identical between questions.

## Part D: `car_vision/pipeline.py`, `car_vision/draw.py`

```python
class Pipeline:
    def search(self, names: Sequence[str]) -> None: ...      # [] stops looking; ValueError for bad names
    def sightings(self) -> SightingLog | None: ...           # the memory, for the routes
```

- Names: 1 to 6 of them, each 1 to 40 characters after trimming, letters, digits, spaces, hyphens and apostrophes only (any script), compared without regard to case, duplicates dropped. Anything else: `ValueError` with a plain message.
- `search` returns at once. Loading the model, setting the names and everything else slow happens in one background thread; a newer `search` replaces an older one that has not finished ("the newest wins", like the phone-picture rounds). While that thread works on the model, the loop simply skips the named detection.
- While names are set and ready, the loop runs `OpenDetector.detect` on every `open_every_n_frames`-th frame and keeps those results on screen in between (at most 1 second without being seen again). They are added to the frame's detections with these rules: a named result that overlaps an everyday detection by more than 0.6 (intersection over union) replaces it (the visitor's word is the more useful one); named track numbers are kept apart from the everyday ones (add 100000); the track epoch goes up when the names change.
- `status()` gains `"search"`: `{"names": [...], "state": "off" | "loading" | "ready" | "failed", "note": "<plain reason when failed>", "found": [{"name": "wifi router", "count": 2, "place": "right"}], "missing": ["power bank"]}`, and each entry of `status()["objects"]` gains `"found": true` when it came from the search. `place` is `left`, `middle` or `right`: which third of the picture the box centre is in (for several of one name, the place of the most confident one).
- `draw.py`: `draw_detections(frame, detections, wanted=())`: boxes whose track number is in `wanted` are drawn with a thicker line in one fixed bright colour, so the thing being looked for stands out.
- Memory: when `memory.file` is set, the pipeline owns a `SightingLog`, calls `seen` for every tracked detection of each processed frame (people with the recognised name when there is one; objects with a cut-out of at most 160 pixels on the longer side when `memory.pictures` is on), and puts `summary()` into the `WorldState` every 5 seconds. Untracked detections are not remembered. A change of camera or frame size gives new keys through the epoch.
- Start-up never loads the open model and never needs the internet for it. With recognition off or the memory file unusable, everything else works.
- Tests with fakes for the open detector and a temporary memory file: the name rules; newest search wins; the loop skips while loading; every n-th frame; results kept in between and dropped after a second; the overlap rule; ids apart; the epoch; each `search` state in the status including `failed` with the detector's message; `found`, `missing`, `place`; the highlight ids reach `draw`; sightings recorded for tracked things only, people without pictures; the memory lines reach the `WorldState`.

## Part E: `car_vision/app.py`, `templates/index.html` (after A to D)

| Route | Behaviour |
| --- | --- |
| `POST /api/search` | Body `{"names": ["power bank", "keys"]}` or `{"text": "power bank, keys"}` (split on commas and the word `and`). 400 with the pipeline's message for bad names. Answers `{"ok": true, "names": [...]}`. |
| `DELETE /api/search` | Stops looking. |
| `GET /api/memory?limit=50&text=bottle` | `{"sightings": [{"id", "kind", "label", "name", "first_seen", "last_seen", "place", "confidence", "has_picture"}]}`. |
| `GET /api/memory/<id>/picture` | The small JPEG; 404 when there is none. |
| `DELETE /api/memory` | Empties the memory. |

The page:

- **Look for.** A field over the bottom-left of the video frame, also in the full view: `Look for ...` and a **Find** button; Enter works. While looking, the names show as small chips beside it with a **Stop** button, and each chip says what the status says: `loading the finder ...` the first time, `found on the right` / `found (2)`, or `not in view`. Found things stand out in the video (the server draws them) and their pills in the In view list are marked.
- **Find by words.** A sentence typed in the chat or spoken into any microphone that is only a request to find something (`find ...`, `look for ...`, `where is/are (the/my) ...`, `can you see (the/my/a) ...`) starts a search for that thing instead of going to the assistant, and the page answers in the chat or the caption from the status: `I can see the wifi router, on the right.` / `I can't see a power bank right now. I'll keep looking.` Questions about the past (`where did you last see ...`, `when did ...`) go to the assistant as before; they are answered from the memory.
- **Voice** follows the owner's rule: the page speaks only in answer to something the visitor just did. The answer to a find request is spoken (when voice is on). If the thing was not in view and appears later while the search is still on, the page says so **once** per name per search (`I can see the power bank now, on the left.`): the visitor asked to be told. Nothing else about the search is ever spoken, and only the page that started the search speaks.
- **Timeline.** A section after People (a link in the navigation bar): what the camera remembers, newest first, as a list of rows with the small picture (objects only), the name, `seen 3 minutes ago · in view for 2 minutes · on the left`, and a filter field. It loads when it scrolls into view and refreshes every 10 seconds while visible, never otherwise. A **Clear memory** button asks once in the page before it acts.
- Privacy line gains: `The camera's memory (what was seen and when, with small pictures of objects, never of people) stays on this laptop. A chat question sends its text lines, without pictures, to the assistant.`
- All page rules of M7, M8, M10 and M11 hold (text only via `textContent`, no outside resources, no endless animation, no blur over the video, inert page behind the full view, the picture address checked before it is shown).

## What to tell the reader plainly in the README (the orchestrator writes this)

- Looking for something by name uses a second, slower model beside the everyday one; while it runs the video is a few frames a second slower. It is off until asked for and loads on first use (about 15 seconds; the very first time it needs the internet to fetch 250 MB).
- It finds what looks like the words. It can be wrong, and it cannot tell your keys from someone else's.
- The memory is a local file. It records labels, names of enrolled people, times and small pictures of objects.
