# M7 spec: click an object, get its details

The project owner's request: the demo is a website that shows the live video and every object the camera detects. A click on an object opens a side panel that says what the object is and gives its specifications. For a person the panel shows what the team wrote about them and a small photo. When the detector has no name for the thing, the hosted API looks at the picture and searches the web.

This spec builds on M5 (the page, already written) and M6 (person folders). Where M7 and an older spec differ, M7 wins. Read `CLAUDE.md` first.

## What a click does

1. The page sends the click to the server (`POST /api/inspect`).
2. The server finds the **target**: the detected object under the click, or, when no box is there, the area around the click. It cuts a small photo of the target out of the clean camera frame (no boxes drawn) and answers at once.
3. The panel opens with the photo, the detector's label and confidence.
4. What happens next depends on the kind of target:

| Kind | Meaning | Panel content | API call |
| --- | --- | --- | --- |
| `person`, recognised | a `person` box or a face box that the face recognizer matched | name, role, card lines and story from that person's `about.md` | none |
| `person`, not recognised | a `person` box with no matched face | "Unknown person" and one line on how to add them | none |
| `object` | any other detected object | what it is, a short description, a list of specifications, sources when the web was used | one (`POST /api/describe`) |
| `area` | a click where the detector found nothing | the same as `object`; the API identifies the thing | one |

A picture of a person is never sent to the API because of a click on that person. A cut-out of an object can still have a person in the background; the prompt forbids describing them.

The work is three parts, each in the modules that own it. The parts are built at the same time by different people, so **the interfaces below are fixed**: build against them, and use fakes for the other parts in tests.

---

## Part A: `car_vision/assistant.py`

Finish M6 Part A as rewritten in `specs/M6-briefs-and-look.md` (briefs in the system prompt, the objects line). The helpers `load_briefs`, `build_system_prompt(notes, briefs)`, `objects_line`, `check_image` and `user_turn` are already in the file but not yet used. `ask` keeps its signature: the chat does not take pictures.

New:

```python
@dataclass(frozen=True)
class ObjectInfo:
    name: str                              # what the thing is, e.g. "Laptop" or "Office chair"
    summary: str                           # two or three sentences
    specs: tuple[tuple[str, str], ...]     # (name, value) pairs, at most six
    used_web: bool = False
    sources: tuple[str, ...] = ()

class Assistant:
    def describe(self, image: bytes, label: str | None = None) -> ObjectInfo: ...
```

- `image` is one JPEG, checked with `check_image` (`ValueError` before anything is sent). `label` is the detector's label or `None` for an area; a label longer than 60 characters or one that is not a string raises `ValueError`.
- One call, no memory: `describe` does not read or change the chat history, and the chat does not see it. It sends its own system prompt (a module constant, so it is the same for every call), one user turn made with `user_turn(text, image)`, and allows web search when `assistant.web_search` is on. It goes through the same `LLMClient.complete(system, turns, allow_web)`; the client and its request shape do not change.
- Its own lock, so a slow look-up does not block the chat: a second `describe` waits up to 30 seconds for the first, then raises `AssistantError` with a plain "still busy" message.
- The user text says what the detector thinks: `The object detector labelled this "laptop".` or `The object detector has no label for this. The visitor clicked the middle of the picture.`
- The system prompt tells the model: the picture is a small cut-out from the camera of a student robot car; say what the thing in the middle is, as exactly as the picture allows; the detector's label is a hint, not the truth; name a brand or model only when it can be read or is unmistakable; give up to six specifications a curious visitor would want (size, weight, material, power, typical price and the like), marked as typical when the exact model is not known; when it cannot tell what the thing is, or a brand or model is visible whose details it does not know, search the web if the tool is available; never invent; when it still cannot tell, say so with the name `Unknown object`; if the thing is a person, answer with the name `Person` and nothing about who they are or how they look; say nothing about people elsewhere in the picture; answer in English, in exactly this shape:

  ```
  NAME: <a few words>
  ABOUT: <two or three sentences>
  SPECS:
  - <name>: <value>
  - <name>: <value>
  ```

  Plain text is used on purpose: the API's JSON output mode cannot be combined with web search citations.
- `parse_object_info(text: str, label: str | None) -> tuple[str, str, tuple[tuple[str, str], ...]]` is a pure function and must be forgiving: keys in any letter case, `*` or `-` bullets, Markdown bold around keys, an `ABOUT` that runs over several lines, more than six specs (keep six), spec lines without a colon (skipped). When no `NAME` line is found, the name is the label with a capital first letter, or `Unknown object`, and the whole text is the summary.
- Refusal and `max_tokens` keep raising `AssistantError` as for `ask`.
- Tests (fakes only, no network): the system prompt for `describe` is identical between calls; the picture and the label line reach the client in the user turn; web search follows the setting; the chat history is untouched by `describe` and `describe` sends none of it; the parser cases above; bad pictures and bad labels are refused before the client is called; the busy case; sources and `used_web` are passed through; one request-shape test with a picture in `tests/test_assistant_wire.py`.

---

## Part B: `car_vision/faces.py`, `recognizer.py`, `enroll.py`

Do M6 Part B as written in `specs/M6-briefs-and-look.md`, with this `Person`:

```python
@dataclass(frozen=True)
class Person:
    person_id: str
    name: str
    role: str = ""
    lines: tuple[str, ...] = ()     # every bullet line of about.md, in order
    story: str = ""                 # the text that is left: no heading, no Role line, no bullets
    brief: str = ""                 # the whole about.md as written
```

- The card drawn on the video keeps showing the first three `lines`; the page's panel shows all of them.
- `story` keeps paragraph breaks (blank lines) and has no blank lines at its start or end. HTML comments are not part of `lines` or `story`.
- `Recognizer.person(person_id) -> Person | None` and `load_known_faces(settings) -> (people, matcher)` keep their names and meaning; the people now come from the folders under `recognizer.photos_dir`.
- A person with an `about.md` but no photos (not enrolled yet) is still loaded as a `Person`; they are simply never matched. The project owner will add the photos later, so this case is the normal one today and must give a clear note, not an error.
- You may delete the line `people_file: data/people.json` from `config.yaml` (the orchestrator allows this one change) and must update `data/people.example*` as M6 says. Do not delete `data/people.json` itself or anything under `data/people/`.

---

## Part C: `car_vision/pipeline.py`, `car_vision/app.py`, `car_vision/templates/index.html`

### `pipeline.py`

```python
@dataclass(frozen=True)
class Target:
    kind: str                    # "object", "person" or "area"
    label: str                   # the detector's label; "" for an area
    confidence: float            # 0.0 for an area
    track_id: int | None
    photo: bytes                 # JPEG cut from the clean frame
    person: Person | None = None # the recognised person, for kind "person"

class Pipeline:
    def target(self, x: float | None = None, y: float | None = None,
               track_id: int | None = None, label: str | None = None) -> Target | None: ...
```

- The loop keeps, under its lock, one consistent **view**: a clean copy of the newest frame (taken before anything is drawn on it) with the detections and faces of that same frame. `target` works on that view only.
- `x` and `y` are the click as fractions of the picture, 0 to 1 from the top-left corner.
- Finding the target, in this order:
  1. `track_id` given: the detection with that track id (and that `label`, when given). None left in view: return `None`.
  2. Otherwise the boxes that hold the point: detection boxes (only those with that `label`, when one is given) and face boxes. The smallest box wins, so a phone in someone's hand beats the person holding it. A face box means that person: use the `person` detection the face belongs to when there is one, else the face box itself.
  3. No box holds the point and no `label` was asked for: an `area` target, half the frame wide and half the frame high, centred on the click and moved to stay inside the frame.
  4. A `label` was asked for and nothing matches, or there is no frame yet: `None`.
- A `person` target is recognised when a known face has the same track id, or its box centre lies inside the person box. `Target.person` is then `recognizer.person(face.person_id)`.
- The photo: the box grown by 15 % on each side, at least 160 pixels wide and high where the frame allows, clipped to the frame, encoded with `app.jpeg_quality`. Never enlarged.
- `status()["objects"]` changes to one entry per detection, people first, then by label, then by track id:

  ```python
  {"track_id": 7, "label": "laptop", "confidence": 0.83,
   "box": [0.212, 0.401, 0.587, 0.932],   # x1, y1, x2, y2 as fractions of the frame
   "name": None}                          # the display name, for a recognised person
  ```

  The other status keys stay as they are.

### `app.py`

| Route | Behaviour |
| --- | --- |
| `POST /api/inspect` | Body: `{"x": 0.42, "y": 0.61}` from a click on the video, or `{"track_id": 7, "label": "laptop", "x": ..., "y": ...}` from the list (the point is the box centre, used when the entry has no track id). Numbers outside 0 to 1, a missing point without a track id, or wrong types: 400. `pipeline.target(...)` gives `None`: 404 with `{"error": "That is no longer in view."}`. |
| `POST /api/describe` | Body: `{"id": "<inspection id>"}`. Calls `assistant.describe(photo, label or None)`. An unknown or expired id: 404, asking to click the object again. A person target: 400. `AssistantError`: status 200 with `{"error": ...}`, as for `/api/ask`. `ValueError`: 400. |

`/api/inspect` answers:

```python
{"id": "kq3...",                       # random, from the secrets module
 "kind": "object", "label": "laptop", "confidence": 0.83, "track_id": 7,
 "photo": "data:image/jpeg;base64,...",
 "person": None,                       # or {"id", "name", "role", "lines", "story"}
 "describable": True,                  # False for every person target
 "info": None}                         # the cached answer, when there is one
```

`/api/describe` answers `{"name", "summary", "specs": [{"name", "value"}, ...], "used_web", "sources"}`.

- The server remembers the last 20 inspections (id, label, track id, photo) so that `/api/describe` needs only the id; the browser never sends a picture.
- Answers are cached for ten minutes by `(track_id, label)`, at most 50 of them, so a second click on the same object is instant and costs nothing. Targets without a track id, and areas, are not cached. Both stores are small classes with a lock, tested without Flask.
- `UnavailableAssistant` gets a `describe` that raises the same error as its `ask`.
- The server never logs a photo or a person's details.

### The page

The M5 page stays (video, banners, chat, status line), with these changes:

- **Click on the video.** The pointer shows that the picture can be clicked. The click position is worked out against the picture itself, not the element, so it stays right when the picture is letter-boxed.
- **In view.** A list of everything in `status.objects`, each entry a button: label, track number, confidence in percent; a recognised person shows their name and role instead. It replaces the separate person cards and updates only when something changed. Empty: a quiet line.
- **Details panel.** Opens at the side on a click, without covering the video; a close button and the Escape key close it. It shows the photo, a title (the person's name, the name from the API, or the label until the answer arrives; `Something the detector has no name for` for an area), and a small line with the detector's label, confidence and track number.
  - Recognised person: role, all card lines, the story as paragraphs.
  - Unrecognised person: `Unknown person` and `Add a folder with photos and an about.md for them in data/people to show their details here.`
  - Object or area: `Looking it up ...` until `/api/describe` answers, then the summary, the specifications as a two-column list, and for a web answer the mark `From the web` with its sources. An error shows in the error colour, and the photo and label stay.
  - When the user clicks something else before the answer arrives, the late answer is ignored.
- **Sources**, in the panel and in the chat: at most three links, shown by host name, then `+N more`.
- One small line near the list: `A click on an object sends a small cut-out of the picture to the assistant's API. A click on a person sends nothing.`
- The rules of M5 hold: server text goes in with `textContent`, never as HTML; links only for `http:` and `https:` URLs; no outside resources. The photo is shown only when its address starts with `data:image/jpeg;base64,`.

### Look and feel (added by the project owner; overrides M5's "one screen, light text on dark")

The owner does not want the page to look like a generic AI tool or a dark dashboard. The reference is a modern phone maker's product page: calm, spacious, alive, with translucent "liquid glass" surfaces and smooth scrolling. Take the manner, not the brand: no logos, product pictures, names or copied text from any company.

- **A scrolling page, not a dashboard.** Sections from top to bottom: a hero (a very large, tight headline such as `Robot Car Vision`, one line under it such as `See what the car sees.`, two pill buttons that scroll to the live view and to the chat), the live view, the chat, a short footer with the privacy line. `scroll-behavior: smooth`; anchors land below the navigation bar.
- **Navigation bar.** Fixed at the top, translucent glass, with the project name, links to the sections and a small live status pill (a pulsing dot, source, FPS, device; `Reconnecting` or `Program not running` when that is the state). It becomes more compact once the page is scrolled.
- **Type.** No downloaded fonts. Use the system stack (`"Segoe UI Variable Display", "Segoe UI", system-ui, -apple-system, "SF Pro Display", "Helvetica Neue", Arial, sans-serif`), big sizes with `clamp()`, tight letter-spacing on headlines, a relaxed weight for body text. Few colours: near-black text, one mid grey, one blue accent for actions, red only for the warning. No emoji, no gradient text, no neon.
- **Colour.** Light by default (page around `#f5f5f7`, white surfaces), dark when the system asks for it (`prefers-color-scheme`). Colours are CSS variables on `:root`. Text must pass normal contrast on both.
- **Liquid glass.** The navigation bar, the In view pills, the details panel, the chat card and the banners over the video are glass: a translucent fill, `backdrop-filter: blur(...) saturate(...)` (with the `-webkit-` form), a hairline light border, a soft inner highlight along the top edge, large radii (20 to 32 px), soft wide shadows. Behind the page, two or three large soft colour shapes give the glass something to bend. They move with the scroll position (`animation-timeline: scroll(root)`), never with time: shapes that drifted all the time made the browser redraw every glass surface on every frame, and on the demo laptop that cut the models from 22 to about 5 frames a second. Where `backdrop-filter` is missing, the surfaces fall back to an almost opaque fill.
- **The live view is the product shot.** The video sits in a large rounded frame with a deep soft shadow, as wide as the layout allows, keeping its own shape. Greeting and warning banners are glass capsules over its top edge. The In view list is a row of glass pills under it (wrapping, not a table). The details panel is a glass sheet that slides in from the right; from 1280 px wide the video makes room for it instead of being covered, on narrower windows it lies over the page.
- **Alive.** Sections fade and rise into place as they scroll into view (`IntersectionObserver`); pills and chat messages ease in when they appear; buttons and pills react to hover and press with a short spring-like transform; the panel slides with an ease-out curve; the hero headline moves slightly slower than the page. Numbers in the status pill change without the pill jumping in width (`font-variant-numeric: tabular-nums`).
- **Keep it cheap.** The graphics card is busy with the models. Animate only `transform` and `opacity`; no blur animation; keep glass that lies over the moving video small (the banners); no animation loop in JavaScript beyond what the page already polls. With `prefers-reduced-motion: reduce`, the drifting shapes, the parallax and the reveals are off.
- **Still true:** one HTML file with inline CSS and JavaScript, no outside resources of any kind, `textContent` only, keyboard use works (focus rings visible, Escape closes the panel, Enter sends), and everything stays readable from the back of a room at 1280 x 720 and larger.

### Tests

- `Pipeline.target`: by track id; by point; smallest box wins; a face box gives the person; recognised and unrecognised people; an area (centred, and pushed back inside at an edge); a label that is no longer there; no frame yet; the photo decodes, is cut from the clean frame (no box colour in it) and is never larger than the frame. The new `objects` shape and order.
- Routes with fakes: both routes' good cases; each 400 and 404 above; a person target is never passed to `assistant.describe`; the cache gives the second click its answer without a second call; an `AssistantError` from `describe`; the stores' limits and expiry with an injected clock.
- The template: the list, the panel and the privacy line are present; still no `innerHTML` and no outside resources.

---

## Limits to state plainly in the README (the orchestrator writes this)

- The detector knows 80 everyday kinds of object. Anything else has no box; a click on it asks the API what it is.
- Specifications come from a language model looking at a small picture. For a generic object they are typical values, and they can be wrong.
- Each look-up is one paid API call and takes a few seconds; repeated clicks on the same tracked object are answered from memory.
