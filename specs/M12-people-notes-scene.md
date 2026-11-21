# M12 spec: add a person from the page, edit the notes on the page, describe the scene

Three features the project owner chose:

1. **Add a person from the page.** Type a name, a role and a few lines, stand in front of the camera, and the page takes the photos and enrols the person. No folders or files made by hand. (Today the system knows one person from two photos, because making folders is tedious.)
2. **Edit the notes on the page.** `data/knowledge.md` is what the chat knows about the project; it still holds placeholder lines.
3. **Describe the scene.** One button: the assistant looks at the whole picture and says what it sees in two or three sentences.

Read `CLAUDE.md` first (its rules on face photos, the key and the module boundaries all apply), then `specs/M6-briefs-and-look.md` ("The person folder") and `specs/M7-click-to-inspect.md`.

Three parts. Parts A and B are built at the same time; Part C (routes and page) follows and is built against the interfaces below, which are fixed.

---

## Part A: the assistant (`car_vision/assistant.py`, its tests)

```python
class Assistant:
    def describe_scene(self, image: bytes) -> Answer: ...
    def reload(self) -> None: ...
```

- `describe_scene`: one call with no memory, like `describe`. `image` is one JPEG checked with `check_image`. Its own system prompt (a module constant): the picture is the whole view of a student robot car's camera; say what the scene is and what stands out, in two or three plain sentences a listener can follow (the answer may be read aloud); the lines from the camera that come with it (who the face recogniser names, which objects the detector reports) are hints; a recognised person may be mentioned by the name the camera line gives; never guess who anyone is, and say nothing about how people look beyond that a person is there and what they are doing in general terms; no lists, no Markdown; do not invent; English. The user turn is the picture followed by the same camera lines `ask` builds (people line, objects line) and the sentence `Describe the scene.` No web search for this call. It shares `describe`'s lock and busy message. The chat history is neither read nor changed.
- `reload`: reads the notes file and the person briefs again and builds the chat's system prompt from them, exactly as at start-up, then starts a fresh conversation (a changed system prompt cannot continue an old one). It waits for a question in progress like `reset` does. A notes file that cannot be read raises `AssistantError` and leaves the old prompt and the old conversation in place.
- Tests with the fake client: the exact turn and prompt for `describe_scene`, with and without a `WorldState`; no web tool; history untouched; a bad picture refused before the client is called; `reload` picks up changed notes and a new brief, clears the history, and keeps the old state when the notes file is unreadable; one wire test for `describe_scene`.

## Part B: people and photos on the vision side (new `car_vision/people.py`, `recognizer.py`, `enroll.py`, `faces.py` if needed, `pipeline.py`, their tests)

### `car_vision/people.py` (standard library, plus `faces.py`)

```python
@dataclass(frozen=True)
class PersonEntry:
    person: Person          # as faces.py reads it from about.md
    photos: int             # jpg/png files in the folder

class PeopleBook:
    def __init__(self, photos_dir: Path, removed_dir: Path) -> None: ...
    def all(self) -> list[PersonEntry]: ...                                  # sorted by id
    def save(self, name: str, role: str = "", lines: Sequence[str] = (), story: str = "",
             person_id: str | None = None) -> Person: ...
    def remove(self, person_id: str) -> None: ...

def make_id(name: str, taken: Collection[str]) -> str: ...
def about_text(name: str, role: str, lines: Sequence[str], story: str) -> str: ...
```

- `save` with no `person_id` makes a new person: the id comes from `make_id` (small letters, digits and underscores from the name; other characters dropped or turned into `_`; a name with no usable characters, such as one written in Arabic, gives `person`; a taken id gets `_2`, `_3`, ...). With a `person_id` it rewrites that person's `about.md` and leaves the photos alone; an unknown id is a `KeyError`.
- `about_text` writes the file `faces.parse_about` reads: parsing its output gives back the same name, role, lines and story. Text is cleaned so it cannot change the file's meaning: no line breaks in the name or role, a line that starts with `#`, `-`, `*` or `Role:` inside the story is kept as story (escape or indent it as needed and prove it with the round-trip test), HTML comment marks removed.
- Limits, each a `ValueError` with a plain message: name 1 to 80 characters; role up to 80; at most 12 lines of up to 120 characters; story up to 4000 characters; at most 50 people.
- `remove` does not delete: it moves the person's folder to `removed_dir/<id>-<YYYYmmdd-HHMMSS>/` (git-ignored; the orchestrator adds the line). Unknown id: `KeyError`.
- Every path is built from a validated id inside `photos_dir`; no input can reach outside it.

### Enrolling without a restart

- `Recognizer.reload_known(people, matcher) -> None` swaps the known people and the matcher in one step, safely while `recognize` runs on the loop thread.
- `Pipeline.reload_people() -> list[str]`: runs enrolment for every folder with the face model that is already loaded (never loading a second one), saves `embeddings.npz`, swaps the result into the recogniser, forgets remembered faces, and returns the notes enrolment produced (people without photos, ignored files, ...). It runs off the loop thread and must not stop the video for more than a moment: decide how the loop and enrolment share the face model (a lock around its use is acceptable) and say what you chose. Recognition off (`self._recognizer is None`): `RuntimeError` with the note the page already shows.
- `Pipeline.enrolled_ids() -> set[str]`.

### Taking photos from the live camera

```python
class Pipeline:
    def start_photos(self, person_id: str, count: int = 5) -> None: ...
    def photo_progress(self) -> dict | None: ...
    def cancel_photos(self) -> None: ...
    def scene_photo(self, max_side: int = 960) -> bytes | None: ...
```

- `start_photos` begins a session for an existing person folder (unknown id: `KeyError`; recognition off, another session running, or `count` outside 3 to 10: `ValueError` with a plain message). The loop then looks at the clean frame on its face-check frames and takes a photo when all of these hold: exactly one face in view; the face at least 80 pixels wide; the face sharp enough (variance of the Laplacian of the face cut-out above a threshold you choose and justify with a test on a sharp and a blurred generated image); at least 1 second since the last photo. The photo is the clean frame at its current size, saved as the next free `photo_<n>.jpg` in the person's folder, so photos can be added to someone who already has some.
- `photo_progress()` is `None` when no session ran yet, else `{"person_id", "taken", "wanted", "state", "hint", "notes"}`: `state` is `"taking"`, `"enrolling"`, `"done"` or `"failed"`; `hint` says what the person should do right now (`Stand in front of the camera`, `Only one person in the picture, please`, `Come closer`, `Hold still`, or empty); `notes` are enrolment's notes when done.
- When `count` photos are taken, the session enrols (`reload_people`) and ends as `done`. After 60 seconds: with 3 or more photos it enrols and ends `done` with a note that fewer were taken; with fewer it ends `failed` (the photos taken stay in the folder) and says how many were clear. `cancel_photos` ends it as `failed` with the hint `Cancelled`.
- A session survives a change of picture size and a phone that drops out and returns (the timeout still runs); `use_phone` cancels it.
- Nothing about a face, a photo or a person's details is logged beyond the person id and counts.
- `scene_photo`: the newest clean frame as JPEG, shrunk so its longer side is at most `max_side`, never enlarged; `None` before the first frame or while the view is cleared.

### Tests (fakes only: no real face model, no camera)

`make_id` (plain, spaces, capitals, Arabic, taken ids, digits first); `about_text` round trip including awkward stories and lines; every limit; `save` new and update; `remove` moves and a second person with the same id later does not collide; path safety with hostile ids; `reload_known` under a running `recognize`; `reload_people` with a fake face model (new person recognised afterwards, removed person no longer); a photo session for each rule (no face, two faces, small, blurred, too soon), the count reached, the timeout with 3 and with 2, cancel, a second session refused, `use_phone` cancelling; `scene_photo` sizes.

## Part C: routes and page (`car_vision/app.py`, `templates/index.html`, their tests): after A and B

| Route | Behaviour |
| --- | --- |
| `GET /api/people` | `[{"id", "name", "role", "lines", "story", "photos", "enrolled"}]` and whether recognition is on. |
| `POST /api/people` | Body `{"name", "role", "lines", "story"}`: a new person. 400 with the plain message for bad input. Then `assistant.reload()`. |
| `PUT /api/people/<id>` | The same fields: rewrite that person's `about.md`, then `assistant.reload()`. 404 for an unknown id. |
| `DELETE /api/people/<id>` | `PeopleBook.remove`, then `pipeline.reload_people()` and `assistant.reload()`. |
| `POST /api/people/<id>/photos` | Body `{"count": 5}` (optional): `pipeline.start_photos`. |
| `GET /api/people/photos` | `pipeline.photo_progress()`. When a session has just ended `done`, the server calls `assistant.reload()` once. |
| `DELETE /api/people/photos` | `pipeline.cancel_photos()`. |
| `GET /api/notes` | `{"text": <data/knowledge.md>}`. |
| `PUT /api/notes` | Body `{"text"}` up to 50 000 characters: keeps the old file as `knowledge.md.bak`, writes the new one (UTF-8, atomically), then `assistant.reload()`. |
| `POST /api/scene` | `assistant.describe_scene(pipeline.scene_photo())`; no picture yet: a plain error. Answers `{"text"}`; `AssistantError` as elsewhere. |

The page, in the manner of `M7` "Look and feel":

- A **People** section after the chat (a link in the navigation bar): one glass card per person (name, role, number of photos, `Recognised` or `Not enrolled yet`), with **Edit**, **Take photos** and **Remove**; and an **Add a person** button. The form has Name, Role, Card lines (one per line) and About (free text). **Take photos** shows the live progress under the form: the hint in large text, `3 of 5`, and a Cancel button; the visitor watches the video above (a small copy of the hint also appears as a capsule over the video, so it is visible in the full view). Remove asks once before it acts and says the folder is moved aside, not deleted.
- One line under the heading: `Take photos only with the person's permission. Photos and details stay on this laptop.`
- A **Notes** section after People: a large text area with the notes, **Save**, and a result line; a quiet warning when the text still contains `[REPLACE]`.
- A **Describe the scene** button on the video frame (beside the full-view button, also in the full view). The answer appears in the caption capsule over the video and in the main chat, and is spoken when voice is on. While it works the button shows it. The privacy line gains: `Describe the scene sends the whole picture.`
- The page rules of M7, M8 and M10 hold. No picture ever comes from the browser.
