# M3 spec: face recognition and info card

Recognise the people the team has enrolled and show their info card beside their face. No model is trained: enrolment turns photos into saved embeddings once, and the live view compares each face against them.

Read `CLAUDE.md` first. `car_vision/state.py`, `car_vision/config.py`, `car_vision/stream.py`, `car_vision/detector.py`, `car_vision/draw.py` and `config.yaml` exist and are fixed, except that you add functions to `draw.py` as listed below.

## Files

| File | What it holds |
| --- | --- |
| `car_vision/faces.py` | Pure logic: people file, embedding store, matching. Imports only `numpy`, `json`, the standard library and `car_vision`. |
| `car_vision/recognizer.py` | Finds and encodes faces with InsightFace, names them with `faces.py`. |
| `car_vision/enroll.py` | Command that turns photo folders into `data/embeddings.npz`. |
| `car_vision/draw.py` | Add `draw_faces` and `draw_info_card`. Do not change what is there. |
| `data/people.example.json` | A two-person example with made-up names. |
| `tests/test_faces.py`, `tests/test_recognizer.py`, `tests/test_enroll.py` | Unit tests with made-up embeddings and a fake face model. |

Only `recognizer.py` and `enroll.py` may import `insightface` or `onnxruntime`, and they share one loader function that lives in `recognizer.py`.

## Data formats

`data/people.json` is written by hand, so it must be forgiving and its errors must be readable:

```json
{
  "sara": {
    "name": "Sara Ahmed",
    "role": "Team lead",
    "lines": ["Built the car chassis", "Fourth-year student"]
  }
}
```

The key is the person id and the name of their photo folder, `data/people/sara/`. `name` is required; `role` defaults to an empty string and `lines` to an empty list; at most 3 lines are shown. Names and card text are English letters for now.

`data/embeddings.npz` holds two arrays: `ids` (one person id per row) and `embeddings` (one L2-normalised face vector per row, float32). Several rows per person, one per usable photo.

## `car_vision/faces.py`

```python
@dataclass(frozen=True)
class Person:
    person_id: str
    name: str
    role: str = ""
    lines: tuple[str, ...] = ()

def load_people(path: Path) -> dict[str, Person]: ...
def save_embeddings(path: Path, ids: list[str], embeddings: np.ndarray) -> None: ...
def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]: ...

class FaceMatcher:
    def __init__(self, ids: list[str], embeddings: np.ndarray, threshold: float) -> None: ...
    def match(self, embedding: np.ndarray) -> tuple[str | None, float]: ...
```

- `load_people` raises `ConfigError` with the file name and the reason for: missing file, invalid JSON, a top level that is not an object, an entry without `name`, wrong value types.
- `FaceMatcher.match` L2-normalises the input, takes the cosine similarity to every stored row, and scores each person by their best row. It returns `(person_id, score)` when the best score is at least `threshold`, otherwise `(None, best_score)`. With no stored rows it returns `(None, 0.0)`. A zero vector returns `(None, 0.0)` and raises nothing.
- `load_embeddings` raises `ConfigError` telling the user to run the enroll command when the file is missing, and when the two arrays disagree in length.

```python
class FaceMemory:
    """Keeps the last recognised faces on screen between face checks."""
    def __init__(self, max_age_seconds: float = 1.0) -> None: ...
    def remember(self, faces: Sequence[RecognizedFace], detections: Sequence[Detection], now: float) -> None: ...
    def current(self, detections: Sequence[Detection], now: float) -> list[RecognizedFace]: ...
```

Faces are checked only every few frames, so `FaceMemory` fills the frames in between. `remember` stores the faces from a face check together with the box of each face's person track at that moment. `current` returns them for a later frame: a face with a `track_id` whose track is still among `detections` is returned with its box shifted by the same offset the person box has moved, and is dropped as soon as that track is gone; a face without a `track_id` is returned unchanged until it is older than `max_age_seconds`, then dropped. Pure logic, no clock of its own: the caller passes `now`. The live view here and the full program in M5 both use it.

## `car_vision/recognizer.py`

```python
def load_face_model(settings: dict): ...      # InsightFace FaceAnalysis, detection + recognition only

class Recognizer:
    def __init__(self, settings: dict, face_model=None,
                 people: dict[str, Person] | None = None,
                 matcher: FaceMatcher | None = None) -> None: ...
    def recognize(self, frame: np.ndarray,
                  detections: Sequence[Detection] = ()) -> list[RecognizedFace]: ...
    def person(self, person_id: str) -> Person | None: ...

def main(argv: list[str] | None = None) -> int: ...
```

- `settings` is the `recognizer:` section. Read every key with `require`.
- `load_face_model` loads only the detection and recognition models of the pack (not landmarks, gender or age), stores model files under `models_dir`, and uses the graphics card when `device` allows it. ONNX Runtime needs the CUDA libraries that PyTorch ships: call `onnxruntime.preload_dlls()` before creating the model, and confirm in your report which execution provider the loaded models actually use. If the graphics card provider cannot load, fall back to the processor with a logged warning rather than crashing.
- Verify the InsightFace 2.1 API you use (`FaceAnalysis`, `prepare`, `get`, the face object's `bbox` and normalised embedding field) against the installed package, and say what you checked.
- `recognize` finds the faces in the frame, skips any whose box is smaller than `min_face_pixels` on its shorter side, matches each, and returns one `RecognizedFace` per face: `name` is the person's display name or `UNKNOWN`, `similarity` the score, `box` four ints clipped to the frame, `person_id` or `None`.
- `track_id`: when `detections` are given, a face gets the `track_id` of the `person` detection whose box contains the centre of the face box (the smallest such box when several do), otherwise `None`.
- When two faces in one frame match the same person, only the higher score keeps the name; the other becomes `UNKNOWN`.
- When the people file lists someone with no embeddings, or the embeddings name an id missing from the people file, log one warning at start-up and carry on.

`main` shows the live view: `FrameSource`, the `Detector` on every new frame, the `Recognizer` on every `every_n_frames`-th frame, both written into one `WorldState`. Between face checks, show the faces from `FaceMemory.current`, and write those into the `WorldState` too. Draw detections, faces, and the info card of each known face. Print the similarity scores on screen beside each name so the threshold can be tuned. Same arguments as the detector: `--source`, `--seconds`, `--no-window`. In `--no-window` mode print one line per second with the FPS and the names and scores seen.

## `car_vision/enroll.py`

```python
def enroll(settings: dict, face_model=None) -> dict[str, int]: ...   # person id -> photos used
def main(argv: list[str] | None = None) -> int: ...
```

- For every person in the people file, read each image in `photos_dir/<person_id>/` (jpg, jpeg, png, any letter case).
- A photo is used only when exactly one face is found in it; with several faces, use the largest one only if it is at least twice the area of the next, otherwise skip the photo. Print one line per skipped photo with the reason: unreadable, no face, several faces.
- Save all embeddings with `save_embeddings`. Print a table: person, photos found, photos used. A person with fewer than 3 used photos gets a warning line; a person with none is reported as not enrolled. Folders in `photos_dir` that match no id in the people file are listed as ignored.
- Exit code 1 when nobody was enrolled, 0 otherwise.
- Never print or log an embedding.

## `car_vision/draw.py` additions

```python
def draw_faces(frame, faces: Iterable[RecognizedFace], show_scores: bool = False) -> None: ...
def draw_info_card(frame, face: RecognizedFace, person: Person) -> None: ...
```

A known face gets a green box and its name; an unknown face a grey box and `Unknown`. The info card is a filled panel beside the face with the name, the role and up to 3 lines; it goes to the right of the face, or to the left when there is no room, and is always kept fully inside the frame. Long lines are cut with `...` to fit the card.

## Tests

All pass with no camera, no network, no graphics card and no real photos. Use made-up embeddings (for example one-hot vectors with a little noise) and a fake face model that returns faces with a `bbox` and an embedding.

- `load_people`: the good case and every error case above.
- Embeddings round-trip through save and load; missing file and mismatched lengths raise `ConfigError`.
- `FaceMatcher`: a known person matches; a stranger returns `None` with its score; the best of several rows per person wins; exactly at the threshold matches; empty store and zero vector are safe.
- `FaceMemory`: a tracked face follows its person box; it disappears when the track does; an untracked face expires after `max_age_seconds`; a new `remember` replaces the old faces.
- `Recognizer`: two different known people in one frame; one known and one unknown; two faces matching the same person (only the better keeps the name); a face smaller than `min_face_pixels` is skipped; track ids are assigned from person boxes, including the nested-box case and the no-detections case; boxes are clipped.
- `enroll` with a fake face model and tiny generated images in a temporary folder: no face, one face, several faces with a clear largest, several faces without one, an unreadable file, a person with no folder, an extra folder.
- `draw_info_card` stays inside the frame for faces in all four corners.
- One test that reads every `.py` file under `car_vision/` and fails if any file other than `recognizer.py` and `enroll.py` imports `insightface` or `onnxruntime`.

## Exit test on the laptop

1. `.venv\Scripts\python.exe -m pytest` passes.
2. With no photos present, `-m car_vision.enroll` explains what is missing and exits with code 1 without a traceback.
3. `-m car_vision.recognizer --source webcam --seconds 20 --no-window` runs with an empty or missing embeddings file handled as a readable error, and, once at least one person is enrolled, reports faces with scores at 12 or more FPS.

Recognition of each enrolled person at 1 and 2 metres from the car's camera is tested later by the user.
