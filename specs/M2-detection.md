# M2 spec: object detection and tracking

Label the objects in the live view with a pretrained YOLO model, and give each object a track number that follows it between frames. No training.

Read `CLAUDE.md` first. `car_vision/state.py`, `car_vision/config.py`, `car_vision/stream.py` and `config.yaml` already exist and are fixed; read them, do not change them.

## Files to create

| File | What it holds |
| --- | --- |
| `car_vision/detector.py` | The only module that imports `ultralytics` or `torch`. |
| `car_vision/draw.py` | Drawing helpers. Imports only `cv2`, `numpy` and `car_vision.state`. |
| `tests/test_detector.py` | Unit tests with a fake model. |
| `tests/test_draw.py` | Unit tests for the drawing helpers. |

## `car_vision/detector.py`

```python
class Detector:
    def __init__(self, settings: dict, model=None) -> None: ...
    def detect(self, frame: np.ndarray) -> list[Detection]: ...
    @property
    def device(self) -> str: ...        # "cuda" or "cpu", the one actually in use

def main(argv: list[str] | None = None) -> int: ...
```

- `settings` is the `detector:` section of `config.yaml`. Read every key with `car_vision.config.require`: `model`, `models_dir`, `confidence`, `image_size`, `device`, `tracker`.
- `model` is for tests: when given, use it and import nothing heavy. When `None`, load Ultralytics `YOLO` from `<models_dir>/<model>`; the weights download on the first run and must land inside `models_dir`, not in the project root. Import `ultralytics` and `torch` inside the function that loads the model, so importing `car_vision.detector` stays cheap for tests.
- `device: auto` means the graphics card when PyTorch can see it, otherwise the processor. `cuda` with no graphics card available raises `ConfigError` with a plain message.
- `detect` runs the tracker on one frame: `model.track(frame, persist=True, tracker=<tracker>, conf=<confidence>, imgsz=<image_size>, device=<device>, verbose=False)`. Always pass `tracker=` explicitly; the library default is a different tracker.
- Never turn on half precision. GTX 16-series cards return empty or NaN results with it.
- Convert the first result into `Detection` objects: `label` from the result's class names, `confidence` as a float, `box` as four ints clipped to the frame, `track_id` as an int or `None`. The track ids field is `None` whenever the tracker has nothing confirmed; handle that without an exception.
- A `None` or empty frame raises `ValueError`. No boxes returns `[]`.
- Verify the result field names (`boxes.xyxy`, `.cls`, `.conf`, `.id`, `names`) against the installed Ultralytics before relying on them, and say in your report what you checked.

`main` shows the live result. It builds a `FrameSource` from the `stream:` section, runs `detect` on each new frame (use the sequence number from `FrameSource.read()` to skip frames already processed), writes the list into a `WorldState` with `update_detections`, draws, and shows the detection FPS in the status line.

Arguments: `--source {phone,webcam,file}` overrides `stream.source`; `--seconds N` stops by itself; `--no-window` shows no video and prints one line per second with the FPS, the average time one `detect` call took in milliseconds, the device and the labels seen. `q` quits the window. On exit print one summary line with the average detection FPS, the average `detect` time in milliseconds (leaving out the first call, which warms the model up) and the device.

The laptop webcam delivers only about 10 frames a second in room light, so the live FPS is capped by the camera, not by the model. The `detect` time is the number that shows what the model can do.

## `car_vision/draw.py`

```python
def color_for(label: str) -> tuple[int, int, int]: ...          # same label, same colour, every run
def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> None: ...   # draws in place
def draw_status(frame: np.ndarray, text: str) -> None: ...      # same look as stream.draw_status
```

Each box gets a rectangle and a caption such as `person 0.87 #3` (no `#` part when `track_id` is `None`). The caption must stay inside the frame when the box touches the top edge. English text only.

## Tests

All pass with no camera, no network and no graphics card, in a few seconds.

- A fake model whose `track()` returns objects shaped like the real results: boxes with ids, boxes with `id = None`, and no boxes.
- `detect` returns the right labels, confidences, int boxes and track ids; clips boxes that go past the frame; passes `tracker`, `conf`, `imgsz` and `persist=True` through.
- `None` frame raises `ValueError`.
- `device: cuda` without a graphics card raises `ConfigError` (fake the availability check).
- `color_for` is stable; `draw_detections` changes pixels inside the frame, raises nothing for a box at the frame edge, and leaves the frame untouched for an empty list.
- One test that reads every `.py` file under `car_vision/` and fails if any file other than `detector.py` imports `ultralytics` or `torch`.

## Exit test on the laptop

1. `.venv\Scripts\python.exe -m pytest` passes.
2. `.venv\Scripts\python.exe -m car_vision.detector --source webcam --seconds 20 --no-window` reports device `cuda`, an average `detect` time of 66 ms or less (fast enough for 15 FPS), a live FPS that keeps up with the camera, and labels a person sitting in front of the laptop as `person` with a track id. A short recording, `data/samples/webcam_test.mp4`, can be used with `--source file` when nobody is in front of the webcam; pass the path by temporarily overriding it in code or a test, not by editing `config.yaml`.

The phone test (1 to 3 metres, people and common objects) is done later by the user.
