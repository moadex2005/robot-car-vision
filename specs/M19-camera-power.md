# M19 spec: turn the camera off and on from the page

The owner wants a button on the page that turns the camera off and on. Off means really off: the laptop camera's light goes out, or the phone's stream is closed, and no model runs. On starts the same camera again.

Read `CLAUDE.md`, then the docstring of `car_vision/pipeline.py` and its `use_phone`, `stop`, `_forget_old_picture`, `_run`, `_start_afresh` and `_show_stream_lost`.

## Part A: `car_vision/pipeline.py`

```python
CAMERA_OFF = "Camera off"                       # the line on the still picture, and nothing else is drawn
HINT_CAMERA_OFF = "Cancelled: the camera was switched off"

class Pipeline:
    def set_camera(self, on: bool) -> None: ...
```

- The camera is **on at every start**; the choice is not remembered between runs.
- `set_camera(False)`: returns at once. The live source is let go in a thread of its own (`let_go_in_background`), which releases the webcam or closes the phone's stream. Everything the old picture showed is forgotten exactly as at a camera switch (`_forget_old_picture`: a new generation, no view, no finds, an empty `WorldState`, no reactions), a running photo session is cancelled with `HINT_CAMERA_OFF`, and the page gets one still picture: plain dark, the size of the last picture, with the line `Camera off`. Before any picture ever arrived there is nothing to darken; the page then shows its own text.
- While it is off: the loop thread keeps running but handles no frames; no detector, recogniser, open detector or lookout call happens; the memory's `tick()` still runs, so open sightings are closed and written. Auto-discover stays switched on if it was, but makes no look and spends nothing while there is no picture. Search names are kept. `target()` and `scene_photo()` give `None` (there is no view), which the routes already answer plainly.
- `set_camera(True)`: a new source is made from the settings in use now (the phone at the address in use when the camera was a phone, otherwise the source the program started with) and started; a new generation begins; the still picture says `Connecting to the webcam ...` (the existing line) until the first frame; the phone's picture size is checked again as after a Connect. If the loop had ended after too many failures it is started again, as `use_phone` does.
- `set_camera` with the value it already has does nothing. Anything but `True` or `False` raises `ValueError("The camera is switched with true or false.")`.
- `use_phone(url)` while the camera is off switches it on with that phone: typing an address and pressing Connect means "use this camera".
- `set_phone_picture` while off only stores the choice; the phone is asked when the camera is switched on.
- `stop()` works in both states and never starts a camera. `start()` after `stop()` is not needed.
- Calls from several web threads at once (`set_camera`, `use_phone`, `stop`) happen one after the other under `self._switch_lock`; the last one wins; none of them waits for a camera to let go.
- `status()` gains `"camera": {"on": true | false}`. Add the key directly after the `"source"` line of the returned dictionary (another change adds a key elsewhere in it). While off: `"connected": false`, `"ended": false`, `"error": null`, `"source"` still names the camera that will be used again, `"fps": 0.0`, and no objects, people, reactions or codes.
- A suggestion, not a rule: a small stand-in source for "off" (`source` name, `connected` and `ended` False, `read()` gives `(None, 0)`, `start`/`stop`/`set_box`/`reopen` do nothing) lets the loop's existing code for a switched camera do most of the work; `_show_stream_lost` then needs to know the off case for its line.
- Put new fields of `__init__` directly after `self._switch_lock = ...`; put `set_camera` directly after `use_phone`.

## Part B: `car_vision/app.py`

`POST /api/camera/power` with `{"on": true}` or `{"on": false}` calls `pipeline.set_camera` and answers `{"ok": true, "on": <bool>}`. Anything else is a 400 with `{"error": 'Send {"on": true} or {"on": false}.'}`. Put the route directly after `camera_picture`. The existing same-origin check (`only_this_page`) covers it like every other POST.

## Part C: `car_vision/templates/index.html`

- A round button in the tools at the top right of the video (`.frame-tools`), first in the row, so it is reachable in the normal view and in the full view: a camera icon, with a line through it while the camera is off. `aria-pressed` is `true` while the camera is on; its `title` and `aria-label` say what a press will do: `Turn the camera off` or `Turn the camera on`.
- A press sends the request, shows the button as busy until the answer comes, and after that the page follows the status (`status.camera.on`), so two tabs agree. A failed request shows a short plain line near the button for a few seconds and leaves the button as the status says.
- While the camera is off: the status pill says `Camera off` with a quiet grey dot (a new state class `is-off`, checked before the other states of `drawStatusLine`, after an error); over the dark picture, centred, a calm line `Camera off` with a **Turn on** button that does the same as the round button; the hover hand and clicks on the video do nothing (there is nothing to inspect).
- When the program is not running (`setOnline(false)`) the button is hidden, like the finder.
- Nothing is spoken when the camera is switched. A voice question about finding something while the camera is off gets the page's existing "camera is down" answer; do not add another.
- The privacy line in the footer needs no change.
- Page rules, all of them: text through `textContent`, nodes through `make()`, no outside resources, no endless animation, no blur over the video, redraw only what changed, the full view's inert page.
- Where to put it, so other changes going on in parallel merge cleanly: the CSS directly after the `.frame-button` rules; the JavaScript as a section of its own, `// ---- the camera: off and on ----`, directly after the function `drawStatusLine`; in `poll()` call `drawCamera(status);` on a new line directly after `drawStatusLine(status);`.

## Tests (new files)

- `tests/test_pipeline_camera.py`, with the fakes of `tests/test_pipeline.py` (a source factory that records what it made and whether each source was started and stopped): on at start; off stops the old source (in the background: wait for it), empties the status and the `WorldState`, and the picture changes to one that is not the last frame; no detector call happens while off, however long; on makes and starts a new source of the same kind and frames flow again; off then on for a phone keeps its address; twice the same value does nothing (no new source); `use_phone` while off switches on; a photo session is cancelled with the hint; auto-discover on makes no look while off and looks again after on; `stop()` while off returns and does not start anything; wrong values raise; many `set_camera` calls from threads end in the state of the last one with exactly one live source.
- `tests/test_app_camera.py`: the route (true, false, missing body, a string, a number, a list; the pipeline fake records the call), the same-origin refusal, and string checks on the template: the button in `.frame-tools`, `drawCamera(status)` after `drawStatusLine(status)`, the `is-off` state, the Turn on button.
- `tests/test_pipeline.py` has a set of the expected status keys (`STATUS_KEYS`): add `"camera"` to it. Fake pipelines in the other test files get `set_camera` only where a test needs it.

## Exit test

The whole test suite passes. The orchestrator checks it live on the laptop camera: off turns the camera's light out and the page says `Camera off`; on brings the video back with boxes within a few seconds.
