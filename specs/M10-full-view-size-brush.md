# M10 spec: full screen everywhere, the chosen size really shown, a brush on the photo

Three requests from the project owner after trying the page on the phone:

1. Full screen "still doesn't work".
2. The size chosen in the Camera box must change what is seen: "sometimes I need a good quality".
3. After clicking a thing, a brush to mark a part of its photo, to ask about exactly that part.

Read `CLAUDE.md`, then `specs/M7-click-to-inspect.md` (page rules, "Look and feel"), `specs/M8-theme-camera-delay.md` (B4 full screen, B5 follow-up chat) and `specs/M9-phone-picture-size.md`.

The work is three parts for three people at the same time. Files are split so that nobody edits a file someone else edits; the interfaces between the parts are fixed below.

---

## Part A: the page and the server (`templates/index.html`, `app.py`, new `car_vision/marking.py`, their tests)

### A1. Full view that always works

What was found: the Fullscreen API works in a real Edge window (checked by the orchestrator: the stage fills the screen, the pills sit along the bottom, clicks work). But the owner often looks at the page inside the Claude desktop app's browser pane, and that pane neither grants nor refuses `requestFullscreen()`: the button then does nothing at all. So full view must not depend on the browser saying yes.

- Pressing the button or `F` **always** switches the live view to fill the whole browser window at once: the `is-full` class goes on the stage (its CSS already covers the window with `position: fixed`), and the page behind stops scrolling. Then, where the Fullscreen API exists, the page also asks for real full screen; if that is granted the same layout simply fills the screen; if it is refused, ignored or missing, the window-filling view stays. The button is never hidden.
- Leaving: the button, `F`, or Escape. In real full screen the browser's own Escape leaves it (`fullscreenchange` with no full-screen element), and the page then leaves the full view as well. Without real full screen, Escape leaves the full view. In both cases the details panel stays open, and Escape order is: camera box, then full view, then panel.
- The full view must really cover the window: check that no ancestor of the stage has a transform, filter, `contain` or `will-change` that would trap `position: fixed`, and that the navigation bar does not show through.
- Everything from M8 B4 holds in both kinds of full view (layout, pills along the bottom, panel in its own column with a solid fill, click maths).

How to check it for real: a small throw-away script, `cdp.ps1` (kept outside the project), drives a headless Edge (which grants real full screen): `cdp.ps1 -Url http://127.0.0.1:<your port>/ -Tag <name> -Steps @("<javascript>", ...)` runs each step as a user gesture, prints its value and saves `<name>-<n>.jpg` beside the script. The built-in browser pane shows the other case (no answer from the API). Check both.

### A2. The words about picture size

Part B makes a chosen size the size that is shown. The help under the select changes to match:
- Automatic or "Leave the phone as it is": `Smaller pictures arrive sooner. The page shows <W> × <H>.`
- A chosen size: `Sharper, but a little later and slower. The page shows <W> × <H>.`
`<W> × <H>` stays the real size of the video's pictures, as now.

### A3. The brush

In the details panel of an object or an area (never a person), under the photo:

- A **Mark a part** button (a brush drawn inline) switches the brush on and off, and a **Clear** button appears once something is marked. With the brush on, dragging over the photo (mouse, pen or finger; Pointer Events, `touch-action: none` on the photo while the brush is on) paints a translucent stroke in the accent colour on a canvas that lies exactly over the picture. The brush width is 6 % of the photo's shorter side.
- The marking is kept as strokes of points, each point as fractions of the photo (0 to 1 from its top-left corner). At most 12 strokes and 400 points a stroke; a new point is added only after the pointer moved at least 0.5 % of the photo's width; at the limits the brush just stops adding.
- While a part is marked: the follow-up field's placeholder reads `Ask about the marked part`, and a quick button **What is this part?** sits beside the field (it sends that question). A question sent with a marking shows a small `marked part` tag on its bubble.
- The marking belongs to the thing shown: it stays after a question (to ask more), goes when Clear is pressed, and is forgotten when another thing is clicked or the panel closes.
- Keyboard: both buttons are reachable and operable; the brush itself is pointer-only, which is acceptable.
- No endless animation, nothing set as HTML. The canvas must stay aligned with the picture when the panel resizes and in full view.

`POST /api/follow_up` gains an optional `"strokes"`: a list of `{"points": [[x, y], ...]}`.
- Refused with 400 and a plain message: not a list, more than 12 strokes, a stroke with no point or more than 400, a point that is not two finite numbers from 0 to 1, or strokes sent for a person.
- With strokes, the server draws the marking on a copy of the stored photo (`marking.mark_photo`) and calls `assistant.follow_up(photo, label, info, earlier, question, marked=<that copy>)`. The browser still sends no picture.
- The pair stored for later questions keeps the question text with `(about a part the visitor marked) ` in front, so later turns know.

New module `car_vision/marking.py` (`cv2` and `numpy` only):

```python
Stroke = Sequence[tuple[float, float]]
def mark_photo(photo: bytes, strokes: Sequence[Stroke], quality: int = 85) -> bytes: ...
```

- Decodes the JPEG, paints the strokes into a mask (lines with round ends, 6 % of the shorter side wide; a single point is a filled circle), and draws only the **outline** of the marked region on the picture: a magenta line with a thin white line outside it, so it shows on any background and the marked content itself stays untouched. Line width grows with the photo (at least 2 pixels). Returns JPEG bytes of the same size. `ValueError` for bytes that are not a picture or for no strokes.

The privacy line gains: `A marked part is sent as the same cut-out with your outline drawn on it.`

Tests: the template has the brush button, Clear, the canvas, the quick button; `mark_photo` (same size out, magenta present near the stroke, the middle of a closed loop unchanged, a single point, bad input); the route (strokes reach `mark_photo` and the assistant gets `marked`; each refusal; a person; no strokes means `marked` is not passed or is `None`; the stored pair's prefix).

---

## Part B: the chosen size is the size shown (`stream.py`, `pipeline.py`, `draw.py`, their tests)

Today every picture is shrunk into the `stream.width` x `stream.height` box (640 x 480, so 640 x 360 for a wide phone), whatever the phone sends. The owner picked 1920 x 1080 in the Camera box and saw no difference. From now on:

- **Automatic** and **Leave the phone as it is**: as today, the box from `config.yaml`.
- **A chosen size** (in the Camera box, or `stream.phone_picture: 1280x720` in `config.yaml`): the box becomes that size, at most 1920 x 1080 (`MAX_SHOWN_SIZE`, a constant; a phone asked for 3840 x 2160 is shown at 1920 x 1080). So the detector, the face recogniser, the cut-outs sent to the assistant and the video on the page all get the sharper picture. Going back to Automatic puts the box back.
- `FrameSource.set_box(max_width: int, max_height: int) -> None`: takes effect from the next frame, safe to call from another thread while the reader runs. Nothing else in `FrameSource` changes.
- `Pipeline.set_phone_picture` sets the box at once (it does not wait for the phone), and the box follows the choice across `use_phone` and a phone that drops out and returns. A configured size the phone does not offer keeps the config box, with the existing warning.
- `draw.py`: everything drawn on a frame scales with the frame, so labels are as readable at 1920 wide as at 640 wide: line thickness, text size, padding, the status line and the info card all grow with `frame width / 640` (never below today's sizes). One helper decides the scale; no caller changes.
- The photo cut-out rules are unchanged (they are in fractions and in "at least 160 pixels").
- Tests: `set_box` while frames flow; the box for each kind of choice, including the 1920 x 1080 cap and going back to Automatic; a switch of phone keeps it; drawing at 640 and at 1920 wide (a label's height roughly triples; nothing is drawn outside the frame).
- Measure, with fakes only, what one 1920 x 1080 frame costs in `draw` plus JPEG encoding compared with 640 x 360, and report the two numbers: the orchestrator needs them to tell the owner what the sharper picture costs.

## Part C: a marked picture in a follow-up (`assistant.py`, its tests)

```python
def follow_up(self, image, label, info, earlier, question, marked: bytes | None = None) -> Answer
```

- `marked` is the same cut-out with the visitor's outline drawn on it, checked with `check_image`. When given, the last user turn carries that picture followed by the text: `The visitor brushed over part of the picture. In this second copy that part is outlined in magenta. The question is about the outlined part: <question>`.
- The follow-up system prompt gains: when a second copy with a magenta outline is given, answer about what is inside the outline; say what that part is before anything else; the outline itself is not part of the scene.
- When `info` is `None` and there are no earlier pairs, the picture turn and the question are one user turn (as now): the marked picture and its text go into that same turn after the first picture and the label sentence.
- Everything else as before. Tests: the turns with `marked` in each of the shapes (no info; info; info and pairs); a bad `marked` refused before the client is called; one wire test with two pictures.
