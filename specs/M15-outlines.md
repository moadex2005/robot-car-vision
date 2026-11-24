# M15 spec: outlines instead of boxes

The find-by-name detector (YOLOE, `detector.open_model: yoloe-26s-seg.pt`) is a segmentation model: for every find it also computes the thing's own shape. Today that shape is thrown away and a rectangle is drawn. After this milestone a thing found by name, or named by auto-discover, is drawn with its real outline. The everyday detector (`yolo26n.pt`) has no shapes, so its boxes stay boxes.

Read `CLAUDE.md`, then the "Looking for things by a typed name" part of the docstring of `car_vision/pipeline.py`.

## Already done by the orchestrator (do not change)

`car_vision/state.py`, `Detection` has one new field at the end:

```python
outline: tuple[tuple[int, int], ...] = ()   # corner points (x, y) in frame pixels; empty = only the box is known
```

It has a default, so every existing `Detection(...)` call still works. No new setting: outlines are always used when the model gives them.

## Part A: `car_vision/detector.py`

```python
OUTLINE_MAX_POINTS = 48

def simplify_outline(points: Any, width: int, height: int,
                     max_points: int = OUTLINE_MAX_POINTS) -> tuple[tuple[int, int], ...]: ...

def to_detections(result: Any, width: int, height: int, names: Any = None,
                  outlines: bool = False) -> list[Detection]: ...
```

- `simplify_outline` takes one polygon as the library gives it (an N x 2 array or list of x, y in frame pixels, floats) and returns whole-pixel points kept inside the frame (`0..width-1`, `0..height-1`), with at most `max_points` points. Use `cv2.approxPolyDP` on the closed polygon, raising its tolerance step by step until the count fits. It returns `()` when the result has fewer than 3 points, when the polygon has no area, or when `points` is empty, `None` or not N x 2. It never raises for bad input.
- `to_detections(..., outlines=True)` reads `result.masks.xy` (a list with one polygon per box, in the same order as `result.boxes`) and gives each `Detection` its `outline`. No outlines at all (every `outline` stays `()`) when `result.masks` is missing or `None`, when the number of polygons is not the number of boxes, or when reading the masks raises: log one warning line and carry on with boxes. With `outlines=False` (the default, used by the everyday `Detector`) nothing about masks is read.
- `OpenDetector.detect` calls `to_detections(..., outlines=True)`. Change the comment that says the masks are ignored. Nothing else in `OpenDetector` changes: same `track()` arguments, no half precision.
- The everyday `Detector` is untouched.

## Part B: `car_vision/draw.py`

```python
OUTLINE_FILL = 0.2          # how strongly the shape is tinted with its colour (0 = not at all)

def draw_detections(frame, detections, wanted=()) -> None: ...   # same signature as today
def draw_outline(frame: np.ndarray, outline: Sequence[tuple[int, int]],
                 color: tuple[int, int, int], thickness: int) -> None: ...
```

- A detection whose `outline` has 3 or more points is drawn as its shape instead of its rectangle: the inside tinted with the colour at `OUTLINE_FILL`, then a closed anti-aliased line along the outline. Colour and line thickness are exactly what the rectangle would have had: the label's colour and the usual thickness, or `WANTED_COLOR` and `WANTED_THICKNESS` when its track number is in `wanted`. The caption is drawn where it is today (from the box's top-left corner).
- A detection with no outline is drawn exactly as today. Existing tests of `draw_detections` must pass unchanged.
- The tint must be cheap: blend only inside the outline's bounding rectangle (clipped to the frame), never a copy of the whole frame per detection. Points outside the frame must not raise.
- `draw_outline` does the tint and the line; `draw_detections` stays a short function.

## Part C: `car_vision/pipeline.py`

No behaviour change is expected: a named find is a `Detection`, and `named_apart` uses `dataclasses.replace`, so the outline travels with it through `merge_named` into the drawing. Check that this is true by a test, and change the pipeline only if the test shows the outline is lost somewhere. A new frame size already forgets the kept finds (`_forget_named_finds`), so an outline in the old size's pixels is never drawn.

The page needs no change: it shows the video the server draws. The click, the cut-out photo for the assistant and the memory keep using the box.

## Tests (new files, so they do not collide with other work going on in parallel)

- `tests/test_outlines.py`: `simplify_outline` (a square, a circle of 400 points cut to at most 48, points outside the frame clipped, two points, a straight line, `None`, a wrong shape); `to_detections` with a fake result that has `masks.xy` (outlines set in box order; `outlines=False` reads nothing, shown by a fake whose `masks` raises when touched; a count mismatch; `masks` is `None`; a polygon that is empty for one box); `OpenDetector.detect` with a fake model gives outlines.
- `tests/test_draw_outlines.py`: a pixel on the outline has the line colour; the rectangle's own corner pixel is not drawn when an outline exists; a pixel well inside is tinted (changed, but not the full colour) and a pixel outside the shape is untouched; a wanted find uses `WANTED_COLOR`; an outline partly outside the frame does not raise; a detection without an outline still gets its rectangle.
- One test, in `tests/test_outlines.py`, that `merge_named` (with a `fresh` find that has an outline and a track number) returns a detection that still has that outline, with `NAMED_ID_OFFSET` added to its track number.

## Exit test

The whole test suite passes. With a fake open detector that returns an outline, a frame run through `Pipeline._process` produces a picture in which the outline is drawn and the rectangle is not (one test, using the fakes that `tests/test_pipeline.py` already has; put it in `tests/test_outlines.py`). The orchestrator checks it live with the real model.
