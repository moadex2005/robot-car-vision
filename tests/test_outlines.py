"""Outlines of the named finds: read from the model's masks, kept through the pipeline, drawn.

Fake model results and fake pipelines only: no camera, model, network or graphics card.
"""

import logging

import numpy as np
import pytest

from car_vision import pipeline as pipeline_module
from car_vision.detector import (OUTLINE_MAX_POINTS, Detector, OpenDetector, simplify_outline,
                                 to_detections)
from car_vision.state import Detection
from test_detector import OPEN_SETTINGS, SETTINGS, FakeBoxes, FakeModel, FakeResult, FakeYoloe, frame
from test_pipeline import FakeOpenDetector, FakeSource, decoded, make, searching, wait_until  # noqa: F401

SQUARE = [(10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0)]
OTHER_SQUARE = [(300.2, 50.4), (400.0, 50.0), (399.6, 150.0), (300.0, 149.7)]


def circle(points=400, centre=(320, 240), radius=100.0):
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False)
    return np.stack([centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)], axis=1)


class FakeMasks:
    """Shaped like ultralytics.engine.results.Masks: xy is one N x 2 polygon per box."""

    def __init__(self, polygons) -> None:
        self.xy = [np.asarray(polygon, dtype=np.float32).reshape(-1, 2) for polygon in polygons]


class BrokenMasks:
    @property
    def xy(self):
        raise RuntimeError("the masks could not be read")


class MasksMustNotBeRead(FakeResult):
    """A result whose masks raise when touched, to show they are never read."""

    @property
    def masks(self):
        raise AssertionError("the masks were read")


def two_boxes(result_class=FakeResult):
    return result_class(FakeBoxes(xyxy=[[10, 10, 110, 110], [300, 50, 400, 150]],
                                  conf=[0.8, 0.6], cls=[0, 56], ids=[3, 7]))


def with_masks(polygons):
    result = two_boxes()
    result.masks = FakeMasks(polygons)
    return result


# ---- simplify_outline ---------------------------------------------------------------


def test_a_square_keeps_its_four_corners():
    assert set(simplify_outline(SQUARE, 640, 480)) == {(10, 10), (110, 10), (110, 110), (10, 110)}


def test_a_circle_of_400_points_is_cut_to_at_most_48():
    outline = simplify_outline(circle(), 640, 480)
    assert 3 <= len(outline) <= OUTLINE_MAX_POINTS == 48
    for x, y in outline:
        assert isinstance(x, int) and isinstance(y, int)
        assert abs(np.hypot(x - 320, y - 240) - 100) < 5        # still on the circle


def test_max_points_can_be_smaller():
    assert 3 <= len(simplify_outline(circle(), 640, 480, max_points=8)) <= 8


def test_points_outside_the_frame_are_kept_inside_it():
    outline = simplify_outline([(-20, -20), (700, -20), (700, 500), (-20, 500)], 640, 480)
    assert set(outline) == {(0, 0), (639, 0), (639, 479), (0, 479)}


def test_a_torch_like_tensor_is_read_too():
    class Tensor:
        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(SQUARE, dtype=np.float32)

    assert len(simplify_outline(Tensor(), 640, 480)) == 4


@pytest.mark.parametrize("points", [
    [(10, 10), (50, 50)],                           # two points
    [(0, 0), (50, 50), (100, 100), (150, 150)],     # a straight line: no area
    [(5, 5)] * 10,                                  # one point many times
    None,
    [],
    np.zeros((0, 2)),                               # what the library gives for an empty mask
    np.zeros((5, 3)),                               # not N x 2
    np.zeros(8),
    [(1, 2), (3,), (4, 5, 6)],                      # ragged
    "not points",
    [(0, 0), (np.nan, 5), (10, 10)],
])
def test_no_shape_gives_an_empty_outline_and_never_raises(points):
    assert simplify_outline(points, 640, 480) == ()


def test_a_tensor_that_cannot_be_copied_gives_no_outline():
    class BrokenTensor:
        def cpu(self):
            raise RuntimeError("the graphics card is gone")

    assert simplify_outline(BrokenTensor(), 640, 480) == ()


def test_a_frame_without_size_gives_no_outline():
    assert simplify_outline(SQUARE, 0, 0) == ()


# ---- to_detections --------------------------------------------------------------


def test_outlines_are_given_in_the_order_of_the_boxes():
    found = to_detections(with_masks([SQUARE, OTHER_SQUARE]), 640, 480, outlines=True)
    assert set(found[0].outline) == {(10, 10), (110, 10), (110, 110), (10, 110)}
    assert set(found[1].outline) == {(300, 50), (400, 50), (400, 150), (300, 150)}
    assert [d.track_id for d in found] == [3, 7]
    assert found[0].box == (10, 10, 110, 110)                   # the box stays as it was


def test_without_outlines_the_masks_are_never_read():
    found = to_detections(two_boxes(MasksMustNotBeRead), 640, 480)
    assert [d.outline for d in found] == [(), ()]
    assert [d.outline for d in to_detections(two_boxes(MasksMustNotBeRead), 640, 480, outlines=False)] == [(), ()]


def test_a_polygon_count_that_is_not_the_box_count_gives_no_outlines(caplog):
    with caplog.at_level(logging.WARNING, logger="car_vision.detector"):
        found = to_detections(with_masks([SQUARE]), 640, 480, outlines=True)
    assert [d.outline for d in found] == [(), ()]
    assert [d.box for d in found] == [(10, 10, 110, 110), (300, 50, 400, 150)]
    assert len(caplog.records) == 1


@pytest.mark.parametrize("masks", [None, BrokenMasks(), object()])
def test_missing_or_broken_masks_give_boxes_only(masks, caplog):
    result = two_boxes()
    result.masks = masks
    with caplog.at_level(logging.WARNING, logger="car_vision.detector"):
        found = to_detections(result, 640, 480, outlines=True)
    assert [d.outline for d in found] == [(), ()]
    assert len(found) == 2
    assert len(caplog.records) == 1


def test_a_result_with_no_masks_attribute_gives_boxes_only():
    assert [d.outline for d in to_detections(two_boxes(), 640, 480, outlines=True)] == [(), ()]


def test_an_empty_polygon_leaves_only_that_box_without_an_outline():
    found = to_detections(with_masks([np.zeros((0, 2)), OTHER_SQUARE]), 640, 480, outlines=True)
    assert found[0].outline == ()
    assert len(found[1].outline) == 4


def test_open_detector_gives_outlines():
    result = with_masks([SQUARE, OTHER_SQUARE])
    result.names = {}
    detector = OpenDetector(OPEN_SETTINGS, model=FakeYoloe(result))
    detector.set_names(["keys", "wallet"])
    found = detector.detect(frame())
    assert [len(d.outline) for d in found] == [4, 4]
    assert found[0].label == "keys"


def test_open_detector_warns_only_once_about_missing_masks(caplog):
    results = [two_boxes() for _ in range(10)]          # no masks: a model that is not a segmentation one
    for result in results:
        result.names = {}
    detector = OpenDetector(OPEN_SETTINGS, model=FakeYoloe(*results))
    detector.set_names(["keys"])
    with caplog.at_level(logging.WARNING, logger="car_vision.detector"):
        for _ in range(10):
            assert [d.outline for d in detector.detect(frame())] == [(), ()]
    assert len([r for r in caplog.records if "outlines" in r.getMessage()]) == 1


def test_a_warn_function_is_told_instead_of_the_log(caplog):
    told = []
    with caplog.at_level(logging.WARNING, logger="car_vision.detector"):
        to_detections(two_boxes(), 640, 480, outlines=True, warn=told.append)
    assert len(told) == 1 and "drawing boxes" in told[0]
    assert caplog.records == []


def test_the_everyday_detector_never_reads_masks():
    detector = Detector(SETTINGS, model=FakeModel(two_boxes(MasksMustNotBeRead)))
    assert [d.outline for d in detector.detect(frame())] == [(), ()]


# ---- the pipeline -------------------------------------------------------------------

DIAMOND = ((340, 60), (380, 100), (340, 140), (300, 100))
KEYS_WITH_OUTLINE = Detection("keys", 0.7, (300, 60, 380, 140), 7, outline=DIAMOND)


def test_merge_named_keeps_the_outline_and_sets_the_named_track_number():
    shown, kept = pipeline_module.merge_named([], pipeline_module.NamedFinds(), [KEYS_WITH_OUTLINE],
                                              0, 10.0, ["keys"])
    assert len(shown) == 1
    assert shown[0].outline == DIAMOND
    assert shown[0].track_id == 7 + pipeline_module.NAMED_ID_OFFSET
    assert kept.detections[0].outline == DIAMOND


def test_a_named_find_is_drawn_by_its_outline_and_not_by_its_box(make):
    pipeline, _ = searching(make, names=["keys"], open_detector=FakeOpenDetector(finds=[KEYS_WITH_OUTLINE]))
    yellow = np.array(pipeline_module.draw.WANTED_COLOR)
    background = np.array([60, 60, 60])

    def outline_is_drawn():
        jpeg = pipeline.latest_jpeg()
        return jpeg is not None and np.abs(decoded(jpeg)[80, 360].astype(int) - yellow).sum() < 60

    assert wait_until(outline_is_drawn)                         # the middle of the top-right edge
    drawn = decoded(pipeline.latest_jpeg()).astype(int)
    assert np.abs(drawn[140, 380] - background).sum() < 30      # the box's bottom-right corner is not drawn
    assert np.abs(drawn[100, 300 + 2] - yellow).sum() < 60     # the left corner of the outline is
    inside = drawn[100, 340]
    assert np.abs(inside - background).sum() > 20               # the inside is tinted ...
    assert np.abs(inside - yellow).sum() > 200                  # ... but not filled
