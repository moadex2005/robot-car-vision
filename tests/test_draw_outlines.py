"""Drawing a find by its outline instead of its box. Plain numpy frames only."""

import numpy as np

from car_vision import draw
from car_vision.state import Detection

BACKGROUND = 60
BOX = (100, 100, 300, 300)
TRAPEZOID = ((150, 120), (250, 120), (280, 280), (120, 280))   # inside BOX, away from its corners
ON_THE_LINE = (120, 200)          # (y, x): the middle of the top edge
INSIDE = (200, 200)
OUTSIDE = (200, 110)              # left of the left edge, still inside the box
BOX_CORNER = (300, 300)           # bottom-right; the caption covers the top-left one


def blank(width=640, height=360):
    return np.full((height, width, 3), BACKGROUND, dtype=np.uint8)


def cup(outline=TRAPEZOID, track_id=1):
    return Detection("cup", 0.9, BOX, track_id, outline=outline)


def pixel(frame, at):
    return tuple(int(value) for value in frame[at])


def test_a_pixel_on_the_outline_has_the_line_colour():
    frame = blank()
    draw.draw_detections(frame, [cup()])
    assert pixel(frame, ON_THE_LINE) == draw.color_for("cup")


def test_the_box_corner_is_not_drawn_when_there_is_an_outline():
    frame = blank()
    draw.draw_detections(frame, [cup()])
    assert pixel(frame, BOX_CORNER) == (BACKGROUND,) * 3
    assert pixel(frame, (300, 100)) == (BACKGROUND,) * 3     # bottom-left corner too


def test_the_inside_is_tinted_and_the_outside_untouched():
    frame = blank()
    draw.draw_detections(frame, [cup()])
    color = np.array(draw.color_for("cup"))
    expected = np.rint((1 - draw.OUTLINE_FILL) * BACKGROUND + draw.OUTLINE_FILL * color)
    inside = np.array(pixel(frame, INSIDE))
    assert np.abs(inside - expected).max() <= 1
    assert pixel(frame, INSIDE) != (BACKGROUND,) * 3
    assert pixel(frame, INSIDE) != tuple(color)
    assert pixel(frame, OUTSIDE) == (BACKGROUND,) * 3


def test_a_wanted_find_is_drawn_in_the_wanted_colour():
    frame = blank()
    draw.draw_detections(frame, [cup()], wanted=[1])
    assert pixel(frame, ON_THE_LINE) == draw.WANTED_COLOR


def test_the_wanted_line_is_as_thick_as_a_wanted_box():
    plain, wanted = blank(), blank()
    draw.draw_detections(plain, [cup()])
    draw.draw_detections(wanted, [cup()], wanted=[1])
    column = 200
    plain_rows = np.flatnonzero((plain[100:140, column] != BACKGROUND).any(axis=1))
    wanted_rows = np.flatnonzero((wanted[100:140, column] != BACKGROUND).any(axis=1))
    assert len(wanted_rows) > len(plain_rows)


def test_an_outline_partly_outside_the_frame_does_not_raise():
    frame = blank()
    draw.draw_detections(frame, [cup(outline=((-50, -50), (700, 10), (300, 500)))])
    assert (frame != BACKGROUND).any()


def test_an_outline_wholly_outside_the_frame_does_not_raise():
    frame = blank()
    draw.draw_detections(frame, [cup(outline=((-90, -90), (-10, -90), (-10, -10)))])


def test_a_detection_without_an_outline_still_gets_its_rectangle():
    frame = blank()
    draw.draw_detections(frame, [cup(outline=())])
    assert pixel(frame, BOX_CORNER) == draw.color_for("cup")
    assert pixel(frame, INSIDE) == (BACKGROUND,) * 3


def test_an_outline_of_two_points_is_drawn_as_the_rectangle():
    frame = blank()
    draw.draw_detections(frame, [cup(outline=((150, 120), (250, 120)))])
    assert pixel(frame, BOX_CORNER) == draw.color_for("cup")


def test_a_large_outline_on_a_large_frame_is_tinted_inside_only():
    frame = blank(1920, 1080)
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    circle = tuple((int(960 + 400 * np.cos(a)), int(540 + 400 * np.sin(a))) for a in angles)
    draw.draw_detections(frame, [Detection("cup", 0.9, (560, 140, 1360, 940), 1, outline=circle)])
    color = np.array(draw.color_for("cup"))
    expected = np.rint((1 - draw.OUTLINE_FILL) * BACKGROUND + draw.OUTLINE_FILL * color)
    assert np.abs(np.array(pixel(frame, (540, 960))) - expected).max() <= 1     # the middle
    assert np.abs(np.array(pixel(frame, (900, 960))) - expected).max() <= 1     # near the bottom edge
    assert pixel(frame, (150, 570)) == (BACKGROUND,) * 3                         # a box corner, outside
    assert pixel(frame, (540, 100)) == (BACKGROUND,) * 3                         # far outside


def test_draw_outline_tints_only_inside_the_shape():
    frame = blank()
    draw.draw_outline(frame, TRAPEZOID, (0, 0, 255), 2)
    assert pixel(frame, (10, 10)) == (BACKGROUND,) * 3
    assert pixel(frame, INSIDE)[2] > BACKGROUND
