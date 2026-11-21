"""mark_photo: the outline of a brushed part drawn on a copy of a photo. Generated pictures only."""

import cv2
import numpy as np
import pytest
from car_vision.marking import MAGENTA, brush_width, mark_photo, outline_width


def photo(width=320, height=240, colour=(40, 120, 60)):
    """A plain JPEG of one colour; a gradient band keeps it from being too easy to compare."""
    image = np.full((height, width, 3), colour, np.uint8)
    image[:, :, 0] = np.linspace(20, 80, width, dtype=np.uint8)[None, :]
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def decoded(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def is_magenta(pixel):
    blue, green, red = (int(value) for value in pixel)
    return blue > 180 and red > 180 and green < 90


def magenta_near(image, x, y, reach):
    """True when some pixel within `reach` of (x, y) is magenta."""
    patch = image[max(0, y - reach):y + reach + 1, max(0, x - reach):x + reach + 1].reshape(-1, 3)
    return any(is_magenta(pixel) for pixel in patch)


def close(a, b, tolerance=12):
    """Two pixels look the same (JPEG changes them a little)."""
    return int(np.abs(a.astype(int) - b.astype(int)).max()) <= tolerance


SQUARE_LOOP = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75), (0.25, 0.25)]


def test_the_marked_photo_has_the_same_size_and_is_a_jpeg():
    original = photo(320, 240)
    marked = mark_photo(original, [[(0.2, 0.5), (0.8, 0.5)]])
    assert marked[:2] == b"\xff\xd8"
    assert decoded(marked).shape == decoded(original).shape


def test_magenta_is_drawn_near_the_stroke_and_not_far_away():
    marked = decoded(mark_photo(photo(320, 240), [[(0.2, 0.5), (0.8, 0.5)]]))
    width = brush_width(320, 240)
    above_line = round(0.5 * 239) - width // 2 - 2               # just outside the top edge of the stroke
    assert magenta_near(marked, 160, above_line, reach=3)
    assert not magenta_near(marked, 160, 20, reach=5)             # far from the stroke: untouched
    assert not magenta_near(marked, 300, 220, reach=5)


def test_the_brushed_content_itself_is_not_painted():
    original = decoded(photo(320, 240))
    marked = decoded(mark_photo(photo(320, 240), [[(0.2, 0.5), (0.8, 0.5)]]))
    middle = (round(0.5 * 239), 160)                              # on the stroke itself
    assert close(marked[middle], original[middle])


def test_the_middle_of_a_closed_loop_is_unchanged():
    original = decoded(photo(320, 240))
    marked = decoded(mark_photo(photo(320, 240), [SQUARE_LOOP]))
    assert close(marked[120, 160], original[120, 160])
    assert not magenta_near(marked, 160, 120, reach=30)           # no line around the inside of the loop
    outer_left = round(0.25 * 319) - brush_width(320, 240) // 2 - 2
    assert magenta_near(marked, outer_left, 120, reach=3)         # but one around the outside


def test_a_single_point_is_a_circle_with_an_outline():
    marked = decoded(mark_photo(photo(320, 240), [[(0.5, 0.5)]]))
    radius = brush_width(320, 240) // 2
    centre_x, centre_y = round(0.5 * 319), round(0.5 * 239)
    assert magenta_near(marked, centre_x + radius + 2, centre_y, reach=2)
    assert magenta_near(marked, centre_x, centre_y - radius - 2, reach=2)
    assert not is_magenta(marked[centre_y, centre_x])


def test_a_white_line_lies_outside_the_magenta_one():
    marked = decoded(mark_photo(photo(320, 240, colour=(40, 40, 40)), [[(0.5, 0.5)]]))
    centre_x, centre_y = round(0.5 * 319), round(0.5 * 239)
    row = marked[centre_y, centre_x:]
    first_magenta = next(i for i, pixel in enumerate(row) if is_magenta(pixel))
    after = row[first_magenta:first_magenta + 12]
    assert any(int(pixel.min()) > 200 for pixel in after)         # white just beyond the magenta


def test_the_line_grows_with_the_photo():
    assert outline_width(160, 160) == 2                           # never thinner than 2 pixels
    assert outline_width(1920, 1080) > outline_width(320, 240)
    assert brush_width(1000, 500) == 30                           # 6 % of the shorter side


def test_points_on_the_edges_and_several_strokes_work():
    marked = mark_photo(photo(200, 200), [[(0.0, 0.0), (1.0, 1.0)], [(1.0, 0.0)], [(0.0, 1.0), (0.5, 0.5)]])
    assert decoded(marked).shape == (200, 200, 3)


@pytest.mark.parametrize("bad", [b"", b"not a picture", b"\xff\xd8\xff broken"])
def test_bytes_that_are_not_a_picture_are_refused(bad):
    with pytest.raises(ValueError):
        mark_photo(bad, [[(0.5, 0.5)]])


@pytest.mark.parametrize("strokes", [[], [[]], [[], []]])
def test_no_strokes_is_refused(strokes):
    with pytest.raises(ValueError):
        mark_photo(photo(), strokes)


@pytest.mark.parametrize("point", [(0.5,), (0.5, 0.5, 0.5), (float("nan"), 0.5), (0.5, float("inf"))])
def test_a_bad_point_is_refused(point):
    with pytest.raises(ValueError):
        mark_photo(photo(), [[point]])
