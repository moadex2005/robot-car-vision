"""The drawing helpers, tested on blank frames."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from car_vision import draw
from car_vision.draw import caption_for, color_for, draw_detections, draw_status
from car_vision.faces import Person
from car_vision.state import Detection, RecognizedFace

ROOT = Path(__file__).resolve().parent.parent


def blank(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_color_for_is_stable_within_a_run():
    assert color_for("person") == color_for("person")
    assert all(isinstance(v, int) and 0 <= v <= 255 for v in color_for("person"))


def test_color_for_is_stable_between_runs():
    # A new Python process gets a different hash() seed; the colour must not change.
    code = "from car_vision.draw import color_for; print(color_for('person'))"
    output = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                            text=True, timeout=60, check=True).stdout
    assert output.strip() == str(color_for("person"))


def test_different_labels_usually_differ():
    labels = ["person", "chair", "cup", "bottle", "laptop", "dog"]
    assert len({color_for(label) for label in labels}) == len(labels)


def test_caption_with_and_without_track_id():
    assert caption_for(Detection("person", 0.871, (0, 0, 10, 10), 3)) == "person 0.87 #3"
    assert caption_for(Detection("cup", 0.5, (0, 0, 10, 10))) == "cup 0.50"


def test_draw_detections_changes_pixels_inside_the_frame():
    frame = blank()
    draw_detections(frame, [Detection("person", 0.9, (100, 120, 300, 400), 1)])
    assert frame.any()
    assert frame[120:401, 100:301].any()          # the box itself
    assert frame[100:120, 100:200].any()          # the caption above it


def test_box_at_the_frame_edge_raises_nothing_and_caption_stays_inside():
    frame = blank(320, 240)
    draw_detections(frame, [Detection("person", 0.87, (0, 0, 319, 239), 12)])
    # The caption band is drawn just inside the box, not cut off above row 0.
    assert frame[5:15, 10:60].any(axis=2).mean() > 0.5


def test_caption_near_the_right_edge_is_pulled_inside():
    frame = blank(320, 240)
    draw_detections(frame, [Detection("traffic light", 0.66, (315, 100, 319, 120), 4)])
    assert frame[80:100, 200:315].any()           # band slid left so the text fits


def test_empty_list_leaves_the_frame_untouched():
    frame = blank()
    draw_detections(frame, [])
    assert not frame.any()


def test_a_wanted_box_is_drawn_thicker_in_the_wanted_colour():
    keys = Detection("keys", 0.7, (100, 120, 300, 400), 100_007)
    plain, wanted = blank(), blank()
    draw_detections(plain, [keys])
    draw_detections(wanted, [keys], wanted=[100_007])
    expected = blank()
    cv2.rectangle(expected, (100, 120), (300, 400), draw.WANTED_COLOR, draw.WANTED_THICKNESS)
    assert np.array_equal(wanted[130:], expected[130:])                 # below the caption band
    assert draw.WANTED_THICKNESS > draw.BOX_THICKNESS
    assert not np.array_equal(plain, wanted)
    assert tuple(wanted[118, 101]) == draw.WANTED_COLOR                 # the caption band too (its padding)


def test_only_the_wanted_track_numbers_stand_out():
    chair = Detection("chair", 0.8, (100, 120, 300, 400), 2)
    keys = Detection("keys", 0.7, (400, 120, 500, 200), 100_007)
    untracked = Detection("keys", 0.6, (520, 300, 600, 380))
    frame = blank()
    draw_detections(frame, [chair, keys, untracked], wanted={100_007, None, 99})
    assert tuple(frame[300, 100]) == color_for("chair")
    assert tuple(frame[160, 400]) == draw.WANTED_COLOR
    assert tuple(frame[340, 520]) == color_for("keys")                  # no track number: never wanted


def test_the_wanted_colour_is_no_label_colour():
    # color_for() never gives a channel below 60, so no label can look like a wanted box.
    assert min(draw.WANTED_COLOR) < 60


def test_a_wanted_box_line_grows_with_the_frame():
    box = (300, 360, 900, 900)
    new = blank(1920, 1080)
    draw_detections(new, [Detection("keys", 0.87, box, 100_003)], wanted=[100_003])
    expected = blank(1920, 1080)
    cv2.rectangle(expected, box[:2], box[2:], draw.WANTED_COLOR, draw.WANTED_THICKNESS * 3)
    assert np.array_equal(new[400:], expected[400:])


def test_draw_status_writes_in_the_top_left_corner():
    frame = blank()
    draw_status(frame, "Detect FPS 15.0")
    assert frame[0:40, 0:200].any()
    assert not frame[100:, :].any()


# ---- drawing on a larger frame ---------------------------------------------------


def painted_rows(frame):
    """(first, last) row with anything drawn on it."""
    rows = np.flatnonzero(frame.any(axis=(1, 2)))
    return int(rows[0]), int(rows[-1])


def painted_columns(frame):
    columns = np.flatnonzero(frame.any(axis=(0, 2)))
    return int(columns[0]), int(columns[-1])


def test_drawing_scale_follows_the_frame_width_and_never_shrinks():
    assert draw.drawing_scale(640) == 1.0
    assert draw.drawing_scale(320) == 1.0                  # never below today's sizes
    assert draw.drawing_scale(1280) == 2.0
    assert draw.drawing_scale(1920) == 3.0


def caption_band_height(width, height):
    """How many rows the caption band of a box takes on a frame of this size."""
    frame = blank(width, height)
    draw.draw_caption(frame, "person 0.87 #3", width // 4, height // 2, (200, 200, 200))
    first, last = painted_rows(frame)
    return last - first + 1


def test_a_caption_at_1920_wide_is_about_three_times_as_tall_as_at_640():
    small, large = caption_band_height(640, 360), caption_band_height(1920, 1080)
    assert 2.5 <= large / small <= 3.5


def test_at_640_wide_the_drawing_is_exactly_as_before():
    """The same pixels as the drawing code before frames were scaled, written out by hand."""
    text, color = "person 0.87 #3", color_for("person")
    new = blank(640, 360)
    draw_detections(new, [Detection("person", 0.87, (100, 120, 300, 300), 3)])
    draw_status(new, "FPS 15.0")

    old = blank(640, 360)
    cv2.rectangle(old, (100, 120), (300, 300), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = 120 - (text_height + baseline + 6)
    cv2.rectangle(old, (100, top), (100 + text_width + 6, 120), color, cv2.FILLED)
    cv2.putText(old, text, (103, top + 3 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                draw.text_color_on(color), 1, cv2.LINE_AA)
    cv2.putText(old, "FPS 15.0", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(old, "FPS 15.0", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    assert np.array_equal(new, old)


def test_box_lines_grow_with_the_frame():
    box = (300, 360, 900, 900)
    new = blank(1920, 1080)
    draw_detections(new, [Detection("person", 0.87, box, 3)])
    expected = blank(1920, 1080)
    cv2.rectangle(expected, box[:2], box[2:], color_for("person"), 6)    # 3 times the 2 used at 640
    assert np.array_equal(new[400:], expected[400:])                     # below the caption


@pytest.mark.parametrize("box", [(0, 0, 1919, 1079), (1880, 500, 1919, 560), (0, 1060, 30, 1079),
                                 (1700, 0, 1919, 40)])
def test_at_1920_wide_the_caption_is_never_cut_off_at_an_edge(box):
    text = caption_for(Detection("traffic light", 0.66, box, 4))
    (text_width, text_height), baseline = cv2.getTextSize(text, draw.FONT, 0.5 * 3, 3)
    band = blank(1920, 1080)
    draw.draw_caption(band, text, box[0], box[1], (200, 200, 200))
    left, right = painted_columns(band)
    top, bottom = painted_rows(band)
    # The band is the text plus 9 pixels of padding on each side (3 at 640): all of it is
    # in the frame. (At the right and bottom edges its last line of pixels falls just outside,
    # as it always did at 640.)
    assert right - left + 1 >= text_width + 2 * 9
    assert bottom - top + 1 >= text_height + baseline + 2 * 9


def test_the_status_line_grows_with_the_frame():
    small, large = blank(640, 360), blank(1920, 1080)
    draw_status(small, "Detect FPS 15.0")
    draw_status(large, "Detect FPS 15.0")
    small_top, small_bottom = painted_rows(small)
    large_top, large_bottom = painted_rows(large)
    assert 2.5 <= (large_bottom - large_top) / (small_bottom - small_top) <= 3.5
    assert painted_columns(large)[1] < 1919 and large_bottom < 1080


@pytest.mark.parametrize("corner", ["top-left", "bottom-right"])
def test_at_1920_wide_the_info_card_is_three_times_as_wide_and_stays_inside(corner):
    width, height, size = 1920, 1080, 240
    left = 0 if corner == "top-left" else width - 1 - size
    top = 0 if corner == "top-left" else height - 1 - size
    face = RecognizedFace("Sara Ahmed", 0.7, (left, top, left + size, top + size), "sara")
    frame = blank(width, height)
    draw.draw_info_card(frame, face, Person("sara", "Sara Ahmed", "Team lead", ("Built the chassis",)))
    x1, x2 = painted_columns(frame)
    y1, y2 = painted_rows(frame)
    assert x2 - x1 + 1 == draw.CARD_WIDTH * 3
    assert 0 <= x1 and x2 <= width - 1 and 0 <= y1 and y2 <= height - 1
    middle = (y1 + y2) // 2
    assert all(tuple(frame[middle, x]) == draw.KNOWN_FACE_COLOR for x in (x1, x1 + 2, x2 - 2, x2))   # 3 px border
    assert tuple(frame[middle, x1 + 3]) == tuple(frame[middle, x2 - 3]) == draw.CARD_BACKGROUND
    small = blank(640, 360)
    draw.draw_info_card(small, RecognizedFace("Sara Ahmed", 0.7, (0, 0, 80, 80), "sara"),
                        Person("sara", "Sara Ahmed", "Team lead", ("Built the chassis",)))
    small_y1, small_y2 = painted_rows(small)
    assert 2.5 <= (y2 - y1) / (small_y2 - small_y1) <= 3.5


ARABIC_NAME = "سارة"          # Sara, in Arabic letters


def test_non_ascii_text_is_replaced_by_the_id_or_left_off_the_card():
    person = Person("sara", ARABIC_NAME, "Team lead",
                    ("Built the chassis", "طالبة", "Café owner", "fourth"))
    assert draw.drawable("Sara Ahmed") and not draw.drawable(ARABIC_NAME)
    rows = [row[0] for row in draw._card_rows(person, 200)]
    assert rows == ["sara", "Team lead", "Built the chassis"]          # of the first three lines


def test_a_face_with_a_non_ascii_name_is_captioned_with_its_id(monkeypatch):
    captions = []
    monkeypatch.setattr(draw, "draw_caption", lambda frame, text, x, y, color: captions.append(text))
    faces = [RecognizedFace(ARABIC_NAME, 0.7, (10, 10, 60, 60), "sara"),
             RecognizedFace("Omar Ali", 0.7, (100, 10, 160, 60), "omar")]
    draw.draw_faces(blank(), faces, show_scores=True)
    assert captions == ["sara 0.70", "Omar Ali 0.70"]


def test_very_long_card_text_is_cut_before_it_is_measured(monkeypatch):
    measured = []
    real_size = cv2.getTextSize

    def counting_size(text, *args):
        measured.append(len(text))
        return real_size(text, *args)

    monkeypatch.setattr(cv2, "getTextSize", counting_size)
    cut = draw._fit_text("x" * 3000, 200, draw.CARD_TEXT_SCALE, draw.CARD_TEXT_THICKNESS)
    assert cut.endswith("...")
    assert max(measured) <= draw.CARD_MAX_CHARS + len("...")
    assert len(measured) <= draw.CARD_MAX_CHARS + 1


def test_draw_module_imports_only_cv2_numpy_and_state():
    text = (ROOT / "car_vision" / "draw.py").read_text(encoding="utf-8")
    imports = [line.strip() for line in text.splitlines() if line.startswith(("import ", "from "))]
    allowed = {"from __future__ import annotations", "from collections.abc import Iterable, Sequence",
               "import cv2", "import numpy as np", "from car_vision.faces import Person",
               "from car_vision.state import Detection, RecognizedFace"}
    assert set(imports) <= allowed
