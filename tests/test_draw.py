"""The drawing helpers, tested on blank frames."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

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


def test_draw_status_writes_in_the_top_left_corner():
    frame = blank()
    draw_status(frame, "Detect FPS 15.0")
    assert frame[0:40, 0:200].any()
    assert not frame[100:, :].any()


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
    allowed = {"from __future__ import annotations", "from collections.abc import Iterable",
               "import cv2", "import numpy as np", "from car_vision.faces import Person",
               "from car_vision.state import Detection, RecognizedFace"}
    assert set(imports) <= allowed
