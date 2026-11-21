"""The recognizer and the face drawing, tested with a fake face model and made-up vectors.

No camera, network, graphics card or real photos are needed.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from car_vision import draw
from car_vision.config import ConfigError
from car_vision.faces import FaceMatcher, Person
from car_vision.recognizer import Recognizer, choose_providers, describe_faces, track_for
from car_vision.state import UNKNOWN, Detection, RecognizedFace

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "car_vision"
SIZE = 8

SETTINGS = {
    "model_pack": "buffalo_l",
    "models_dir": "models",
    "detection_size": 640,
    "device": "cpu",
    "match_threshold": 0.42,
    "every_n_frames": 5,
    "min_face_pixels": 40,
    "people_file": "data/does-not-exist/people.json",
    "photos_dir": "data/does-not-exist/people",
    "embeddings_file": "data/does-not-exist/embeddings.npz",
}

PEOPLE = {
    "sara": Person("sara", "Sara Ahmed", "Team lead", ("Built the car chassis",)),
    "omar": Person("omar", "Omar Ali", "Driver"),
}


def one_hot(index, noise=0.0, seed=0):
    vector = np.zeros(SIZE, dtype=np.float32)
    vector[index] = 1.0
    if noise:
        vector += np.random.default_rng(seed).normal(0.0, noise, SIZE).astype(np.float32)
    return vector


SARA = one_hot(0)
OMAR = one_hot(1)
STRANGER = one_hot(5)


class FakeFace:
    """Shaped like an InsightFace face: bbox (x1, y1, x2, y2) and normed_embedding."""

    def __init__(self, box, embedding):
        self.bbox = np.asarray(box, dtype=np.float32)
        self.normed_embedding = np.asarray(embedding, dtype=np.float32)


class FakeFaceModel:
    """Returns the same prepared faces for every frame and counts the calls."""

    def __init__(self, *faces):
        self.faces = list(faces)
        self.calls = 0

    def get(self, image):
        self.calls += 1
        return list(self.faces)


def recognizer(*faces, people=None):
    matcher = FaceMatcher(["sara", "sara", "omar"],
                          np.stack([SARA, one_hot(0, 0.1), OMAR]), threshold=0.42)
    return Recognizer(SETTINGS, face_model=FakeFaceModel(*faces),
                      people=PEOPLE if people is None else people, matcher=matcher)


def frame(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---- naming faces -------------------------------------------------------------------


def test_two_different_known_people_in_one_frame():
    found = recognizer(FakeFace((100, 100, 180, 200), one_hot(0, 0.05)),
                       FakeFace((400, 100, 480, 200), one_hot(1, 0.05))).recognize(frame())
    assert [f.name for f in found] == ["Sara Ahmed", "Omar Ali"]
    assert [f.person_id for f in found] == ["sara", "omar"]
    assert all(f.similarity > 0.9 for f in found)
    assert all(isinstance(f.similarity, float) for f in found)


def test_one_known_and_one_unknown():
    found = recognizer(FakeFace((100, 100, 180, 200), SARA),
                       FakeFace((400, 100, 480, 200), STRANGER)).recognize(frame())
    assert [(f.name, f.person_id) for f in found] == [("Sara Ahmed", "sara"), (UNKNOWN, None)]
    assert found[1].similarity < 0.42                 # its score is still reported
    assert not found[1].is_known


def test_two_faces_matching_the_same_person_only_the_better_keeps_the_name():
    weaker = one_hot(0) + 0.6 * one_hot(3)
    found = recognizer(FakeFace((100, 100, 180, 200), weaker),
                       FakeFace((400, 100, 480, 200), SARA)).recognize(frame())
    assert [f.name for f in found] == [UNKNOWN, "Sara Ahmed"]
    assert found[0].person_id is None
    assert 0.42 < found[0].similarity < found[1].similarity      # it keeps its own score
    assert found[1].similarity == pytest.approx(1.0)


def test_face_smaller_than_min_face_pixels_is_skipped():
    found = recognizer(FakeFace((10, 10, 49, 200), SARA),       # 39 pixels wide
                       FakeFace((300, 300, 400, 339), OMAR),    # 39 pixels tall
                       FakeFace((500, 100, 540, 140), OMAR)).recognize(frame())
    assert [f.box for f in found] == [(500, 100, 540, 140)]


def test_boxes_are_clipped_to_the_frame_as_ints():
    found = recognizer(FakeFace((-20.4, -5.0, 90.6, 120.2), SARA),
                       FakeFace((600.0, 300.0, 700.0, 400.0), OMAR)).recognize(frame(640, 360))
    assert found[0].box == (0, 0, 91, 120)
    assert found[1].box == (600, 300, 639, 359)
    assert all(type(v) is int for f in found for v in f.box)


def test_face_without_an_embedding_is_unknown():
    face = FakeFace((100, 100, 180, 200), SARA)
    face.normed_embedding = None
    found = recognizer(face).recognize(frame())
    assert [(f.name, f.similarity) for f in found] == [(UNKNOWN, 0.0)]


def test_no_faces_gives_an_empty_list():
    assert recognizer().recognize(frame()) == []


def test_empty_frame_raises_value_error():
    with pytest.raises(ValueError):
        recognizer().recognize(np.zeros((0, 0, 3), np.uint8))


def test_person_returns_the_info_card():
    found = recognizer()
    assert found.person("sara") is PEOPLE["sara"]
    assert found.person("nobody") is None


# ---- track ids ------------------------------------------------------------------------


def person_box(track_id, box, label="person"):
    return Detection(label, 0.9, box, track_id)


def test_track_ids_come_from_the_person_box_around_the_face():
    detections = [person_box(1, (50, 50, 250, 470)), person_box(2, (350, 50, 550, 470))]
    found = recognizer(FakeFace((100, 80, 180, 180), SARA),
                       FakeFace((400, 80, 480, 180), OMAR)).recognize(frame(), detections)
    assert [f.track_id for f in found] == [1, 2]


def test_nested_person_boxes_give_the_smallest_one():
    # A child in front of an adult: both boxes contain the face; the smaller one is the child's.
    detections = [person_box(7, (0, 0, 600, 479)), person_box(8, (80, 60, 220, 300)),
                  person_box(9, (90, 70, 210, 250), label="chair")]
    found = recognizer(FakeFace((100, 80, 180, 180), SARA)).recognize(frame(), detections)
    assert found[0].track_id == 8


def test_no_detections_and_no_containing_box_give_none():
    face = FakeFace((100, 80, 180, 180), SARA)
    assert recognizer(face).recognize(frame())[0].track_id is None
    elsewhere = [person_box(3, (400, 0, 600, 400))]
    assert recognizer(face).recognize(frame(), elsewhere)[0].track_id is None


def test_track_for_counts_the_box_edge_as_inside():
    assert track_for((0, 0, 20, 20), [person_box(5, (10, 10, 100, 100))]) == 5


# ---- start-up --------------------------------------------------------------------------


def test_mismatch_between_people_and_saved_faces_gives_one_warning(caplog):
    people = {"sara": PEOPLE["sara"], "lina": Person("lina", "Lina Haddad")}
    matcher = FaceMatcher(["sara", "ghost"], np.stack([SARA, OMAR]), threshold=0.42)
    with caplog.at_level(logging.WARNING, logger="car_vision.recognizer"):
        made = Recognizer(SETTINGS, face_model=FakeFaceModel(FakeFace((10, 10, 90, 90), OMAR)),
                          people=people, matcher=matcher)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "lina" in warnings[0].getMessage() and "ghost" in warnings[0].getMessage()
    # It carries on: a saved face with no card is shown by its id.
    assert made.recognize(frame())[0].name == "ghost"


def test_matching_people_and_faces_give_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="car_vision.recognizer"):
        recognizer()
    assert not caplog.records


def test_missing_people_file_is_a_config_error():
    with pytest.raises(ConfigError, match="People file not found"):
        Recognizer(SETTINGS, face_model=FakeFaceModel())


def test_missing_setting_is_reported():
    settings = dict(SETTINGS)
    del settings["min_face_pixels"]
    with pytest.raises(ConfigError, match="min_face_pixels"):
        Recognizer(settings, face_model=FakeFaceModel(), people=PEOPLE,
                   matcher=FaceMatcher([], np.zeros((0, SIZE)), 0.42))


def test_choose_providers():
    both = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert choose_providers("auto", both) == both
    assert choose_providers("cuda", both) == both
    assert choose_providers("cpu", both) == ["CPUExecutionProvider"]
    assert choose_providers("auto", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert choose_providers("cuda", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    with pytest.raises(ConfigError):
        choose_providers("gpu", both)


def test_describe_faces_keeps_the_best_score_per_name():
    faces = [RecognizedFace("Sara Ahmed", 0.61, (0, 0, 1, 1), "sara"),
             RecognizedFace("Sara Ahmed", 0.72, (0, 0, 1, 1), "sara"),
             RecognizedFace(UNKNOWN, 0.2, (0, 0, 1, 1))]
    assert describe_faces(faces) == "Sara Ahmed 0.72, Unknown 0.20"
    assert describe_faces([]) == "no faces"


# ---- which files may use the face libraries ----------------------------------------------


def test_only_recognizer_and_enroll_import_insightface_or_onnxruntime():
    face_import = re.compile(r"^\s*(import|from)\s+(insightface|onnxruntime)\b", re.MULTILINE)
    offenders = [
        path.name
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        if path.name not in ("recognizer.py", "enroll.py")
        and face_import.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_importing_the_recognizer_with_a_fake_model_loads_nothing_heavy():
    code = (
        "import sys, numpy as np\n"
        "import car_vision.recognizer, car_vision.enroll\n"
        "print(sorted(m for m in ('insightface', 'onnxruntime', 'torch', 'ultralytics')"
        " if m in sys.modules))\n"
    )
    output = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                            text=True, timeout=60, check=True).stdout
    assert output.strip() == "[]"


# ---- drawing faces and info cards -----------------------------------------------------------


def blank(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)


def painted_box(image):
    """(left, top, right, bottom) of every pixel that is not black."""
    rows = np.flatnonzero(image.any(axis=(1, 2)))
    cols = np.flatnonzero(image.any(axis=(0, 2)))
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


LONG_CARD = Person("sara", "Sara Ahmed", "Team lead",
                   ("Built the car chassis", "Fourth-year student", "Likes robots", "A fourth line not shown"))


@pytest.mark.parametrize("width, height", [(640, 480), (640, 360)])
@pytest.mark.parametrize("corner", ["top-left", "top-right", "bottom-left", "bottom-right"])
def test_info_card_stays_inside_the_frame_in_every_corner(width, height, corner):
    size = 80
    left = 0 if "left" in corner else width - 1 - size
    top = 0 if "top" in corner else height - 1 - size
    face = RecognizedFace("Sara Ahmed", 0.7, (left, top, left + size, top + size), "sara")

    image = blank(width, height)
    draw.draw_info_card(image, face, LONG_CARD)

    x1, y1, x2, y2 = painted_box(image)
    assert x2 - x1 + 1 == draw.CARD_WIDTH            # the whole card is visible, not cut off
    assert 0 <= x1 and x2 <= width - 1 and 0 <= y1 and y2 <= height - 1
    # Beside the face, not on top of it: right of a left face, left of a right face.
    if "left" in corner:
        assert x1 > left + size
    else:
        assert x2 < left
    # The card border is drawn on all four sides, so nothing was clipped.
    assert image[y1, x1:x2 + 1].any(axis=1).all() and image[y2, x1:x2 + 1].any(axis=1).all()
    assert image[y1:y2 + 1, x1].any(axis=1).all() and image[y1:y2 + 1, x2].any(axis=1).all()


def test_info_card_in_a_frame_too_small_for_it_still_fits():
    image = blank(160, 60)
    face = RecognizedFace("Sara Ahmed", 0.7, (40, 10, 100, 50), "sara")
    draw.draw_info_card(image, face, LONG_CARD)
    x1, y1, x2, y2 = painted_box(image)
    assert (x1, x2) == (0, 159)
    assert 0 <= y1 and y2 <= 59


def test_long_lines_are_cut_with_dots():
    text = "A very long line about building the robot car chassis from scratch"
    cut = draw._fit_text(text, 150, draw.CARD_TEXT_SCALE, draw.CARD_TEXT_THICKNESS)
    assert cut.endswith("...") and len(cut) < len(text)
    assert draw._fit_text("Short", 150, draw.CARD_TEXT_SCALE, draw.CARD_TEXT_THICKNESS) == "Short"


def test_card_shows_at_most_three_lines():
    rows = draw._card_rows(LONG_CARD, 200)
    assert [row[0] for row in rows] == ["Sara Ahmed", "Team lead", "Built the car chassis",
                                        "Fourth-year student", "Likes robots"]


def test_draw_faces_green_for_known_grey_for_unknown():
    image = blank(640, 480)
    known = RecognizedFace("Sara Ahmed", 0.7, (100, 100, 200, 200), "sara")
    unknown = RecognizedFace(UNKNOWN, 0.1, (400, 100, 500, 200))
    draw.draw_faces(image, [known, unknown], show_scores=True)
    assert tuple(image[150, 100]) == draw.KNOWN_FACE_COLOR       # left edge of the known box
    assert tuple(image[150, 400]) == draw.UNKNOWN_FACE_COLOR
    assert image[80:100, 100:200].any()                          # caption above the box


def test_draw_faces_with_nothing_leaves_the_frame_untouched():
    image = blank(320, 240)
    draw.draw_faces(image, [])
    assert not image.any()
