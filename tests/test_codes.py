"""M16 Parts A and B: reading QR codes and barcodes, and drawing their outline.

The round trip uses OpenCV's real QR detector on a picture made in memory:
no camera, model file, network or graphics card. Everything else uses fakes.
"""

import logging

import cv2
import numpy as np
import pytest

from car_vision.codes import MAX_TEXT_CHARS, Code, CodeReader, clean_text, web_link
from car_vision.config import ConfigError
from car_vision.draw import CODE_COLOR, draw_codes

SETTINGS = {"enabled": True, "every_n_frames": 5, "max_width": 960, "hold_seconds": 1.5}
GREY = 128
SECRET = "https://example.com/menu?table=7"      # must never reach the log


def qr_picture(text, module=8):
    """A QR code for `text`, each module `module` pixels wide, with its white quiet zone."""
    code = cv2.QRCodeEncoder.create().encode(text)
    code = cv2.resize(code, None, fx=module, fy=module, interpolation=cv2.INTER_NEAREST)
    quiet = 4 * module
    code = cv2.copyMakeBorder(code, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)


def pasted(picture, size, at):
    """A grey frame of this (width, height) with the picture pasted at (left, top)."""
    width, height = size
    frame = np.full((height, width, 3), GREY, dtype=np.uint8)
    left, top = at
    frame[top:top + picture.shape[0], left:left + picture.shape[1]] = picture
    return frame


def around(box, outer, slack=0):
    """True when the box lies inside the outer box (grown by `slack` pixels)."""
    return (box[0] >= outer[0] - slack and box[1] >= outer[1] - slack
            and box[2] <= outer[2] + slack and box[3] <= outer[3] + slack)


# ---- fakes ---------------------------------------------------------------------


class FakeQr:
    """Answers like QRCodeDetector.detectAndDecodeMulti: (ok, texts, points, straight codes)."""

    def __init__(self, finds=(), fail=False):
        self.finds = list(finds)                   # (text, four corners)
        self.fail = fail
        self.seen = []                             # the shape of every picture it was given

    def detectAndDecodeMulti(self, picture):
        self.seen.append(picture.shape)
        if self.fail:
            raise cv2.error("the QR detector broke")
        if not self.finds:
            return False, (), None, ()
        texts = tuple(text for text, _corners in self.finds)
        points = np.array([corners for _text, corners in self.finds], dtype=np.float32)
        return True, texts, points, ()


class FakeBarcodes:
    """Answers like BarcodeDetector.detectAndDecodeWithType: (ok, texts, formats, points)."""

    def __init__(self, finds=(), fail=False):
        self.finds = list(finds)                   # (text, format, four corners)
        self.fail = fail

    def detectAndDecodeWithType(self, picture):
        if self.fail:
            raise cv2.error("the barcode detector broke")
        if not self.finds:
            return False, (), (), None
        texts = tuple(text for text, _form, _corners in self.finds)
        forms = tuple(form for _text, form, _corners in self.finds)
        points = np.array([corners for _text, _form, corners in self.finds], dtype=np.float32)
        return True, texts, forms, points


def square(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def frame_of(width=640, height=480):
    return np.full((height, width, 3), GREY, dtype=np.uint8)


# ---- the real QR detector: a round trip without a camera ------------------------------


def test_a_qr_code_pasted_into_a_frame_is_read_with_its_text_kind_and_box():
    picture = qr_picture(SECRET)
    frame = pasted(picture, (640, 480), (100, 50))
    codes = CodeReader(SETTINGS).read(frame)
    assert len(codes) == 1
    code = codes[0]
    assert (code.kind, code.text, code.format) == ("qr", SECRET, "")
    where = (100, 50, 100 + picture.shape[1], 50 + picture.shape[0])
    assert around(code.box, where)
    # The box is the code itself, not the whole pasted square with its quiet zone.
    assert code.box[2] - code.box[0] > picture.shape[1] / 2
    assert len(code.corners) == 4 and all(around((x, y, x, y), code.box) for x, y in code.corners)


def test_a_frame_wider_than_max_width_is_shrunk_and_the_box_is_in_its_own_pixels():
    picture = qr_picture(SECRET, module=16)
    frame = pasted(picture, (1920, 1080), (900, 300))
    qr = cv2.QRCodeDetector()
    seen = []

    class Watching:                                  # the real detector, noting the picture's size
        def detectAndDecodeMulti(self, small):
            seen.append(small.shape[:2])
            return qr.detectAndDecodeMulti(small)

    codes = CodeReader(SETTINGS, qr=Watching()).read(frame)
    assert seen == [(540, 960)]                       # searched at max_width, shape kept
    assert [code.text for code in codes] == [SECRET]
    where = (900, 300, 900 + picture.shape[1], 300 + picture.shape[0])
    assert around(codes[0].box, where, slack=4)       # back in the 1920 x 1080 frame's pixels
    assert codes[0].box[0] > 960 / 2                  # not left in the shrunk picture's pixels


def test_a_frame_with_no_code_gives_none():
    assert CodeReader(SETTINGS).read(frame_of()) == []


# Two pictures the plain cv2.QRCodeDetector did not read (checked with OpenCV 4.14): the
# ArUco-based detector the reader uses reads both.


def test_a_300_pixel_code_on_a_full_hd_frame_is_read():
    picture = qr_picture(SECRET, module=10)                  # the code itself is 290 pixels wide
    frame = pasted(picture, (1920, 1080), (777, 123))
    codes = CodeReader(SETTINGS).read(frame)
    assert [(code.kind, code.text) for code in codes] == [("qr", SECRET)]
    assert around(codes[0].box, (777, 123, 777 + picture.shape[1], 123 + picture.shape[0]), slack=4)


def test_a_small_code_on_a_640_by_360_frame_is_read():
    picture = qr_picture(SECRET, module=3)                   # the code itself is 87 pixels wide
    frame = pasted(picture, (640, 360), (400, 100))
    codes = CodeReader(SETTINGS).read(frame)
    assert [code.text for code in codes] == [SECRET]
    assert around(codes[0].box, (400, 100, 400 + picture.shape[1], 100 + picture.shape[0]))


def test_grey_and_bgra_frames_are_read_too():
    frame = pasted(qr_picture(SECRET), (640, 480), (100, 50))
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    for picture in (grey, bgra):
        assert [code.text for code in CodeReader(SETTINGS).read(picture)] == [SECRET]


def test_an_empty_frame_gives_none_without_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="car_vision.codes"):
        assert CodeReader(SETTINGS).read(np.zeros((0, 0, 3), np.uint8)) == []
    assert caplog.records == []


# ---- with fakes ------------------------------------------------------------------------


def test_a_barcode_comes_with_its_format():
    barcodes = FakeBarcodes([("5901234123457", "EAN_13", square(50, 60, 250, 140))])
    codes = CodeReader(SETTINGS, qr=FakeQr(), barcodes=barcodes).read(frame_of())
    assert codes == [Code("barcode", "5901234123457", (50, 60, 250, 140),
                          ((50, 60), (250, 60), (250, 140), (50, 140)), "EAN_13")]


def test_a_code_that_was_found_but_not_decoded_is_left_out():
    qr = FakeQr([("", square(10, 10, 90, 90)), ("menu", square(200, 10, 290, 90))])
    barcodes = FakeBarcodes([("", "", square(10, 200, 200, 260))])
    codes = CodeReader(SETTINGS, qr=qr, barcodes=barcodes).read(frame_of())
    assert [code.text for code in codes] == ["menu"]


def test_the_same_code_twice_counts_once_but_two_stickers_apart_count_twice():
    qr = FakeQr([("menu", square(10, 10, 90, 90)), ("menu", square(12, 12, 92, 92)),
                 ("menu", square(300, 10, 380, 90))])
    codes = CodeReader(SETTINGS, qr=qr, barcodes=FakeBarcodes()).read(frame_of())
    assert [code.box for code in codes] == [(10, 10, 90, 90), (300, 10, 380, 90)]


def test_the_same_text_as_a_qr_code_and_a_barcode_is_two_codes():
    qr = FakeQr([("12345", square(10, 10, 90, 90))])
    barcodes = FakeBarcodes([("12345", "CODE_128", square(10, 10, 90, 90))])
    codes = CodeReader(SETTINGS, qr=qr, barcodes=barcodes).read(frame_of())
    assert [code.kind for code in codes] == ["qr", "barcode"]


def test_corners_outside_the_frame_are_kept_inside_it():
    qr = FakeQr([("menu", [[-20, -5], [700, -5], [700, 500], [-20, 500]])])
    code, = CodeReader(SETTINGS, qr=qr, barcodes=FakeBarcodes()).read(frame_of(640, 480))
    assert code.box == (0, 0, 639, 479)
    assert all(0 <= x <= 639 and 0 <= y <= 479 for x, y in code.corners)


def test_the_text_is_cleaned():
    qr = FakeQr([("  line one\r\nline two\x00  ", square(10, 10, 90, 90))])
    code, = CodeReader(SETTINGS, qr=qr, barcodes=FakeBarcodes()).read(frame_of())
    assert code.text == "line one line two"


def test_a_detector_that_raises_gives_no_codes_and_one_warning_for_many_frames(caplog):
    qr = FakeQr(fail=True)
    barcodes = FakeBarcodes([("5901234123457", "EAN_13", square(50, 60, 250, 140))])
    reader = CodeReader(SETTINGS, qr=qr, barcodes=barcodes)
    with caplog.at_level(logging.WARNING, logger="car_vision.codes"):
        for _ in range(20):
            codes = reader.read(frame_of())
            assert [code.kind for code in codes] == ["barcode"]   # the other kind still works
    assert len(caplog.records) == 1
    assert "QR" in caplog.records[0].getMessage()


def test_each_kind_warns_once_of_its_own():
    reader = CodeReader(SETTINGS, qr=FakeQr(fail=True), barcodes=FakeBarcodes(fail=True))
    logged = []
    handler = logging.Handler()
    handler.emit = logged.append
    logger = logging.getLogger("car_vision.codes")
    logger.addHandler(handler)
    try:
        for _ in range(5):
            assert reader.read(frame_of()) == []
    finally:
        logger.removeHandler(handler)
    assert len(logged) == 2


def test_an_answer_of_ok_with_no_points_gives_no_codes():
    class OkButNothing:
        def detectAndDecodeMulti(self, picture):
            return True, ("menu",), None, ()

    class OkButNothingEither:
        def detectAndDecodeWithType(self, picture):
            return True, ("123",), ("EAN_13",), None

    reader = CodeReader(SETTINGS, qr=OkButNothing(), barcodes=OkButNothingEither())
    assert reader.read(frame_of()) == []


def test_an_answer_of_an_unexpected_shape_does_not_leave_read():
    class Odd:
        def detectAndDecodeMulti(self, picture):
            return (True,)                          # too short

    assert CodeReader(SETTINGS, qr=Odd(), barcodes=FakeBarcodes()).read(frame_of()) == []


def test_the_text_never_reaches_the_log(caplog):
    qr = FakeQr([(SECRET, square(10, 10, 90, 90)), ("", square(200, 10, 290, 90))])
    barcodes = FakeBarcodes(fail=True)
    with caplog.at_level(logging.DEBUG):
        reader = CodeReader(SETTINGS, qr=qr, barcodes=barcodes)
        for _ in range(3):
            reader.read(frame_of())
    assert caplog.records                               # the barcode failure was logged ...
    assert SECRET not in caplog.text and "example.com" not in caplog.text   # ... without any text


def test_the_reader_builds_the_opencv_detectors_itself():
    reader = CodeReader(SETTINGS)
    assert isinstance(reader._qr, cv2.QRCodeDetectorAruco)       # not a kind of cv2.QRCodeDetector
    assert isinstance(reader._barcodes, cv2.barcode.BarcodeDetector)


# ---- altered: the text is not exactly what the code holds --------------------------------


def read_one(text):
    qr = FakeQr([(text, square(10, 10, 90, 90))])
    code, = CodeReader(SETTINGS, qr=qr, barcodes=FakeBarcodes()).read(frame_of())
    return code


def test_a_text_kept_as_it_is_is_not_altered():
    assert read_one(SECRET).altered is False
    assert read_one("  " + SECRET + "\n").altered is False         # only the spaces around it went


def test_a_text_cut_at_500_characters_is_altered():
    long_address = "https://example.com/" + "a" * 600
    code = read_one(long_address)
    assert len(code.text) == MAX_TEXT_CHARS and code.altered is True


def test_a_removed_control_character_or_line_break_makes_the_text_altered():
    assert read_one("https://exa\x00mple.com/").altered is True
    assert read_one("https://example.com/‮").altered is True
    assert read_one("one\ntwo").altered is True


# ---- settings ------------------------------------------------------------------------


@pytest.mark.parametrize("key, value", [
    ("every_n_frames", 0), ("every_n_frames", 2.5), ("every_n_frames", True),
    ("max_width", 159), ("max_width", "960"), ("hold_seconds", -0.1), ("hold_seconds", "1"),
])
def test_wrong_settings_are_refused(key, value):
    with pytest.raises(ConfigError):
        CodeReader({**SETTINGS, key: value}, qr=FakeQr(), barcodes=FakeBarcodes())


@pytest.mark.parametrize("key", ["every_n_frames", "max_width", "hold_seconds"])
def test_a_missing_setting_is_refused(key):
    settings = {name: value for name, value in SETTINGS.items() if name != key}
    with pytest.raises(ConfigError):
        CodeReader(settings, qr=FakeQr(), barcodes=FakeBarcodes())


def test_the_smallest_allowed_settings_are_taken():
    reader = CodeReader({**SETTINGS, "every_n_frames": 1, "max_width": 160, "hold_seconds": 0},
                        qr=FakeQr(), barcodes=FakeBarcodes())
    assert reader.max_width == 160


# ---- clean_text and web_link ----------------------------------------------------------


@pytest.mark.parametrize("raw, clean", [
    ("menu", "menu"),
    ("  padded  ", "padded"),
    ("one\ntwo", "one two"),
    ("one\r\n\r\ntwo\tthree", "one two three"),
    ("bell\x07 and null\x00 and escape\x1b[31m", "bell and null and escape[31m"),
    ("right\u202eto left", "rightto left"),            # an invisible direction change is removed
    ("\n\t \x00", ""),
    ("caf\u00e9 \u2615", "caf\u00e9 \u2615"),          # letters of every kind stay
])
def test_clean_text(raw, clean):
    assert clean_text(raw) == clean


def test_clean_text_cuts_long_text():
    assert clean_text("a" * 2000) == "a" * MAX_TEXT_CHARS
    assert len(clean_text("word " * 400)) <= MAX_TEXT_CHARS


@pytest.mark.parametrize("raw", [None, 42, b"bytes", ["a"], 3.5])
def test_clean_text_of_anything_but_a_string_is_empty(raw):
    assert clean_text(raw) == ""


@pytest.mark.parametrize("text", [
    "https://example.com/menu", "http://example.com", "https://example.com:8443/a?b=c#d",
    "https://www.example.org/path/to/page.html",
])
def test_a_plain_web_address_is_a_link(text):
    assert web_link(text) == text


@pytest.mark.parametrize("text", [
    "javascript:alert(1)", "JavaScript:alert(1)", "data:text/html,<b>hi</b>", "file:///C:/Windows/win.ini",
    "tel:+3312345678", "mailto:someone@example.com", "WIFI:S:home;T:WPA;P:secret;;", "WIFI:S:home;;",
    "BEGIN:VCARD\nFN:Someone\nEND:VCARD", "MECARD:N:Someone;;",
    "https://user:pw@host/", "https://user@example.com/", "https://@example.com/",
    "http://example.com/a b", " https://example.com", "https://example.com\n",
    "https://", "https:///path", "http:example.com", "example.com", "word", "",
    "https://example.com:port/", "https://exa<mple>.com/", "https://evil.example\\@good.example/",
    "ftp://example.com/file",
])
def test_anything_else_is_not_a_link(text):
    assert web_link(text) is None


def test_web_link_of_anything_but_a_string_is_none():
    assert web_link(None) is None
    assert web_link(42) is None


# ---- draw_codes ---------------------------------------------------------------------


def test_the_outline_has_the_code_colour_along_an_edge_from_corner_to_corner():
    frame = frame_of()
    code = Code("qr", SECRET, (100, 100, 300, 260), ((100, 100), (300, 100), (300, 260), (100, 260)))
    draw_codes(frame, [code])
    assert tuple(frame[180, 300]) == CODE_COLOR          # the middle of the right edge
    assert tuple(frame[260, 200]) == CODE_COLOR          # the middle of the bottom edge
    assert tuple(frame[180, 200]) == (GREY, GREY, GREY)  # nothing drawn inside the code


def test_a_tilted_code_is_outlined_through_its_corners():
    frame = frame_of()
    corners = ((200, 100), (300, 200), (200, 300), (100, 200))
    draw_codes(frame, [Code("qr", "menu", (100, 100, 300, 300), corners)])
    assert tuple(frame[150, 250]) == CODE_COLOR          # halfway from the top corner to the right one
    assert tuple(frame[110, 110]) == (GREY, GREY, GREY)  # the box's corner is not drawn


def test_without_four_corners_the_box_is_outlined():
    frame = frame_of()
    draw_codes(frame, [Code("barcode", "123", (100, 300, 400, 380), (), "EAN_13")])
    assert tuple(frame[380, 250]) == CODE_COLOR


def captions_drawn(monkeypatch, code):
    import car_vision.draw as draw_module
    captions = []
    monkeypatch.setattr(draw_module, "draw_caption", lambda frame, text, x, y, color: captions.append((text, x, y, color)))
    draw_codes(frame_of(), [code])
    return captions


def test_the_caption_says_the_kind_and_never_the_text(monkeypatch):
    qr = Code("qr", SECRET, (100, 100, 300, 260), ((100, 100), (300, 100), (300, 260), (100, 260)))
    barcode = Code("barcode", "5901234123457", (50, 300, 250, 380), (), "EAN_13")
    assert captions_drawn(monkeypatch, qr) == [("QR code", 100, 100, CODE_COLOR)]
    assert captions_drawn(monkeypatch, barcode) == [("barcode", 50, 300, CODE_COLOR)]


def test_a_code_at_the_frame_edge_is_drawn_without_error():
    frame = frame_of()
    corners = ((0, 0), (639, 0), (639, 479), (0, 479))
    draw_codes(frame, [Code("qr", "menu", (0, 0, 639, 479), corners),
                       Code("barcode", "1", (600, 450, 639, 479), ())])
    assert tuple(frame[240, 0]) == CODE_COLOR


def test_no_codes_draws_nothing():
    frame = frame_of()
    draw_codes(frame, [])
    assert (frame == GREY).all()
