"""Reads the QR codes and barcodes in a camera frame, on this laptop, with OpenCV alone.

No model file, no internet, nothing is sent anywhere: OpenCV's own QR code
detector and barcode detector look at the picture.

The text of a code was written by a stranger. This module hands it to the
pipeline, which shows it to the visitor and nothing else. It is never
logged here: a warning says only which kind of detector failed.

    reader = CodeReader(config["codes"])
    for code in reader.read(frame):
        print(code.kind, code.box)          # "qr" (or "barcode") and where it is

clean_text() makes a code's text safe to keep (no control characters, at
most MAX_TEXT_CHARS long); web_link() says whether it is a plain web address
the page may offer as a link.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from car_vision.config import ConfigError, require
from car_vision.state import Box

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 500            # a code's text is cut to this many characters
MIN_SEARCH_WIDTH = 160          # codes.max_width must be at least this many pixels
WEB_SCHEMES = ("http", "https")
LINE_BREAKS = re.compile(r"[\t\n\r\v\f\x1c-\x1e\x85  ]+")   # each run becomes one space
DROPPED_KINDS = ("Cc", "Cf", "Cs")   # control, invisible format (e.g. right-to-left override) and broken characters
HOST = re.compile(r"[\w.-]+|[0-9a-f:.]+")   # a host name (any script), or an IP address


@dataclass(frozen=True)
class Code:
    """One QR code or barcode read in a frame."""

    kind: str                                  # "qr" or "barcode"
    text: str                                  # what the code says, cleaned
    box: Box                                   # (x1, y1, x2, y2) in pixels of the frame it was read from
    corners: tuple[tuple[int, int], ...]       # its four corners in the same pixels
    format: str = ""                           # for a barcode, e.g. "EAN_13"; "" for a QR code
    altered: bool = False                      # cleaning removed or cut something: not exactly what the code holds


def checked_settings(settings: dict) -> tuple[int, int, float]:
    """(every_n_frames, max_width, hold_seconds) from the codes section, checked.

    Raises ConfigError for a missing setting or a value out of range.
    """
    every_n_frames = require(settings, "every_n_frames", int)
    if every_n_frames < 1:
        raise ConfigError("codes.every_n_frames must be 1 or more")
    max_width = require(settings, "max_width", int)
    if max_width < MIN_SEARCH_WIDTH:
        raise ConfigError(f"codes.max_width must be {MIN_SEARCH_WIDTH} pixels or more")
    hold_seconds = float(require(settings, "hold_seconds", (int, float)))
    if hold_seconds < 0:
        raise ConfigError("codes.hold_seconds must be 0 or more")
    return every_n_frames, max_width, hold_seconds


class CodeReader:
    """Finds and decodes the QR codes and barcodes in a frame.

    `qr` and `barcodes` are OpenCV's two detectors; the tests hand in fakes.
    The QR detector is the ArUco-based one: on test pictures it read every
    code the plain cv2.QRCodeDetector read and many it missed, and faster.
    Only the thread that calls read() should use one reader.
    """

    def __init__(self, settings: dict, qr: Any = None, barcodes: Any = None) -> None:
        _every_n_frames, self.max_width, _hold_seconds = checked_settings(settings)
        self._qr = qr if qr is not None else cv2.QRCodeDetectorAruco()
        self._barcodes = barcodes if barcodes is not None else cv2.barcode.BarcodeDetector()
        self._failed: set[str] = set()         # the kinds whose failure was logged already

    def read(self, frame: np.ndarray) -> list[Code]:
        """The decoded codes in the frame, with boxes and corners in the frame's own pixels.

        A code that was found but not decoded is left out, and the same code
        found twice counts once. Never raises for an OpenCV error: that kind
        then gives no codes on this frame. An empty frame gives none.
        """
        if frame.size == 0:
            return []
        picture, scale = shrunk_for_search(frame, self.max_width)
        height, width = frame.shape[:2]
        codes: list[Code] = []
        for kind in ("qr", "barcode"):
            for code in self._read_kind(kind, picture, scale, (width, height)):
                if not any(same_code(code, other) for other in codes):
                    codes.append(code)
        return codes

    def _read_kind(self, kind: str, picture: np.ndarray, scale: float, size: tuple[int, int]) -> list[Code]:
        """The codes of one kind; [] (and one warning, the first time) when OpenCV fails."""
        try:
            if kind == "qr":
                finds = qr_finds(self._qr.detectAndDecodeMulti(picture))
            else:
                finds = barcode_finds(self._barcodes.detectAndDecodeWithType(picture))
            codes = [make_code(kind, text, points, form, scale, size) for text, points, form in finds]
        except Exception as error:      # cv2.error, or an answer of an unexpected shape
            if kind not in self._failed:
                self._failed.add(kind)
                log.warning("Reading %s codes failed (%s); frames where it fails show none",
                            "QR" if kind == "qr" else "bar", type(error).__name__)
            return []
        return [code for code in codes if code is not None]


def qr_finds(result: Sequence[Any]) -> list[tuple[Any, Any, str]]:
    """(text, corner points, format) for each QR code in detectAndDecodeMulti's answer.

    The answer is (ok, texts, points, straight codes); points is None when nothing was found.
    """
    _ok, texts, points = result[0], result[1], result[2]
    if points is None:
        return []
    return [(text, corners, "") for text, corners in zip(texts, points)]


def barcode_finds(result: Sequence[Any]) -> list[tuple[Any, Any, str]]:
    """(text, corner points, format) for each barcode in detectAndDecodeWithType's answer.

    The answer is (ok, texts, formats, points); points is None when nothing was found.
    """
    _ok, texts, formats, points = result[0], result[1], result[2], result[3]
    if points is None:
        return []
    return [(text, corners, form) for text, form, corners in zip(texts, formats, points)]


def make_code(kind: str, text: Any, points: Any, form: Any, scale: float, size: tuple[int, int]) -> Code | None:
    """A Code with its corners scaled back to the frame and kept inside it; None when it has no text."""
    raw, text = text, clean_text(text)
    if not text:
        return None                     # found, but not decoded
    corners = tuple(inside(x / scale, y / scale, size)
                    for x, y in np.asarray(points, dtype=float).reshape(-1, 2))
    if not corners:
        return None
    xs = [x for x, _y in corners]
    ys = [y for _x, y in corners]
    box = (min(xs), min(ys), max(xs), max(ys))
    return Code(kind, text, box, corners, clean_text(form) if kind == "barcode" else "", was_altered(raw, text))


def was_altered(raw: str, cleaned: str) -> bool:
    """True when cleaning removed or cut anything besides the spaces around the text."""
    return cleaned != raw.strip()


def inside(x: float, y: float, size: tuple[int, int]) -> tuple[int, int]:
    """A point rounded to whole pixels and kept inside a frame of this (width, height)."""
    width, height = size
    return min(max(round(x), 0), width - 1), min(max(round(y), 0), height - 1)


def shrunk_for_search(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """The frame shrunk to max_width wide (its shape kept) when it is wider, and the scale used."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / width
    small = cv2.resize(frame, (max_width, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


def same_code(a: Code, b: Code) -> bool:
    """True when two finds are one code: the same kind and text, and boxes that overlap."""
    if (a.kind, a.text) != (b.kind, b.text):
        return False
    return a.box[0] <= b.box[2] and b.box[0] <= a.box[2] and a.box[1] <= b.box[3] and b.box[1] <= a.box[3]


def clean_text(text: Any) -> str:
    """A code's text made safe to keep and show; "" for anything that is not a string.

    Line breaks and tabs become one space, other control characters are
    removed, and the result is trimmed and cut to MAX_TEXT_CHARS.
    """
    if not isinstance(text, str):
        return ""
    text = LINE_BREAKS.sub(" ", text)
    text = "".join(letter for letter in text if unicodedata.category(letter) not in DROPPED_KINDS)
    return text.strip()[:MAX_TEXT_CHARS].rstrip()


def web_link(text: str) -> str | None:
    """The text itself when the whole of it is one plain web address, else None.

    A web address here: the scheme http or https, a host, no user name or
    password, and no spaces. javascript:, data:, file:, tel:, WiFi and
    contact codes are not links.
    """
    if not isinstance(text, str) or not text or "\\" in text:
        return None
    if any(letter.isspace() or unicodedata.category(letter)[0] == "C" for letter in text):
        return None
    try:
        parts = urlsplit(text)
        _port = parts.port              # raises ValueError for a port that is not a number
    except ValueError:
        return None
    if parts.scheme not in WEB_SCHEMES or "@" in parts.netloc:
        return None
    host = parts.hostname
    if not host or not HOST.fullmatch(host):
        return None
    return text
