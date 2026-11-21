"""Draw the part of a photo that a visitor brushed over, as an outline.

On the details panel a visitor can brush over part of a clicked thing's photo
to ask about exactly that part. The browser sends only the brush strokes, as
points from 0 to 1 across and down the photo; the server draws them here on
its own copy of the photo, and that copy goes to the assistant with the
question.

Only the outline of the brushed region is drawn (a magenta line with a thin
white line outside it), so the part that was marked stays as it was and the
model can still see it. A loop drawn around something marks everything inside
the loop. A marking over nearly the whole photo leaves no outline to see;
marked_share tells the server so, and it then asks about the whole photo.
Uses OpenCV and numpy only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

Stroke = Sequence[tuple[float, float]]     # points as fractions of the photo, (0, 0) is the top-left corner

BRUSH_SHARE = 0.06            # the brush is this share of the photo's shorter side wide, as on the page
OUTLINE_SHARE = 1 / 120       # the magenta line is this share of the shorter side wide ...
MIN_OUTLINE_PX = 2            # ... and never thinner than this
MAGENTA = (255, 0, 255)       # OpenCV colours are blue, green, red
WHITE = (255, 255, 255)
NEARLY_ALL = 0.9              # a marking over more than this share of the photo marks no part at all
QUALITY_STEP = 10             # when the JPEG is too large, it is saved again this much lower ...
MIN_QUALITY = 25              # ... but never below this


def mark_photo(photo: bytes, strokes: Sequence[Stroke], quality: int = 85,
               max_bytes: int | None = None) -> bytes:
    """The JPEG photo, same size, with the outline of the brushed region drawn on it.

    With `max_bytes`, the JPEG is saved again at lower quality until it fits.
    ValueError when the bytes are not a picture, when there are no strokes,
    when a point is not a pair of numbers, or when it does not fit even at
    MIN_QUALITY.
    """
    image = decode(photo)
    draw_outline(image, marked_region(image, strokes))
    while True:
        encoded = encode(image, quality)
        if max_bytes is None or len(encoded) <= max_bytes:
            return encoded
        if quality <= MIN_QUALITY:
            raise ValueError(f"The marked photo is too large to send ({len(encoded)} bytes, "
                             f"the limit is {max_bytes} bytes).")
        quality = max(MIN_QUALITY, quality - QUALITY_STEP)


def marked_share(photo: bytes, strokes: Sequence[Stroke]) -> float:
    """How much of the photo the marking covers, from 0 to 1 (the inside of loops counts).

    Over NEARLY_ALL there is no outline left to see, so the question is about the whole photo.
    """
    image = decode(photo)
    region = marked_region(image, strokes)
    return float(np.count_nonzero(region)) / region.size


def marked_region(image: np.ndarray, strokes: Sequence[Stroke]) -> np.ndarray:
    """The brushed region on this picture, loops filled in: 255 inside, 0 outside."""
    if not strokes or not any(len(stroke) for stroke in strokes):
        raise ValueError("There is nothing marked to draw.")
    height, width = image.shape[:2]
    return fill_loops(brush_mask(strokes, width, height))


def encode(image: np.ndarray, quality: int) -> bytes:
    """The picture as JPEG bytes with full colour detail (4:4:4).

    The usual JPEG halves colour detail, which turns a thin magenta line into a pale smear.
    """
    options = [cv2.IMWRITE_JPEG_QUALITY, int(quality),
               cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    ok, encoded = cv2.imencode(".jpg", image, options)
    if not ok:
        raise ValueError("The marked photo could not be saved as a JPEG.")
    return encoded.tobytes()


def decode(photo: bytes) -> np.ndarray:
    """The photo as a picture OpenCV can draw on, or ValueError."""
    if not isinstance(photo, (bytes, bytearray)) or not photo:
        raise ValueError("The photo is empty.")
    try:
        image = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None:
        raise ValueError("The photo is not a picture.")
    return image


def brush_width(width: int, height: int) -> int:
    """How wide the brush is in pixels on a photo of this size."""
    return max(1, round(BRUSH_SHARE * min(width, height)))


def outline_width(width: int, height: int) -> int:
    """How wide the magenta line is: it grows with the photo, at least MIN_OUTLINE_PX."""
    return max(MIN_OUTLINE_PX, round(OUTLINE_SHARE * min(width, height)))


def to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    """A point given as fractions of the photo, in pixels. Points outside are pulled to the edge."""
    if len(point) != 2:
        raise ValueError("A point must be two numbers.")
    x, y = float(point[0]), float(point[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("A point must be two finite numbers.")
    x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    return round(x * (width - 1)), round(y * (height - 1))


def brush_mask(strokes: Sequence[Stroke], width: int, height: int) -> np.ndarray:
    """Where the brush went: 255 on the strokes, 0 elsewhere.

    Each stroke is a line with round ends; a stroke of one point is a filled circle.
    """
    mask = np.zeros((height, width), np.uint8)
    size = brush_width(width, height)
    radius = max(1, size // 2)
    for stroke in strokes:
        pixels = [to_pixel(point, width, height) for point in stroke]
        for pixel in pixels:
            cv2.circle(mask, pixel, radius, 255, cv2.FILLED)          # the round ends and joins
        for start, end in zip(pixels, pixels[1:]):
            cv2.line(mask, start, end, 255, size)
    return mask


def fill_loops(mask: np.ndarray) -> np.ndarray:
    """The brushed region with the inside of every closed loop filled in."""
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def grown(mask: np.ndarray, pixels: int) -> np.ndarray:
    """The region made larger by this many pixels on every side."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask, kernel)


def draw_outline(image: np.ndarray, region: np.ndarray) -> None:
    """Draw a magenta line just outside the region, and a thin white line outside that.

    Both lines lie outside the region, so the marked content is not touched.
    """
    height, width = image.shape[:2]
    line = outline_width(width, height)
    magenta_edge = grown(region, line)
    white_edge = grown(magenta_edge, max(1, line // 2))
    image[(white_edge > 0) & (magenta_edge == 0)] = WHITE
    image[(magenta_edge > 0) & (region == 0)] = MAGENTA
