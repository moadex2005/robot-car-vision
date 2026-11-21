"""Draws detection boxes, captions and a status line onto camera frames.

Everything here draws in place on the frame it is given and returns nothing.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from car_vision.faces import Person
from car_vision.state import Detection, RecognizedFace

FONT = cv2.FONT_HERSHEY_SIMPLEX
CAPTION_SCALE = 0.5
CAPTION_THICKNESS = 1
CAPTION_PADDING = 3
BOX_THICKNESS = 2


def color_for(label: str) -> tuple[int, int, int]:
    """A colour (blue, green, red) made from the label's letters.

    The same label always gives the same colour, in every run. Python's own
    hash() changes between runs, so a small hash of our own is used instead.
    """
    value = 7
    for letter in label:
        value = (value * 31 + ord(letter)) % 1_000_003
    blue = 60 + value % 196
    green = 60 + (value // 196) % 196
    red = 60 + (value // (196 * 196)) % 196
    return int(blue), int(green), int(red)


def caption_for(detection: Detection) -> str:
    """Text such as 'person 0.87 #3'; no '#' part when there is no track number."""
    text = f"{detection.label} {detection.confidence:.2f}"
    if detection.track_id is not None:
        text += f" #{detection.track_id}"
    return text


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> None:
    """Draw a rectangle and a caption for each detection."""
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = color_for(detection.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        draw_caption(frame, caption_for(detection), x1, y1, color)


def draw_caption(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Write the caption on a filled band just above (x, y), kept inside the frame.

    When the box touches the top edge there is no room above it, so the band
    goes just inside the box instead.
    """
    height, width = frame.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, CAPTION_SCALE, CAPTION_THICKNESS)
    band_width = text_width + 2 * CAPTION_PADDING
    band_height = text_height + baseline + 2 * CAPTION_PADDING

    top = y - band_height
    if top < 0:
        top = y
    top = max(0, min(top, height - band_height))
    left = max(0, min(x, width - band_width))

    cv2.rectangle(frame, (left, top), (left + band_width, top + band_height), color, cv2.FILLED)
    text_origin = (left + CAPTION_PADDING, top + CAPTION_PADDING + text_height)
    cv2.putText(frame, text, text_origin, FONT, CAPTION_SCALE, text_color_on(color),
                CAPTION_THICKNESS, cv2.LINE_AA)


def text_color_on(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black text on light colours, white text on dark ones."""
    blue, green, red = color
    brightness = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if brightness > 140 else (255, 255, 255)


def draw_status(frame: np.ndarray, text: str) -> None:
    """Write one line of status in the top-left corner, readable on any background.

    Looks the same as car_vision.stream.draw_status.
    """
    cv2.putText(frame, text, (10, 24), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 24), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


# ---- faces and info cards ----------------------------------------------------

KNOWN_FACE_COLOR = (0, 200, 0)          # green (blue, green, red)
UNKNOWN_FACE_COLOR = (160, 160, 160)    # grey
CARD_WIDTH = 220
CARD_GAP = 8                            # space between the face box and the card
CARD_PADDING = 8
CARD_LINE_GAP = 6
CARD_MAX_LINES = 3
CARD_MAX_CHARS = 60                     # no card row is longer than this, even before measuring
CARD_BACKGROUND = (40, 40, 40)
CARD_NAME_SCALE = 0.6
CARD_NAME_THICKNESS = 2
CARD_TEXT_SCALE = 0.45
CARD_TEXT_THICKNESS = 1
ELLIPSIS = "..."


def draw_faces(frame: np.ndarray, faces: Iterable[RecognizedFace], show_scores: bool = False) -> None:
    """A green box and the name for a known face, a grey box and 'Unknown' otherwise.

    With show_scores the similarity is written after the name, e.g. 'Sara Ahmed 0.63',
    so the match threshold can be tuned by watching the numbers.
    """
    for face in faces:
        x1, y1, x2, y2 = face.box
        color = KNOWN_FACE_COLOR if face.is_known else UNKNOWN_FACE_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        text = face.name if drawable(face.name) else (face.person_id or "?")
        if show_scores:
            text += f" {face.similarity:.2f}"
        draw_caption(frame, text, x1, y1, color)


def draw_info_card(frame: np.ndarray, face: RecognizedFace, person: Person) -> None:
    """A filled panel beside the face with the name, the role and up to 3 lines.

    The card goes to the right of the face, or to the left when there is no
    room, and is always kept fully inside the frame.
    """
    height, width = frame.shape[:2]
    card_width = min(CARD_WIDTH, width)
    text_width = card_width - 2 * CARD_PADDING
    rows = _card_rows(person, text_width)
    while len(rows) > 1 and _card_height(rows) > height:
        rows.pop()                                     # a tiny frame: keep what fits
    card_height = min(_card_height(rows), height)

    left, top = _card_position(face.box, card_width, card_height, width, height)
    cv2.rectangle(frame, (left, top), (left + card_width - 1, top + card_height - 1),
                  CARD_BACKGROUND, cv2.FILLED)
    cv2.rectangle(frame, (left, top), (left + card_width - 1, top + card_height - 1),
                  KNOWN_FACE_COLOR, 1)

    y = top + CARD_PADDING
    for text, scale, thickness, color in rows:
        text_height, baseline = _row_size(text, scale, thickness)
        y += text_height
        cv2.putText(frame, text, (left + CARD_PADDING, y), FONT, scale, color, thickness, cv2.LINE_AA)
        y += baseline + CARD_LINE_GAP


def drawable(text: str) -> bool:
    """True when cv2.putText can draw the text; it shows other letters (Arabic, accents) as '?'."""
    return text.isascii()


def _card_rows(person: Person, text_width: int) -> list[tuple[str, float, int, tuple[int, int, int]]]:
    """The text rows of a card, each cut to fit: (text, scale, thickness, colour).

    A name that cannot be drawn is replaced by the person id (the folder name);
    other rows that cannot be drawn are left off. The web page shows them all.
    """
    name = person.name if drawable(person.name) else person.person_id
    rows = [(_fit_text(name, text_width, CARD_NAME_SCALE, CARD_NAME_THICKNESS),
             CARD_NAME_SCALE, CARD_NAME_THICKNESS, KNOWN_FACE_COLOR)]
    texts = ([person.role] if person.role else []) + list(person.lines[:CARD_MAX_LINES])
    for text in filter(drawable, texts):
        rows.append((_fit_text(text, text_width, CARD_TEXT_SCALE, CARD_TEXT_THICKNESS),
                     CARD_TEXT_SCALE, CARD_TEXT_THICKNESS, (255, 255, 255)))
    return rows


def _card_height(rows: list[tuple[str, float, int, tuple[int, int, int]]]) -> int:
    total = 2 * CARD_PADDING - CARD_LINE_GAP
    for text, scale, thickness, _color in rows:
        text_height, baseline = _row_size(text, scale, thickness)
        total += text_height + baseline + CARD_LINE_GAP
    return total


def _row_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    """Height above and below the baseline of one row; an empty row is as tall as a space."""
    (_, text_height), baseline = cv2.getTextSize(text or " ", FONT, scale, thickness)
    return text_height, baseline


def _card_position(face_box: tuple[int, int, int, int], card_width: int, card_height: int,
                   frame_width: int, frame_height: int) -> tuple[int, int]:
    """Top-left corner of the card: right of the face, else left of it, always inside the frame."""
    x1, y1, x2, _y2 = face_box
    left = x2 + CARD_GAP
    if left + card_width > frame_width:
        left = x1 - CARD_GAP - card_width               # no room on the right
    left = max(0, min(left, frame_width - card_width))
    top = max(0, min(y1, frame_height - card_height))
    return left, top


def _fit_text(text: str, max_width: int, scale: float, thickness: int) -> str:
    """The text itself when it fits in max_width pixels, else as much as fits followed by '...'.

    Text longer than CARD_MAX_CHARS is cut first, because measuring is slow for very long text.
    """
    too_long = len(text) > CARD_MAX_CHARS
    text = text[:CARD_MAX_CHARS]
    if not too_long and cv2.getTextSize(text, FONT, scale, thickness)[0][0] <= max_width:
        return text
    for end in range(len(text), 0, -1):
        cut = text[:end].rstrip() + ELLIPSIS
        if cv2.getTextSize(cut, FONT, scale, thickness)[0][0] <= max_width:
            return cut
    return ELLIPSIS
