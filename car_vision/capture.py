"""Take enrolment photos of one person through the camera.

The program watches the camera and saves a photo each time it sees a face,
a moment apart, until it has enough. The person only has to stand in front of
the camera and look at it, turning their head a little between photos.

    .venv\\Scripts\\python.exe -m car_vision.capture sara
    .venv\\Scripts\\python.exe -m car_vision.capture sara --source webcam --count 5

Photos go to data/people/<id>/ at the camera's full resolution. Take them only
with the person's permission. Afterwards add the person to data/people.json
and run the enroll command.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

from car_vision.config import load_config, require, resolve_path, section

SEARCH_WIDTH = 640          # faces are searched on a copy shrunk to this width
MIN_FACE_PIXELS = 36        # smallest face, on that copy, worth a photo
SHARPNESS_FLOOR = 25.0      # below this the picture is too blurred to use


def find_largest_face(frame: np.ndarray, finder: cv2.CascadeClassifier) -> tuple[int, int, int, int] | None:
    """Return the biggest face as (x, y, width, height) in full-frame pixels, or None."""
    scale = SEARCH_WIDTH / frame.shape[1]
    small = cv2.resize(frame, (SEARCH_WIDTH, int(round(frame.shape[0] * scale))))
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    faces = finder.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7,
                                    minSize=(MIN_FACE_PIXELS, MIN_FACE_PIXELS))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return int(x / scale), int(y / scale), int(w / scale), int(h / scale)


def sharpness(frame: np.ndarray, face: tuple[int, int, int, int]) -> float:
    """How crisp the face area is. A moving head gives a low number."""
    x, y, w, h = face
    patch = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def next_photo_number(folder: Path) -> int:
    numbers = [int(match.group(1)) for path in folder.glob("photo_*.jpg")
               if (match := re.fullmatch(r"photo_(\d+)", path.stem))]
    return max(numbers, default=0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Take enrolment photos through the camera.")
    parser.add_argument("person_id", help="short id without spaces, e.g. sara; also the folder name")
    parser.add_argument("--source", choices=("phone", "webcam"), default="phone")
    parser.add_argument("--count", type=int, default=5, help="how many photos to take")
    parser.add_argument("--gap", type=float, default=1.5, help="seconds between photos")
    parser.add_argument("--timeout", type=float, default=90.0, help="give up after this many seconds")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[a-z0-9_]+", args.person_id):
        print("The id may only use small letters, digits and underscores, e.g. sara or omar_k.")
        return 1

    config = load_config()
    stream = section(config, "stream")
    folder = resolve_path(require(section(config, "recognizer"), "photos_dir", str)) / args.person_id
    folder.mkdir(parents=True, exist_ok=True)

    if args.source == "phone":
        capture = cv2.VideoCapture(require(stream, "phone_url", str), cv2.CAP_FFMPEG)
    else:
        capture = cv2.VideoCapture(require(stream, "webcam_index", int), cv2.CAP_DSHOW)
    if not capture.isOpened():
        print(f"Could not open the {args.source} camera.")
        return 1

    finder = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    number = next_photo_number(folder)
    taken = 0
    started = time.monotonic()
    next_photo_at = started
    last_note = started
    print(f"Looking for a face on the {args.source} camera. Stand 1 to 2 metres away and look at it.")
    try:
        while taken < args.count:
            now = time.monotonic()
            if now - started > args.timeout:
                break
            opened, frame = capture.read()
            if not opened or frame is None:
                continue
            if now < next_photo_at:
                continue                      # keep reading so the picture stays fresh
            face = find_largest_face(frame, finder)
            if face is None or sharpness(frame, face) < SHARPNESS_FLOOR:
                if now - last_note > 5.0:
                    print("  no clear face in view yet...", flush=True)
                    last_note = now
                continue
            path = folder / f"photo_{number}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            taken += 1
            number += 1
            next_photo_at = now + args.gap
            last_note = now
            print(f"  photo {taken} of {args.count} saved (face {face[2]} pixels wide)", flush=True)
    finally:
        capture.release()

    print(f"{taken} photo(s) saved in {folder}")
    if taken < args.count:
        print("Not enough photos. Check the camera angle and the light, then run this again.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
