"""Make a short test clip that shows a QR code and a barcode, to check code reading without a camera.

    .venv\\Scripts\\python.exe tools\\make_codes_clip.py data\\samples\\codes_clip.mp4

Then play it on the demo page (the `demo-codes` entry of .claude/launch.json does this):

    .venv\\Scripts\\python.exe -m car_vision.app --file data/samples/codes_clip.mp4
"""

import sys

import cv2
import numpy as np

WIDTH, HEIGHT, FPS, SECONDS = 640, 360, 15, 10
QR_TEXT = "https://example.com/robot-car"
EAN = "5901234123457"

L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"]
R = ["".join("1" if bit == "0" else "0" for bit in code) for code in L]
G = [code[::-1] for code in R]
PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


def ean13_bits(digits: str) -> str:
    left = "".join((L if kind == "L" else G)[int(d)] for d, kind in zip(digits[1:7], PARITY[int(digits[0])]))
    right = "".join(R[int(d)] for d in digits[7:])
    return "101" + left + "01010" + right + "101"


def barcode_picture(digits: str, bar: int = 3, height: int = 90, quiet: int = 12) -> np.ndarray:
    bits = ean13_bits(digits)
    picture = np.full((height + 2 * quiet, (len(bits) + 2 * quiet) * bar), 255, dtype=np.uint8)
    for i, bit in enumerate(bits):
        if bit == "1":
            x = (quiet + i) * bar
            picture[quiet:quiet + height, x:x + bar] = 0
    return picture


def qr_picture(text: str, size: int = 150) -> np.ndarray:
    code = cv2.QRCodeEncoder.create().encode(text)
    code = cv2.copyMakeBorder(code, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(code, (size, size), interpolation=cv2.INTER_NEAREST)


def paste(frame: np.ndarray, picture: np.ndarray, x: int, y: int) -> None:
    h, w = picture.shape[:2]
    frame[y:y + h, x:x + w] = cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)


def main(path: str) -> None:
    rng = np.random.default_rng(7)
    background = cv2.GaussianBlur(rng.integers(90, 150, (HEIGHT, WIDTH, 3), dtype=np.uint8), (0, 0), 6)
    # Sizes a camera can read at 640 x 360: the QR code about 6 pixels a square, the bars 2 pixels.
    qr, bars = qr_picture(QR_TEXT, size=220), barcode_picture(EAN, bar=2, height=70)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    total = FPS * SECONDS
    for i in range(total):
        frame = background.copy()
        slide = int(30 * np.sin(2 * np.pi * i / total))
        paste(frame, qr, 50 + slide, 70)
        if i > total // 3:                       # the barcode comes into view after a while
            paste(frame, bars, 390, 130 - slide // 2)
        # A real camera never gives perfectly sharp edges, and OpenCV's barcode reader
        # misses large bars that are perfectly sharp, so the picture is softened a little.
        writer.write(cv2.GaussianBlur(frame, (0, 0), 0.8))
    writer.release()
    print("wrote", path, total, "frames")


if __name__ == "__main__":
    main(sys.argv[1])
