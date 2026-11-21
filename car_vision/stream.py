"""Live frames from the phone, the laptop webcam or a recorded clip.

A background thread reads the camera as fast as frames arrive and keeps only
the newest one. Whoever asks for a frame always gets the latest picture, so a
slow model never makes the video fall behind the real world.

The phone sends an MJPEG stream: one JPEG picture after another over HTTP.
MjpegCapture reads it with the standard library, keeps only the newest
picture, and decodes it at a reduced size (a 1920 x 1080 picture that will be
shown at 640 x 360 is decoded at 960 x 540, which is much cheaper). When the
phone answers with something other than MJPEG, FFmpeg reads it instead.

Run it on its own to check the stream:

    .venv\\Scripts\\python.exe -m car_vision.stream
    .venv\\Scripts\\python.exe -m car_vision.stream --source phone
    .venv\\Scripts\\python.exe -m car_vision.stream --source file

Keys in the window: q quits, c shows a millisecond clock for the delay test.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import logging
import os
import socket
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Protocol
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from car_vision.config import ConfigError, load_config, require, resolve_path, section

log = logging.getLogger(__name__)

SOURCES = ("phone", "webcam", "file")
OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 5000
RECORD_FPS = 15.0

MAX_PICTURE_BYTES = 4 * 1024 * 1024   # a 1920 x 1080 JPEG is well under 1 MB
MAX_HEADER_BYTES = 8 * 1024           # the headers of one part of the stream
CHUNK_BYTES = 64 * 1024               # read at most this much from the network at once
RELEASE_WAIT_SECONDS = 1.0            # release() waits this long for the network thread
JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
FFMPEG_OPTIONS_VARIABLE = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
FFMPEG_LOW_DELAY = "fflags;nobuffer|flags;low_delay"
REDUCED_DECODE_FLAGS = {
    1: cv2.IMREAD_COLOR,
    2: cv2.IMREAD_REDUCED_COLOR_2,
    4: cv2.IMREAD_REDUCED_COLOR_4,
    8: cv2.IMREAD_REDUCED_COLOR_8,
}


class Capture(Protocol):
    """The small part of cv2.VideoCapture this module uses. Tests pass a fake."""

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...

    def isOpened(self) -> bool: ...


class FpsCounter:
    """Frames per second over the last few seconds."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        self._window = window_seconds
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self, now: float | None = None) -> None:
        stamp = time.monotonic() if now is None else now
        with self._lock:
            self._stamps.append(stamp)
            self._trim(stamp)

    def value(self, now: float | None = None) -> float:
        stamp = time.monotonic() if now is None else now
        with self._lock:
            self._trim(stamp)
            if len(self._stamps) < 2:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def _trim(self, now: float) -> None:
        while self._stamps and now - self._stamps[0] > self._window:
            self._stamps.popleft()


def fitted_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """The largest size with the same shape that fits inside max_width x max_height.

    Never enlarges, and keeps both sides even because video writers need that.
    """
    scale = min(max_width / width, max_height / height, 1.0)
    fitted_width = max(2, int(round(width * scale / 2)) * 2)
    fitted_height = max(2, int(round(height * scale / 2)) * 2)
    return fitted_width, fitted_height


# ---- JPEG pictures -------------------------------------------------------

# Markers that stand alone, without a length: TEM, RST0 to RST7 and SOI.
STANDALONE_MARKERS = frozenset([0x01, *range(0xD0, 0xD9)])
# Start-of-frame markers (baseline, progressive and the rarer kinds); they hold the size.
FRAME_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
END_OF_PICTURE = 0xD9
START_OF_SCAN = 0xDA


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """The (width, height) written in a JPEG's frame header, read without decoding.

    Walks the header segments up to the start-of-frame marker, which baseline
    and progressive pictures both have. None when the size cannot be read.
    """
    if not data.startswith(JPEG_START):
        return None
    position = 2
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            return None                       # not a marker where one should be
        marker = data[position + 1]
        if marker == 0xFF:                    # padding before a marker
            position += 1
        elif marker in STANDALONE_MARKERS:
            position += 2
        elif marker in FRAME_MARKERS:
            return frame_header_size(data, position)
        elif marker in (END_OF_PICTURE, START_OF_SCAN):
            return None                       # the picture data began without a frame header
        else:
            length = int.from_bytes(data[position + 2:position + 4], "big")
            if length < 2:
                return None
            position += 2 + length
    return None


def frame_header_size(data: bytes, position: int) -> tuple[int, int] | None:
    """Width and height from the start-of-frame segment that begins at position.

    The segment is: marker (2 bytes), length (2), precision (1), height (2), width (2).
    """
    header = data[position + 5:position + 9]
    if len(header) < 4:
        return None
    height = int.from_bytes(header[0:2], "big")
    width = int.from_bytes(header[2:4], "big")
    return (width, height) if width > 0 and height > 0 else None


def reduction_for(width: int, height: int, max_width: int, max_height: int) -> int:
    """How many times smaller (1, 2, 4 or 8) a picture can be decoded without losing detail.

    Picks the largest reduction whose result is still at least the size that
    fitted_size will ask for, so the resize afterwards only ever shrinks.
    """
    if width <= 0 or height <= 0:
        return 1
    wanted_width, wanted_height = fitted_size(width, height, max_width, max_height)
    for factor in (8, 4, 2):
        reduced_width = -(-width // factor)    # the decoder rounds up
        reduced_height = -(-height // factor)
        if reduced_width >= wanted_width and reduced_height >= wanted_height:
            return factor
    return 1


def is_whole_jpeg(data: bytes) -> bool:
    """True when the bytes start with the JPEG start marker and end with the end marker.

    Line breaks and zero bytes after the end marker are allowed: some senders pad.
    """
    return data.startswith(JPEG_START) and data.rstrip(b"\r\n\x00").endswith(JPEG_END)


def decode_jpeg(data: bytes, max_width: int, max_height: int) -> np.ndarray | None:
    """Decode a JPEG at the smallest size that still fills max_width x max_height.

    The size comes from the picture's header; when it cannot be read, the
    picture is decoded in full. None when the picture is broken.
    """
    if not is_whole_jpeg(data):
        return None
    size = jpeg_size(data)
    factor = reduction_for(size[0], size[1], max_width, max_height) if size else 1
    try:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), REDUCED_DECODE_FLAGS[factor])
    except cv2.error:
        return None


# ---- cutting an MJPEG stream into pictures -------------------------------


def multipart_boundary(content_type: str) -> bytes | None:
    """The boundary of a multipart content type, without quotes or leading dashes.

    'multipart/x-mixed-replace; boundary="--frame"' gives b"frame". Searching
    for the text without its dashes works whether a server counts the two
    dashes of each boundary line as part of the boundary or not.
    None when the type is not multipart or names no boundary.
    """
    kind, _, parameters = content_type.partition(";")
    if not kind.strip().lower().startswith("multipart/"):
        return None
    for parameter in parameters.split(";"):
        name, _, value = parameter.partition("=")
        if name.strip().lower() == "boundary":
            boundary = value.strip().strip('"').lstrip("-")
            return boundary.encode("latin-1", errors="replace") if boundary else None
    return None


def content_length(value: bytes) -> int | None:
    """The number in a Content-Length header, or None when it is not a positive number."""
    try:
        length = int(value.strip())
    except ValueError:
        return None
    return length if length > 0 else None


class MjpegParser:
    """Cuts the bytes of a multipart MJPEG stream into JPEG pictures.

    Feed it bytes as they arrive, in pieces of any size; it hands back every
    picture that is complete. A part's Content-Length is used when it has one;
    otherwise the picture runs from the JPEG start marker to the end marker.
    (A picture with an embedded thumbnail and no Content-Length would be cut at
    the thumbnail's end marker; MJPEG cameras do not send those.)

    Memory stays small: a picture bigger than max_picture_bytes is dropped and
    the parser waits for the next boundary instead of growing its buffer.
    """

    def __init__(self, boundary: bytes, max_picture_bytes: int = MAX_PICTURE_BYTES) -> None:
        self.boundary = boundary
        self.max_picture_bytes = max_picture_bytes
        self.skipped_parts = 0        # parts thrown away: too big, cut off or malformed
        self.wrong_lengths = 0        # parts whose Content-Length did not fit their picture
        self._buffer = bytearray()
        self._state = "boundary"      # "boundary", then "headers", then "picture"
        self._length: int | None = None
        self._header_bytes = 0
        self._searched_up_to = 0      # where the search for the end marker goes on

    @property
    def buffered_bytes(self) -> int:
        """How many bytes are held while waiting for the rest of a part."""
        return len(self._buffer)

    def feed(self, data: bytes) -> list[bytes]:
        """Add bytes from the stream; return the pictures they completed, oldest first."""
        self._buffer += data
        pictures: list[bytes] = []
        while self._step(pictures):
            pass
        return pictures

    def _step(self, pictures: list[bytes]) -> bool:
        """Do one step of work. False when more bytes are needed."""
        if self._state == "boundary":
            return self._skip_to_boundary()
        if self._state == "headers":
            return self._read_header_line()
        if self._length is not None:
            return self._take_counted_picture(pictures)
        return self._take_marked_picture(pictures)

    def _skip_to_boundary(self) -> bool:
        """Drop everything up to and including the next boundary line."""
        found = self._buffer.find(self.boundary)
        if found < 0:
            self._keep_last(len(self.boundary) - 1)   # a boundary may be cut in two
            return False
        line_end = self._buffer.find(b"\n", found)
        if line_end < 0:
            if len(self._buffer) - found > MAX_HEADER_BYTES:
                del self._buffer[:found + len(self.boundary)]
                return True
            return False
        del self._buffer[:line_end + 1]
        self._start_part()
        return True

    def _start_part(self) -> None:
        self._state = "headers"
        self._length = None
        self._header_bytes = 0
        self._searched_up_to = 0

    def _read_header_line(self) -> bool:
        """Read one header line of a part; an empty line ends the headers."""
        line_end = self._buffer.find(b"\n")
        if line_end < 0:
            if self._header_bytes + len(self._buffer) > MAX_HEADER_BYTES:
                self._skip_part("its headers are too long")
                return True
            return False
        line = bytes(self._buffer[:line_end]).strip()
        del self._buffer[:line_end + 1]
        self._header_bytes += line_end + 1
        if self._header_bytes > MAX_HEADER_BYTES:
            self._skip_part("its headers are too long")
            return True
        if not line:
            self._state = "picture"
            return True
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            self._length = content_length(value)
        return True

    def _take_counted_picture(self, pictures: list[bytes]) -> bool:
        """The part said how long its picture is: wait for that many bytes."""
        length = self._length
        if length > self.max_picture_bytes:
            self._skip_part(f"its picture of {length} bytes is too big")
            return True
        if len(self._buffer) < length:
            return False
        picture = bytes(self._buffer[:length])
        if not is_whole_jpeg(picture):         # the length was wrong: trust the markers
            self._note_wrong_length()
            self._length = None
            return True
        pictures.append(picture)
        del self._buffer[:length]
        self._state = "boundary"
        return True

    def _note_wrong_length(self) -> None:
        """Count a part whose Content-Length did not fit its picture; only the first is logged."""
        self.wrong_lengths += 1
        if self.wrong_lengths == 1:
            log.warning("A Content-Length in the MJPEG stream does not match its picture; "
                        "finding pictures by their JPEG markers when that happens")

    def _take_marked_picture(self, pictures: list[bytes]) -> bool:
        """No length given: the picture runs from the JPEG start to the JPEG end marker."""
        start = self._buffer.find(JPEG_START)
        if start < 0:
            self._keep_last(1)                 # the marker may be cut in two
            return False
        del self._buffer[:start]
        end = self._buffer.find(JPEG_END, max(2, self._searched_up_to))
        if end < 0:
            if len(self._buffer) > self.max_picture_bytes:
                self._skip_part("no JPEG end marker came")
                return True
            self._searched_up_to = len(self._buffer) - 1
            return False
        next_part = self._buffer.find(self.boundary, 0, end)
        if next_part >= 0:                     # this picture was cut off by the next part
            del self._buffer[:next_part]
            self._skip_part("its picture was cut off")
            return True
        pictures.append(bytes(self._buffer[:end + 2]))
        del self._buffer[:end + 2]
        self._state = "boundary"
        return True

    def _skip_part(self, reason: str) -> None:
        """Give up on the part being read and wait for the next boundary.

        Only the first skipped part is logged; later ones are counted.
        """
        self.skipped_parts += 1
        if self.skipped_parts == 1:
            log.warning("Skipped a part of the MJPEG stream: %s; later ones are counted, not logged",
                        reason)
        self._state = "boundary"
        self._length = None
        self._searched_up_to = 0

    def _keep_last(self, count: int) -> None:
        """Forget all but the last count bytes of the buffer."""
        surplus = len(self._buffer) - max(count, 0)
        if surplus > 0:
            del self._buffer[:surplus]


# ---- reading the phone over HTTP -----------------------------------------


def host_of(url: str) -> str:
    """The host and port of an address, for log lines: never a user name or password."""
    return urlsplit(url).netloc.rpartition("@")[2]


def request_headers(url: str) -> dict[str, str]:
    """HTTP headers for the request: a login when the address carries one."""
    parts = urlsplit(url)
    if parts.username is None:
        return {}
    login = f"{unquote(parts.username)}:{unquote(parts.password or '')}"
    return {"Authorization": "Basic " + base64.b64encode(login.encode("utf-8")).decode("ascii")}


class HttpStream:
    """One HTTP GET answer, read piece by piece. Standard library only.

    The socket is kept so that interrupt() can wake a read that is waiting on
    a silent phone, from another thread, at once.
    """

    def __init__(self, url: str, open_timeout: float, read_timeout: float) -> None:
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError("only http:// addresses can be read as MJPEG")
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        self._connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=open_timeout)
        try:
            self._connection.connect()
            self._socket = self._connection.sock   # http.client forgets it for an endless answer
            self._connection.request("GET", path, headers=request_headers(url))
            self._response = self._connection.getresponse()
            self._socket.settimeout(read_timeout)
        except BaseException:
            self._connection.close()
            raise
        self.status: int = self._response.status
        self.content_type: str = self._response.getheader("Content-Type", "")

    def read_chunk(self, size: int) -> bytes:
        """Whatever has arrived, up to size bytes, without waiting for all of them."""
        return self._response.read1(size)

    def interrupt(self) -> None:
        """Wake a read that is waiting, from any thread; that read then fails at once.

        On Windows shutdown() alone leaves the read waiting until its timeout,
        so the socket's handle is also taken from it and closed directly.
        """
        try:
            self._socket.shutdown(socket.SHUT_RDWR)   # tells the phone we are leaving
        except OSError:
            pass                                      # already closed
        try:
            socket.close(self._socket.detach())
        except OSError:
            pass

    def close(self) -> None:
        self._response.close()
        self._connection.close()
        self._socket.close()


class MjpegCapture:
    """Reads a phone's MJPEG stream; has read(), release() and isOpened() like cv2.VideoCapture.

    A small network thread cuts the stream into JPEG pictures and keeps only
    the newest one, undecoded, so the network never waits for decoding. read()
    decodes the newest picture at a reduced size. A picture that is replaced by
    a newer one before read() takes it is dropped, never queued.
    """

    def __init__(self, url: str, max_width: int, max_height: int,
                 open_timeout: float, read_timeout: float, opener=None) -> None:
        self.max_width = max_width
        self.max_height = max_height
        self.read_timeout = read_timeout
        self.other_format = False     # the server answered 200, but not with a multipart stream
        self.dropped = 0              # pictures replaced by a newer one before read() took them
        self.broken = 0               # pictures that could not be decoded
        self._stream = None
        self._parser: MjpegParser | None = None
        self._thread: threading.Thread | None = None
        self._newest: bytes | None = None
        self._receiving = False
        self._summary_logged = False
        self._closing = threading.Event()
        self._condition = threading.Condition()
        self._open(url, open_timeout, read_timeout, opener or HttpStream)

    def isOpened(self) -> bool:
        return self._stream is not None and not self._closing.is_set()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """The newest picture, decoded. (False, None) when no new one came within read_timeout."""
        deadline = time.monotonic() + self.read_timeout
        while True:
            picture = self._take_newest(deadline)
            if picture is None:
                return False, None
            frame = decode_jpeg(picture, self.max_width, self.max_height)
            if frame is not None:
                return True, frame
            self._note_broken_picture()

    def release(self) -> None:
        """Close the stream and end the network thread, within about a second."""
        self._closing.set()
        with self._condition:
            self._condition.notify_all()
        if self._stream is not None:
            self._stream.interrupt()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(RELEASE_WAIT_SECONDS)
        self._log_summary()

    @property
    def skipped_parts(self) -> int:
        """Parts of the stream the parser threw away (too big, cut off or malformed)."""
        return self._parser.skipped_parts if self._parser is not None else 0

    @property
    def wrong_lengths(self) -> int:
        """Parts whose Content-Length did not fit their picture."""
        return self._parser.wrong_lengths if self._parser is not None else 0

    def _log_summary(self) -> None:
        """One line, on the first release, about the trouble that was only counted."""
        if self._summary_logged:
            return
        self._summary_logged = True
        if self.skipped_parts or self.wrong_lengths or self.broken:
            log.info("MJPEG stream closed: skipped %d parts, %d wrong lengths, %d broken pictures",
                     self.skipped_parts, self.wrong_lengths, self.broken)

    # ---- opening -------------------------------------------------------

    def _open(self, url: str, open_timeout: float, read_timeout: float, opener) -> None:
        """Ask for the stream; start the network thread when the answer is MJPEG."""
        try:
            stream = opener(url, open_timeout, read_timeout)
        except (OSError, http.client.HTTPException, ValueError) as error:
            log.warning("Could not reach the phone at %s: %s", host_of(url), error)
            return
        boundary = multipart_boundary(stream.content_type)
        if stream.status != 200 or boundary is None:
            self.other_format = stream.status == 200
            log.warning("The phone at %s answered with status %s and type %r, not an MJPEG stream",
                        host_of(url), stream.status, stream.content_type)
            stream.close()
            return
        self._stream = stream
        self._parser = MjpegParser(boundary)
        self._receiving = True
        self._thread = threading.Thread(target=self._receive, name="mjpeg-reader", daemon=True)
        self._thread.start()

    # ---- the network thread --------------------------------------------

    def _receive(self) -> None:
        """Read pieces of the stream, cut them into pictures, keep the newest."""
        try:
            while not self._closing.is_set():
                piece = self._stream.read_chunk(CHUNK_BYTES)
                if not piece:
                    if not self._closing.is_set():
                        log.warning("The phone ended the stream")
                    break
                for picture in self._parser.feed(piece):
                    self._keep(picture)
        except (OSError, http.client.HTTPException, ValueError) as error:
            if not self._closing.is_set():           # a timeout lands here when the phone goes silent
                log.warning("The phone stream stopped: %s", error)
        finally:
            self._stream.close()
            with self._condition:
                self._receiving = False
                self._condition.notify_all()

    def _keep(self, picture: bytes) -> None:
        """Make this the newest picture, dropping an older one nobody took."""
        with self._condition:
            if self._newest is not None:
                self.dropped += 1
            self._newest = picture
            self._condition.notify_all()

    # ---- the reading side -----------------------------------------------

    def _take_newest(self, deadline: float) -> bytes | None:
        """Wait until deadline for a picture not handed out before; None when none came."""
        with self._condition:
            while self._newest is None and self._receiving and not self._closing.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._closing.is_set():
                return None
            picture, self._newest = self._newest, None
            return picture

    def _note_broken_picture(self) -> None:
        """Count a picture that would not decode; only the first one is logged."""
        self.broken += 1
        if self.broken == 1:
            log.warning("Skipped a broken picture from the phone (cut off or not a JPEG); "
                        "later ones are counted, not logged")


# ---- choosing how to read the phone --------------------------------------


def open_ffmpeg(url: str) -> Capture:
    """Open a stream with FFmpeg, asking it to keep no buffer."""
    # OpenCV reads this variable each time it opens an FFmpeg capture, and it
    # holds for the whole process: every FFmpeg capture opened later uses it too.
    # A value someone set on purpose is left alone.
    if FFMPEG_OPTIONS_VARIABLE not in os.environ:
        os.environ[FFMPEG_OPTIONS_VARIABLE] = FFMPEG_LOW_DELAY
    timeouts = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, timeouts)


def open_phone(
    url: str,
    max_width: int,
    max_height: int,
    open_mjpeg: Callable[..., MjpegCapture] = MjpegCapture,
    open_other: Callable[[str], Capture] = open_ffmpeg,
) -> Capture | None:
    """Open the phone: our own MJPEG reader first, FFmpeg when the phone speaks something else.

    A phone that cannot be reached is not tried again with FFmpeg: that would
    only double the wait. None when the phone could not be opened.
    """
    if urlsplit(url).scheme != "http":
        return open_other(url)                 # our reader speaks plain http only
    capture = open_mjpeg(url, max_width, max_height, OPEN_TIMEOUT_MS / 1000, READ_TIMEOUT_MS / 1000)
    if capture.isOpened():
        return capture
    capture.release()
    if capture.other_format:
        log.info("The phone does not send MJPEG; reading it with FFmpeg instead")
        return open_other(url)
    return None


class FrameSource:
    """Keeps the newest camera frame ready, and reconnects when the stream drops."""

    def __init__(
        self,
        settings: dict[str, Any],
        opener: Callable[[], Capture | None] | None = None,
    ) -> None:
        self.source: str = require(settings, "source", str)
        if self.source not in SOURCES:
            raise ConfigError(f"stream.source must be one of {SOURCES}, got {self.source!r}")
        self.width: int = require(settings, "width", int)
        self.height: int = require(settings, "height", int)
        self.reconnect_seconds: float = float(require(settings, "reconnect_seconds", (int, float)))
        self.loop_file: bool = require(settings, "loop_file", bool)
        self._settings = settings
        self._opener = opener if opener is not None else self._default_opener()

        self.fps = FpsCounter()
        self._lock = threading.Lock()
        self._first_frame = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._connected = False
        self._ended = False
        self._frame_interval = 0.0   # seconds between frames; only used for a clip

    # ---- public --------------------------------------------------------

    def start(self) -> "FrameSource":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="frame-source", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=READ_TIMEOUT_MS / 1000 + 2)
            self._thread = None

    def read(self) -> tuple[np.ndarray | None, int]:
        """Return a copy of the newest frame and its number.

        The number goes up by one for every new frame, so a caller can tell
        whether it has already processed this picture. The frame is None until
        the first one arrives.
        """
        with self._lock:
            if self._frame is None:
                return None, self._sequence
            return self._frame.copy(), self._sequence

    def wait_for_frame(self, timeout: float = 10.0) -> bool:
        """Block until the first frame arrives. False when it never did."""
        return self._first_frame.wait(timeout)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def ended(self) -> bool:
        """True once a clip has played to its end with looping switched off."""
        return self._ended

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # ---- the reader thread ---------------------------------------------

    def _run(self) -> None:
        capture: Capture | None = None
        frames_since_open = 0
        try:
            while not self._stop.is_set():
                if capture is None:
                    capture = self._open()
                    frames_since_open = 0
                    if capture is None:
                        self._connected = False
                        self._stop.wait(self.reconnect_seconds)
                        continue

                started = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    capture.release()
                    capture = None
                    clip_finished = self.source == "file" and frames_since_open > 0
                    if clip_finished and self.loop_file:
                        continue                      # start the clip again at once
                    self._connected = False
                    if clip_finished:
                        self._ended = True
                        return
                    log.warning("Lost the %s stream; trying again in %.1f s",
                                self.source, self.reconnect_seconds)
                    self._stop.wait(self.reconnect_seconds)
                    continue

                frames_since_open += 1
                fitted = self._fit(frame)
                with self._lock:
                    self._frame = fitted
                    self._sequence += 1
                self._connected = True
                self.fps.tick()
                self._first_frame.set()

                if self._frame_interval > 0:           # play a clip at its own speed
                    spare = self._frame_interval - (time.monotonic() - started)
                    if spare > 0:
                        self._stop.wait(spare)
        finally:
            if capture is not None:
                capture.release()
            self._connected = False

    def _open(self) -> Capture | None:
        try:
            capture = self._opener()
        except ConfigError:
            raise
        except Exception:                              # camera drivers raise many kinds
            log.exception("Could not open the %s stream", self.source)
            return None
        if capture is None or not capture.isOpened():
            log.warning("The %s stream did not open", self.source)
            if capture is not None:
                capture.release()
            return None
        if self.source == "file":
            get = getattr(capture, "get", None)
            clip_fps = float(get(cv2.CAP_PROP_FPS)) if callable(get) else 0.0
            self._frame_interval = 1.0 / clip_fps if clip_fps > 1.0 else 1.0 / 30.0
        log.info("Opened the %s stream", self.source)
        return capture

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        """Shrink the frame to fit inside width x height without changing its shape.

        A wide 1920 x 1080 phone picture becomes 640 x 360, not a squashed 640 x 480:
        squashed faces are harder to recognise.
        """
        size = fitted_size(frame.shape[1], frame.shape[0], self.width, self.height)
        if size == (frame.shape[1], frame.shape[0]):
            return frame
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    # ---- real cameras ----------------------------------------------------

    def _default_opener(self) -> Callable[[], Capture | None]:
        settings = self._settings
        if self.source == "phone":
            url: str = require(settings, "phone_url", str)

            def open_phone_stream() -> Capture | None:
                return open_phone(url, self.width, self.height)

            return open_phone_stream

        if self.source == "webcam":
            index: int = require(settings, "webcam_index", int)

            def open_webcam() -> Capture:
                capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # opens fast on Windows
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return capture

            return open_webcam

        path = resolve_path(require(settings, "file_path", str))
        if not path.is_file():
            raise ConfigError(
                f"stream.source is 'file' but the clip does not exist: {path}\n"
                "Record a clip into data/samples/ or set stream.source to 'webcam'."
            )

        def open_file() -> Capture:
            return cv2.VideoCapture(str(path))

        return open_file


# ---- running this file on its own ----------------------------------------


def draw_status(frame: np.ndarray, text: str) -> None:
    """Write one line of status in the top-left corner, readable on any background."""
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def clock_image() -> np.ndarray:
    """A big clock with milliseconds. Point the phone at it to measure the delay:
    the clock itself shows now, the video of the clock shows then."""
    image = np.zeros((160, 560, 3), dtype=np.uint8)
    text = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(image, text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show the live camera stream.")
    parser.add_argument("--source", choices=SOURCES, help="override stream.source from config.yaml")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop by itself after this long")
    parser.add_argument("--no-window", action="store_true", help="print the FPS instead of showing video")
    parser.add_argument("--record", metavar="FILE", help="also save the video, e.g. data/samples/car_clip.mp4")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = dict(section(load_config(), "stream"))
    if args.source:
        settings["source"] = args.source

    source = FrameSource(settings)
    show_clock = False
    with source:
        if not source.wait_for_frame(timeout=15.0):
            print(f"No frame arrived from the {source.source} stream after 15 seconds.")
            if source.source == "phone":
                print("Check that IP Webcam is running, that the phone and the laptop are on "
                      "the same network, and that stream.phone_url in config.yaml is right.")
            return 1

        writer = None
        if args.record:
            first_frame, _ = source.read()
            record_path = resolve_path(args.record)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            frame_size = (first_frame.shape[1], first_frame.shape[0])
            writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     RECORD_FPS, frame_size)
            if not writer.isOpened():
                print(f"Could not create the recording file: {record_path}")
                return 1
            print(f"Recording {frame_size[0]} x {frame_size[1]} video to {record_path}")

        started = time.monotonic()
        last_print = started
        last_sequence = 0
        next_record = started
        while not source.ended:
            now = time.monotonic()
            if args.seconds and now - started >= args.seconds:
                break
            frame, sequence = source.read()
            # Save at a steady rate, so the clip plays back at the speed it was filmed.
            if writer is not None and frame is not None and now >= next_record and sequence != last_sequence:
                writer.write(frame)
                last_sequence = sequence
                next_record = max(next_record + 1.0 / RECORD_FPS, now)
            status = f"FPS {source.fps.value():.1f} | {source.source}"
            if writer is not None:
                status += " | recording"
            if not source.connected:
                status += " | reconnecting"
            if args.no_window:
                if now - last_print >= 1.0:
                    print(status)
                    last_print = now
                time.sleep(0.05)
                continue
            if frame is not None:
                draw_status(frame, status)
                cv2.imshow("Car camera", frame)
            if show_clock:
                cv2.imshow("Clock", clock_image())
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                show_clock = not show_clock
                if not show_clock:
                    cv2.destroyWindow("Clock")
        if writer is not None:
            writer.release()
        print(f"Final FPS {source.fps.value():.1f} from the {source.source} stream")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
