"""The stream logic, tested with a fake camera. No phone or webcam needed.

The MJPEG reader is tested on byte streams built here in memory: no sockets.
"""

import logging
import os
import queue
import socket
import threading
import time

import cv2
import numpy as np
import pytest

from car_vision import stream
from car_vision.config import ConfigError
from car_vision.stream import (
    FpsCounter,
    FrameSource,
    MjpegCapture,
    MjpegParser,
    decode_jpeg,
    fitted_size,
    host_of,
    is_whole_jpeg,
    jpeg_size,
    multipart_boundary,
    open_ffmpeg,
    open_phone,
    reduction_for,
    request_headers,
)

SETTINGS = {
    "source": "webcam",
    "phone_url": "http://192.168.0.2:8080/video",
    "webcam_index": 0,
    "file_path": "data/samples/missing.mp4",
    "width": 64,
    "height": 48,
    "reconnect_seconds": 0.01,
    "loop_file": True,
}


class FakeCapture:
    """Hands out numbered frames, then reports failure when they run out."""

    def __init__(self, frames: int, size=(48, 64), opened: bool = True) -> None:
        self.remaining = frames
        self.size = size
        self.opened = opened
        self.released = False
        self.next_value = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        self.next_value += 1
        return True, np.full((*self.size, 3), self.next_value % 256, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def wait_until(condition, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


def test_fps_counter_measures_rate():
    counter = FpsCounter(window_seconds=2.0)
    for step in range(11):
        counter.tick(now=step * 0.1)          # 10 frames per second
    assert counter.value(now=1.0) == pytest.approx(10.0)


def test_fps_counter_forgets_old_frames():
    counter = FpsCounter(window_seconds=2.0)
    counter.tick(now=0.0)
    counter.tick(now=0.1)
    assert counter.value(now=10.0) == 0.0


def test_read_before_first_frame_returns_none():
    source = FrameSource(SETTINGS, opener=lambda: FakeCapture(0))
    frame, sequence = source.read()
    assert frame is None and sequence == 0


def test_keeps_only_the_latest_frame():
    capture = FakeCapture(frames=5)
    source = FrameSource(SETTINGS, opener=lambda: capture)
    with source:
        assert wait_until(lambda: source.read()[1] == 5)
        frame, sequence = source.read()
    assert sequence == 5
    assert int(frame[0, 0, 0]) == 5           # the newest picture, not the first


def test_read_returns_a_copy():
    source = FrameSource(SETTINGS, opener=lambda: FakeCapture(frames=1))
    with source:
        assert source.wait_for_frame(2.0)
        first, _ = source.read()
        first[:] = 0                          # a caller drawing on its frame
        second, _ = source.read()
    assert int(second[0, 0, 0]) == 1


def test_resizes_to_configured_size():
    source = FrameSource(SETTINGS, opener=lambda: FakeCapture(frames=1, size=(96, 128)))
    with source:
        assert source.wait_for_frame(2.0)
        frame, _ = source.read()
    assert frame.shape == (48, 64, 3)


def test_wide_frame_keeps_its_shape():
    # A 16:9 phone picture must not be squashed into the 4:3 setting.
    source = FrameSource(SETTINGS, opener=lambda: FakeCapture(frames=1, size=(108, 192)))
    with source:
        assert source.wait_for_frame(2.0)
        frame, _ = source.read()
    assert frame.shape == (36, 64, 3)


def test_small_frame_is_not_enlarged():
    source = FrameSource(SETTINGS, opener=lambda: FakeCapture(frames=1, size=(24, 32)))
    with source:
        assert source.wait_for_frame(2.0)
        frame, _ = source.read()
    assert frame.shape == (24, 32, 3)


def test_fitted_size_examples():
    assert fitted_size(1920, 1080, 640, 480) == (640, 360)
    assert fitted_size(640, 480, 640, 480) == (640, 480)
    assert fitted_size(1280, 960, 640, 480) == (640, 480)
    assert fitted_size(320, 240, 640, 480) == (320, 240)


def test_reconnects_after_the_stream_drops():
    captures = [FakeCapture(frames=2), FakeCapture(frames=3)]
    opened = []

    def opener():
        capture = captures[len(opened)] if len(opened) < len(captures) else FakeCapture(0, opened=False)
        opened.append(capture)
        return capture

    source = FrameSource(SETTINGS, opener=opener)
    with source:
        assert wait_until(lambda: source.read()[1] == 5)
    assert len(opened) >= 2
    assert captures[0].released


def test_keeps_trying_when_the_camera_will_not_open():
    attempts = []

    def opener():
        attempts.append(1)
        return FakeCapture(0, opened=False)

    source = FrameSource(SETTINGS, opener=opener)
    with source:
        assert wait_until(lambda: len(attempts) >= 3)
        assert not source.connected
        assert source.read()[0] is None


def test_survives_an_opener_that_raises():
    attempts = []

    def opener():
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("driver hiccup")
        return FakeCapture(frames=1)

    source = FrameSource(SETTINGS, opener=opener)
    with source:
        assert source.wait_for_frame(2.0)


def test_clip_ends_when_looping_is_off():
    settings = {**SETTINGS, "source": "file", "loop_file": False}
    source = FrameSource(settings, opener=lambda: FakeCapture(frames=2))
    with source:
        assert wait_until(lambda: source.ended)
    assert source.read()[1] == 2


def test_clip_loops_when_looping_is_on():
    settings = {**SETTINGS, "source": "file", "loop_file": True}
    source = FrameSource(settings, opener=lambda: FakeCapture(frames=2))
    with source:
        assert wait_until(lambda: source.read()[1] >= 5)
        assert not source.ended


def test_stop_ends_the_thread_quickly():
    source = FrameSource(SETTINGS, opener=lambda: FakeCapture(frames=10_000_000))
    source.start()
    assert source.wait_for_frame(2.0)
    started = time.monotonic()
    source.stop()
    assert time.monotonic() - started < 2.0


def test_unknown_source_is_rejected():
    with pytest.raises(ConfigError):
        FrameSource({**SETTINGS, "source": "drone"})


def test_missing_clip_gives_a_clear_error():
    with pytest.raises(ConfigError, match="does not exist"):
        FrameSource({**SETTINGS, "source": "file"})


# ---- the MJPEG reader ------------------------------------------------------

MULTIPART = "multipart/x-mixed-replace; boundary=frame"


def make_jpeg(value: int = 100, size=(48, 64), progressive: bool = False) -> bytes:
    """A small JPEG of one grey value, made here so no file is needed."""
    image = np.full((*size, 3), value, dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (size[1] // 2, size[0] // 2), (255 - value,) * 3, -1)
    options = [cv2.IMWRITE_JPEG_PROGRESSIVE, 1] if progressive else []
    ok, data = cv2.imencode(".jpg", image, options)
    assert ok
    return data.tobytes()


def make_part(picture: bytes, boundary: bytes = b"--frame", with_length: bool = True) -> bytes:
    """One part of a multipart stream, laid out the way IP Webcam sends it."""
    headers = boundary + b"\r\nContent-Type: image/jpeg\r\n"
    if with_length:
        headers += b"Content-Length: %d\r\n" % len(picture)
    return headers + b"\r\n" + picture + b"\r\n"


def feed_in_pieces(parser: MjpegParser, data: bytes, piece_size: int) -> list[bytes]:
    pictures = []
    for start in range(0, len(data), piece_size):
        pictures += parser.feed(data[start:start + piece_size])
    return pictures


def grey_value(frame: np.ndarray) -> int:
    """The grey value in the bottom-right corner, where make_jpeg draws no square."""
    return int(frame[-1, -1, 0])


class FakeStream:
    """Stands in for HttpStream: hands out the pieces it is given, in order.

    When it has none, it waits like a silent phone until interrupt() is called.
    """

    def __init__(self, pieces=(), status: int = 200, content_type: str = MULTIPART,
                 ends: bool = False) -> None:
        self.status = status
        self.content_type = content_type
        self.pieces: queue.Queue = queue.Queue()
        for piece in pieces:
            self.pieces.put(piece)
        if ends:
            self.pieces.put(b"")
        self.interrupted = False
        self.closed = False

    def send(self, piece: bytes) -> None:
        self.pieces.put(piece)

    def read_chunk(self, size: int) -> bytes:
        return self.pieces.get(timeout=10)

    def interrupt(self) -> None:
        self.interrupted = True
        self.pieces.put(b"")

    def close(self) -> None:
        self.closed = True


def capture_for(fake: FakeStream, read_timeout: float = 1.0, max_size=(64, 48)) -> MjpegCapture:
    return MjpegCapture("http://192.168.0.2:8080/video", max_size[0], max_size[1],
                        open_timeout=1.0, read_timeout=read_timeout,
                        opener=lambda url, open_timeout, read_timeout: fake)


# -- the content type

@pytest.mark.parametrize("content_type", [
    "multipart/x-mixed-replace; boundary=frame",
    'multipart/x-mixed-replace; boundary="frame"',
    "multipart/x-mixed-replace;boundary=--frame",
    'Multipart/X-Mixed-Replace; charset=utf-8; Boundary="--frame"',
])
def test_boundary_with_or_without_quotes_and_dashes(content_type):
    assert multipart_boundary(content_type) == b"frame"


@pytest.mark.parametrize("content_type", ["image/jpeg", "text/html; boundary=frame",
                                          "multipart/x-mixed-replace", 'multipart/mixed; boundary=""'])
def test_no_boundary_when_not_multipart_or_missing(content_type):
    assert multipart_boundary(content_type) is None


# -- the parser

def test_parser_uses_content_length():
    first, second = make_jpeg(10), make_jpeg(200)
    parser = MjpegParser(b"frame")
    assert parser.feed(make_part(first) + make_part(second)) == [first, second]


def test_parser_finds_pictures_without_content_length():
    first, second = make_jpeg(10), make_jpeg(200)
    parser = MjpegParser(b"frame")
    data = make_part(first, with_length=False) + make_part(second, with_length=False)
    assert parser.feed(data) == [first, second]


def test_parser_works_whether_the_server_adds_dashes_or_not():
    # Declared as boundary=--frame: some servers then send "--frame", others "----frame".
    first, second = make_jpeg(10), make_jpeg(200)
    parser = MjpegParser(multipart_boundary('multipart/x-mixed-replace; boundary="--frame"'))
    data = make_part(first, boundary=b"--frame") + make_part(second, boundary=b"----frame")
    assert parser.feed(data) == [first, second]


@pytest.mark.parametrize("with_length", [True, False])
def test_parser_joins_a_picture_split_across_many_reads(with_length):
    first, second = make_jpeg(10), make_jpeg(200)
    data = make_part(first, with_length=with_length) + make_part(second, with_length=with_length)
    for piece_size in (1, 7, 100):
        assert feed_in_pieces(MjpegParser(b"frame"), data, piece_size) == [first, second]


@pytest.mark.parametrize("with_length", [True, False])
def test_parser_skips_junk_between_parts(with_length):
    first, second = make_jpeg(10), make_jpeg(200)
    junk = b"\r\nnoise\xff\xd8 not a picture\xff\xd9\r\n\r\n"
    data = junk + make_part(first, with_length=with_length) + junk + make_part(second, with_length=with_length)
    assert MjpegParser(b"frame").feed(data) == [first, second]


@pytest.mark.parametrize("with_length", [True, False])
def test_parser_holds_back_a_cut_off_picture(with_length):
    picture = make_jpeg(10)
    data = make_part(picture, with_length=with_length)
    parser = MjpegParser(b"frame")
    assert parser.feed(data[:len(data) // 2]) == []      # the stream stops here


def test_parser_drops_a_cut_off_picture_when_the_next_part_begins():
    picture = make_jpeg(200)
    cut = make_part(make_jpeg(10), with_length=False)[:200]
    parser = MjpegParser(b"frame")
    assert parser.feed(cut + make_part(picture, with_length=False)) == [picture]
    assert parser.skipped_parts == 1


def test_parser_refuses_a_picture_that_is_too_big():
    picture = make_jpeg(200)
    huge = b"--frame\r\nContent-Length: 999999999\r\n\r\n" + b"x" * 5000
    parser = MjpegParser(b"frame", max_picture_bytes=4000)
    assert parser.feed(huge + make_part(picture)) == [picture]
    assert parser.skipped_parts == 1


def test_parser_memory_stays_bounded_without_an_end_marker():
    parser = MjpegParser(b"frame", max_picture_bytes=10_000)
    parser.feed(b"--frame\r\n\r\n\xff\xd8")
    for _ in range(100):
        parser.feed(b"\x00" * 1000)                    # a picture that never ends
        assert parser.buffered_bytes <= 11_000
    assert parser.skipped_parts >= 1
    picture = make_jpeg(200)
    assert parser.feed(make_part(picture, with_length=False)) == [picture]


def test_parser_memory_stays_bounded_without_any_boundary():
    parser = MjpegParser(b"frame")
    for _ in range(100):
        parser.feed(b"\x00" * 10_000)
    assert parser.buffered_bytes < 10


def make_part_with_length(picture: bytes, length: int) -> bytes:
    """A part whose Content-Length says length, whatever the picture's real size."""
    return b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % length + picture + b"\r\n"


@pytest.mark.parametrize("error", [+100, -50])
def test_parser_survives_a_wrong_content_length(error, caplog):
    pictures = [make_jpeg(value) for value in (10, 100, 200)]
    data = b"".join(make_part_with_length(picture, len(picture) + error) for picture in pictures)
    data += make_part(make_jpeg(50))                     # one more, so the last wrong one completes
    with caplog.at_level(logging.WARNING, logger="car_vision.stream"):
        for piece_size in (len(data), 1000):
            parser = MjpegParser(b"frame")
            assert feed_in_pieces(parser, data, piece_size)[:3] == pictures
            assert parser.wrong_lengths == 3
            assert parser.skipped_parts == 0
    assert sum("Content-Length" in record.getMessage() for record in caplog.records) == 2   # once per parser


def test_parser_logs_only_the_first_skipped_part(caplog):
    huge = b"--frame\r\nContent-Length: 999999999\r\n\r\n"
    parser = MjpegParser(b"frame", max_picture_bytes=4000)
    with caplog.at_level(logging.WARNING, logger="car_vision.stream"):
        parser.feed(huge * 5)
    assert parser.skipped_parts == 5
    assert sum("Skipped a part" in record.getMessage() for record in caplog.records) == 1


# -- the JPEG header and the decode size

@pytest.mark.parametrize("progressive", [False, True])
def test_jpeg_size_reads_the_header(progressive):
    assert jpeg_size(make_jpeg(size=(108, 192), progressive=progressive)) == (192, 108)


def test_jpeg_size_none_for_a_non_jpeg_or_a_cut_header():
    assert jpeg_size(b"not a picture") is None
    assert jpeg_size(make_jpeg()[:20]) is None


def test_reduction_for_examples():
    assert reduction_for(1920, 1080, 640, 480) == 2    # 4 would give 480 x 270: too small
    assert reduction_for(1280, 720, 640, 480) == 2
    assert reduction_for(640, 480, 640, 480) == 1
    assert reduction_for(32, 24, 640, 480) == 1
    assert reduction_for(3840, 2160, 640, 480) == 4
    assert reduction_for(0, 0, 640, 480) == 1


def test_decode_jpeg_decodes_at_a_reduced_size():
    frame = decode_jpeg(make_jpeg(size=(1080, 1920)), 640, 480)
    assert frame.shape == (540, 960, 3)


def test_decode_jpeg_decodes_in_full_when_the_size_is_unknown(monkeypatch):
    monkeypatch.setattr(stream, "jpeg_size", lambda data: None)
    assert decode_jpeg(make_jpeg(size=(1080, 1920)), 640, 480).shape == (1080, 1920, 3)


def test_a_jpeg_may_end_with_padding():
    picture = make_jpeg(200)
    for padding in (b"\r\n", b"\x00" * 16, b"\x00\r\n\x00"):
        assert is_whole_jpeg(picture + padding)
        assert decode_jpeg(picture + padding, 640, 480) is not None
    assert not is_whole_jpeg(picture[:-2] + b"\x00\x00")


def test_decode_jpeg_refuses_broken_pictures():
    picture = make_jpeg()
    assert decode_jpeg(b"not a picture", 640, 480) is None
    assert decode_jpeg(picture[:len(picture) // 2], 640, 480) is None


# -- MjpegCapture

def test_capture_opens_on_a_multipart_answer():
    fake = FakeStream()
    capture = capture_for(fake)
    assert capture.isOpened()
    capture.release()
    assert not capture.isOpened()


@pytest.mark.parametrize("status, content_type, other_format", [
    (404, MULTIPART, False),
    (200, "text/html", True),
    (200, "multipart/x-mixed-replace", True),          # multipart, but no boundary
])
def test_capture_does_not_open_on_another_answer(status, content_type, other_format):
    fake = FakeStream(status=status, content_type=content_type)
    capture = capture_for(fake)
    assert not capture.isOpened()
    assert capture.other_format is other_format
    assert fake.closed
    assert capture.read() == (False, None)
    capture.release()


@pytest.mark.parametrize("error", [ConnectionRefusedError("refused"), socket.timeout("timed out")])
def test_capture_does_not_open_when_the_phone_is_not_there(error):
    def opener(url, open_timeout, read_timeout):
        raise error

    capture = MjpegCapture("http://192.168.0.2:8080/video", 64, 48, 1.0, 1.0, opener=opener)
    assert not capture.isOpened()
    assert not capture.other_format
    capture.release()


def test_capture_hands_out_the_newest_of_three_waiting_pictures():
    pieces = [make_part(make_jpeg(value)) for value in (10, 100, 200)]
    fake = FakeStream([b"".join(pieces)])
    capture = capture_for(fake)
    try:
        assert wait_until(lambda: capture.dropped == 2)
        ok, frame = capture.read()
        assert ok and abs(grey_value(frame) - 200) <= 3
    finally:
        capture.release()


def test_capture_read_decodes_small():
    fake = FakeStream([make_part(make_jpeg(size=(1080, 1920)))])
    capture = capture_for(fake, max_size=(640, 480))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok and frame.shape == (540, 960, 3)


def test_capture_read_times_out_and_never_repeats_a_picture():
    fake = FakeStream([make_part(make_jpeg(10))])
    capture = capture_for(fake, read_timeout=0.3)
    try:
        assert capture.read()[0]
        started = time.monotonic()
        assert capture.read() == (False, None)          # the phone has gone silent
        waited = time.monotonic() - started
    finally:
        capture.release()
    assert 0.25 <= waited < 1.0


def test_capture_read_answers_at_once_when_the_stream_ended():
    fake = FakeStream([make_part(make_jpeg(10))], ends=True)
    capture = capture_for(fake, read_timeout=5.0)
    try:
        assert capture.read()[0]
        started = time.monotonic()
        assert capture.read() == (False, None)
        assert time.monotonic() - started < 1.0
    finally:
        capture.release()


def test_capture_skips_broken_pictures_and_logs_once(caplog):
    fake = FakeStream()
    capture = capture_for(fake, read_timeout=0.3)
    try:
        with caplog.at_level(logging.WARNING, logger="car_vision.stream"):
            # Both have JPEG start and end markers, so only the decoder can tell they are broken.
            fake.send(make_part(b"\xff\xd8 not a jpeg at all \xff\xd9"))
            assert capture.read() == (False, None)
            fake.send(make_part(b"\xff\xd8\xff\xe0 garbage \xff\xd9"))
            assert capture.read() == (False, None)
        fake.send(make_part(make_jpeg(200)))
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok and abs(grey_value(frame) - 200) <= 3
    assert capture.broken == 2
    assert sum("broken picture" in record.getMessage() for record in caplog.records) == 1


def test_release_ends_the_thread_of_a_silent_phone():
    fake = FakeStream()
    capture = capture_for(fake)
    started = time.monotonic()
    capture.release()
    assert time.monotonic() - started < 1.0
    assert fake.interrupted and fake.closed
    assert not capture._thread.is_alive()


def test_release_logs_one_summary_of_counted_trouble(caplog):
    huge = b"--frame\r\nContent-Length: 999999999\r\n\r\n"
    fake = FakeStream([huge * 3])
    capture = capture_for(fake)
    assert wait_until(lambda: capture.skipped_parts == 3)
    with caplog.at_level(logging.INFO, logger="car_vision.stream"):
        capture.release()
        capture.release()
    summaries = [record.getMessage() for record in caplog.records if "skipped" in record.getMessage()]
    assert summaries == ["MJPEG stream closed: skipped 3 parts, 0 wrong lengths, 0 broken pictures"]


def test_release_wakes_a_waiting_read():
    fake = FakeStream()
    capture = capture_for(fake, read_timeout=5.0)
    results = []
    reader = threading.Thread(target=lambda: results.append(capture.read()))
    reader.start()
    time.sleep(0.1)
    capture.release()
    reader.join(1.0)
    assert results == [(False, None)]


def test_frame_source_reads_the_phone_through_mjpeg_capture():
    picture = make_jpeg(size=(108, 192))                # becomes 64 x 36 in the end
    settings = {**SETTINGS, "source": "phone"}

    def opener():
        return capture_for(FakeStream([make_part(picture)] * 3), max_size=(64, 48))

    source = FrameSource(settings, opener=opener)
    with source:
        assert source.wait_for_frame(2.0)
        frame, _ = source.read()
    assert frame.shape == (36, 64, 3)


# -- which reader opens the phone

class FakeMjpegCapture:
    def __init__(self, opened: bool, other_format: bool) -> None:
        self.opened = opened
        self.other_format = other_format
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.released = True


class Openers:
    """Fake openers for open_phone that remember how they were called."""

    def __init__(self, opened: bool = True, other_format: bool = False) -> None:
        self.mjpeg_calls = []
        self.other_calls = []
        self.mjpeg = FakeMjpegCapture(opened, other_format)
        self.other = FakeCapture(frames=1)

    def open_mjpeg(self, *args):
        self.mjpeg_calls.append(args)
        return self.mjpeg

    def open_other(self, url):
        self.other_calls.append(url)
        return self.other

    def open(self, url="http://192.168.0.2:8080/video"):
        return open_phone(url, 640, 480, open_mjpeg=self.open_mjpeg, open_other=self.open_other)


def test_open_phone_uses_mjpeg_when_it_opens():
    openers = Openers(opened=True)
    assert openers.open() is openers.mjpeg
    assert openers.mjpeg_calls == [("http://192.168.0.2:8080/video", 640, 480, 5.0, 5.0)]
    assert openers.other_calls == []


def test_open_phone_falls_back_when_the_phone_speaks_something_else():
    openers = Openers(opened=False, other_format=True)
    assert openers.open() is openers.other
    assert openers.other_calls == ["http://192.168.0.2:8080/video"]
    assert openers.mjpeg.released


def test_open_phone_does_not_fall_back_when_the_phone_is_not_there():
    openers = Openers(opened=False, other_format=False)
    assert openers.open() is None
    assert openers.other_calls == []
    assert openers.mjpeg.released


def test_open_phone_leaves_https_to_ffmpeg():
    openers = Openers()
    assert openers.open("https://192.168.0.2:8080/video") is openers.other
    assert openers.mjpeg_calls == []


def test_frame_source_opens_the_phone_with_open_phone(monkeypatch):
    calls = []
    monkeypatch.setattr(stream, "open_phone", lambda *args: calls.append(args))
    FrameSource({**SETTINGS, "source": "phone"})._opener()
    assert calls == [("http://192.168.0.2:8080/video", 64, 48)]


class FakeVideoCapture:
    def __init__(self, *args) -> None:
        self.args = args
        self.options = os.environ.get(stream.FFMPEG_OPTIONS_VARIABLE)


def test_ffmpeg_fallback_asks_for_low_delay(monkeypatch):
    monkeypatch.setenv(stream.FFMPEG_OPTIONS_VARIABLE, "placeholder")
    monkeypatch.delenv(stream.FFMPEG_OPTIONS_VARIABLE)    # restored after the test
    monkeypatch.setattr(stream.cv2, "VideoCapture", FakeVideoCapture)
    capture = open_ffmpeg("http://192.168.0.2:8080/video")
    assert capture.options == "fflags;nobuffer|flags;low_delay"
    assert capture.args[:2] == ("http://192.168.0.2:8080/video", cv2.CAP_FFMPEG)


def test_ffmpeg_fallback_keeps_options_someone_set(monkeypatch):
    monkeypatch.setenv(stream.FFMPEG_OPTIONS_VARIABLE, "rtsp_transport;tcp")
    monkeypatch.setattr(stream.cv2, "VideoCapture", FakeVideoCapture)
    assert open_ffmpeg("http://192.168.0.2:8080/video").options == "rtsp_transport;tcp"


# -- addresses

def test_host_of_leaves_out_the_login():
    assert host_of("http://user:pw@192.168.0.2:8080/video") == "192.168.0.2:8080"
    assert host_of("http://192.168.0.2:8080/video") == "192.168.0.2:8080"


def test_request_headers_carry_a_login_only_when_given():
    assert request_headers("http://192.168.0.2:8080/video") == {}
    assert request_headers("http://user:pw@192.168.0.2:8080/video") == {"Authorization": "Basic dXNlcjpwdw=="}