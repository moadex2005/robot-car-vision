"""The stream logic, tested with a fake camera. No phone or webcam needed."""

import os
import time

import cv2
import numpy as np
import pytest

from car_vision import stream
from car_vision.config import ConfigError
from car_vision.stream import FpsCounter, FrameSource, fitted_size, open_ffmpeg

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


# ---- set_box(): a new box while the video runs --------------------------------


def frame_shape(source):
    frame, _ = source.read()
    return None if frame is None else frame.shape[:2]


def test_set_box_takes_effect_from_the_next_frame():
    source = FrameSource(SETTINGS, opener=lambda: FakeCapture(frames=10_000_000, size=(108, 192)))
    with source:
        assert wait_until(lambda: frame_shape(source) == (36, 64))        # the config box: 64 x 48
        source.set_box(128, 128)
        assert wait_until(lambda: frame_shape(source) == (72, 128))
        source.set_box(1920, 1080)                                       # larger than the picture ...
        assert wait_until(lambda: frame_shape(source) == (108, 192))     # ... which is never enlarged
        source.set_box(64, 48)                                           # and back
        assert wait_until(lambda: frame_shape(source) == (36, 64))


def test_set_box_from_another_thread_never_mixes_two_boxes():
    # The picture is 192 x 108. The box 96 x 1000 makes it 96 x 54, the box 1000 x 27
    # makes it 48 x 28. A frame fitted with the width of the second box and the
    # height of the first (1000 x 1000) would stay 192 x 108, so it would show.
    boxes = [(96, 1000), (1000, 27)]
    assert fitted_size(192, 108, 1000, 1000) == (192, 108)
    source = FrameSource(SETTINGS, opener=lambda: FakeCapture(frames=10_000_000, size=(108, 192)))
    seen = set()
    with source:
        assert source.wait_for_frame(2.0)
        for turn in range(2000):
            source.set_box(*boxes[turn % 2])
            seen.add(frame_shape(source))
    assert seen <= {(54, 96), (28, 48), (36, 64)}         # (36, 64): the config box, before the first set_box
    assert (54, 96) in seen and (28, 48) in seen


def test_reopen_opens_the_stream_again_without_calling_it_lost():
    """After a phone changes its picture size the old capture keeps the old frame size."""
    captures = []

    def opener():
        # The first capture gives small frames for ever; the one opened after reopen() gives large ones.
        size = (36, 64) if not captures else (72, 128)
        captures.append(FakeCapture(frames=10_000_000, size=size))
        return captures[-1]

    source = FrameSource({**SETTINGS, "width": 640, "height": 480}, opener=opener)
    with source:
        assert wait_until(lambda: frame_shape(source) == (36, 64))
        source.reopen()
        assert wait_until(lambda: frame_shape(source) == (72, 128))
        assert source.connected                     # never shown as a lost stream
        assert captures[0].released and len(captures) == 2


def test_set_box_refuses_sides_that_are_not_positive_whole_numbers():
    source = FrameSource(SETTINGS, opener=lambda: FakeCapture(0))
    for bad in [(0, 48), (64, -1), (64.0, 48), (True, 48), ("64", 48)]:
        with pytest.raises(ValueError):
            source.set_box(*bad)


def test_set_box_leaves_the_configured_size_for_the_webcam_alone(monkeypatch):
    calls = []

    class FakeWebcam:
        def __init__(self, *args):
            pass

        def set(self, prop, value):
            calls.append((prop, value))

    monkeypatch.setattr(stream.cv2, "VideoCapture", FakeWebcam)
    source = FrameSource(SETTINGS)
    source.set_box(1920, 1080)
    source._opener()
    assert calls == [(cv2.CAP_PROP_FRAME_WIDTH, 64), (cv2.CAP_PROP_FRAME_HEIGHT, 48)]


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



# ---- the phone, read with FFmpeg -------------------------------------------


class FakeVideoCapture:
    def __init__(self, *args) -> None:
        self.args = args
        self.options = os.environ.get(stream.FFMPEG_OPTIONS_VARIABLE)


def test_ffmpeg_asks_for_low_delay_and_the_timeouts(monkeypatch):
    monkeypatch.setenv(stream.FFMPEG_OPTIONS_VARIABLE, "placeholder")
    monkeypatch.delenv(stream.FFMPEG_OPTIONS_VARIABLE)    # restored after the test
    monkeypatch.setattr(stream.cv2, "VideoCapture", FakeVideoCapture)
    capture = open_ffmpeg("http://192.168.0.2:8080/video")
    assert capture.options == "fflags;nobuffer|flags;low_delay"
    assert capture.args == ("http://192.168.0.2:8080/video", cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, stream.OPEN_TIMEOUT_MS,
                             cv2.CAP_PROP_READ_TIMEOUT_MSEC, stream.READ_TIMEOUT_MS])


def test_frame_source_opens_the_phone_with_ffmpeg(monkeypatch):
    calls = []
    monkeypatch.setattr(stream, "open_ffmpeg", lambda url: calls.append(url))
    FrameSource({**SETTINGS, "source": "phone"})._opener()
    assert calls == ["http://192.168.0.2:8080/video"]


def test_ffmpeg_keeps_options_someone_set(monkeypatch):
    monkeypatch.setenv(stream.FFMPEG_OPTIONS_VARIABLE, "rtsp_transport;tcp")
    monkeypatch.setattr(stream.cv2, "VideoCapture", FakeVideoCapture)
    assert open_ffmpeg("http://192.168.0.2:8080/video").options == "rtsp_transport;tcp"
