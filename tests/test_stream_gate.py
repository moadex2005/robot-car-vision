"""Only one reader may have the webcam at a time (stream.WEBCAM_GATE).

Seen live: the camera switched off and on quickly opened the webcam while the
capture before it was still being closed, and OpenCV's Windows driver ended
Python with corrupted memory. These tests use fake captures and a gate of
their own, so they need no camera.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from car_vision import stream
from car_vision.stream import FrameSource

SETTINGS = {"source": "webcam", "webcam_index": 0, "width": 64, "height": 48,
            "reconnect_seconds": 0.01, "loop_file": True,
            "file_path": "data/does-not-exist/clip.mp4"}
WAIT = 3.0


class Cameras:
    """Hands out fake captures and counts how many are open at the same moment."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.open_now = 0
        self.most_open = 0
        self.opened = 0

    def open(self) -> "FakeCapture":
        with self.lock:
            self.opened += 1
            self.open_now += 1
            self.most_open = max(self.most_open, self.open_now)
        return FakeCapture(self)

    def closed(self) -> None:
        with self.lock:
            self.open_now -= 1


class FakeCapture:
    def __init__(self, cameras: Cameras) -> None:
        self._cameras = cameras
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def read(self):
        time.sleep(0.002)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self) -> None:
        if self._open:
            self._open = False
            self._cameras.closed()


def wait_until(condition, seconds: float = WAIT) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


def test_the_real_webcam_gets_the_shared_gate_and_other_sources_get_none():
    assert FrameSource(SETTINGS)._gate is stream.WEBCAM_GATE
    assert FrameSource({**SETTINGS, "source": "phone", "phone_url": "http://192.168.0.2:8080/video"})._gate is None
    # A test's own opener means no real camera, so no gate unless one is handed in.
    assert FrameSource(SETTINGS, opener=Cameras().open)._gate is None


def test_a_second_reader_waits_until_the_first_has_let_the_camera_go():
    cameras, gate = Cameras(), threading.Lock()
    first = FrameSource(SETTINGS, opener=cameras.open, gate=gate).start()
    assert first.wait_for_frame(WAIT)
    second = FrameSource(SETTINGS, opener=cameras.open, gate=gate).start()
    time.sleep(0.2)
    assert cameras.opened == 1                  # the second has not touched the camera
    assert second.read()[0] is None

    first.stop()
    assert second.wait_for_frame(WAIT)          # now it is the second one's turn
    assert cameras.opened == 2
    assert cameras.most_open == 1               # never two captures at once
    second.stop()
    assert cameras.open_now == 0
    assert gate.acquire(blocking=False)         # and the gate is free again
    gate.release()


def test_a_reader_stopped_while_it_waits_never_opens_the_camera():
    cameras, gate = Cameras(), threading.Lock()
    first = FrameSource(SETTINGS, opener=cameras.open, gate=gate).start()
    assert first.wait_for_frame(WAIT)
    second = FrameSource(SETTINGS, opener=cameras.open, gate=gate).start()
    started = time.monotonic()
    second.stop()                               # switched off again before its turn
    assert time.monotonic() - started < 1.0     # stop() does not wait for the first reader
    first.stop()
    assert cameras.opened == 1
    assert cameras.most_open == 1


def test_quick_offs_and_ons_never_have_two_captures_open():
    cameras, gate = Cameras(), threading.Lock()
    source = FrameSource(SETTINGS, opener=cameras.open, gate=gate).start()
    assert source.wait_for_frame(WAIT)
    for _ in range(8):                          # as the pipeline does: let go in a thread, start at once
        threading.Thread(target=source.stop, daemon=True).start()
        source = FrameSource(SETTINGS, opener=cameras.open, gate=gate).start()
    assert source.wait_for_frame(WAIT)          # the last one gets the camera
    assert cameras.most_open == 1
    source.stop()
    assert wait_until(lambda: cameras.open_now == 0)


def test_the_gate_is_free_again_when_the_reader_ends_by_an_error():
    gate = threading.Lock()

    def broken():
        raise stream.ConfigError("no such camera")

    source = FrameSource(SETTINGS, opener=broken, gate=gate)
    with pytest.raises(stream.ConfigError):
        source._run()                           # the reader's own code, in this thread
    assert gate.acquire(blocking=False)
    gate.release()
