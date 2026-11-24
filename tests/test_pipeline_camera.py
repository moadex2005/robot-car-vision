"""M19 Part A: turning the camera off and on (Pipeline.set_camera), run with the fakes of test_pipeline.py.

No camera, model, network or graphics card is needed. Nothing here sleeps: every wait is on
a condition with a timeout. "Nothing happens while the camera is off" is checked over many
turns of the loop, which the loop counts itself by ticking the memory on every turn.
"""

import json
import threading

import numpy as np
import pytest

from car_vision import pipeline as pipeline_module
from car_vision.config import ConfigError
from car_vision.pipeline import CAMERA_OFF, HINT_CAMERA_OFF, CameraOff, settings_to_turn_on
from test_pipeline import (CONFIG, LINA_ID, PHONE, FakeDetector, FakeLookout, FakeOpenDetector, FakeRecognizer,
                           FakeSightings, FakeSource, NoPhone, SlowToStopSource, decoded, discover_config,
                           people_config, phone_source, progress_is, ready, search_of, wait_until)
from test_pipeline import book, make, quick  # noqa: F401  (the fixtures, used by name)

WEBCAM_CONFIG = {**CONFIG, "stream": {**CONFIG["stream"], "source": "webcam"}}


# ---- fakes and helpers ------------------------------------------------------------------------


class Factory:
    """Stands in for FrameSource: makes a new fake for every call (a phone for phone settings)
    and keeps the settings and every source it made, which record whether they were started
    and stopped."""

    def __init__(self, frames=10_000, connected=True):
        self.frames = frames
        self.connected = connected
        self.settings = []
        self.made = []
        self._lock = threading.Lock()

    def __call__(self, settings):
        if settings["source"] == "phone":
            source = phone_source(frames=self.frames, connected=self.connected)
        else:
            source = FakeSource(frames=self.frames)
            source.source = settings["source"]
            source.connected = self.connected
        with self._lock:
            self.settings.append(settings)
            self.made.append(source)
        return source


def running(make, source=None, factory=None, **kwargs):
    """A started pipeline that has shown its first picture, with a factory and a memory that counts turns."""
    factory = factory if factory is not None else Factory()
    sightings = kwargs.pop("sightings", None) or FakeSightings()
    pipeline = make(source=source if source is not None else FakeSource(frames=10_000),
                    source_factory=factory, sightings=sightings, **kwargs).start()
    assert wait_until(lambda: pipeline.latest_jpeg() is not None)
    return pipeline, factory, sightings


def turns(sightings, count=20):
    """Wait until the loop has gone round `count` more times (it ticks the memory on every turn)."""
    start = getattr(sightings, "ticks", 0)
    assert wait_until(lambda: getattr(sightings, "ticks", 0) >= start + count, timeout=5.0)


def live(source):
    return source.started and not source.stopped


def camera_of(pipeline):
    return pipeline.status()["camera"]


def dark_below_the_line(pipeline):
    picture = pipeline.latest_jpeg()
    return picture is not None and np.abs(decoded(picture)[60:] - pipeline_module.DARK_LEVEL).max() <= 4


# ---- off and on -----------------------------------------------------------------------------------


def test_the_camera_is_on_at_every_start(make):
    pipeline = make(source=FakeSource(frames=3))
    assert camera_of(pipeline) == {"on": True}
    pipeline.start()
    assert wait_until(lambda: pipeline.latest_jpeg() is not None)
    assert camera_of(pipeline) == {"on": True}


def test_off_lets_go_of_the_camera_and_forgets_everything_it_showed(make):
    old = FakeSource(frames=10_000)
    pipeline, factory, _sightings = running(make, source=old)
    assert wait_until(lambda: pipeline.status()["objects"] and pipeline.state.current_people() == ["Sara Ahmed"])
    last_frame = pipeline.latest_jpeg()

    pipeline.set_camera(False)
    assert wait_until(lambda: old.stopped)                  # let go in a thread of its own
    status = pipeline.status()
    assert status["camera"] == {"on": False}
    assert (status["source"], status["connected"], status["ended"], status["error"]) == ("file", False, False, None)
    assert status["fps"] == 0.0
    assert status["objects"] == status["people"] == status["reactions"] == []
    json.dumps(status)
    snapshot = pipeline.state.snapshot()
    assert snapshot.detections == () and snapshot.faces == ()
    assert pipeline.state.current_people() == []
    assert pipeline.target(track_id=2) is None and pipeline.target(x=0.5, y=0.5) is None
    assert pipeline.scene_photo() is None
    assert factory.made == []                               # nothing new is made to be off

    # The page gets one still picture: plain dark, the size of the last one, with the line on it.
    assert wait_until(lambda: dark_below_the_line(pipeline))
    assert pipeline._lost_line == CAMERA_OFF
    assert pipeline.latest_jpeg() is not last_frame
    picture = decoded(pipeline.latest_jpeg())
    assert picture.shape == decoded(last_frame).shape
    assert np.abs(picture[:40] - pipeline_module.DARK_LEVEL).max() > 100      # the line is written on it
    still = pipeline.latest_jpeg()
    turns(pipeline._sightings, 10)
    assert pipeline.latest_jpeg() is still                  # drawn once


def test_off_does_not_wait_for_the_camera_to_let_go(make):
    old = SlowToStopSource()
    pipeline, _factory, _sightings = running(make, source=old)
    try:
        pipeline.set_camera(False)                          # returns although stop() hangs
        assert old.stopping.wait(2)
        assert camera_of(pipeline) == {"on": False}         # and the lock is free meanwhile
        pipeline.latest_jpeg()
        pipeline.set_camera(True)
        assert camera_of(pipeline) == {"on": True}
        assert not old.stopped
    finally:
        old.may_stop.set()
    assert wait_until(lambda: old.stopped)


def test_no_model_runs_while_off_however_long_but_the_memory_still_ticks(make):
    detector, recognizer = FakeDetector(), FakeRecognizer()
    pipeline, _factory, sightings = running(make, detector=detector, recognizer=recognizer)
    assert wait_until(lambda: detector.calls > 3 and recognizer.calls > 0)
    pipeline.set_camera(False)
    turns(sightings, 2)                                     # a frame that was in the models has finished
    detects, checks = detector.calls, recognizer.calls
    turns(sightings, 40)                                    # the memory's tick() still runs on every turn
    assert (detector.calls, recognizer.calls) == (detects, checks)


def test_on_makes_and_starts_a_new_source_of_the_same_kind_and_frames_flow_again(make):
    detector = FakeDetector()
    pipeline, factory, _sightings = running(make, detector=detector)
    pipeline.set_camera(False)
    pipeline.set_camera(True)
    assert factory.settings == [CONFIG["stream"]]           # the source the program started with
    new = factory.made[0]
    assert new.source == "file" and live(new)
    assert camera_of(pipeline) == {"on": True}
    calls = detector.calls
    assert wait_until(lambda: detector.calls > calls + 3)
    assert wait_until(lambda: pipeline.target(track_id=2) is not None)
    assert pipeline.status()["connected"] is True
    assert wait_until(lambda: pipeline.status()["objects"])   # Windows' clock moves in steps of about 15 ms


def test_on_shows_the_connecting_line_until_the_first_frame(make):
    webcam = FakeSource(frames=3)
    webcam.source = "webcam"
    pipeline, factory, _sightings = running(make, source=webcam, factory=Factory(frames=0, connected=False),
                                            config=WEBCAM_CONFIG)
    pipeline.set_camera(False)
    assert wait_until(lambda: pipeline._lost_line == CAMERA_OFF)
    pipeline.set_camera(True)
    assert factory.settings == [WEBCAM_CONFIG["stream"]]
    assert wait_until(lambda: pipeline._lost_line == "Connecting to the webcam ...")
    assert wait_until(lambda: dark_below_the_line(pipeline))
    status = pipeline.status()
    assert (status["camera"], status["source"], status["connected"]) == ({"on": True}, "webcam", False)


def test_off_then_on_for_a_phone_keeps_its_address_and_asks_the_phone_again(make):
    fetch = NoPhone()
    pipeline, factory, sightings = running(make, phone_fetch=fetch)
    pipeline.use_phone(PHONE)
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (320, 240))
    phone = factory.made[0]
    pipeline.set_camera(False)
    assert wait_until(lambda: phone.stopped)
    status = pipeline.status()
    assert (status["source"], status["phone_url"], status["phone_picture"]) == ("phone", PHONE, None)

    # While off, a picture choice is only stored: the phone is not asked.
    asked = len(fetch.calls)
    pipeline.set_phone_picture("keep")
    assert not any(thread.name == "phone-picture" for thread in threading.enumerate())
    assert len(fetch.calls) == asked

    pipeline.set_camera(True)
    assert factory.settings[-1] == {**CONFIG["stream"], "source": "phone", "phone_url": PHONE}
    again = factory.made[-1]
    assert again is not phone and again.source == "phone" and live(again)
    assert wait_until(lambda: len(fetch.calls) > asked)     # the picture size is checked again
    assert fetch.calls[-1].startswith("http://192.168.1.7:8080")
    assert pipeline._picture_choice == "keep"
    assert wait_until(lambda: pipeline.status()["connected"] is True)


def test_the_value_it_already_has_does_nothing(make):
    old = FakeSource(frames=10_000)
    pipeline, factory, _sightings = running(make, source=old)
    generation = pipeline._generation
    pipeline.set_camera(True)
    assert factory.made == [] and pipeline._generation == generation and not old.stopped
    pipeline.set_camera(False)
    pipeline.set_camera(False)
    assert pipeline._generation == generation + 1
    assert factory.made == []
    pipeline.set_camera(True)
    pipeline.set_camera(True)
    assert len(factory.made) == 1 and pipeline._generation == generation + 2


@pytest.mark.parametrize("value", [1, 0, "true", "off", None, [True], {"on": True}])
def test_anything_but_true_or_false_is_refused(make, value):
    pipeline = make(source=FakeSource(frames=3))
    with pytest.raises(ValueError, match="The camera is switched with true or false."):
        pipeline.set_camera(value)
    assert camera_of(pipeline) == {"on": True}


def test_use_phone_while_off_turns_the_camera_on_with_that_phone(make):
    pipeline, factory, _sightings = running(make)
    pipeline.set_camera(False)
    pipeline.use_phone(PHONE)
    status = pipeline.status()
    assert (status["camera"], status["source"], status["phone_url"]) == ({"on": True}, "phone", PHONE)
    assert factory.settings == [{**CONFIG["stream"], "source": "phone", "phone_url": PHONE}]
    assert live(factory.made[0])
    assert wait_until(lambda: pipeline.state.snapshot().frame_size == (320, 240))


def test_off_cancels_a_photo_session_with_its_own_hint(make, tmp_path, book, quick):
    pipeline, _factory, _sightings = running(make, recognizer=FakeRecognizer(), config=people_config(tmp_path))
    pipeline.start_photos(LINA_ID, 3)
    assert wait_until(progress_is(pipeline, state="taking"))
    pipeline.set_camera(False)
    progress = pipeline.photo_progress()
    assert (progress["state"], progress["hint"]) == ("failed", HINT_CAMERA_OFF)


def test_auto_discover_makes_no_look_while_off_and_looks_again_after_on(make):
    lookout = FakeLookout(["router"])
    open_detector = FakeOpenDetector()
    pipeline, _factory, sightings = running(make, detector=FakeDetector(delay=0.005), open_detector=open_detector,
                                            lookout=lookout, config=discover_config(0.02))
    pipeline.discover(True)
    assert wait_until(lambda: lookout.looks >= 2)
    pipeline.set_camera(False)
    turns(sightings, 10)                                    # a look that was under way has finished
    looks, spent = lookout.looks, pipeline.status()["discover"]["spent_usd"]
    turns(sightings, 40)
    discover = pipeline.status()["discover"]
    assert lookout.looks == looks and discover["spent_usd"] == spent
    assert discover["on"] is True                           # it stays switched on
    pipeline.set_camera(True)
    assert wait_until(lambda: lookout.looks > looks)


def test_search_names_are_kept_and_the_open_detector_rests_while_off(make):
    open_detector = FakeOpenDetector()
    pipeline, _factory, sightings = running(make, open_detector=open_detector)
    pipeline.search(["keys"])
    assert wait_until(ready(pipeline))
    assert wait_until(lambda: open_detector.count("detect") > 0)
    pipeline.set_camera(False)
    turns(sightings, 2)
    detects = open_detector.count("detect")
    turns(sightings, 30)
    assert open_detector.count("detect") == detects
    assert search_of(pipeline)["names"] == ["keys"]
    pipeline.set_camera(True)
    assert wait_until(lambda: open_detector.count("detect") > detects)


def test_a_search_asked_for_while_off_waits_for_the_camera(make):
    open_detector = FakeOpenDetector()
    pipeline, _factory, sightings = running(make, open_detector=open_detector)
    pipeline.set_camera(False)
    pipeline.search(["keys"])
    turns(sightings, 20)
    assert open_detector.calls == []                        # not loaded, no names set, no warm-up
    assert search_of(pipeline)["state"] == "loading"
    pipeline.set_camera(True)
    assert wait_until(ready(pipeline))
    assert ("set_names", ("keys",)) in open_detector.calls


def test_stop_while_off_returns_and_starts_nothing(make):
    old = FakeSource(frames=10_000)
    pipeline, factory, _sightings = running(make, source=old)
    pipeline.set_camera(False)
    pipeline.stop()
    assert wait_until(lambda: old.stopped)
    assert factory.made == []
    assert camera_of(pipeline) == {"on": False}
    pipeline.set_camera(True)                               # after stop(): made, never started
    assert all(not source.started for source in factory.made)


def test_off_before_start_starts_no_camera_until_it_is_on(make):
    old = FakeSource(frames=3)
    factory = Factory()
    pipeline = make(source=old, source_factory=factory)
    pipeline.set_camera(False)
    pipeline.start()
    assert not old.started and wait_until(lambda: old.stopped)
    assert camera_of(pipeline) == {"on": False} and pipeline.latest_jpeg() is None
    pipeline.set_camera(True)
    assert live(factory.made[0])
    assert wait_until(lambda: pipeline.latest_jpeg() is not None)


def test_before_any_picture_there_is_nothing_to_darken(make):
    nothing_yet = FakeSource(frames=0)
    nothing_yet.connected = False
    pipeline = make(source=nothing_yet, source_factory=Factory()).start()
    pipeline.set_camera(False)
    assert wait_until(lambda: pipeline._lost_line == CAMERA_OFF)
    assert pipeline.latest_jpeg() is None                   # the page shows its own text


def test_the_loop_starts_again_when_it_had_stopped_after_ten_failures(make):
    detector = FakeDetector(fail_on=set(range(1, 11)))      # the first ten frames fail, then all is well
    pipeline = make(source=FakeSource(frames=50), detector=detector, source_factory=Factory()).start()
    assert wait_until(lambda: pipeline.status()["error"] is not None)
    pipeline.set_camera(False)
    assert pipeline.status()["error"] is None
    assert wait_until(lambda: pipeline._lost_line == CAMERA_OFF)     # the loop runs again
    pipeline.set_camera(True)
    assert wait_until(lambda: pipeline.target(track_id=2) is not None)
    assert pipeline.status()["connected"] is True


def stopped_after_failures(make, **kwargs):
    """A pipeline whose loop drew two good frames, then stopped after ten failures in a row."""
    detector = FakeDetector(fail_on=set(range(3, 13)))
    pipeline = make(source=FakeSource(frames=50), detector=detector, **kwargs).start()
    assert wait_until(lambda: pipeline.status()["error"] is not None)
    assert pipeline._last_drawn is not None                 # the last good frame, boxes and names drawn on it
    return pipeline


def test_off_after_the_loop_stopped_shows_the_plain_dark_picture_not_the_last_frame(make):
    pipeline = stopped_after_failures(make, source_factory=Factory())
    pipeline.set_camera(False)
    assert wait_until(lambda: pipeline._lost_line == CAMERA_OFF)
    assert wait_until(lambda: dark_below_the_line(pipeline))   # nothing of the old camera's picture is left


def test_use_phone_after_the_loop_stopped_shows_the_plain_dark_picture_too(make):
    pipeline = stopped_after_failures(make, source_factory=Factory(frames=0, connected=False))
    pipeline.use_phone(PHONE)
    assert wait_until(lambda: pipeline._lost_line == "Connecting to the phone ...")
    assert wait_until(lambda: dark_below_the_line(pipeline))


class HoldingDetector(FakeDetector):
    """Holds its second frame inside detect() until the test lets it go."""

    def __init__(self):
        super().__init__()
        self.holding = threading.Event()
        self.go_on = threading.Event()

    def detect(self, frame):
        if self.calls == 1:
            self.holding.set()
            assert self.go_on.wait(5)
        return super().detect(frame)


def test_a_frame_in_the_models_when_the_camera_goes_off_runs_no_more_models_and_leaves_nothing(make):
    every_frame = {**CONFIG, "recognizer": {**CONFIG["recognizer"], "every_n_frames": 1},
                   "detector": {**CONFIG["detector"], "open_every_n_frames": 1}}
    detector, recognizer, open_detector = HoldingDetector(), FakeRecognizer(), FakeOpenDetector()
    sightings = FakeSightings()
    pipeline = make(source=FakeSource(frames=2), detector=detector, recognizer=recognizer,
                    open_detector=open_detector, sightings=sightings, source_factory=Factory(), config=every_frame)
    pipeline.search(["keys"])
    assert wait_until(ready(pipeline))
    pipeline.start()
    assert detector.holding.wait(3)                         # frame 2 is inside detect()
    checks, finds = recognizer.calls, open_detector.count("detect")
    assert (checks, finds) == (1, 1)                        # both ran on frame 1
    pipeline.set_camera(False)
    detector.go_on.set()
    assert wait_until(lambda: detector.calls == 2)
    turns(sightings, 5)                                     # the loop has gone on after frame 2
    assert (recognizer.calls, open_detector.count("detect")) == (checks, finds)
    assert pipeline._view is None and pipeline.target(x=0.5, y=0.5) is None
    assert pipeline.state.snapshot().detections == () and pipeline.state.snapshot().faces == ()
    status = pipeline.status()
    assert status["objects"] == status["people"] == [] and status["search"]["found"] == []
    assert wait_until(lambda: dark_below_the_line(pipeline))
    assert pipeline._lost_line == CAMERA_OFF


def test_no_photos_while_off(make, tmp_path, book):
    pipeline = make(source=FakeSource(frames=3), config=people_config(tmp_path), source_factory=Factory())
    pipeline.set_camera(False)
    with pytest.raises(ValueError, match="Turn the camera on first to take photos."):
        pipeline.start_photos(LINA_ID, 3)
    assert pipeline.photo_progress() is None
    pipeline.set_camera(True)
    pipeline.start_photos(LINA_ID, 3)                       # on again: allowed
    assert pipeline.photo_progress()["state"] == "taking"


def test_while_off_auto_and_keep_are_stored_but_a_size_is_refused(make):
    pipeline, _factory, _sightings = running(make)
    pipeline.use_phone(PHONE)
    pipeline.set_camera(False)
    with pytest.raises(ValueError, match="Turn the camera on first to choose a picture size."):
        pipeline.set_phone_picture("640x360")
    pipeline.set_phone_picture("keep")
    assert pipeline._picture_choice == "keep"
    pipeline.set_phone_picture("auto")
    assert pipeline._picture_choice == "auto"


def test_a_camera_that_cannot_be_made_leaves_it_off_and_consistent(make):
    class Broken:
        def __call__(self, settings):
            raise ConfigError("stream.file_path: the clip does not exist")

    old = FakeSource(frames=10_000)
    pipeline, _factory, _sightings = running(make, source=old, factory=Broken())
    generation = pipeline._generation
    pipeline.set_camera(False)
    with pytest.raises(ConfigError):
        pipeline.set_camera(True)
    status = pipeline.status()
    assert (status["camera"], status["connected"], status["error"]) == ({"on": False}, False, None)
    assert isinstance(pipeline._source, CameraOff) and pipeline._generation == generation + 1
    assert wait_until(lambda: old.stopped)
    assert not pipeline._switch_lock.locked()               # nothing is left held
    pipeline._make_source = Factory()                       # the clip is back
    pipeline.set_camera(True)
    assert wait_until(lambda: pipeline.status()["connected"] is True)


def test_many_switches_at_once_end_in_the_state_of_the_last_one_with_one_live_source(make, monkeypatch):
    old = FakeSource(frames=10_000)
    pipeline, factory, _sightings = running(make, source=old)
    changes = []                                            # the value of every switch that changed something
    forget = pipeline._forget_old_picture
    monkeypatch.setattr(pipeline, "_forget_old_picture", lambda: changes.append(pipeline._camera_on) or forget())
    values = [[True, False, "phone"][index % 3] for index in range(15)]   # on, off and a Connect, mixed
    start_together = threading.Barrier(len(values))

    def switch(value):
        start_together.wait(5)
        if value == "phone":
            pipeline.use_phone(PHONE)
        else:
            pipeline.set_camera(value)

    threads = [threading.Thread(target=switch, args=(value,)) for value in values]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    on = camera_of(pipeline)["on"]
    assert changes and changes[-1] == on                    # the last switch that changed it decides
    sources = [old, *factory.made]
    if on:
        assert wait_until(lambda: [live(source) for source in sources].count(True) == 1)
        assert pipeline._source is factory.made[-1] and live(factory.made[-1])
    else:
        assert isinstance(pipeline._source, CameraOff)
    assert wait_until(lambda: all(source.stopped for source in sources if source is not pipeline._source))
    pipeline.set_camera(True)                               # and it still works afterwards
    assert wait_until(lambda: pipeline.status()["connected"] is True)


@pytest.mark.parametrize("attempt", range(5))
def test_stop_racing_turning_on_leaves_no_live_source(make, attempt):
    old = FakeSource(frames=10_000)
    pipeline, factory, _sightings = running(make, source=old)
    pipeline.set_camera(False)
    start_together = threading.Barrier(2)

    def at_once(call):
        start_together.wait(5)
        call()

    threads = [threading.Thread(target=at_once, args=(call,))
               for call in (pipeline.stop, lambda: pipeline.set_camera(True))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert wait_until(lambda: not any(live(source) for source in [old, *factory.made]))


# ---- the pieces ---------------------------------------------------------------------------------


def test_the_stand_in_gives_no_frames_and_does_nothing():
    off = CameraOff("webcam")
    assert (off.source, off.connected, off.ended) == ("webcam", False, False)
    assert off.read() == (None, 0)
    assert off.start() is off
    off.stop()
    off.set_box(640, 480)
    off.reopen()


def test_the_settings_to_turn_on_again():
    stream = CONFIG["stream"]
    assert settings_to_turn_on(stream, "phone", PHONE) == {**stream, "source": "phone", "phone_url": PHONE}
    assert settings_to_turn_on(stream, "file", PHONE) == stream
    assert settings_to_turn_on(stream, "webcam", None) == stream
    assert settings_to_turn_on(stream, "phone", None) == stream
    assert settings_to_turn_on(stream, "file", None) is not stream     # a copy: the settings are never changed
