"""M25: watch for a thing, and get one quiet event each time it appears.

The loop runs with the fakes of tests/test_pipeline.py and an injected clock: the pipeline
module's `time` is replaced by FakeClock, the camera gives a frame only when the test asks
for one, and the test waits until the loop has finished that frame. Nothing sleeps, and
no timing luck is needed. No camera, model, network or graphics card is used.

The owner's two rules are proved here for the pipeline: nothing about a watch is spoken or
made into a reaction, and nothing about it costs money (no model is loaded for an everyday
label, the lookout is never asked).
"""

import threading
import types

import cv2
import numpy as np
import pytest

from car_vision import pipeline as pipeline_module
from car_vision.pipeline import (NO_PEOPLE, Search, Watcher, clean_names, watch_of, watch_settings,
                                 watched_place)
from car_vision.config import ConfigError
from car_vision.state import Detection, RecognizedFace
from test_pipeline import (CONFIG, OFFSET, PERSON, SARA, SARA_FACE, FakeDetector, FakeLookout, FakeOpenDetector,
                           FakeRecognizer, FakeSightings, FakeSource, memory_config, wait_until)
from test_pipeline import make  # noqa: F401  (the fixture, used by name)

WALL_START = 1_795_500_000.0          # the wall clock when the fake monotonic clock reads 0
STEP = 0.25                           # seconds between frames: exact in binary, so 4 steps are 1.0 s
BOTTLE = Detection("bottle", 0.8, (20, 200, 80, 330), 5)          # on the left of 640 x 360
CUP = Detection("cup", 0.7, (560, 200, 620, 300), 6)              # on the right
ROUTER = Detection("wifi router", 0.8, (300, 20, 340, 60), 4)     # a finder find, in the middle
KEYS_IN_HAND = Detection("keys", 0.7, (150, 200, 190, 240), 7)    # inside PERSON (100, 60, 260, 300)


# ---- the clock, the camera and a running pipeline ------------------------------------------


class FakeClock:
    """time.monotonic() and time.time() for the pipeline module; moves only when told."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def time(self):
        return WALL_START + self.now

    def advance(self, seconds):
        self.now += seconds


class StepSource(FakeSource):
    """A camera that gives a new frame only when the test calls step()."""

    def __init__(self, size=(640, 360)):
        super().__init__(frames=0, size=size)

    def step(self):
        with self._lock:
            self.frames += 1


class EnrolledRecognizer(FakeRecognizer):
    """Sara is enrolled; her face is recognised while `faces` holds it."""

    def enrolled_ids(self):
        return {"sara"}


class WatchingSightings(FakeSightings):
    """Shaped like memory.SightingLog, with appeared() too."""

    def __init__(self):
        super().__init__()
        self.appeared_calls = []

    def appeared(self, label, place, now, name=None):
        self.appeared_calls.append((label, place, now, name))


class Run:
    """A running pipeline, stepped one frame at a time on the fake clock."""

    def __init__(self, pipeline, source, detector, clock, done):
        self.pipeline, self.source, self.detector, self.clock, self.done = pipeline, source, detector, clock, done

    def frame(self, after=STEP):
        """Move the clock on by `after`, then let the loop finish one new frame."""
        self.clock.advance(after)
        before = self.done[0]
        self.source.step()
        assert wait_until(lambda: self.done[0] > before)

    def frames(self, seconds, after=STEP):
        for _ in range(round(seconds / after)):
            self.frame(after)

    def show(self, *detections):
        self.detector.detections = list(detections)

    def watch_status(self):
        return self.pipeline.status()["watch"]

    def events(self):
        return self.watch_status()["events"]

    def entry(self, name):
        return next(entry for entry in self.watch_status()["names"] if entry["name"] == name)


@pytest.fixture
def clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(pipeline_module, "time", types.SimpleNamespace(monotonic=clock.monotonic, time=clock.time))
    return clock


@pytest.fixture
def run(make, clock, monkeypatch):
    """Build and start a pipeline whose frames the test steps; the camera starts empty."""

    def build(detections=(), recognizer=None, source_factory=None, config=CONFIG, **kwargs):
        source = StepSource()
        detector = FakeDetector(detections=list(detections))
        pipeline = make(source=source, detector=detector,
                        recognizer=recognizer if recognizer is not None else FakeRecognizer(faces=()),
                        source_factory=source_factory, config=config, **kwargs)
        done = [0]
        real = pipeline._watch_for_things

        def counted(*args, **kw):
            real(*args, **kw)
            done[0] += 1

        monkeypatch.setattr(pipeline, "_watch_for_things", counted)
        pipeline.start()
        return Run(pipeline, source, detector, clock, done)

    return build


def finder_ready(current):
    """The open detector has the names of the newest round (also when only watched names are given)."""
    return lambda: current.pipeline._search.state == "ready"


def watch_config(**changes):
    return {**CONFIG, "watch": {**CONFIG["watch"], **changes}}


def bottle_watched(run, **kwargs):
    """Watching for a bottle, with nothing in view for the first 1.5 seconds."""
    current = run(**kwargs)
    current.pipeline.watch(["bottle"])
    current.frames(1.5)
    return current


# ---- appearing ---------------------------------------------------------------------------------


def test_a_thing_appears_after_one_second_in_view_and_not_before(run, clock):
    current = bottle_watched(run)
    assert current.entry("bottle") == {"name": "bottle", "state": "watching", "in_view": False,
                                       "last_at": None, "person": False}
    current.show(BOTTLE)
    current.frame()
    assert current.entry("bottle")["in_view"] is True and current.events() == []
    current.frames(0.75)
    assert current.events() == []                                   # 0.75 s in view: not yet
    current.frame()
    assert current.events() == [{"id": 1, "name": "bottle", "place": "left", "at": clock.time(), "person": False}]
    assert current.entry("bottle")["last_at"] == clock.time()


def test_a_flicker_gives_one_event(run):
    current = bottle_watched(run)
    current.show(BOTTLE)
    current.frames(1.5)
    assert len(current.events()) == 1
    for _ in range(5):                                              # gone and back, again and again
        current.show()
        current.frames(0.5)
        current.show(BOTTLE)
        current.frames(1.5)
    current.show()
    current.frame(after=3.0)                                        # a real break, but short
    current.show(BOTTLE)
    current.frames(2.0)
    assert len(current.events()) == 1


def test_the_same_thing_is_announced_again_only_after_again_after_seconds_away(run):
    current = bottle_watched(run)
    current.show(BOTTLE)
    current.frames(1.5)
    current.show()
    current.frame(after=59.0)                                       # away 59 s: not long enough
    current.show(BOTTLE)
    current.frames(2.0)
    assert [event["id"] for event in current.events()] == [1]
    current.show()
    current.frame(after=60.0)                                       # away 60 s
    current.show(BOTTLE)
    current.frames(1.0)
    assert len(current.events()) == 1                               # appearing again takes its second
    current.frame()
    assert [event["id"] for event in current.events()] == [1, 2]


def test_a_shorter_again_after_setting_is_used(run):
    current = bottle_watched(run, config=watch_config(again_after_seconds=5))
    current.show(BOTTLE)
    current.frames(1.5)
    current.show()
    current.frame(after=5.0)
    current.show(BOTTLE)
    current.frames(1.5)
    assert [event["id"] for event in current.events()] == [1, 2]


def test_a_thing_already_in_view_when_the_watch_starts_gives_no_event_but_is_in_view(run):
    current = run(detections=[BOTTLE])
    current.frames(1.0)                                             # in view before the watch
    current.pipeline.watch(["bottle"])
    current.frames(3.0)
    assert current.events() == []
    assert current.entry("bottle")["in_view"] is True and current.entry("bottle")["state"] == "watching"
    current.show()
    current.frame(after=60.0)                                       # away long enough, then back
    current.show(BOTTLE)
    current.frames(1.25)
    assert len(current.events()) == 1


def test_a_thing_there_at_the_start_is_announced_once_put_back_after_a_short_break(run):
    current = run(detections=[BOTTLE])
    current.pipeline.watch(["bottle"])
    current.frames(2.0)                                             # on the table from the start
    for away, events in ((2.0, 0), (4.0, 1), (10.0, 1), (61.0, 2)):
        current.show()
        current.frame(after=away)                                   # away this long ...
        current.show(BOTTLE)
        current.frames(0.75)                                        # ... then back, not yet 1 s
        current.frames(1.0)                                         # 1.75 s in view
        assert len(current.events()) == events, away
    current.show()
    current.frame(after=4.0)                                        # after a real event a short break is not enough
    current.show(BOTTLE)
    current.frames(2.0)
    assert len(current.events()) == 2


def test_a_thing_in_the_first_second_of_a_new_picture_is_announced_after_a_short_break(run):
    current = bottle_watched(run)
    current.source.picture = np.full((480, 640, 3), 60, dtype=np.uint8)    # a new frame size
    current.show(BOTTLE)
    current.frames(2.0)
    assert current.events() == []
    current.show()
    current.frame(after=4.0)
    current.show(BOTTLE)
    current.frames(1.5)
    assert len(current.events()) == 1


def test_a_watch_of_a_finder_name_never_makes_a_running_search_lie(run):
    keys = Detection("keys", 0.7, (20, 300, 60, 340), 7)
    open_detector = FakeOpenDetector(finds=[keys, ROUTER])
    current = run(open_detector=open_detector)
    current.pipeline.search(["keys"])
    assert wait_until(finder_ready(current))
    current.frames(0.5)

    def answer():
        search = current.pipeline.status()["search"]
        return search["state"], search["seen"], [entry["name"] for entry in search["found"]], search["missing"]

    assert answer() == ("ready", True, ["keys"], [])
    for names in (["wifi router"], []):                             # start a watch, then stop it
        open_detector.names_gate = threading.Event()               # set_names waits until the test lets it go
        open_detector.inside.clear()
        current.pipeline.watch(names)
        assert wait_until(open_detector.inside.is_set)
        for _ in range(12):                                         # 3 s while the open detector cannot look
            current.frame()
            assert answer() == ("ready", True, ["keys"], []), names
        open_detector.names_gate.set()
        assert wait_until(lambda: open_detector.names == ("keys", *names))
        assert wait_until(finder_ready(current))
        current.frames(0.5)
        assert answer() == ("ready", True, ["keys"], [])


def test_until_the_loop_has_looked_a_new_name_is_getting_ready(run):
    current = run()
    current.frame()
    current.pipeline.watch(["bottle"])
    assert current.entry("bottle")["state"] == "loading"
    current.frame()
    assert current.entry("bottle")["state"] == "watching"


# ---- what answers which name --------------------------------------------------------------------


def test_an_everyday_label_needs_no_finder_and_costs_nothing(run):
    lookout = FakeLookout()
    current = bottle_watched(run, lookout=lookout)
    current.show(BOTTLE)
    current.frames(1.5)
    assert len(current.events()) == 1
    assert current.pipeline._open_detector is None                  # never built, never loaded
    assert current.pipeline.status()["search"]["state"] == "off"
    assert lookout.looks == 0                                       # auto-discover is never asked


def test_a_known_persons_name_is_answered_by_their_face(run):
    recognizer = EnrolledRecognizer(faces=(), people={"sara": SARA})
    open_detector = FakeOpenDetector()
    current = run(recognizer=recognizer, open_detector=open_detector,
                  config={**CONFIG, "recognizer": {**CONFIG["recognizer"], "every_n_frames": 1}})
    current.pipeline.watch(["sara"])
    current.frames(1.5)
    assert current.entry("Sara Ahmed")["person"] is True and current.entry("Sara Ahmed")["in_view"] is False
    current.show(PERSON)
    recognizer.faces = [SARA_FACE]
    current.frames(1.25)
    [event] = current.events()
    assert (event["name"], event["place"], event["person"]) == ("Sara Ahmed", "left", True)
    assert open_detector.calls == []                                # a person is never given to the finder


def test_another_name_goes_to_the_finder_after_the_own_names_and_is_not_found_in_the_search(run):
    open_detector = FakeOpenDetector(finds=[ROUTER])
    current = run(open_detector=open_detector)
    current.pipeline.search(["keys"])
    current.pipeline.watch(["wifi router", "bottle"])
    assert wait_until(finder_ready(current))
    assert open_detector.names == ("keys", "wifi router")           # the own name first; bottle is everyday
    current.frames(1.5)                                             # the finder looks: the router is there already
    assert current.events() == [] and current.entry("wifi router")["in_view"] is True
    status = current.pipeline.status()
    router = next(entry for entry in status["objects"] if entry["label"] == "wifi router")
    assert "found" not in router and "discovered" not in router     # drawn like a discovered one, but not tagged
    assert status["search"]["found"] == [] and status["search"]["missing"] == ["keys"]


def test_finder_names_come_after_the_own_names_and_before_the_discovered_ones():
    search = Search(names=("keys",), discovered=("lamp",), watched=("wifi router",))
    assert search.finder_names == ("keys", "wifi router", "lamp")


def test_a_finder_name_appears_when_found_after_the_watch_began(run):
    open_detector = FakeOpenDetector(finds=[])
    current = run(open_detector=open_detector)
    current.pipeline.watch(["wifi router"])
    assert wait_until(finder_ready(current))
    current.frames(1.5)
    assert current.entry("wifi router")["state"] == "watching"
    open_detector.finds = [ROUTER]
    current.frames(1.5)
    [event] = current.events()
    assert (event["name"], event["place"]) == ("wifi router", "middle")


def test_a_maybe_never_counts(run):
    window = Detection("window", 0.22, (300, 0, 620, 340), 9)       # between the maybe and the sure limit
    open_detector = FakeOpenDetector(finds=[])
    current = run(open_detector=open_detector)
    current.pipeline.search(["window"])                             # an own name: it may get a maybe
    current.pipeline.watch(["window"])
    assert wait_until(finder_ready(current))
    current.frames(1.5)
    open_detector.finds = [window]
    current.frames(2.0)
    assert current.pipeline.status()["search"]["found"][0]["sure"] is False      # shown as a maybe
    assert current.events() == [] and current.entry("window")["in_view"] is False


def test_a_find_on_a_person_counts(run):
    open_detector = FakeOpenDetector(finds=[])
    sightings = WatchingSightings()
    current = run(detections=[PERSON], open_detector=open_detector, sightings=sightings,
                  config=memory_config())
    current.pipeline.watch(["keys"])
    assert wait_until(finder_ready(current))
    current.frames(1.5)
    open_detector.finds = [KEYS_IN_HAND]
    current.frames(1.5)
    [event] = current.events()
    assert (event["name"], event["place"]) == ("keys", "left")
    assert not any(call["label"] == "keys" for call in sightings.seen_calls)    # still no sighting of it
    assert sightings.appeared_calls == [("keys", "left", event["at"], None)]


# ---- a new picture starts afresh ---------------------------------------------------------------


def test_a_camera_switch_gives_no_event_for_what_the_new_camera_shows_at_once(run):
    new_camera = StepSource()
    current = bottle_watched(run, source_factory=lambda settings: new_camera)
    current.pipeline.use_phone("http://192.168.0.9:8080/video")
    current.source = new_camera
    current.show(BOTTLE)
    current.frames(3.0)
    assert current.events() == [] and current.entry("bottle")["in_view"] is True


def test_turning_the_camera_off_and_on_gives_no_event(run):
    cameras = []

    def new_camera(settings):
        cameras.append(StepSource())
        return cameras[-1]

    current = bottle_watched(run, source_factory=new_camera)
    current.pipeline.set_camera(False)
    current.clock.advance(5.0)
    assert current.entry("bottle") == {"name": "bottle", "state": "watching", "in_view": False,
                                       "last_at": None, "person": False}     # the list is kept
    current.pipeline.set_camera(True)
    current.source = cameras[-1]
    current.show(BOTTLE)
    current.frames(3.0)
    assert current.events() == [] and current.entry("bottle")["in_view"] is True


def test_a_new_frame_size_gives_no_event(run):
    current = bottle_watched(run)
    current.source.picture = np.full((480, 640, 3), 60, dtype=np.uint8)
    current.show(BOTTLE)
    current.frames(3.0)
    assert current.events() == []


def test_without_a_new_picture_the_same_moment_would_have_been_an_event(run):
    current = bottle_watched(run)                                   # the control for the three tests above
    current.show(BOTTLE)
    current.frames(3.0)
    assert len(current.events()) == 1


# ---- room in the finder, ids, the memory, and what is never done -----------------------------


def test_a_name_without_room_in_the_finder_waits(run):
    open_detector = FakeOpenDetector(finds=[])
    current = run(open_detector=open_detector)
    current.pipeline.search(["lamp", "mug", "pen", "fan", "box"])
    current.pipeline.watch(["wifi router", "curtain"])              # room for one more of six
    assert wait_until(finder_ready(current))
    current.frames(0.5)
    assert current.entry("wifi router")["state"] == "watching"
    assert current.entry("curtain")["state"] == "waiting"
    assert "curtain" not in open_detector.names and len(open_detector.names) == 6
    current.pipeline.search([])                                     # the search ends: now there is room
    assert wait_until(lambda: "curtain" in open_detector.names)
    assert wait_until(finder_ready(current))
    current.frames(0.5)
    assert current.entry("curtain")["state"] == "watching"


def test_event_ids_rise_by_one_and_only_the_newest_ten_are_kept(run):
    current = bottle_watched(run, config=watch_config(again_after_seconds=0))
    for _ in range(12):
        current.show(BOTTLE)
        current.frames(1.25)
        current.show()
        current.frame(after=2.0)
    events = current.events()
    assert [event["id"] for event in events] == list(range(3, 13))
    assert [event["at"] for event in events] == sorted(event["at"] for event in events)


def test_the_memory_is_told_once_per_event_and_never_given_a_picture(run, clock):
    sightings = WatchingSightings()
    current = bottle_watched(run, sightings=sightings, config=memory_config())
    current.show(BOTTLE)
    current.frames(3.0)
    assert sightings.appeared_calls == [("bottle", "left", current.events()[0]["at"], None)]


def test_a_person_is_told_to_the_memory_by_name(run):
    recognizer = EnrolledRecognizer(faces=(), people={"sara": SARA})
    sightings = WatchingSightings()
    config = {**memory_config(), "recognizer": {**CONFIG["recognizer"], "every_n_frames": 1}}
    current = run(recognizer=recognizer, sightings=sightings, config=config)
    current.pipeline.watch(["Sara"])
    current.frames(1.5)
    current.show(PERSON)
    recognizer.faces = [SARA_FACE]
    current.frames(1.25)
    assert sightings.appeared_calls == [("person", "left", current.events()[0]["at"], "Sara Ahmed")]


def test_with_the_memory_off_the_events_still_come(run):
    current = bottle_watched(run)                                   # CONFIG has no memory file
    assert current.pipeline.sightings() is None
    current.show(BOTTLE)
    current.frames(1.5)
    assert len(current.events()) == 1


def test_nothing_is_spoken_and_no_reaction_is_made(run):
    lookout = FakeLookout()
    current = bottle_watched(run, lookout=lookout)
    current.show(BOTTLE, CUP)
    current.frames(1.5)
    status = current.pipeline.status()
    assert len(status["watch"]["events"]) == 1
    assert status["reactions"] == []                                # the event is no greeting or warning
    assert current.pipeline._on_screen == []
    assert lookout.looks == 0 and current.pipeline._lookout is lookout
    assert current.pipeline.state.memory() == ()                    # the assistant is told nothing new


def test_watch_with_no_names_stops_watching(run):
    open_detector = FakeOpenDetector(finds=[])
    current = run(open_detector=open_detector)
    current.pipeline.watch(["wifi router"])
    assert wait_until(lambda: open_detector.names == ("wifi router",))
    current.pipeline.watch([])
    assert current.watch_status()["names"] == []
    assert wait_until(lambda: open_detector.names == () or current.pipeline._search.finder_names == ())
    assert current.pipeline._search.finder_names == ()


@pytest.mark.parametrize("names, message", [
    (["person"], NO_PEOPLE),
    (["my hand"], NO_PEOPLE),
    (["bottle", "cup", "keys", "remote", "lamp"], "Watch for at most 4 things at once."),
    ("bottle", "Give the names as a list"),
    ([""], "Each name must be 1 to 40 characters long."),
    (["bottle!"], "A name may only have letters"),
])
def test_wrong_names_raise_and_change_nothing(run, names, message):
    current = run()
    current.pipeline.watch(["bottle"])
    with pytest.raises(ValueError) as raised:
        current.pipeline.watch(names)
    assert str(raised.value).startswith(message)
    assert [entry["name"] for entry in current.watch_status()["names"]] == ["bottle"]


# ---- the small pure parts ------------------------------------------------------------------------


def test_the_watcher_on_its_own():
    watcher = Watcher(again_after=60)
    assert watcher.look("cup", False, 0.0) is False                 # the first judged frame
    assert watcher.look("cup", True, 1.0) is False                  # 1.0 s after the start: there already
    assert watcher.judged("cup") and watcher.in_view("cup", 1.5) and not watcher.in_view("cup", 2.5)
    watcher.start_afresh()
    assert not watcher.judged("cup")
    assert watcher.look("cup", False, 10.0) is False
    assert watcher.look("cup", True, 11.5) is False
    assert watcher.look("cup", True, 12.0) is False
    assert watcher.look("cup", True, 12.5) is True                  # 1.0 s in view (a gap under 1 s is no break)
    assert watcher.look("cup", True, 13.0) is False
    watcher.keep(["bottle"])
    assert not watcher.judged("cup")


def test_watch_of_shows_a_person_as_enrolled_and_lists_them_once():
    watch = watch_of(("sara", "bottle", "Sara Ahmed"), {"sara": "Sara Ahmed"})
    assert watch.names == ("sara", "bottle") and watch.shown == ("Sara Ahmed", "bottle")
    assert watch.person("sara").person_ids == ("sara",) and watch.person("bottle") is None
    assert watch.finder_names == ()                                 # a person and an everyday label
    assert watch_of(("keys", "Bottle"), {}).finder_names == ("keys",)


def test_watched_place_judges_a_finder_name_only_once_the_finder_looked():
    watch = watch_of(("keys",), {})
    search = Search(watched=("keys",), state="ready")
    keys = Detection("keys", 0.7, (500, 0, 540, 40), OFFSET + 7)
    assert watched_place("keys", watch, search, set(), [], {}, [keys], [], 640) == (False, None)
    assert watched_place("keys", watch, search, {"keys"}, [], {}, [keys], [], 640) == (True, "right")
    assert watched_place("keys", watch, search, {"keys"}, [], {}, [keys], [keys], 640) == (True, None)   # a maybe
    assert watched_place("keys", watch, Search(), {"keys"}, [], {}, [keys], [], 640) == (False, None)   # no room


def test_the_watch_settings_are_checked():
    assert watch_settings({"max_names": 4, "again_after_seconds": 60}) == (4, 60.0)
    for wrong in ({"max_names": 0, "again_after_seconds": 60}, {"max_names": 7, "again_after_seconds": 60},
                  {"max_names": 4, "again_after_seconds": -1}, {"max_names": 4}):
        with pytest.raises(ConfigError):
            watch_settings(wrong)
    assert clean_names(["a", "b", "c", "d", "e", "f"]) == ("a", "b", "c", "d", "e", "f")   # search keeps 6


# ---- a picture kept by the memory never shows a face (the orchestrator's addition) ----------------


def test_a_picture_for_the_memory_has_the_faces_covered_with_grey(make):
    red = (0, 0, 255)                                               # BGR: the face is painted pure red
    picture = np.full((360, 640, 3), 60, dtype=np.uint8)
    x1, y1, x2, y2 = SARA_FACE.box
    picture[y1:y2, x1:x2] = red
    phone = Detection("cell phone", 0.8, (130, 50, 260, 200), 8)   # held beside the cheek: the face is inside
    sightings = FakeSightings()
    pipeline = make(source=FakeSource(frames=3, picture=picture), detector=FakeDetector([PERSON, phone]),
                    sightings=sightings, config=memory_config(min_seconds=0.0)).start()
    assert wait_until(lambda: any(call["label"] == "cell phone" and call["picture"] for call in sightings.seen_calls))
    jpeg = next(call["picture"] for call in sightings.seen_calls if call["label"] == "cell phone" and call["picture"])
    kept = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).astype(int)
    blue, green, red_channel = kept[..., 0], kept[..., 1], kept[..., 2]
    assert not ((red_channel > 150) & (green < 100) & (blue < 100)).any()      # no red face pixel is left
    grey = (abs(kept - 128) < 12).all(axis=2)
    assert grey.sum() > 0.25 * (x2 - x1) * (y2 - y1)                # the face's place is plain grey
