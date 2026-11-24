"""M20 Part B: "maybe" for the visitor's own names.

The open detector gives everything at or above detector.open_maybe_confidence; the
pipeline sorts: at or above open_confidence is found, else one "maybe" per own name.
A track that was sure stays sure while it goes on. Discovered names never get a maybe,
and a maybe is never told to the memory.

Everything is a fake: no camera, model, network or graphics card is needed, and no test
sleeps: they wait for frames.
"""

import cv2
import numpy as np
import pytest

from car_vision import draw
from car_vision import pipeline as pipeline_module
from car_vision.config import ConfigError, load_config, section
from car_vision.detector import OpenDetector, open_limits
from car_vision.pipeline import NamedFinds, Pipeline, Search, merge_named, search_status, sure_or_maybe
from car_vision.state import Detection
from test_pipeline import (CONFIG, OFFSET, FakeDetector, FakeLookout, FakeOpenDetector, FakeRecognizer,
                           FakeSightings, FakeSource, RecordingState, discover_config, memory_config, search_of,
                           wait_until)
from test_pipeline import make  # noqa: F401  (the fixture, used by name)

SURE = 0.35                                    # open_confidence in config.yaml


def frames_later(detector, count=2):
    """Wait until `count` more frames have started: the frames before them are then done."""
    start = detector.calls
    assert wait_until(lambda: detector.calls >= start + count)


def every_frame(pipeline, monkeypatch, do):
    """Call do() once on every frame, right after the frame was published (and told to the memory)."""
    real = pipeline._remember

    def remember(*args, **kwargs):
        real(*args, **kwargs)
        do()

    monkeypatch.setattr(pipeline, "_remember", remember)


class ScriptedOpenDetector(FakeOpenDetector):
    """Gives script[i] on its i-th detect, then the last one again and again."""

    def __init__(self, *script):
        super().__init__(finds=())
        self.script = [list(step) for step in script]
        self.detects = 0

    def detect(self, frame):
        super().detect(frame)                   # notes the call
        step = self.script[min(self.detects, len(self.script) - 1)]
        self.detects += 1
        return [find for find in step if find.label in self._names]


# ---- the settings check -----------------------------------------------------------------------


def detector_settings(**change):
    return {**CONFIG["detector"], **change}


@pytest.mark.parametrize("maybe", [0, -0.1, 0.26, 2])
def test_the_maybe_limit_must_be_above_0_and_at_most_open_confidence(maybe):
    with pytest.raises(ConfigError, match="open_maybe_confidence"):
        open_limits(detector_settings(open_maybe_confidence=maybe))      # open_confidence is 0.25 here
    with pytest.raises(ConfigError, match="open_maybe_confidence"):
        OpenDetector(detector_settings(open_maybe_confidence=maybe))
    with pytest.raises(ConfigError, match="open_maybe_confidence"):
        Pipeline({**CONFIG, "detector": detector_settings(open_maybe_confidence=maybe)},
                 source=FakeSource(), detector=FakeDetector(), recognizer=FakeRecognizer())


def test_a_missing_maybe_limit_is_reported():
    settings = detector_settings()
    del settings["open_maybe_confidence"]
    with pytest.raises(ConfigError, match="open_maybe_confidence"):
        OpenDetector(settings)


def test_the_limits_as_read():
    assert open_limits(detector_settings(open_maybe_confidence=0.25)) == (0.25, 0.25)   # equal is fine
    detector = OpenDetector(detector_settings())
    assert (detector.confidence, detector.maybe_confidence) == (0.25, 0.2)
    assert open_limits(section(load_config(), "detector")) == (0.35, 0.2)                # config.yaml


# ---- found against maybe against missing ----------------------------------------------------------


def find(label, confidence, track, box=(10, 10, 50, 50)):
    return Detection(label, confidence, box, track)


def merged(fresh, own=("window", "curtain", "lamp"), names=None, sure_tracks=()):
    names = own if names is None else names
    shown, _kept = merge_named([], NamedFinds(1), fresh, 1, 10.0, names, own=own, sure=SURE, sure_tracks=sure_tracks)
    return shown


def test_found_maybe_and_missing():
    shown = merged([find("window", 0.26, 1), find("curtain", 0.70, 2, (400, 0, 600, 300))])
    assert [(d.label, d.confidence) for d in shown] == [("window", 0.26), ("curtain", 0.70)]
    search = Search(("window", "curtain", "lamp"), "ready", "", 1)
    status = search_status(search, shown, 640, True, maybes=[shown[0]])
    assert status["found"] == [{"name": "window", "count": 1, "place": "left", "sure": False},
                               {"name": "curtain", "count": 1, "place": "right", "sure": True}]
    assert status["missing"] == ["lamp"]                     # a maybe is found, not missing


def test_one_maybe_per_name_the_most_confident():
    shown = merged([find("window", 0.21, 1), find("window", 0.30, 2), find("window", 0.26, 3)])
    assert [(d.track_id, d.confidence) for d in shown] == [(OFFSET + 2, 0.30)]


def test_no_maybe_when_the_name_has_a_sure_find():
    shown = merged([find("window", 0.30, 1), find("window", 0.40, 2), find("window", 0.36, 3)])
    assert sorted(d.confidence for d in shown) == [0.36, 0.40]   # every sure one, no maybe


def test_discovered_names_never_get_a_maybe():
    fresh = [find("window", 0.26, 1), find("lamp", 0.30, 2), find("rug", 0.50, 3)]
    shown = merged(fresh, own=("window",), names=("window", "lamp", "rug"))
    assert [d.label for d in shown] == ["window", "rug"]          # the discovered lamp below SURE is dropped


def test_a_track_that_was_sure_stays_sure_on_a_dip():
    fresh = [find("window", 0.26, 1), find("window", 0.30, 2), find("lamp", 0.22, 3)]
    shown = merged(fresh, own=("window",), names=("window", "lamp"), sure_tracks={OFFSET + 1, OFFSET + 3})
    # Track 1 dipped: still sure, so track 2 is no maybe and is dropped; the discovered lamp stays.
    assert [(d.label, d.track_id) for d in shown] == [("window", OFFSET + 1), ("lamp", OFFSET + 3)]


def test_a_maybe_held_on_screen_gives_way_to_a_sure_find():
    finds = ((find("window", 0.26, OFFSET + 1), 9.5), (find("window", 0.5, OFFSET + 2), 10.0))
    assert [item[0].confidence for item in sure_or_maybe(finds, ["window"], SURE)] == [0.5]


def test_sure_tracks_now_keeps_the_sure_ones_that_go_on():
    finds = [find("window", 0.22, 7), find("lamp", 0.5, 8), find("rug", 0.22, 9), find("cup", 0.9, None)]
    assert pipeline_module.sure_tracks_now(finds, {7, 4}, 0.25) == {7, 8}      # 4 ended, 9 never was sure


# ---- in the running pipeline -------------------------------------------------------------------

WINDOW_MAYBE = Detection("window", 0.22, (20, 20, 90, 90), 30)      # CONFIG: sure 0.25, maybe 0.2


def maybe_pipeline(make):
    sightings = FakeSightings()
    detector = FakeDetector()
    open_detector = FakeOpenDetector(finds=[WINDOW_MAYBE])
    pipeline = make(source=FakeSource(frames=10_000), detector=detector, open_detector=open_detector,
                    sightings=sightings, config=memory_config(), state=RecordingState())
    pipeline.search(["window", "lamp"])
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    pipeline.start()
    assert wait_until(lambda: search_of(pipeline)["found"])
    return pipeline, sightings, detector


def test_a_maybe_is_shown_listed_and_marked_in_the_status(make):
    pipeline, _sightings, _detector = maybe_pipeline(make)
    search = search_of(pipeline)
    assert search["found"] == [{"name": "window", "count": 1, "place": "left", "sure": False}]
    assert search["missing"] == ["lamp"]
    window = next(o for o in pipeline.status()["objects"] if o["label"] == "window")
    assert (window["found"], window["maybe"], window["track_id"]) == (True, True, OFFSET + 30)
    assert pipeline.target(track_id=OFFSET + 30).kind == "object"     # it can be clicked like any find


def test_a_maybe_is_never_told_to_the_memory(make):
    pipeline, sightings, detector = maybe_pipeline(make)
    assert wait_until(lambda: any(call["label"] == "chair" for call in sightings.seen_calls))
    frames_later(detector)
    assert not any(call["label"] == "window" for call in sightings.seen_calls)


def test_a_maybe_is_drawn_unsure_and_a_sure_find_wanted(make, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline_module.draw, "draw_detections",
                        lambda frame, detections, wanted=(), unsure=():
                        calls.append((sorted(wanted), [d.track_id for d in unsure])))
    sure_keys = Detection("keys", 0.7, (500, 300, 540, 340), 31)
    open_detector = FakeOpenDetector(finds=[WINDOW_MAYBE, sure_keys])
    pipeline = make(source=FakeSource(frames=10_000), open_detector=open_detector)
    pipeline.search(["window", "keys"])
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    pipeline.start()
    assert wait_until(lambda: ([OFFSET + 31], [OFFSET + 30]) in calls)


def test_a_maybe_without_a_track_number_is_still_drawn_as_a_maybe(make, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline_module.draw, "draw_detections",
                        lambda frame, detections, wanted=(), unsure=():
                        calls.append([(d.label, d.track_id) for d in unsure]))
    untracked = Detection("window", 0.22, (20, 20, 90, 90), None)
    pipeline = make(source=FakeSource(frames=10_000), open_detector=FakeOpenDetector(finds=[untracked]))
    pipeline.search(["window"])
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    pipeline.start()
    assert wait_until(lambda: [("window", None)] in calls)


def test_a_discovered_find_below_the_sure_limit_is_dropped(make):
    lamp = Detection("lamp", 0.22, (500, 20, 560, 80), 32)
    rug = Detection("rug", 0.5, (300, 300, 400, 350), 33)
    pipeline, open_detector = discovering(make, FakeLookout(["lamp", "rug"]), finds=[lamp, rug])
    assert wait_until(lambda: any(o["label"] == "rug" for o in pipeline.status()["objects"]))
    assert wait_until(lambda: open_detector.count("detect") >= 3)
    labels = {o["label"]: o for o in pipeline.status()["objects"]}
    assert "lamp" not in labels and labels["rug"]["discovered"] is True and "maybe" not in labels["rug"]


def discovering(make, lookout, finds):
    open_detector = FakeOpenDetector(finds=finds)
    pipeline = make(source=FakeSource(frames=10_000), detector=FakeDetector(), open_detector=open_detector,
                    lookout=lookout, config=discover_config(0.02)).start()
    pipeline.discover(True)
    return pipeline, open_detector


def test_a_sure_find_that_dips_stays_sure_and_the_chip_does_not_flip(make, monkeypatch):
    sure, dipped = Detection("window", 0.5, (20, 20, 90, 90), 40), Detection("window", 0.22, (20, 20, 90, 90), 40)
    open_detector = ScriptedOpenDetector([sure], [dipped])
    detector = FakeDetector()
    pipeline = make(source=FakeSource(frames=10_000), detector=detector, open_detector=open_detector)
    pipeline.search(["window"])
    assert wait_until(lambda: search_of(pipeline)["state"] == "ready")
    seen = []
    every_frame(pipeline, monkeypatch, lambda: seen.append((search_of(pipeline)["found"],
                                                             [o for o in pipeline.status()["objects"]
                                                              if o["label"] == "window"])))
    pipeline.start()
    assert wait_until(lambda: open_detector.detects >= 6)
    frames_later(detector)
    shown = [(found, objects) for found, objects in seen if found]
    assert shown and all(found[0]["sure"] is True for found, _objects in shown)
    assert all("maybe" not in entry for _found, objects in shown for entry in objects)
    assert all(objects and objects[0]["confidence"] in (0.5, 0.22) for _found, objects in shown)


def test_a_discovered_find_that_dips_is_not_lost(make, monkeypatch):
    sure, dipped = Detection("rug", 0.5, (300, 300, 400, 350), 41), Detection("rug", 0.22, (300, 300, 400, 350), 41)
    open_detector = ScriptedOpenDetector([sure], [dipped])
    detector = FakeDetector()
    pipeline = make(source=FakeSource(frames=10_000), detector=detector, open_detector=open_detector,
                    lookout=FakeLookout(["rug"]), config=discover_config(0.02)).start()
    pipeline.discover(True)
    assert wait_until(lambda: any(o["label"] == "rug" for o in pipeline.status()["objects"]))
    present = []
    every_frame(pipeline, monkeypatch,
                lambda: present.append(any(o["label"] == "rug" for o in pipeline.status()["objects"])))
    start = open_detector.detects
    assert wait_until(lambda: open_detector.detects >= start + 6)
    frames_later(detector)
    assert present and all(present)                                # held through every dip


def test_a_track_that_starts_below_the_sure_limit_is_nothing_for_a_discovered_name(make):
    low = Detection("rug", 0.22, (300, 300, 400, 350), 42)
    detector = FakeDetector()
    open_detector = ScriptedOpenDetector([low], [Detection("rug", 0.5, (300, 300, 400, 350), 42)])
    pipeline = make(source=FakeSource(frames=10_000), detector=detector, open_detector=open_detector,
                    lookout=FakeLookout(["rug"]), config=discover_config(0.02)).start()
    pipeline.discover(True)
    assert wait_until(lambda: open_detector.detects >= 1)
    assert wait_until(lambda: any(o["label"] == "rug" for o in pipeline.status()["objects"]))
    assert open_detector.detects >= 2                               # only once it reached the sure limit


# ---- drawing a maybe -----------------------------------------------------------------------------

BACKGROUND = 60
SQUARE = Detection("window", 0.26, (100, 100, 200, 200), OFFSET + 3)
SHAPE = Detection("door", 0.71, (300, 150, 400, 250), OFFSET + 4,
                  outline=((300, 150), (400, 160), (390, 250), (310, 240)))


def blank():
    return np.full((360, 640, 3), BACKGROUND, np.uint8)


def test_the_caption_of_a_maybe_has_a_question_mark():
    assert draw.caption_for(Detection("window", 0.26, (0, 0, 9, 9), 100003), unsure=True) == "window? 0.26 #100003"
    assert draw.caption_for(Detection("window", 0.26, (0, 0, 9, 9), 100003)) == "window 0.26 #100003"
    assert draw.caption_for(Detection("window", 0.26, (0, 0, 9, 9)), unsure=True) == "window? 0.26"


def test_a_maybe_is_drawn_in_the_wanted_colour_with_the_ordinary_line(monkeypatch):
    captions = []
    monkeypatch.setattr(draw, "draw_caption", lambda frame, text, x, y, color: captions.append((text, color)))
    maybe, wanted = blank(), blank()
    draw.draw_detections(maybe, [SQUARE], unsure=[SQUARE])
    draw.draw_detections(wanted, [SQUARE], wanted=[OFFSET + 3])
    assert tuple(maybe[150, 100]) == draw.WANTED_COLOR                   # on the edge: the bright yellow
    assert tuple(maybe[150, 103]) == (BACKGROUND,) * 3                   # the ordinary line is thin ...
    assert tuple(wanted[150, 102]) == draw.WANTED_COLOR                  # ... the wanted one is thicker
    assert captions == [("window? 0.26 #100003", draw.WANTED_COLOR), ("window 0.26 #100003", draw.WANTED_COLOR)]


def test_a_maybe_is_marked_by_the_object_itself_not_its_track_number(monkeypatch):
    captions = []
    monkeypatch.setattr(draw, "draw_caption", lambda frame, text, x, y, color: captions.append((text, color)))
    untracked = Detection("window", 0.26, (100, 100, 200, 200))
    twin = Detection("window", 0.26, (300, 100, 400, 200))               # the same values, another object
    frame = blank()
    draw.draw_detections(frame, [untracked, twin], unsure=[untracked])
    assert captions[0] == ("window? 0.26", draw.WANTED_COLOR)
    assert captions[1][0] == "window 0.26" and captions[1][1] != draw.WANTED_COLOR
    assert tuple(frame[150, 100]) == draw.WANTED_COLOR


def test_a_maybe_has_no_tint_inside_its_outline():
    maybe, sure = blank(), blank()
    draw.draw_detections(maybe, [SHAPE], unsure=[SHAPE])
    draw.draw_detections(sure, [SHAPE], wanted=[OFFSET + 4])
    assert tuple(maybe[200, 350]) == (BACKGROUND,) * 3                   # inside: untouched
    assert tuple(sure[200, 350]) != (BACKGROUND,) * 3                    # a sure find is tinted


def base_caption(frame, text, x, y, color):
    """The caption as the code at the base drew it, with plain cv2 calls (640 wide: scale 1)."""
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    band_width, band_height = text_width + 6, text_height + baseline + 6
    top = y - band_height if y - band_height >= 0 else y
    top = max(0, min(top, frame.shape[0] - band_height))
    left = max(0, min(x, frame.shape[1] - band_width))
    cv2.rectangle(frame, (left, top), (left + band_width, top + band_height), color, cv2.FILLED)
    cv2.putText(frame, text, (left + 3, top + 3 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)                                            # black text on the bright yellow


def base_picture():
    """SQUARE and SHAPE, both wanted, as the code at the base drew them."""
    frame, yellow = blank(), (0, 255, 255)
    cv2.rectangle(frame, (100, 100), (200, 200), yellow, 5)
    base_caption(frame, "window 0.26 #100003", 100, 100, yellow)
    points = np.array(SHAPE.outline, dtype=np.int32).reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(points)
    area = frame[y:y + h, x:x + w]
    inside = np.zeros(area.shape[:2], dtype=np.uint8)
    cv2.fillPoly(inside, [points - np.array([x, y], dtype=np.int32)], 255)
    tinted = cv2.addWeighted(area, 0.8, np.full_like(area, yellow), 0.2, 0)
    cv2.copyTo(tinted, inside, area)
    cv2.polylines(frame, [points], True, yellow, 5, cv2.LINE_AA)
    base_caption(frame, "door 0.71 #100004", 300, 150, yellow)
    return frame


def test_calls_without_unsure_draw_exactly_as_at_the_base():
    expected = base_picture()
    for call in (lambda frame: draw.draw_detections(frame, [SQUARE, SHAPE], [OFFSET + 3, OFFSET + 4]),
                 lambda frame: draw.draw_detections(frame, [SQUARE, SHAPE], wanted=[OFFSET + 3, OFFSET + 4])):
        frame = blank()
        call(frame)
        assert np.array_equal(frame, expected)
