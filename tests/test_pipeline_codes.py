"""M16 Part C: the loop reads QR codes and barcodes, keeps them for a moment, draws them and
lists them in status()["codes"].

A fake code reader and the fakes of tests/test_pipeline.py: no camera, model or graphics card.
"""

import json
import logging
import time

import cv2
import numpy as np
import pytest

from car_vision import pipeline as pipeline_module
from car_vision.codes import Code
from car_vision.config import ConfigError
from car_vision.draw import CODE_COLOR
from test_codes import pasted, qr_picture
from test_pipeline import (CONFIG, PHONE, FakeDetector, FakeSightings, FakeSource, SourceFactory, make,  # noqa: F401
                           phone_source, wait_until)

SECRET = "https://example.com/menu?table=7"
MENU = Code("qr", SECRET, (64, 36, 192, 180), ((64, 36), (192, 36), (192, 180), (64, 180)))
SMALL = Code("barcode", "5901234123457", (400, 300, 440, 320), (), "EAN_13")
BIG = Code("qr", "WIFI:S:home;;", (300, 40, 620, 340), ((300, 40), (620, 40), (620, 340), (300, 340)))


def codes_config(**changes):
    return {**CONFIG, "codes": {**CONFIG["codes"], "enabled": True, **changes}}


class FakeReader:
    """Shaped like codes.CodeReader: gives `codes` on every read, or raises."""

    def __init__(self, codes=(MENU,), fail=False):
        self.codes = list(codes)
        self.fail = fail
        self.calls = 0

    def read(self, frame):
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"the reader broke on {SECRET}")
        return list(self.codes)


def reading(make, reader=None, frames=4, config=None, **kwargs):
    """A started pipeline with a fake reader; 20 ms per frame so the clock moves between frames."""
    reader = reader if reader is not None else FakeReader()
    detector = FakeDetector(delay=0.02)
    pipeline = make(source=kwargs.pop("source", FakeSource(frames=frames)), detector=detector,
                    config=config or codes_config(), code_reader=reader, **kwargs).start()
    assert wait_until(lambda: detector.calls == frames)
    return pipeline, reader


# ---- when the reader runs ----------------------------------------------------------------


def test_the_reader_is_called_on_the_first_and_every_fifth_frame(make):
    pipeline, reader = reading(make, frames=11)
    assert wait_until(lambda: reader.calls == 3)     # frames 1, 6 and 11
    time.sleep(0.05)
    assert reader.calls == 3


def test_the_reader_reads_the_clean_frame(make):
    seen = []

    class Looking(FakeReader):
        def read(self, frame):
            seen.append(frame.copy())
            return super().read(frame)

    reading(make, reader=Looking(), frames=6)
    assert seen and all((frame == 60).all() for frame in seen)   # nothing drawn on it yet


def test_with_codes_off_no_reader_is_built_and_the_status_has_no_codes(make, monkeypatch):
    built = []
    monkeypatch.setattr(pipeline_module, "CodeReader", lambda settings: built.append(settings))
    detector = FakeDetector(delay=0.02)
    pipeline = make(source=FakeSource(frames=4), detector=detector).start()     # CONFIG has enabled: false
    assert wait_until(lambda: detector.calls == 4 and pipeline.status()["objects"])
    assert built == []
    assert pipeline.status()["codes"] == []


def test_with_codes_off_a_reader_handed_in_is_not_used(make):
    reader = FakeReader()
    detector = FakeDetector(delay=0.02)
    pipeline = make(source=FakeSource(frames=4), detector=detector, code_reader=reader).start()
    assert wait_until(lambda: detector.calls == 4 and pipeline.status()["objects"])
    assert reader.calls == 0 and pipeline.status()["codes"] == []


def test_with_codes_on_the_real_reader_is_built_from_the_settings(make, monkeypatch):
    built = []
    monkeypatch.setattr(pipeline_module, "CodeReader", lambda settings: built.append(settings) or FakeReader())
    make(config=codes_config())
    assert built == [codes_config()["codes"]]


@pytest.mark.parametrize("key, value", [("enabled", "yes"), ("every_n_frames", 0), ("hold_seconds", -1),
                                        ("max_width", 100)])
def test_wrong_code_settings_stop_the_start(make, key, value):
    with pytest.raises(ConfigError):
        make(config=codes_config(**{key: value}), code_reader=FakeReader())


# ---- keeping the codes on screen -----------------------------------------------------------


def frame_of():
    return np.full((360, 640, 3), 60, dtype=np.uint8)


def test_a_code_stays_for_hold_seconds_after_its_last_read_and_then_goes(make):
    pipeline = make(config=codes_config(), code_reader=FakeReader())
    assert pipeline._codes_to_show(frame_of(), 0, 10.0) == (MENU,)     # read
    pipeline._code_reader.codes = []                                   # the code has left the view
    assert pipeline._codes_to_show(frame_of(), 1, 10.04) == (MENU,)    # not a read frame: kept
    assert pipeline._codes_to_show(frame_of(), 5, 10.2) == (MENU,)     # read 0.2 s later, not seen: kept
    assert pipeline._codes_to_show(frame_of(), 7, 11.5) == (MENU,)     # 1.5 s after its last read
    assert pipeline._codes_to_show(frame_of(), 8, 11.6) == ()          # 1.6 s: gone


class MissingReader(FakeReader):
    """A code held still that the reader misses on some of its runs (blur, a reflection)."""

    def __init__(self, missed_runs):
        super().__init__()
        self.missed_runs = set(missed_runs)

    def read(self, frame):
        codes = super().read(frame)
        return [] if self.calls in self.missed_runs else codes


@pytest.mark.parametrize("frames_per_second, missed_runs", [
    (3, {4}),          # every 1.67 s, longer than hold_seconds, and one run misses the code
    (2, ()),           # every 2.5 s: even the frames between two runs are 2 s apart from the last
])
def test_at_a_low_frame_rate_a_steady_code_does_not_blink(make, frames_per_second, missed_runs):
    """The reader runs on every 5th frame. A code stays for hold_seconds (1.5 s), or for twice the
    time between the reader's last two runs when that is longer, so it never blinks between runs.
    The rhythm is known from the reader's second run on (frame 5)."""
    pipeline = make(config=codes_config(), code_reader=MissingReader(missed_runs))
    shown = [bool(pipeline._codes_to_show(frame_of(), index, 100.0 + index / frames_per_second))
             for index in range(36)]                                   # the code held still all along
    assert all(shown[5:])


def test_at_a_low_frame_rate_a_code_that_left_goes_after_twice_the_rhythm(make):
    reader = FakeReader()
    pipeline = make(config=codes_config(), code_reader=reader)
    for index in range(31):                                            # 3 frames a second
        pipeline._codes_to_show(frame_of(), index, 100.0 + index / 3)
    reader.codes = []                                                  # now it leaves the view
    last_read = 100.0 + 30 / 3                                         # frame 30 was the last run
    assert pipeline._codes_to_show(frame_of(), 31, last_read + 3.3) == (MENU,)   # within 2 x 1.67 s
    assert pipeline._codes_to_show(frame_of(), 32, last_read + 3.4) == ()


def test_at_a_high_frame_rate_hold_seconds_still_counts(make):
    pipeline = make(config=codes_config(), code_reader=FakeReader())
    for index in range(11):                                            # 50 frames a second
        pipeline._codes_to_show(frame_of(), index, 10.0 + index * 0.02)
    pipeline._code_reader.codes = []
    assert pipeline._codes_to_show(frame_of(), 11, 10.2 + 1.5) == (MENU,)
    assert pipeline._codes_to_show(frame_of(), 12, 10.2 + 1.6) == ()


def test_a_code_read_again_keeps_its_newest_box(make):
    reader = FakeReader()
    pipeline = make(config=codes_config(every_n_frames=1), code_reader=reader)
    pipeline._codes_to_show(frame_of(), 0, 10.0)
    moved = Code("qr", SECRET, (70, 40, 200, 190), ((70, 40), (200, 40), (200, 190), (70, 190)))
    reader.codes = [moved]
    assert pipeline._codes_to_show(frame_of(), 1, 11.0) == (moved,)
    assert pipeline._codes_to_show(frame_of(), 2, 12.4) == (moved,)    # its hold counts from the new read


def test_with_no_hold_a_code_shows_only_on_the_frame_it_was_read(make):
    reader = FakeReader()
    pipeline = make(config=codes_config(hold_seconds=0), code_reader=reader)
    assert pipeline._codes_to_show(frame_of(), 0, 10.0) == (MENU,)
    assert pipeline._codes_to_show(frame_of(), 1, 10.02) == ()


# ---- the status ---------------------------------------------------------------------------


def test_status_codes_have_the_shape_the_page_reads(make):
    pipeline, _reader = reading(make)
    assert wait_until(lambda: pipeline.status()["codes"])
    status = pipeline.status()
    assert status["codes"] == [{"kind": "qr", "format": "", "text": SECRET, "link": SECRET,
                                "altered": False, "box": [0.1, 0.1, 0.3, 0.5]}]
    assert json.loads(json.dumps(status)) == status
    keys = list(status)
    assert keys.index("codes") == keys.index("objects") + 1


def test_the_text_is_nowhere_in_the_status_outside_codes(make):
    pipeline, _reader = reading(make)
    assert wait_until(lambda: pipeline.status()["codes"])
    status = pipeline.status()
    assert json.dumps(status).count("example.com") == 2           # the text and the link, in "codes"
    rest = {key: value for key, value in status.items() if key != "codes"}
    assert "example.com" not in json.dumps(rest)


def test_an_altered_text_is_never_a_link():
    cut = Code("qr", SECRET, (0, 0, 10, 10), (), altered=True)
    plain = Code("qr", SECRET, (20, 0, 40, 10), ())
    first, second = pipeline_module.codes_in_view([cut, plain], (640, 360))
    assert (second["link"], second["altered"]) == (None, True)
    assert (first["link"], first["altered"]) == (SECRET, False)


def test_status_lists_the_largest_first_with_a_link_only_for_a_web_address(make):
    pipeline, _reader = reading(make, reader=FakeReader([SMALL, MENU, BIG]))
    assert wait_until(lambda: pipeline.status()["codes"])
    codes = pipeline.status()["codes"]
    assert [(c["kind"], c["text"], c["format"], c["link"]) for c in codes] == [
        ("qr", "WIFI:S:home;;", "", None), ("qr", SECRET, "", SECRET), ("barcode", "5901234123457", "EAN_13", None)]


def test_status_lists_at_most_eight_codes(make):
    many = [Code("qr", f"code {n}", (10 * n, 0, 10 * n + 5 + n, 20), ()) for n in range(12)]
    pipeline, _reader = reading(make, reader=FakeReader(many))
    assert wait_until(lambda: pipeline.status()["codes"])
    texts = [code["text"] for code in pipeline.status()["codes"]]
    assert texts == [f"code {n}" for n in range(11, 3, -1)]


def test_status_codes_are_empty_after_a_camera_switch(make):
    new = phone_source(frames=0, connected=False)               # a phone that sends nothing yet
    pipeline, _reader = reading(make, source_factory=SourceFactory(new))
    assert wait_until(lambda: pipeline.status()["codes"])
    pipeline.use_phone(PHONE)
    assert pipeline.status()["codes"] == []
    time.sleep(0.05)
    assert pipeline.status()["codes"] == []
    assert pipeline._codes == {}                                # the loop forgot them too


def test_a_new_frame_size_forgets_the_codes(make):
    pipeline = make(config=codes_config(), code_reader=FakeReader())
    pipeline._note_frame_size(frame_of())
    pipeline._codes_to_show(frame_of(), 0, 10.0)
    pipeline._note_frame_size(np.zeros((480, 640, 3), np.uint8))
    assert pipeline._codes == {}


def test_status_leaves_out_the_codes_of_a_stale_stream(make, monkeypatch):
    monkeypatch.setattr(pipeline_module, "STALE_SECONDS", 0.2)
    source = FakeSource(frames=3)
    pipeline, _reader = reading(make, frames=3, source=source)
    assert wait_until(lambda: pipeline.status()["codes"])
    source.connected = False
    assert wait_until(lambda: not pipeline.status()["codes"])


def test_the_codes_are_drawn_on_the_picture(make):
    pipeline, _reader = reading(make)
    jpeg = pipeline.latest_jpeg()
    picture = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    edge = picture[100, 186:199].astype(int)                     # across the right edge of MENU
    closest = np.abs(edge - np.array(CODE_COLOR)).max(axis=1).min()
    assert closest <= 40                                         # JPEG blurs colours a little


# ---- a reader that breaks; the text stays on the page ------------------------------------------


def test_a_reader_that_raises_does_not_stop_the_loop_and_is_logged_once(make, caplog):
    with caplog.at_level(logging.WARNING, logger="car_vision.pipeline"):
        pipeline, reader = reading(make, reader=FakeReader(fail=True), frames=16)
        assert wait_until(lambda: reader.calls == 4 and pipeline.latest_jpeg() is not None)
    status = pipeline.status()
    assert status["error"] is None and status["connected"] is True and status["codes"] == []
    assert status["objects"]                                     # the frames were all shown
    warnings = [r for r in caplog.records if "codes" in r.getMessage()]
    assert len(warnings) == 1
    assert SECRET not in caplog.text                             # not even from the error's message


def test_a_real_reader_in_the_loop_finds_a_qr_code_and_keeps_its_text_to_the_status(make, caplog):
    """OpenCV's own detectors, a QR picture made in memory, the real loop: no camera or model."""
    frame = pasted(qr_picture(SECRET), (640, 360), (100, 30))
    detector = FakeDetector(delay=0.02)
    with caplog.at_level(logging.DEBUG):
        pipeline = make(source=FakeSource(frames=6, picture=frame), detector=detector,
                        config=codes_config()).start()             # no reader handed in: the real one
        assert wait_until(lambda: detector.calls == 6 and pipeline.status()["codes"], timeout=10)
    codes = pipeline.status()["codes"]
    assert [(c["kind"], c["text"], c["link"], c["altered"]) for c in codes] == [("qr", SECRET, SECRET, False)]
    x1, y1, x2, y2 = codes[0]["box"]
    assert 100 / 640 <= x1 < x2 <= (100 + 232 + 64) / 640 and 30 / 360 <= y1 < y2 <= (30 + 232 + 64) / 360
    assert "example.com" not in caplog.text
    snapshot = pipeline.state.snapshot()
    assert "example.com" not in repr(snapshot)


def test_the_text_is_not_in_the_log_the_world_state_or_the_memory(make, caplog):
    sightings = FakeSightings()
    config = {**codes_config(), "memory": {**CONFIG["memory"], "min_seconds": 0.0}}
    with caplog.at_level(logging.DEBUG):
        pipeline, _reader = reading(make, frames=6, config=config, sightings=sightings)
        assert wait_until(lambda: pipeline.status()["codes"])
    assert SECRET not in caplog.text and "example.com" not in caplog.text
    snapshot = pipeline.state.snapshot()
    assert SECRET not in repr(snapshot) and "example.com" not in repr(snapshot)
    assert "example.com" not in repr(sightings.seen_calls)
    assert all(call["label"] != "qr" for call in sightings.seen_calls)
