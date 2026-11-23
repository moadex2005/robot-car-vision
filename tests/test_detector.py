"""The detector logic, tested with a fake YOLO model. No camera, network or graphics card needed."""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from car_vision import detector as detector_module
from car_vision.config import ConfigError
from car_vision.detector import (Detector, DetectorError, LabelVotes, LiveStats, OpenDetector, clip_box,
                                 describe, main, tracker_names)
from car_vision.stream import FpsCounter

SETTINGS = {
    "model": "yolo26n.pt",
    "models_dir": "models",
    "confidence": 0.4,
    "image_size": 640,
    "device": "cpu",
    "tracker": "bytetrack.yaml",
}

NAMES = {0: "person", 16: "dog", 41: "cup", 56: "chair", 57: "couch"}
PACKAGE_DIR = Path(__file__).resolve().parent.parent / "car_vision"
HEAVY = {"ultralytics", "torch"}


# ---- fakes shaped like the Ultralytics results -----------------------------


class FakeTensor:
    """Behaves like a torch tensor on the graphics card: needs .cpu().numpy()."""

    def __init__(self, values) -> None:
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._values

    def __len__(self) -> int:
        return len(self._values)


class FakeBoxes:
    """Shaped like ultralytics.engine.results.Boxes: xyxy, conf, cls and id."""

    def __init__(self, xyxy, conf, cls, ids=None) -> None:
        self.xyxy = FakeTensor(np.reshape(xyxy, (-1, 4)))
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)
        self.id = None if ids is None else FakeTensor(ids)   # None when nothing is tracked yet

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes
        self.names = NAMES


class FakeModel:
    """Records every track() call and hands back the prepared results in turn."""

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls = []
        self.frames = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        self.frames.append(frame)
        return [self.results.pop(0)] if self.results else []


class FakeTracker:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class FakePredictor:
    def __init__(self) -> None:
        self.trackers = [FakeTracker()]


class FakeModelWithTracker(FakeModel):
    """Like the real YOLO object: its trackers sit on model.predictor.trackers."""

    def __init__(self, *results) -> None:
        super().__init__(*results)
        self.predictor = FakePredictor()

    @property
    def resets(self) -> int:
        return self.predictor.trackers[0].resets


def frame(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


def tracked_result():
    return FakeResult(FakeBoxes(
        xyxy=[[10.4, 20.6, 110.2, 220.9], [300.0, 50.0, 400.0, 150.0]],
        conf=[0.87, 0.55],
        cls=[0, 56],
        ids=[3, 7],
    ))


def one_box(cls, track_id):
    return FakeResult(FakeBoxes(xyxy=[[10, 10, 50, 50]], conf=[0.7], cls=[cls], ids=None if track_id is None else [track_id]))


@pytest.fixture
def no_graphics_card(monkeypatch):
    monkeypatch.setattr(detector_module, "cuda_available", lambda: False)


@pytest.fixture
def graphics_card(monkeypatch):
    monkeypatch.setattr(detector_module, "cuda_available", lambda: True)


# ---- detect() --------------------------------------------------------------


def test_detect_returns_labels_confidences_boxes_and_track_ids():
    found = Detector(SETTINGS, model=FakeModel(tracked_result())).detect(frame())

    assert [d.label for d in found] == ["person", "chair"]
    assert found[0].confidence == pytest.approx(0.87)
    assert isinstance(found[0].confidence, float)
    assert found[0].box == (10, 21, 110, 221)
    assert all(isinstance(v, int) for d in found for v in d.box)
    assert [d.track_id for d in found] == [3, 7]
    assert all(isinstance(d.track_id, int) for d in found)


def test_boxes_without_track_ids_get_none():
    found = Detector(SETTINGS, model=FakeModel(one_box(41, None))).detect(frame())
    assert len(found) == 1
    assert found[0].label == "cup"
    assert found[0].track_id is None


def test_no_boxes_returns_empty_list():
    empty = FakeResult(FakeBoxes(xyxy=np.zeros((0, 4)), conf=[], cls=[], ids=None))
    assert Detector(SETTINGS, model=FakeModel(empty)).detect(frame()) == []


def test_missing_boxes_and_no_results_return_empty_list():
    assert Detector(SETTINGS, model=FakeModel(FakeResult(None))).detect(frame()) == []
    assert Detector(SETTINGS, model=FakeModel()).detect(frame()) == []


def test_boxes_past_the_frame_are_clipped():
    result = FakeResult(FakeBoxes(xyxy=[[-15.0, -3.0, 700.0, 500.0]], conf=[0.6], cls=[0], ids=[1]))
    found = Detector(SETTINGS, model=FakeModel(result)).detect(frame(640, 480))
    assert found[0].box == (0, 0, 639, 479)


def test_clip_box_keeps_inner_boxes():
    assert clip_box(np.array([5.0, 6.0, 7.0, 8.0]), 640, 480) == (5, 6, 7, 8)


def test_settings_are_passed_to_the_tracker():
    model = FakeModel(tracked_result())
    settings = {**SETTINGS, "confidence": 0.25, "image_size": 320, "tracker": "botsort.yaml"}
    Detector(settings, model=model).detect(frame())

    call = model.calls[0]
    assert call["persist"] is True
    assert call["tracker"] == "botsort.yaml"
    assert call["conf"] == pytest.approx(0.25)
    assert call["imgsz"] == 320
    assert call["device"] == "cpu"
    assert call["verbose"] is False
    # Half precision breaks GTX 16-series cards. Ultralytics 8.4 sets it with
    # quantize= (half= is the old name); neither may be passed.
    assert "half" not in call and "quantize" not in call


def test_none_frame_raises_value_error():
    with pytest.raises(ValueError):
        Detector(SETTINGS, model=FakeModel()).detect(None)


def test_empty_frame_raises_value_error():
    with pytest.raises(ValueError):
        Detector(SETTINGS, model=FakeModel()).detect(np.zeros((0, 0, 3), dtype=np.uint8))


# ---- device choice (fix 2) ---------------------------------------------------


def test_cuda_without_graphics_card_raises_config_error(no_graphics_card):
    with pytest.raises(ConfigError, match="graphics card"):
        Detector({**SETTINGS, "device": "cuda"}, model=FakeModel())


def test_cuda_with_graphics_card_is_used(graphics_card):
    model = FakeModel(tracked_result())
    found_on = Detector({**SETTINGS, "device": "cuda"}, model=model)
    found_on.detect(frame())
    assert found_on.device == "cuda"
    assert model.calls[0]["device"] == "cuda"


def test_auto_with_injected_model_skips_the_torch_check(monkeypatch):
    def must_not_run():
        raise AssertionError("cuda_available() was called for an injected model")

    monkeypatch.setattr(detector_module, "cuda_available", must_not_run)
    assert Detector({**SETTINGS, "device": "auto"}, model=FakeModel()).device == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(ConfigError):
        Detector({**SETTINGS, "device": "gpu"}, model=FakeModel())


# ---- settings checks (fix 7) -------------------------------------------------


def test_missing_setting_is_reported():
    settings = dict(SETTINGS)
    del settings["tracker"]
    with pytest.raises(ConfigError, match="tracker"):
        Detector(settings, model=FakeModel())


@pytest.mark.parametrize("confidence", [0, 0.0, -0.1, 1.01, 5])
def test_confidence_outside_zero_to_one_is_rejected(confidence):
    with pytest.raises(ConfigError, match="confidence"):
        Detector({**SETTINGS, "confidence": confidence}, model=FakeModel())


@pytest.mark.parametrize("confidence", [0.01, 0.4, 1, 1.0])
def test_confidence_inside_zero_to_one_is_accepted(confidence):
    Detector({**SETTINGS, "confidence": confidence}, model=FakeModel())


@pytest.mark.parametrize("size", [0, -32, 100, 650])
def test_image_size_must_be_a_positive_multiple_of_32(size):
    with pytest.raises(ConfigError, match="image_size"):
        Detector({**SETTINGS, "image_size": size}, model=FakeModel())


@pytest.mark.parametrize("size", [32, 320, 640])
def test_good_image_sizes_are_accepted(size):
    Detector({**SETTINGS, "image_size": size}, model=FakeModel())


def test_tracker_names_come_from_the_installed_package():
    names = tracker_names()
    assert "bytetrack.yaml" in names and "botsort.yaml" in names
    assert all(name.endswith(".yaml") for name in names)


def test_unknown_tracker_is_rejected():
    with pytest.raises(ConfigError, match="bytetrack.yaml"):
        Detector({**SETTINGS, "tracker": "sorty.yaml"}, model=FakeModel())


def test_tracker_list_is_read_not_guessed(monkeypatch):
    monkeypatch.setattr(detector_module, "tracker_names", lambda: ["only.yaml"])
    Detector({**SETTINGS, "tracker": "only.yaml"}, model=FakeModel())
    with pytest.raises(ConfigError):
        Detector(SETTINGS, model=FakeModel())


# ---- tracker reset on a new frame size (fix 3) ------------------------------


def test_tracker_resets_when_the_frame_size_changes():
    model = FakeModelWithTracker()
    detector = Detector(SETTINGS, model=model)
    detector.detect(frame(640, 480))
    detector.detect(frame(640, 480))
    assert model.resets == 0
    detector.detect(frame(640, 360))
    assert model.resets == 1
    detector.detect(frame(640, 360))
    assert model.resets == 1


def test_size_change_with_a_fake_model_without_tracker_is_fine():
    detector = Detector(SETTINGS, model=FakeModel())
    detector.detect(frame(640, 480))
    detector.detect(frame(320, 240))            # no predictor.trackers: nothing to reset


def test_size_change_forgets_label_votes():
    model = FakeModelWithTracker(one_box(57, 3), one_box(57, 3), one_box(16, 3))
    detector = Detector(SETTINGS, model=model)
    detector.detect(frame(640, 480))
    detector.detect(frame(640, 480))
    found = detector.detect(frame(640, 360))     # new size: track 3 starts with a clean slate
    assert found[0].label == "dog"


# ---- label votes per track (fix 4) ------------------------------------------


def test_a_still_object_keeps_its_most_common_label():
    model = FakeModel(one_box(57, 3), one_box(57, 3), one_box(16, 3), one_box(57, 3))
    detector = Detector(SETTINGS, model=model)
    labels = [detector.detect(frame())[0].label for _ in range(4)]
    assert labels == ["couch", "couch", "couch", "couch"]


def test_label_vote_keeps_box_confidence_and_track():
    model = FakeModel(one_box(57, 3), one_box(57, 3), one_box(16, 3))
    detector = Detector(SETTINGS, model=model)
    detector.detect(frame())
    detector.detect(frame())
    flipped = detector.detect(frame())[0]
    assert (flipped.label, flipped.track_id, flipped.box) == ("couch", 3, (10, 10, 50, 50))
    assert flipped.confidence == pytest.approx(0.7)


def test_untracked_detections_keep_their_own_label():
    model = FakeModel(one_box(57, None), one_box(16, None))
    detector = Detector(SETTINGS, model=model)
    assert detector.detect(frame())[0].label == "couch"
    assert detector.detect(frame())[0].label == "dog"


def test_tie_goes_to_the_newest_label():
    votes = LabelVotes()
    assert votes.vote(3, "couch", now=0.0) == "couch"
    assert votes.vote(3, "dog", now=0.1) == "dog"          # 1 to 1: newest wins
    assert votes.vote(3, "couch", now=0.2) == "couch"      # 2 to 1
    assert votes.vote(3, "dog", now=0.3) == "dog"          # 2 to 2: newest wins again


def test_tracks_have_separate_votes():
    votes = LabelVotes()
    votes.vote(1, "cup", now=0.0)
    assert votes.vote(2, "bottle", now=0.0) == "bottle"
    assert votes.vote(1, "bottle", now=0.0) == "bottle"    # 1 to 1 for track 1, newest


def test_votes_are_forgotten_after_30_seconds_unseen():
    votes = LabelVotes()
    votes.vote(1, "cup", now=0.0)
    votes.vote(2, "dog", now=20.0)
    votes.forget_old(now=29.0)
    assert len(votes) == 2
    votes.forget_old(now=31.0)
    assert len(votes) == 1                                 # track 1 gone, track 2 kept
    assert votes.vote(1, "bottle", now=31.0) == "bottle"


def test_detector_forgets_old_tracks_by_its_clock():
    model = FakeModel(one_box(57, 3), one_box(57, 3), one_box(16, 3))
    detector = Detector(SETTINGS, model=model)
    detector._clock = lambda: 0.0
    detector.detect(frame())
    detector.detect(frame())                               # couch 2, dog 0
    detector._clock = lambda: 100.0
    assert detector.detect(frame())[0].label == "dog"      # the old couch votes were dropped first
    assert len(detector._votes) == 1


def test_vote_table_stays_small_over_many_tracks():
    votes = LabelVotes()
    for second in range(1000):
        votes.vote(second, "person", now=float(second))
        votes.forget_old(now=float(second))
    assert len(votes) <= 31


# ---- warm-up when the real model loads (fix 5) ------------------------------


def test_real_model_is_warmed_up_and_left_without_tracks(monkeypatch, capsys, graphics_card):
    model = FakeModelWithTracker(one_box(0, 1))
    monkeypatch.setattr(Detector, "_load_model", lambda self: model)
    detector = Detector({**SETTINGS, "device": "auto"})

    assert "Warming up the model" in capsys.readouterr().out
    assert detector.device == "cuda"
    assert len(model.calls) == 1
    assert model.calls[0]["device"] == "cuda"
    assert not model.frames[0].any() and model.frames[0].shape == (640, 640, 3)
    assert model.resets == 1                     # tracks from the blank frame are dropped
    assert len(detector._votes) == 0

    detector.detect(frame(640, 480))             # a different size from the blank frame
    assert model.resets == 1                     # but no second reset: the warm-up left no size


def test_injected_model_is_not_warmed_up():
    model = FakeModel()
    Detector(SETTINGS, model=model)
    assert model.calls == []


# ---- the live loop in main() (fixes 1, 6, 8 and --file) ---------------------


class FakeSource:
    """Stands in for FrameSource: hands out the given frames, then reports the end."""

    instances = []

    def __init__(self, settings, frames=3, repeats=1, fail_first_frame=False) -> None:
        self.settings = settings
        self.source = settings["source"]
        self.fps = FpsCounter()
        self.connected = True
        self.frames_left = frames
        self.repeats = repeats          # how often read() returns the same frame number
        self.reads = 0
        self.sequence = 0
        self.no_frame = fail_first_frame
        self.started = False
        self.stopped = False
        FakeSource.instances.append(self)

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    def wait_for_frame(self, timeout=10.0):
        return not self.no_frame

    @property
    def ended(self):
        return self.frames_left <= 0

    def read(self):
        if self.reads % self.repeats == 0:
            self.sequence += 1
            self.frames_left -= 1
        self.reads += 1
        return np.zeros((48, 64, 3), dtype=np.uint8), self.sequence


class FakeDetector:
    def __init__(self, settings, problem=None, after=0) -> None:
        self.device = "cpu"
        self.model_name = "fake.pt"
        self.calls = 0
        self.problem = problem
        self.after = after

    def detect(self, frame):
        self.calls += 1
        if self.problem is not None and self.calls > self.after:
            raise self.problem
        return [detector_module.Detection("person", 0.9, (1, 1, 10, 10), 1)]


@pytest.fixture
def fake_loop(monkeypatch):
    """Replace the camera, the model and the OpenCV window with fakes."""
    FakeSource.instances = []
    windows = {"shown": 0, "closed": 0}
    sleeps = []
    options = {"source": {}, "detector": {}}

    monkeypatch.setattr(detector_module, "FrameSource",
                        lambda settings: FakeSource(settings, **options["source"]))
    monkeypatch.setattr(detector_module, "Detector",
                        lambda settings: FakeDetector(settings, **options["detector"]))
    monkeypatch.setattr(detector_module.cv2, "imshow",
                        lambda *a: windows.__setitem__("shown", windows["shown"] + 1))
    monkeypatch.setattr(detector_module.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(detector_module.cv2, "destroyAllWindows",
                        lambda: windows.__setitem__("closed", windows["closed"] + 1))
    monkeypatch.setattr(detector_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    return {"windows": windows, "sleeps": sleeps, "options": options}


def test_main_runs_to_the_end_of_a_clip(fake_loop, capsys):
    assert main(["--source", "webcam", "--no-window"]) == 0
    source = FakeSource.instances[0]
    assert source.started and source.stopped
    assert fake_loop["windows"]["closed"] == 1
    assert "Average detection FPS" in capsys.readouterr().out


def test_file_argument_implies_file_source_and_overrides_the_path(fake_loop):
    assert main(["--file", "data/samples/people_clip.mp4", "--no-window"]) == 0
    settings = FakeSource.instances[0].settings
    assert settings["source"] == "file"
    assert settings["file_path"] == "data/samples/people_clip.mp4"


def test_file_argument_cannot_be_combined_with_another_source():
    with pytest.raises(SystemExit):
        main(["--file", "clip.mp4", "--source", "webcam"])


def test_missing_clip_gives_one_plain_message(capsys):
    # The real FrameSource and config.yaml; the clip does not exist, so nothing loads.
    assert main(["--file", "data/samples/no_such_clip.mp4", "--no-window"]) == 1
    output = capsys.readouterr()
    assert "Cannot start" in output.out and "does not exist" in output.out
    assert "Traceback" not in output.out + output.err


def test_config_error_from_the_detector_gives_one_plain_message(fake_loop, monkeypatch, capsys):
    def broken(settings):
        raise ConfigError("detector.device is 'cuda' but PyTorch cannot see a graphics card.")

    monkeypatch.setattr(detector_module, "Detector", broken)
    assert main(["--source", "webcam", "--no-window"]) == 1
    assert "Cannot start" in capsys.readouterr().out


def test_ctrl_c_still_releases_the_camera_and_prints_the_summary(fake_loop, capsys):
    fake_loop["options"]["source"] = {"frames": 10}
    fake_loop["options"]["detector"] = {"problem": KeyboardInterrupt(), "after": 3}
    assert main(["--source", "webcam"]) == 0
    source = FakeSource.instances[0]
    assert source.stopped
    assert fake_loop["windows"]["closed"] == 1
    output = capsys.readouterr().out
    assert "Stopped with Ctrl+C" in output
    assert "over 2 calls" in output              # 3 calls finished, the first left out


def test_an_exception_still_releases_the_camera_and_prints_the_summary(fake_loop, capsys):
    fake_loop["options"]["source"] = {"frames": 10}
    fake_loop["options"]["detector"] = {"problem": RuntimeError("model broke"), "after": 1}
    with pytest.raises(RuntimeError):
        main(["--source", "webcam", "--no-window"])
    assert FakeSource.instances[0].stopped
    assert fake_loop["windows"]["closed"] == 1
    assert "Average detection FPS" in capsys.readouterr().out


def test_no_frame_from_the_phone_prints_the_hint(fake_loop, capsys):
    fake_loop["options"]["source"] = {"fail_first_frame": True}
    assert main(["--source", "phone", "--no-window"]) == 1
    output = capsys.readouterr().out
    assert "No frame arrived from the phone stream" in output
    assert detector_module.PHONE_HINT in output
    assert FakeSource.instances[0].stopped


def test_window_loop_waits_when_no_new_frame_has_arrived(fake_loop):
    fake_loop["options"]["source"] = {"frames": 3, "repeats": 4}   # each frame read 4 times
    assert main(["--source", "webcam"]) == 0
    assert fake_loop["windows"]["shown"] == 3                      # drawn once per new frame
    assert len(fake_loop["sleeps"]) >= 6
    assert all(0 < s <= 0.05 for s in fake_loop["sleeps"])


def test_window_mode_does_not_keep_seen_detections():
    stats = LiveStats(keep_labels=False)
    found = [detector_module.Detection("person", 0.9, (1, 1, 10, 10), 1)]
    for step in range(1000):
        stats.record(20.0, found, now=step * 0.1)
    assert stats.second_seen == []
    assert stats.calls == 1000


def test_console_mode_clears_seen_detections_every_second():
    stats = LiveStats(keep_labels=True)
    found = [detector_module.Detection("person", 0.9, (1, 1, 10, 10), 1)]
    stats.record(5000.0, found, now=0.0)
    stats.record(20.0, found, now=0.1)
    stats.record(30.0, found, now=0.2)
    assert len(stats.second_seen) == 3
    assert stats.second_average_ms() == pytest.approx(25.0)
    stats.new_second()
    assert stats.second_seen == [] and stats.second_ms == []
    assert "average detect 25.0 ms over 2 calls" in stats.summary("cpu")


def test_describe_lists_labels_once_with_track_ids():
    found = Detector(SETTINGS, model=FakeModel(tracked_result())).detect(frame())
    assert describe(found + found) == "person #3, chair #7"
    assert describe([]) == "nothing"


# ---- keeping heavy libraries in one place (fix 2 and fix 9) -----------------


def test_importing_the_detector_with_a_fake_model_loads_nothing_heavy():
    code = (
        "import sys, numpy as np\n"
        "from car_vision.detector import Detector\n"
        "class M:\n"
        "    def track(self, frame, **kw): return []\n"
        "settings = dict(model='m.pt', models_dir='models', confidence=0.4, image_size=640,\n"
        "                device='auto', tracker='bytetrack.yaml')\n"
        "d = Detector(settings, model=M())\n"
        "d.detect(np.zeros((4, 4, 3), np.uint8))\n"
        "print(d.device, sorted(m for m in ('torch', 'ultralytics') if m in sys.modules))\n"
    )
    root = PACKAGE_DIR.parent
    output = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                            text=True, timeout=60, check=True).stdout
    assert output.strip() == "cpu []"


def heavy_imports(source: str) -> list[str]:
    """Every import of ultralytics or torch anywhere in the code, nested or not."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        found += [name for name in names if name.split(".")[0] in HEAVY]
    return found


def test_import_guard_finds_imports_anywhere():
    source = (
        "import os, torch.cuda\n"
        "def load():\n"
        "    from ultralytics.engine import model\n"
        "    if True:\n"
        "        import ultralytics as u\n"
        "class A:\n"
        "    def f(self):\n"
        "        from torch import nn\n"
        "x = 1; import torch\n"
        "from . import torchvision_helpers\n"
        "import torchvision\n"
    )
    expected = ["torch.cuda", "ultralytics.engine", "ultralytics", "torch", "torch"]
    assert sorted(heavy_imports(source)) == sorted(expected)   # torchvision is not torch


def test_only_detector_imports_ultralytics_or_torch():
    offenders = {
        path.name: heavy_imports(path.read_text(encoding="utf-8"))
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        if path.name != "detector.py"
    }
    assert {name: found for name, found in offenders.items() if found} == {}


def test_detector_itself_imports_them_only_inside_functions():
    tree = ast.parse((PACKAGE_DIR / "detector.py").read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert heavy_imports(ast.unparse(ast.Module(body=top_level, type_ignores=[]))) == []


# ---- OpenDetector: finding things by a typed name (M13 part A) -------------

OPEN_SETTINGS = {
    **SETTINGS,
    "open_model": "yoloe-26s-seg.pt",
    "open_confidence": 0.25,
    "open_every_n_frames": 3,
}


class FakeYoloe(FakeModelWithTracker):
    """Shaped like ultralytics.YOLOE: get_text_pe() and set_classes() besides track().

    It notes the working directory at each text call, like the real one that
    looks for its text encoder there.
    """

    def __init__(self, *results, fail_text=False) -> None:
        super().__init__(*results)
        self.classes = []
        self.text_calls = []
        self.fail_text = fail_text

    def get_text_pe(self, names):
        self.text_calls.append((list(names), Path.cwd()))
        if self.fail_text:
            raise ConnectionError("no internet")
        return f"embeddings for {names}"

    def set_classes(self, names, embeddings):
        assert embeddings == f"embeddings for {names}"
        self.classes.append(list(names))


def open_result(cls, ids=None, names=None):
    """One box per class number in cls; names is what result.names says."""
    count = len(cls)
    result = FakeResult(FakeBoxes(xyxy=[[10, 10, 50, 50]] * count, conf=[0.6] * count, cls=cls, ids=ids))
    result.names = names if names is not None else {}
    return result


@pytest.fixture
def no_real_yoloe(monkeypatch):
    """Make sure nothing in a test reaches the real Ultralytics loader."""
    def must_not_run(weights):
        raise AssertionError(f"the real model was loaded from {weights}")

    monkeypatch.setattr(detector_module, "load_yoloe", must_not_run)


# settings and construction


def test_open_detector_reads_its_settings(no_real_yoloe):
    detector = OpenDetector(OPEN_SETTINGS)
    assert detector.model_name == "yoloe-26s-seg.pt"
    assert detector.confidence == pytest.approx(0.25)
    assert detector.models_dir == "models"


@pytest.mark.parametrize("key", ["open_model", "open_confidence", "models_dir", "image_size", "device", "tracker"])
def test_open_detector_missing_setting_is_reported(key):
    settings = dict(OPEN_SETTINGS)
    del settings[key]
    with pytest.raises(ConfigError, match=key):
        OpenDetector(settings)


@pytest.mark.parametrize("confidence", [0, -0.1, 1.01])
def test_open_confidence_outside_zero_to_one_is_rejected(confidence):
    with pytest.raises(ConfigError, match="open_confidence"):
        OpenDetector({**OPEN_SETTINGS, "open_confidence": confidence})


@pytest.mark.parametrize("change", [{"image_size": 100}, {"tracker": "sorty.yaml"}, {"device": "gpu"},
                                    {"open_model": "  "}, {"open_confidence": True}])
def test_open_detector_bad_settings_are_rejected(change):
    with pytest.raises(ConfigError):
        OpenDetector({**OPEN_SETTINGS, **change})


def test_nothing_is_loaded_at_construction(monkeypatch, no_real_yoloe):
    def must_not_run():
        raise AssertionError("cuda_available() was called at construction")

    monkeypatch.setattr(detector_module, "cuda_available", must_not_run)
    detector = OpenDetector({**OPEN_SETTINGS, "device": "cuda"})
    assert detector.loaded is False
    assert detector.names == ()


def test_constructing_the_open_detector_imports_nothing_heavy():
    code = (
        "import sys, numpy as np\n"
        "from car_vision.detector import OpenDetector\n"
        "settings = dict(open_model='m.pt', models_dir='models', open_confidence=0.25, image_size=640,\n"
        "                device='auto', tracker='bytetrack.yaml')\n"
        "d = OpenDetector(settings)\n"
        "d.set_names([])\n"
        "print(d.loaded, d.detect(np.zeros((4, 4, 3), np.uint8)),\n"
        "      sorted(m for m in ('torch', 'ultralytics', 'clip') if m in sys.modules))\n"
    )
    output = subprocess.run([sys.executable, "-c", code], cwd=PACKAGE_DIR.parent, capture_output=True,
                            text=True, timeout=60, check=True).stdout
    assert output.strip() == "False [] []"


# load()


def test_load_uses_the_weights_in_models_dir_and_keeps_the_working_directory(monkeypatch, tmp_path, graphics_card):
    seen = []
    model = FakeYoloe()

    def fake_load(weights):
        seen.append((weights, Path.cwd()))
        return model

    monkeypatch.setattr(detector_module, "load_yoloe", fake_load)
    monkeypatch.chdir(tmp_path)
    detector = OpenDetector({**OPEN_SETTINGS, "models_dir": str(tmp_path / "weights"), "device": "auto"})
    detector.load()

    assert detector.loaded
    assert seen == [(tmp_path / "weights" / "yoloe-26s-seg.pt", tmp_path)]
    assert (tmp_path / "weights").is_dir()
    detector.load()                                    # already loaded: nothing happens
    assert len(seen) == 1

    detector.set_names(["keys"])
    detector.detect(frame())
    assert model.calls[0]["device"] == "cuda"


def test_relative_models_dir_starts_at_the_project_folder(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(detector_module, "load_yoloe", lambda weights: seen.append(weights) or FakeYoloe())
    monkeypatch.chdir(tmp_path)
    OpenDetector(OPEN_SETTINGS).load()
    assert seen == [PACKAGE_DIR.parent / "models" / "yoloe-26s-seg.pt"]


def test_load_failure_gives_a_detector_error(monkeypatch, tmp_path):
    def no_internet(weights):
        raise ConnectionError("download failed")

    monkeypatch.setattr(detector_module, "load_yoloe", no_internet)
    monkeypatch.chdir(tmp_path)
    detector = OpenDetector({**OPEN_SETTINGS, "models_dir": str(tmp_path)})
    with pytest.raises(DetectorError, match="internet") as caught:
        detector.load()
    assert "yoloe-26s-seg.pt" in str(caught.value)
    assert not detector.loaded
    assert Path.cwd() == tmp_path


def test_cuda_without_a_card_gives_a_detector_error_at_load(no_graphics_card, no_real_yoloe):
    detector = OpenDetector({**OPEN_SETTINGS, "device": "cuda"})
    with pytest.raises(DetectorError, match="graphics card"):
        detector.load()


# set_names()


def test_set_names_loads_the_encoder_inside_models_dir_and_switches_back(monkeypatch, tmp_path):
    model = FakeYoloe()
    folder = tmp_path / "weights"
    monkeypatch.chdir(tmp_path)
    detector = OpenDetector({**OPEN_SETTINGS, "models_dir": str(folder)}, model=model)
    detector.set_names(["power bank", "keys"])

    assert model.text_calls == [(["power bank", "keys"], folder)]
    assert model.classes == [["power bank", "keys"]]
    assert detector.names == ("power bank", "keys")
    assert Path.cwd() == tmp_path


def test_set_names_failure_gives_a_detector_error_and_switches_back(monkeypatch, tmp_path):
    model = FakeYoloe(fail_text=True)
    monkeypatch.chdir(tmp_path)
    detector = OpenDetector({**OPEN_SETTINGS, "models_dir": str(tmp_path / "weights")}, model=model)
    with pytest.raises(DetectorError, match="internet"):
        detector.set_names(["keys"])
    assert Path.cwd() == tmp_path
    assert detector.names == ()                        # the old names stay


def test_set_names_loads_the_model_when_needed(monkeypatch):
    model = FakeYoloe()
    monkeypatch.setattr(detector_module, "load_yoloe", lambda weights: model)
    detector = OpenDetector(OPEN_SETTINGS)
    detector.set_names(["keys"])
    assert detector.loaded and model.classes == [["keys"]]


def test_same_names_change_nothing():
    model = FakeYoloe()
    detector = OpenDetector(OPEN_SETTINGS, model=model)
    detector.set_names(["keys", "cup"])
    detector.set_names(("keys", "cup"))
    assert model.classes == [["keys", "cup"]]
    assert model.resets == 1


def test_new_names_are_set_and_start_the_tracks_again():
    model = FakeYoloe()
    detector = OpenDetector(OPEN_SETTINGS, model=model)
    detector.set_names(["keys"])
    detector.set_names(["wifi router"])
    assert model.classes == [["keys"], ["wifi router"]]
    assert detector.names == ("wifi router",)
    assert model.resets == 2


def test_empty_list_clears_the_names_without_the_encoder():
    model = FakeYoloe()
    detector = OpenDetector(OPEN_SETTINGS, model=model)
    detector.set_names(["keys"])
    detector.set_names([])
    assert detector.names == ()
    assert len(model.text_calls) == 1
    assert detector.detect(frame()) == []
    assert model.calls == []


def test_clearing_names_before_load_loads_nothing(no_real_yoloe):
    detector = OpenDetector(OPEN_SETTINGS)
    detector.set_names([])
    assert not detector.loaded


@pytest.mark.parametrize("names", [["keys", "  "], ["keys", 3]])
def test_blank_or_non_text_names_are_refused(names):
    with pytest.raises(ValueError):
        OpenDetector(OPEN_SETTINGS, model=FakeYoloe()).set_names(names)


def test_one_string_instead_of_a_list_is_refused():
    with pytest.raises(TypeError):
        OpenDetector(OPEN_SETTINGS, model=FakeYoloe()).set_names("keys")


# detect()


def test_detect_with_no_names_does_not_touch_the_model():
    model = FakeYoloe(open_result([0], ids=[1]))
    detector = OpenDetector(OPEN_SETTINGS, model=model)
    assert detector.detect(frame()) == []
    assert model.calls == []


def test_detect_before_load_with_no_names_returns_nothing(no_real_yoloe):
    assert OpenDetector(OPEN_SETTINGS).detect(frame()) == []


def test_labels_are_the_typed_names_and_track_ids_pass_through():
    stale = {0: "something old", 1: "other"}            # what the model says must not win
    model = FakeYoloe(open_result([1, 0], ids=[4, 9], names=stale))
    detector = OpenDetector(OPEN_SETTINGS, model=model)
    detector.set_names(["power bank", "wifi router"])
    found = detector.detect(frame())

    assert [d.label for d in found] == ["wifi router", "power bank"]
    assert [d.track_id for d in found] == [4, 9]
    assert found[0].box == (10, 10, 50, 50)
    assert found[0].confidence == pytest.approx(0.6)


def test_untracked_named_boxes_get_no_track_id():
    model = FakeYoloe(open_result([0], ids=None))
    detector = OpenDetector(OPEN_SETTINGS, model=model)
    detector.set_names(["keys"])
    assert detector.detect(frame())[0].track_id is None


def test_detect_passes_the_open_settings_and_never_half_precision():
    model = FakeYoloe(open_result([0], ids=[1]))
    detector = OpenDetector({**OPEN_SETTINGS, "image_size": 320}, model=model)
    detector.set_names(["keys"])
    detector.detect(frame())

    call = model.calls[0]
    assert call["persist"] is True
    assert call["tracker"] == "bytetrack.yaml"
    assert call["conf"] == pytest.approx(0.25)
    assert call["imgsz"] == 320
    assert call["device"] == "cpu"
    assert call["verbose"] is False
    assert "half" not in call and "quantize" not in call


def test_masks_in_the_result_are_ignored():
    result = open_result([0], ids=[1])
    result.masks = object()                            # a segmentation model returns these
    detector = OpenDetector(OPEN_SETTINGS, model=FakeYoloe(result))
    detector.set_names(["keys"])
    assert [d.label for d in detector.detect(frame())] == ["keys"]


def test_empty_frame_with_names_raises_value_error():
    detector = OpenDetector(OPEN_SETTINGS, model=FakeYoloe())
    detector.set_names(["keys"])
    with pytest.raises(ValueError):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))


def test_reset_tracks_and_a_new_frame_size_start_the_tracks_again():
    model = FakeYoloe()
    detector = OpenDetector(OPEN_SETTINGS, model=model)
    detector.set_names(["keys"])                       # 1 reset
    detector.detect(frame(640, 480))
    detector.detect(frame(640, 480))
    assert model.resets == 1
    detector.detect(frame(640, 360))
    assert model.resets == 2
    detector.reset_tracks()
    assert model.resets == 3


def test_reset_tracks_before_load_is_fine(no_real_yoloe):
    OpenDetector(OPEN_SETTINGS).reset_tracks()
