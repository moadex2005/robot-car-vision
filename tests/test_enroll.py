"""Enrolment with a fake face model and tiny generated images in a temporary folder."""

import json

import cv2
import numpy as np
import pytest

from car_vision import enroll as enroll_module
from car_vision.config import ConfigError
from car_vision.enroll import enroll
from car_vision.faces import load_embeddings

SIZE = 8


def one_hot(index):
    vector = np.zeros(SIZE, dtype=np.float32)
    vector[index] = 3.0                      # not unit length: enroll must normalise
    return vector


class FakeFace:
    def __init__(self, box, embedding):
        self.bbox = np.asarray(box, dtype=np.float32)
        self.normed_embedding = np.asarray(embedding, dtype=np.float32)


# The fake model decides what it "sees" from the image width, which survives jpg compression.
NO_FACE, ONE_FACE, CLEAR_LARGEST, NO_CLEAR_LARGEST = 20, 21, 22, 23
SCENES = {
    NO_FACE: [],
    ONE_FACE: [FakeFace((0, 0, 50, 50), one_hot(0))],
    CLEAR_LARGEST: [FakeFace((0, 0, 30, 30), one_hot(2)), FakeFace((0, 0, 100, 100), one_hot(1))],
    NO_CLEAR_LARGEST: [FakeFace((0, 0, 90, 90), one_hot(3)), FakeFace((0, 0, 100, 100), one_hot(4))],
}


class FakeFaceModel:
    def __init__(self):
        self.calls = 0

    def get(self, image):
        self.calls += 1
        return list(SCENES[image.shape[1]])


def save_image(path, width):
    ok, data = cv2.imencode(path.suffix.lower() if path.suffix.lower() != ".jpeg" else ".jpg",
                            np.full((16, width, 3), 128, dtype=np.uint8))
    assert ok
    path.write_bytes(data.tobytes())


@pytest.fixture
def team(tmp_path):
    """people.json with three people and a photo folder with a bit of everything."""
    people = {
        "sara": {"name": "Sara Ahmed"},
        "omar": {"name": "Omar Ali"},
        "lina": {"name": "Lina Haddad"},          # has no folder at all
    }
    (tmp_path / "people.json").write_text(json.dumps(people), encoding="utf-8")
    photos = tmp_path / "people"
    sara = photos / "sara"
    sara.mkdir(parents=True)
    save_image(sara / "one.png", ONE_FACE)
    save_image(sara / "two.JPG", ONE_FACE)                 # capital letters count too
    save_image(sara / "group.jpeg", CLEAR_LARGEST)
    save_image(sara / "crowd.png", NO_CLEAR_LARGEST)
    save_image(sara / "empty.png", NO_FACE)
    (sara / "broken.jpg").write_bytes(b"this is not an image")
    (sara / "notes.txt").write_text("not a photo", encoding="utf-8")
    omar = photos / "omar"
    omar.mkdir()
    save_image(omar / "blurry.png", NO_FACE)
    (photos / "stranger").mkdir()                          # not in people.json
    save_image(photos / "stranger" / "x.png", ONE_FACE)

    settings = {
        "model_pack": "buffalo_l", "models_dir": "models", "detection_size": 640, "device": "cpu",
        "match_threshold": 0.42, "every_n_frames": 5, "min_face_pixels": 40,
        "people_file": str(tmp_path / "people.json"),
        "photos_dir": str(photos),
        "embeddings_file": str(tmp_path / "embeddings.npz"),
    }
    return settings, tmp_path


def test_enroll_uses_only_photos_with_one_clear_face(team, capsys):
    settings, folder = team
    model = FakeFaceModel()
    used = enroll(settings, face_model=model)

    assert used == {"sara": 3, "omar": 0, "lina": 0}
    assert model.calls == 6                                # broken.jpg and notes.txt never reach it
    ids, vectors = load_embeddings(folder / "embeddings.npz")
    assert ids == ["sara", "sara", "sara"]
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
    # Photos are read in name order, so the first one used is group.jpeg: its largest face.
    np.testing.assert_allclose(vectors[0], one_hot(1) / 3.0)

    out = capsys.readouterr().out
    assert "Skipped sara/broken.jpg: unreadable" in out
    assert "Skipped sara/empty.png: no face" in out
    assert "Skipped sara/crowd.png: several faces" in out
    assert "Skipped omar/blurry.png: no face" in out
    assert "group.jpeg" not in out and "one.png" not in out
    assert "Warning: sara has only 3" not in out
    assert "Not enrolled: omar (no usable photo)" in out
    assert "Not enrolled: lina (no photos found)" in out
    assert "Ignored folders (no entry in the people file): stranger" in out
    assert "stranger/x.png" not in out


def test_enroll_prints_a_table_of_found_and_used(team, capsys):
    settings, _ = team
    enroll(settings, face_model=FakeFaceModel())
    lines = capsys.readouterr().out.splitlines()
    header = next(i for i, line in enumerate(lines) if "photos found" in line)
    assert "photos used" in lines[header]
    rows = {line.split()[0]: line.split()[1:] for line in lines[header + 1:header + 4]}
    assert rows == {"sara": ["6", "3"], "omar": ["1", "0"], "lina": ["0", "0"]}


def test_fewer_than_three_photos_gives_a_warning(team, capsys):
    settings, folder = team
    for name in ("group.jpeg", "two.JPG"):
        (folder / "people" / "sara" / name).unlink()
    assert enroll(settings, face_model=FakeFaceModel())["sara"] == 1
    assert "Warning: sara has only 1 usable photo" in capsys.readouterr().out


def test_nobody_enrolled_saves_nothing_and_main_exits_1(team, capsys, monkeypatch):
    settings, folder = team
    for photo in (folder / "people" / "sara").iterdir():
        photo.unlink()
    assert enroll(settings, face_model=FakeFaceModel()) == {"sara": 0, "omar": 0, "lina": 0}
    assert not (folder / "embeddings.npz").exists()
    assert "Nobody was enrolled" in capsys.readouterr().out

    monkeypatch.setattr(enroll_module, "load_config", lambda: {"recognizer": settings})
    monkeypatch.setattr(enroll_module, "load_face_model", lambda s: FakeFaceModel())
    assert enroll_module.main([]) == 1


def test_main_exits_0_when_someone_is_enrolled(team, monkeypatch):
    settings, _ = team
    monkeypatch.setattr(enroll_module, "load_config", lambda: {"recognizer": settings})
    monkeypatch.setattr(enroll_module, "load_face_model", lambda s: FakeFaceModel())
    assert enroll_module.main([]) == 0


def test_no_photos_at_all_never_loads_the_face_model(tmp_path, monkeypatch, capsys):
    (tmp_path / "people.json").write_text('{"sara": {"name": "Sara Ahmed"}}', encoding="utf-8")
    settings = {"people_file": str(tmp_path / "people.json"), "photos_dir": str(tmp_path / "people"),
                "embeddings_file": str(tmp_path / "embeddings.npz")}

    def must_not_load(_settings):
        raise AssertionError("the face model was loaded with no photos to read")

    monkeypatch.setattr(enroll_module, "load_face_model", must_not_load)
    assert enroll(settings) == {"sara": 0}
    out = capsys.readouterr().out
    assert "photo folder does not exist" in out
    assert "Not enrolled: sara (no photos found)" in out


def test_missing_people_file_is_a_readable_error_and_exit_1(tmp_path, monkeypatch, capsys):
    settings = {"people_file": str(tmp_path / "people.json"), "photos_dir": str(tmp_path / "people"),
                "embeddings_file": str(tmp_path / "embeddings.npz")}
    with pytest.raises(ConfigError, match="People file not found"):
        enroll(settings, face_model=FakeFaceModel())

    monkeypatch.setattr(enroll_module, "load_config", lambda: {"recognizer": settings})
    assert enroll_module.main([]) == 1
    assert "Cannot enroll: People file not found" in capsys.readouterr().out


def test_enroll_never_prints_an_embedding(team, capsys):
    settings, _ = team
    enroll(settings, face_model=FakeFaceModel())
    out = capsys.readouterr().out
    assert "[" not in out and "0.333" not in out and "1.0" not in out
