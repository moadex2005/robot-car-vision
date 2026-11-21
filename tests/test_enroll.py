"""Enrolment with a fake face model and tiny generated images in a temporary folder."""

import cv2
import numpy as np
import pytest

from car_vision import enroll as enroll_module
from car_vision.enroll import enroll
from car_vision.faces import load_embeddings, save_embeddings
from car_vision.recognizer import FaceModelError

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


def about(folder, text):
    (folder / "about.md").write_text(text, encoding="utf-8")


@pytest.fixture
def team(tmp_path):
    """A photo folder with a bit of everything: every sub-folder is a person."""
    photos = tmp_path / "people"
    sara = photos / "sara"
    sara.mkdir(parents=True)
    about(sara, "# Sara Ahmed\nRole: Team lead\n")
    save_image(sara / "one.png", ONE_FACE)
    save_image(sara / "two.JPG", ONE_FACE)                 # capital letters count too
    save_image(sara / "group.jpeg", CLEAR_LARGEST)
    save_image(sara / "crowd.png", NO_CLEAR_LARGEST)
    save_image(sara / "empty.png", NO_FACE)
    (sara / "broken.jpg").write_bytes(b"this is not an image")
    (sara / "notes.txt").write_text("not a photo", encoding="utf-8")
    omar = photos / "omar"                                 # photos, but no about.md
    omar.mkdir()
    save_image(omar / "blurry.png", NO_FACE)
    lina = photos / "lina"                                 # about.md, but no photos yet
    lina.mkdir()
    about(lina, "# Lina Haddad\n")
    (photos / "Not Valid").mkdir()                         # not a valid id: skipped
    save_image(photos / "Not Valid" / "x.png", ONE_FACE)

    settings = {
        "model_pack": "buffalo_l", "models_dir": "models", "detection_size": 640, "device": "cpu",
        "match_threshold": 0.42, "every_n_frames": 5, "min_face_pixels": 40,
        "photos_dir": str(photos),
        "embeddings_file": str(tmp_path / "embeddings.npz"),
    }
    return settings, tmp_path


def test_enroll_uses_only_photos_with_one_clear_face(team, capsys):
    settings, folder = team
    model = FakeFaceModel()
    used = enroll(settings, face_model=model)

    assert used == {"lina": 0, "omar": 0, "sara": 3}
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
    assert "Note: omar has no about.md" in out
    assert "lina has no about.md" not in out and "sara has no about.md" not in out
    assert "x.png" not in out
    assert out.count("Note: Skipped the folder 'Not Valid'") == 1
    assert "small letters, digits and underscores" in out
    assert "Ignored (not jpg or png): sara/notes.txt" in out


def test_enroll_prints_a_table_of_found_and_used(team, capsys):
    settings, _ = team
    enroll(settings, face_model=FakeFaceModel())
    lines = capsys.readouterr().out.splitlines()
    header = next(i for i, line in enumerate(lines) if "photos found" in line)
    assert "photos used" in lines[header]
    rows = {line.split()[0]: line.split()[1:] for line in lines[header + 1:header + 4]}
    assert rows == {"sara": ["6", "3"], "omar": ["1", "0"], "lina": ["0", "0"]}
    assert lines[header + 4].strip() == ""                 # only the three valid folders are listed


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
    assert enroll(settings, face_model=FakeFaceModel()) == {"lina": 0, "omar": 0, "sara": 0}
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


def must_not_load(_settings):
    raise AssertionError("the face model was loaded with no photos to read")


def small_settings(tmp_path):
    return {"photos_dir": str(tmp_path / "people"), "embeddings_file": str(tmp_path / "embeddings.npz")}


def test_about_but_no_photos_never_loads_the_face_model(tmp_path, monkeypatch, capsys):
    (tmp_path / "people" / "sara").mkdir(parents=True)
    about(tmp_path / "people" / "sara", "# Sara Ahmed\n")
    monkeypatch.setattr(enroll_module, "load_face_model", must_not_load)
    assert enroll(small_settings(tmp_path)) == {"sara": 0}
    out = capsys.readouterr().out
    assert "Not enrolled: sara (no photos found)" in out
    assert "Nobody was enrolled" in out


def test_missing_photo_folder_gives_a_note_and_exit_1(tmp_path, monkeypatch, capsys):
    settings = small_settings(tmp_path)
    monkeypatch.setattr(enroll_module, "load_face_model", must_not_load)
    assert enroll(settings) == {}
    assert "photo folder does not exist" in capsys.readouterr().out

    monkeypatch.setattr(enroll_module, "load_config", lambda: {"recognizer": settings})
    assert enroll_module.main([]) == 1


def test_empty_photo_folder_says_there_are_no_person_folders(tmp_path, monkeypatch, capsys):
    (tmp_path / "people").mkdir()
    monkeypatch.setattr(enroll_module, "load_face_model", must_not_load)
    assert enroll(small_settings(tmp_path)) == {}
    out = capsys.readouterr().out
    assert "no person folders" in out
    assert "photos found" not in out                       # no empty table


def test_about_in_a_wrong_encoding_still_enrols_and_prints_one_note(tmp_path, monkeypatch, capsys):
    settings = small_settings(tmp_path)
    sara = tmp_path / "people" / "sara"
    sara.mkdir(parents=True)
    (sara / "about.md").write_bytes("# Sára\n".encode("cp1252"))
    save_image(sara / "one.png", ONE_FACE)
    assert enroll(settings, face_model=FakeFaceModel()) == {"sara": 1}
    out = capsys.readouterr().out
    assert out.count("not saved as UTF-8") == 1
    assert str(sara / "about.md") in out
    assert "sara has no about.md" not in out

    monkeypatch.setattr(enroll_module, "load_config", lambda: {"recognizer": settings})
    monkeypatch.setattr(enroll_module, "load_face_model", lambda s: FakeFaceModel())
    assert enroll_module.main([]) == 0


def test_nobody_enrolled_removes_an_old_embeddings_file(tmp_path, monkeypatch, capsys):
    settings = small_settings(tmp_path)
    old = tmp_path / "embeddings.npz"
    save_embeddings(old, ["gone"], np.stack([one_hot(0)]))
    (tmp_path / "people" / "sara").mkdir(parents=True)          # no photos
    monkeypatch.setattr(enroll_module, "load_face_model", must_not_load)
    assert enroll(settings) == {"sara": 0}
    assert not old.exists()
    assert "Removed the old embeddings.npz" in capsys.readouterr().out


def test_files_in_other_formats_are_listed_once_per_person(tmp_path, capsys):
    settings = small_settings(tmp_path)
    sara = tmp_path / "people" / "sara"
    sara.mkdir(parents=True)
    about(sara, "# Sara\n")
    save_image(sara / "one.png", ONE_FACE)
    for name in ("b.webp", "a.heic", "c.BMP", "Thumbs.db", "desktop.ini", ".hidden"):
        (sara / name).write_bytes(b"x")
    omar = tmp_path / "people" / "omar"
    omar.mkdir()
    (omar / "x.jfif").write_bytes(b"x")
    enroll(settings, face_model=FakeFaceModel())
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Ignored")]
    assert lines == ["Ignored (not jpg or png): omar/x.jfif",
                     "Ignored (not jpg or png): sara/a.heic, sara/b.webp, sara/c.BMP"]


def test_a_photo_that_makes_the_model_fail_is_skipped_and_the_run_goes_on(tmp_path, capsys):
    settings = small_settings(tmp_path)
    sara = tmp_path / "people" / "sara"
    sara.mkdir(parents=True)
    save_image(sara / "a_bad.png", NO_FACE)
    save_image(sara / "b_good.png", ONE_FACE)

    class BreaksOnSomePhotos(FakeFaceModel):
        def get(self, image):
            if image.shape[1] == NO_FACE:
                raise RuntimeError("model failure")
            return super().get(image)

    assert enroll(settings, face_model=BreaksOnSomePhotos()) == {"sara": 1}
    assert "Skipped sara/a_bad.png: could not be processed" in capsys.readouterr().out


def test_main_prints_cannot_start_when_the_face_model_cannot_load(team, monkeypatch, capsys):
    settings, _ = team

    def no_model(_settings):
        raise FaceModelError("the face models could not be loaded: no network")

    monkeypatch.setattr(enroll_module, "load_config", lambda: {"recognizer": settings})
    monkeypatch.setattr(enroll_module, "load_face_model", no_model)
    assert enroll_module.main([]) == 1
    assert "Cannot start: the face models could not be loaded: no network" in capsys.readouterr().out


def test_about_with_only_a_comment_counts_as_no_about(tmp_path, capsys):
    settings = small_settings(tmp_path)
    (tmp_path / "people" / "sara").mkdir(parents=True)
    about(tmp_path / "people" / "sara", "<!-- write about Sara here -->\n")
    enroll(settings, face_model=FakeFaceModel())
    assert "Note: sara has no about.md (or an empty one)" in capsys.readouterr().out


def test_enroll_never_prints_an_embedding(team, capsys):
    settings, _ = team
    enroll(settings, face_model=FakeFaceModel())
    out = capsys.readouterr().out
    assert "[" not in out and "0.333" not in out and "1.0" not in out
