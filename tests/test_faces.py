"""People file, embeddings file, matching and face memory, tested with made-up numbers."""

import json
from pathlib import Path

import numpy as np
import pytest

from car_vision.config import ConfigError
from car_vision.faces import (
    FaceMatcher,
    FaceMemory,
    Person,
    load_embeddings,
    load_people,
    save_embeddings,
)
from car_vision.state import Detection, RecognizedFace

ROOT = Path(__file__).resolve().parent.parent
SIZE = 8                       # real face vectors have 512 numbers; 8 is enough here


def one_hot(index: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    vector = np.zeros(SIZE, dtype=np.float32)
    vector[index] = 1.0
    if noise:
        vector += np.random.default_rng(seed).normal(0.0, noise, SIZE).astype(np.float32)
    return vector


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "people.json"
    path.write_text(text, encoding="utf-8")
    return path


# ---- load_people ------------------------------------------------------------------


def test_load_people_reads_names_roles_and_lines(tmp_path):
    path = write(tmp_path, json.dumps({
        "sara": {"name": "Sara Ahmed", "role": "Team lead", "lines": ["Built the car chassis", "Fourth-year student"]},
        "omar": {"name": "Omar Ali"},
    }))
    people = load_people(path)
    assert people["sara"] == Person("sara", "Sara Ahmed", "Team lead",
                                    ("Built the car chassis", "Fourth-year student"))
    assert people["omar"] == Person("omar", "Omar Ali", "", ())


def test_load_people_accepts_the_notepad_byte_order_mark(tmp_path):
    path = tmp_path / "people.json"
    path.write_text('{"sara": {"name": "Sara Ahmed"}}', encoding="utf-8-sig")
    assert load_people(path)["sara"].name == "Sara Ahmed"


def test_the_example_people_file_loads():
    people = load_people(ROOT / "data" / "people.example.json")
    assert len(people) == 2
    assert all(person.name for person in people.values())


def test_missing_people_file(tmp_path):
    with pytest.raises(ConfigError, match="people.json") as caught:
        load_people(tmp_path / "people.json")
    assert "not found" in str(caught.value)


def test_invalid_json_names_the_file_and_the_line(tmp_path):
    path = write(tmp_path, '{\n  "sara": {"name": "Sara Ahmed",}\n}')
    with pytest.raises(ConfigError, match=r"people\.json is not valid JSON.*line 2"):
        load_people(path)


def test_top_level_must_be_an_object(tmp_path):
    with pytest.raises(ConfigError, match="people.json must be one object"):
        load_people(write(tmp_path, '[{"name": "Sara Ahmed"}]'))


def test_entry_without_name(tmp_path):
    with pytest.raises(ConfigError, match=r"people\.json, person 'sara': 'name' is missing"):
        load_people(write(tmp_path, '{"sara": {"role": "Team lead"}}'))


@pytest.mark.parametrize("entry, field", [
    ('"Sara Ahmed"', "must be an object"),
    ('{"name": 7}', "'name' must be text"),
    ('{"name": ""}', "'name' must be text"),
    ('{"name": "Sara", "role": ["lead"]}', "'role' must be text"),
    ('{"name": "Sara", "lines": "Built the chassis"}', "'lines' must be a list of text"),
    ('{"name": "Sara", "lines": ["ok", 3]}', "'lines' must be a list of text"),
])
def test_wrong_value_types(tmp_path, entry, field):
    with pytest.raises(ConfigError, match="people.json") as caught:
        load_people(write(tmp_path, '{"sara": ' + entry + '}'))
    assert field in str(caught.value)


# ---- embeddings file ---------------------------------------------------------------


def test_embeddings_round_trip(tmp_path):
    path = tmp_path / "embeddings.npz"
    vectors = np.stack([one_hot(0), one_hot(1), one_hot(1, 0.05)])
    save_embeddings(path, ["sara", "omar", "omar"], vectors)

    ids, loaded = load_embeddings(path)
    assert ids == ["sara", "omar", "omar"]
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, vectors)
    assert not list(tmp_path.glob("*partial*"))     # the temporary file is gone


def test_missing_embeddings_file_says_to_enroll(tmp_path):
    with pytest.raises(ConfigError, match="car_vision.enroll"):
        load_embeddings(tmp_path / "embeddings.npz")


def test_mismatched_lengths_say_to_enroll(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez(path, ids=np.array(["sara", "omar"]), embeddings=np.zeros((3, SIZE), np.float32))
    with pytest.raises(ConfigError, match="2 ids but 3") as caught:
        load_embeddings(path)
    assert "car_vision.enroll" in str(caught.value)


def test_damaged_embeddings_file(tmp_path):
    path = tmp_path / "embeddings.npz"
    path.write_bytes(b"not a numpy file")
    with pytest.raises(ConfigError, match="enroll"):
        load_embeddings(path)


# ---- FaceMatcher ---------------------------------------------------------------------


def matcher(threshold=0.5):
    ids = ["sara", "sara", "omar"]
    rows = np.stack([one_hot(0), one_hot(1), one_hot(2)])
    return FaceMatcher(ids, rows, threshold)


def test_known_person_matches():
    person_id, score = matcher().match(one_hot(2, noise=0.05) * 3.0)   # length does not matter
    assert person_id == "omar"
    assert score > 0.9


def test_stranger_returns_none_with_its_score():
    person_id, score = matcher().match(one_hot(5))
    assert person_id is None
    assert score == pytest.approx(0.0, abs=1e-6)

    stranger = one_hot(5) + 0.4 * one_hot(0)
    person_id, score = matcher().match(stranger)
    assert person_id is None
    assert 0.0 < score < 0.5


def test_best_of_several_rows_per_person_wins():
    # Close to sara's second photo only, and a little like omar.
    face = one_hot(1) + 0.3 * one_hot(2)
    person_id, score = matcher().match(face)
    assert person_id == "sara"
    assert score == pytest.approx(1 / np.sqrt(1.09), abs=1e-5)


def test_exactly_at_the_threshold_matches():
    face = one_hot(0) + one_hot(4)
    _, score = matcher(threshold=0.0).match(face)
    assert matcher(threshold=score).match(face) == ("sara", score)
    assert matcher(threshold=score + 1e-4).match(face)[0] is None


def test_empty_store_is_safe():
    assert FaceMatcher([], np.zeros((0, SIZE)), 0.4).match(one_hot(0)) == (None, 0.0)
    assert FaceMatcher([], np.array([]), 0.4).match(one_hot(0)) == (None, 0.0)


def test_zero_vector_is_safe():
    assert matcher().match(np.zeros(SIZE)) == (None, 0.0)


def test_score_is_a_plain_float():
    _, score = matcher().match(one_hot(0))
    assert type(score) is float


# ---- FaceMemory --------------------------------------------------------------------


def person(track_id, box):
    return Detection("person", 0.9, box, track_id)


def face(box, track_id=None, name="Sara Ahmed", person_id="sara"):
    return RecognizedFace(name, 0.7, box, person_id, track_id)


def test_tracked_face_follows_its_person_box():
    memory = FaceMemory()
    memory.remember([face((120, 60, 160, 110), track_id=4)], [person(4, (100, 50, 200, 400))], now=0.0)

    moved = memory.current([person(4, (130, 70, 230, 420))], now=5.0)   # 30 right, 20 down
    assert len(moved) == 1
    assert moved[0].box == (150, 80, 190, 130)
    assert moved[0].name == "Sara Ahmed"
    assert moved[0].track_id == 4


def test_tracked_face_disappears_with_its_track():
    memory = FaceMemory()
    memory.remember([face((120, 60, 160, 110), track_id=4)], [person(4, (100, 50, 200, 400))], now=0.0)
    assert memory.current([person(9, (100, 50, 200, 400))], now=0.1) == []
    # It stays gone, even if a box with that number shows up again later.
    assert memory.current([person(4, (100, 50, 200, 400))], now=0.2) == []


def test_untracked_face_expires_after_max_age():
    memory = FaceMemory(max_age_seconds=1.0)
    unknown = face((10, 10, 60, 60), name="Unknown", person_id=None)
    memory.remember([unknown], [], now=10.0)
    assert memory.current([], now=10.5) == [unknown]
    assert memory.current([], now=11.0) == [unknown]
    assert memory.current([], now=11.01) == []


def test_new_remember_replaces_the_old_faces():
    memory = FaceMemory()
    memory.remember([face((10, 10, 60, 60))], [], now=0.0)
    newer = face((300, 10, 350, 60), name="Omar Ali", person_id="omar")
    memory.remember([newer], [], now=0.2)
    assert memory.current([], now=0.3) == [newer]

    memory.remember([], [], now=0.4)
    assert memory.current([], now=0.5) == []
