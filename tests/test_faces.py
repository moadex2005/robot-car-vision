"""Person folders, embeddings file, matching and face memory, tested with made-up data."""

import logging
from pathlib import Path

import numpy as np
import pytest

from car_vision.config import ConfigError
from car_vision.faces import (
    TRACK_GAP_SECONDS,
    FaceMatcher,
    FaceMemory,
    Person,
    has_about_text,
    load_embeddings,
    load_people,
    parse_about,
    photos_in,
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


def person_folder(photos_dir: Path, person_id: str, about: str | bytes | None = None) -> Path:
    """Make photos_dir/person_id/, with an about.md when one is given."""
    folder = photos_dir / person_id
    folder.mkdir(parents=True)
    if isinstance(about, str):
        (folder / "about.md").write_text(about, encoding="utf-8")
    elif isinstance(about, bytes):
        (folder / "about.md").write_bytes(about)
    return folder


SARA_ABOUT = """\
# Sara Ahmed
Role: Team lead

- Built the car chassis
- Fourth-year student

Sara joined the project in the first week and wrote the motor control code.
"""


# ---- parse_about ------------------------------------------------------------------


def test_parse_about_reads_name_role_lines_story_and_brief():
    person = parse_about("sara", SARA_ABOUT)
    assert person == Person(
        "sara", "Sara Ahmed", "Team lead",
        ("Built the car chassis", "Fourth-year student"),
        "Sara joined the project in the first week and wrote the motor control code.",
        SARA_ABOUT,
    )


def test_no_heading_gives_the_folder_name():
    person = parse_about("sara", "Role: Team lead\n- Built the chassis\n")
    assert person.name == "sara"
    assert person.role == "Team lead"


def test_no_role_and_no_bullets():
    person = parse_about("sara", "# Sara Ahmed\n\nJust a story.\n")
    assert (person.role, person.lines, person.story) == ("", (), "Just a story.")


def test_role_in_any_letter_case():
    assert parse_about("sara", "ROLE:  Driver \n").role == "Driver"
    assert parse_about("sara", "role: Driver\n").role == "Driver"


def test_star_and_dash_bullets_all_kept_in_order():
    text = "# Sara\n- one\n* two\n- three\n  - four\n* five\n"
    assert parse_about("sara", text).lines == ("one", "two", "three", "four", "five")


def test_story_keeps_paragraphs_and_has_no_blank_ends():
    text = ("\n\n# Sara\n\nFirst paragraph,\nstill the first.\n\n- a bullet\n\n\n"
            "Second paragraph.\nRole: Team lead\n\n\n")
    assert parse_about("sara", text).story == "First paragraph,\nstill the first.\n\nSecond paragraph."


def test_only_the_first_heading_is_the_name():
    person = parse_about("sara", "# Sara Ahmed\n# Later heading\n## Hobbies\nChess.\n")
    assert person.name == "Sara Ahmed"
    assert person.story == "# Later heading\n## Hobbies\nChess."


def test_html_comments_are_left_out_of_lines_and_story_but_kept_in_brief():
    text = ("<!-- a note\nfor the team -->\n# Sara\n- Built the chassis <!-- check -->\n"
            "<!-- - hidden bullet -->\nStory <!-- secret --> here.\n")
    person = parse_about("sara", text)
    assert person.name == "Sara"
    assert person.lines == ("Built the chassis",)
    assert person.story == "Story  here."
    assert person.brief == text


def test_heading_loses_its_closing_hashes():
    assert parse_about("sara", "# Sara Ahmed ##\n").name == "Sara Ahmed"
    assert parse_about("sara", "# ##\n").name == "sara"


@pytest.mark.parametrize("line", ["Role: Team lead", "Role : Team lead", "**Role:** Team lead",
                                  "**Role**: Team lead", "ROLE:Team lead"])
def test_role_line_forms(line):
    person = parse_about("sara", f"# Sara\n{line}\n")
    assert person.role == "Team lead"
    assert person.story == ""


def test_only_the_first_role_line_is_the_role_a_later_one_stays_in_the_story():
    person = parse_about("sara", "Role: Team lead\nRole: Driver in 2025\n")
    assert person.role == "Team lead"
    assert person.story == "Role: Driver in 2025"


def test_tab_plus_and_bullet_sign_bullets():
    text = "-\tone\n+ two\n• three\n•\tfour\n-\n-5 degrees is cold\n---\n"
    person = parse_about("sara", text)
    assert person.lines == ("one", "two", "three", "four")
    assert person.story == "-5 degrees is cold\n---"


def test_has_about_text_ignores_comments():
    assert not has_about_text(parse_about("sara", "<!-- only a comment -->\n\n"))
    assert not has_about_text(Person("sara", "sara"))
    assert has_about_text(parse_about("sara", "<!-- note -->\n# Sara\n"))


def test_empty_about_gives_just_the_folder_name():
    assert parse_about("sara", "") == Person("sara", "sara")


def test_windows_line_endings():
    person = parse_about("sara", "# Sara\r\nRole: Driver\r\n- one\r\n\r\nStory.\r\n")
    assert (person.name, person.role, person.lines, person.story) == ("Sara", "Driver", ("one",), "Story.")


# ---- load_people ------------------------------------------------------------------


def test_photos_in_lists_jpg_and_png_in_any_letter_case_in_name_order(tmp_path):
    for name in ("b.PNG", "a.jpg", "c.JPEG", "d.webp", "about.md"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "e.jpg").mkdir()                            # a folder is not a photo
    assert [p.name for p in photos_in(tmp_path)] == ["a.jpg", "b.PNG", "c.JPEG"]
    assert photos_in(tmp_path / "missing") == []


def test_load_people_makes_one_person_per_folder_in_name_order(tmp_path):
    person_folder(tmp_path, "sara", SARA_ABOUT)
    person_folder(tmp_path, "omar")                         # photos only, no about.md
    (tmp_path / "readme.txt").write_text("not a folder", encoding="utf-8")
    people = load_people(tmp_path)
    assert list(people) == ["omar", "sara"]
    assert people["sara"].name == "Sara Ahmed"
    assert people["omar"] == Person("omar", "omar")


def test_more_than_three_bullets_are_all_kept(tmp_path):
    person_folder(tmp_path, "sara", "# Sara\n- one\n- two\n- three\n- four\n- five\n")
    assert load_people(tmp_path)["sara"].lines == ("one", "two", "three", "four", "five")


def test_byte_order_mark_is_accepted(tmp_path):
    person_folder(tmp_path, "sara", "﻿# Sara Ahmed\n- one\n".encode("utf-8"))
    person = load_people(tmp_path)["sara"]
    assert person.name == "Sara Ahmed"
    assert not person.brief.startswith("﻿")


def test_wrong_encoding_keeps_that_person_by_folder_name_and_reports_it_once(tmp_path):
    person_folder(tmp_path, "sara", "# Sára Ahmed\n- one\n".encode("cp1252"))
    person_folder(tmp_path, "omar", "# Omar Ali\n")
    problems = {}
    people = load_people(tmp_path, problems)
    assert people["sara"] == Person("sara", "sara")             # no role, lines, story or brief
    assert people["omar"].name == "Omar Ali"                    # the others load normally
    assert list(problems) == ["sara"]
    assert "not saved as UTF-8" in problems["sara"]
    assert str(Path("sara") / "about.md") in problems["sara"]


def test_wrong_encoding_without_a_problems_dict_gives_one_warning(tmp_path, caplog):
    person_folder(tmp_path, "sara", "# Sára Ahmed\n".encode("cp1252"))
    with caplog.at_level(logging.WARNING, logger="car_vision.faces"):
        people = load_people(tmp_path)
    assert people == {"sara": Person("sara", "sara")}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "about.md" in warnings[0] and "UTF-8" in warnings[0]


def test_invalid_folder_names_are_skipped_with_one_warning_each(tmp_path, caplog):
    person_folder(tmp_path, "sara", SARA_ABOUT)
    for bad in ("Omar", "lina haddad", "nora-f"):
        person_folder(tmp_path, bad)
    with caplog.at_level(logging.WARNING, logger="car_vision.faces"):
        people = load_people(tmp_path)
    assert list(people) == ["sara"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3
    assert all(any(bad in message for message in warnings) for bad in ("Omar", "lina haddad", "nora-f"))


def test_invalid_folder_names_go_into_the_problems_dict_with_the_rule(tmp_path):
    person_folder(tmp_path, "sara", SARA_ABOUT)
    person_folder(tmp_path, "Luran")
    person_folder(tmp_path, "sara-ahmed")
    problems = {}
    assert list(load_people(tmp_path, problems)) == ["sara"]
    assert sorted(problems) == ["Luran", "sara-ahmed"]
    assert all("small letters, digits and underscores" in message for message in problems.values())
    assert "'Luran'" in problems["Luran"]


def test_missing_or_empty_photo_folder_gives_nobody(tmp_path):
    assert load_people(tmp_path / "does-not-exist") == {}
    assert load_people(tmp_path) == {}


def test_the_example_about_file_parses():
    text = (ROOT / "data" / "people.example" / "about.md").read_text(encoding="utf-8-sig")
    person = parse_about("nora", text)
    assert person.name == "Nora Fielding"
    assert person.role and len(person.lines) > 3 and person.story
    assert "<!--" not in person.story
    assert not list((ROOT / "data" / "people.example").glob("*.jpg"))      # never enrolled


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


def test_one_dimensional_embeddings_say_to_enroll(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez(path, ids=np.array(["sara"]), embeddings=np.zeros(1, np.float32))
    with pytest.raises(ConfigError, match="one face vector per row") as caught:
        load_embeddings(path)
    assert "car_vision.enroll" in str(caught.value)


def test_vectors_of_the_wrong_size_say_to_enroll(tmp_path):
    path = tmp_path / "embeddings.npz"
    save_embeddings(path, ["sara"], np.stack([one_hot(0)]))
    assert load_embeddings(path, vector_size=SIZE)[0] == ["sara"]
    with pytest.raises(ConfigError, match=f"{SIZE} numbers, but the face model makes 512"):
        load_embeddings(path, vector_size=512)


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


def test_tracked_face_disappears_when_its_track_is_gone_too_long():
    memory = FaceMemory()
    memory.remember([face((120, 60, 160, 110), track_id=4)], [person(4, (100, 50, 200, 400))], now=0.0)
    assert len(memory.current([person(9, (100, 50, 200, 400))], now=0.1)) == 1     # missing since 0.1
    assert len(memory.current([], now=0.1 + TRACK_GAP_SECONDS - 0.01)) == 1         # still within the gap
    assert memory.current([], now=0.1 + TRACK_GAP_SECONDS + 0.01) == []
    # It stays gone, even if a box with that number shows up again later.
    assert memory.current([person(4, (100, 50, 200, 400))], now=0.5) == []


def test_tracked_face_waits_where_it_was_and_follows_the_track_when_it_comes_back():
    memory = FaceMemory()
    memory.remember([face((120, 60, 160, 110), track_id=4)], [person(4, (100, 50, 200, 400))], now=0.0)
    assert memory.current([person(4, (110, 50, 210, 400))], now=0.05)[0].box == (130, 60, 170, 110)
    # One frame without the person box: the face stays at its last place.
    assert memory.current([], now=0.1)[0].box == (130, 60, 170, 110)
    # The box is back within the gap: the face follows it again.
    assert memory.current([person(4, (130, 70, 230, 420))], now=0.3)[0].box == (150, 80, 190, 130)
    # And a new gap starts from zero.
    assert len(memory.current([], now=0.35)) == 1
    assert len(memory.current([], now=0.35 + TRACK_GAP_SECONDS - 0.01)) == 1


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
