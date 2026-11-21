"""Adding, editing and removing person folders, tested in temporary folders only."""

import random
import re
from pathlib import Path

import pytest

from car_vision import people as people_module
from car_vision.faces import PERSON_ID, parse_about
from car_vision.people import (
    MAX_LINE,
    MAX_LINES,
    MAX_NAME,
    MAX_PEOPLE,
    MAX_ROLE,
    MAX_STORY,
    PeopleBook,
    PersonEntry,
    about_text,
    checked_fields,
    clean_story,
    make_id,
)

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "car_vision"


@pytest.fixture
def book(tmp_path):
    return PeopleBook(tmp_path / "people", tmp_path / "people.removed")


def fields_of(person):
    return person.name, person.role, list(person.lines), person.story


def round_trip(name, role, lines, story):
    """What faces.parse_about reads back from about_text."""
    return fields_of(parse_about("someone", about_text(name, role, lines, story)))


# ---- make_id -------------------------------------------------------------------------


@pytest.mark.parametrize("name, expected", [
    ("Sara", "sara"),
    ("Sara Ahmed", "sara_ahmed"),
    ("OMAR ALI", "omar_ali"),
    ("  Lina   Haddad  ", "lina_haddad"),
    ("Sára Ahméd", "sara_ahmed"),
    ("Anne-Marie O'Brien, Jr.", "anne_marie_obrien_jr"),
    ("2nd Driver", "2nd_driver"),
    ("007", "007"),
    ("سارة أحمد", "person"),
    ("Sara سارة", "sara"),
    ("!!!", "person"),
    ("", "person"),
    ("Photos", "photos_2"),          # "photos" is part of the page's addresses, so it is never an id
])
def test_make_id(name, expected):
    made = make_id(name, taken=())
    assert made == expected
    assert PERSON_ID.fullmatch(made)


def test_a_taken_id_gets_a_number():
    assert make_id("Sara", {"sara"}) == "sara_2"
    assert make_id("Sara", {"sara", "sara_2"}) == "sara_3"
    assert make_id("Sara", {"Sara"}) == "sara_2"              # Windows folder names ignore letter case
    assert make_id("سارة", {"person"}) == "person_2"


def test_names_windows_does_not_allow_as_folders_are_avoided():
    assert make_id("Con", ()) == "con_2"
    assert make_id("NUL", ()) == "nul_2"


# ---- about_text and the round trip ------------------------------------------------------


ORDINARY = [
    ("Sara Ahmed", "Team lead", ["Built the car chassis", "Fourth-year student"],
     "Sara joined in the first week.\n\nShe wrote the motor control code."),
    ("Omar", "", [], ""),
    ("Lina Haddad", "Driver", [], "Lina drives the car in the demo."),
    ("Yusuf", "", ["Likes robots"], ""),
    ("C#", "Language", [], ""),                                 # a name ending in # keeps it
    ("#", "", [], ""),
    ("Ana María Núñez", "Diseño", ["Señal: 5 GHz"], "Habla español."),
    ("سارة أحمد", "قائدة الفريق", ["بنت الهيكل"], "انضمت في الأسبوع الأول."),
    ("Ken", "Lead * builder", ["- starts with a dash", "* starts with a star", "Role: in a card line"],
     "-5 degrees is a line that is not a bullet.\n**bold** text stays.\n# A hash line stays in the story."),
]


@pytest.mark.parametrize("name, role, lines, story", ORDINARY)
def test_ordinary_details_come_back_exactly(name, role, lines, story):
    assert round_trip(name, role, lines, story) == (name, role, lines, story)


def test_the_file_looks_like_a_hand_written_one():
    assert about_text("Sara Ahmed", "Team lead", ["Built the car chassis"], "Joined in week one.") == (
        "# Sara Ahmed\nRole: Team lead\n\n- Built the car chassis\n\nJoined in week one.\n")


AWKWARD_STORY = """\
- not a card line
* nor this
+ nor this
• nor this
-
   - indented bullet
Role: not the role
**Role:** not the role either
role : nor this
# Not a heading
## Nor this"""


def test_story_lines_that_look_like_card_lines_or_a_role_stay_in_the_story():
    name, role, lines, story = round_trip("Sara", "Team lead", ["Real line"], AWKWARD_STORY)
    assert (name, role, lines) == ("Sara", "Team lead", ["Real line"])
    assert story == clean_story(AWKWARD_STORY)
    assert story.splitlines() == [
        "\\- not a card line", "\\* nor this", "\\+ nor this", "\\• nor this", "\\-",
        "   \\- indented bullet", "\\Role: not the role", "\\**Role:** not the role either",
        "\\role : nor this", "# Not a heading", "## Nor this"]


def test_a_role_line_in_the_story_is_not_taken_as_the_role_when_there_is_none():
    assert round_trip("Sara", "", [], "Role: boss of everyone") == ("Sara", "", [], "\\Role: boss of everyone")


def test_line_breaks_cannot_move_text_into_another_field():
    name, role, lines, story = round_trip(
        "Sara\n- injected card line\nRole: injected role",
        "Driver\r\n- another injected line",
        ["first\n- second", "third\u2028fourth"],
        "Plain story.")
    assert name == "Sara - injected card line Role: injected role"
    assert role == "Driver - another injected line"
    assert lines == ["first - second", "third fourth"]
    assert story == "Plain story."


@pytest.mark.parametrize("separator", ["\r", "\x0b", "\x0c", "\x1c", "\x85", "\u2028", "\u2029"])
def test_every_kind_of_line_break_is_removed_from_the_name(separator):
    name, role, lines, _story = round_trip(f"Sara{separator}- x", "", [], "")
    assert (name, role, lines) == ("Sara - x", "", [])


def test_html_comment_marks_are_removed_so_nothing_is_hidden():
    text = about_text("Sara <!-- hidden", "Lead -->", ["a <!-- b", "c --> d"],
                      "Before <!-- a comment --> after.\n<!<!---->-- nested")
    assert "<!--" not in text and "-->" not in text
    assert round_trip("Sara <!-- hidden", "Lead -->", ["a <!-- b", "c --> d"],
                      "Before <!-- a comment --> after.") == (
        "Sara  hidden", "Lead", ["a  b", "c  d"], "Before  a comment  after.")


def test_stars_and_spaces_around_the_role_are_dropped_as_the_reader_would():
    assert round_trip("Sara", "  **Team lead**  ", [], "") == ("Sara", "Team lead", [], "")


def test_empty_card_lines_are_left_out_and_blank_lines_in_the_story_are_kept_as_paragraphs():
    assert round_trip("Sara", "", ["", "  ", "one"], "\n\nfirst\n\n\n\nsecond\n\n") == (
        "Sara", "", ["one"], "first\n\nsecond")


def test_windows_line_ends_in_the_story_come_back_as_plain_ones():
    assert round_trip("Sara", "", [], "one\r\ntwo\r\n\r\nthree") == ("Sara", "", [], "one\ntwo\n\nthree")


def test_cleaning_twice_changes_nothing():
    once = round_trip("Sara\nx", " *Lead* ", ["a\nb"], AWKWARD_STORY)
    assert round_trip(*once) == once


def test_an_empty_name_cannot_be_written():
    with pytest.raises(ValueError, match="name"):
        about_text("  \n ", "Lead", [], "")


AWKWARD_PIECES = ["\n", "\r\n", "\r", "\u2028", "\x85", " ", "\t", "-", "*", "+", "•", "#", "Role:",
                  "**Role:**", "role :", "<!--", "-->", "<!", "--", ">", "\\", "a", "Sara", "ÿ", "س"]


def random_text(rng, pieces):
    return "".join(rng.choice(AWKWARD_PIECES) for _ in range(pieces))


def test_random_awkward_input_always_round_trips_and_keeps_every_field_apart():
    rng = random.Random(12)
    for _ in range(400):
        name = "N" + random_text(rng, 6)                       # never empty after cleaning
        role, story = random_text(rng, 6), random_text(rng, 30)
        lines = [random_text(rng, 4) for _ in range(rng.randint(0, 4))]
        cleaned = checked_fields(name, role, lines, story)
        text = about_text(name, role, lines, story)
        assert "<!--" not in text and "-->" not in text
        assert round_trip(name, role, lines, story) == (cleaned[0], cleaned[1], cleaned[2], cleaned[3])


# ---- limits ------------------------------------------------------------------------------


@pytest.mark.parametrize("change, message", [
    ({"name": ""}, "name is needed"),
    ({"name": " \n "}, "name is needed"),
    ({"name": "x" * (MAX_NAME + 1)}, "name may be at most 80"),
    ({"role": "x" * (MAX_ROLE + 1)}, "role may be at most 80"),
    ({"lines": ["x"] * (MAX_LINES + 1)}, "At most 12 card lines"),
    ({"lines": ["x" * (MAX_LINE + 1)]}, "at most 120"),
    ({"story": "x" * (MAX_STORY + 1)}, "at most 4000"),
    ({"name": 5}, "name must be text"),
    ({"role": None}, "role must be text"),
    ({"story": ["a"]}, "about text must be text"),
    ({"lines": "one line as a string"}, "list of lines"),
    ({"lines": ["ok", 3]}, "list of lines"),
])
def test_every_limit_is_a_value_error_with_a_plain_message(book, change, message):
    details = {"name": "Sara", "role": "Lead", "lines": ["one"], "story": "Story."} | change
    with pytest.raises(ValueError, match=message):
        book.save(**details)
    assert book.all() == []                                     # nothing was made


def test_the_longest_allowed_details_are_accepted(book):
    person = book.save("n" * MAX_NAME, "r" * MAX_ROLE, ["l" * MAX_LINE] * MAX_LINES, "s" * MAX_STORY)
    assert fields_of(person) == ("n" * MAX_NAME, "r" * MAX_ROLE, ["l" * MAX_LINE] * MAX_LINES, "s" * MAX_STORY)


def test_at_most_50_people(book):
    for number in range(MAX_PEOPLE):
        (book.photos_dir / f"p{number}").mkdir(parents=True)
    with pytest.raises(ValueError, match="already 50 people"):
        book.save("One Too Many")
    book.save("Changed", person_id="p3")                         # editing someone is still fine


# ---- save, all and remove ------------------------------------------------------------------


def test_save_makes_a_new_person_folder_with_an_about_file(book):
    person = book.save("Sára Ahmed", "Team lead", ["Built the chassis"], "Joined in week one.")
    assert person.person_id == "sara_ahmed"
    assert fields_of(person) == ("Sára Ahmed", "Team lead", ["Built the chassis"], "Joined in week one.")
    about = book.photos_dir / "sara_ahmed" / "about.md"
    assert about.read_text(encoding="utf-8") == person.brief
    assert sorted(p.name for p in about.parent.iterdir()) == ["about.md"]   # no temporary file left
    assert book.all() == [PersonEntry(person, 0)]


def test_the_same_name_twice_gives_two_people(book):
    first, second = book.save("Sara"), book.save("Sara")
    assert (first.person_id, second.person_id) == ("sara", "sara_2")


def test_an_existing_folder_in_other_letter_case_is_not_used_again(book):
    (book.photos_dir / "Sara").mkdir(parents=True)              # not a valid id, but the name is taken
    assert book.save("sara").person_id == "sara_2"


def test_save_with_an_id_rewrites_the_about_file_and_leaves_the_photos(book):
    book.save("Sara", "Driver")
    photo = book.photos_dir / "sara" / "photo_1.jpg"
    photo.write_bytes(b"photo bytes")
    person = book.save("Sara Ahmed", "Team lead", ["New line"], "", person_id="sara")
    assert fields_of(person) == ("Sara Ahmed", "Team lead", ["New line"], "")
    assert photo.read_bytes() == b"photo bytes"
    assert book.all() == [PersonEntry(person, 1)]


def test_save_with_an_unknown_id_is_a_key_error(book):
    with pytest.raises(KeyError):
        book.save("Sara", person_id="nobody")
    assert not (book.photos_dir / "nobody").exists()


def test_all_is_sorted_by_id_counts_photos_and_skips_invalid_folders(book):
    for person_id in ("zoe", "adam", "Not Valid"):
        (book.photos_dir / person_id).mkdir(parents=True)
    for name in ("a.jpg", "b.JPEG", "c.png", "d.webp", "notes.txt"):
        (book.photos_dir / "zoe" / name).write_bytes(b"x")
    (book.photos_dir / "adam" / "about.md").write_text("# Adam Smith\n", encoding="utf-8")
    entries = book.all()
    assert [(e.person.person_id, e.person.name, e.photos) for e in entries] == [
        ("adam", "Adam Smith", 0), ("zoe", "zoe", 3)]


def test_all_with_no_folder_yet_is_empty(book):
    assert book.all() == []


def test_remove_moves_the_folder_aside_and_deletes_nothing(book):
    book.save("Sara", "Lead")
    (book.photos_dir / "sara" / "photo_1.jpg").write_bytes(b"photo bytes")
    book.remove("sara")
    assert not (book.photos_dir / "sara").exists()
    [moved] = list(book.removed_dir.iterdir())
    assert re.fullmatch(r"sara-\d{8}-\d{6}", moved.name)
    assert (moved / "photo_1.jpg").read_bytes() == b"photo bytes"
    assert (moved / "about.md").is_file()
    assert book.all() == []


def test_a_second_person_with_the_same_id_removed_later_does_not_collide(book, monkeypatch):
    class SameSecond:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime(2026, 11, 21, 10, 30, 0)

    monkeypatch.setattr(people_module, "datetime", SameSecond)
    book.save("Sara", "First")
    book.remove("sara")
    assert book.save("Sara", "Second").person_id == "sara"     # the id is free again
    book.remove("sara")
    moved = sorted(p.name for p in book.removed_dir.iterdir())
    assert moved == ["sara-20261121-103000", "sara-20261121-103000-2"]
    roles = sorted(parse_about("x", (book.removed_dir / name / "about.md").read_text(encoding="utf-8")).role
                   for name in moved)
    assert roles == ["First", "Second"]


def test_remove_an_unknown_id_is_a_key_error(book):
    with pytest.raises(KeyError):
        book.remove("nobody")
    with pytest.raises(KeyError):
        book.remove(None)


HOSTILE_IDS = ["..", "../outside", "sara/../../outside", "..\\outside", "C:\\outside", "/outside",
               "", " ", "sara\n", "Sara", "sara.md", "sara ", 5, ["sara"]]   # None means a new person


@pytest.mark.parametrize("person_id", HOSTILE_IDS)
def test_hostile_ids_never_reach_outside_the_photos_folder(book, tmp_path, person_id):
    book.save("Sara", "Lead")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "about.md").write_text("# Keep me\n", encoding="utf-8")
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    with pytest.raises(KeyError):
        book.save("Mallory", person_id=person_id)
    with pytest.raises(KeyError):
        book.remove(person_id)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    assert (outside / "about.md").read_text(encoding="utf-8") == "# Keep me\n"


def test_a_name_never_builds_a_path_outside_the_photos_folder(book, tmp_path):
    for name in ("../../outside", "..\\..\\x", "C:\\Windows", "/etc/passwd", ".."):
        person = book.save(name)
        assert (book.photos_dir / person.person_id).parent == book.photos_dir
    assert sorted(p.parent for p in tmp_path.rglob("about.md")) == sorted(
        book.photos_dir / e.person.person_id for e in book.all())


# ---- which libraries people.py uses ------------------------------------------------------------


def test_people_uses_only_the_standard_library_and_faces():
    import ast
    import sys

    tree = ast.parse((PACKAGE_DIR / "people.py").read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    outside = {name for name in found if name.split(".")[0] not in sys.stdlib_module_names}
    assert outside == {"car_vision.faces"}
