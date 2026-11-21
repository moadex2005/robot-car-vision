"""The M12 routes and page parts: people from the page, the notes editor, describe the scene.

The people folder is a real PeopleBook in a temporary folder; the pipeline and the
assistant are fakes that record what they were asked, in order.
"""

import re

import pytest

from car_vision import app as app_module
from car_vision.app import (MAX_NOTES_CHARS, PhotoWatch, UnavailableAssistant, build_people_book, create_app,
                            notes_path)
from car_vision.assistant import Answer, AssistantError
from car_vision.config import resolve_path
from car_vision.people import PeopleBook

SCENE_JPEG = b"\xff\xd8 the newest clean frame \xff\xd9"


class FakePipeline:
    def __init__(self, recognition=True, enrolled=(), photo=SCENE_JPEG, calls=None):
        self.recognition = recognition
        self.enrolled = set(enrolled)
        self.photo = photo
        self.calls = calls if calls is not None else []     # shared with the assistant: the order of effects
        self.progress = None
        self.start_error = None
        self.reload_error = None
        self.photos_dir = None

    def status(self):
        return {"recognition": self.recognition}

    def enrolled_ids(self):
        return set(self.enrolled)

    def reload_people(self):
        self.calls.append("reload_people")
        if self.reload_error is not None:
            raise self.reload_error
        return []

    def start_photos(self, person_id, count=5):
        if self.photos_dir is not None and not (self.photos_dir / person_id).is_dir():
            raise KeyError(person_id)                           # as the real pipeline: no such folder
        self.calls.append(("start_photos", person_id, count))
        if self.start_error is not None:
            raise self.start_error
        self.progress = {"person_id": person_id, "taken": 0, "wanted": count, "state": "taking",
                         "hint": "Stand in front of the camera", "notes": []}

    def photo_progress(self):
        return self.progress

    def cancel_photos(self):
        self.calls.append("cancel_photos")
        if self.progress is not None and self.progress["state"] == "taking":
            self.progress = {**self.progress, "state": "failed", "hint": "Cancelled"}

    def scene_photo(self, max_side=960):
        self.calls.append("scene_photo")
        return self.photo


class FakeAssistant:
    def __init__(self, calls, reload_error=None, scene_error=None, people_dir=None):
        self.calls = calls
        self.reload_error = reload_error
        self.scene_error = scene_error
        self.people_dir = people_dir
        self.seen_on_reload = []            # the person folders that existed when reload ran
        self.scenes = []

    def reload(self):
        self.calls.append("assistant.reload")
        if self.people_dir is not None:
            self.seen_on_reload.append(sorted(p.name for p in self.people_dir.iterdir() if p.is_dir()))
        if self.reload_error is not None:
            raise self.reload_error

    def describe_scene(self, image):
        self.scenes.append(image)
        if self.scene_error is not None:
            raise self.scene_error
        return Answer("A desk with a laptop. Sara Ahmed is standing beside it.", False, ())


@pytest.fixture
def world(tmp_path):
    """A temporary people folder with Sara (2 photos) and Omar (none), a notes file, fakes."""
    photos_dir = tmp_path / "people"
    sara = photos_dir / "sara"
    sara.mkdir(parents=True)
    (sara / "about.md").write_text("# Sara Ahmed\nRole: Team lead\n\n- Built the chassis\n\nShe plays chess.\n",
                                   encoding="utf-8")
    (sara / "photo_1.jpg").write_bytes(b"not read here")
    (sara / "photo_2.jpg").write_bytes(b"not read here")
    omar = photos_dir / "omar"
    omar.mkdir()
    (omar / "about.md").write_text("# Omar\n", encoding="utf-8")
    calls = []
    pipeline = FakePipeline(enrolled={"sara"}, calls=calls)
    pipeline.photos_dir = photos_dir
    assistant = FakeAssistant(calls, people_dir=photos_dir)
    notes = tmp_path / "knowledge.md"
    notes.write_text("# Notes\n[REPLACE] the team\n", encoding="utf-8")
    book = PeopleBook(photos_dir, tmp_path / "people.removed")
    app = create_app(pipeline, assistant, people=book, notes_file=notes)
    app.testing = True
    return {"client": app.test_client(), "pipeline": pipeline, "assistant": assistant, "calls": calls,
            "photos_dir": photos_dir, "removed_dir": tmp_path / "people.removed", "notes": notes, "tmp": tmp_path}


def bare_client(**kwargs):
    calls = []
    app = create_app(FakePipeline(calls=calls), FakeAssistant(calls), **kwargs)
    app.testing = True
    return app.test_client()


# ---- without a people folder or a notes file -----------------------------------------------


@pytest.mark.parametrize("method, url", [
    ("get", "/api/people"), ("post", "/api/people"), ("put", "/api/people/sara"), ("delete", "/api/people/sara"),
    ("post", "/api/people/sara/photos"), ("get", "/api/people/photos"), ("delete", "/api/people/photos"),
])
def test_people_routes_answer_a_plain_error_without_a_people_book(method, url):
    response = getattr(bare_client(), method)(url, json={"name": "X"})
    assert response.status_code == 503
    assert "recognizer.photos_dir" in response.get_json()["error"]


@pytest.mark.parametrize("method", ["get", "put"])
def test_notes_routes_answer_a_plain_error_without_a_notes_file(method):
    response = getattr(bare_client(), method)("/api/notes", json={"text": "x"})
    assert response.status_code == 503
    assert "assistant.notes_file" in response.get_json()["error"]


def test_the_page_hides_people_and_notes_when_the_server_has_neither():
    page = bare_client().get("/").get_data(as_text=True)
    assert '<section class="section" id="people" aria-label="People" hidden>' in page
    assert '<section class="section" id="knowledge" aria-label="Notes" hidden>' in page
    assert '<a href="#people" id="peopleLink" hidden>' in page and '<a href="#knowledge" id="notesLink" hidden>' in page


def test_the_page_shows_people_and_notes_when_the_server_has_both(world):
    page = world["client"].get("/").get_data(as_text=True)
    assert '<section class="section" id="people" aria-label="People">' in page
    assert '<section class="section" id="knowledge" aria-label="Notes">' in page
    assert '<a href="#people" id="peopleLink">People</a>' in page


# ---- the list -----------------------------------------------------------------------------


def test_the_list_gives_details_and_photo_counts_never_pictures(world):
    response = world["client"].get("/api/people")
    data = response.get_json()
    assert data["recognition"] is True
    assert data["people"] == [
        {"id": "omar", "name": "Omar", "role": "", "lines": [], "story": "", "photos": 0, "enrolled": False},
        {"id": "sara", "name": "Sara Ahmed", "role": "Team lead", "lines": ["Built the chassis"],
         "story": "She plays chess.", "photos": 2, "enrolled": True},
    ]
    assert b"data:image" not in response.data and b"base64" not in response.data


def test_the_list_says_when_recognition_is_off(world):
    world["pipeline"].recognition = False
    assert world["client"].get("/api/people").get_json()["recognition"] is False


# ---- adding and changing --------------------------------------------------------------------


def test_adding_a_person_writes_the_folder_then_reloads_the_assistant(world):
    response = world["client"].post("/api/people", json={"name": "Lina Haddad", "role": "Driver",
                                                          "lines": ["Drives the car", ""], "story": "Hi."})
    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "id": "lina_haddad"}
    about = (world["photos_dir"] / "lina_haddad" / "about.md").read_text(encoding="utf-8")
    assert "# Lina Haddad" in about and "Role: Driver" in about and "- Drives the car" in about
    assert world["calls"] == ["assistant.reload"]                      # no enrolment for a new person
    assert "lina_haddad" in world["assistant"].seen_on_reload[0]      # the file came first


@pytest.mark.parametrize("body, message", [
    ({"name": ""}, "The name is needed."),
    ({"name": "x" * 81}, "The name may be at most 80 characters long."),
    ({"name": "Lina", "lines": "one line"}, "The card lines must be a list"),
    ({"name": "Lina", "lines": ["x"] * 13}, "At most 12 card lines"),
    ({"name": "Lina", "story": "x" * 4001}, "The about text may be at most 4000 characters long."),
    ({"name": None}, "The name must be text."),
])
def test_bad_details_are_refused_with_the_books_message_and_nothing_changes(world, body, message):
    before = sorted(p.name for p in world["photos_dir"].iterdir())
    response = world["client"].post("/api/people", json=body)
    assert response.status_code == 400
    assert message in response.get_json()["error"]
    assert sorted(p.name for p in world["photos_dir"].iterdir()) == before
    assert world["calls"] == []


def test_a_body_that_is_not_json_is_refused(world):
    response = world["client"].post("/api/people", data="name=Lina", content_type="text/plain")
    assert response.status_code == 400 and "Send the person as JSON" in response.get_json()["error"]


def test_changing_a_person_rewrites_about_md_and_keeps_the_photos(world):
    response = world["client"].put("/api/people/sara", json={"name": "Sara Ahmed", "role": "Captain",
                                                             "lines": [], "story": ""})
    assert response.status_code == 200
    about = (world["photos_dir"] / "sara" / "about.md").read_text(encoding="utf-8")
    assert "Role: Captain" in about and "Built the chassis" not in about
    assert sorted(p.name for p in (world["photos_dir"] / "sara").glob("*.jpg")) == ["photo_1.jpg", "photo_2.jpg"]
    # The camera side reads the new name and role too, so the label on the video changes at once.
    assert world["calls"] == ["reload_people", "assistant.reload"]


@pytest.mark.parametrize("person_id", ["nobody", "..", "Sara", "sara%20x", "%2e%2e", "a" * 101, "con."])
def test_an_unknown_or_hostile_id_is_404_and_touches_nothing(world, person_id):
    client = world["client"]
    before = sorted(str(p) for p in world["tmp"].rglob("*"))
    for response in (client.put(f"/api/people/{person_id}", json={"name": "X"}),
                     client.delete(f"/api/people/{person_id}"),
                     client.post(f"/api/people/{person_id}/photos", json={})):
        assert response.status_code == 404, (person_id, response.status_code)
        assert response.get_json()["error"]
    assert sorted(str(p) for p in world["tmp"].rglob("*")) == before
    assert world["calls"] == []


def test_a_failing_assistant_reload_still_answers_ok_with_a_warning(world):
    world["assistant"].reload_error = AssistantError("The notes file was not found.")
    data = world["client"].post("/api/people", json={"name": "Lina"}).get_json()
    assert data["ok"] is True and "The notes file was not found." in data["warning"]
    assert (world["photos_dir"] / "lina" / "about.md").exists()          # the change did happen


# ---- removing ---------------------------------------------------------------------------------


def test_removing_moves_the_folder_aside_then_enrols_then_reloads(world):
    response = world["client"].delete("/api/people/sara")
    assert response.status_code == 200 and response.get_json() == {"ok": True}
    assert not (world["photos_dir"] / "sara").exists()
    moved, = world["removed_dir"].iterdir()
    assert moved.name.startswith("sara-") and (moved / "photo_1.jpg").exists()   # nothing deleted
    assert world["calls"] == ["reload_people", "assistant.reload"]
    assert "sara" not in world["assistant"].seen_on_reload[0]


def test_removing_with_recognition_off_skips_enrolment_without_a_warning(world):
    world["pipeline"].reload_error = RuntimeError("Face recognition is off.")
    data = world["client"].delete("/api/people/sara").get_json()
    assert data == {"ok": True}


def test_a_failing_enrolment_after_a_removal_is_a_warning(world):
    world["pipeline"].reload_error = OSError("disk full")
    world["assistant"].reload_error = AssistantError("busy")
    data = world["client"].delete("/api/people/sara").get_json()
    assert data["ok"] is True
    assert "could not be enrolled again" in data["warning"] and "busy" in data["warning"]
    assert world["calls"] == ["reload_people", "assistant.reload"]      # the assistant still reloads


def test_removing_a_person_whose_photos_are_being_taken_cancels_them_first(world):
    world["client"].post("/api/people/omar/photos", json={})
    world["calls"].clear()
    world["client"].delete("/api/people/omar")
    assert world["calls"] == ["cancel_photos", "reload_people", "assistant.reload"]


# ---- photos -------------------------------------------------------------------------------------


def test_taking_photos_starts_a_session_with_five_or_the_count_given(world):
    client = world["client"]
    assert client.post("/api/people/sara/photos").get_json() == {"ok": True}          # no body at all
    world["pipeline"].progress = None
    client.post("/api/people/omar/photos", json={"count": 7})
    assert world["calls"] == [("start_photos", "sara", 5), ("start_photos", "omar", 7)]


@pytest.mark.parametrize("error, status", [
    (KeyError("omar"), 404),
    (ValueError("Photos are being taken already. Wait until they are done, or cancel them."), 400),
    (ValueError("Take 3 to 10 photos."), 400),
])
def test_a_refused_session_gives_the_pipelines_message(world, error, status):
    world["pipeline"].start_error = error
    response = world["client"].post("/api/people/omar/photos", json={"count": 2})
    assert response.status_code == status
    if status == 400:
        assert response.get_json()["error"] == str(error)


def test_progress_before_any_session_is_null(world):
    response = world["client"].get("/api/people/photos")
    assert response.get_json() == {"progress": None}
    assert response.headers["Cache-Control"] == "no-store"


def test_the_end_of_a_session_reloads_the_assistant_exactly_once(world):
    client, pipeline = world["client"], world["pipeline"]
    client.post("/api/people/omar/photos", json={})
    assert client.get("/api/people/photos").get_json()["progress"]["state"] == "taking"
    pipeline.progress = {**pipeline.progress, "state": "enrolling"}
    client.get("/api/people/photos")
    pipeline.progress = {**pipeline.progress, "state": "done", "taken": 5}
    for _page in range(3):                                       # three pages poll the same end
        assert client.get("/api/people/photos").get_json()["progress"]["state"] == "done"
    assert world["calls"].count("assistant.reload") == 1
    client.post("/api/people/omar/photos", json={})              # a second session ends done too
    pipeline.progress = {**pipeline.progress, "state": "done"}
    client.get("/api/people/photos")
    client.get("/api/people/photos")
    assert world["calls"].count("assistant.reload") == 2


def test_a_failed_session_does_not_reload_the_assistant(world):
    client, pipeline = world["client"], world["pipeline"]
    client.post("/api/people/omar/photos", json={})
    pipeline.progress = {**pipeline.progress, "state": "failed", "hint": "Only 1 photo was clear."}
    client.get("/api/people/photos")
    pipeline.progress = {**pipeline.progress, "state": "done"}   # a done seen later, without a new start
    client.get("/api/people/photos")
    assert "assistant.reload" not in world["calls"]


def test_a_failing_reload_at_the_end_of_a_session_is_a_warning(world):
    world["assistant"].reload_error = AssistantError("busy")
    world["client"].post("/api/people/omar/photos", json={})
    world["pipeline"].progress = {**world["pipeline"].progress, "state": "done"}
    data = world["client"].get("/api/people/photos").get_json()
    assert data["progress"]["state"] == "done" and "busy" in data["warning"]


def test_cancelling_photos_is_not_taken_for_removing_a_person_called_photos(world):
    world["client"].post("/api/people/omar/photos", json={})
    response = world["client"].delete("/api/people/photos")
    assert response.get_json() == {"ok": True}
    assert world["calls"][-1] == "cancel_photos"
    assert (world["photos_dir"] / "omar").exists()


def test_photo_watch_says_done_once_and_ignores_ends_it_did_not_start():
    watch = PhotoWatch()
    assert watch.ended_done("done") is False                    # nothing was started from the page
    watch.started()
    assert watch.ended_done("taking") is False
    assert watch.ended_done("done") is True
    assert watch.ended_done("done") is False


# ---- notes -------------------------------------------------------------------------------------


def test_reading_the_notes(world):
    response = world["client"].get("/api/notes")
    assert response.get_json() == {"text": "# Notes\n[REPLACE] the team\n"}


def test_reading_missing_notes_gives_an_empty_text(world):
    world["notes"].unlink()
    assert world["client"].get("/api/notes").get_json() == {"text": ""}


def test_saving_the_notes_keeps_a_backup_writes_utf8_then_reloads(world):
    response = world["client"].put("/api/notes", json={"text": "# Notes\nSara builds the car. Ünïcode ✓\n"})
    assert response.get_json() == {"ok": True}
    assert world["notes"].read_bytes() == "# Notes\nSara builds the car. Ünïcode ✓\n".encode("utf-8")
    backup = world["notes"].with_name("knowledge.md.bak")
    assert backup.read_text(encoding="utf-8") == "# Notes\n[REPLACE] the team\n"
    assert [p.name for p in world["tmp"].iterdir() if p.name.endswith(".partial")] == []
    assert world["calls"] == ["assistant.reload"]


@pytest.mark.parametrize("raw, message", [
    ('{"text": 5}', "Send the notes as JSON"),
    ('{"words": "x"}', "Send the notes as JSON"),
    ('{"text": "' + "x" * (MAX_NOTES_CHARS + 1) + '"}', "at most 50000 characters"),
    ('{"text": "half a pair \\ud800 here"}', "cannot be saved"),
])
def test_notes_that_cannot_be_saved_are_refused_and_nothing_is_written(world, raw, message):
    response = world["client"].put("/api/notes", data=raw, content_type="application/json")
    assert response.status_code == 400 and message in response.get_json()["error"]
    assert world["notes"].read_text(encoding="utf-8") == "# Notes\n[REPLACE] the team\n"
    assert not world["notes"].with_name("knowledge.md.bak").exists()
    assert world["calls"] == []


def test_notes_at_the_limit_are_saved(world):
    assert world["client"].put("/api/notes", json={"text": "x" * MAX_NOTES_CHARS}).status_code == 200


def test_a_failed_write_leaves_the_old_notes_and_no_temporary_file(world, monkeypatch):
    def broken_replace(source, target):
        raise PermissionError("the file is open in another program")

    monkeypatch.setattr(app_module.os, "replace", broken_replace)
    response = world["client"].put("/api/notes", json={"text": "new"})
    assert response.status_code == 500 and "could not be saved" in response.get_json()["error"]
    assert world["notes"].read_text(encoding="utf-8") == "# Notes\n[REPLACE] the team\n"
    assert [p.name for p in world["tmp"].iterdir() if p.name.endswith(".partial")] == []
    assert world["calls"] == []


def test_saving_notes_with_a_failing_reload_answers_ok_with_a_warning(world):
    world["assistant"].reload_error = AssistantError("The notes file could not be read.")
    data = world["client"].put("/api/notes", json={"text": "new"}).get_json()
    assert data["ok"] is True and "could not be read" in data["warning"]
    assert world["notes"].read_text(encoding="utf-8") == "new"


def test_a_new_notes_file_gets_no_backup(world):
    world["notes"].unlink()
    world["client"].put("/api/notes", json={"text": "first"})
    assert world["notes"].read_text(encoding="utf-8") == "first"
    assert not world["notes"].with_name("knowledge.md.bak").exists()


# ---- describe the scene ----------------------------------------------------------------------


def test_the_scene_is_described_from_the_servers_own_picture(world):
    response = world["client"].post("/api/scene", json={"image": "data:image/jpeg;base64,AAAA"})
    assert response.get_json() == {"text": "A desk with a laptop. Sara Ahmed is standing beside it."}
    assert world["assistant"].scenes == [SCENE_JPEG]                    # never a picture from the browser


def test_no_picture_yet_is_a_plain_error(world):
    world["pipeline"].photo = None
    response = world["client"].post("/api/scene")
    assert response.status_code == 503 and "no picture from the camera yet" in response.get_json()["error"]
    assert world["assistant"].scenes == []


def test_an_assistant_error_while_describing_is_shown_like_elsewhere(world):
    world["assistant"].scene_error = AssistantError("The assistant is still busy.")
    response = world["client"].post("/api/scene")
    assert response.status_code == 200 and response.get_json() == {"error": "The assistant is still busy."}


def test_the_unavailable_assistant_explains_itself_for_the_new_calls():
    standing_in = UnavailableAssistant("The assistant is not available: no key.")
    with pytest.raises(AssistantError, match="no key"):
        standing_in.describe_scene(SCENE_JPEG)
    with pytest.raises(AssistantError, match="Restart the program"):
        standing_in.reload()


# ---- building the book and the notes path from the settings ---------------------------------


def test_the_book_and_the_notes_come_from_the_settings():
    config = {"recognizer": {"photos_dir": "data/people"}, "assistant": {"notes_file": "data/knowledge.md"}}
    book = build_people_book(config)
    assert book.photos_dir == resolve_path("data/people")
    assert book.removed_dir == resolve_path("data/people.removed")
    assert notes_path(config) == resolve_path("data/knowledge.md")


def test_missing_settings_turn_the_sections_off():
    assert build_people_book({}) is None and notes_path({"assistant": {}}) is None


# ---- the page ---------------------------------------------------------------------------------


@pytest.fixture
def page(world):
    return world["client"].get("/").get_data(as_text=True)


def script_part(page, start, end):
    return page[page.index(start):page.index(end, page.index(start))]


def test_the_page_has_the_people_section_its_form_and_its_permission_line(page):
    assert "Take photos only with the person's permission. Photos and details stay on this laptop." in page
    for element in ('id="peopleList"', 'id="addPerson" type="button">Add a person</button>',
                    '<label class="field-label" for="personName">Name</label>',
                    '<label class="field-label" for="personRole">Role</label>',
                    '<label class="field-label" for="personLines">Card lines (one per line)</label>',
                    '<label class="field-label" for="personStory">About</label>',
                    'id="photoProgress"', 'id="photoHint"', 'id="photoCount"', 'id="photoCancel"'):
        assert element in page, element
    assert '<input id="personName" type="text" maxlength="80" required>' in page
    assert '"Recognised"' in page and '"Not enrolled yet"' in page
    assert '" of "' in page                                              # 3 of 5


def test_the_page_removes_only_after_an_in_page_question(page):
    assert "confirm(" not in page and "window.confirm" not in page and "alert(" not in page
    ask = script_part(page, "function askRemove(", "async function removePerson(")
    assert "moved aside into people.removed, not deleted." in ask and "Remove \" + name" in ask
    assert "removePerson(person" in ask                                  # only the in-page Remove button acts


def test_the_page_follows_photos_only_while_a_session_runs(page):
    assert "PHOTO_POLL_MS = 400" in page and "setInterval" not in page
    assert page.count("setTimeout(checkPhotos, PHOTO_POLL_MS)") == 1
    check = script_part(page, "async function checkPhotos()", "function hintText(")
    assert "if (RUNNING.includes(progress.state)) followPhotos();" in check   # stops at done or failed
    assert 'const RUNNING = ["taking", "enrolling"];' in page
    callers = [m.start() for m in re.finditer(r"followPhotos\(\);", page)]
    assert len(callers) == 3                       # after Take photos, while running, and a session found at load
    resume = script_part(page, "async function followRunningPhotos()", "// ---- notes")
    assert "RUNNING.includes(progress.state)" in resume


def test_the_photo_hint_is_also_a_solid_capsule_over_the_video(page):
    from test_app import css_rule, ids_around
    around = ids_around(page)
    assert "frame" in around["photoCapsule"] and "watch" in around["photoCapsule"]
    capsule = css_rule(page, ".photo-capsule")
    assert "backdrop-filter" not in capsule and "background: #1c1c1e" in capsule


def test_the_page_has_the_notes_editor(page):
    assert '<textarea id="notesText" maxlength="50000" rows="18" spellcheck="true" disabled></textarea>' in page
    assert 'id="notesSave" type="submit" disabled>Save</button>' in page    # off until the notes were loaded
    assert 'notesText.value.includes("[REPLACE]")' in page
    assert 'sendJson("PUT", "/api/notes", { text: notesText.value })' in page
    load = script_part(page, "async function loadNotes()", 'notesText.addEventListener("input"')
    assert load.index("notesText.value = data.text;") < load.index("notesSave.disabled = false;")


def test_the_scene_button_sits_on_the_video_and_answers_in_caption_chat_and_voice(page):
    from test_app import ids_around
    around = ids_around(page)
    assert "frame" in around["sceneButton"] and "watch" in around["sceneButton"]
    assert 'id="sceneButton" type="button" title="Describe the scene"' in page
    assert page.index('id="sceneButton"') < page.index('id="fullButton"')   # beside it, also in the full view
    scene = script_part(page, "async function describeScene()", 'sceneButton.addEventListener("click"')
    assert "if (sceneBusy) return;" in scene and "setSceneBusy(true" in scene and "setSceneBusy(false" in scene
    assert 'addMessage("question", SCENE_QUESTION);' in scene and "addAnswer({ text });" in scene
    assert 'answerArrived("scene", text, isProblem);' in scene             # caption, and say() when voice is on
    assert 'const SCENE_QUESTION = "Describe the scene";' in page
    assert 'postJson("/api/scene", {})' in page                            # no picture from the browser
    assert page.count("Describe the scene sends the whole picture.") == 2  # under the list and in the footer


def test_the_new_sections_are_outside_the_stage_so_the_full_view_makes_them_inert(page):
    from test_app import ids_around
    around = ids_around(page)
    for element in ("people", "knowledge", "personForm", "notesText"):
        assert "watch" not in around[element], element


def test_the_navigation_bar_links_the_new_sections_and_folds_its_links_earlier(page):
    assert '<a href="#people" id="peopleLink">People</a>' in page
    assert '<a href="#knowledge" id="notesLink">Notes</a>' in page
    assert "@media (max-width: 1199.98px)" in page
    narrow = page[page.index("@media (max-width: 1199.98px)"):]
    assert "display: none" in narrow[:narrow.index("}")]


def test_the_people_routes_are_named_in_the_page_as_the_server_has_them(page):
    for text in ('getJson("/api/people")', 'postJson("/api/people", body)', 'sendJson("PUT", personPath(id), body)',
                 'sendJson("DELETE", personPath(person.id))', 'personPath(person.id) + "/photos"',
                 'getJson("/api/people/photos")', 'sendJson("DELETE", "/api/people/photos")',
                 '"/api/people/" + encodeURIComponent(id)'):
        assert text in page, text


def test_the_server_never_logs_a_persons_details(world, caplog):
    caplog.set_level("INFO")
    world["client"].post("/api/people", json={"name": "Lina Haddad", "role": "Secret role", "story": "Private story"})
    text = caplog.text
    assert "Secret role" not in text and "Private story" not in text and "Lina Haddad" not in text
