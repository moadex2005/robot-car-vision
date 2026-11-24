"""The M13 routes and page parts: looking for things by name, and the camera's memory.

The pipeline and its memory are fakes that record what they were asked. The page's
script cannot run here (no JavaScript engine), so its rules are checked by the shape
of the script, and the find-request rule also through a small Python copy of it that
reads its word lists and its opening pattern from the page itself.
"""

import json
import re
import unicodedata

import pytest

from car_vision.app import MAX_FILTER_CHARS, MAX_SEARCH_CHARS, create_app, split_names
from car_vision.memory import Sighting
from test_app import TEMPLATE, between, css_rule, ids_around

JPEG = b"\xff\xd8 a small picture of a bottle \xff\xd9"
BOTTLE = Sighting(3, "object", "bottle", None, 1000.0, 1090.0, "left", 0.81234, True)
SARA = Sighting(2, "person", "person", "Sara Ahmed", 900.0, 940.0, "middle", 0.9, True)   # never shown with a picture
STRANGER = Sighting(1, "person", "person", None, 800.0, 810.0, "right", 0.7, False)


class FakeMemory:
    def __init__(self, sightings=(BOTTLE, SARA, STRANGER), pictures=None):
        self.sightings = list(sightings)
        self.pictures = {3: JPEG} if pictures is None else pictures
        self.asked = []                       # (limit, text) of every recent() call
        self.picture_ids = []                 # every id picture() was asked for
        self.cleared = 0

    def recent(self, limit=50, text=None):
        self.asked.append((limit, text))
        return self.sightings[:limit]

    def picture(self, sighting_id):
        self.picture_ids.append(sighting_id)
        return self.pictures.get(sighting_id)

    def clear(self):
        self.cleared += 1
        self.sightings = []


class FakePipeline:
    def __init__(self, memory=None, search_error=None):
        self.memory = memory
        self.search_error = search_error
        self.searches = []                    # every list search() was given

    def search(self, names):
        if self.search_error is not None:
            raise self.search_error
        self.searches.append(list(names))

    def sightings(self):
        return self.memory

    def status(self):
        return {}


class FakeAssistant:
    pass


def client_for(pipeline):
    app = create_app(pipeline, FakeAssistant())
    app.testing = True
    return app.test_client()


@pytest.fixture
def pipeline():
    return FakePipeline(memory=FakeMemory())


@pytest.fixture
def client(pipeline):
    return client_for(pipeline)


def page_text():
    return TEMPLATE.read_text(encoding="utf-8")


# ---- POST and DELETE /api/search ----------------------------------------------------------


def test_search_with_names_starts_looking_and_answers_the_names(client, pipeline):
    reply = client.post("/api/search", json={"names": ["power bank", " keys ", "Keys"]})
    assert reply.status_code == 200
    assert reply.get_json() == {"ok": True, "names": ["power bank", "keys"]}   # trimmed, the same name once
    assert pipeline.searches == [["power bank", "keys"]]


def test_search_with_text_is_split_on_commas_and_the_word_and(client, pipeline):
    reply = client.post("/api/search", json={"text": "power bank, keys AND the remote,  wifi router"})
    assert reply.get_json() == {"ok": True, "names": ["power bank", "keys", "the remote", "wifi router"]}
    assert pipeline.searches == [["power bank", "keys", "the remote", "wifi router"]]


@pytest.mark.parametrize("text, names", [
    ("hand band", ["hand band"]),                            # "and" inside a word is not a separator
    ("salt and pepper", ["salt", "pepper"]),
    ("keys,,remote, and ,cup", ["keys", "remote", "cup"]),  # empty parts are dropped
    ("Sandwich", ["Sandwich"]),
])
def test_split_names(text, names):
    assert split_names(text) == names


@pytest.mark.parametrize("body", [
    None, [], "keys", 5, {}, {"names": ["keys"], "text": "keys"}, {"names": "keys"}, {"names": None},
    {"text": None}, {"text": ["keys"]}, {"name": ["keys"]},
])
def test_search_with_a_body_of_the_wrong_shape_is_400_with_a_plain_message(client, pipeline, body):
    reply = client.post("/api/search", data=json.dumps(body), content_type="application/json")   # None: null
    assert reply.status_code == 400
    assert "Send what to look for as JSON" in reply.get_json()["error"]
    assert pipeline.searches == []


def test_search_with_a_body_that_is_not_json_is_refused(client, pipeline):
    reply = client.post("/api/search", data="keys", content_type="text/plain")
    assert reply.status_code == 415 and "error" in reply.get_json()
    reply = client.post("/api/search", data="keys", content_type="application/json")
    assert reply.status_code == 400 and "error" in reply.get_json()
    assert pipeline.searches == []


@pytest.mark.parametrize("body, message", [
    ({"names": []}, "Type what to look for"),
    ({"text": ""}, "Type what to look for"),
    ({"text": " , and , "}, "Type what to look for"),
    ({"names": ["keys", 5]}, "Each name must be text."),
    ({"names": [["keys"]]}, "Each name must be text."),
    ({"names": ["   "]}, "Each name must be 1 to 40 characters long."),
    ({"names": ["x" * 41]}, "Each name must be 1 to 40 characters long."),
    ({"names": ["keys; drop"]}, "A name may only have letters"),
    ({"text": "keys <script>"}, "A name may only have letters"),
    ({"names": [f"thing {n}" for n in range(7)]}, "Look for at most 6 things at once."),
    ({"text": "a" * (MAX_SEARCH_CHARS + 1)}, "That is too long to look for"),
])
def test_search_with_bad_names_is_400_with_the_pipelines_message(client, pipeline, body, message):
    reply = client.post("/api/search", json=body)
    assert reply.status_code == 400
    assert message in reply.get_json()["error"]
    assert pipeline.searches == []


def test_search_passes_on_a_value_error_from_the_pipeline_as_400():
    client = client_for(FakePipeline(search_error=ValueError("The finder is switched off.")))
    reply = client.post("/api/search", json={"names": ["keys"]})
    assert reply.status_code == 400 and reply.get_json() == {"error": "The finder is switched off."}


def test_search_names_in_any_script_are_fine(client, pipeline):
    reply = client.post("/api/search", json={"text": "مفاتيح, Schlüssel and power-bank, cat's bowl"})
    assert reply.status_code == 200
    assert pipeline.searches == [["مفاتيح", "Schlüssel", "power-bank", "cat's bowl"]]


@pytest.mark.parametrize("names", [["man"], ["face"], ["person"], ["keys", "person"]])
def test_words_for_people_are_refused_with_the_pipelines_plain_message(client, pipeline, names):
    """The page shows this message as the answer to "can you see my face" (never spoken)."""
    reply = client.post("/api/search", json={"names": names})
    assert reply.status_code == 400
    assert reply.get_json() == {"error": "Looking for people by name is not supported. "
                                         "People the camera knows are recognised by their face."}
    assert pipeline.searches == []


def test_delete_search_stops_looking(client, pipeline):
    reply = client.delete("/api/search", json={})
    assert reply.status_code == 200 and reply.get_json() == {"ok": True}
    assert pipeline.searches == [[]]


# ---- GET /api/memory --------------------------------------------------------------------------


def test_memory_lists_the_sightings_newest_first_as_plain_json(client, pipeline):
    reply = client.get("/api/memory")
    assert reply.status_code == 200
    assert reply.headers["Cache-Control"] == "no-store"
    assert pipeline.memory.asked == [(50, None)]
    sightings = reply.get_json()["sightings"]
    assert sightings[0] == {"id": 3, "kind": "object", "label": "bottle", "name": None, "first_seen": 1000.0,
                            "last_seen": 1090.0, "place": "left", "confidence": 0.81, "has_picture": True}
    assert sightings[1]["name"] == "Sara Ahmed" and sightings[2]["name"] is None
    assert [s["has_picture"] for s in sightings[1:]] == [False, False]      # never a picture of a person


@pytest.mark.parametrize("query, asked", [
    ("?limit=10", (10, None)), ("?limit=0", (1, None)), ("?limit=1", (1, None)), ("?limit=200", (200, None)),
    ("?limit=999999", (200, None)), ("?text=bottle", (50, "bottle")), ("?text=%20%20Sara%20", (50, "Sara")),
    ("?text=", (50, None)), ("?limit=5&text=cup", (5, "cup")), ("?text=" + "b" * MAX_FILTER_CHARS, (50, "b" * MAX_FILTER_CHARS)),
])
def test_memory_holds_the_limit_between_1_and_200_and_passes_the_filter(client, pipeline, query, asked):
    assert client.get("/api/memory" + query).status_code == 200
    assert pipeline.memory.asked == [asked]


@pytest.mark.parametrize("query, message", [
    ("?limit=abc", "limit must be a whole number"), ("?limit=-1", "limit must be a whole number"),
    ("?limit=1.5", "limit must be a whole number"), ("?limit=", "limit must be a whole number"),
    ("?limit=1e3", "limit must be a whole number"), ("?limit=99999999999999999999", "limit must be a whole number"),
    ("?text=" + "b" * (MAX_FILTER_CHARS + 1), "The filter can have at most"),
])
def test_memory_with_a_bad_address_is_400_with_a_plain_message(client, pipeline, query, message):
    reply = client.get("/api/memory" + query)
    assert reply.status_code == 400 and message in reply.get_json()["error"]
    assert pipeline.memory.asked == []


def test_with_the_memory_off_the_memory_routes_answer_empty():
    client = client_for(FakePipeline(memory=None))
    assert client.get("/api/memory").get_json() == {"sightings": []}
    assert client.get("/api/memory/3/picture").status_code == 404
    assert client.delete("/api/memory", json={}).get_json() == {"ok": True}


def test_a_pipeline_without_a_memory_method_counts_as_memory_off():
    class OldPipeline:
        def status(self):
            return {}

    client = client_for(OldPipeline())
    assert client.get("/api/memory").get_json() == {"sightings": []}
    assert 'data-memory="off"' in client.get("/").get_data(as_text=True)


# ---- GET /api/memory/<id>/picture ---------------------------------------------------------------


def test_the_picture_of_a_remembered_object(client, pipeline):
    reply = client.get("/api/memory/3/picture")
    assert reply.status_code == 200
    assert reply.mimetype == "image/jpeg" and reply.data == JPEG
    # Never kept by the browser: after Clear memory the same id may belong to another row.
    assert reply.headers["Cache-Control"] == "no-store"
    assert pipeline.memory.picture_ids == [3]


@pytest.mark.parametrize("sighting_id", ["2", "1", "999"])          # a person, a person, unknown
def test_no_picture_is_404_as_json(client, pipeline, sighting_id):
    reply = client.get(f"/api/memory/{sighting_id}/picture")
    assert reply.status_code == 404 and reply.get_json() == {"error": "There is no picture of that."}


@pytest.mark.parametrize("sighting_id", ["abc", "-1", "1.5", "1e3", "%D9%A3", "%C2%B2", "1" * 19, "3%20", "0x3"])
def test_an_id_that_is_not_a_plain_number_is_404_before_the_memory_is_asked(client, pipeline, sighting_id):
    reply = client.get(f"/api/memory/{sighting_id}/picture")
    assert reply.status_code == 404 and "error" in reply.get_json()
    assert pipeline.memory.picture_ids == []


def test_the_largest_id_that_fits_is_asked_for(client, pipeline):
    assert client.get("/api/memory/" + "9" * 18 + "/picture").status_code == 404
    assert pipeline.memory.picture_ids == [int("9" * 18)]


def test_only_get_is_allowed_on_a_picture(client):
    assert client.post("/api/memory/3/picture").status_code == 405


# ---- DELETE /api/memory ---------------------------------------------------------------------------


def test_delete_memory_empties_it(client, pipeline):
    reply = client.delete("/api/memory", json={})
    assert reply.status_code == 200 and reply.get_json() == {"ok": True}
    assert pipeline.memory.cleared == 1
    assert client.get("/api/memory").get_json() == {"sightings": []}


# ---- the page: Look for on the video --------------------------------------------------------------


@pytest.fixture
def rendered(client):
    return client.get("/").get_data(as_text=True)


def test_the_look_for_field_sits_on_the_video_also_in_the_full_view(rendered):
    around = ids_around(rendered)
    for element in ("finder", "finderForm", "finderText", "finderFind", "findChips", "finderStop"):
        assert "frame" in around[element] and "watch" in around[element], element
    assert f'<input id="finderText" type="text" maxlength="{MAX_SEARCH_CHARS}"' in rendered
    assert 'placeholder="Look for ..."' in rendered
    assert '<label class="visually-hidden" for="finderText">' in rendered
    assert 'id="finderFind" type="submit">Find</button>' in rendered               # Enter works: a real form
    assert 'id="finderStop" type="button" hidden>Stop</button>' in rendered
    assert 'id="findChips"' in rendered and re.search(r'id="findChips"[^>]*\shidden>', rendered)


def test_the_look_for_row_is_solid_and_keeps_out_of_the_way():
    page = page_text()
    for selector in (".finder", ".finder-form", ".find-chip", ".finder-button"):
        assert "backdrop-filter" not in css_rule(page, selector), selector
    assert "background: #1c1c1e" in css_rule(page, ".finder-form") and "background: #1c1c1e" in css_rule(page, ".find-chip")
    finder = css_rule(page, ".finder")
    assert "left: var(--frame-edge)" in finder and "bottom: var(--frame-edge)" in finder   # bottom-left
    assert "max-width: calc(100% - 2 * var(--frame-edge))" in finder
    assert "pointer-events: none" in finder                                       # between its parts, the video
    # In the full view it sits above the row of pills; the caption always sits above it.
    assert "2.9rem + 36px" in css_rule(page, ".stage.is-full .finder")
    assert "var(--finder-room)" in css_rule(page, ".caption")
    assert "var(--finder-room)" in css_rule(page, ".stage:not(.is-full) .caption")
    room = between(page, "function fitFinderRoom()", "\n}\n")
    assert 'frame.style.setProperty("--finder-room", room)' in room and "finder.offsetHeight" in room
    assert "new ResizeObserver(fitFinderRoom).observe(finder)" in page
    # The chips wrap upwards instead of hiding in a scrolled row.
    assert "flex-wrap: wrap-reverse" in css_rule(page, ".find-chips")


def test_the_chips_say_what_the_status_says():
    page = page_text()
    state = between(page, "function chipState(", "\n}\n")
    for text in ('"loading the finder ..."', '"could not start"', '"not in view"', '"found ("', '"found " + placeWords'):
        assert text in state, text
    assert "left: \"on the left\", middle: \"in the middle\", right: \"on the right\"" in page
    draw = between(page, "function drawChips(", "\n}\n")
    assert 'changed("search"' in draw and "setText(item.firstChild, chip.name)" in draw   # reused, only text changes
    assert "drawSearch(status, number);" in between(page, "async function poll(", "\n}\n")
    assert "finder.hidden = !isOnline;" in page                                       # nothing to look for offline


def test_found_things_are_marked_in_the_in_view_list():
    page = page_text()
    assert "found: raw.found === true" in between(page, "function readObject(", "\n}\n")
    assert 'button.classList.toggle("is-found", thing.found);' in page
    assert "#ffd60a" in css_rule(page, ".thing.is-found")
    assert "backdrop-filter" not in css_rule(page, ".thing.is-found")


def test_the_field_sends_what_was_typed_and_the_server_splits_it():
    page = page_text()
    assert 'postJson("/api/search", { text })' in page
    submit = between(page, 'finderForm.addEventListener("submit"', "\n});\n")
    assert "const text = finderText.value.trim();" in submit and 'where: "stage"' in submit
    assert 'sendJson("DELETE", "/api/search")' in page


# ---- the page: what is said about a search ------------------------------------------------------


def test_only_this_pages_own_search_speaks_and_each_name_once():
    """The owner's rule: the answer to the request, and one "I can see it now" per name and
    search. `finding` exists only on the page that started the search; `waitingFor` is the
    names still owed their "now"."""
    page = page_text()
    start = between(page, "async function startFind(", "\n}\n")
    assert "waitingFor: new Set()" in start and "finding = mine;" in start       # a new search, a new empty set
    assert "mine.fromPoll = pollCount + 1;" in start                            # no status from before it started
    # tellFound is the "now" outside a conversation, called only from tellNewlyFound, only after the answer.
    assert page.count("tellFound(") == 2                                        # its definition and that one call
    follow = between(page, "function followFind(", "\n}\n")
    assert follow.index("if (mine.answered) {") < follow.index("tellNewlyFound(mine, search);")
    assert "if (!mine || mine.names === null || number < mine.fromPoll) return;" in follow
    # Another search (from another page), or one stopped: let go, and nothing more is said.
    assert "sameNames(search.names, mine.names)" in follow and 'search.state !== "off"' in follow
    assert "endFind(mine," in follow
    end = between(page, "function endFind(", "\n}\n")
    assert "if (finding === mine) finding = null;" in end
    assert 'findAnswered(mine, problem, "problem")' in end                      # a problem is never spoken
    stop = between(page, 'finderStop.addEventListener("click"', "\n});\n")
    assert stop.index("endFind(finding,") < stop.index('sendJson("DELETE", "/api/search")')   # silent before it stops
    answered = between(page, "function findAnswered(", "\n}\n")
    assert "clearTimeout(mine.timer);" in answered
    assert 'if (kind !== "answer" && finding === mine) finding = null;' in answered


def test_a_name_leaves_the_waiting_set_only_once_it_was_really_told():
    """Fix: in conversation mode the "now" was shown but never spoken (the page was listening),
    and the name was already out of the waiting set. Now it leaves the set only when its
    sentence was said or shown where the visitor sees it; a conversation says it between turns."""
    page = page_text()
    tell = between(page, "function tellNewlyFound(", "\n}\n")
    assert "!mine.waitingFor.has(key)" in tell and ".delete(" not in tell.split("else if")[0]
    assert "if (conversation) conversationOwes(conversation, mine, name);" in tell
    assert "else if (tellFound(mine, seenSentence(entry, true))) mine.waitingFor.delete(key);" in tell
    told = between(page, "function tellFound(", "\n}\n")
    assert "if (voiceActive() && (talking || hearing)) return false;" in told   # never over speech, never unheard
    assert "if (!mine.canTell()) return false;" in told                         # nor where it would not be seen
    assert told.index("mine.tell(text);") < told.index("return true;")
    # Each place a search answers says whether its "now" can be seen there now.
    chat = between(page, "function findFromChat(", "\n}\n")
    assert "canTell: () => !fullView || !captionWait," in chat
    submit = between(page, 'finderForm.addEventListener("submit"', "\n});\n")
    assert "canTell: () => !captionWait," in submit
    # The conversation: owed once per search and name, said only between two turns, and then the
    # name leaves the set (with the newest status, so a thing that went away again is not called in view).
    owes = between(page, "function conversationOwes(", "\n}\n")
    assert "owed.mine === mine && sameName(owed.name, name)" in owes
    owed = between(page, "function takeOwed(", "\n}\n")
    assert "foundEntry(mine.search, name)" in owed
    assert owed.index("!mine.waitingFor.has(key)") < owed.index("mine.waitingFor.delete(key);")
    assert "mine.tell(text);" in owed and "seenSentence(entry, true)" in owed
    decide = between(page, "function conversationNext(", "\n}\n")
    assert decide.count("nextTurn(conv);") == 2                                 # after an answer, after silence
    assert "owed: []" in between(page, "function startConversation(", "\n}\n")


def test_the_answers_go_back_where_the_request_came_from():
    page = page_text()
    chat = between(page, "function findFromChat(", "\n}\n")
    assert 'answerArrived("chat", text, kind === "problem", kind === "refused");' in chat
    assert "setBusy(true);" in chat and "setBusy(false);" in chat
    assert 'addMessage(kind === "problem" ? "error" : "answer", text);' in chat   # a refusal looks like an answer
    submit = between(page, 'finderForm.addEventListener("submit"', "\n});\n")
    assert 'deliver: (answer, kind) => answerArrived("stage", answer, kind === "problem", kind === "refused")' in submit
    assert 'captionWait = { where: "stage", question };' in submit
    # answerArrived speaks only an answer, and lets a waiting conversation speak it instead.
    arrived = between(page, "function answerArrived(", "\n}\n")
    assert '} else if (!isProblem && !quiet) say(text, "answer");' in arrived
    assert 'isProblem || quiet ? "problem" : "answer"' in arrived              # a refusal ends a conversation quietly
    # The panel never starts a search: typed or spoken, its questions are about the clicked thing.
    assert "function findFromPanel(" not in page


def test_a_refused_name_is_answered_in_the_pages_own_words_and_never_spoken():
    page = page_text()
    start = between(page, "async function startFind(", "\n}\n")
    assert 'if (response.status === 400 && data && typeof data.error === "string") {' in start
    assert 'findAnswered(mine, refusedWords(text, data.error), "refused");' in start
    words = between(page, "function refusedWords(", "\n}\n")
    assert "name.trim().length > FIND_LONGEST_NAME" in words and "TOO_LONG_NAME : message" in words
    assert "const FIND_LONGEST_NAME = 40;" in page
    assert ('const TOO_LONG_NAME = "That name is too long for me to look for. '
            'Please try a shorter one, like power bank.";') in page


def test_the_answer_waits_for_the_finder_and_the_camera_but_not_for_ever():
    page = page_text()
    assert "const FIND_WAIT_MS = 20000;" in page and "const FIND_SETTLE_MS = 1500;" in page
    assert "seen: search.seen === true," in between(page, "function readSearch(", "\n}\n")
    follow = between(page, "function followFind(", "\n}\n")
    # No answer until the finder has looked at a picture for these names (status.search.seen) ...
    assert 'if (search.state !== "ready" || !search.seen) {' in follow
    assert "performance.now() - mine.seenAt >= FIND_SETTLE_MS" in follow
    # ... and while the camera sends nothing, after 20 seconds of that, it says so instead.
    assert "if (!mine.answered && cameraDown) answerCameraDown(mine);" in follow
    assert "const CAMERA_WAIT_MS = 20000;" in page
    assert ('const CAMERA_DOWN = "The camera is not sending pictures right now, so I can\'t look.";') in page
    down = between(page, "function cameraStaysDown(", "\n}\n")
    assert "mine.downSince = 0;" in down and "performance.now() - mine.downSince >= CAMERA_WAIT_MS" in down
    camera = between(page, "function answerCameraDown(", "\n}\n")
    assert 'findAnswered(mine, CAMERA_DOWN, "answer");' in camera and "finding = null;" in camera
    draw = between(page, "function drawSearch(", "\n}\n")
    assert "const cameraUp = status.connected === true && !status.ended && !status.error;" in draw
    late = between(page, "function findTimedOut(", "\n}\n")
    assert "mine.waitingFor.add(name.toLowerCase())" in late                    # then each name gets its "now"
    assert "the first time takes about 15 seconds" in late
    assert late.index("if (mine.downSince) {") < late.index("the first time takes")   # the camera answer wins


def test_the_answer_sentences_are_built_from_the_status():
    page = page_text()
    seen = between(page, "function seenSentence(", "\n}\n")
    assert '"I can see the " + entry.name + when + ", " + placeWords(entry.place) + "."' in seen
    assert "pluralOf(entry.name)" in seen and '" now"' in seen
    words = between(page, "function findSentences(", "\n}\n")
    assert '"I can\'t see " + joinWith(missed, "or") + " right now. I\'ll keep looking."' in words   # M20
    assert "isPersonName(search, name) ? name : anyOf(name)" in words     # no "a" before a person's name
    assert 'return (startsWithVowelSound(name) ? "an " : "a ") + name;' in page
    assert 'if (looksPlural(name)) return "any " + name;' in page


# ---- the find-request rule ------------------------------------------------------------------------


def rule_parts(page):
    """The opening pattern, the "where" pattern and the word lists of the page's findRequest."""
    opening = re.search(r"const FIND_OPENING = /(.+)/i;", page).group(1)
    where = re.search(r"const FIND_WHERE = /(.+)/i;", page).group(1)
    little = re.findall(r'"([^"]+)"', between(page, "const FIND_LITTLE_WORDS = [", "];"))
    not_a_name = re.findall(r'"([^"]+)"', between(page, "const FIND_NOT_A_NAME = [", "];"))
    return re.compile(opening, re.IGNORECASE), re.compile(where, re.IGNORECASE), little, not_a_name


def find_request(text):
    """A line-by-line Python copy of the page's findRequest, over the page's own lists."""
    opening, where, little, not_a_name = rule_parts(page_text())
    sentence = re.sub(r"\s+", " ", text.replace("‘", "'").replace("’", "'")).strip()
    sentence = re.sub(r",?\s+please$", "", re.sub(r"[.!?]+$", "", sentence), flags=re.IGNORECASE).strip()
    if not sentence or len(sentence) > 80:
        return None
    match = opening.match(sentence)
    if not match:
        return None
    needs_little_word = where.match(match.group(1)) is not None
    names = []
    for part in re.split(r"\s*,\s*|\s+and\s+", match.group(2), flags=re.IGNORECASE):
        every = [word for word in part.split(" ") if word]
        if needs_little_word and not (every and every[0].lower() in little):
            return None
        words = [word for word in every if word.lower() not in little]
        if not words or len(words) > 4:
            return None
        for word in words:
            if word.lower() in not_a_name:
                return None
            if not all(unicodedata.category(c)[0] in "LN" or c == "-" for c in word):
                return None
            if len(word) > 1 and unicodedata.category(word[0]) == "Lu" and unicodedata.category(word[1]) == "Ll":
                return None
        names.append(" ".join(words))
    return ", ".join(names) if len(names) <= 6 else None


@pytest.mark.parametrize("sentence, wanted", [
    ("find my keys", "keys"), ("Find my keys.", "keys"), ("look for the remote", "remote"),
    ("where is the power bank", "power bank"), ("Where is the power bank?", "power bank"),
    ("where are my keys", "keys"), ("where's the remote?", "remote"), ("where’s the remote", "remote"),
    ("can you see a bottle", "bottle"), ("Can you see a bottle?", "bottle"), ("do you see the cup", "cup"),
    ("please find my keys", "keys"), ("find my keys please", "keys"), ("find my keys, please", "keys"),
    ("find my keys and the remote", "keys, remote"), ("where is the TV remote", "TV remote"),
    ("look for a wifi router, a power bank and my keys", "wifi router, power bank, keys"),
    ("can you see an umbrella", "umbrella"), ("find the USB cable", "USB cable"),
    ("find power bank", "power bank"), ("look for keys", "keys"), ("where are my keys and the remote", "keys, remote"),
    ("where is a cup", "cup"), ("can you see my face", "face"),          # people words: the server answers
    ("find my car keys", "car keys"), ("find the charging station", "charging station"),
])
def test_a_sentence_that_is_only_a_find_request_is_taken(sentence, wanted):
    assert find_request(sentence) == wanted


@pytest.mark.parametrize("sentence", [
    "where did you last see the bottle", "when did you see Sara", "where is the project's code",
    "can you see well in the dark", "can you see me", "find out who built the car", "look for the keys on the table",
    "where is Sara", "what can you see", "is there a cup", "find my keys and tell me where they are",
    "where is the car going", "can you see anything", "who came by today", "find", "find the", "where is it",
    "where are you", "find and", "look for one, two, three, four, five, six, seven",
    "find the big old red metal toolbox", "how do I find my keys",
    "Can you find out what the team built and where the code is kept for the next demo day at the university?",
    # Questions for the web, not for the camera (fix after review):
    "find a good restaurant nearby", "find python tutorials", "where is the nearest hospital", "where is the capital",
    "where is the robot", "find the best way to clean a lens", "look for the latest news", "find the weather",
    "where is the closest pharmacy", "look for a website about robots", "find information on lidar",
    "find a link to the docs", "find cheap flights", "where is the capital of France",
    # "where is" and "where are" need the, my, a or an before every name:
    "where is paris", "where is london", "where are keys", "where is power bank", "where are my keys and remote",
    "where's paris",
    # People are not looked for by these words: "me" and the like stay with the assistant.
    "can you see me", "can you see us", "do you see him",
])
def test_anything_else_goes_to_the_assistant(sentence):
    assert find_request(sentence) is None


def test_the_rule_is_one_small_function_near_the_top_used_by_the_chat_and_its_microphones():
    page = page_text()
    script = page[page.index("<script>\n\"use strict\";"):]
    assert script.index("function findRequest(") < script.index("// ---- the page coming alive")   # near the top
    rule = between(page, "function findRequest(", "\n}\n")
    assert "fetch(" not in rule and "document" not in rule                    # pure: text in, names or null out
    for word in ("the", "my", "a", "an"):
        assert word in rule_parts(page)[2]
    assert len(set(rule_parts(page)[3])) == len(rule_parts(page)[3])           # each word once
    # The chat field and the chat microphone (also the stage one without a panel, and conversation
    # mode) send through sendChat, which asks the rule first. The panel's field and its microphone
    # send through askFollowUp, never the rule: their questions are about the clicked thing.
    chat = between(page, "function sendChat()", "\n}\n")
    assert "const wanted = findRequest(question);" in chat and "findFromChat(question, wanted)" in chat
    panel = between(page, "function sendFollowUp()", "\n}\n")
    assert "findRequest(" not in panel and "askFollowUp(question, true);" in panel
    submit = between(page, 'followForm.addEventListener("submit"', "\n});\n")
    assert "findRequest(" not in submit
    assert page.count("function findRequest(") == 1 and page.count("findRequest(question)") == 1
    # A conversation sends what it heard through the same functions (sendHeard calls target.send()).
    assert "send: sendChat" in page and "send: sendFollowUp" in page


# ---- the page: the timeline ---------------------------------------------------------------------


def test_the_timeline_section_follows_people_and_has_a_link(rendered):
    around = ids_around(rendered)
    assert rendered.index('id="people"') < rendered.index('id="timeline"') < rendered.index('<footer')
    assert "watch" not in around["timeline"]                                  # inert behind the full view
    links = rendered[rendered.index('<nav class="nav-links"'):rendered.index("</nav>")]
    assert '<a href="#timeline">Timeline</a>' in links
    assert 'data-memory="on"' in rendered
    for element in ("timelineFilter", "timelineClear", "timelineList", "timelineEmpty", "timelineConfirm"):
        assert "timeline" in around[element], element
    assert f'maxlength="{MAX_FILTER_CHARS}"' in rendered
    assert '<div class="confirm-row" id="timelineConfirm" hidden>' in rendered


def test_with_the_memory_off_the_timeline_says_so():
    rendered = client_for(FakePipeline(memory=None)).get("/").get_data(as_text=True)
    assert 'data-memory="off"' in rendered
    assert "The camera's memory is off" in rendered
    assert 'id="timelineList"' not in rendered and 'id="timelineClear"' not in rendered
    assert "const memoryOn = timelineSection.dataset.memory === \"on\";" in rendered


def test_the_navigation_links_fold_before_they_overflow():
    page = page_text()
    assert "@media (max-width: 1179.98px) {\n    .nav-links { display: none; }" in page


def test_the_timeline_loads_only_while_it_is_seen():
    page = page_text()
    assert "const TIMELINE_EVERY_MS = 10000;" in page
    wanted = re.search(r"const timelineWanted = \(\) => (.+);", page).group(1)
    for part in ("memoryOn", "timelineInView", "!fullView", 'document.visibilityState === "visible"'):
        assert part in wanted, part
    changed = between(page, "function timelineChanged()", "\n}\n")
    assert "clearTimeout(timelineTimer);" in changed
    load = between(page, "async function loadTimeline()", "\n}\n")
    assert "if (timelineWanted()) timelineTimer = setTimeout(loadTimeline, TIMELINE_EVERY_MS);" in load
    assert "if (round !== timelineRound) return;" in load                      # an older filter's answer is dropped
    setup = between(page, "function setUpTimeline()", "\n}\n")
    assert "new IntersectionObserver(" in setup and ".observe(timelineBox)" in setup
    assert 'document.addEventListener("visibilitychange", timelineChanged);' in setup
    for function in ("function enterFullView()", "function leaveFullView()"):
        assert "timelineChanged();" in between(page, function, "\n}\n"), function
    assert page.count("setTimeout(loadTimeline") == 1 and "setInterval" not in page


def test_the_timeline_rows_stay_in_place_and_their_times_are_written_on_the_page():
    page = page_text()
    draw = between(page, "function drawTimeline(", "\n}\n")
    assert "momentItems.get(moment.id)" in draw and "placeInOrder(timelineList, items);" in draw
    assert "const now = Date.now() / 1000;" in draw
    order = between(page, "function placeInOrder(", "\n}\n")
    assert "list.insertBefore(item, list.children[i] || null)" in order and "replaceChildren" not in order
    assert '"seen " + timeAgo(now - moment.last) + " · in view for " + howLong(moment.last - moment.first)' in page
    for text in ('"just now"', '"yesterday"', '" ago"', '"less than a second"'):
        assert text in between(page, "function countOf(", "// \"seen 3 minutes ago"), text
    detail = css_rule(page, ".moment-detail")
    assert "white-space: nowrap" in detail and "tabular-nums" in detail
    assert "overflow-y: auto" in css_rule(page, ".timeline-list")              # a new row never moves the page


def test_a_timeline_picture_comes_only_from_the_numeric_id_and_only_for_objects():
    page = page_text()
    address = between(page, "function pictureAddress(", "\n}\n")
    assert 'Number.isSafeInteger(id) && id > 0 ? "/api/memory/" + id + "/picture" : null' in address
    read = between(page, "function readMoment(", "\n}\n")
    assert 'picture: kind === "object" && raw.has_picture === true' in read
    assert "!Number.isSafeInteger(raw.id)" in read
    item = between(page, "function momentItem(", "\n}\n")
    assert "const address = moment.picture ? pictureAddress(moment.id) : null;" in item
    assert "face.src = address;" in item and page.count("face.src =") == 1       # set in one place only


def test_clear_memory_asks_once_in_the_page():
    page = page_text()
    assert "window.confirm" not in page and "confirm(" not in page.replace("timelineConfirm", "")
    setup = between(page, "function setUpTimeline()", "\n}\n")
    assert "timelineConfirm.hidden = false;" in setup and "timelineClearNo.focus({ preventScroll: true });" in setup
    assert 'timelineClearYes.addEventListener("click", clearMemory);' in setup
    assert 'sendJson("DELETE", "/api/memory")' in between(page, "async function clearMemory()", "\n}\n")


def test_the_privacy_line_names_the_memory(rendered):
    footer = rendered[rendered.index('<footer class="footer">'):]
    assert ("The camera's memory (what was seen and when, with small pictures of objects, never of people) "
            "stays on this laptop. A chat question sends its text lines, without pictures, to the assistant.") in footer


def test_the_new_parts_follow_the_page_rules():
    page = page_text()
    start = page.index("// ---- looking for things by name: the Look for field")
    new_parts = page[start:page.index('// ---- keys: Escape')]
    assert "innerHTML" not in new_parts and "insertAdjacentHTML" not in new_parts
    assert "requestAnimationFrame" not in new_parts and "setInterval" not in new_parts
    for css in (css_rule(page, ".moment"), css_rule(page, ".timeline-list"), css_rule(page, ".find-chip")):
        assert "infinite" not in css and "backdrop-filter" not in css
