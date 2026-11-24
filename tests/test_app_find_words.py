"""M20 Part D: the page's words for a maybe, a find on a person, and a known person's name.

The page's script cannot run here (no JavaScript engine), so its rules are checked by the
shape of the script, in the style of tests/test_app_find.py. The route that starts a search
is checked with a fake pipeline.
"""

from car_vision.app import create_app
from test_app import TEMPLATE, between, css_rule


def page_text():
    return TEMPLATE.read_text(encoding="utf-8")


# ---- the chips ---------------------------------------------------------------------------------


def test_the_chips_say_found_maybe_or_in_view():
    state = between(page_text(), "function chipState(", "\n}\n")
    assert '"found " + placeWords(entry.place)' in state                  # a sure find, as before
    assert '"maybe, " + placeWords(entry.place)' in state                 # "maybe, on the left"
    assert '"in view, " + placeWords(entry.place)' in state               # a person: "in view, on the left"
    assert "if (!entry.sure)" in state and "if (entry.person)" in state


def test_a_maybe_chip_looks_quieter():
    page = page_text()
    assert 'item.classList.toggle("is-maybe", chip.maybe);' in between(page, "function drawChips(", "\n}\n")
    assert "dashed" in css_rule(page, ".find-chip.is-maybe")
    assert "#ffd60a" not in css_rule(page, ".find-chip.is-maybe")        # a dimmer yellow than a sure find


def test_the_search_status_is_read_with_sure_person_and_people():
    read = between(page_text(), "function readSearch(", "\n}\n")
    assert "sure: entry.sure !== false" in read
    assert "person: entry.person === true" in read
    assert "people: asList(search.people)" in read


# ---- the pills under In view -----------------------------------------------------------------------


def test_a_maybe_pill_gets_the_maybe_tag_and_its_title():
    page = page_text()
    assert 'const MAYBE_TAG = "maybe";' in page
    assert 'const MAYBE_TITLE = "Not sure: it may be something else";' in page
    assert "maybe: raw.maybe === true && raw.found === true" in between(page, "function readObject(", "\n}\n")
    draw = between(page, "function drawObjects(", "\n}\n")
    assert "setText(tag, thing.maybe ? MAYBE_TAG : DISCOVERED_TAG);" in draw
    assert "tag.hidden = !thing.discovered && !thing.maybe;" in draw
    assert "if (thing.maybe) titles.push(MAYBE_TITLE);" in draw
    assert 'button.classList.toggle("is-maybe", thing.maybe);' in draw
    assert "dashed" in css_rule(page, ".thing.is-maybe")


def test_a_pill_on_a_person_says_a_click_shows_the_person():
    page = page_text()
    assert 'const ON_PERSON_TITLE = "On a person: a click shows the person";' in page
    assert "onPerson: raw.on_person === true" in between(page, "function readObject(", "\n}\n")
    assert "onPerson: entry.onPerson" in between(page, "function thingText(", "\n}\n")
    assert "if (thing.onPerson) titles.push(ON_PERSON_TITLE);" in between(page, "function drawObjects(", "\n}\n")


# ---- the sentences the page writes and speaks ------------------------------------------------------


def test_a_person_is_named_without_a_before_their_name():
    page = page_text()
    seen = between(page, "function seenSentence(", "\n}\n")
    assert 'if (entry.person) return "I can see " + entry.name + when + ", " + placeWords(entry.place) + ".";' in seen
    sentences = between(page, "function findSentences(", "\n}\n")
    assert "isPersonName(search, name) ? name : anyOf(name)" in sentences
    assert '"I can\'t see " + joinWith(missed, "or") + " right now. I\'ll keep looking."' in sentences
    assert "search.people.some((person) => sameName(person, name))" in page
    timed_out = between(page, "function findTimedOut(", "\n}\n")
    assert 'isPersonName(mine.search, name) ? name : "the " + name' in timed_out


def test_a_maybe_is_said_as_not_sure():
    page = page_text()
    seen = between(page, "function seenSentence(", "\n}\n")
    assert '"I think I can see " + someOf(entry.name) + when + ", " + placeWords(entry.place) + ", but I am not sure."' \
        in seen
    assert 'return looksPlural(name) ? "some " + name : anyOf(name);' in between(page, "function someOf(", "\n}\n")


def test_a_maybe_counts_as_an_answer():
    """The page answers once every name has a found entry, and a maybe is one: it does not wait for a sure find."""
    follow = between(page_text(), "function followFind(", "\n}\n")
    assert "const allSeen = mine.names.every((name) => foundEntry(search, name));" in follow
    assert "sure" not in follow


def test_a_name_answered_as_a_maybe_is_waited_for_and_its_sure_find_told_once():
    page = page_text()
    answer = between(page, "function answerFind(", "\n}\n")
    assert "if (!entry || !entry.sure) mine.waitingFor.add(name.toLowerCase());" in answer
    told = between(page, "function tellNewlyFound(", "\n}\n")
    assert "if (!entry || !entry.sure || !mine.waitingFor.has(key)) continue;" in told
    assert "mine.waitingFor.delete(key)" in told                          # told once, then no more


def test_the_privacy_paragraph_says_faces_are_covered():
    privacy = between(page_text(), '<p class="privacy" id="privacy">', "</p>")
    assert "A click on a person sends nothing. Faces in a cut-out are covered before it is sent." in privacy


def test_the_new_code_stays_in_the_finders_section_and_the_pills():
    page = page_text()
    finder = between(page, "// ---- looking for things by name", "// ---- auto-discover")
    for part in ("function someOf(", "const isPersonName", "people: asList(search.people)"):
        assert part in finder, part
    pills = between(page, "// ---- in view: one pill per detected thing", "// ---- codes:")
    for part in ("const MAYBE_TAG", "const ON_PERSON_TITLE", "onPerson: raw.on_person"):
        assert part in pills, part


# ---- the route answers with the names as the status shows them -------------------------------------


class NamingPipeline:
    """search() answers like the real pipeline: a known person's name as they were enrolled."""

    def __init__(self, shown):
        self.shown = shown
        self.searches = []

    def search(self, names):
        self.searches.append(list(names))
        return self.shown

    def sightings(self):
        return None

    def status(self):
        return {}


def client_for(pipeline):
    app = create_app(pipeline, object())
    app.testing = True
    return app.test_client()


def test_the_search_route_answers_with_the_names_as_shown():
    pipeline = NamingPipeline(("keys", "Mohamed Adel"))
    reply = client_for(pipeline).post("/api/search", json={"text": "keys and mohamed"})
    assert reply.status_code == 200
    assert reply.get_json() == {"ok": True, "names": ["keys", "Mohamed Adel"]}
    assert pipeline.searches == [["keys", "mohamed"]]


def test_a_pipeline_that_does_not_say_keeps_the_cleaned_names():
    pipeline = NamingPipeline(None)
    reply = client_for(pipeline).post("/api/search", json={"names": [" keys "]})
    assert reply.get_json() == {"ok": True, "names": ["keys"]}
