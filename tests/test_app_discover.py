"""M14 Part C (auto-discover: the route and the page's switch) and the guard that lets only
this page change anything: every POST, PUT and DELETE under /api/ must be JSON (415
otherwise), and a request from another website (its Origin header) is refused (403).

The pipeline is a fake that records what it was asked. The page's script cannot run here
(no JavaScript engine), so its rules are checked by the shape of the script.
"""

import json
import re

import pytest

from car_vision.app import (DISCOVER_SHAPE, NOT_JSON, OTHER_WEBSITE, create_app, discover_interval,
                            interval_words, refusal)
from test_app import TEMPLATE, between, css_rule, ids_around

PRIVACY = ("Auto-discover sends a small picture of the whole view to a Hugging Face model every few seconds. "
           "Anyone in view is in that picture.")


class FakePipeline:
    def __init__(self, discover=None):
        self.switched = []                    # every value discover() was given
        self.discover_status = discover
        self.calls = []                       # any other method a refused request must never reach

    def discover(self, on):
        self.switched.append(on)

    def status(self):
        return {} if self.discover_status is None else {"discover": self.discover_status}

    def sightings(self):                      # the page asks whether there is a memory
        return None

    def __getattr__(self, name):              # search, use_phone, target, ...: recorded, never answered
        def called(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"a refused request reached pipeline.{name}")
        return called


class FakeAssistant:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def called(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"a refused request reached assistant.{name}")
        return called


@pytest.fixture
def world():
    pipeline, assistant = FakePipeline(), FakeAssistant()
    app = create_app(pipeline, assistant)
    app.testing = True
    return {"app": app, "client": app.test_client(), "pipeline": pipeline, "assistant": assistant}


def page_text():
    return TEMPLATE.read_text(encoding="utf-8")


# ---- POST /api/discover --------------------------------------------------------------------


@pytest.mark.parametrize("on", [True, False])
def test_discover_switches_the_pipeline_and_answers_at_once(world, on):
    reply = world["client"].post("/api/discover", json={"on": on})
    assert reply.status_code == 200 and reply.get_json() == {"ok": True, "on": on}
    assert world["pipeline"].switched == [on]


@pytest.mark.parametrize("body", [
    None, {}, [], "on", True, 1, {"on": 1}, {"on": 0}, {"on": "true"}, {"on": None}, {"on": [True]},
    {"switch": True}, {"on": True, "every": 5},
])
def test_discover_with_anything_else_is_400(world, body):
    reply = world["client"].post("/api/discover", data=json.dumps(body), content_type="application/json")
    assert reply.status_code == 400 and reply.get_json() == {"error": DISCOVER_SHAPE}
    assert world["pipeline"].switched == []


def test_discover_with_a_body_that_is_not_json_is_refused(world):
    assert world["client"].post("/api/discover", data="on=1", content_type="application/x-www-form-urlencoded").status_code == 415
    assert world["client"].post("/api/discover", data="not json", content_type="application/json").status_code == 400
    assert world["pipeline"].switched == []


def test_only_post_is_allowed_on_discover(world):
    reply = world["client"].get("/api/discover")
    assert reply.status_code == 405 and "error" in reply.get_json()


def test_the_interval_comes_from_lookout_every_seconds():
    assert discover_interval({"lookout": {"every_seconds": 10}}) == 10.0
    assert discover_interval({"lookout": {"every_seconds": 2.5}}) == 2.5
    for config in ({}, {"lookout": None}, {"lookout": {}}, {"lookout": {"every_seconds": "10"}},
                   {"lookout": {"every_seconds": 0}}, {"lookout": {"every_seconds": -1}},
                   {"lookout": {"every_seconds": True}}):
        assert discover_interval(config) is None, config
    assert (interval_words(10.0), interval_words(2.5), interval_words(None)) == ("10", "2.5", "")


def test_the_page_gets_the_interval_for_its_words():
    app = create_app(FakePipeline(), FakeAssistant(), discover_every=10.0)
    page = app.test_client().get("/").get_data(as_text=True)
    assert 'id="discoverState" data-every="10" hidden' in page
    page = create_app(FakePipeline(), FakeAssistant()).test_client().get("/").get_data(as_text=True)
    assert 'data-every=""' in page


# ---- only this page may change anything -------------------------------------------------------


EXAMPLE_IDS = {"person_id": "sara", "sighting_id": "3"}
CHANGING = ("POST", "PUT", "DELETE")


def changing_routes(app):
    """(method, address) for every POST, PUT and DELETE under /api/, read from the app itself."""
    routes = []
    for rule in app.url_map.iter_rules():
        if not rule.rule.startswith("/api/"):
            continue
        address = re.sub(r"<(?:[a-z]+:)?([a-z_]+)>", lambda m: EXAMPLE_IDS[m.group(1)], rule.rule)
        routes += [(method, address) for method in sorted(rule.methods & set(CHANGING))]
    return routes


def test_the_walk_finds_every_route_that_changes_something(world):
    found = set(changing_routes(world["app"]))
    expected = {("POST", "/api/ask"), ("POST", "/api/reset"), ("POST", "/api/inspect"), ("POST", "/api/describe"),
                ("POST", "/api/follow_up"), ("POST", "/api/camera"), ("POST", "/api/camera/picture"),
                ("POST", "/api/people"), ("PUT", "/api/people/sara"), ("DELETE", "/api/people/sara"),
                ("POST", "/api/people/sara/photos"), ("DELETE", "/api/people/photos"), ("POST", "/api/scene"),
                ("POST", "/api/search"), ("DELETE", "/api/search"), ("DELETE", "/api/memory"),
                ("POST", "/api/discover")}
    assert expected <= found, expected - found


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded",
                                          "multipart/form-data; boundary=x", "application/jsonp"])
def test_every_change_that_is_not_json_is_415_and_reaches_nothing(world, content_type):
    """A plain form on another website cannot send JSON: what it can send is refused."""
    client = world["client"]
    for method, address in changing_routes(world["app"]):
        kwargs = {"data": "x=1"} if content_type is None else {"data": "x=1", "content_type": content_type}
        reply = client.open(address, method=method, **kwargs)
        assert reply.status_code == 415, (method, address, reply.status_code)
        assert reply.get_json() == {"error": NOT_JSON}
    assert world["pipeline"].switched == [] and world["pipeline"].calls == [] and world["assistant"].calls == []


@pytest.mark.parametrize("origin", ["http://evil.example", "https://localhost", "http://localhost:5001",
                                    "null", "http://127.0.0.1", ""])
def test_every_change_from_another_origin_is_403_and_reaches_nothing(world, origin):
    client = world["client"]
    for method, address in changing_routes(world["app"]):
        reply = client.open(address, method=method, json={}, headers={"Origin": origin})
        assert reply.status_code == 403, (method, address, reply.status_code)
        assert reply.get_json() == {"error": OTHER_WEBSITE}
    assert world["pipeline"].switched == [] and world["pipeline"].calls == [] and world["assistant"].calls == []


def test_a_read_from_another_origin_is_refused_too(world):
    assert world["client"].get("/api/status", headers={"Origin": "http://evil.example"}).status_code == 403
    assert world["client"].get("/api/status").status_code == 200


def test_this_pages_own_requests_go_through(world):
    """The test client's page is http://localhost: its own origin, or no Origin header at all."""
    client = world["client"]
    assert client.post("/api/discover", json={"on": True}, headers={"Origin": "http://localhost"}).status_code == 200
    assert client.post("/api/discover", json={"on": False}).status_code == 200
    reply = client.post("/api/discover", data='{"on": true}', content_type="application/json; charset=utf-8")
    assert reply.status_code == 200
    assert world["pipeline"].switched == [True, False, True]


def test_the_page_and_the_video_are_not_guarded(world):
    assert world["client"].get("/", headers={"Origin": "http://evil.example"}).status_code == 200


def test_an_unknown_address_and_a_wrong_method_still_answer_404_and_405(world):
    client = world["client"]
    assert client.post("/api/no-such-thing", data="x", content_type="text/plain").status_code == 404
    assert client.post("/api/status", data="x", content_type="text/plain").status_code == 405


@pytest.mark.parametrize("method, path, origin, mimetype, refused", [
    ("POST", "/api/ask", None, "application/json", None),
    ("POST", "/api/ask", "http://127.0.0.1:5000", "application/json", None),
    ("POST", "/api/ask", None, "text/plain", (NOT_JSON, 415)),
    ("PUT", "/api/people/sara", None, "", (NOT_JSON, 415)),
    ("DELETE", "/api/search", None, "", (NOT_JSON, 415)),
    ("GET", "/api/status", None, "", None),
    ("POST", "/api/ask", "http://evil.example", "application/json", (OTHER_WEBSITE, 403)),
    ("POST", "/api/ask", "http://evil.example", "text/plain", (OTHER_WEBSITE, 403)),
    ("POST", "/video", "http://evil.example", "text/plain", None),       # not under /api/
])
def test_the_refusal_rule(method, path, origin, mimetype, refused):
    assert refusal(method, path, origin, mimetype, "http://127.0.0.1:5000/") == refused


def test_every_change_the_page_sends_is_json():
    """postJson and sendJson always send JSON with its Content-Type, an empty object when there is
    nothing to send; no other fetch in the page sends anything."""
    page = page_text()
    post = between(page, "async function postJson(", "\n}\n")
    assert 'headers: { "Content-Type": "application/json" }' in post and "body: JSON.stringify(body)" in post
    send = between(page, "async function sendJson(", "\n}\n")
    assert 'if (method !== "GET") {' in send
    assert 'options.headers = { "Content-Type": "application/json" };' in send
    assert "options.body = JSON.stringify(body === undefined ? {} : body);" in send
    script = page[page.index('<script>\n"use strict";'):]
    fetches = [m.start() for m in re.finditer(r"\bfetch\(", script)]
    # M23: the hosted voice's two requests (tests/test_app_voice.py checks what they send).
    allowed = [between(script, name, "\n}\n") for name in
               ("async function postJson(", "async function sendJson(", "async function poll(",
                "async function fetchClip(", "async function sendRecording(")]
    for position in fetches:
        assert any(script.index(body) <= position < script.index(body) + len(body) for body in allowed), position
    assert 'fetch("/api/status", { cache: "no-store", signal: controller.signal })' in script   # a GET


# ---- the page: the Auto-discover switch ---------------------------------------------------------


@pytest.fixture
def rendered(world):
    return world["client"].get("/").get_data(as_text=True)


def test_the_switch_sits_beside_the_in_view_heading_and_is_off(rendered):
    around = ids_around(rendered)
    for element in ("discoverRow", "discoverSwitch", "discoverState", "discoverPrivacy"):
        assert "watch" in around[element], element                     # in the live view, with the pills
    head = rendered[rendered.index('<div class="inview-head">'):rendered.index('<ul class="things"')]
    assert head.index("<h3>In view</h3>") < head.index('id="discoverSwitch"') < head.index('id="discoverState"')
    assert '<input type="checkbox" id="discoverSwitch" role="switch" aria-describedby="discoverState">' in rendered
    assert "<span>Auto-discover</span>" in rendered
    assert '<label class="switch discover-switch" id="discoverRow" hidden>' in rendered   # until the server has it
    assert " checked" not in tag(rendered, "discoverSwitch")                              # off on every page load


def tag(page, element_id):
    return re.search(r'<[a-z]+\b[^>]*\bid="' + element_id + r'"[^>]*>', page).group(0)


def test_the_privacy_sentence_is_under_the_switch_and_in_the_footer(rendered):
    assert rendered.count(PRIVACY) == 2
    footer = rendered[rendered.index('<footer class="footer">'):]
    assert PRIVACY in footer
    assert f'<p class="discover-privacy" id="discoverPrivacy" hidden>{PRIVACY}</p>' in rendered


def test_the_page_never_switches_it_on_by_itself_and_remembers_nothing():
    page = page_text()
    part = between(page, "// ---- auto-discover: a small hosted model", "// ---- people:")
    assert "localStorage" not in part and "readStored(" not in part and "store(" not in part
    assert page.count('postJson("/api/discover"') == 1
    assert page.count("sendDiscover(") == 3                    # its definition, the switch, the press made meanwhile
    change = between(page, 'discoverSwitch.addEventListener("change"', "\n});\n")
    assert "sendDiscover(discoverSwitch.checked)" in change
    # The switch shows what the server says, except while a press is on its way.
    draw = between(page, "function drawDiscover(", "\n}\n")
    assert "const fresh = !discoverSending && discoverNext === null && number >= discoverFromPoll;" in draw
    assert "if (fresh && discoverSwitch.checked !== discover.on) discoverSwitch.checked = discover.on;" in draw
    assert "drawDiscover(status, number);" in between(page, "async function poll(", "\n}\n")
    assert "discoverFromPoll = pollCount + 1;" in between(page, "async function sendDiscover(", "\n}\n")


def test_the_first_use_line_shows_once_per_visit_and_stays_while_it_is_on():
    page = page_text()
    assert 'let discoverFirst = "not yet";' in page
    draw = between(page, "function drawDiscover(", "\n}\n")
    assert 'if (on && discoverFirst === "not yet") discoverFirst = "showing";' in draw
    assert 'if (!on && discoverFirst === "showing") discoverFirst = "done";' in draw
    assert 'discoverPrivacy.hidden = discoverFirst !== "showing";' in draw


def test_the_words_beside_the_switch():
    page = page_text()
    words = between(page, "function discoverWords(", "\n}\n")
    assert '"looking every " + discoverEvery + " s"' in words and '"looking every few seconds"' in words
    assert '" · no looks yet today"' in words and '"1 look today"' in words and '" looks today"' in words
    assert "aboutDollars(discover.spent)" in words
    assert 'if (discover.state === "paused") return ["Paused", discover.note];' in words   # the note, in the page's layout
    draw = between(page, "function drawDiscover(", "\n}\n")
    assert 'make("strong", "", bold)' in draw and "discoverState.replaceChildren(...parts);" in draw
    assert 'changed("discover"' in draw                                                   # redrawn only on change


def about_dollars(amount):
    """A Python copy of the page's aboutDollars, checked against its text below."""
    if amount >= 0.01:
        return f"about ${amount:.2f}"
    if amount < 0.0000005:
        return "less than $0.000001"
    import math
    digits = min(6, math.ceil(-math.log10(amount)))
    return "about $" + f"{amount:.{digits}f}".rstrip("0").rstrip(".")


@pytest.mark.parametrize("amount, words", [
    (0.000182, "about $0.0002"), (0.000013, "about $0.00001"), (0.0019, "about $0.002"), (0.001, "about $0.001"),
    (0.0123, "about $0.01"), (0.25, "about $0.25"), (0.5, "about $0.50"), (0.0, "less than $0.000001"),
])
def test_money_is_shown_with_one_digit_that_matters_below_a_cent(amount, words):
    page = page_text()
    rule = between(page, "function aboutDollars(", "\n}\n")
    for piece in ('if (amount >= 0.01) return "about $" + amount.toFixed(2);',
                  'if (amount < 0.0000005) return "less than $0.000001";',
                  "const digits = Math.min(6, Math.ceil(-Math.log10(amount)));",
                  'return "about $" + amount.toFixed(digits).replace(/0+$/, "").replace(/\\.$/, "");'):
        assert piece in rule, piece
    assert about_dollars(amount) == words


def test_discovered_things_carry_a_small_tag_on_their_pills():
    page = page_text()
    assert "discovered: raw.discovered === true && raw.found !== true," in between(page, "function readObject(", "\n}\n")
    item = between(page, "function thingItem(", "\n}\n")
    assert 'make("span", "thing-tag", DISCOVERED_TAG)' in item and "tag.hidden = true;" in item
    assert 'button.append(make("span", "thing-name"), tag, make("span", "thing-detail"));' in item
    draw = between(page, "function drawObjects(", "\n}\n")
    assert 'button.classList.toggle("is-discovered", thing.discovered);' in draw
    assert 'button.querySelector(".thing-tag").hidden = !thing.discovered;' in draw
    assert 'const DISCOVERED_TAG = "discovered";' in page
    tag_css = css_rule(page, ".thing-tag")
    assert "backdrop-filter" not in tag_css and "animation" not in tag_css


def test_auto_discover_is_silent_and_follows_the_page_rules():
    page = page_text()
    part = between(page, "// ---- auto-discover: a small hosted model", "// ---- people:")
    for word in ("say(", "speak(", "innerHTML", "insertAdjacentHTML", "requestAnimationFrame", "setInterval"):
        assert word not in part, word
    assert "discover" not in between(page, "function say(", "\n}\n")
    for selector in (".inview-head", ".discover-state", ".discover-privacy"):
        rule = css_rule(page, selector)
        assert "backdrop-filter" not in rule and "infinite" not in rule, selector
    # In the full view the heading row and the line go with the rest of the page's words.
    assert ".stage.is-full .inview-head, .stage.is-full .discover-privacy," in page
    # Offline: the switch goes, like the other parts that need the program.
    assert "hideDiscover();" in between(page, "function setOnline(", "\n}\n")
