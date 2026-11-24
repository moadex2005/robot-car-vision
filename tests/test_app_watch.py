"""M25 Parts C and D: POST and DELETE /api/watch, and the page's watch parts.

The routes run against the real pipeline built with fakes (never started: no frames are
needed to watch). The page is checked as text, in the style of tests/test_app_find.py; the
watch sentences themselves were also run in a real browser (see the milestone's report).

The owner's two rules: the notice is never spoken (no say(), speak() or answerArrived() is
reachable from it), and nothing about a watch costs money (no request reaches the assistant,
the hosted voice or the lookout).
"""

import re

import pytest

from car_vision.app import NOTHING_TO_WATCH, WATCH_SHAPE, create_app
from car_vision.pipeline import NO_PEOPLE
from test_app import TEMPLATE, between, css_rule, ids_around
from test_pipeline import SARA, FakeDetector, FakeLookout, FakeRecognizer, FakeSource
from test_pipeline import make  # noqa: F401  (the fixture, used by name)


class EnrolledRecognizer(FakeRecognizer):
    def enrolled_ids(self):
        return {"sara"}


class Untouchable:
    """Fails the test the moment anything is asked of it (the assistant, the hosted voice)."""

    def __init__(self, what):
        self._what = what

    def __getattr__(self, name):
        raise AssertionError(f"a watch reached the {self._what} ({name})")


@pytest.fixture
def served(make):
    """A client for a pipeline with fakes, where Sara is enrolled; nothing may reach the assistant or the voice."""
    lookout = FakeLookout()
    pipeline = make(source=FakeSource(), detector=FakeDetector(), lookout=lookout,
                    recognizer=EnrolledRecognizer(faces=(), people={"sara": SARA}))
    app = create_app(pipeline, Untouchable("assistant"), voice=Untouchable("hosted voice"))
    app.testing = True
    return app.test_client(), pipeline, lookout


def page_text():
    return TEMPLATE.read_text(encoding="utf-8")


# ---- POST and DELETE /api/watch ---------------------------------------------------------------


def test_post_watch_starts_watching_and_answers_the_names_as_shown(served):
    client, pipeline, lookout = served
    reply = client.post("/api/watch", json={"names": ["bottle", "sara", " Keys "]})
    assert reply.status_code == 200
    assert reply.get_json() == {"ok": True, "names": ["bottle", "Sara Ahmed", "Keys"]}
    status = client.get("/api/status").get_json()["watch"]
    assert [entry["name"] for entry in status["names"]] == ["bottle", "Sara Ahmed", "Keys"]
    assert [entry["person"] for entry in status["names"]] == [False, True, False]
    assert status["events"] == [] and lookout.looks == 0


def test_post_watch_replaces_the_list(served):
    client, pipeline, _lookout = served
    client.post("/api/watch", json={"names": ["bottle", "cup"]})
    client.post("/api/watch", json={"names": ["cup"]})
    assert [entry["name"] for entry in pipeline.status()["watch"]["names"]] == ["cup"]


@pytest.mark.parametrize("body", [{}, {"names": "bottle"}, {"text": "bottle"}, {"names": ["bottle"], "x": 1}, [], "bottle"])
def test_a_body_of_the_wrong_shape_is_400_with_a_plain_message(served, body):
    client, pipeline, _lookout = served
    reply = client.post("/api/watch", json=body)
    assert reply.status_code == 400 and reply.get_json() == {"error": WATCH_SHAPE}
    assert pipeline.status()["watch"]["names"] == []


def test_an_empty_list_is_400(served):
    client, _pipeline, _lookout = served
    reply = client.post("/api/watch", json={"names": []})
    assert reply.status_code == 400 and reply.get_json() == {"error": NOTHING_TO_WATCH}


@pytest.mark.parametrize("names, message", [
    (["person"], NO_PEOPLE),
    (["a", "b", "c", "d", "e"], "Watch for at most 4 things at once."),
    (["x" * 41], "Each name must be 1 to 40 characters long."),
    ([7], "Each name must be text."),
])
def test_a_value_error_from_the_pipeline_is_400_with_its_message(served, names, message):
    client, pipeline, _lookout = served
    reply = client.post("/api/watch", json={"names": names})
    assert reply.status_code == 400 and reply.get_json() == {"error": message}
    assert pipeline.status()["watch"]["names"] == []


def test_delete_watch_stops_watching(served):
    client, pipeline, _lookout = served
    client.post("/api/watch", json={"names": ["bottle"]})
    reply = client.delete("/api/watch", json={})
    assert reply.status_code == 200 and reply.get_json() == {"ok": True}
    assert pipeline.status()["watch"]["names"] == []


def test_the_watch_routes_refuse_another_website(served):
    client, pipeline, _lookout = served
    reply = client.post("/api/watch", json={"names": ["bottle"]}, headers={"Origin": "https://example.com"})
    assert reply.status_code == 403
    assert pipeline.status()["watch"]["names"] == []


def test_the_watch_routes_sit_directly_after_the_search_routes():
    source = (TEMPLATE.parents[1] / "app.py").read_text(encoding="utf-8")
    routes = re.findall(r'@app\.(?:get|post|delete)\("([^"]+)"\)', source)
    at = routes.index("/api/search")
    assert routes[at:at + 4] == ["/api/search", "/api/search", "/api/watch", "/api/watch"]


# ---- the page: the bell, the chips, the notice ----------------------------------------------------


@pytest.fixture
def rendered(served):
    client, _pipeline, _lookout = served
    return client.get("/").get_data(as_text=True)


def test_the_bell_sits_beside_find_and_the_chips_beside_the_search_chips_also_in_the_full_view(rendered):
    around = ids_around(rendered)
    assert "finderForm" in around["finderWatch"] and "watch" in around["finderWatch"]
    for element in ("watchChips", "notices"):
        assert "frame" in around[element] and "watch" in around[element], element
    form = between(rendered, '<form class="finder-form"', "</form>")
    assert form.index('id="finderFind"') < form.index('id="finderWatch"')
    assert 'aria-label="Tell me when it appears"' in form and 'class="bell"' in form
    assert rendered.index('id="findChips"') < rendered.index('id="watchChips"')
    assert '<div class="notices" id="notices" aria-live="polite"></div>' in rendered
    assert between(rendered, '<div class="banners">', '<div class="offline"').count('id="notices"') == 1


def test_the_chips_say_what_the_status_says_and_each_can_stop_its_name():
    page = page_text()
    words = between(page, "const WATCH_WORDS", "\n")
    for text in ('"getting ready"', '"waiting for room"', '"could not start"'):
        assert text in words
    chip = between(page, "function watchChipText(", "\n}\n")
    assert '"in view"' in chip and '"watching"' in chip
    item = between(page, "function watchChip(", "\n}\n")
    assert 'make("li", "find-chip watch-chip")' in item and 'make("button", "watch-chip-stop", "×")' in item
    assert "stopWatchingOne(item.dataset.name)" in item
    stop = between(page, "async function stopWatchingOne(", "\n}\n")
    assert "watchList.filter(" in stop and "sendWatchList(rest)" in stop
    draw = between(page, "function drawWatchChips(", "\n}\n")
    assert 'changed("watchChips", chips)' in draw                 # redrawn only when something changed
    assert '"Stop watching for " + chip.name' in draw


def test_the_watch_follows_every_status_and_is_cleared_when_the_program_is_not_running():
    page = page_text()
    assert "drawWatch(status, number);" in between(page, "function drawSearch(", "\n}\n")
    assert "clearWatch();" in between(page, "function clearSearch(", "\n}\n")
    clear = between(page, "function clearWatch(", "\n}\n")
    for part in ("watchChips.replaceChildren()", "notices.replaceChildren()", "noticeAfter = null", "drawTitle()"):
        assert part in clear


def test_the_notice_shows_each_new_event_once_and_none_from_before_a_reload():
    page = page_text()
    assert "const NOTICE_MS = 8000;" in page and "const NOTICE_RECENT_S = 10;" in page
    take = between(page, "function takeEvents(", "\n}\n")
    assert "event.id > noticeAfter" in take and "event.at >= recent" in take
    assert 'capitalised(event.name) + " appeared, " + placeWords(event.place)' in take
    show = between(page, "function showNotice(", "\n}\n")
    assert 'make("div", "capsule watch-notice")' in show and "noticeQueue.shift()" in show
    assert "NOTICE_MS" in show and '"is-leaving"' in show
    assert "innerHTML" not in between(page, "// ---- watching for things:", "// ---- auto-discover")


def test_the_tab_title_counts_notices_while_the_page_is_in_the_background():
    page = page_text()
    queue = between(page, "function queueNotice(", "\n}\n")
    assert 'document.visibilityState === "hidden"' in queue and "unseenNotices += 1" in queue
    title = between(page, "function drawTitle(", "\n}\n")
    assert '"(" + unseenNotices + ") " + pageTitle' in title and "document.title = title" in title
    seen = between(page, 'document.addEventListener("visibilitychange", () => {\n  if (document.visibilityState !== "visible")',
                    "\n});\n")
    assert "unseenNotices = 0" in seen and "drawTitle()" in seen


FORBIDDEN = ("say", "speak", "answerArrived", "showCaption", "fetch", "postJson", "sendJson", "getJson",
             "conversationNext", "conversationOwes", "tellFound")
KEYWORDS = {"if", "for", "while", "return", "function", "switch", "catch", "typeof", "new"}


def reachable(page, names):
    """Every function of the page reachable from these, with the text of each (name -> body)."""
    bodies, waiting = {}, list(names)
    while waiting:
        name = waiting.pop()
        if name in bodies:
            continue
        if f"function {name}(" in page:
            body = between(page, f"function {name}(", "\n}\n")
        elif re.search(rf"^const {name} = ", page, re.MULTILINE):
            body = re.search(rf"^const {name} = .*$", page, re.MULTILINE).group(0)
        else:
            continue                                            # a browser function: setTimeout, Math.max, ...
        bodies[name] = body
        for called in re.findall(r"(?<![.\w])([A-Za-z_]\w*)\s*\(", body):
            if called not in KEYWORDS:
                waiting.append(called)
    return bodies


def test_no_say_or_speak_is_reachable_from_the_notice():
    page = page_text()
    seen = between(page, 'document.addEventListener("visibilitychange", () => {\n  if (document.visibilityState !== "visible")',
                   "\n});\n")
    bodies = reachable(page, ["takeEvents", "queueNotice", "showNotice", "drawTitle", "clearWatch"])
    assert {"takeEvents", "queueNotice", "showNotice", "drawTitle", "capitalised", "placeWords", "bell"} <= set(bodies)
    for name, body in [*bodies.items(), ("visibilitychange", seen)]:
        for word in FORBIDDEN:
            assert not re.search(rf"(?<![.\w]){word}\s*\(", body), (name, word)
    # clearWatch may end this page's own pending request (an answer, never the notice): that path
    # is checked by itself below, so the notice's own functions are proved silent without it.
    notice = reachable(page, ["takeEvents", "queueNotice", "showNotice", "drawTitle"])
    assert "watchAnswered" not in notice


def test_clear_watch_only_ends_a_pending_request_and_never_speaks_a_notice():
    clear = between(page_text(), "function clearWatch(", "\n}\n")
    assert 'watchAnswered(watching, "Program not running: the watch could not be followed.", "problem")' in clear


def test_the_answer_to_starting_a_watch_goes_where_a_finds_answer_goes():
    page = page_text()
    sentences = between(page, "function watchSentences(", "\n}\n")
    assert '" in view right now"' in sentences and '"I\'ll tell you when I see "' in sentences
    assert '"I\'ll tell you when " + appearsAgain(there) + " again."' in sentences
    assert "entry.person ? entry.name : anyOf(entry.name)" in page              # no "a" before a person
    chat = between(page, "function watchFromChat(", "\n}\n")
    assert 'answerArrived("chat", text, kind === "problem", kind === "refused")' in chat
    bell = between(page, 'finderWatch.addEventListener("click"', "\n});\n")
    assert 'answerArrived("stage", answer, kind === "problem", kind === "refused")' in bell
    start = between(page, "async function startWatch(", "\n}\n")
    assert "const before = [...watchList];" in start and "sendWatchList(all)" in start


def test_a_watch_sentence_never_goes_to_the_assistant():
    page = page_text()
    chat = between(page, "function sendChat()", "\n}\n")
    assert chat.index("watchRequest(question)") < chat.index("findRequest(question)") < chat.index("askChat(question)")
    assert "watchFromChat(question, watched);\n    return true;" in chat
    rule = between(page, "function watchRequest(", "\n}\n") + between(page, "function watchedNames(", "\n}\n")
    assert "fetch(" not in rule and "document" not in rule                # pure: text in, names or null out
    script = page[page.index("<script>\n\"use strict\";"):]
    assert script.index("function watchRequest(") < script.index("// ---- the page coming alive")


def test_the_timeline_shows_an_appearance_with_its_time_and_a_bell():
    page = page_text()
    assert '["person", "watch"].includes(raw.kind)' in between(page, "function readMoment(", "\n}\n")
    name = between(page, "function momentName(", "\n}\n")
    assert 'capitalised(moment.name || moment.label) + " appeared, " + placeWords(moment.place)' in name
    assert '"at " + clock' in between(page, "function momentDetail(", "\n}\n")
    item = between(page, "function momentItem(", "\n}\n")
    assert 'moment.kind === "watch"' in item and "face.append(bell())" in item


def test_the_watch_parts_are_solid_calm_and_fit_a_phone():
    page = page_text()
    notice = css_rule(page, ".capsule.watch-notice")
    assert "background: #1c1c1e" in notice and "backdrop-filter" not in notice       # no blur over the video
    assert "animation: bell-ring 0.9s var(--ease-out) 0.25s 1" in css_rule(page, ".watch-notice .bell")   # rings once
    styles = between(page, "/* Watching for things (M25)", ".is-full .frame-button .expand")
    assert "infinite" not in styles and "backdrop-filter" not in styles
    phone = page[page.index("/* ---- phone: up to 640 px wide ----"):]
    assert ".capsule.watch-notice {" in phone and ".watch-chip-stop { width: 40px; height: 40px; }" in phone
    assert ".stage:not(.is-full) .notices { position: absolute;" in phone


def test_the_page_keeps_its_own_list_so_two_quick_presses_lose_no_name():
    page = page_text()
    send = between(page, "function sendWatchList(", "\n}\n")
    assert send.index("watchList = [...names];") < send.index("postJson(")      # changed at once
    assert "watchSent.then(" in send                                         # one change after the other
    assert 'postJson("/api/watch", { names })' in send and 'sendJson("DELETE", "/api/watch")' in send
    draw = between(page, "function drawWatch(", "\n}\n")
    assert "if (!watchSending && number >= watchListFrom) watchList = " in draw    # the server's list, once newer
    assert "sendWatchList([])" in between(page, "async function stopWatching(", "\n}\n")


def test_a_notice_that_waited_more_than_a_minute_is_dropped():
    page = page_text()
    assert "const NOTICE_STALE_S = 60;" in page
    show = between(page, "function showNotice(", "\n}\n")
    assert "while (noticeQueue.length && noticeQueue[0].at < stale) noticeQueue.shift();" in show
    assert "unseenNotices += 1" in between(page, "function queueNotice(", "\n}\n")     # still counted in the title