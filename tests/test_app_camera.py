"""M19 Parts B and C: POST /api/camera/power, and the page's button that turns the camera off and on.

The pipeline is a fake that records what it was asked. The page's script cannot run here
(no JavaScript engine), so its rules are checked by the shape of the script.
"""

import json

import pytest

from car_vision.app import CAMERA_POWER_SHAPE, NOT_JSON, OTHER_WEBSITE, create_app
from car_vision.config import ConfigError
from test_app import TEMPLATE, between, css_rule, ids_around
from test_pipeline import LINA_ID, FakeSource, people_config
from test_pipeline import book, make  # noqa: F401  (the fixtures, used by name)


class FakePipeline:
    def __init__(self, error=None):
        self.switched = []                    # every value set_camera() was given
        self.calls = []                       # any other method a request must never reach
        self.error = error                    # raised by set_camera(), as a source that cannot be made

    def set_camera(self, on):
        self.switched.append(on)
        if self.error is not None:
            raise self.error

    def status(self):
        return {}

    def sightings(self):                      # the page asks whether there is a memory
        return None

    def __getattr__(self, name):
        def called(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"a request reached pipeline.{name}")
        return called


class FakeAssistant:
    def __getattr__(self, name):
        raise AssertionError(f"a camera request reached assistant.{name}")


@pytest.fixture
def world():
    pipeline = FakePipeline()
    app = create_app(pipeline, FakeAssistant())
    app.testing = True
    return {"client": app.test_client(), "pipeline": pipeline}


def page_text():
    return TEMPLATE.read_text(encoding="utf-8")


# ---- POST /api/camera/power ----------------------------------------------------------------


@pytest.mark.parametrize("on", [True, False])
def test_the_route_switches_the_camera_and_answers_at_once(world, on):
    reply = world["client"].post("/api/camera/power", json={"on": on})
    assert reply.status_code == 200 and reply.get_json() == {"ok": True, "on": on}
    assert world["pipeline"].switched == [on]


@pytest.mark.parametrize("body", [
    {}, "on", "true", 1, 0, 1.5, [], [True], None, {"on": "true"}, {"on": 1}, {"on": 0}, {"on": None},
    {"on": [True]}, {"power": True}, {"on": True, "now": 1},
])
def test_anything_else_is_400_with_a_plain_message(world, body):
    reply = world["client"].post("/api/camera/power", data=json.dumps(body), content_type="application/json")
    assert reply.status_code == 400 and reply.get_json() == {"error": 'Send {"on": true} or {"on": false}.'}
    assert CAMERA_POWER_SHAPE == 'Send {"on": true} or {"on": false}.'
    assert world["pipeline"].switched == []


def test_a_missing_body_is_400(world):
    reply = world["client"].post("/api/camera/power", data="", content_type="application/json")
    assert reply.status_code == 400 and reply.get_json() == {"error": CAMERA_POWER_SHAPE}
    reply = world["client"].post("/api/camera/power", data="not json", content_type="application/json")
    assert reply.status_code == 400
    assert world["pipeline"].switched == []


def test_another_website_cannot_switch_the_camera(world):
    client = world["client"]
    reply = client.post("/api/camera/power", json={"on": False}, headers={"Origin": "http://evil.example"})
    assert reply.status_code == 403 and reply.get_json() == {"error": OTHER_WEBSITE}
    reply = client.post("/api/camera/power", data="on=false", content_type="application/x-www-form-urlencoded")
    assert reply.status_code == 415 and reply.get_json() == {"error": NOT_JSON}
    assert world["pipeline"].switched == [] and world["pipeline"].calls == []
    assert client.post("/api/camera/power", json={"on": False}, headers={"Origin": "http://localhost"}).status_code == 200


def test_a_camera_that_cannot_be_started_is_a_plain_400():
    error = ConfigError("stream.file_path: the clip data/samples/gone.mp4 does not exist\nmore detail")
    pipeline = FakePipeline(error=error)
    client = create_app(pipeline, FakeAssistant()).test_client()
    reply = client.post("/api/camera/power", json={"on": True})
    assert reply.status_code == 400
    assert reply.get_json() == {
        "error": "The camera could not be started: stream.file_path: the clip data/samples/gone.mp4 does not exist"}
    assert pipeline.switched == [True]


def test_no_photos_are_taken_while_the_camera_is_off(make, tmp_path, book):
    """The real pipeline (with fakes) behind the real route: the page gets the plain line with a 400."""
    pipeline = make(source=FakeSource(frames=3), config=people_config(tmp_path))
    pipeline.set_camera(False)
    app = create_app(pipeline, FakeAssistant(), people=book)
    reply = app.test_client().post(f"/api/people/{LINA_ID}/photos", json={"count": 3})
    assert reply.status_code == 400 and reply.get_json() == {"error": "Turn the camera on first to take photos."}
    assert pipeline.photo_progress() is None


def test_only_post_is_allowed(world):
    reply = world["client"].get("/api/camera/power")
    assert reply.status_code == 405 and "error" in reply.get_json()


# ---- the page: the button and the "Camera off" line ----------------------------------------------


@pytest.fixture
def rendered(world):
    return world["client"].get("/").get_data(as_text=True)


def test_the_button_is_first_in_the_tools_on_the_video(rendered):
    tools = between(rendered, '<div class="frame-tools">', "</div>")
    first = tools[tools.index("<button"):tools.index(">", tools.index("<button"))]
    assert 'id="cameraPower"' in first and 'class="frame-button power-button"' in first
    assert 'aria-pressed="true"' in first                                   # on at every start
    assert 'aria-label="Turn the camera off"' in first and 'title="Turn the camera off"' in first
    around = ids_around(rendered)
    assert "frame" in around["cameraPower"] and "watch" in around["cameraPower"]   # also in the full view
    assert 'class="power-strike"' in tools                                   # the line through the camera


def test_the_camera_off_line_has_a_turn_on_button_over_the_picture(rendered):
    around = ids_around(rendered)
    assert "cameraOff" in around["cameraOn"] and "frame" in around["cameraOff"]
    assert '<div class="camera-off" id="cameraOff" hidden>' in rendered
    part = between(rendered, '<div class="camera-off" id="cameraOff" hidden>', '<div class="frame-tools">')
    assert "<span>Camera off</span>" in part
    assert '<button class="pill primary small" id="cameraOn" type="button">Turn on</button>' in part


def test_the_page_follows_the_status_after_each_poll():
    page = page_text()
    poll = between(page, "async function poll(", "\n}\n")
    assert "drawStatusLine(status);\n    drawCamera(status);\n" in poll
    draw = between(page, "function drawCamera(", "\n}\n")
    assert "showCamera(isCameraOff(status));" in draw
    assert "if (powerHeldBack()) return;" in draw                           # busy until the answer comes
    held = between(page, "function powerHeldBack(", "\n}\n")
    assert "return powerSending || pollCount < powerFromPoll;" in held
    assert "status.camera.on === false" in between(page, "function isCameraOff(", "\n}\n")


def test_the_status_pill_is_held_back_with_the_button():
    """A status asked for before the answer must not say Live next to a button that says off."""
    page = page_text()
    shown = between(page, "function cameraOffShown(", "\n}\n")
    assert "return powerHeldBack() ? cameraIsOff : isCameraOff(status);" in shown
    assert "cameraOffShown(status)" in between(page, "function drawStatusLine(", "\n}\n")


def test_the_press_gives_up_after_eight_seconds():
    page = page_text()
    assert "const POWER_WAIT_MS = 8000;" in page
    send = between(page, "async function sendCameraPower(", "\n}\n")
    assert "const controller = new AbortController();" in send
    assert "setTimeout(() => controller.abort(), POWER_WAIT_MS);" in send
    assert 'postJson("/api/camera/power", { on }, controller.signal)' in send
    assert '"AbortError"' in send and "The camera did not answer." in send
    assert "clearTimeout(timer);" in send
    post = between(page, "async function postJson(", "\n}\n")
    assert "async function postJson(url, body, signal) {" in post and "    signal,\n" in post
    assert page.count(", controller.signal)") == 1                         # no other request has a limit


def test_the_status_pill_says_camera_off_after_an_error():
    page = page_text()
    assert '"is-off"' in between(page, "const STATE_CLASSES = [", "];")
    line = between(page, "function drawStatusLine(", "\n}\n")
    stopped = line.index('setState("is-stopped"')
    off = line.index('else if (cameraOffShown(status)) setState("is-off", "Camera off");')
    assert stopped < off < line.index('setState("is-ended"') < line.index('setState("is-live"')
    assert "background: var(--grey)" in css_rule(page, ".is-off .dot")


def test_a_press_sends_the_wish_and_shows_the_button_as_busy():
    page = page_text()
    part = between(page, "// ---- the camera: off and on ----", "// ---- in view:")
    assert page.index("function drawStatusLine(") < page.index("// ---- the camera: off and on ----")
    assert page.count('postJson("/api/camera/power"') == 1
    send = between(page, "async function sendCameraPower(", "\n}\n")
    assert "setPowerBusy(true);" in send and "setPowerBusy(false);" in send
    assert "showPowerProblem(problem)" in send                              # a failed press says so
    assert 'cameraPower.addEventListener("click", () => sendCameraPower(cameraIsOff));' in part
    assert 'cameraOnButton.addEventListener("click", () => sendCameraPower(true));' in part
    show = between(page, "function showCamera(", "\n}\n")
    for piece in ('cameraPower.setAttribute("aria-pressed", off ? "false" : "true");',
                  'const words = off ? "Turn the camera on" : "Turn the camera off";',
                  'cameraPower.setAttribute("aria-label", words);', "cameraPower.title = words;",
                  "cameraOffBox.hidden = !off;"):
        assert piece in show, piece
    problem = between(page, "function showPowerProblem(", "\n}\n")
    assert "setText(powerProblem, text);" in problem and "POWER_PROBLEM_MS" in problem


def test_nothing_can_be_inspected_while_the_camera_is_off():
    page = page_text()
    move = between(page, 'video.addEventListener("mousemove"', "\n});\n")
    click = between(page, 'video.addEventListener("click"', "\n});\n")
    assert "if (cameraIsOff) return;" in move and "if (cameraIsOff) return;" in click
    assert 'if (off) video.classList.remove("is-clickable");' in between(page, "function showCamera(", "\n}\n")


def test_the_button_goes_with_the_program():
    online = between(page_text(), "function setOnline(", "\n}\n")
    assert "cameraPower.hidden = !isOnline;" in online
    assert "showCamera(false);" in online                                  # a program that starts again is on


def test_the_camera_switch_is_silent_and_follows_the_page_rules():
    page = page_text()
    part = between(page, "// ---- the camera: off and on ----", "// ---- in view:")
    for word in ("say(", "speak(", "innerHTML", "insertAdjacentHTML", "requestAnimationFrame", "setInterval",
                 "localStorage", "fetch("):
        assert word not in part, word
    for selector in ('.frame-button.power-button[aria-pressed="false"]', ".capsule.power-problem",
                     ".camera-off", ".capsule.camera-off-capsule", ".power-button .power-strike"):
        rule = css_rule(page, selector)
        assert "backdrop-filter" not in rule and "infinite" not in rule and "blur" not in rule, selector
    assert page.index(".frame-button .shrink { display: none; }") < page.index(".frame-button.power-button {")
