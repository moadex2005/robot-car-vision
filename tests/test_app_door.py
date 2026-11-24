"""M21 Part B and C: the phone door in app.py and the Camera box, with Flask's test client.

A request "comes through the phone door" here exactly as it does from the real phone
door's server: its WSGI environ has car_vision.door = "phone" (environ_overrides).
Everything else goes through the laptop's own door, which must not change at all.

No port is opened, no OpenSSL runs and .env is never read: main() is run with the
server, the certificate and the password replaced by fakes. The password used here
is made up for the tests. Every response of these tests, and every log line, is
checked for it.
"""

import json
import logging
import re
import socket
import ssl
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from werkzeug.test import Client

from car_vision import app as app_module
from car_vision import door as door_module
from car_vision.app import (DOOR_OFF, NOT_A_DOOR_ADDRESS, NOT_JSON, ONLY_ON_THE_LAPTOP, OTHER_WEBSITE,
                            PASSWORD_SHAPE, SIGN_IN_FIRST, TOO_BIG, WRONG_PASSWORD, PhoneDoorHandler,
                            close_phone_door, create_app, main, make_door_server, through_phone_door, wrap_listener)
from car_vision.door import NO_PASSWORD, DoorError, DoorSettings, PhoneDoor
from test_app import TEMPLATE, TEST_CONFIG, FakeAssistant, FakePipeline, between, fake_main  # noqa: F401

DOOR_TEMPLATE = TEMPLATE.parent / "door.html"
TEST_PASSWORD = "made-up-test-words"            # only for these tests
ADDRESS = "https://192.168.43.105:5443/"
PHONE_BASE = "https://192.168.43.105:5443"
PHONE_ORIGIN = "https://192.168.43.105:5443"
PHONE_ENVIRON = {"car_vision.door": "phone", "REMOTE_ADDR": "192.168.43.20"}
SETTINGS = DoorSettings(port=5443, certificate_dir=Path("unused"), session_seconds=3600.0, max_tries=3,
                        pause_seconds=300.0)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# ---- every response and log line of this file is checked for the password ----------------

SEEN = []                                       # every response the test clients got
FlaskClient = type(create_app(object(), object()).test_client())   # flask itself is imported only by app.py


class RecordingClient(FlaskClient):
    def open(self, *args, **kwargs):
        response = super().open(*args, **kwargs)
        SEEN.append(response)
        return response


@pytest.fixture(autouse=True)
def password_never_shows(caplog):
    caplog.set_level(logging.DEBUG)
    SEEN.clear()
    yield
    for response in SEEN:
        assert TEST_PASSWORD not in str(response.headers)
        if response.mimetype != "multipart/x-mixed-replace":      # the video never ends
            assert TEST_PASSWORD.encode("utf-8") not in response.get_data()
    assert TEST_PASSWORD not in caplog.text


# ---- fakes and helpers -----------------------------------------------------------------


class UntouchedPipeline:
    """Records any use: a request that is closed must never reach the pipeline."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def called(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"a closed request reached pipeline.{name}")
        return called


class UntouchedAssistant(UntouchedPipeline):
    pass


def make_door(clock=None, **changes):
    door = PhoneDoor(replace(SETTINGS, **changes), clock=clock or FakeClock())
    door.set_password(TEST_PASSWORD)
    door.opened([ADDRESS])
    return door


def make_app(door=None, pipeline=None, assistant=None):
    app = create_app(pipeline or FakePipeline(), assistant or FakeAssistant(), door=door)
    app.testing = True
    app.test_client_class = RecordingClient
    return app


def phone(client, method, path, environ=None, **kwargs):
    """One request through the phone door."""
    return client.open(path, method=method, base_url=PHONE_BASE,
                       environ_overrides={**PHONE_ENVIRON, **(environ or {})}, **kwargs)


def sign_in(client, password=TEST_PASSWORD, environ=None):
    return phone(client, "POST", "/door", json={"password": password}, environ=environ,
                 headers={"Origin": PHONE_ORIGIN})


@pytest.fixture
def door_world():
    door = make_door()
    app = make_app(door)
    return {"door": door, "app": app, "client": app.test_client()}


EXAMPLES = {"person_id": "sara", "sighting_id": "3", "filename": "style.css"}


def every_route(app):
    """(method, address) for every rule and method of the app, read from the app itself."""
    routes = []
    for rule in app.url_map.iter_rules():
        address = re.sub(r"<(?:[a-z]+:)?([a-z_]+)>", lambda m: EXAMPLES.get(m.group(1), "x"), rule.rule)
        routes += [(method, address) for method in sorted(rule.methods)]
    return routes


def is_closed(response, path, method="GET"):
    """401 for data and video, a redirect to the sign-in page for anything else."""
    if path.startswith("/api/") or path == "/video" or path.startswith("/video/"):
        if response.status_code != 401:
            return False
        return method == "HEAD" or response.get_json() == {"error": SIGN_IN_FIRST}   # HEAD has no body
    return response.status_code == 302 and response.headers["Location"].endswith("/door")


# ---- through the phone door, without a session: every address is closed ------------------


def test_every_route_of_the_app_is_closed_without_a_session():
    pipeline, assistant = UntouchedPipeline(), UntouchedAssistant()
    app = make_app(make_door(), pipeline, assistant)
    client = app.test_client()
    routes = [(method, address) for method, address in every_route(app) if address not in ("/door", "/door/leave")]
    assert ("GET", "/video") in routes and ("GET", "/") in routes and ("POST", "/api/ask") in routes
    assert len(routes) > 30
    for method, address in routes:
        response = phone(client, method, address, json={})
        assert is_closed(response, address, method), (method, address, response.status_code)
    assert pipeline.calls == [] and assistant.calls == []


@pytest.mark.parametrize("path", ["/", "/nothing", "/door/", "/DOOR", "/static/x.css", "/api", "/videos"])
def test_pages_redirect_to_the_sign_in_page(door_world, path):
    assert is_closed(phone(door_world["client"], "GET", path), path)


@pytest.mark.parametrize("path", ["/api/status", "/api/nothing", "/video", "/video/x", "/api/door", "/api/door/qr"])
def test_data_and_video_answer_401(door_world, path):
    response = phone(door_world["client"], "GET", path)
    assert response.status_code == 401 and response.get_json() == {"error": SIGN_IN_FIRST}


def test_the_sign_in_page_needs_no_session(door_world):
    response = phone(door_world["client"], "GET", "/door")
    assert response.status_code == 200 and response.mimetype == "text/html"
    page = response.get_data(as_text=True)
    assert 'type="password"' in page and "Sign in" in page
    assert response.headers["Cache-Control"] == "no-store"


def test_a_made_up_cookie_host_or_header_never_opens_the_door(door_world):
    client = door_world["client"]
    client.set_cookie("car_vision_door", "made-up-token", domain="192.168.43.105")
    client.set_cookie("session", "made-up-token", domain="192.168.43.105")
    tricks = [
        {},
        {"headers": {"Host": "127.0.0.1:5000"}},
        {"headers": {"Host": "localhost"}},
        {"headers": {"car_vision.door": "laptop"}},
        {"headers": {"Car-Vision-Door": "laptop", "X-Forwarded-For": "127.0.0.1"}},
        {"environ": {"REMOTE_ADDR": "127.0.0.1"}},
    ]
    for trick in tricks:
        for path in ("/", "/api/status", "/video"):
            assert is_closed(phone(client, "GET", path, **trick), path), (trick, path)


def test_a_header_never_makes_a_laptop_request_a_phone_request():
    app = make_app(make_door())
    client = app.test_client()
    headers = {"car_vision.door": "phone", "Car-Vision-Door": "phone", "Car_Vision.Door": "phone"}
    assert client.get("/", headers=headers).status_code == 200              # the laptop's door: no sign-in
    assert client.get("/door", headers=headers).status_code == 404          # and no /door


def test_with_no_door_nothing_comes_through_a_phone_door():
    client = make_app(None).test_client()
    for path in ("/", "/door", "/api/status", "/video"):
        response = phone(client, "GET", path)
        assert response.status_code == 401 and response.get_json() == {"error": ONLY_ON_THE_LAPTOP}
    assert sign_in(client).status_code == 401


@pytest.mark.parametrize("remote", ["192.168.43.20", "10.0.0.5", "0.0.0.0", "", "example.com"])
def test_an_unmarked_request_from_elsewhere_is_a_phone_request(remote):
    """Only a request from this computer itself counts as the laptop's, even without the mark."""
    client = make_app(make_door()).test_client()
    elsewhere = {"REMOTE_ADDR": remote}
    assert is_closed(client.get("/", environ_overrides=elsewhere), "/")
    assert is_closed(client.get("/api/door", environ_overrides=elsewhere), "/api/door")
    assert is_closed(client.get("/video", environ_overrides=elsewhere), "/video")
    no_door = make_app(None).test_client().get("/", environ_overrides=elsewhere)
    assert no_door.status_code == 401 and no_door.get_json() == {"error": ONLY_ON_THE_LAPTOP}


@pytest.mark.parametrize("remote", ["127.0.0.1", "127.0.0.5", "::1"])
def test_an_unmarked_request_from_this_computer_is_the_laptops(remote):
    client = make_app(make_door()).test_client()
    assert client.get("/", environ_overrides={"REMOTE_ADDR": remote}).status_code == 200
    assert client.get("/door", environ_overrides={"REMOTE_ADDR": remote}).status_code == 404


# ---- body sizes through the phone door -------------------------------------------------------


def test_a_sign_in_body_over_4_kb_is_413_before_it_is_read(door_world):
    client = door_world["client"]
    big = json.dumps({"password": "x" * 4100})
    for path in ("/door", "/door/leave"):
        response = phone(client, "POST", path, data=big, content_type="application/json")
        assert response.status_code == 413 and response.get_json() == {"error": TOO_BIG}
    assert door_world["door"].throttle.may_try("192.168.43.20")             # nothing was counted
    small = phone(client, "POST", "/door", data=json.dumps({"password": "x" * 4000}), content_type="application/json")
    assert small.status_code == 403                                          # under 4 kB: read, and wrong


def test_any_other_body_over_3_mb_is_413(door_world):
    client = door_world["client"]
    sign_in(client)
    big = json.dumps({"question": "x" * (3 * 1024 * 1024)})
    response = phone(client, "POST", "/api/ask", data=big, content_type="application/json",
                     headers={"Origin": PHONE_ORIGIN})
    assert response.status_code == 413 and response.get_json() == {"error": TOO_BIG}
    unsigned = make_app(make_door()).test_client()
    assert phone(unsigned, "POST", "/api/ask", data=big, content_type="application/json").status_code == 413


@pytest.mark.parametrize("method, path", [("POST", "/door"), ("POST", "/api/ask"), ("PUT", "/api/people/sara"),
                                          ("DELETE", "/api/memory")])
def test_a_change_that_does_not_say_its_size_is_413(door_world, method, path):
    client = door_world["client"]
    sign_in(client)
    response = phone(client, method, path, data=b'{"password": "x"}', content_type="application/json",
                     headers={"Transfer-Encoding": "chunked", "Origin": PHONE_ORIGIN})
    assert response.status_code == 413 and response.get_json() == {"error": TOO_BIG}


def test_the_laptop_door_takes_any_body_as_before():
    client = make_app(make_door()).test_client()
    big = json.dumps({"question": "x" * (3 * 1024 * 1024)})
    response = client.post("/api/ask", data=big, content_type="application/json")
    assert response.status_code == 400 and "too long" in response.get_json()["error"]


# ---- odd sign-in bodies ------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ['{"password": "made-up\\ud800words"}',
                                 '{"password": ' + "[" * 2000 + "]" * 2000 + "}",
                                 "[" * 3000 + "]" * 1000])
def test_an_unreadable_password_is_400_counted_and_logs_no_traceback(door_world, caplog, raw):
    assert len(raw.encode("utf-8")) < 4096
    client = door_world["client"]
    response = phone(client, "POST", "/door", data=raw, content_type="application/json")
    assert response.status_code == 400 and response.get_json() == {"error": PASSWORD_SHAPE}
    assert door_world["door"].throttle._failures["192.168.43.20"][0] == 1   # counted as a try
    assert "Traceback" not in caplog.text and not [r for r in caplog.records if r.levelno >= logging.ERROR]


# ---- signing in ---------------------------------------------------------------------------


def test_the_right_password_gives_a_strict_session_cookie(door_world):
    response = sign_in(door_world["client"])
    assert response.status_code == 200 and response.get_json() == {"ok": True}
    (cookie,) = response.headers.getlist("Set-Cookie")
    name, token = cookie.split(";")[0].split("=", 1)
    assert name == "car_vision_door" and len(token) >= 40
    flags = [part.strip() for part in cookie.split(";")[1:]]
    assert "Secure" in flags and "HttpOnly" in flags and "Path=/" in flags and "SameSite=Strict" in flags
    assert "Max-Age=3600" in flags
    assert door_world["door"].sessions.valid(token)


def test_with_a_session_the_page_the_data_and_the_video_open(door_world):
    client = door_world["client"]
    sign_in(client)
    page = phone(client, "GET", "/")
    assert page.status_code == 200 and "<title>Robot car vision</title>" in page.get_data(as_text=True)
    assert phone(client, "GET", "/api/status").status_code == 200
    answer = phone(client, "POST", "/api/ask", json={"question": "Who built the car?"}, headers={"Origin": PHONE_ORIGIN})
    assert answer.status_code == 200 and answer.get_json()["text"]
    video = phone(client, "GET", "/video", buffered=False)
    try:
        assert video.status_code == 200 and video.mimetype == "multipart/x-mixed-replace"
        assert next(iter(video.response)).startswith(b"--frame")
    finally:
        video.close()


def test_the_same_origin_rule_still_holds_through_the_phone_door(door_world):
    client = door_world["client"]
    sign_in(client)
    refused = phone(client, "POST", "/api/ask", json={"question": "Hi"}, headers={"Origin": "https://evil.example"})
    assert refused.status_code == 403 and refused.get_json() == {"error": OTHER_WEBSITE}


def test_signed_in_the_sign_in_page_goes_to_the_page(door_world):
    client = door_world["client"]
    sign_in(client)
    response = phone(client, "GET", "/door")
    assert response.status_code == 302 and response.headers["Location"].endswith("/")


def test_a_wrong_password_is_403_and_gives_no_cookie(door_world):
    for wrong in ("", "made-up-test-word", TEST_PASSWORD.upper(), TEST_PASSWORD + " ", "x" * 300):
        response = sign_in(door_world["client"], wrong, environ={"REMOTE_ADDR": f"10.0.0.{len(wrong) % 200}"})
        assert response.status_code == 403 and response.get_json() == {"error": WRONG_PASSWORD}
        assert "Set-Cookie" not in response.headers
    assert is_closed(phone(door_world["client"], "GET", "/"), "/")


@pytest.mark.parametrize("body", [None, {}, [], "secret", {"password": 12345678}, {"password": None}, {"pass": "x"}])
def test_a_wrong_shape_is_400_and_counts_as_a_try(door_world, body):
    response = phone(door_world["client"], "POST", "/door", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 400 and response.get_json() == {"error": PASSWORD_SHAPE}
    assert door_world["door"].throttle._failures["192.168.43.20"][0] == 1


def test_wrong_shapes_use_up_the_tries(door_world):
    client = door_world["client"]
    for _ in range(3):
        assert phone(client, "POST", "/door", data="{}", content_type="application/json").status_code == 400
    assert sign_in(client).status_code == 429


def test_too_many_wrong_passwords_from_everybody_closes_sign_in_for_all():
    clock = FakeClock()
    door = make_door(clock, max_tries=1, pause_seconds=60.0)             # 6 wrong ones for everybody
    client = make_app(door).test_client()
    assert sign_in(client, environ={"REMOTE_ADDR": "192.168.43.50"}).status_code == 200   # a phone signs in first
    for number in range(6):
        assert sign_in(client, "not-the-password", environ={"REMOTE_ADDR": f"10.0.0.{number}"}).status_code == 403
    refused = sign_in(client, environ={"REMOTE_ADDR": "192.168.43.51"})  # a new phone, the right password
    assert refused.status_code == 429 and refused.get_json() == {"error": "Too many tries. Wait 1 minute."}
    signed_in = phone(client, "GET", "/", environ={"REMOTE_ADDR": "192.168.43.50"})
    assert signed_in.status_code == 200                                  # a phone already in is not affected
    clock.now += 60
    assert sign_in(client, environ={"REMOTE_ADDR": "192.168.43.51"}).status_code == 200


def test_too_many_tries_is_429_even_with_the_right_password(door_world):
    client = door_world["client"]
    for _ in range(3):
        assert sign_in(client, "not-the-password").status_code == 403
    for password in ("not-the-password", TEST_PASSWORD):
        response = sign_in(client, password)
        assert response.status_code == 429
        assert response.get_json() == {"error": "Too many tries. Wait 5 minutes."}
        assert "Set-Cookie" not in response.headers
    tricked = sign_in(client, environ={}, password=TEST_PASSWORD)
    assert tricked.status_code == 429                                        # same address
    forwarded = phone(client, "POST", "/door", json={"password": TEST_PASSWORD},
                      headers={"X-Forwarded-For": "10.9.9.9", "Origin": PHONE_ORIGIN})
    assert forwarded.status_code == 429                                      # a header is not an address
    other = sign_in(client, environ={"REMOTE_ADDR": "192.168.43.21"})
    assert other.status_code == 200                                          # another phone may try


def test_the_pause_comes_from_the_settings_and_ends():
    clock = FakeClock()
    app = make_app(make_door(clock, max_tries=1, pause_seconds=60.0))
    client = app.test_client()
    assert sign_in(client, "not-the-password").status_code == 403
    assert sign_in(client).get_json() == {"error": "Too many tries. Wait 1 minute."}
    clock.now += 60
    assert sign_in(client).status_code == 200


def test_the_session_ends_with_its_lifetime():
    clock = FakeClock()
    app = make_app(make_door(clock))
    client = app.test_client()
    sign_in(client)
    clock.now += 3599
    assert phone(client, "GET", "/").status_code == 200
    clock.now += 1
    assert is_closed(phone(client, "GET", "/"), "/")


def test_leaving_ends_the_session(door_world):
    client = door_world["client"]
    response = sign_in(client)
    token = response.headers["Set-Cookie"].split(";")[0].split("=", 1)[1]
    left = phone(client, "POST", "/door/leave", json={}, headers={"Origin": PHONE_ORIGIN})
    assert left.status_code == 200 and left.get_json() == {"ok": True}
    assert "car_vision_door=;" in left.headers["Set-Cookie"] and "Secure" in left.headers["Set-Cookie"]
    assert not door_world["door"].sessions.valid(token)
    client.set_cookie("car_vision_door", token, domain="192.168.43.105")    # an old cookie kept by the phone
    assert is_closed(phone(client, "GET", "/"), "/")


@pytest.mark.parametrize("path", ["/door", "/door/leave"])
def test_the_sign_in_addresses_follow_the_same_origin_rule(door_world, path):
    client = door_world["client"]
    form = phone(client, "POST", path, data={"password": TEST_PASSWORD})
    assert form.status_code == 415 and form.get_json() == {"error": NOT_JSON}
    other = phone(client, "POST", path, json={"password": TEST_PASSWORD}, headers={"Origin": "https://evil.example"})
    assert other.status_code == 403 and other.get_json() == {"error": OTHER_WEBSITE}
    assert "Set-Cookie" not in form.headers and "Set-Cookie" not in other.headers
    assert door_world["door"].throttle.may_try("192.168.43.20")             # refused before any try counted


def test_the_password_is_compared_only_through_the_door(door_world):
    seen = []
    original = door_world["door"].matches
    door_world["door"].matches = lambda given: seen.append(given) or original(given)
    sign_in(door_world["client"])
    assert seen == [TEST_PASSWORD]
    source = Path(app_module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"(password|given)\w*\s*[!=]=|[!=]=\s*\w*(password|given)", source)
    door_source = Path(app_module.__file__).with_name("door.py").read_text(encoding="utf-8")
    assert "hmac.compare_digest(given.encode(\"utf-8\"), real.encode(\"utf-8\"))" in door_source


# ---- the laptop's door does not change --------------------------------------------------


def test_the_laptop_door_answers_byte_for_byte_as_without_a_door():
    without, with_door = make_app(None).test_client(), make_app(make_door()).test_client()
    for path in ("/", "/api/status", "/no-such-page", "/api/no-such-thing"):
        old, new = without.get(path), with_door.get(path)
        assert (old.status_code, old.get_data(), old.headers.get("Content-Type")) == \
               (new.status_code, new.get_data(), new.headers.get("Content-Type")), path
        assert "Set-Cookie" not in new.headers


def test_the_laptop_door_needs_no_password_and_sets_no_cookie():
    client = make_app(make_door()).test_client()
    responses = [client.get("/"), client.get("/api/status"),
                 client.post("/api/ask", json={"question": "Who built the car?"}),
                 client.post("/api/camera", json={"address": "192.168.1.7:8080"})]
    video = client.get("/video", buffered=False)
    try:
        responses.append(video)
        assert next(iter(video.response)).startswith(b"--frame")
    finally:
        video.close()
    assert [response.status_code for response in responses] == [200] * 5
    assert all("Set-Cookie" not in response.headers for response in responses)


@pytest.mark.parametrize("method, path", [("GET", "/door"), ("POST", "/door"), ("POST", "/door/leave"),
                                          ("GET", "/door/leave"), ("PUT", "/door")])
def test_the_laptop_door_has_no_door_page(method, path):
    client = make_app(make_door()).test_client()
    response = client.open(path, method=method, json={"password": TEST_PASSWORD})
    unknown = client.get("/no-such-page")
    assert response.status_code == 404 and response.get_data() == unknown.get_data()
    assert "Set-Cookie" not in response.headers


# ---- /api/door and its QR codes (laptop's door only) ---------------------------------------


def test_api_door_says_where_a_phone_opens_the_page():
    response = make_app(make_door()).test_client().get("/api/door")
    assert response.get_json() == {"open": True, "addresses": [ADDRESS]}
    assert response.headers["Cache-Control"] == "no-store"


def test_api_door_says_why_when_shut_or_off():
    door = make_door()
    door.shut("Phone door closed: no password.")
    assert make_app(door).test_client().get("/api/door").get_json() == \
           {"open": False, "why": "Phone door closed: no password."}
    assert make_app(None).test_client().get("/api/door").get_json() == {"open": False, "why": DOOR_OFF}


def test_the_qr_code_reads_back_to_the_address():
    response = make_app(make_door()).test_client().get("/api/door/qr", query_string={"address": ADDRESS})
    assert response.status_code == 200 and response.mimetype == "image/png"
    picture = cv2.imdecode(np.frombuffer(response.get_data(), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert picture.shape[0] >= 140
    text, _points, _code = cv2.QRCodeDetectorAruco().detectAndDecode(picture)
    assert text == ADDRESS


@pytest.mark.parametrize("address", ["", "https://evil.example/", "https://192.168.43.105:5443", "javascript:x",
                                     ADDRESS + "x"])
def test_any_other_qr_address_is_400(address):
    response = make_app(make_door()).test_client().get("/api/door/qr", query_string={"address": address})
    assert response.status_code == 400 and response.get_json() == {"error": NOT_A_DOOR_ADDRESS}
    no_door = make_app(None).test_client().get("/api/door/qr", query_string={"address": ADDRESS})
    assert no_door.status_code == 400


def test_api_door_is_not_there_through_the_phone_door(door_world):
    client = door_world["client"]
    sign_in(client)
    assert phone(client, "GET", "/api/door").status_code == 404
    assert phone(client, "GET", "/api/door/qr", query_string={"address": ADDRESS}).status_code == 404


# ---- the certificate's key is never served or shown --------------------------------------------


KEY_MARK = "-----BEGIN PRIVATE KEY----- made-up-key-for-a-test"


def test_a_key_that_does_not_load_is_an_error_without_its_content(tmp_path):
    certificate, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    certificate.write_text("not a certificate", encoding="utf-8")
    key.write_text(KEY_MARK, encoding="utf-8")
    with pytest.raises(DoorError) as caught:
        app_module.tls_context(certificate, key)
    assert "made-up-key" not in str(caught.value) and "PRIVATE KEY" not in str(caught.value)


@pytest.mark.parametrize("path", ["/data/door/key.pem", "/key.pem", "/static/../data/door/key.pem",
                                  "/static/..%2F..%2Fdata%2Fdoor%2Fkey.pem", "/door/key.pem"])
def test_no_address_serves_the_key(door_world, path):
    client = door_world["client"]
    sign_in(client)
    for response in (phone(client, "GET", path), make_app(make_door()).test_client().get(path)):
        assert response.status_code in (302, 404)
        assert b"PRIVATE KEY" not in response.get_data()


# ---- the phone door's server -----------------------------------------------------------------


def test_only_the_phone_doors_server_marks_requests():
    seen = []

    def app(environ, start_response):
        seen.append(environ.get("car_vision.door"))
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"ok"]

    Client(through_phone_door(app)).get("/", environ_overrides={"car_vision.door": "laptop"})
    Client(app).get("/", headers={"car_vision.door": "phone"})
    assert seen == ["phone", None]


class FakeServer:
    def __init__(self):
        self.served = threading.Event()
        self.shut = False
        self.closed = False

    def serve_forever(self):
        self.served.set()

    def shutdown(self):
        self.shut = True

    def server_close(self):
        self.closed = True


@pytest.fixture
def door_main(fake_main, monkeypatch, tmp_path):  # noqa: F811 (fake_main is a fixture from test_app)
    """main() with phone_door enabled, and the password, addresses, certificate and server faked."""
    config = {**TEST_CONFIG, "app": {**TEST_CONFIG["app"], "log_file": str(fake_main["log_file"])},
              "phone_door": {"enabled": True, "port": 5443, "certificate_dir": str(tmp_path / "door"),
                             "session_hours": 12, "max_tries": 5, "pause_minutes": 5}}
    made = {"servers": [], "certificates": []}

    def fake_make_door_server(app, host, port, context):
        server = FakeServer()
        made["servers"].append({"host": host, "port": port, "app": app, "context": context, "server": server})
        return server

    def fake_certificate(folder, addresses):
        made["certificates"].append((folder, list(addresses)))
        return folder / "cert.pem", folder / "key.pem"

    monkeypatch.setattr(app_module, "load_config", lambda: config)
    monkeypatch.setattr(app_module, "read_password", lambda: TEST_PASSWORD)
    monkeypatch.setattr(app_module, "lan_addresses", lambda: ["192.168.43.105"])
    monkeypatch.setattr(app_module, "ensure_certificate", fake_certificate)
    monkeypatch.setattr(app_module, "tls_context", lambda certificate, key: ("context", certificate, key))
    monkeypatch.setattr(app_module, "make_door_server", fake_make_door_server)
    return {**fake_main, "config": config, "made": made}


def test_main_opens_the_phone_door_and_closes_it_at_the_end(door_main, capsys, tmp_path):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Phone door: https://192.168.43.105:5443/  (password: PAGE_PASSWORD in .env)" in out
    assert TEST_PASSWORD not in out
    (made,) = door_main["made"]["servers"]
    assert (made["host"], made["port"]) == ("0.0.0.0", 5443)
    assert made["context"] == ("context", tmp_path / "door" / "cert.pem", tmp_path / "door" / "key.pem")
    assert door_main["made"]["certificates"] == [(tmp_path / "door", ["192.168.43.105"])]
    assert made["server"].served.wait(2.0)                                   # served in a thread of its own
    assert made["server"].shut and made["server"].closed                     # closed with the program
    (app, kwargs), = door_main["calls"]["run"]
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 5000          # the laptop's door as before
    app.test_client_class = RecordingClient
    assert app.test_client().get("/api/door").get_json() == {"open": True, "addresses": [ADDRESS]}
    assert made["app"] is app                                                # the same app behind both doors
    phone_side = Client(through_phone_door(made["app"]))                     # what the phone door serves
    assert phone_side.get("/", base_url=PHONE_BASE).status_code == 302
    assert phone_side.get("/door", base_url=PHONE_BASE).status_code == 200
    assert app.test_client().get("/door").status_code == 404                 # the laptop's server: no door


def raises(error):
    def broken(*args, **kwargs):
        raise error
    return broken


@pytest.mark.parametrize("problem", ["password", "unreadable env", "network", "port", "server", "certificate write",
                                     "certificate value", "anything else"])
def test_a_door_problem_prints_one_line_and_the_program_runs_on(door_main, monkeypatch, capsys, caplog, problem):
    secret = "secret-detail-" + TEST_PASSWORD
    if problem == "password":
        monkeypatch.setattr(app_module, "read_password", raises(DoorError(NO_PASSWORD)))
    elif problem == "unreadable env":                    # the real read_password, on a .env that is not UTF-8
        env_file = door_main["log_file"].parent.parent / "bad.env"
        env_file.write_bytes(b"PAGE_PASSWORD=\xff\xfe\n")
        monkeypatch.delenv("PAGE_PASSWORD", raising=False)
        monkeypatch.setattr(app_module, "read_password", lambda: door_module.read_password(env_file))
    elif problem == "network":
        monkeypatch.setattr(app_module, "lan_addresses", lambda: [])
    elif problem == "port":
        monkeypatch.setattr(app_module, "already_serving", lambda host, port: port == 5443)
    elif problem == "server":
        monkeypatch.setattr(app_module, "make_door_server", raises(OSError(10048, "address in use")))
    elif problem == "certificate write":
        monkeypatch.setattr(app_module, "ensure_certificate", raises(PermissionError(13, secret)))
    elif problem == "certificate value":
        monkeypatch.setattr(app_module, "ensure_certificate", raises(ValueError(secret)))
    else:
        monkeypatch.setattr(app_module, "lan_addresses", raises(RuntimeError(secret)))
    assert main([]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "Phone door" in line]
    assert len(lines) == 1 and lines[0].startswith("Phone door closed: ")
    assert TEST_PASSWORD not in out and "secret-detail" not in out and "Traceback" not in out
    assert "Traceback" not in caplog.text and "secret-detail" not in caplog.text
    (app, _kwargs), = door_main["calls"]["run"]
    assert app.test_client().get("/api/door").get_json() == {"open": False, "why": lines[0]}
    assert FakePipeline.instances[0].started and FakePipeline.instances[0].stopped


def test_nothing_about_the_door_runs_when_it_is_off(door_main, monkeypatch, capsys):
    off = {**door_main["config"], "phone_door": {**door_main["config"]["phone_door"], "enabled": False}}
    monkeypatch.setattr(app_module, "load_config", lambda: off)
    for name in ("read_password", "lan_addresses", "ensure_certificate", "make_door_server"):
        monkeypatch.setattr(app_module, name, lambda *args, **kwargs: pytest.fail("the door ran"))
    assert main([]) == 0
    assert "Phone door" not in capsys.readouterr().out


def test_a_wrong_door_setting_stops_before_anything_is_loaded(door_main, monkeypatch, capsys):
    wrong = {**door_main["config"], "phone_door": {**door_main["config"]["phone_door"], "port": "5443"}}
    monkeypatch.setattr(app_module, "load_config", lambda: wrong)
    assert main([]) == 1
    assert "Cannot start" in capsys.readouterr().out
    assert FakePipeline.instances == []


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.43.105", "::", "example.com", ""])
def test_the_laptops_door_must_stay_on_the_laptop(door_main, monkeypatch, capsys, host):
    wide = {**door_main["config"], "app": {**door_main["config"]["app"], "host": host}}
    monkeypatch.setattr(app_module, "load_config", lambda: wide)
    assert main([]) == 1
    out = capsys.readouterr().out
    assert out.startswith("Cannot start: app.host must be this laptop itself") and "phone_door" in out
    assert FakePipeline.instances == [] and door_main["calls"]["run"] == []   # before any model is loaded


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "::1"])
def test_loopback_hosts_are_the_laptops_door(door_main, monkeypatch, host):
    local = {**door_main["config"], "app": {**door_main["config"]["app"], "host": host}}
    monkeypatch.setattr(app_module, "load_config", lambda: local)
    assert main([]) == 0
    assert door_main["calls"]["run"][0][1]["host"] == host


# ---- a stalled client, and closing the door ---------------------------------------------------


class StuckServer(FakeServer):
    """A server whose shutdown never returns, as when its accept loop is stuck."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def shutdown(self):
        self.release.wait(30)


def test_closing_the_door_gives_up_within_its_limit(monkeypatch):
    monkeypatch.setattr(app_module, "DOOR_CLOSE_SECONDS", 0.3)
    server = StuckServer()
    started = time.monotonic()
    assert close_phone_door(server) is False
    assert time.monotonic() - started < 2.0
    server.release.set()
    assert close_phone_door(FakeServer()) is True and close_phone_door(None) is True


def test_the_pipeline_stops_even_when_the_door_does_not_close(door_main, monkeypatch, capsys):
    monkeypatch.setattr(app_module, "DOOR_CLOSE_SECONDS", 0.3)
    stuck = StuckServer()
    monkeypatch.setattr(app_module, "make_door_server", lambda app, host, port, context: stuck)
    started = time.monotonic()
    assert main([]) == 0
    assert time.monotonic() - started < 3.0
    assert FakePipeline.instances[0].stopped
    assert "did not close in time" in capsys.readouterr().out
    stuck.release.set()


def test_the_door_closes_within_three_seconds_by_default():
    assert app_module.DOOR_CLOSE_SECONDS == 3.0


def test_each_phone_door_connection_has_a_ten_second_limit():
    assert app_module.DOOR_SOCKET_SECONDS == 10 and PhoneDoorHandler.timeout == 10


def test_the_listener_does_not_handshake_inside_accept():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    with socket.socket() as plain:                       # never bound: no port is opened
        listener = wrap_listener(plain, context)
        try:
            assert isinstance(listener, ssl.SSLSocket)
            assert listener.server_side is True and listener.do_handshake_on_connect is False
        finally:
            listener.close()


class FakeConnection:
    def __init__(self, error):
        self.error = error
        self.handshakes = 0

    def do_handshake(self):
        self.handshakes += 1
        raise self.error


@pytest.mark.parametrize("error", [TimeoutError("timed out"), ssl.SSLError(1, "certificate unknown"),
                                   ConnectionResetError(10054, "reset")])
def test_a_failed_handshake_ends_that_connection_quietly(caplog, error):
    handler = PhoneDoorHandler.__new__(PhoneDoorHandler)  # no socket: only handle() is tried
    handler.connection = FakeConnection(error)
    handler.client_address = ("192.168.43.20", 50000)
    handler.handle()                                      # would fail on its missing parts if it went on
    assert handler.connection.handshakes == 1
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def openssl_found():
    try:
        door_module.find_openssl()
    except DoorError:
        return False
    return True


@pytest.mark.skipif(not openssl_found(), reason="no OpenSSL on this computer")
def test_a_silent_client_does_not_stop_the_next_one(tmp_path, monkeypatch):
    """The one test with a real socket, on 127.0.0.1 only: a client that connects and sends
    nothing must not hold up the next phone, and is dropped after the time limit."""
    monkeypatch.setattr(PhoneDoorHandler, "timeout", 1.5)  # the limit, shortened for the test
    certificate, key = door_module.ensure_certificate(tmp_path / "door", ["127.0.0.1"])
    server = make_door_server(make_app(make_door()), "127.0.0.1", 0, app_module.tls_context(certificate, key))
    port = server.socket.getsockname()[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    silent = socket.create_connection(("127.0.0.1", port), timeout=4)
    try:
        time.sleep(0.2)                                   # the server has accepted the silent one
        client_context = ssl.create_default_context()
        client_context.check_hostname = False
        client_context.verify_mode = ssl.CERT_NONE
        with socket.create_connection(("127.0.0.1", port), timeout=4) as raw:
            with client_context.wrap_socket(raw) as tls:
                tls.sendall(b"GET /door HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                reply = b""
                while chunk := tls.recv(65536):
                    reply += chunk
        assert reply.startswith(b"HTTP/1.1 200") and b"Sign in" in reply
        assert silent.recv(1) == b""                      # dropped after the limit, well within 4 s
    finally:
        silent.close()
        assert close_phone_door(server) is True


# ---- the page sends a phone whose session ended back to the sign-in page ---------------------


def test_a_401_from_the_status_sends_the_page_to_the_sign_in_page():
    poll = between(index_page(), "async function poll() {", "\n}\n")
    line = 'if (response.status === 401) window.location.replace("/door");'
    assert line in poll
    assert poll.index('fetch("/api/status"') < poll.index(line) < poll.index("if (!response.ok)")


# ---- the sign-in page ------------------------------------------------------------------------


def door_page():
    return DOOR_TEMPLATE.read_text(encoding="utf-8")


def test_the_sign_in_page_has_one_field_and_one_button():
    page = door_page()
    assert page.count("<input") == 1 and 'type="password"' in page and 'autocomplete="current-password"' in page
    assert page.count("<button") == 1 and ">Sign in</button>" in page
    assert 'maxlength="{{ max_password_chars }}"' in page


def test_the_sign_in_page_loads_nothing_from_outside_and_writes_text_only():
    page = door_page()
    assert "innerHTML" not in page and "outerHTML" not in page and "insertAdjacentHTML" not in page
    assert "http://" not in page and "https://" not in page
    assert not re.search(r"<link\b", page, re.IGNORECASE) and "@import" not in page
    assert not re.search(r"<script\b[^>]*\bsrc\s*=", page, re.IGNORECASE)
    assert "result.textContent = text;" in page


def test_the_sign_in_page_sends_the_password_as_json_in_the_body():
    script = between(door_page(), "async function signIn() {", "\n}\n")
    assert 'fetch("/door", {' in script and 'method: "POST"' in script
    assert '"Content-Type": "application/json"' in script
    assert "body: JSON.stringify({ password: field.value })" in script
    assert "localStorage.setItem" not in door_page() and "console." not in door_page()


def test_the_sign_in_page_has_a_light_and_a_dark_look():
    page = door_page()
    assert ':root, :root[data-theme="light"] {' in page and ':root[data-theme="dark"] {' in page
    assert "backdrop-filter: blur(" in page and 'name="viewport"' in page
    assert "prefers-reduced-motion: reduce" in page
    assert "infinite" not in page                                           # nothing moves all the time


# ---- the Camera box --------------------------------------------------------------------------


def index_page():
    return TEMPLATE.read_text(encoding="utf-8")


def test_the_camera_box_has_the_phone_door_part_under_the_delay_switch():
    page = index_page()
    box = between(page, '<div class="camera-box glass" id="cameraBox"', "\n</div>\n")
    assert box.index("Delay test") < box.index('id="doorPart"') < box.index('id="doorLeave"')
    assert '<div class="door-part" id="doorPart" hidden>' in box and "Open on your phone" in box
    assert "Same network. The phone warns once about the certificate; continue, then type the password." in box
    assert '<p class="camera-help door-shut" id="doorShut" hidden></p>' in box
    assert '<div class="door-leave" id="doorLeave" hidden>' in box and ">Sign out</button>" in box


def test_the_camera_box_asks_once_when_first_opened():
    page = index_page()
    assert "askDoor();" in between(page, "function openCameraBox() {", "\n}\n")
    assert page.count("askDoor()") == 2                                     # defined once, called once
    ask = between(page, "async function askDoor() {", "\n}\n")
    assert "if (doorAsked) return;" in ask and 'getJson("/api/door")' in ask
    assert "response.status === 404" in ask and "doorLeave.hidden = false;" in ask
    assert "setText(doorShut, asText(data.why));" in ask


def test_the_addresses_and_their_qr_codes_are_drawn_safely():
    draw = between(index_page(), "function drawDoor(addresses) {", "\n}\n")
    assert 'picture.src = "/api/door/qr?address=" + encodeURIComponent(address);' in draw
    assert "picture.width = 140;" in draw and 'make("figcaption", "", address)' in draw
    assert "innerHTML" not in draw
    rule = between(index_page(), ".camera-box .door-qr {", "}")
    assert "width: 140px" in rule


def test_sign_out_ends_the_session_and_goes_to_the_sign_in_page():
    page = index_page()
    part = between(page, 'doorSignOut.addEventListener("click"', "\n});\n")
    assert 'postJson("/door/leave", {})' in part and 'window.location.replace("/door");' in part
