"""The page answers only to its own address, never to a web name pointed at this computer.

A hostile website can point its own name at 127.0.0.1 ("DNS rebinding"). The
browser then sends this server requests whose Origin matches their Host, so
the same-origin rule alone would let them through: the website could read the
video, and spend the hosted voice's budget. Such a request always carries the
website's name in its Host header, and that is what is refused here.
"""

from __future__ import annotations

import pytest

from car_vision.app import WRONG_HOST, create_app, host_name, names_this_computer
from test_app import FakeAssistant, FakePipeline

PHONE_DOOR = {"car_vision.door": "phone"}


@pytest.fixture
def client():
    return create_app(FakePipeline(), FakeAssistant()).test_client()


@pytest.mark.parametrize("host, name", [
    ("127.0.0.1:5000", "127.0.0.1"), ("127.0.0.1", "127.0.0.1"), ("localhost:5000", "localhost"),
    ("[::1]:5000", "::1"), ("[::1]", "::1"), ("evil.example:5000", "evil.example"), ("", ""),
    ("[::1", ""), (" 127.0.0.1:5000 ", "127.0.0.1"),
])
def test_host_name_drops_the_port(host, name):
    assert host_name(host) == name


@pytest.mark.parametrize("host", ["127.0.0.1:5000", "127.0.0.1", "localhost:5000", "LOCALHOST", "[::1]:5000",
                                  "127.5.6.7:5000"])
def test_the_laptops_own_addresses_are_this_computer(host):
    assert names_this_computer(host, laptop=True)


@pytest.mark.parametrize("host", ["evil.example:5000", "evil.example", "127.0.0.1.evil.example",
                                  "localhost.evil.example:5000", "192.168.1.5:5000", "8.8.8.8", "", "[::1",
                                  "0x7f.1", "a:b:c"])
def test_any_other_name_is_not_the_laptops_door(host):
    assert not names_this_computer(host, laptop=True)


def test_the_phone_door_also_answers_to_a_private_network_address():
    assert names_this_computer("192.168.43.105:5443", laptop=False)
    assert names_this_computer("10.0.0.7:5443", laptop=False)
    assert names_this_computer("127.0.0.1:5443", laptop=False)
    assert not names_this_computer("8.8.8.8:5443", laptop=False)        # not a private address
    assert not names_this_computer("phone.evil.example:5443", laptop=False)


@pytest.mark.parametrize("path", ["/", "/api/status", "/video", "/api/voice", "/nothing-here"])
def test_a_web_name_pointed_at_this_computer_gets_nothing(client, path):
    reply = client.get(path, headers={"Host": "evil.example:5000"})
    assert reply.status_code == 403
    assert reply.get_json() == {"error": WRONG_HOST}


def test_a_matching_origin_does_not_help_a_web_name(client):
    """The rebinding case itself: the Origin equals the Host, so refusal() alone would pass it."""
    headers = {"Host": "evil.example:5000", "Origin": "http://evil.example:5000"}
    for path, body in (("/api/voice/say", {"text": "hello"}), ("/api/ask", {"question": "hi"}),
                       ("/api/camera/power", {"on": False})):
        reply = client.post(path, json=body, headers=headers)
        assert reply.status_code == 403, path
        assert reply.get_json() == {"error": WRONG_HOST}


@pytest.mark.parametrize("host", ["127.0.0.1:5000", "localhost:5000", "[::1]:5000"])
def test_the_page_opens_by_its_own_addresses(client, host):
    assert client.get("/", headers={"Host": host}).status_code == 200
    assert client.get("/api/status", headers={"Host": host}).status_code == 200


def test_through_the_phone_door_a_web_name_is_refused_before_anything_else(client):
    reply = client.get("/api/status", headers={"Host": "phone.evil.example:5443"}, environ_overrides=PHONE_DOOR)
    assert reply.status_code == 403
    assert reply.get_json() == {"error": WRONG_HOST}
    # The door's own kind of address gets the door's own answer (no door in this app: 401).
    reply = client.get("/api/status", headers={"Host": "192.168.43.105:5443"}, environ_overrides=PHONE_DOOR)
    assert reply.status_code == 401
