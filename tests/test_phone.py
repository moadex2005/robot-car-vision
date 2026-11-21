"""The phone's address: what may be typed, what is refused, and the remembered file.

Pure logic and a temporary folder: no network, no phone.
"""

import ast
import json
import logging
import sys
from pathlib import Path

import pytest

from car_vision import phone
from car_vision.phone import load_remembered, phone_url, remember

PHONE_FILE = Path(__file__).resolve().parent.parent / "car_vision" / "phone.py"


@pytest.mark.parametrize("typed, expected", [
    ("192.168.1.7", "http://192.168.1.7:8080/video"),
    ("192.168.1.7:8080", "http://192.168.1.7:8080/video"),
    ("http://192.168.1.7:8080", "http://192.168.1.7:8080/video"),
    ("http://192.168.1.7:8080/", "http://192.168.1.7:8080/video"),
    ("http://192.168.1.7:8080/video", "http://192.168.1.7:8080/video"),
    ("  192.168.1.7:8080  ", "http://192.168.1.7:8080/video"),
    ("\t192.168.1.7\n", "http://192.168.1.7:8080/video"),
    ("HTTP://192.168.1.7:4747", "http://192.168.1.7:4747/video"),
    ("192.168.1.7/video", "http://192.168.1.7:8080/video"),
    ("10.0.0.5:1", "http://10.0.0.5:1/video"),
    ("172.16.0.9:65535", "http://172.16.0.9:65535/video"),
    ("172.31.255.254", "http://172.31.255.254:8080/video"),
    ("127.0.0.1:8080", "http://127.0.0.1:8080/video"),
])
def test_accepted_forms(typed, expected):
    assert phone_url(typed) == expected


@pytest.mark.parametrize("typed, message", [
    ("", phone.EMPTY),
    ("   ", phone.EMPTY),
    (None, phone.EMPTY),
    ("phone.local:8080", phone.NOT_IPV4),
    ("localhost", phone.NOT_IPV4),
    ("192.168.1", phone.NOT_IPV4),
    ("192.168.1.300", phone.NOT_IPV4),
    ("192.168.01.7", phone.NOT_IPV4),
    ("192.168.1.7.5", phone.NOT_IPV4),
    ("::1", phone.NOT_IPV4),
    ("[::1]:8080", phone.NOT_IPV4),
    ("http://[fe80::1]:8080/video", phone.NOT_IPV4),
    ("http://", phone.NOT_IPV4),
    ("8.8.8.8:8080", phone.NOT_LOCAL),
    ("172.32.0.1", phone.NOT_LOCAL),
    ("169.254.1.1", phone.NOT_LOCAL),
    ("0.0.0.0", phone.NOT_LOCAL),
    ("100.64.0.1", phone.NOT_LOCAL),
    ("192.168.1.7:0", phone.BAD_PORT),
    ("192.168.1.7:65536", phone.BAD_PORT),
    ("192.168.1.7:", phone.BAD_PORT),
    ("192.168.1.7:80a", phone.BAD_PORT),
    ("192.168.1.7:-1", phone.BAD_PORT),
    ("192.168.1.7:" + "9" * 5000, phone.BAD_PORT),
    ("https://192.168.1.7:8080/video", phone.NO_HTTPS),
    ("rtsp://192.168.1.7:8080/video", phone.ONLY_HTTP),
    ("file:///etc/passwd", phone.ONLY_HTTP),
    ("http://user:secret@192.168.1.7:8080/video", phone.NO_LOGIN),
    ("admin@192.168.1.7", phone.NO_LOGIN),
    ("192.168.1.7:8080/shot.jpg", phone.BAD_PATH),
    ("192.168.1.7:8080/video/", phone.BAD_PATH),
    ("192.168.1.7:8080//video", phone.BAD_PATH),
    ("192.168.1.7:8080?x=1", phone.BAD_PATH),
    ("192.168.1.7:8080/video#top", phone.BAD_PATH),
    ("192.168.1.7:8080/@8.8.8.8", phone.BAD_PATH),
])
def test_refusals_each_have_their_own_message(typed, message):
    with pytest.raises(ValueError) as refused:
        phone_url(typed)
    assert str(refused.value) == message


def test_every_refusal_message_is_different_and_short():
    messages = [phone.EMPTY, phone.NO_HTTPS, phone.ONLY_HTTP, phone.NO_LOGIN, phone.BAD_PATH,
                phone.NOT_IPV4, phone.BAD_PORT, phone.NOT_LOCAL]
    assert len(set(messages)) == len(messages)
    assert all(len(message) < 140 for message in messages)


# ---- the remembered address --------------------------------------------------------


def test_remember_then_load(tmp_path):
    path = tmp_path / "data" / "camera.json"                 # the folder does not exist yet
    remember(path, "http://192.168.1.7:8080/video")
    assert json.loads(path.read_text(encoding="utf-8")) == {"phone_url": "http://192.168.1.7:8080/video"}
    assert load_remembered(path) == "http://192.168.1.7:8080/video"


def test_remember_writes_the_full_address_and_refuses_a_bad_one(tmp_path):
    path = tmp_path / "camera.json"
    remember(path, "192.168.1.9:4747")
    assert load_remembered(path) == "http://192.168.1.9:4747/video"
    with pytest.raises(ValueError):
        remember(path, "8.8.8.8")
    assert load_remembered(path) == "http://192.168.1.9:4747/video"   # the old one is kept


def test_a_missing_file_gives_none_and_one_log_line(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="car_vision.phone")
    assert load_remembered(tmp_path / "camera.json") is None
    assert len(caplog.records) == 1


@pytest.mark.parametrize("content", [
    b"not json at all",
    b"",
    b"\xff\xfe\x00broken",
    b'["http://192.168.1.7:8080/video"]',
    b'{"address": "192.168.1.7"}',
    b'{"phone_url": 42}',
    b'{"phone_url": "http://8.8.8.8:8080/video"}',
    b'{"phone_url": "https://192.168.1.7:8080/video"}',
])
def test_a_broken_or_invalid_file_is_ignored_with_one_log_line(tmp_path, caplog, content):
    path = tmp_path / "camera.json"
    path.write_bytes(content)
    caplog.set_level(logging.INFO, logger="car_vision.phone")
    assert load_remembered(path) is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


def test_a_very_deeply_nested_file_is_ignored_with_one_log_line(tmp_path, caplog):
    path = tmp_path / "camera.json"
    path.write_text("[" * 200_000 + "]" * 200_000, encoding="utf-8")   # json raises RecursionError
    caplog.set_level(logging.INFO, logger="car_vision.phone")
    assert load_remembered(path) is None
    assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.WARNING


def test_any_failure_while_reading_is_one_log_line_and_none(tmp_path, caplog, monkeypatch):
    path = tmp_path / "camera.json"
    path.write_text('{"phone_url": "http://192.168.1.7:8080/video"}', encoding="utf-8")

    def broken(text):
        raise MemoryError("out of memory")

    monkeypatch.setattr(phone.json, "loads", broken)
    caplog.set_level(logging.INFO, logger="car_vision.phone")
    assert load_remembered(path) is None
    assert len(caplog.records) == 1


@pytest.mark.parametrize("first, second, same", [
    ("http://user:pw@192.168.1.7:8080/video", "http://192.168.1.7:8080/video", True),
    ("http://192.168.1.7/video", "http://192.168.1.7:8080/video", True),       # 8080 when none is given
    ("HTTP://192.168.1.7:8080/video", "192.168.1.7:8080", True),
    ("http://user:pw@192.168.1.7:8080/video", "http://192.168.1.7:4747/video", False),
    ("http://user:pw@192.168.1.8:8080/video", "http://192.168.1.7:8080/video", False),
    ("http://192.168.1.7:port/video", "http://192.168.1.7:8080/video", False),   # unreadable
    ("", "", False),
])
def test_same_phone_compares_host_and_port_only(first, second, same):
    assert phone.same_phone(first, second) is same


@pytest.mark.parametrize("url, shown", [
    ("http://user:secret@192.168.1.7:8080/video", "http://192.168.1.7:8080/video"),
    ("http://user:se/cr@et@192.168.1.7:8080/video", "http://192.168.1.7:8080/video"),
    ("http://192.168.1.7:8080/video", "http://192.168.1.7:8080/video"),
    ("user:pw@192.168.1.7:8080", "192.168.1.7:8080"),
])
def test_without_login_never_shows_the_password(url, shown):
    assert phone.without_login(url) == shown


def test_an_unreadable_file_is_ignored(tmp_path, caplog):
    folder = tmp_path / "camera.json"
    folder.mkdir()                                          # a folder where the file should be
    caplog.set_level(logging.INFO, logger="car_vision.phone")
    assert load_remembered(folder) is None
    assert len(caplog.records) == 1


def test_a_remembered_file_written_by_hand_is_normalised(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text('{"phone_url": " 192.168.1.7 "}', encoding="utf-8")
    assert load_remembered(path) == "http://192.168.1.7:8080/video"


def test_phone_uses_only_the_standard_library():
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    found = set()
    for node in ast.walk(ast.parse(PHONE_FILE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    assert found - allowed == set()
