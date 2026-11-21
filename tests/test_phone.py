"""The phone's address: what may be typed, what is refused, and the remembered file.

Pure logic and a temporary folder: no network, no phone.
"""

import ast
import base64
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlsplit

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


# ---- the picture size ------------------------------------------------------------------

# What the real phone answered (the list as it came, in no useful order).
REAL_OFFERED = ["960x540", "800x600", "3840x2160", "1920x1080", "1280x720", "1600x1200",
                "640x480", "640x360", "352x288", "320x240", "176x144"]
STREAM = "http://192.168.1.7:8080/video"


def status_json(current="1920x1080", offered=REAL_OFFERED):
    return json.dumps({"video_connections": 1, "audio_connections": 0,
                       "curvals": {"video_size": current, "quality": "50", "orientation": "landscape"},
                       "avail": {"video_size": offered, "quality": ["1", "100"]}})


class FakeFetch:
    """Stands in for the network: records every address and hands back prepared answers."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append((url, timeout))
        answer = self.answers.pop(0) if self.answers else "nothing prepared"
        if isinstance(answer, BaseException):
            raise answer
        return answer


REAL_SIZES = [phone.parse_size(text) for text in REAL_OFFERED]


@pytest.mark.parametrize("text, size", [
    ("640x360", (640, 360)), ("1920x1080", (1920, 1080)), ("1x1", (1, 1)),
    ("640X360", None), ("640 x 360", None), (" 640x360", None), ("640x", None), ("x360", None),
    ("0x360", None), ("640x0", None), ("-640x360", None), ("640x360x2", None), ("640.5x360", None),
    ("123456x2", None), ("", None), (None, None), (640, None), ("٦٤٠x٣٦٠", None),
])
def test_parse_size(text, size):
    assert phone.parse_size(text) == size


def test_best_size_for_the_real_phone_is_640x360():
    assert phone.best_size(REAL_SIZES, (1920, 1080), 640, 480) == (640, 360)


def test_best_size_for_a_4_by_3_phone_is_640x480():
    assert phone.best_size(REAL_SIZES, (1920, 1440), 640, 480) == (640, 480)


def test_best_size_is_none_when_the_phone_already_sends_it():
    assert phone.best_size(REAL_SIZES, (640, 360), 640, 480) is None


def test_best_size_is_never_larger_than_the_current_size():
    # The page could show 640 x 360, but the phone sends 352 x 198: nothing smaller fits, nothing larger is asked for.
    assert phone.best_size([(640, 360), (1280, 720)], (352, 198), 640, 480) is None
    assert phone.best_size([(1280, 720)], (960, 540), 640, 480) is None


def test_best_size_never_picks_a_size_too_small_or_of_another_shape():
    assert phone.best_size([(320, 180), (176, 144)], (1920, 1080), 640, 480) is None   # too small to be sharp
    assert phone.best_size([(640, 480), (800, 600)], (1920, 1080), 640, 480) is None   # another shape
    assert phone.best_size([(1280, 720), (960, 540)], (1920, 1080), 640, 480) == (960, 540)


def test_best_size_allows_a_shape_two_percent_off_but_not_more():
    assert phone.best_size([(640, 356)], (1920, 1080), 640, 356) == (640, 356)   # 1.1 % wider
    assert phone.best_size([(640, 352)], (1920, 1080), 640, 352) is None         # 2.3 % wider


def test_best_size_with_nothing_offered_is_none():
    assert phone.best_size([], (1920, 1080), 640, 480) is None


def test_read_picture_sizes_on_the_real_answer():
    fetch = FakeFetch(status_json())
    current, offered = phone.read_picture_sizes(STREAM, fetch)
    assert current == (1920, 1080)
    assert offered == REAL_SIZES
    assert fetch.calls == [("http://192.168.1.7:8080/status.json?show_avail=1", 3.0)]


def test_read_picture_sizes_skips_junk_in_the_list():
    fetch = FakeFetch(status_json(offered=["640x360", "big", 7, None, "640x360", "0x0", "1280x720"]))
    assert phone.read_picture_sizes(STREAM, fetch) == ((1920, 1080), [(640, 360), (1280, 720)])


def test_collecting_the_offered_sizes_stops_at_max_sizes(monkeypatch):
    listed = [f"{width}x480" for width in range(1, 5001)]          # 5000 different sizes
    parsed = []
    real_parse = phone.parse_size
    monkeypatch.setattr(phone, "parse_size", lambda text: parsed.append(text) or real_parse(text))
    offered = phone.offered_sizes(listed)
    assert offered == [(width, 480) for width in range(1, phone.MAX_SIZES + 1)]
    assert len(parsed) == phone.MAX_SIZES                            # the rest is not even looked at
    assert phone.offered_sizes(["640x360"] * 5000 + ["1280x720"]) == [(640, 360), (1280, 720)]


@pytest.mark.parametrize("answer", [
    "<html>Another camera app</html>",
    "",
    "[1, 2, 3]",
    '{"curvals": {"video_size": "huge"}, "avail": {"video_size": ["640x360"]}}',
    '{"curvals": "1920x1080"}',
    '{"avail": {"video_size": ["640x360"]}}',
    "[" * 100_000 + "]" * 100_000,                       # json raises RecursionError
    b'{"curvals": {"video_size": "1920x1080"}}',         # not text
    None,
    OSError("No route to host"),
    TimeoutError("timed out"),
    ConnectionRefusedError("refused"),
    RuntimeError("anything at all"),
])
def test_read_picture_sizes_gives_up_quietly(answer, caplog):
    caplog.set_level(logging.DEBUG, logger="car_vision.phone")
    assert phone.read_picture_sizes(STREAM, FakeFetch(answer)) == (None, [])
    assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.DEBUG


@pytest.mark.parametrize("stream_url", [
    "http://8.8.8.8:8080/video",            # not on the local network
    "http://phone.local:8080/video",        # a name, not an address
    "https://192.168.1.7:8080/video",
    "http://[::1]:8080/video",
    "http://192.168.1.7:port/video",
    "",
    None,
])
def test_nothing_is_sent_for_an_address_that_is_not_a_local_phone(stream_url):
    fetch = FakeFetch(status_json(), "<result>Ok</result>")
    assert phone.read_picture_sizes(stream_url, fetch) == (None, [])
    assert phone.ask_for_size(stream_url, (640, 360), fetch) is False
    assert fetch.calls == []


def test_the_requests_go_to_the_host_and_port_of_the_stream_only():
    fetch = FakeFetch(status_json(), "<result>Ok</result>")
    phone.read_picture_sizes("http://10.0.0.5:4747/video", fetch)
    phone.ask_for_size("http://10.0.0.5:4747/", (640, 360), fetch)
    assert [url for url, _timeout in fetch.calls] == [
        "http://10.0.0.5:4747/status.json?show_avail=1",
        "http://10.0.0.5:4747/settings/video_size?set=640x360"]
    fetch = FakeFetch(status_json())
    phone.read_picture_sizes("http://192.168.1.7/video", fetch)              # no port: IP Webcam's 8080
    assert fetch.calls[0][0] == "http://192.168.1.7:8080/status.json?show_avail=1"


def test_ask_for_size_on_ok():
    fetch = FakeFetch('<?xml version="1.0" ?>\n<result>Ok</result>\n')
    assert phone.ask_for_size(STREAM, (640, 360), fetch) is True
    assert fetch.calls == [("http://192.168.1.7:8080/settings/video_size?set=640x360", 3.0)]


@pytest.mark.parametrize("answer", ["<result>Error</result>", "", "Okay", "<html>404</html>", None, b"Ok",
                                    OSError("refused"), TimeoutError("timed out"), ValueError("odd")])
def test_ask_for_size_on_another_answer_or_a_failure_is_false(answer, caplog):
    caplog.set_level(logging.DEBUG, logger="car_vision.phone")
    assert phone.ask_for_size(STREAM, (640, 360), FakeFetch(answer)) is False
    assert len(caplog.records) == 1


@pytest.mark.parametrize("size", [(0, 360), (640, 0), (-640, 360), (8001, 360), (640, 8001),
                                  (640.0, 360), (True, 360), ("640", "360"), [640, 360], (640,), None])
def test_ask_for_size_refuses_a_silly_size_and_sends_nothing(size):
    fetch = FakeFetch("<result>Ok</result>")
    assert phone.ask_for_size(STREAM, size, fetch) is False
    assert fetch.calls == []


def test_ask_for_size_accepts_8000():
    assert phone.ask_for_size(STREAM, (8000, 8000), FakeFetch("<result>Ok</result>")) is True


def test_a_login_goes_along_and_is_never_logged(caplog):
    caplog.set_level(logging.DEBUG, logger="car_vision.phone")
    stream = "http://me:secret@192.168.1.7:8080/video"
    failing = FakeFetch(OSError("could not reach http://me:secret@192.168.1.7:8080"),
                        "junk", OSError("me:secret"), "<result>Error</result>")
    assert phone.read_picture_sizes(stream, failing) == (None, [])
    assert phone.read_picture_sizes(stream, failing) == (None, [])
    assert phone.ask_for_size(stream, (640, 360), failing) is False
    assert phone.ask_for_size(stream, (640, 360), failing) is False
    assert len(caplog.records) == 4
    assert "secret" not in caplog.text
    assert failing.calls[0][0] == "http://me:secret@192.168.1.7:8080/status.json?show_avail=1"


def test_the_login_becomes_a_basic_auth_header():
    header = phone.login_header(urlsplit("http://me:se%40cret@192.168.1.7:8080/status.json"))
    assert header == {"Authorization": "Basic " + base64.b64encode(b"me:se@cret").decode("ascii")}
    assert phone.login_header(urlsplit("http://192.168.1.7:8080/status.json")) == {}


def test_phone_uses_only_the_standard_library():
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    found = set()
    for node in ast.walk(ast.parse(PHONE_FILE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    assert found - allowed == set()
