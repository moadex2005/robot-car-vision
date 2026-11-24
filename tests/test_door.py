"""M21 Part A: car_vision/door.py, the parts of the phone door that need no web server.

Fake clocks, a fake `run` and a fake PATH: no OpenSSL, network or waiting is needed,
except in the one real certificate test, which is skipped where no OpenSSL is found
and writes only into its own temporary folder. The password used here is made up
for the tests and is not anybody's real one.
"""

import os
import ssl
import subprocess
import threading
import time
from pathlib import Path

import pytest

from car_vision import door as door_module
from car_vision.config import ConfigError, resolve_path
from car_vision.door import (LONG_PASSWORD, MOST_SESSIONS, NO_OPENSSL, DoorError, DoorSettings, PhoneDoor, Sessions,
                             Throttle, can_be_typed, ensure_certificate, find_openssl, is_loopback, lan_addresses,
                             minutes_words, password_matches, read_password, read_settings)

TEST_PASSWORD = "made-up-test-words"      # only for these tests

DOOR_CONFIG = {"enabled": True, "port": 5443, "certificate_dir": "data/door", "session_hours": 12,
               "max_tries": 5, "pause_minutes": 5}


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def settings(**changes):
    return read_settings({"phone_door": {**DOOR_CONFIG, **changes}})


# ---- settings ---------------------------------------------------------------------


def test_settings_are_read_in_seconds_with_a_full_path():
    assert settings() == DoorSettings(port=5443, certificate_dir=resolve_path("data/door"),
                                      session_seconds=12 * 3600.0, max_tries=5, pause_seconds=300.0)


def test_the_door_is_off_when_not_enabled_or_not_configured():
    assert settings(enabled=False) is None
    assert read_settings({"phone_door": {"enabled": False}}) is None     # nothing else is read
    assert read_settings({}) is None                                      # older settings files


@pytest.mark.parametrize("changes", [
    {"enabled": "yes"}, {"enabled": 1}, {"port": "5443"}, {"port": 0}, {"port": 70000}, {"port": True},
    {"port": 5443.0}, {"certificate_dir": 3}, {"session_hours": 0}, {"session_hours": -1},
    {"session_hours": "12"}, {"session_hours": True}, {"max_tries": 0}, {"max_tries": 2.5},
    {"pause_minutes": 0}, {"pause_minutes": None},
])
def test_wrong_values_are_config_errors(changes):
    with pytest.raises(ConfigError):
        settings(**changes)


@pytest.mark.parametrize("missing", ["enabled", "port", "certificate_dir", "session_hours", "max_tries",
                                     "pause_minutes"])
def test_a_missing_setting_is_a_config_error(missing):
    values = {key: value for key, value in DOOR_CONFIG.items() if key != missing}
    with pytest.raises(ConfigError):
        read_settings({"phone_door": values})


def test_a_phone_door_that_is_not_a_section_is_a_config_error():
    with pytest.raises(ConfigError):
        read_settings({"phone_door": True})


def test_the_pause_in_words():
    assert (minutes_words(300), minutes_words(60), minutes_words(30)) == ("5 minutes", "1 minute", "0.5 minutes")


# ---- the password -------------------------------------------------------------------


@pytest.fixture
def no_password_in_environment(monkeypatch):
    monkeypatch.delenv("PAGE_PASSWORD", raising=False)


def test_the_password_is_read_from_the_env_file(tmp_path, no_password_in_environment):
    env_file = tmp_path / ".env"
    env_file.write_text(f"OTHER=1\nPAGE_PASSWORD={TEST_PASSWORD}\n", encoding="utf-8")
    assert read_password(env_file) == TEST_PASSWORD
    assert "PAGE_PASSWORD" not in os.environ                 # not copied into the environment


def test_the_environment_wins_over_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("PAGE_PASSWORD=from-the-file-1\n", encoding="utf-8")
    monkeypatch.setenv("PAGE_PASSWORD", "from-the-environment")
    assert read_password(env_file) == "from-the-environment"


@pytest.mark.parametrize("text", [None, "", "OTHER=1\n", "PAGE_PASSWORD=\n", "PAGE_PASSWORD=   \n"])
def test_a_missing_password_is_a_plain_door_error(tmp_path, no_password_in_environment, text):
    env_file = tmp_path / ".env"
    if text is not None:
        env_file.write_text(text, encoding="utf-8")
    with pytest.raises(DoorError) as caught:
        read_password(env_file)
    assert "PAGE_PASSWORD" in str(caught.value) and ".env" in str(caught.value)


def test_a_short_password_is_refused_without_showing_it(tmp_path, no_password_in_environment):
    env_file = tmp_path / ".env"
    env_file.write_text("PAGE_PASSWORD=abc1234\n", encoding="utf-8")     # 7 characters
    with pytest.raises(DoorError) as caught:
        read_password(env_file)
    assert "abc1234" not in str(caught.value) and "abc1234" not in repr(caught.value)
    assert "8 characters" in str(caught.value)


def test_eight_characters_are_enough(tmp_path, no_password_in_environment):
    env_file = tmp_path / ".env"
    env_file.write_text("PAGE_PASSWORD=abcd1234\n", encoding="utf-8")
    assert read_password(env_file) == "abcd1234"


def test_a_password_longer_than_200_characters_is_refused(tmp_path, no_password_in_environment):
    env_file = tmp_path / ".env"
    env_file.write_text("PAGE_PASSWORD=" + "a" * 201 + "\n", encoding="utf-8")
    with pytest.raises(DoorError) as caught:
        read_password(env_file)
    assert str(caught.value) == LONG_PASSWORD and "aaaa" not in str(caught.value)
    env_file.write_text("PAGE_PASSWORD=" + "a" * 200 + "\n", encoding="utf-8")
    assert read_password(env_file) == "a" * 200


def test_an_env_file_that_is_not_utf8_is_a_plain_door_error(tmp_path, no_password_in_environment):
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"PAGE_PASSWORD=caf\xe9-\xff\xfe-made-up\n")
    with pytest.raises(DoorError) as caught:
        read_password(env_file)
    assert "could not be read (UnicodeDecodeError)" in str(caught.value) and "made-up" not in str(caught.value)


def test_an_env_file_that_cannot_be_read_is_a_plain_door_error(tmp_path, no_password_in_environment, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("PAGE_PASSWORD=whatever-it-is\n", encoding="utf-8")

    def refused(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(door_module.dotenv, "dotenv_values", refused)
    with pytest.raises(DoorError) as caught:
        read_password(env_file)
    assert "could not be read (PermissionError)" in str(caught.value)


def test_text_with_a_lone_surrogate_cannot_be_typed_and_never_matches():
    assert can_be_typed("ثلاثة-أبواب-🚗") and not can_be_typed("abc\ud800defgh")
    assert password_matches("abc\ud800defgh", TEST_PASSWORD) is False      # no UnicodeEncodeError
    assert password_matches(TEST_PASSWORD, "abc\ud800defgh") is False


@pytest.mark.parametrize("address, loopback", [
    ("127.0.0.1", True), ("127.8.9.10", True), ("localhost", True), ("LOCALHOST", True), ("::1", True),
    ("[::1]", True), ("::ffff:127.0.0.1", True), ("0.0.0.0", False), ("192.168.43.105", False), ("::", False),
    ("10.0.0.1", False), ("example.com", False), ("", False), (None, False),
])
def test_is_loopback(address, loopback):
    assert is_loopback(address) is loopback


def test_password_matches():
    assert password_matches(TEST_PASSWORD, TEST_PASSWORD)
    assert not password_matches(TEST_PASSWORD + "x", TEST_PASSWORD)
    assert not password_matches("", TEST_PASSWORD)
    assert not password_matches(TEST_PASSWORD.upper(), TEST_PASSWORD)
    assert password_matches("ثلاثة-أبواب-🚗", "ثلاثة-أبواب-🚗")              # not only ASCII
    assert not password_matches("ثلاثة-أبواب-🚗", "ثلاثة-أبواب-🚙")
    assert not password_matches("café-door", "café-door")             # compared as typed


def test_the_door_never_shows_its_password():
    door = PhoneDoor(settings())
    assert not door.matches(TEST_PASSWORD)                                  # no password set yet
    door.set_password(TEST_PASSWORD)
    assert door.matches(TEST_PASSWORD) and not door.matches("something-else")
    assert TEST_PASSWORD not in repr(door) and TEST_PASSWORD not in str(door)


# ---- sessions -------------------------------------------------------------------------


def test_a_session_token_is_long_and_random():
    sessions = Sessions(60)
    tokens = {sessions.open() for _ in range(20)}
    assert len(tokens) == 20 and all(len(token) >= 40 for token in tokens)
    assert all(sessions.valid(token) for token in tokens)


def test_a_session_expires_with_the_clock():
    clock = FakeClock()
    sessions = Sessions(3600, clock=clock)
    token = sessions.open()
    clock.now += 3599
    assert sessions.valid(token)
    clock.now += 1
    assert not sessions.valid(token)


def test_a_closed_session_is_gone_and_others_stay():
    sessions = Sessions(60)
    first, second = sessions.open(), sessions.open()
    sessions.close(first)
    assert not sessions.valid(first) and sessions.valid(second)
    sessions.close("not-a-token")
    sessions.close(None)
    assert sessions.valid(second)


@pytest.mark.parametrize("token", [None, "", "made-up", 42, b"bytes"])
def test_anything_else_is_not_a_session(token):
    sessions = Sessions(60)
    sessions.open()
    assert not sessions.valid(token)


def test_at_most_fifty_sessions_and_the_oldest_goes_first():
    clock = FakeClock()
    sessions = Sessions(3600, clock=clock)
    tokens = []
    for _ in range(MOST_SESSIONS + 1):
        tokens.append(sessions.open())
        clock.now += 1
    assert MOST_SESSIONS == 50
    assert not sessions.valid(tokens[0])
    assert all(sessions.valid(token) for token in tokens[1:])


# ---- the throttle ---------------------------------------------------------------------


def test_the_throttle_allows_max_tries_wrong_passwords():
    throttle = Throttle(3, 300, clock=FakeClock())
    for _ in range(3):
        assert throttle.may_try("10.0.0.2")
        throttle.failed("10.0.0.2")
    assert not throttle.may_try("10.0.0.2")


def test_the_pause_ends_and_the_client_is_forgotten():
    clock = FakeClock()
    throttle = Throttle(2, 300, clock=clock)
    throttle.failed("10.0.0.2")
    throttle.failed("10.0.0.2")
    clock.now += 299
    assert not throttle.may_try("10.0.0.2")
    clock.now += 1
    assert throttle.may_try("10.0.0.2")
    throttle.failed("10.0.0.2")                         # counting starts again from nothing
    assert throttle.may_try("10.0.0.2")
    assert len(throttle._failures) == 1


def test_the_right_password_starts_the_count_again():
    throttle = Throttle(2, 300, clock=FakeClock())
    throttle.failed("10.0.0.2")
    throttle.passed("10.0.0.2")
    throttle.failed("10.0.0.2")
    assert throttle.may_try("10.0.0.2")


def test_clients_are_counted_apart():
    throttle = Throttle(1, 300, clock=FakeClock())
    throttle.failed("10.0.0.2")
    assert not throttle.may_try("10.0.0.2")
    assert throttle.may_try("10.0.0.3")


def test_take_checks_and_counts_in_one_step():
    throttle = Throttle(2, 300, clock=FakeClock())
    assert throttle.take("10.0.0.2") and throttle.take("10.0.0.2")
    assert not throttle.take("10.0.0.2")                   # used up: nothing more is counted
    assert throttle._failures["10.0.0.2"][0] == 2


def test_a_right_password_takes_its_try_back():
    throttle = Throttle(1, 300, clock=FakeClock())
    assert throttle.take("10.0.0.2")
    throttle.passed("10.0.0.2")
    assert throttle.may_try("10.0.0.2") and len(throttle._recent) == 0


def test_all_addresses_together_have_six_times_the_tries():
    clock = FakeClock()
    throttle = Throttle(2, 300, clock=clock)
    for number in range(12):                               # 6 * max_tries, from many addresses
        assert throttle.take(f"10.0.0.{number}")
    assert not throttle.take("10.0.1.1") and not throttle.may_try("10.0.1.1")   # a new address waits too
    clock.now += 299
    assert not throttle.take("10.0.1.1")
    clock.now += 1
    assert throttle.take("10.0.1.1")                       # the pause is over for everybody


def test_the_overall_count_keeps_the_wrong_tries_of_a_client_that_then_passed():
    throttle = Throttle(1, 300, clock=FakeClock())         # 6 for everybody
    for number in range(5):
        throttle.take(f"10.0.0.{number}")
    assert throttle.take("10.0.0.9")
    throttle.passed("10.0.0.9")                            # its own try is taken back, the others stay
    assert len(throttle._recent) == 5
    assert throttle.take("10.0.0.10") and not throttle.take("10.0.0.11")


def test_guesses_sent_at_the_same_moment_cannot_all_slip_through():
    throttle = Throttle(3, 300, clock=FakeClock())
    start = threading.Barrier(40)
    allowed = []

    def guess():
        start.wait()
        if throttle.take("10.0.0.2"):
            allowed.append(1)

    threads = [threading.Thread(target=guess) for _ in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert len(allowed) == 3


# ---- the certificate ---------------------------------------------------------------


class FakeRun:
    """Stands in for subprocess.run: writes the two files OpenSSL would, or fails."""

    def __init__(self, returncode=0, write=True):
        self.returncode = returncode
        self.write = write
        self.commands = []
        self.options = []

    def __call__(self, command, **options):
        self.commands.append(command)
        self.options.append(options)
        if self.write:
            Path(command[command.index("-keyout") + 1]).write_text("fake key", encoding="utf-8")
            Path(command[command.index("-out") + 1]).write_text("fake certificate", encoding="utf-8")
        return subprocess.CompletedProcess(command, self.returncode, stdout=b"SECRET OUTPUT",
                                           stderr=b"SECRET ERROR OUTPUT")


@pytest.fixture
def fake_openssl(monkeypatch, tmp_path):
    program = tmp_path / "bin" / "openssl.exe"
    monkeypatch.setattr(door_module, "find_openssl", lambda: program)
    return program


def test_the_certificate_is_made_with_these_arguments(tmp_path, fake_openssl):
    run = FakeRun()
    folder = tmp_path / "door"
    certificate, key = ensure_certificate(folder, ["192.168.43.105", "10.0.0.7"], run=run)
    assert (certificate, key) == (folder / "cert.pem", folder / "key.pem")
    (command,) = run.commands
    assert command == [str(fake_openssl), "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
                       "-subj", "/CN=Robot Car Vision",
                       "-addext", "subjectAltName=IP:192.168.43.105,IP:10.0.0.7,IP:127.0.0.1,DNS:localhost",
                       "-keyout", str(key), "-out", str(certificate)]
    assert run.options[0]["check"] is False and run.options[0]["capture_output"] is True
    assert (folder / "addresses.txt").read_text(encoding="utf-8") == "10.0.0.7\n192.168.43.105\n"


def test_the_certificate_is_made_once(tmp_path, fake_openssl):
    run = FakeRun()
    ensure_certificate(tmp_path, ["192.168.43.105"], run=run)
    ensure_certificate(tmp_path, ["192.168.43.105"], run=run)
    assert len(run.commands) == 1


def test_the_certificate_is_made_again_for_other_addresses(tmp_path, fake_openssl):
    run = FakeRun()
    ensure_certificate(tmp_path, ["192.168.43.105"], run=run)
    ensure_certificate(tmp_path, ["192.168.1.20"], run=run)
    assert len(run.commands) == 2 and "IP:192.168.1.20," in run.commands[1][run.commands[1].index("-addext") + 1]
    ensure_certificate(tmp_path, ["192.168.1.20"], run=run)
    assert len(run.commands) == 2


def test_the_certificate_is_made_again_when_old(tmp_path, fake_openssl):
    run = FakeRun()
    certificate, _key = ensure_certificate(tmp_path, ["192.168.43.105"], run=run)
    old = time.time() - 301 * 86400
    os.utime(certificate, (old, old))
    ensure_certificate(tmp_path, ["192.168.43.105"], run=run)
    assert len(run.commands) == 2


@pytest.mark.parametrize("missing", ["cert.pem", "key.pem", "addresses.txt"])
def test_the_certificate_is_made_again_when_a_file_is_missing(tmp_path, fake_openssl, missing):
    run = FakeRun()
    ensure_certificate(tmp_path, ["192.168.43.105"], run=run)
    (tmp_path / missing).unlink()
    ensure_certificate(tmp_path, ["192.168.43.105"], run=run)
    assert len(run.commands) == 2


def test_a_failing_run_names_the_exit_code_and_not_the_output(tmp_path, fake_openssl):
    with pytest.raises(DoorError) as caught:
        ensure_certificate(tmp_path, ["192.168.43.105"], run=FakeRun(returncode=3))
    message = str(caught.value)
    assert "exit code 3" in message and "SECRET" not in message
    assert not (tmp_path / "addresses.txt").exists()          # so the next start tries again


def test_a_run_that_writes_nothing_is_a_door_error(tmp_path, fake_openssl):
    with pytest.raises(DoorError):
        ensure_certificate(tmp_path, ["192.168.43.105"], run=FakeRun(write=False))


def test_a_certificate_folder_that_cannot_be_written_is_a_door_error(tmp_path, fake_openssl):
    def run_then_block(command, **options):
        result = FakeRun()(command, **options)
        (tmp_path / "addresses.txt").mkdir()               # the note beside the files cannot be written
        return result

    with pytest.raises(DoorError) as caught:
        ensure_certificate(tmp_path, ["192.168.43.105"], run=run_then_block)
    assert "cannot be written" in str(caught.value)
    with pytest.raises(DoorError):                         # and a folder that is not a folder
        ensure_certificate(tmp_path / "cert.pem", ["192.168.43.105"], run=FakeRun())


def test_a_run_that_cannot_start_is_a_door_error(tmp_path, fake_openssl):
    def broken(command, **options):
        raise FileNotFoundError(2, "not found", command[0])

    with pytest.raises(DoorError) as caught:
        ensure_certificate(tmp_path, ["192.168.43.105"], run=broken)
    assert "FileNotFoundError" in str(caught.value)


# ---- finding OpenSSL --------------------------------------------------------------


@pytest.fixture
def empty_world(monkeypatch, tmp_path):
    """A PATH and program folders with nothing in them."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "nothing"))
    return tmp_path


def make_program(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_openssl_on_the_path_is_found_without_git(empty_world, monkeypatch):
    program = make_program(empty_world / "tools" / "openssl.exe")
    monkeypatch.setenv("PATH", str(program.parent))
    assert find_openssl() == program.resolve()


def test_the_git_folders_openssl_comes_before_the_path(empty_world, monkeypatch):
    make_program(empty_world / "tools" / "openssl.exe")
    program = make_program(empty_world / "Programs" / "Git" / "usr" / "bin" / "openssl.exe")
    monkeypatch.setenv("PATH", str(empty_world / "tools"))
    monkeypatch.setenv("ProgramFiles", str(empty_world / "Programs"))
    assert find_openssl() == program.resolve()


def test_openssl_in_the_current_folder_or_a_relative_path_is_never_used(empty_world, monkeypatch):
    here = empty_world / "project"
    make_program(here / "openssl.exe")
    make_program(here / "tools" / "openssl.exe")
    make_program(here / "Git" / "cmd" / "git.exe")
    make_program(here / "Git" / "usr" / "bin" / "openssl.exe")
    monkeypatch.chdir(here)
    monkeypatch.setenv("PATH", os.pathsep.join([".", "tools", str(here / "tools"), str(here / "Git" / "cmd")]))
    with pytest.raises(DoorError):
        find_openssl()
    outside = make_program(empty_world / "outside" / "openssl.exe")
    monkeypatch.setenv("PATH", os.pathsep.join([".", str(here / "tools"), str(outside.parent)]))
    assert find_openssl() == outside.resolve()


def test_openssl_beside_the_git_on_the_path_is_found(empty_world, monkeypatch):
    make_program(empty_world / "Git" / "cmd" / "git.exe")
    program = make_program(empty_world / "Git" / "usr" / "bin" / "openssl.exe")
    monkeypatch.setenv("PATH", str(empty_world / "Git" / "cmd"))
    assert find_openssl().resolve() == program.resolve()


def test_openssl_in_the_usual_git_folder_is_found(empty_world, monkeypatch):
    program = make_program(empty_world / "Programs" / "Git" / "usr" / "bin" / "openssl.exe")
    monkeypatch.setenv("ProgramFiles", str(empty_world / "Programs"))
    assert find_openssl().resolve() == program.resolve()


def test_no_openssl_is_a_plain_door_error(empty_world):
    with pytest.raises(DoorError) as caught:
        find_openssl()
    assert str(caught.value) == NO_OPENSSL


# ---- this laptop's addresses --------------------------------------------------------


def test_only_private_ipv4_addresses_are_given(monkeypatch):
    found = ["192.168.43.105", "127.0.0.1", "169.254.10.3", "10.0.0.7", "172.20.1.4", "172.32.0.1",
             "100.64.0.9", "8.8.8.8", "192.168.43.105"]
    entries = [(2, 1, 6, "", (address, 0)) for address in found]
    monkeypatch.setattr(door_module.socket, "gethostname", lambda: "laptop")
    monkeypatch.setattr(door_module.socket, "getaddrinfo", lambda host, port, family: entries)
    assert lan_addresses() == ["192.168.43.105", "10.0.0.7", "172.20.1.4"]


def test_no_network_gives_no_addresses(monkeypatch):
    def broken(host, port, family):
        raise OSError("no network")

    monkeypatch.setattr(door_module.socket, "gethostname", lambda: "laptop")
    monkeypatch.setattr(door_module.socket, "getaddrinfo", broken)
    assert lan_addresses() == []


# ---- one real certificate ------------------------------------------------------------


def openssl_or_none():
    try:
        return find_openssl()
    except DoorError:
        return None


@pytest.mark.skipif(openssl_or_none() is None, reason="no OpenSSL on this computer")
def test_a_real_certificate_loads_in_python(tmp_path):
    certificate, key = ensure_certificate(tmp_path / "door", ["192.168.43.105"])
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    assert certificate.parent == tmp_path / "door"
