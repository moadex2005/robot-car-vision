"""The phone door: what a second, password-protected HTTPS door to the page needs.

The laptop opens the page at http://127.0.0.1:5000 with no password, as always. When
phone_door.enabled is true in config.yaml, app.py also serves the same page on
https://<this laptop's address>:5443, so a phone on the same network can open it
after typing the password (PAGE_PASSWORD in .env).

This module holds the parts that need no web server, so they can be tested alone:
the settings, the password, this laptop's network addresses, a certificate made with
OpenSSL, the signed-in phones (Sessions) and the count of wrong passwords (Throttle).
Everything that waits or runs a program takes its clock or its `run` from outside.

The password is never printed, logged, put in an error or shown by repr. The
certificate's key file is never read here, logged or served.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
import socket
import subprocess
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dotenv

from car_vision.config import PROJECT_ROOT, ConfigError, require, resolve_path

ENV_FILE = PROJECT_ROOT / ".env"
PASSWORD_VARIABLE = "PAGE_PASSWORD"
SHORTEST_PASSWORD = 8
LONGEST_PASSWORD = 200             # the sign-in page takes no more; a longer one could never match
NO_PASSWORD = (f"The phone door needs a password: put {PASSWORD_VARIABLE}= and at least "
               f"{SHORTEST_PASSWORD} characters in the .env file in the project folder, "
               "then start the program again.")
SHORT_PASSWORD = (f"{PASSWORD_VARIABLE} in .env is shorter than {SHORTEST_PASSWORD} characters. "
                  "Choose a longer one, then start the program again.")
LONG_PASSWORD = (f"{PASSWORD_VARIABLE} in .env is longer than {LONGEST_PASSWORD} characters. "
                 "Choose a shorter one, then start the program again.")
ODD_PASSWORD = (f"{PASSWORD_VARIABLE} has characters that cannot be typed. "
                "Choose another one, then start the program again.")
UNREADABLE_ENV = ("The .env file in the project folder could not be read ({kind}). "
                  "Save it as plain UTF-8 text, then start the program again.")

# The private IPv4 networks a home router or a phone's hotspot hands out.
PRIVATE_NETWORKS = (ipaddress.ip_network("10.0.0.0/8"),
                    ipaddress.ip_network("172.16.0.0/12"),
                    ipaddress.ip_network("192.168.0.0/16"))

CERTIFICATE_FILE = "cert.pem"
KEY_FILE = "key.pem"
ADDRESSES_FILE = "addresses.txt"   # the addresses the certificate was made for, one a line
CERTIFICATE_DAYS = 365             # what OpenSSL is asked for
RENEW_AFTER_DAYS = 300             # an older certificate is made again, long before it runs out
CERTIFICATE_NAME = "/CN=Robot Car Vision"
OPENSSL_SECONDS = 120              # making a 2048-bit key takes a second or two
NO_OPENSSL = ("OpenSSL was not found, so no certificate can be made. Install Git for Windows "
              "(it brings OpenSSL) or put openssl.exe on the PATH, then start the program again.")

MOST_SESSIONS = 50                 # signed-in phones kept at once; the oldest goes first
ALL_CLIENTS_TRIES = 6              # all addresses together get this many times max_tries in one pause


class DoorError(Exception):
    """A plain message for the start-up line. It never holds the password or a key."""


# ---- the settings ------------------------------------------------------------------


@dataclass(frozen=True)
class DoorSettings:
    """phone_door from config.yaml, checked, in seconds and with a full path."""

    port: int
    certificate_dir: Path
    session_seconds: float
    max_tries: int
    pause_seconds: float


def read_settings(config: dict[str, Any]) -> DoorSettings | None:
    """The phone door's settings, or None when phone_door.enabled is false (or the section is missing).

    A wrong value is a ConfigError that names the setting.
    """
    door = config.get("phone_door")
    if door is None:
        return None
    if not isinstance(door, dict):
        raise ConfigError("phone_door in config.yaml must be a section with enabled, port and the others")
    if not require(door, "enabled", bool):
        return None
    port = require(door, "port", int)
    if not 1 <= port <= 65535:
        raise ConfigError(f"phone_door.port must be from 1 to 65535, got {port}")
    max_tries = require(door, "max_tries", int)
    if not 1 <= max_tries <= 1000:
        raise ConfigError(f"phone_door.max_tries must be from 1 to 1000, got {max_tries}")
    return DoorSettings(
        port=port,
        certificate_dir=resolve_path(require(door, "certificate_dir", str)),
        session_seconds=positive(door, "session_hours", 24 * 365) * 3600,
        max_tries=max_tries,
        pause_seconds=positive(door, "pause_minutes", 24 * 60) * 60,
    )


def positive(door: dict[str, Any], key: str, most: float) -> float:
    """A number above 0 and at most `most` from the phone_door section."""
    value = require(door, key, (int, float))
    if not 0 < value <= most:
        raise ConfigError(f"phone_door.{key} must be above 0 and at most {most:g}, got {value}")
    return float(value)


def minutes_words(seconds: float) -> str:
    """"5 minutes", "1 minute" or "0.5 minutes": how the page writes a pause."""
    minutes = seconds / 60
    return "1 minute" if minutes == 1 else f"{minutes:g} minutes"


# ---- the password -------------------------------------------------------------------


def read_password(env_file: Path | None = None) -> str:
    """PAGE_PASSWORD from the environment or from .env. Never printed or logged.

    As with load_dotenv in lookout.py, a value already in the environment wins.
    The .env file is only read here, not copied into the environment, so programs
    this one starts (OpenSSL) never inherit the password.
    """
    password = os.environ.get(PASSWORD_VARIABLE)
    if password is None:
        password = password_in_file(ENV_FILE if env_file is None else env_file)
    if not password or not password.strip():
        raise DoorError(NO_PASSWORD)
    if len(password) < SHORTEST_PASSWORD:
        raise DoorError(SHORT_PASSWORD)
    if len(password) > LONGEST_PASSWORD:
        raise DoorError(LONG_PASSWORD)
    if not can_be_typed(password):
        raise DoorError(ODD_PASSWORD)
    return password


def password_in_file(path: Path) -> str | None:
    """PAGE_PASSWORD from this .env file, or None. A file that cannot be read is a plain DoorError."""
    try:
        if not path.is_file():
            return None
        return dotenv.dotenv_values(path, encoding="utf-8").get(PASSWORD_VARIABLE)
    except (OSError, ValueError) as error:            # no permission, not UTF-8, ...
        raise DoorError(UNREADABLE_ENV.format(kind=type(error).__name__)) from None


def can_be_typed(text: str) -> bool:
    """False for text with a lone surrogate, which has no UTF-8 form and no key to type it."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def password_matches(given: str, real: str) -> bool:
    """True when the typed password is the real one; it takes as long wherever they differ.

    Text that cannot be typed (a lone surrogate) never matches, and never raises.
    """
    if not (can_be_typed(given) and can_be_typed(real)):
        return False
    return hmac.compare_digest(given.encode("utf-8"), real.encode("utf-8"))


# ---- this laptop's addresses --------------------------------------------------------


def lan_addresses() -> list[str]:
    """This laptop's private IPv4 addresses, e.g. ["192.168.43.105"]; [] with no network.

    Asks for the addresses of this computer's own name, which gives one for every
    network adapter. Nothing is sent over the network.
    """
    try:
        found = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    addresses: list[str] = []
    for *_rest, socket_address in found:
        address = str(socket_address[0])
        if is_private(address) and address not in addresses:
            addresses.append(address)
    return addresses


def is_loopback(address: str | None) -> bool:
    """True for this computer's own addresses: localhost, 127.x.x.x and ::1 (also ::ffff:127.x.x.x)."""
    if not isinstance(address, str):
        return False
    if address.lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(address.strip("[]"))
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)           # an IPv4 address written as IPv6
    return ip.is_loopback or (mapped is not None and mapped.is_loopback)


def is_private(address: str) -> bool:
    """True for an IPv4 address in one of the PRIVATE_NETWORKS."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip.version == 4 and any(ip in network for network in PRIVATE_NETWORKS)


# ---- the certificate ----------------------------------------------------------------


def find_openssl() -> Path:
    """The OpenSSL program, as a full path: the one in a Git for Windows folder first (beside
    the git on the PATH, or in the usual folders), else one on the PATH.

    A program in the current folder, or found through a relative PATH entry, is never
    used: a file planted in the project folder must not be run.
    """
    beside_git = [folder / name for folder in git_folders()
                  for name in ("usr/bin/openssl.exe", "mingw64/bin/openssl.exe", "ucrt64/bin/openssl.exe")]
    for candidate in beside_git + on_path(program_name("openssl")):
        if candidate.is_file() and outside_current_folder(candidate):
            return candidate.resolve()
    raise DoorError(NO_OPENSSL)


def git_folders() -> list[Path]:
    """Where Git for Windows may be: above the git on the PATH, then the usual folders."""
    folders: list[Path] = []
    for git in on_path(program_name("git")):
        if outside_current_folder(git):
            folders += list(git.parents)[:3]        # ...\Git\cmd\git.exe or ...\Git\mingw64\bin\git.exe
    for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        if os.environ.get(variable):
            folders.append(Path(os.environ[variable]) / "Git")
    if os.environ.get("LOCALAPPDATA"):
        folders.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Git")
    return [folder for folder in folders if folder.is_absolute()]


def program_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def on_path(name: str) -> list[Path]:
    """Every file with this name in the absolute folders of the PATH, in PATH order."""
    found = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        folder = Path(entry.strip('"')) if entry.strip() else None
        if folder is not None and folder.is_absolute() and (folder / name).is_file():
            found.append(folder / name)
    return found


def outside_current_folder(path: Path) -> bool:
    """True for a full path that is not in the current folder or below it."""
    try:
        resolved = path.resolve()
        return path.is_absolute() and not resolved.is_relative_to(Path.cwd().resolve())
    except OSError:
        return False


def ensure_certificate(folder: Path, addresses: Sequence[str],
                       run: Callable[..., Any] = subprocess.run) -> tuple[Path, Path]:
    """(cert.pem, key.pem) in `folder`, made with OpenSSL when needed.

    They are made again when one is missing, when they are older than RENEW_AFTER_DAYS,
    or when they were made for other addresses (kept in addresses.txt beside them).
    A failed OpenSSL run is a DoorError that names its exit code, never its output.
    """
    certificate, key, made_for = folder / CERTIFICATE_FILE, folder / KEY_FILE, folder / ADDRESSES_FILE
    wanted = addresses_text(addresses)
    if still_good(certificate, key, made_for, wanted):
        return certificate, key
    try:
        folder.mkdir(parents=True, exist_ok=True)
        made_for.unlink(missing_ok=True)           # a failure below leaves no "made for" behind
    except (OSError, ValueError) as error:
        raise DoorError(f"The certificate folder {folder} cannot be written ({type(error).__name__}).") from None
    command = certificate_command(find_openssl(), certificate, key, addresses)
    try:
        result = run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=OPENSSL_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise DoorError(f"OpenSSL could not be run to make the certificate ({type(error).__name__}).") from None
    if result.returncode != 0 or not (certificate.is_file() and key.is_file()):
        raise DoorError(f"OpenSSL could not make the certificate (exit code {result.returncode}).")
    try:
        made_for.write_text(wanted, encoding="utf-8")
    except (OSError, ValueError) as error:
        raise DoorError(f"The certificate folder {folder} cannot be written ({type(error).__name__}).") from None
    return certificate, key


def addresses_text(addresses: Sequence[str]) -> str:
    """The addresses as addresses.txt keeps them: sorted, one a line."""
    return "".join(f"{address}\n" for address in sorted(set(addresses)))


def still_good(certificate: Path, key: Path, made_for: Path, wanted: str) -> bool:
    """True when both files are there, young enough and made for these addresses."""
    try:
        if not (certificate.is_file() and key.is_file()):
            return False
        age_days = (time.time() - certificate.stat().st_mtime) / 86400
        return age_days < RENEW_AFTER_DAYS and made_for.read_text(encoding="utf-8") == wanted
    except (OSError, ValueError):                  # e.g. addresses.txt is missing or not text
        return False


def certificate_command(openssl: Path, certificate: Path, key: Path, addresses: Sequence[str]) -> list[str]:
    """The OpenSSL command that makes a certificate for these addresses, 127.0.0.1 and localhost."""
    names = ",".join([f"IP:{address}" for address in addresses] + ["IP:127.0.0.1", "DNS:localhost"])
    return [str(openssl), "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-days", str(CERTIFICATE_DAYS), "-subj", CERTIFICATE_NAME,
            "-addext", f"subjectAltName={names}",
            "-keyout", str(key), "-out", str(certificate)]


# ---- signed-in phones and wrong passwords -------------------------------------------


class Sessions:
    """The signed-in phones: a random token each, valid for `seconds`. Safe to use from many threads."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds = seconds
        self._clock = clock
        self._opened: OrderedDict[str, float] = OrderedDict()   # token -> when it was opened
        self._lock = threading.Lock()

    def open(self) -> str:
        """A new random token; the oldest session goes when there are more than MOST_SESSIONS."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._forget_old()
            self._opened[token] = self._clock()
            while len(self._opened) > MOST_SESSIONS:
                self._opened.popitem(last=False)
        return token

    def valid(self, token: str | None) -> bool:
        """True for a token from open() that has not expired or been closed."""
        if not isinstance(token, str) or not token:
            return False
        with self._lock:
            self._forget_old()
            return token in self._opened

    def close(self, token: str | None) -> None:
        """End this session. Nothing happens for an unknown token."""
        if isinstance(token, str):
            with self._lock:
                self._opened.pop(token, None)

    def _forget_old(self) -> None:
        now = self._clock()
        for token, opened in list(self._opened.items()):
            if now - opened >= self.seconds:
                del self._opened[token]


class Throttle:
    """Wrong passwords: after `max_tries` from one client address, that client waits `pause_seconds`;
    after ALL_CLIENTS_TRIES * max_tries from all addresses together within `pause_seconds`,
    every client waits until the oldest of them is `pause_seconds` old.

    A client is forgotten once its last wrong password is `pause_seconds` old, and at
    once when it types the right one. Safe to use from many threads: take() checks and
    counts in one step, so guesses sent at the same moment cannot all slip through.
    """

    def __init__(self, max_tries: int, pause_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.max_tries = max_tries
        self.most_tries = ALL_CLIENTS_TRIES * max_tries   # from all addresses together
        self.pause_seconds = pause_seconds
        self._clock = clock
        self._failures: dict[str, tuple[int, float]] = {}       # who -> (wrong tries, when the last was)
        self._recent: deque[tuple[float, str]] = deque()        # (when, who) of every recent wrong try
        self._lock = threading.Lock()

    def take(self, who: str) -> bool:
        """One try for this client: False (and nothing counted) while it, or everybody, must wait.

        The try is counted as wrong at once; passed() takes it back when the password was right.
        """
        with self._lock:
            self._forget_old()
            if not self._may_try(who):
                return False
            self._count_failure(who)
            return True

    def may_try(self, who: str) -> bool:
        """False while this client, or everybody, has used up the tries and the pause is not over."""
        with self._lock:
            self._forget_old()
            return self._may_try(who)

    def failed(self, who: str) -> None:
        """Count one wrong password from this client (take() already does this)."""
        with self._lock:
            self._forget_old()
            self._count_failure(who)

    def passed(self, who: str) -> None:
        """The right password: this client starts again from no wrong tries, and the try that
        take() counted is taken back. Its earlier wrong tries still count for everybody."""
        with self._lock:
            self._failures.pop(who, None)
            for index in range(len(self._recent) - 1, -1, -1):
                if self._recent[index][1] == who:
                    del self._recent[index]
                    break

    def _may_try(self, who: str) -> bool:
        count, _last = self._failures.get(who, (0, 0.0))
        return count < self.max_tries and len(self._recent) < self.most_tries

    def _count_failure(self, who: str) -> None:
        now = self._clock()
        count, _last = self._failures.get(who, (0, 0.0))
        self._failures[who] = (count + 1, now)
        self._recent.append((now, who))

    def _forget_old(self) -> None:
        now = self._clock()
        for who, (_count, last) in list(self._failures.items()):
            if now - last >= self.pause_seconds:
                del self._failures[who]
        while self._recent and now - self._recent[0][0] >= self.pause_seconds:
            self._recent.popleft()


# ---- the door as the page sees it ----------------------------------------------------


class PhoneDoor:
    """What app.py needs to know about the phone door: the password, the sessions, the
    tries, and whether the door opened (its https addresses) or why it stayed shut.

    The password is kept in a private attribute and is never part of repr.
    """

    def __init__(self, settings: DoorSettings, clock: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings
        self.sessions = Sessions(settings.session_seconds, clock)
        self.throttle = Throttle(settings.max_tries, settings.pause_seconds, clock)
        self.addresses: tuple[str, ...] = ()        # https addresses, once the door is open
        self.why = "The phone door is still opening."
        self._password: str | None = None           # until it is set, no password matches

    def set_password(self, password: str) -> None:
        self._password = password

    def matches(self, given: str) -> bool:
        """True when `given` is the password (always False before set_password)."""
        return self._password is not None and password_matches(given, self._password)

    def opened(self, addresses: Sequence[str]) -> None:
        """The door's server runs: these are the addresses a phone opens."""
        self.addresses = tuple(addresses)

    def shut(self, why: str) -> None:
        """The door could not open; `why` is the start-up line."""
        self.addresses = ()
        self.why = why

    def __repr__(self) -> str:
        return f"PhoneDoor(port={self.settings.port}, open={bool(self.addresses)})"
