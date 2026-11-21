"""The phone's camera address: what a person may type, and the one that is remembered.

The demo page has a box where the visitor types the address the IP Webcam
app shows on the phone's screen, like 192.168.1.7:8080. phone_url() turns
that into the full stream address, http://192.168.1.7:8080/video, or refuses
it with a short message the page can show.

This is the only thing between a text field and a network connection made by
the server, so it is strict: only a plain http address, only an IPv4 address
on the local network (private or loopback), no user name or password, and no
path except /video.

The last address that was asked for is kept in data/camera.json, so the next
start uses the same phone. config.yaml is never written. A phone that needs a
login keeps the address from config.yaml (with its login) when the remembered
one is for the same host and port: see same_phone().

Standard library only.
"""

from __future__ import annotations

import ipaddress
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

log =logging.getLogger(__name__)

DEFAULT_PORT = 8080              # the port IP Webcam uses unless it was changed
STREAM_PATH = "/video"           # where IP Webcam serves its MJPEG stream
ALLOWED_PATHS = ("", "/", STREAM_PATH)

# The phone is always on the same local network as the laptop.
LOCAL_NETWORKS = (
    ipaddress.IPv4Network("10.0.0.0/8"),        # private
    ipaddress.IPv4Network("172.16.0.0/12"),     # private
    ipaddress.IPv4Network("192.168.0.0/16"),    # private (most home Wi-Fi)
    ipaddress.IPv4Network("127.0.0.0/8"),       # loopback: this laptop itself
)

EMPTY = "Type the phone's address, like 192.168.1.7:8080."
NO_HTTPS = "IP Webcam serves plain http, not https. Type the address without https://."
ONLY_HTTP = "Only http:// addresses work. Type the address as IP Webcam shows it, like 192.168.1.7:8080."
NO_LOGIN = "Leave out the user name and password: type only the address, like 192.168.1.7:8080."
BAD_PATH = "Leave out anything after the port except /video, like 192.168.1.7:8080/video."
NOT_IPV4 = "Type the phone's IP address as four numbers, like 192.168.1.7. Names and IPv6 addresses do not work."
BAD_PORT = "The port must be a number from 1 to 65535, like 8080."
NOT_LOCAL = ("That address is not on the local network. The phone's address starts with "
             "192.168., 10. or 172.16. to 172.31.")


def phone_url(text: str) -> str:
    """The full stream address for what a person typed, e.g. "http://192.168.1.7:8080/video".

    Raises ValueError with a short plain message when the text is refused.
    """
    address = text.strip() if isinstance(text, str) else ""
    if not address:
        raise ValueError(EMPTY)
    rest = without_scheme(address)
    host_and_port, path = split_path(rest)
    if "@" in host_and_port:
        raise ValueError(NO_LOGIN)
    if path not in ALLOWED_PATHS:
        raise ValueError(BAD_PATH)
    host, port = split_port(host_and_port)
    ip = ipv4_address(host)
    if not any(ip in network for network in LOCAL_NETWORKS):
        raise ValueError(NOT_LOCAL)
    return f"http://{ip}:{port}{STREAM_PATH}"


def without_scheme(address: str) -> str:
    """The address without "http://" in front; any other scheme is refused."""
    if "://" not in address:
        return address
    scheme, rest = address.split("://", 1)
    if scheme.lower() == "https":
        raise ValueError(NO_HTTPS)
    if scheme.lower() != "http":
        raise ValueError(ONLY_HTTP)
    return rest


def split_path(rest: str) -> tuple[str, str]:
    """("192.168.1.7:8080", "/video"): the part before the first /, ? or #, and the part from it."""
    cut = min((rest.find(mark) for mark in "/?#" if mark in rest), default=len(rest))
    return rest[:cut], rest[cut:]


def split_port(host_and_port: str) -> tuple[str, int]:
    """("192.168.1.7", 8080). IPv6 addresses have more than one colon and are refused."""
    if host_and_port.startswith("[") or host_and_port.count(":") > 1:
        raise ValueError(NOT_IPV4)
    if ":" not in host_and_port:
        return host_and_port, DEFAULT_PORT
    host, port_text = host_and_port.split(":")
    is_number = port_text.isascii() and port_text.isdigit() and len(port_text) <= 5
    if not is_number or not 1 <= int(port_text) <= 65535:
        raise ValueError(BAD_PORT)
    return host, int(port_text)


def ipv4_address(host: str) -> ipaddress.IPv4Address:
    """Four numbers from 0 to 255 with dots between them, nothing else."""
    try:
        return ipaddress.IPv4Address(host)
    except ValueError:
        raise ValueError(NOT_IPV4) from None


# ---- the remembered address --------------------------------------------------------


def load_remembered(path: Path) -> str | None:
    """The address kept by remember(), or None. A missing or broken file gives one log line."""
    try:
        return read_remembered(Path(path))
    except FileNotFoundError:
        log.info("No phone address remembered yet (%s); using stream.phone_url from config.yaml", path)
        return None
    except Exception as error:                  # anything at all: start-up must go on without it
        log.warning("Ignored %s: %s", path, first_line(error))
        return None


def read_remembered(path: Path) -> str:
    """The checked address in the file; raises when the file is missing, unreadable or wrong."""
    data = json.loads(path.read_text(encoding="utf-8"))
    url = data.get("phone_url") if isinstance(data, dict) else None
    if not isinstance(url, str):
        raise ValueError("it has no phone_url text")
    return phone_url(url)


def first_line(error: Exception) -> str:
    """The first line of an error message, or the error's type when it has no message."""
    lines = str(error).strip().splitlines()
    return lines[0][:200] if lines else type(error).__name__


def same_phone(first: str, second: str) -> bool:
    """True when two stream addresses point at the same host and port; a login in either is ignored."""
    one, other = host_and_port(first), host_and_port(second)
    return one is not None and one == other


def host_and_port(url: str) -> tuple[str, int] | None:
    """("192.168.1.7", 8080) for an http address, or None when it cannot be read."""
    text = url.strip() if isinstance(url, str) else ""
    try:
        parts = urlsplit(text if "://" in text else "http://" + text)
        port = parts.port or DEFAULT_PORT
    except ValueError:                          # e.g. a port that is not a number
        return None
    return (parts.hostname.lower(), port) if parts.hostname else None


def without_login(url: str) -> str:
    """The address with any user name and password taken out, so it can be shown on the page.

    Everything up to the last @ goes, so even a password with a / in it is never shown.
    """
    scheme, separator, rest = url.partition("://")
    if not separator:
        return url.rpartition("@")[2]
    return f"{scheme}://{rest.rpartition('@')[2]}"


def remember(path: Path, url: str) -> None:
    """Keep this address for the next start. Its folder is created when missing."""
    checked = phone_url(url)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"phone_url": checked}, indent=2) + "\n", encoding="utf-8")
