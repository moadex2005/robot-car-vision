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

The picture size: the page never shows more than about 640 x 360, but the
phone may send 1920 x 1080, which takes about twice as long to arrive.
read_picture_sizes() asks IP Webcam which sizes it offers, best_size() picks
the smallest one that is still sharp enough, and ask_for_size() asks the
phone to send it. These talk only to the host and port of the stream address,
wait at most 3 seconds, and never raise: another camera app simply gives
"no answer".

Standard library only.
"""

from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import SplitResult, unquote, urlsplit

log = logging.getLogger(__name__)

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


# ---- the picture size the phone sends ------------------------------------------------
#
# IP Webcam answers two addresses beside its stream (checked on the real phone):
#   /status.json?show_avail=1          JSON: curvals.video_size is the current size ("1920x1080"),
#                                      avail.video_size the sizes it offers, in no useful order
#   /settings/video_size?set=640x360   asks for another size; answers a little XML text with "Ok"

Size = tuple[int, int]                      # (width, height) in pixels
Fetch = Callable[[str, float], str]         # (address, timeout in seconds) -> the answer as text

ASK_TIMEOUT_SECONDS = 3.0        # a phone that does not answer within this is given up on
MAX_SIDE = 8000                  # no camera sends pictures wider or higher than this
SAME_SHAPE = 0.02                # two sizes have the same shape when width / height differ by at most 2 %
MAX_ANSWER_BYTES = 64 * 1024     # the status answer is a few kilobytes; anything more is not read
MAX_SIZES = 100                  # at most this many offered sizes are kept from one answer
STATUS_PATH = "/status.json?show_avail=1"
SET_SIZE_PATH = "/settings/video_size?set="
SIZE_PATTERN = re.compile(r"([1-9][0-9]{0,4})x([1-9][0-9]{0,4})")
OK_ANSWER = re.compile(r"\bOk\b")


def parse_size(text: str) -> Size | None:
    """(640, 360) for the text "640x360"; None for anything else."""
    if not isinstance(text, str):
        return None
    match = SIZE_PATTERN.fullmatch(text)
    return (int(match[1]), int(match[2])) if match else None


def size_text(size: Size) -> str:
    """The size the way IP Webcam writes it, e.g. "640x360"."""
    return f"{size[0]}x{size[1]}"


def is_usable_size(size: object) -> bool:
    """True for a (width, height) pair of whole numbers from 1 to 8000."""
    return (isinstance(size, tuple) and len(size) == 2
            and all(isinstance(side, int) and not isinstance(side, bool) and 1 <= side <= MAX_SIDE
                    for side in size))


def shown_size(size: Size, max_width: int, max_height: int) -> Size:
    """The size the page shows a picture of this size at.

    The same arithmetic as stream.fitted_size (the largest size with the same
    shape inside the box, never enlarged, both sides even), repeated here so
    this module needs nothing but the standard library.
    """
    width, height = size
    scale = min(max_width / width, max_height / height, 1.0)
    return max(2, int(round(width * scale / 2)) * 2), max(2, int(round(height * scale / 2)) * 2)


def same_shape(one: Size, other: Size) -> bool:
    """True when the two sizes have the same width / height, within 2 %."""
    return abs((one[0] / one[1]) / (other[0] / other[1]) - 1) <= SAME_SHAPE


def area(size: Size) -> int:
    return size[0] * size[1]


def best_size(offered: Sequence[Size], current: Size, max_width: int, max_height: int) -> Size | None:
    """The smallest offered size that is still sharp enough for the page, or None.

    "Sharp enough" means the same shape as the current picture and at least as
    large as what the page will show. Never a size larger than the current one.
    None when the current size is already that one, or when nothing fits.
    """
    if not is_usable_size(current):
        return None
    shown = shown_size(current, max_width, max_height)
    sharp_enough = [size for size in offered
                    if is_usable_size(size) and same_shape(size, current)
                    and shown[0] <= size[0] <= current[0] and shown[1] <= size[1] <= current[1]]
    if not sharp_enough:
        return None
    smallest = min(sharp_enough, key=area)
    return None if smallest == current else smallest


def read_picture_sizes(stream_url: str, fetch: Fetch | None = None) -> tuple[Size | None, list[Size]]:
    """(the size the phone sends now, the sizes it offers); (None, []) when it does not say.

    Never raises: no phone, another camera app, a timeout or a junk answer is
    one debug log line.
    """
    address = phone_address(stream_url, STATUS_PATH)
    if address is None:
        log.debug("Not asking %s for picture sizes: not a plain http address on the local network",
                  without_login(str(stream_url)))
        return None, []
    try:
        return sizes_in_status((fetch or fetch_text)(address, ASK_TIMEOUT_SECONDS))
    except Exception as error:                 # anything at all: the video must go on without it
        log.debug("The phone at %s gave no picture sizes (%s)", without_login(stream_url), type(error).__name__)
        return None, []


def sizes_in_status(answer: str) -> tuple[Size, list[Size]]:
    """The current and the offered sizes in IP Webcam's status answer; ValueError when it has none."""
    if not isinstance(answer, str):
        raise ValueError("the answer is not text")
    data = json.loads(answer)
    current_values = data.get("curvals") if isinstance(data, dict) else None
    available = data.get("avail") if isinstance(data, dict) else None
    current = parse_size(current_values.get("video_size")) if isinstance(current_values, dict) else None
    if current is None:
        raise ValueError("the answer has no current video_size")
    listed = available.get("video_size") if isinstance(available, dict) else None
    offered: list[Size] = []
    for item in listed if isinstance(listed, list) else []:
        size = parse_size(item)
        if size is not None and size not in offered:
            offered.append(size)
    return current, offered[:MAX_SIZES]


def ask_for_size(stream_url: str, size: Size, fetch: Fetch | None = None) -> bool:
    """Ask the phone to send pictures of this size. True when it answered "Ok".

    A size that is not positive or is larger than 8000 on a side is refused
    without sending anything. Never raises.
    """
    if not is_usable_size(size):
        return False
    address = phone_address(stream_url, SET_SIZE_PATH + size_text(size))
    if address is None:
        return False
    try:
        answer = (fetch or fetch_text)(address, ASK_TIMEOUT_SECONDS)
    except Exception as error:                 # anything at all: the video must go on without it
        log.debug("The phone at %s did not take the size %s (%s)",
                  without_login(stream_url), size_text(size), type(error).__name__)
        return False
    if isinstance(answer, str) and OK_ANSWER.search(answer):
        return True
    log.debug("The phone at %s did not answer Ok to the size %s", without_login(stream_url), size_text(size))
    return False


def phone_address(stream_url: str, path: str) -> str | None:
    """`path` on exactly the host and port of the stream address, keeping its login.

    None unless the stream address is plain http to an IPv4 address on the
    local network, so these requests can never go anywhere else.
    """
    try:
        parts = urlsplit(stream_url.strip())
        port = parts.port or DEFAULT_PORT
        ip = ipaddress.IPv4Address(parts.hostname or "")
    except (AttributeError, TypeError, ValueError):     # not text, a bad port, not an IPv4 address
        return None
    if parts.scheme.lower() != "http" or not any(ip in network for network in LOCAL_NETWORKS):
        return None
    login = parts.netloc.rpartition("@")[0]
    return f"http://{login + '@' if login else ''}{ip}:{port}{path}"


def fetch_text(url: str, timeout: float) -> str:
    """GET an http address and return the answer as text. Raises on any failure.

    A login in the address is sent as Basic auth, the way the stream reader sends it.
    """
    parts = urlsplit(url)
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    try:
        connection.request("GET", path, headers=login_header(parts))
        response = connection.getresponse()
        if response.status != 200:
            raise OSError(f"the phone answered with status {response.status}")
        return response.read(MAX_ANSWER_BYTES).decode("utf-8", errors="replace")
    finally:
        connection.close()


def login_header(parts: SplitResult) -> dict[str, str]:
    """The Basic auth header for a login in the address; no header without one."""
    if parts.username is None:
        return {}
    login = f"{unquote(parts.username)}:{unquote(parts.password or '')}"
    return {"Authorization": "Basic " + base64.b64encode(login.encode("utf-8")).decode("ascii")}
