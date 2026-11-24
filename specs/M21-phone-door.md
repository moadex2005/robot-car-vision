# M21 spec: the phone door

Today only the laptop itself can open the page (`app.host: 127.0.0.1`). The owner wants to open it from a phone on the same network, behind a password. This milestone adds a second, protected door for that. The laptop's own door does not change at all: same address, no password.

The layout for a small screen is a separate milestone (M22). Read `CLAUDE.md`, then `create_app`, `refusal`, `only_this_page` and `main` in `car_vision/app.py`.

## Why a second door, and why HTTPS

- A password sent over plain HTTP on a hotspot can be read by anyone on it, and so can the video with people's faces. A phone's browser also gives a page its microphone only over HTTPS. So the phone door speaks HTTPS, with a certificate made on this laptop. The phone does not know that certificate and shows a warning once; the page and the README say so.
- The laptop keeps its plain `http://127.0.0.1:5000`, so nothing about the demo on the laptop changes and no test of today breaks.

## Settings (already in `config.yaml`, added by the orchestrator)

```yaml
phone_door:
  enabled: false            # true: the page can also be opened from a phone on the same network, behind a password
  port: 5443                # the phone opens https://<this laptop's address>:5443
  certificate_dir: data/door    # a certificate is made here on first use
  session_hours: 12         # a phone stays signed in this long
  max_tries: 5              # this many wrong passwords from one phone, then ...
  pause_minutes: 5          # ... it must wait this long
```

The password is `PAGE_PASSWORD` in `.env`, read the way `lookout.py` reads `HF_TOKEN`. The rules for keys hold for it: never printed, logged, committed, put in a test or sent back in any answer. `data/door/` is git-ignored already.

## Part A: `car_vision/door.py` (new; standard library only, no `flask`)

```python
class DoorError(Exception): ...               # a plain message for the start-up line

@dataclass(frozen=True)
class DoorSettings:
    port: int
    certificate_dir: Path
    session_seconds: float
    max_tries: int
    pause_seconds: float

def read_settings(config: dict) -> DoorSettings | None: ...     # None when phone_door.enabled is false
def read_password(env_file: Path | None = None) -> str: ...     # DoorError when missing or shorter than 8 characters
def lan_addresses() -> list[str]: ...                           # this laptop's private IPv4 addresses, e.g. ["192.168.43.105"]
def find_openssl() -> Path: ...                                 # DoorError with a plain message when there is none
def ensure_certificate(folder: Path, addresses: Sequence[str], run=subprocess.run) -> tuple[Path, Path]: ...

class Sessions:
    def __init__(self, seconds: float, clock=time.monotonic) -> None: ...
    def open(self) -> str: ...                 # a new random token (secrets.token_urlsafe)
    def valid(self, token: str | None) -> bool: ...
    def close(self, token: str | None) -> None: ...

class Throttle:
    def __init__(self, max_tries: int, pause_seconds: float, clock=time.monotonic) -> None: ...
    def may_try(self, who: str) -> bool: ...
    def failed(self, who: str) -> None: ...
    def passed(self, who: str) -> None: ...

def password_matches(given: str, real: str) -> bool: ...        # hmac.compare_digest on the UTF-8 bytes
```

- `ensure_certificate` returns `(cert.pem, key.pem)` in `folder`, making them when they are missing, older than 300 days, or made for other addresses (keep the addresses in a small text file beside them). It runs OpenSSL: `openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=Robot Car Vision" -addext "subjectAltName=IP:<each address>,IP:127.0.0.1,DNS:localhost"`. This was tried on this laptop with the OpenSSL that ships with Git for Windows (`C:\Program Files\Git\usr\bin\openssl.exe`, OpenSSL 3.5) and Python loaded the result. `find_openssl` looks on the `PATH`, then beside the Git it finds on the `PATH` or in the usual Git folders. No new Python package: `cryptography` is not installed and must not be added.
- The key file is private: it is never logged or served. A failed OpenSSL run is a `DoorError` that names the exit code, never the command's full output.
- `Sessions` keeps at most 50 sessions (the oldest goes first) and forgets expired ones. `Throttle` counts per client address and forgets a client after its pause.
- Everything takes its clock and its `run` from outside, so the tests need no OpenSSL, network or waiting.

## Part B: `car_vision/app.py`

- `create_app(...)` gains a last parameter `door: Any = None` (a small object holding the password, a `Sessions` and a `Throttle`; `None` means there is no phone door, as in every test of today).
- A request **came through the phone door** exactly when the server that took it says so: the phone server wraps the app in a few lines of WSGI that set `environ["car_vision.door"] = "phone"`. Nothing the browser sends (not the `Host` header, not a cookie) decides this.
- For a request through the phone door, before anything else: with a valid session cookie it goes on as usual. Without one: `GET /` and any other page address redirect to `/door`; anything under `/api/` or `/video` answers 401 with `{"error": "Sign in first."}`. `/door` and its own style need no session.
- `GET /door`: a small sign-in page (`templates/door.html`: one password field, one button, the page's look in a light and a dark theme, no outside resource, text through `textContent`). `POST /door` with JSON `{"password": "..."}`: right password: a session cookie named `car_vision_door` with `Secure`, `HttpOnly`, `SameSite=Strict`, `Path=/` and the session's lifetime, answer `{"ok": true}`. Wrong: 403 `{"error": "That is not the password."}`. Too many tries from that address: 429 `{"error": "Too many tries. Wait 5 minutes."}` (the number from the settings), also when the password would have been right. `POST /door/leave` ends the session. The same-origin rule covers these three like every other changing request (they must be JSON from this page).
- Through the laptop's door `/door` answers 404, and nothing else changes: no cookie, no password, every existing test passes untouched.
- `GET /api/door`: through the laptop's door `{"open": true, "addresses": ["https://192.168.43.105:5443/"]}` or `{"open": false, "why": "<the start-up line>"}`; through the phone door 404. `GET /api/door/qr?address=<one of those addresses>`: a PNG of a QR code for that address, made with `cv2.QRCodeEncoder` (cv2 is allowed in `app.py` as elsewhere outside the library rule); any other address is a 400. Laptop door only.
- `main()`: after the pipeline is built, when the door is enabled: read the password, find the addresses, make the certificate, start a second server for the same app on `0.0.0.0:<port>` with TLS in a thread of its own (`werkzeug.serving.make_server(..., threaded=True, ssl_context=...)`), and print one line: `Phone door: https://192.168.43.105:5443/  (password: PAGE_PASSWORD in .env)`. Any `DoorError` (no password, no OpenSSL, no network, the port taken) prints one plain line that says what to do, and the program runs on without the door. The door closes with the program. Nothing about the door runs when it is not enabled.
- The password is compared only through `password_matches`. It never appears in a log line, an error, a response or a `repr`.

## Part C: `templates/index.html`, the Camera box only

Under the Delay test switch, a part that is shown only when `GET /api/door` says the door is open (asked once when the box is first opened): the line `Open on your phone` and, for each address, the address as text and its QR picture (`/api/door/qr?...`, about 140 pixels), with the note `Same network. The phone warns once about the certificate; continue, then type the password.` When the door is shut, one quiet line with the `why`. Through the phone door this part is not shown, and a **Sign out** button is shown there instead. Other work is going on in this file in parallel: touch only the Camera box's markup, its CSS (`.camera-box` rules) and its JavaScript section.

## Tests (new files)

- `tests/test_door.py`: settings (off gives `None`, wrong values are `ConfigError`); the password (missing, short, present; never in an exception's text); `Sessions` (a token is long and random, expires with a fake clock, closes, the 50 limit); `Throttle` (tries, the pause, the reset after a pass, clients kept apart); `password_matches` (equal, not equal, non-ASCII); `ensure_certificate` with a fake `run` (the command's arguments, made once, made again when the addresses change or the files are old, a failing run is a `DoorError` without the output); `find_openssl` with a fake `PATH`; `lan_addresses` returns only private addresses.
- `tests/test_app_door.py`, with Flask's test client and `environ_overrides={"car_vision.door": "phone"}` for the phone door: every case of Part B, including that a `Host` header, a made-up cookie or a made-up `car_vision.door` request header never opens the door; the cookie's flags; the three status codes; the laptop door unchanged (one request per existing route kind, no cookie set); `/api/door` and `/api/door/qr` (a PNG that `cv2.QRCodeDetectorAruco` reads back to the address); the password never in any response body or log line of these tests; the template's new part by string checks.
- One real test of `ensure_certificate` marked `@pytest.mark.skipif` when no OpenSSL is found: it makes a certificate in `tmp_path` and `ssl.SSLContext.load_cert_chain` loads it.

## Exit test

The whole test suite passes. You do not start the site. The orchestrator checks live that the laptop's page is unchanged, that `https://127.0.0.1:5443/` asks for the password and then shows the page with video, and that the Camera box shows the address and its QR code.
