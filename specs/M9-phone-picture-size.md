# M9 spec: ask the phone for a smaller picture

## Why (measured on the owner's phone, 2026-11-21)

The page never shows more than 640 x 360 (`stream.width` x `stream.height` is a 640 x 480 box and the phone's picture is wide), but the phone was sending 1920 x 1080. Timing how long a change on the phone takes to become a decoded picture on the laptop (median of 8, phone's encoder + Wi-Fi + reader + decoder; not the camera sensor, the models or the browser):

| Phone picture | Old reader (FFmpeg) | New reader (`MjpegCapture`) | Bytes per picture |
| --- | --- | --- | --- |
| 1920 x 1080 | 122 ms, spikes to 280 | 103 ms, spikes to 285 | about 120 000 |
| 640 x 360 | 66 ms | 59 to 79 ms | about 14 000 |

So the picture size is the big lever: it halves this part of the delay and removes the spikes. The IP Webcam app lets the laptop change it, checked on the real phone:

- `GET http://<phone>:<port>/status.json?show_avail=1` answers JSON. `curvals.video_size` is the current size as text, e.g. `"1920x1080"`. `avail.video_size` is a list of the sizes the phone offers, e.g. `["960x540", "800x600", "3840x2160", ..., "640x480", "640x360", "352x288", ...]`, in no useful order.
- `GET http://<phone>:<port>/settings/video_size?set=640x360` answers a small XML text containing `Ok`. The running `/video` stream changes size after a few pictures without being closed (about a quarter of a second without pictures); `MjpegCapture` and `FrameSource` carried on without reconnecting.
- Other camera apps have neither address. Then nothing is changed and nothing is shown.

Read `CLAUDE.md` first, then `specs/M8-theme-camera-delay.md` (Part B), then `car_vision/phone.py`, `pipeline.py`, `app.py`, `templates/index.html` and their tests.

## What to build

### `car_vision/phone.py` (standard library only)

```python
Size = tuple[int, int]

def parse_size(text: str) -> Size | None: ...                 # "640x360" -> (640, 360); anything else -> None
def best_size(offered: Sequence[Size], current: Size, max_width: int, max_height: int) -> Size | None: ...
def read_picture_sizes(stream_url: str, fetch=None) -> tuple[Size | None, list[Size]]: ...   # (current, offered)
def ask_for_size(stream_url: str, size: Size, fetch=None) -> bool: ...
```

- `best_size`: among the offered sizes with the same shape as the current one (width / height equal within 2 %), the smallest that is at least as large as what the page will show (`fitted_size` of the current size into the box; do not import `stream.py`, the arithmetic is three lines and may be repeated here with a comment). `None` when the current size is already that one, when nothing fits, or when the list is empty. Never a size larger than the current one.
- `read_picture_sizes` and `ask_for_size` talk only to the host and port of the given stream address, with a 3 second timeout, and never raise: any failure (no such address, another app, bad JSON, a timeout) is one debug log line and `(None, [])` or `False`. `fetch(url, timeout) -> str` can be replaced in tests. A login in the stream address is passed on as Basic auth and never logged.
- `ask_for_size` refuses (returns `False`, sends nothing) for a size that is not positive or larger than 8000 on a side.

### Setting

`config.yaml` gains `stream.phone_picture: auto` (the orchestrator adds the line). Values: `auto` (ask the phone for the best size whenever a phone is connected), `keep` (never change the phone), or a size like `1280x720` (ask for exactly that if the phone offers it). Anything else is a `ConfigError` naming the three forms.

### `car_vision/pipeline.py`

- Whenever the phone source is started (at start-up with `source: phone`, and after `use_phone`), a short background thread reads the sizes and, depending on the setting, asks for one. It must not delay the video or the web server, must not run for the webcam or a clip, and must give up quietly. One at a time; a newer switch makes an older thread's result irrelevant.
- `Pipeline.set_phone_picture(choice: str) -> None` (`"auto"`, `"keep"` or a size): applies for this run (config.yaml is not written) and acts at once in the background. A size the phone does not offer raises `ValueError` with a plain message.
- `status()` gains `"phone_picture"`: `{"current": "640x360", "offered": ["640x360", "960x540", ...], "choice": "auto"}`, with `offered` sorted from small to large by area, or `None` when the phone does not answer such requests or the source is not the phone. `current` is updated after a change (read back from the phone, not assumed).

### `car_vision/app.py`

- `POST /api/camera/picture` with `{"choice": "auto" | "keep" | "<W>x<H>"}`: 400 with a plain message for anything else or for a size the phone does not offer; otherwise `{"ok": true}` at once.
- `VIDEO_MAX_FPS` goes from 20 to 30: the phone sends 30 pictures a second at the small size, and a cap of 20 made a finished picture wait up to 50 ms. Update the tests that name the old value.

### The page

In the Camera box, under the address, a row **Picture from the phone**: a `<select>` with `Automatic (smallest that is sharp enough)`, `Leave the phone as it is`, and the offered sizes (`640 × 360`, ...), showing the current choice; beside or under it the size the phone sends right now. One line of help: `Smaller pictures arrive sooner. The page shows 640 × 360 either way.` (use the real fitted size if it is known, else leave the second sentence out). The row is hidden when `status.phone_picture` is `null`. Changing the select posts at once; an error shows in the result line.

### Tests (fakes only: no phone, no sockets)

- `parse_size`, `best_size` (the real list above with 1920x1080 current and a 640x480 box gives 640x360; a 4:3 phone at 1920x1440 gives 640x480; already smallest gives `None`; never larger; empty list), `read_picture_sizes` on the real JSON shape and on junk, `ask_for_size` on `Ok`, on another answer, on an exception, on a silly size; the login never in a log line.
- Pipeline with a fake fetch: `auto` asks once per connect and reports the size read back; `keep` asks nothing; a fixed size not offered is ignored at start-up with one warning; webcam and clip never fetch; `set_phone_picture` each form; `status()["phone_picture"]`.
- The route's good and bad cases; the template has the row; the existing template rules still pass.

Do not open the phone stream or any address on the local network: the orchestrator checks the result on the real phone.
