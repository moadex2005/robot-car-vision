# M16 spec: QR codes and barcodes

The camera reads QR codes and barcodes that are in view, on this laptop, with OpenCV alone: no new model, no internet, nothing sent anywhere. The video marks each code, and the page shows what it says when the visitor asks.

Read `CLAUDE.md` first. The installed OpenCV is 4.14.0 and has both `cv2.QRCodeDetector` and `cv2.barcode.BarcodeDetector`; check their exact call results with the project's Python before you write against them.

## What a code's text is

Text from a code is written by a stranger. It is shown to the visitor and nothing else:

- never logged (counts and errors only), never spoken, never stored in the memory, never put into the `WorldState`, never given to the assistant;
- on the page only through `textContent`;
- a link only when it is a plain `http:` or `https:` address, and it is never opened by itself.

## Settings (already in `config.yaml`, added by the orchestrator)

```yaml
codes:
  enabled: true             # read QR codes and barcodes in the picture
  every_n_frames: 5         # codes are looked for on every 5th frame
  max_width: 960            # a wider picture is shrunk to this width before it is searched
  hold_seconds: 1.5         # a code stays on screen this long after it was last read
```

`tests/test_pipeline.py` `CONFIG` already has this section with `enabled: False`.

## Part A: `car_vision/codes.py` (new; `cv2` and the standard library only)

```python
MAX_TEXT_CHARS = 500

@dataclass(frozen=True)
class Code:
    kind: str                                  # "qr" or "barcode"
    text: str                                  # what the code says, cleaned
    box: Box                                   # (x1, y1, x2, y2) in pixels of the frame it was read from
    corners: tuple[tuple[int, int], ...]       # its four corners in the same pixels
    format: str = ""                           # for a barcode, e.g. "EAN_13"; "" for a QR code

class CodeReader:
    def __init__(self, settings: dict, qr: Any = None, barcodes: Any = None) -> None: ...
    def read(self, frame: np.ndarray) -> list[Code]: ...

def web_link(text: str) -> str | None: ...
def clean_text(text: Any) -> str: ...
```

- `CodeReader` checks its settings like the other modules do (`require`; `every_n_frames` 1 or more, `max_width` 160 or more, `hold_seconds` 0 or more, else `ConfigError`). `qr` and `barcodes` are the two OpenCV detectors; tests hand in fakes, otherwise they are built in `__init__` (no model files, no super-resolution).
- `read` shrinks a frame wider than `max_width` (keeping its shape), searches it for QR codes and for barcodes, and returns the decoded ones with boxes and corners scaled back to the frame's own pixels and kept inside the frame. A code that was found but not decoded (empty text) is left out. The same code reported twice (same kind and text, boxes overlapping) counts once.
- `clean_text`: a string with control characters removed (line breaks and tabs become one space), trimmed, cut to `MAX_TEXT_CHARS`; anything that is not a string gives `""`.
- `web_link(text)` returns the text itself when the whole text is one web address: scheme exactly `http` or `https`, a host, no user name or password, no spaces. Otherwise `None`. `javascript:`, `data:`, `file:`, `tel:`, WiFi and contact codes are not links.
- An exception from OpenCV never leaves `read`: it logs one warning the first time for that kind (not once per frame) and that kind gives no codes on that frame.

## Part B: `car_vision/draw.py`

```python
CODE_COLOR = (255, 170, 0)   # (blue, green, red): one fixed colour for every code

def draw_codes(frame: np.ndarray, codes: Iterable[Code]) -> None: ...
```

A closed line through the four corners (the box when there are not four) and the caption `QR code` or `barcode` from the box's top-left corner, with `draw_caption`. The code's text is never drawn. Put the new code in a section of its own at the end of the file.

## Part C: `car_vision/pipeline.py`

- `Pipeline.__init__` gains a last parameter `code_reader=None`. It reads the `codes` section; with `enabled: true` and no reader handed in it builds a `CodeReader`. With `enabled: false` no reader is built and nothing below happens.
- In `_process`, on every `codes.every_n_frames`-th frame of this camera (and on the first), the reader reads the clean frame. Codes are kept by `(kind, text)` and stay on screen until `hold_seconds` have passed since they were last read (with their newest box). The kept codes are drawn on the picture with `draw.draw_codes` after the detections and faces, and stored under the lock together with the frame's other results. A camera switch, a new frame size and a stream that went stale forget them, exactly as the finds are forgotten.
- `status()` gains `"codes"`: a list, empty when nothing is fresh, of

  ```python
  {"kind": "qr", "format": "", "text": "https://example.com/menu", "link": "https://example.com/menu",
   "box": [0.212, 0.401, 0.387, 0.632]}        # fractions of the frame, like the objects
  ```

  `link` is `web_link(text)` (the address, or `None`). At most 8 codes, largest box first. Add the key directly after the `"objects"` line of the returned dictionary (another change adds a key elsewhere in it).
- A reader that raises must never stop the video (the reader itself already swallows OpenCV errors; the loop treats any other error from it as "no codes" and logs it once).
- Put new fields of `__init__` directly after the memory settings (after `keep_pictures = ...`) and after `self._memory_failed = False`; put new methods in a section of their own, `# ---- codes (the loop thread) ----`, directly before `# ---- the memory (the loop thread)`.

## Part D: `car_vision/templates/index.html` (no change to `app.py`: `/api/status` already passes the status through)

- Under the In view pills (inside `.inview`, after `<ul id="things">` and its "Nothing detected" line), a row that is hidden when there are no codes: one pill per code, saying `QR code` or `Barcode` and a short preview (for a link: the site's name, as `siteName()` gives it; otherwise the first 28 characters).
- A click on a pill opens a small card under the row: the kind (`QR code`, or `Barcode · EAN-13`), the whole text (it wraps; long text scrolls inside the card), and two buttons: **Open link**, only when the status gave a `link` and the page's own `webAddress()` also accepts it (an `<a>` that opens in a new tab with `rel="noopener noreferrer"` and shows the full address in its `title`), and **Copy** (the clipboard; it says `Copied` or `Could not copy` for two seconds). A close button and `Esc` close the card. The card keeps its text when the code leaves the view, until it is closed or another pill is clicked, so the visitor can still copy it.
- A click on the video that lands inside a code's box (from the newest status) opens that code's card and does not send `/api/inspect`. Change the video's click handler for this and nothing else in it.
- The row and the card must work in the full view (`.stage.is-full`), where the pills lie over the video.
- The privacy paragraph (`#privacy`) gains one sentence: `QR codes and barcodes are read on this laptop; their text is sent nowhere.`
- Page rules, all of them: text only through `textContent`, nodes through `make()`, no outside resources, no endless animation, no blur over the video, redraw only what changed (`changed()`), cleared when the program is not running (`setOnline(false)`).
- Where to put it, so other changes going on in parallel merge cleanly: the CSS directly after the `.thing` rules (before the full-view rules); the HTML as said above; the JavaScript as a section of its own, `// ---- codes: QR codes and barcodes in view ----`, directly before `// ---- the chat ----`; in `poll()` call `drawCodes(status);` on a new line directly after `drawObjects(status);`.

## Tests (new files)

- `tests/test_codes.py`: a real round trip without a camera: make a QR picture with `cv2.QRCodeEncoder`, paste it into a larger grey frame, and `CodeReader.read` returns its text, kind and a box around where it was pasted, also from a frame wider than `max_width`. With fakes: a barcode with its format; an undecoded find left out; duplicates; a detector that raises (no exception, one log line for many frames); `clean_text` and `web_link` on awkward input (control characters, 2000 characters, `javascript:alert(1)`, `https://user:pw@host/`, `http://example.com/a b`, `WIFI:S:home;;`, a bare word); settings checks; the text never appears in the log. `draw_codes`: the line colour on a corner-to-corner edge, the caption words, nothing raised for a code at the frame's edge.
- `tests/test_pipeline_codes.py`, with a fake reader and the fakes of `tests/test_pipeline.py`: the reader is called on every 5th frame; a code stays for `hold_seconds` and then goes; `status()["codes"]` has the shape above and is empty after a camera switch; `enabled: False` builds no reader and gives `"codes": []`; a reader that raises does not stop the loop; the text is not in the log, the `WorldState` or the memory.
- `tests/test_app_codes.py`: the page has the row, the card, `drawCodes(status)` after `drawObjects(status)`, the privacy sentence, and uses `textContent`/`make()` for the text (string checks on the template, as `tests/test_app_discover.py` does).
- `tests/test_pipeline.py` has a set of the expected status keys (`STATUS_KEYS`): add `"codes"` to it.

## Exit test

The whole test suite passes. The orchestrator checks it live: a QR code held to the webcam gets a marked outline in the video and a pill on the page whose card shows its text.
