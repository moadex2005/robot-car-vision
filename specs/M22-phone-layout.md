# M22 spec: the page on a phone

The page was drawn for a laptop. Opened on a phone (through the phone door of M21, built in parallel) it must be just as calm and usable: one column, large touch targets, nothing cut off, no sideways scrolling. This milestone is CSS and a few lines of markup in `car_vision/templates/index.html`. No Python changes, no new behaviour.

Read `CLAUDE.md` and the "Look and feel" part of `specs/M7-click-to-inspect.md`: a phone maker's product page, light, spacious, liquid glass, small motion; never a dark dashboard, no endless animation, no blur over the video.

## What "phone" means

A viewport up to 640 CSS pixels wide, held upright (about 360 to 430 wide) or on its side (about 640 to 930 wide and under 450 high). Everything above 640 wide and 450 high keeps today's layout pixel for pixel.

## What the page must do at phone width

1. **No sideways scrolling**, at 320, 360, 390 and 430 wide: nothing is wider than the screen, long words and addresses wrap.
2. **Top bar**: one row that fits. The section links (`Live view`, `Ask`, ...) are hidden; the brand shortens or hides; the status pill shows its dot and state word only (source, device and the word `fps` hidden; the number may stay); the Camera, voice and theme buttons stay, each at least 44 by 44 pixels to touch.
3. **Hero**: smaller title, the two buttons side by side or stacked, no empty screen-high gap before the video.
4. **Live view**: the video spans the full width with a small margin; the tools at its top right (camera, Describe, microphone, full view) stay touchable (at least 40 pixels, 8 apart) and do not cover the middle of the picture; the Describe button shows its icon only; the finder (the Look for field, its chips and Stop) sits under the video instead of over its bottom edge, so it never covers the picture, and its field is as wide as the screen allows.
5. **In view**: the pills wrap or scroll sideways inside their own row (never the page); the Auto-discover switch and its small text stack under the heading.
6. **Details panel**: no longer a column beside the video. It comes under the video and the pills as a full-width sheet with the same glass look, without the width grip. Closing it and `Esc` work as today. The follow-up form fits: field on one row, its buttons on the next when they do not fit.
7. **Chat**, **People**, **Timeline**: full width, forms one field per row, buttons at least 44 pixels high, the Timeline's pictures and text in one column.
8. **The boxes** opened from the top bar (Camera, Voice): full width with a margin, inside the screen, scrollable when taller than the screen.
9. **Typing**: every input, select and textarea has a font size of at least 16 pixels at phone width, so iPhone Safari does not zoom the page when one is tapped.
10. **Full view** on a phone: the picture fills the screen (upright: full width, centred; on its side: full height), the pills in one scrolling row at the bottom, the details as a sheet over the lower part of the screen, never wider than the screen. The safe areas of a phone with a notch are respected (`env(safe-area-inset-*)`), and the page's `viewport` meta has `viewport-fit=cover` (add the meta if it is missing; `width=device-width, initial-scale=1`).
11. **On its side** (wide and low): the top bar does not take a third of the height (smaller padding), and the video is not taller than the screen.
12. **Touch**: no hover-only control. Whatever appears on hover today (the hand over the video does not count) must also be reachable by a tap. `:hover` looks may stay.
13. Both themes, light and dark.

## How

- All new rules go into one block at the very end of the style sheet, directly before `</style>`, under the comment `/* ---- phone: up to 640 px wide ---- */`, inside `@media (max-width: 640px)` and, for the low sideways case, `@media (max-height: 450px) and (orientation: landscape)`. Do not edit the existing rules above it; override them in the block. Other coders are adding rules higher up in the same sheet at this moment, and nobody else writes at its end.
- Where the layout cannot be reached by CSS alone, change the markup as little as possible (a wrapper, a class) and say each change in your report. Do not move or rename elements with an `id`: the JavaScript and the tests find them by `id`.
- JavaScript: only if a rule above truly needs it (for example the panel's width code must not set a width on a phone). Say each change in your report. The functions `fitDetails`, `fitPanelWidth`, `fitVideo`, `fitFinderRoom` and `enterFullView` measure the layout; read them before you decide.
- New controls added in parallel by others (a part in the Camera box with an address and a small QR picture, a row in the Voice box, a small language button in the top bar, chips that say `maybe`) will arrive after you. Write the phone rules so that an extra button in the top bar and extra rows in the two boxes still fit: wrap, do not assume a fixed count.

## Checking it for real

The test suite cannot see a layout. Before you report, look at the page yourself in a real browser engine at phone size, without starting the program: make a scratch copy of the template in your temporary folder with the Jinja parts filled in by hand (`{{ ... }}` values and `{% if %}` blocks), stop its script from fetching (`window.fetch = () => new Promise(() => {})` before the page's script, and point the video at nothing), then fill the lists by calling the page's own functions with sample data (`drawObjects`, `drawChips`, `addMessage`, `showInfo`, `drawTimeline`, ...) in a small script at the end. Take pictures with headless Edge:

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu --hide-scrollbars --user-data-dir="<a folder under your temp folder>" --window-size=390,844 --screenshot="<out>.png" "file:///<path>/phone.html"
```

at 320 x 640, 390 x 844, 430 x 932 and 844 x 390, in the light and the dark theme, with the details panel open and closed, and once at 1280 x 800 to show the laptop layout did not change. Read each picture. In the scratch page also compute `document.documentElement.scrollWidth <= window.innerWidth` and the size of every button and input, and write the results into the page so `--dump-dom` shows them. Put the table of these checks in your report. Do not use the Claude browser pane (it belongs to the orchestrator) and do not commit scratch files.

## Tests: `tests/test_app_phone.py` (new)

String checks on the template: the viewport meta with `viewport-fit=cover`; the phone block is the last thing in the style sheet and holds both media queries; inside it: the 16-pixel rule for inputs, the hidden section links, the details panel no longer in a side column, `env(safe-area-inset-bottom)`; no `id` that existed before is gone (compare with a list you take from the template at the base commit); no rule above the block was changed (the text of the style sheet before the block equals the base's, apart from what the other milestones add: check instead that every selector line you did not write is still there).

## Exit test

The whole test suite passes, and the pictures show every numbered point. The orchestrator checks it live at phone size in the browser pane.
