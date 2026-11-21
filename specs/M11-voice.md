# M11 spec: voice on the page

The project owner asked for voice. Two things, both in the browser, with nothing new on the server and no new library: the page **speaks** (answers, greetings, the close warning) and the visitor can **ask by speaking**.

Read `CLAUDE.md`, then `specs/M7-click-to-inspect.md` (page rules, "Look and feel"), `specs/M8-theme-camera-delay.md` (B4, B5) and `specs/M10-full-view-size-brush.md` (A1, A3), then `car_vision/templates/index.html` and the template tests in `tests/test_app.py`.

Files: `car_vision/templates/index.html` and `tests/test_app.py` only.

## What the browser offers (use feature detection, never assume)

- Speaking: `window.speechSynthesis` and `SpeechSynthesisUtterance`. Works offline with the voices installed on the computer; Edge and Chrome add better online voices. It may only start after the visitor has done something on the page (a click), so the first speech must follow the click that switched voice on.
- Listening: `window.SpeechRecognition` or `window.webkitSpeechRecognition`. Present in Edge and Chrome, missing in Firefox and probably in the Claude desktop app's browser pane. **In Edge and Chrome the sound of the question is sent to the browser maker's speech service to be turned into text, and it needs the internet.** The page must say so (see the privacy line).
- Where one of the two is missing, its controls are hidden and everything else works as before.

## Speaking

- A **Voice** button in the navigation bar (a small speaker drawn inline, with a label for screen readers) switches speaking on and off. Off by default; the choice is remembered in `localStorage` (wrapped in `try`/`catch`). Remembered-on only takes effect after the visitor's first click or key press on the page, because browsers refuse speech before that.
- One small module in the script owns speaking: `say(text, kind)` and `hush()`. Rules:
  - A new thing to say replaces what is being said (cancel, then speak), except that a warning is never cut off by anything but another warning or `hush()`.
  - Text is cleaned for the ear before it is spoken: no URLs, no `From the web` mark, no Markdown signs; at most about 600 characters, cut at a sentence end.
  - The voice: the best English voice the browser lists (prefer names containing `Natural` or `Online`, then any `en-` voice, then the default). Voices arrive late in some browsers (`voiceschanged`).
  - `Escape` hushes before it closes anything (first press hushes, the next behaves as today). Switching voice off hushes.
- What is spoken while voice is on:
  - A chat answer.
  - An object's description when it arrives in the details panel: its name, then its summary. Not the specifications list, not the sources.
  - A follow-up answer in the panel.
  - A recognised person's panel: their name and role.
  - A greeting when it appears (`Hello, Sara Ahmed!`), once per greeting.
  - The close warning when it appears: `Careful. A person is close in front of the car.` Once each time the warning comes on, not while it stays on.
  - Errors are not spoken.
- While something is being spoken, the Voice button shows it (a quiet pulsing mark made of `transform`/`opacity` on that small button only, stopped the moment speech ends; this is not an endless animation and must not run when nothing is spoken).

## Asking by voice

- A microphone button beside the main chat field and beside the panel's `Ask about this` field (drawn inline, labelled). Press it, speak, and the words appear in that field as they are heard. When the browser says the sentence is finished, the question is sent as if Send had been pressed. Pressing the button again while it listens cancels without sending.
- In the full view, where the chat is not visible, a microphone button sits on the stage (near the full-view button). A question asked there goes to the open details panel's thing when a panel with a follow-up chat is open, otherwise to the main chat. The answer is shown as a **caption**: a solid capsule over the bottom of the video (above the pills) with the question in small text and the answer under it, for 12 seconds or until the next one; no blur over the video.
- While listening: the button shows it, the field's placeholder says `Listening ...`, and any speaking is hushed (so the page does not hear itself). Recognition language `en-US`.
- Plain messages, in the field's usual error place: the microphone was not allowed; nothing was heard; no internet for speech; the browser stopped listening.
- Asking by voice works whether or not speaking is switched on.

## Privacy line

Add, near the main chat and in the footer: `Asking by voice sends the sound of your question to your browser's speech service (Microsoft or Google) to turn it into text. Speaking the answers stays on this computer unless you picked an online voice.` Shown only where listening exists.

## Rules that still hold

- One HTML file, inline script and style, no outside resources, `textContent` only, links only for http/https, every `focus()` with `preventScroll`, no `scrollIntoView`, no endless animation (the existing test lists the two allowed loops; the speaking mark runs only during speech and must be written so that test still passes or is extended honestly), no blurred surface over the video.
- Keyboard: every new button reachable and operable; in the full view the page behind stays inert.
- Nothing on the server changes. No audio is ever sent to the project's own server.

## Tests

- Template tests in `tests/test_app.py`: the Voice button, the three microphone buttons, the caption element and the privacy sentence are present; `speechSynthesis` and both recognition names are feature-detected (the page must not throw where they are missing); the existing template rules still pass.
- In a browser, with a fake server (see the checking notes in your brief): with `window.speechSynthesis` and the recognition class replaced by small recording fakes before the page script runs, check that each event in "What is spoken" calls speak once with the cleaned text, that a new answer cancels the old speech, that the warning is not cut off, that Escape hushes first, that a fake recognition result fills the field and sends once, and that the page works with both APIs deleted.
