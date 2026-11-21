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
  - A new thing to say replaces what is being said (cancel, then speak).
  - One answer is spoken in one voice: the voice is settled before the first piece is spoken (waiting a moment for a voice list that has not arrived yet), and the pieces follow one another, never overlapping.
  - Text is cleaned for the ear before it is spoken: no URLs, no `From the web` mark, no Markdown signs; at most about 600 characters, cut at a sentence end.
  - The voice: the best English voice the browser lists (prefer names containing `Natural` or `Online`, then any `en-` voice, then the default). Voices arrive late in some browsers (`voiceschanged`).
  - `Escape` hushes before it closes anything (first press hushes, the next behaves as today). Switching voice off hushes.
- What is spoken while voice is on. **Only things that answer something the visitor just did on this page** (the owner's decision after hearing the first build: "i need the voice just in the chat when i clicked the photo or when i text with it"):
  - A chat answer, typed or asked by microphone.
  - The scene description.
  - An object's description when the visitor clicked it: its name, then its summary. Not the specifications list, not the sources.
  - A follow-up answer in the panel.
  - A recognised person's name and role, when the visitor opened that person's panel by clicking.
  - **Not spoken:** greetings and the close warning (their banners stay, silent), anything that arrives through the status poll, and errors. The first build spoke greetings and the warning; the owner did not want the warning at all, and with the page open in two places the greeting was spoken twice at once in two different voices.
- While something is being spoken, the Voice button shows it (a quiet pulsing mark made of `transform`/`opacity` on that small button only, stopped the moment speech ends; this is not an endless animation and must not run when nothing is spoken).

## Asking by voice

- A microphone button beside the main chat field and beside the panel's `Ask about this` field (drawn inline, labelled). Press it, speak, and the words appear in that field as they are heard. When the browser says the sentence is finished, the question is sent as if Send had been pressed. Pressing the button again while it listens cancels without sending.
- In the full view, where the chat is not visible, a microphone button sits on the stage (near the full-view button). A question asked there goes to the open details panel's thing when a panel with a follow-up chat is open, otherwise to the main chat. The answer is shown as a **caption**: a solid capsule over the bottom of the video (above the pills) with the question in small text and the answer under it, for 12 seconds or until the next one; no blur over the video.
- While listening: the button shows it, the field's placeholder says `Listening ...`, and any speaking is hushed (so the page does not hear itself). Recognition language `en-US`.
- Plain messages, in the field's usual error place: the microphone was not allowed; nothing was heard; no internet for speech; the browser stopped listening.
- Asking by voice works whether or not speaking is switched on.

## Conversation mode (added after the first build)

The owner asked for voice "like in the Claude app". The Claude API has no speech (checked: its reference has no audio input or output, and the app's voice mode is not offered to developers), so the listening and the speaking stay with the browser. What can be matched is the back-and-forth: one press, then a spoken conversation without pressing a button for every question.

- In the Voice box, a switch **Conversation: keep listening after each answer** (off by default, remembered like the other voice choices; shown only where listening exists).
- With it on, a press on any microphone starts a conversation: listen, send the question, show and **speak** the answer (the conversation speaks even when `Speak the answers` is off; with no speech synthesis in the browser the answer is shown and the page waits 1.5 seconds), then listen again by itself. The question goes where that microphone's questions go today (main chat, the open panel's thing, or from the stage: the panel's thing when one is open, else the main chat).
- It ends when: the same microphone or Escape is pressed; the visitor says only `stop`, `goodbye`, `bye`, `that's all` or `thank you` (any letter case, with or without a full stop; these words are not sent as a question, and the page answers `Okay.` in the caption or chat and stops); nothing is heard two turns in a row; any recognition error; the details panel it was talking about closes; the page leaves or enters the full view while the stage microphone drives it.
- A press on the microphone while the page is speaking stops the speech and listens at once (the visitor interrupts).
- The page never listens while it speaks. After speech ends it waits the same short pause as today before listening.
- While a conversation runs, the microphone that drives it shows which of the three states it is in (listening, thinking, speaking) by its label for screen readers and a visible mark; the marks follow the page's animation rule (they run only during the conversation and the template test must still pass or be extended honestly).
- Tests with the recording fakes, as for the rest of voice: three turns in a row with one press; each way of ending; an interruption; silence twice; a greeting and a warning arriving mid-conversation are not spoken; the switch off behaves exactly as before.

## Privacy line

Add, near the main chat and in the footer: `Asking by voice sends the sound of your question to your browser's speech service (Microsoft or Google) to turn it into text. Speaking the answers stays on this computer unless you picked an online voice.` Shown only where listening exists.

## Rules that still hold

- One HTML file, inline script and style, no outside resources, `textContent` only, links only for http/https, every `focus()` with `preventScroll`, no `scrollIntoView`, no endless animation (the existing test lists the two allowed loops; the speaking mark runs only during speech and must be written so that test still passes or is extended honestly), no blurred surface over the video.
- Keyboard: every new button reachable and operable; in the full view the page behind stays inert.
- Nothing on the server changes. No audio is ever sent to the project's own server.

## Tests

- Template tests in `tests/test_app.py`: the Voice button, the three microphone buttons, the caption element and the privacy sentence are present; `speechSynthesis` and both recognition names are feature-detected (the page must not throw where they are missing); the existing template rules still pass.
- In a browser, with a fake server (see the checking notes in your brief): with `window.speechSynthesis` and the recognition class replaced by small recording fakes before the page script runs, check that each event in "What is spoken" calls speak once with the cleaned text, that a new answer cancels the old speech, that the warning is not cut off, that Escape hushes first, that a fake recognition result fills the field and sends once, and that the page works with both APIs deleted.
