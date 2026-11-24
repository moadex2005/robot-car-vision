# M26 spec: Arabic for the voice and for listening

M24 made the assistant answer in Arabic when the `EN` / `ع` switch says so. This milestone makes the page speak those answers with an Arabic voice and understand a question spoken in Arabic. It is a change to the voice code of `car_vision/templates/index.html` only (and its tests).

Read `CLAUDE.md`, `specs/M23-hosted-voice.md` and `specs/M24-arabic-answers.md`, then all of the page's voice code: the cleaning for the ear (`earText`, `readableLines`, `cutAtSentence`, `speechPieces`), choosing a voice (`voiceRank`, `englishVoices`, `chosenVoice`, `drawVoiceChoices`), speaking (`say`, `speak`, `startTalking`, `speakPiece`, `utterance`, `hush`, the hosted path `speaksHosted` / `hostedClip`), and listening (`findRecognition`, `startListening`, `startRecording`, `sendRecording`, conversation mode, `isStopWords`).

## What is true today

- The hosted voice (Kokoro) speaks English only: the server refuses text with Arabic letters, and the page then falls back to the browser's voice for that piece, but with an English voice, which reads Arabic as noise or not at all.
- The browser's recogniser is always given `en-US`. Hosted listening already sends `language=<answerLanguage()>` and its model understands Arabic.
- `textDirection(text)` says `"rtl"` when at least a third of a text's letters are Arabic. `answerLanguage()` says `"en"` or `"ar"`.

## Speaking

- Each piece of what is spoken is judged by itself: a piece is **Arabic** when `textDirection(piece)` is `"rtl"`. So an Arabic answer is spoken in Arabic, and the page's own English sentences (`I can see the keys, on the left.`) stay English, whatever the switch says.
- An Arabic piece is spoken by the browser's own speech with an Arabic voice: the best one the browser lists whose language starts with `ar` (rank them as `voiceRank` ranks English voices: "Natural" first, then "Online" or "Google", then any other; within each group `ar-EG`, then `ar-SA`, then the rest). Its `lang` is the voice's own. It is never sent to the hosted voice (no request that will be refused), and never spoken with a voice that is not Arabic.
- With no Arabic voice in this browser, an Arabic piece is not spoken at all; the answer is still shown. The talk goes on to the next piece and ends as usual, so everything that waits for the end of speaking (the speaker button's waves, conversation mode) still gets it.
- An English piece is spoken exactly as today (the hosted voice when its switch is on, else the chosen browser voice).
- Cleaning for the ear must know Arabic punctuation: `؟` and `۔` end a sentence, `،` and `؛` are places to cut a long piece, and no ASCII full stop is added after a line that ends in `؟`, `،`, `؛` or `۔`. Arabic-Indic digits and Arabic text are left as they are. English text is cleaned and cut exactly as before: every existing test of these functions passes untouched.
- The Voice box, under the list of voices, gets one quiet line that is shown only while the switch says Arabic: `Arabic answers are spoken by: <the voice's short name>` or `This browser has no Arabic voice, so Arabic answers are shown but not spoken. Microsoft Edge has Arabic voices.` It is redrawn when the browser's list of voices arrives late (the existing waiting logic) and when the switch changes.
- The Test button speaks `مرحباً. هذا هو الصوت الذي ستستخدمه الصفحة.` while the switch says Arabic, and today's sentence otherwise.
- Nothing new is ever spoken by itself: only what `say()` is called for today.

## Listening

- The browser's recogniser gets `ar-EG` while the switch says Arabic and `en-US` otherwise. Hosted listening is unchanged.
- What was heard goes where it goes today. An Arabic sentence is never a find or watch request (those are recognised in English only), so it goes to the assistant, which answers in Arabic.
- Conversation mode also stops on these Arabic words, alone in a sentence: `توقف`, `قف`, `خلاص`, `كفاية`, `شكراً`, `شكرا`, `مع السلامة`. Ignore Arabic vowel marks and a final `.`, `!`, `؟` when comparing.
- Pressing the `EN` / `ع` switch while the page is speaking or listening stops both at once (`hush()` and the existing cancel), so no piece is spoken and no question is heard in the wrong language.
- The line under the microphone that says where the sound goes (`drawVoicePrivacy`) is unchanged.

## Where to work

Another coder is changing the finder, the banners over the video and the Timeline in this file at the same time. Touch only the voice sections named above, the Voice box's markup and CSS, and the one line where the language switch is pressed (to stop speaking and listening). Do not change `textDirection`, `answerLanguage`, `linkParts` or anything in the assistant or the server.

## Checking the JavaScript for real

Before you report, run the page's real script in headless Edge on a scratch page outside the project, with a fake `speechSynthesis` whose list of voices you control, a fake `fetch` and a fake recogniser, and report the results of at least these cases: an Arabic answer with an Arabic voice present (which voice and `lang` each piece got; no hosted request made); the same with no Arabic voice (nothing spoken, the talk still ends, the note in the Voice box); an English answer while the switch says Arabic; an answer that mixes an English and an Arabic paragraph; the voices arriving one second late; the hosted voice switched on with an Arabic answer (no request) and with an English answer (requests as before); the recogniser's `lang` for both switch positions; each Arabic stop word ending a conversation; the switch pressed in the middle of speaking; the cleaning functions on Arabic text with `؟`, `،` and a bullet list, and on the English samples the existing tests use (unchanged output). Do not use the Claude browser pane.

## Tests: `tests/test_app_arabic_voice.py` (new)

String checks on the template in the style of the existing voice tests: an Arabic piece never reaches the hosted request; the Arabic voice ranking; the recogniser's language; the stop words; the Test sentence; the note; the switch stopping speech and listening; no ASCII full stop added after Arabic punctuation. Existing tests are edited only where a quoted line had to change; list each in your report.

## Exit test

The whole test suite passes and every case of the browser check passes. The orchestrator checks live in Microsoft Edge that an Arabic answer is spoken with an Arabic voice.
