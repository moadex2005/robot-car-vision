# M24 spec: Arabic answers

The owner wants a switch that makes the assistant answer in Arabic. This milestone does the answers: the switch, the assistant's language, and showing Arabic text properly. Speaking and listening in Arabic is the next milestone (M26), built after the hosted voice (M23) that is being written in parallel; do not touch the voice code here.

Read `CLAUDE.md`, then `car_vision/assistant.py` (`SYSTEM_PROMPT`, `DESCRIBE_PROMPT`, `FOLLOW_UP_PROMPT`, `SCENE_PROMPT`, `parse_object_info`, the `Assistant` class) and the routes `ask`, `describe_object`, `follow_up` and `scene` in `car_vision/app.py`.

## The switch (`templates/index.html`)

- A small round button in the top bar, between the Camera button and the voice button: it shows `EN` while answers are in English and `ع` while they are in Arabic. `aria-pressed` is true for Arabic; its `title` and `aria-label` say what a press does (`Answer in Arabic` / `Answer in English`).
- The choice is remembered in this browser (`readStored` / `store`, as the theme is) and is English until chosen.
- `answerLanguage()` already exists near the top of the script and returns `"en"`. Make it return the choice (`"en"` or `"ar"`). The hosted voice being built in parallel calls it too.
- The page's own words (buttons, headings, hints, the sentences the page writes itself such as `I can see the keys, on the left.`) stay English. Only the assistant's answers change language.
- Other work is going on in this file in parallel (the camera box, the voice code and Voice box, the finder, new CSS at the very end of the style sheet). Touch only: the top bar's markup for the new button, its CSS directly after the `.theme-button` rules, a section of its own `// ---- the language of the answers ----` directly after the theme section, the bodies of the four requests named below, and the few places where an answer's text is put on the page.

## Sending the language (`templates/index.html`, `app.py`)

- The four requests that ask the assistant for words gain `"language": answerLanguage()` in their JSON body: `/api/ask`, `/api/describe`, `/api/follow_up`, `/api/scene`.
- The four routes read it: missing means `"en"`; anything but `"en"` or `"ar"` is a 400 with a plain message. They pass it on as `language=`.
- The cache of looked-up objects (`AnswerCache`, `cache_key`) keeps the two languages apart: a thing looked up in English is looked up again when it is clicked in Arabic.
- A click on a person still sends nothing and shows what was written about them, as written.

## The assistant (`assistant.py`)

```python
LANGUAGES = ("en", "ar")
ARABIC_STYLES = ("standard", "egyptian")

class Assistant:
    def ask(self, question: str, language: str = "en") -> Answer: ...
    def describe(self, image: bytes, label: str | None = None, language: str = "en") -> ObjectInfo: ...
    def follow_up(self, image, label, info, earlier, question, marked=None, language: str = "en") -> Answer: ...
    def describe_scene(self, image: bytes, language: str = "en") -> Answer: ...
```

- New setting, already in `config.yaml`: `assistant.arabic_style: standard` (`standard` or `egyptian`, else `ConfigError`). `standard` is Modern Standard Arabic in simple, clear sentences; `egyptian` is everyday Egyptian Arabic.
- With `language="ar"` one short instruction is added to what the model is told for that call (a function `language_instruction(language, style) -> str`, empty for English): answer in Arabic in that style; keep people's names, product names, model numbers, units and web addresses as they are normally written; do not add a translation or a transliteration. English calls send exactly what they send today, byte for byte: the existing tests of the prompts must pass untouched.
- `describe` in Arabic: the answer's three headings stay in English exactly as the prompt asks for them today (`parse_object_info` reads them), and the values (the name line, the description, each specification's value) are in Arabic. Specification names may be Arabic. Check `parse_object_info` with Arabic values, Arabic-Indic digits and the Arabic comma and colon, and fix it only where it loses or mangles them.
- The conversation of the main chat may hold both languages; the instruction applies to the answer being asked for now.
- `UnavailableAssistant` in `app.py` takes the same new arguments.
- Anything but `"en"` or `"ar"` is a `ValueError` with a plain message.
- The assistant's own fixed sentences (`BUSY_MESSAGE` and the like) and every error message stay English.

## Showing Arabic (`templates/index.html`)

- Arabic reads from right to left. Every element that shows an answer's text gets `dir="auto"`, so each paragraph takes its direction from its own first strong letter and an English answer looks exactly as today: the chat's answer paragraphs, the follow-up answers, the object's name and summary, each specification's name and value, the caption over the video.
- The visitor's own typed questions get `dir="auto"` too, and so do the chat field and the follow-up field.
- Lists, bullets and the bold, italic and code rendering of M18 must sit on the correct side in an Arabic paragraph (check `• ` bullets and a bold label followed by a colon). A web link's visible word stays as it is.
- Text is still put in only through `textContent` and `make()`.
- The font stack must show Arabic letters well on Windows and on phones: add `"Segoe UI"`, `"Noto Naskh Arabic"`/`"Noto Sans Arabic"` and `Tahoma` where the stack would otherwise fall to a serif for Arabic. No outside font file.

## Tests (new files; edit existing ones only where an assertion must change)

- `tests/test_assistant_language.py`, with the fake client the assistant tests use: the instruction is in what is sent for each of the four calls with `"ar"` and absent with `"en"`; both styles; a wrong language and a wrong style; English requests are byte for byte what they were (compare with a request built at the start of the test without the argument); `parse_object_info` with an Arabic answer (the name, the description, three specifications, Arabic comma, Arabic-Indic digits).
- `tests/test_app_language.py`: each route passes the language on (a fake assistant records it), a missing language is English, a wrong one is 400, the cache keeps the languages apart, a person click is unchanged, and string checks on the template: the button, `answerLanguage()` reading the stored choice, the four request bodies, `dir="auto"` in each place listed above.
- Fake assistants in existing test files get the new argument only where a test would otherwise fail; list each such edit in your report.

## Checking the page for real

Before you report, look at Arabic and English answers side by side in a real browser engine, without starting the program: a scratch page outside the project that holds the template's style sheet and the functions that put an answer on the page (`make`, `messageText`, `addMessage`, `answerNodes` and what they need, `showInfo`), fed with an Arabic answer that has a bold label, a bullet list, a number, an English product name and a web link, and with an English answer; take pictures with headless Edge (`--headless=new --screenshot=... --window-size=900,900`, with `--user-data-dir` set to a folder under your temp folder) in the light and the dark theme, read them, and say in your report what you saw. Do not use the Claude browser pane and do not commit scratch files.

## Exit test

The whole test suite passes. The orchestrator checks live, with the real assistant, an Arabic answer in the chat, an Arabic look-up of a clicked object with its specifications, and that switching back gives English.
