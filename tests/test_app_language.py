"""Tests for the Arabic switch on the page and the language the routes pass on (M24).

A fake assistant records the language of every call. No camera, model,
network, graphics card or API key is needed. The page is checked as text.
"""

import json
import re
import shutil
import subprocess
import urllib.parse
from pathlib import Path

import pytest
from car_vision.app import UnavailableAssistant, create_app
from car_vision.assistant import Answer, AssistantError, ObjectInfo
from test_app import LAPTOP, PHOTO, SARA_TARGET, FakePipeline, between, ids_around

ARABIC_INFO = ObjectInfo(name="حاسوب محمول", summary="حاسوب خفيف.", specs=(("الوزن", "١٫٢ كجم"),))
ENGLISH_INFO = ObjectInfo(name="Laptop", summary="A light computer.", specs=(("Weight", "1.2 kg"),))
STROKES = [{"points": [[0.4, 0.4], [0.45, 0.45], [0.5, 0.5]]}]


class RecordingAssistant:
    """Answers in the language it was asked for, and records (call, language) of every call."""

    def __init__(self):
        self.calls = []
        self.questions = []

    def ask(self, question, language="en"):
        self.calls.append(("ask", language))
        self.questions.append(question)
        return Answer("مرحبا" if language == "ar" else "Hello")

    def describe(self, image, label=None, language="en"):
        self.calls.append(("describe", language))
        return ARABIC_INFO if language == "ar" else ENGLISH_INFO

    def follow_up(self, image, label, info, earlier, question, marked=None, language="en"):
        self.calls.append(("follow_up", language))
        return Answer("جواب" if language == "ar" else "An answer")

    def describe_scene(self, image, language="en"):
        self.calls.append(("scene", language))
        return Answer("مكتب عليه حاسوب." if language == "ar" else "A desk with a laptop.")

    def reset(self):
        pass


class ScenePipeline(FakePipeline):
    def scene_photo(self):
        return PHOTO


def client_with(helper, target=LAPTOP):
    app = create_app(ScenePipeline(target=target), helper)
    app.testing = True
    return app.test_client()


def inspect(client, language=None):
    body = {"x": 0.5, "y": 0.5} if language is None else {"x": 0.5, "y": 0.5, "language": language}
    return client.post("/api/inspect", json=body).get_json()


def with_language(body, language):
    return body if language is None else dict(body, language=language)


# ---- each route passes the language on -------------------------------------------------


@pytest.mark.parametrize("language, expected", [(None, "en"), ("en", "en"), ("ar", "ar")])
def test_ask_passes_the_language_on(language, expected):
    helper = RecordingAssistant()
    response = client_with(helper).post("/api/ask", json=with_language({"question": "Hi"}, language))
    assert response.status_code == 200
    assert helper.calls == [("ask", expected)]


@pytest.mark.parametrize("language, expected", [(None, "en"), ("en", "en"), ("ar", "ar")])
def test_describe_passes_the_language_on(language, expected):
    helper = RecordingAssistant()
    client = client_with(helper)
    inspection = inspect(client, language)
    response = client.post("/api/describe", json=with_language({"id": inspection["id"]}, language))
    assert response.status_code == 200
    assert helper.calls == [("describe", expected)]
    assert response.get_json()["name"] == (ARABIC_INFO if expected == "ar" else ENGLISH_INFO).name


@pytest.mark.parametrize("language, expected", [(None, "en"), ("en", "en"), ("ar", "ar")])
@pytest.mark.parametrize("strokes", [None, STROKES])
def test_follow_up_passes_the_language_on(language, expected, strokes):
    helper = RecordingAssistant()
    client = client_with(helper)
    body = {"id": inspect(client)["id"], "question": "How heavy is it?"}
    if strokes is not None:
        body["strokes"] = strokes
    response = client.post("/api/follow_up", json=with_language(body, language))
    assert response.status_code == 200
    assert helper.calls == [("follow_up", expected)]


def test_a_follow_up_about_a_person_goes_to_the_chat_in_that_language():
    helper = RecordingAssistant()
    client = client_with(helper, target=SARA_TARGET)
    body = {"id": inspect(client, "ar")["id"], "question": "ما دورها؟", "language": "ar"}
    response = client.post("/api/follow_up", json=body)
    assert response.get_json() == {"text": "مرحبا", "used_web": False, "sources": []}
    assert helper.calls == [("ask", "ar")]
    assert helper.questions == ["This question is about Sara Ahmed, who the camera recognises. ما دورها؟"]


@pytest.mark.parametrize("language, expected", [(None, "en"), ("en", "en"), ("ar", "ar")])
def test_scene_passes_the_language_on(language, expected):
    helper = RecordingAssistant()
    response = client_with(helper).post("/api/scene", json=with_language({}, language))
    assert response.status_code == 200
    assert helper.calls == [("scene", expected)]


# ---- a wrong language ---------------------------------------------------------------------


@pytest.mark.parametrize("language", ["fr", "AR", "", None, 1, ["ar"]])
@pytest.mark.parametrize("route", ["/api/ask", "/api/describe", "/api/follow_up", "/api/scene", "/api/inspect"])
def test_a_wrong_language_is_400_with_a_plain_message(route, language):
    helper = RecordingAssistant()
    client = client_with(helper)
    inspection_id = inspect(client)["id"]
    body = {"question": "Hi", "id": inspection_id, "x": 0.5, "y": 0.5, "language": language}
    response = client.post(route, json=body)
    assert response.status_code == 400
    assert response.get_json() == {"error": 'The answer language must be "en" (English) or "ar" (Arabic), '
                                            'like {"language": "ar"}.'}
    assert helper.calls == []


# ---- the cache keeps the languages apart -----------------------------------------------------


def test_a_thing_looked_up_in_english_is_looked_up_again_in_arabic():
    helper = RecordingAssistant()
    client = client_with(helper)
    first = inspect(client, "en")
    client.post("/api/describe", json={"id": first["id"], "language": "en"})
    in_arabic = inspect(client, "ar")
    assert in_arabic["info"] is None                         # the English description is not shown
    reply = client.post("/api/describe", json={"id": in_arabic["id"], "language": "ar"}).get_json()
    assert reply["name"] == "حاسوب محمول"
    assert helper.calls == [("describe", "en"), ("describe", "ar")]
    assert inspect(client, "en")["info"]["name"] == "Laptop"       # each language is kept
    assert inspect(client, "ar")["info"]["name"] == "حاسوب محمول"
    assert inspect(client)["info"]["name"] == "Laptop"             # no language is English
    assert len(helper.calls) == 2


def test_a_second_click_in_the_same_language_is_free():
    helper = RecordingAssistant()
    client = client_with(helper)
    for _ in range(2):
        inspection = inspect(client, "ar")
        client.post("/api/describe", json={"id": inspection["id"], "language": "ar"})
    assert helper.calls == [("describe", "ar")]


# ---- a click on a person is unchanged ----------------------------------------------------


def test_a_click_on_a_person_in_arabic_sends_nothing_and_shows_what_was_written():
    helper = RecordingAssistant()
    client = client_with(helper, target=SARA_TARGET)
    english, arabic = inspect(client, "en"), inspect(client, "ar")
    for reply in (english, arabic):
        assert reply["person"] == {"id": "sara", "name": "Sara Ahmed", "role": "Team lead"}
        assert reply["describable"] is False and reply["info"] is None
    refused = client.post("/api/describe", json={"id": arabic["id"], "language": "ar"})
    assert refused.status_code == 400
    assert helper.calls == []


def test_the_unavailable_assistant_takes_the_language_too():
    stand_in = UnavailableAssistant("The notes file is missing.")
    for call in (lambda: stand_in.ask("Hi", language="ar"),
                 lambda: stand_in.describe(PHOTO, "laptop", language="ar"),
                 lambda: stand_in.follow_up(PHOTO, "laptop", None, (), "Hi", marked=None, language="ar"),
                 lambda: stand_in.describe_scene(PHOTO, language="ar")):
        with pytest.raises(AssistantError, match="notes file is missing"):
            call()


# ---- the page --------------------------------------------------------------------------------


@pytest.fixture
def page():
    return client_with(RecordingAssistant()).get("/").get_data(as_text=True)


def test_the_switch_sits_in_the_bar_between_camera_and_voice(page):
    assert "nav" in ids_around(page)["languageButton"]
    assert page.index('id="cameraButton"') < page.index('id="languageButton"') < page.index('id="voiceButton"')
    assert ('<button class="round language-button" id="languageButton" type="button" aria-pressed="false"\n'
            '          aria-label="Answer in Arabic" title="Answer in Arabic">EN</button>') in page


def test_the_switch_shows_en_or_ain_and_says_what_a_press_does(page):
    part = between(page, "function showLanguage()", "\n}\n")
    assert 'const label = arabicWanted ? "Answer in English" : "Answer in Arabic";' in part
    assert 'languageButton.textContent = arabicWanted ? "ع" : "EN";' in part
    assert 'languageButton.setAttribute("aria-pressed", arabicWanted ? "true" : "false");' in part
    assert 'languageButton.setAttribute("aria-label", label);' in part and "languageButton.title = label;" in part


def test_the_choice_is_remembered_and_english_until_chosen(page):
    assert 'let arabicWanted = readStored(LANGUAGE_KEY) === "ar";' in page
    click = between(page, 'languageButton.addEventListener("click"', "\n});\n")
    assert 'store(LANGUAGE_KEY, arabicWanted ? "ar" : null);' in click and "showLanguage();" in click
    assert 'return arabicWanted ? "ar" : "en";' in between(page, "function answerLanguage()", "\n}\n")


def test_the_choice_is_declared_right_above_answer_language(page):
    """answerLanguage() may be called early (the hosted voice): what it reads exists before it."""
    assert ('const LANGUAGE_KEY = "answerLanguage";\nlet arabicWanted = readStored(LANGUAGE_KEY) === "ar";\n'
            "function answerLanguage() {") in page
    assert page.count("let arabicWanted") == 1 and page.count("const LANGUAGE_KEY") == 1


def test_the_language_section_and_its_css_sit_where_the_spec_says(page):
    section = page.index("// ---- the language of the answers")
    assert page.index("// ---- light and dark") < section < page.index("// ---- the camera box")
    assert page.index("applyTheme(shownTheme(), false);") < section
    css = page.index(".language-button {")
    assert page.index(':root[data-theme="dark"] .theme-button .moon') < css < page.index(".nav .round { flex: none; }")


@pytest.mark.parametrize("request_line", [
    'postJson("/api/ask", { question, language: answerLanguage() })',
    'postJson("/api/describe", { id, language: answerLanguage() })',
    "const body = { id, question, language: answerLanguage() };",          # /api/follow_up
    'postJson("/api/scene", { language: answerLanguage() })',
    'postJson("/api/inspect", { ...body, language: answerLanguage() })',   # a kept description
])
def test_each_request_for_words_sends_the_language(page, request_line):
    assert request_line in page


def test_answers_take_their_direction_from_text_direction(page):
    message = between(page, "function messageText(", "\n}\n")
    assert "paragraph.dir = textDirection(text);" in message           # the chat's and the panel's answers
    info = between(page, "function showInfo(", "\n}\n")
    assert "const card = textDirection(cardText);" in info
    assert "specs.flatMap((spec) => [spec.name, spec.value])" in info and "[info.summary, " in info
    assert "detailTitle.dir = card;" in info and "list.dir = card;" in info
    assert "summary.dir = textDirection(info.summary);" in info
    assert "value.dir = textDirection(spec.value);" in info
    assert "captionAnswer.dir = textDirection(captionAnswer.textContent);" in between(page, "function showCaption(", "\n}\n")
    assert 'detailTitle.dir = "auto";' in between(page, "function showWaiting(", "\n}\n")   # a new click
    assert 'dir="auto"' not in between(page, "function showInfo(", "\n}\n")
    assert '<p class="caption-answer" id="captionAnswer"></p>' in page


def test_text_direction_counts_the_arabic_letters(page):
    direction = between(page, "function textDirection(", "\n}\n")
    assert "match(/\\p{L}/gu)" in direction and "/\\p{Script=Arabic}/u" in direction
    assert 'letters.length > 0 && arabic * 3 >= letters.length ? "rtl" : "ltr"' in direction


def test_typed_questions_and_fields_keep_dir_auto(page):
    assert 'node.dir = "auto";' in between(page, "function withOwnDirection(", "\n}\n")
    message = between(page, "function messageText(", "\n}\n")
    assert 'if (kind === "question") return withOwnDirection(make("p", "text", text));' in message
    for element in ('id="detailTitle" dir="auto"', 'id="captionQuestion" dir="auto"',
                    'placeholder="Ask a question" dir="auto"', 'placeholder="Ask about this" dir="auto"'):
        assert element in page, element


def js_string(page, name):
    match = re.search(r"const " + name + r' = ("(?:[^"\\]|\\.)*");', page)
    assert match, name
    return json.loads(match.group(1))


def test_marks_open_and_close_beside_arabic_punctuation(page):
    assert "«" in js_string(page, "MARK_OPEN_AFTER")
    closes = js_string(page, "MARK_CLOSE_BEFORE")
    for mark in "،؛؟»۔":
        assert mark in closes, mark


def test_a_bare_address_ends_before_arabic_punctuation(page):
    ends = js_string(page, "ADDRESS_END")
    for mark in "،؛؟»":
        assert mark in ends, mark


def test_text_is_still_put_in_only_through_text_content(page):
    for part in (between(page, "// ---- the language of the answers", "// ---- the camera box"),
                 between(page, "function messageText(", "function addMessage("),
                 between(page, "function showInfo(", "const errorText")):
        assert "innerHTML" not in part and "insertAdjacentHTML" not in part


def test_arabic_letters_have_fonts_on_windows_and_phones(page):
    stack = between(page, "--font:", ";")
    for font in ('"Segoe UI"', '"Noto Sans Arabic"', '"Noto Naskh Arabic"', "Tahoma"):
        assert font in stack, font
    assert stack.rstrip().endswith("sans-serif")


def test_right_to_left_answers_are_not_letter_spaced_and_keep_their_gaps(page):
    assert ".details-title:dir(rtl) { letter-spacing: 0; }" in page
    assert ".specs:dir(rtl) dt { padding-right: 0; padding-left: 1rem; }" in page
    assert '.specs[dir="rtl"] dd { text-align: right; }' in page and '.specs[dir="ltr"] dd { text-align: left; }' in page


# ---- the page's own functions, run in headless Microsoft Edge -----------------------------
#
# Skipped by default (marked hardware: it needs Edge on this laptop, not the network). Run it with
#   .venv\Scripts\python.exe -m pytest tests/test_app_language.py -m hardware

EDGE_PLACES = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
               r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")

# Each case: a JavaScript expression run on the page, and what it must give.
EDGE_CASES = {
    "bold before an Arabic comma": ('shown("- المعالج: **Intel Core i5**، وهو سريع")',
                                    {"text": "• المعالج: Intel Core i5، وهو سريع", "strong": ["Intel Core i5"],
                                     "em": []}),
    "bold before an Arabic question mark": ('shown("هل تفهم **ذلك**؟")',
                                            {"text": "هل تفهم ذلك؟", "strong": ["ذلك"], "em": []}),
    "italic inside guillemets": ('shown("قال «*مرحبا*» للزوار")',
                                 {"text": "قال «مرحبا» للزوار", "strong": [], "em": ["مرحبا"]}),
    "an address before an Arabic comma": ('link("الصفحة: https://example.com/ar، وهي جيدة")',
                                          {"href": "https://example.com/ar", "text": "link",
                                           "after": "، وهي جيدة"}),
    "Arabic sentence starting with a product name": ('textDirection("iPhone 15 Pro هو أحدث هاتف من آبل، وكاميرته ممتازة.")',
                                                     "rtl"),
    "English sentence with one Arabic word": ('textDirection("The word مرحبا means hello in Arabic.")', "ltr"),
    "digits only": ('textDirection("١٢٣ 456")', "ltr"),
    "empty": ('textDirection("")', "ltr"),
    "half and half": ('textDirection("abcd ابجد")', "rtl"),
    "one Arabic letter in three": ('textDirection("ab ج")', "rtl"),
    "one Arabic letter in four": ('textDirection("abc ج")', "ltr"),
    "an Arabic card whose first name is CPU": ("card()", {"title": "rtl", "list": "rtl", "values": ["ltr", "rtl"]}),
    "an English card": ("englishCard()", {"title": "ltr", "list": "ltr", "values": ["ltr", "ltr"]}),
    "an answer paragraph": ('answerDir("iPhone 15 Pro هاتف جديد.")', "rtl"),
}

EDGE_HARNESS = """<!doctype html><html lang="en"><head><meta charset="utf-8"></head><body>
<div id="title"></div><div id="content"></div><ol id="messages"><li id="hint"></li></ol>
<script>
const MAX_SOURCES = 3;
const say = () => {};
const sentences = () => "";
const detailTitle = document.getElementById("title"), detailContent = document.getElementById("content");
const messages = document.getElementById("messages"), hint = document.getElementById("hint");
FUNCTIONS
function holder(text) { const p = document.createElement("p"); p.append(...answerNodes(text)); return p; }
function shown(text) {
  const p = holder(text);
  return { text: p.textContent, strong: [...p.querySelectorAll("strong")].map((n) => n.textContent),
           em: [...p.querySelectorAll("em")].map((n) => n.textContent) };
}
function link(text) {
  const a = holder(text).querySelector("a");
  return { href: a.href.replace(/\\/$/, ""), text: a.textContent, after: a.nextSibling ? a.nextSibling.textContent : "" };
}
function cardOf(info) {
  showInfo(info);
  return { title: detailTitle.dir, list: detailContent.querySelector("dl").dir,
           values: [...detailContent.querySelectorAll("dd")].map((dd) => dd.dir) };
}
const card = () => cardOf({ name: "حاسوب", summary: "حاسوب خفيف للعمل.", used_web: false,
                            specs: [{ name: "CPU", value: "Intel Core i5" }, { name: "الوزن", value: "١٫٢ كجم" }] });
const englishCard = () => cardOf({ name: "Laptop", summary: "A light laptop.", used_web: false,
                                   specs: [{ name: "CPU", value: "Intel Core i5" }, { name: "Weight", value: "1.2 kg" }] });
const answerDir = (text) => messageText("answer", text).dir;
const results = {};
for (const [name, expression] of CASES) {
  try { results[name] = eval(expression); } catch (error) { results[name] = "error: " + error; }
}
document.body.textContent = "OUT:" + encodeURIComponent(JSON.stringify(results)) + ":END";
</script></body></html>"""


def edge_page(page):
    """The harness with the page's own helpers and answer functions in it."""
    script = page[page.index("// ---- small helpers"):]
    functions = (between(script, "function make(", "// The language of the assistant")
                 + between(script, "// ---- the chat ----", "function setBusy(")
                 + between(script, "function showInfo(", "const errorText"))
    cases = json.dumps([[name, expression] for name, (expression, _) in EDGE_CASES.items()], ensure_ascii=False)
    return EDGE_HARNESS.replace("FUNCTIONS", functions).replace("for (const [name, expression] of CASES)",
                                                                 f"for (const [name, expression] of {cases})")


@pytest.mark.hardware
def test_the_page_functions_in_headless_edge(page, tmp_path):
    edge = next((place for place in EDGE_PLACES if Path(place).is_file()), shutil.which("msedge"))
    if edge is None:
        pytest.skip("Microsoft Edge is not installed")
    harness = tmp_path / "harness.html"
    harness.write_text(edge_page(page), encoding="utf-8")
    run = subprocess.run([edge, "--headless=new", "--disable-gpu", f"--user-data-dir={tmp_path / 'profile'}",
                          "--dump-dom", harness.as_uri()], capture_output=True, text=True, encoding="utf-8",
                         timeout=90)
    found = re.search(r"OUT:(.*?):END", run.stdout)
    assert found, run.stdout[-2000:]
    results = json.loads(urllib.parse.unquote(found.group(1)))
    for name, (_, expected) in EDGE_CASES.items():
        assert results[name] == expected, name
