"""M26: Arabic for the voice and for listening, checked by the shape of the page's script.

There is no JavaScript engine here. The rules that are plain regular expressions (where a line
gets a pause, the Arabic stop words) are copied out of the page and tried in Python, which reads
these patterns the same way.
"""

from __future__ import annotations

import re
import unicodedata

import pytest

from test_app import TEMPLATE, between, ids_around, rendered_page

ARABIC_TEST_SENTENCE = "مرحباً. هذا هو الصوت الذي ستستخدمه الصفحة."
NO_VOICE_NOTE = ("This browser has no Arabic voice, so Arabic answers are shown but not spoken. "
                 "Microsoft Edge has Arabic voices.")


def page_text():
    return TEMPLATE.read_text(encoding="utf-8")


def body(page, function):
    """A function's code, from its first line to its closing brace."""
    return between(page, function, "\n}\n")


# ---- speaking: an Arabic piece goes to an Arabic voice, or is not spoken -------------------------


def test_each_piece_is_judged_by_itself():
    page = page_text()
    rule = body(page, "function isArabicPiece(")
    assert "/\\p{Script=Arabic}/u" in rule and "/\\p{Script=Latin}/u" in rule
    assert "return arabic > latin;" in rule                   # a tie is English
    assert "textDirection(" not in rule                       # the one-third rule is for the layout
    piece = body(page, "function speakPiece(")
    # The Arabic check comes before the hosted voice is even looked at.
    assert piece.index("if (isArabicPiece(talk.pieces[talk.next])) {") < piece.index("if (talk.hosted) {")
    assert "speakArabicPiece(talk);" in piece and "browserSpeaksPiece(talk);" in piece


def test_an_arabic_piece_never_reaches_the_hosted_voice():
    page = page_text()
    arabic = body(page, "function speakArabicPiece(")
    for hosted in ("playHostedPiece(", "fetchClip(", "/api/voice/say"):
        assert hosted not in arabic, hosted
    # The next piece is fetched early only when it is English, and only once the Arabic piece has
    # really started (its "start"), so a hush before that fetches nothing.
    early = body(page, "function fetchNextHostedPiece(")
    assert "!isArabicPiece(talk.pieces[next])" in early and "hostedClip(talk, next)" in early
    assert "if (talking !== talk || !talk.hosted" in early
    assert "fetchNextHostedPiece(" not in arabic
    words = between(page, "function utterance(", "function pieceStarted(")
    start = words[words.index("words.onstart = () => {"):words.index("words.onend")]
    assert start.index("pieceStarted(talk, piece);") < start.index("fetchNextHostedPiece(talk);")
    # And the hosted request itself still refuses Arabic letters, as M23 made it.
    assert "ARABIC_LETTERS.test(piece) || piece.length > hosted.maxChars" in body(page, "async function fetchClip(")


def test_an_arabic_piece_is_said_only_by_an_arabic_voice_or_not_at_all():
    page = page_text()
    arabic = body(page, "function speakArabicPiece(")
    assert "talk.arabicVoice = arabicVoice();" in arabic
    assert arabic.index("if (talk.arabicVoice) {") < arabic.index("browserSpeaksPiece(talk);")
    assert arabic.count("pieceEnded(talk);") == 2            # no speech, or no Arabic voice: the talk goes on
    assert arabic.index("if (!speech) {") < arabic.index("whenVoicesListed(")   # the voices may arrive late
    assert arabic.count("if (talking !== talk) return;") == 2   # also after waiting for late voices
    words = between(page, "function utterance(", "function pieceStarted(")
    assert "words.voice = talk.arabicVoice;" in words and "words.lang = asText(talk.arabicVoice.lang);" in words
    # The one place the browser is handed a piece, for both languages.
    assert page.count("speech.synth.speak(") == 1
    assert "speech.synth.speak(" in body(page, "function browserSpeaksPiece(")


def test_arabic_voices_are_ranked_natural_online_then_others_egypt_first():
    page = page_text()
    assert "const isArabicVoice = (voice) => /^ar(-|$)/.test(langOf(voice));" in page
    rank = body(page, "function arabicVoiceRank(")
    assert "/natural/i.test(name) ? 0 : /online/i.test(name) || /^google/i.test(name) ? 1 : 2" in rank
    assert 'lang === "ar-eg" ? 0 : lang === "ar-sa" ? 1 : 2' in rank
    assert "return group * 3 + region;" in rank
    best = body(page, "function arabicVoice()")
    assert "voices.filter((voice) => voice && isArabicVoice(voice))" in best   # never another language
    assert "arabicVoiceRank(a) - arabicVoiceRank(b)" in best and "[0] || null" in best


@pytest.mark.parametrize("lang, arabic", [("ar", True), ("ar-EG", True), ("ar_SA", True), ("ar-sa", True),
                                          ("en-US", False), ("fr-FR", False), ("arn-CL", False), ("", False)])
def test_only_a_language_that_starts_with_ar_is_arabic(lang, arabic):
    rule = re.search(r"const isArabicVoice = \(voice\) => /(.+)/\.test", page_text()).group(1)
    assert bool(re.search(rule, lang.replace("_", "-").lower())) is arabic


def test_sentences_of_two_languages_are_never_one_piece():
    page = page_text()
    pieces = body(page, "function speechPieces(")
    assert "sameLanguage(pieces[last], rest)" in pieces
    same = body(page, "function sameLanguage(")
    assert "isArabicPiece(one) === isArabicPiece(other)" in same
    assert "!hasLetters(one) || !hasLetters(other)" in same       # "١." joins its Arabic sentence


def arabic_for_speaking(text):
    """isArabicPiece in Python: more Arabic letters than Latin letters."""
    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    arabic = sum(1 for char in letters if unicodedata.name(char, "").startswith("ARABIC"))
    latin = sum(1 for char in letters if unicodedata.name(char, "").startswith("LATIN"))
    return arabic > latin


@pytest.mark.parametrize("text, arabic", [
    ("Name: Omar أحمد.", False),                                      # an Arabic name in an English sentence
    ("I can see محمد عبد الرحمن now, on the left.", False),           # the page's own find sentence
    ("I can see سارة, on the right.", False),
    ("ab سب", False),                                                  # a tie is English
    ("إنه هاتف iPhone 15 Pro رائع.", True),                            # an Arabic answer with a product name
    ("هذه سيارة روبوت صغيرة تعمل بالبطارية.", True),
    ("I can see the keys, on the left.", False),
])
def test_a_piece_is_arabic_only_when_arabic_letters_outnumber_latin_ones(text, arabic):
    assert arabic_for_speaking(text) is arabic


def test_an_arabic_piece_waits_once_for_arabic_voices_that_come_late():
    page = page_text()
    assert "voicesChangedOnce = true;" in body(page, "function voicesChanged()")
    arabic = body(page, "function speakArabicPiece(")
    assert "} else if (!voicesChangedOnce && !arabicVoicesWaitedFor) {" in arabic
    assert "whenArabicVoicesMayCome(() => speakArabicPiece(talk));" in arabic
    late = body(page, "function whenArabicVoicesMayCome(")
    assert "arabicVoicesWaitedFor = true;" in late                 # never again in this visit
    assert "voiceWaiters.push(go);" in late and "setTimeout(go, VOICES_WAIT_MS);" in late
    assert "if (done) return;" in late                             # the change and the timer: one look only


# ---- cleaning for the ear: Arabic punctuation --------------------------------------------------------


def pause_rule(page):
    lines = body(page, "function readableLines(")
    return re.compile(re.search(r"\.map\(\(line\) => \(/(.+)/\.test\(line\)", lines).group(1))


@pytest.mark.parametrize("line", ["هل هذه سيارة؟", "نعم،", "إنها صغيرة؛", "وتعمل بالبطارية۔",
                                  "Done.", "Really?", "Wow!", "A list:", "first,", "second;"])
def test_no_full_stop_is_added_after_a_pause(line):
    assert pause_rule(page_text()).search(line)


@pytest.mark.parametrize("line", ["السرعة ١٢ كم", "A line without a stop", "١٢٣"])
def test_a_full_stop_is_added_after_a_line_without_one(line):
    assert not pause_rule(page_text()).search(line)


def test_arabic_sentence_ends_and_cuts():
    page = page_text()
    cut = body(page, "function cutAtSentence(")
    assert 'head.lastIndexOf("؟ ")' in cut and 'head.lastIndexOf("۔ ")' in cut
    assert 'head.lastIndexOf(". "), head.lastIndexOf("! "), head.lastIndexOf("? ")' in cut   # English as before
    pieces = body(page, "function speechPieces(")
    assert "text.split(/(?<=[.!?؟۔])\\s+/)" in pieces
    assert 'rest.lastIndexOf(", ", PIECE_CHARS), rest.lastIndexOf("، ", PIECE_CHARS)' in pieces
    assert 'rest.lastIndexOf("؛ ", PIECE_CHARS)' in pieces
    # Arabic-Indic digits are left alone: the list-number rule matches ASCII digits only (no "u" flag).
    assert r"line.replace(/^\s*(#{1,6}|>|[-*+]|\d+[.)])\s+/, " in body(page, "function readableLines(")


# ---- the Voice box: the Test sentence and the note -------------------------------------------------------


def test_the_test_button_speaks_arabic_while_the_switch_says_arabic():
    page = page_text()
    assert f'const ARABIC_TEST_SPEECH = "{ARABIC_TEST_SENTENCE}";' in page
    assert 'const TEST_SPEECH = "Hello. This is the voice the page will use.";' in page
    assert 'return answerLanguage() === "ar" ? ARABIC_TEST_SPEECH : TEST_SPEECH;' in body(page, "function testSentence()")
    assert 'voiceTest.addEventListener("click", () => speak(testSentence(), "test"));' in page


def test_the_note_under_the_voices_is_shown_only_for_arabic():
    page = rendered_page()
    assert "voiceBox" in ids_around(page)["voiceArabicNote"]
    assert '<p class="voice-arabic-note" id="voiceArabicNote" aria-live="polite" hidden></p>' in page
    assert page.index('id="voiceHelp"') < page.index('id="voiceArabicNote"')     # under the list of voices
    script = page_text()
    parts = re.search(r'const NO_ARABIC_VOICE = "([^"]+)" \+\s+"([^"]+)";', script)
    assert parts and parts.group(1) + parts.group(2) == NO_VOICE_NOTE
    note = body(script, "function drawArabicVoiceNote()")
    assert 'voiceArabicNote.hidden = answerLanguage() !== "ar";' in note
    assert '"Arabic answers are spoken by: " + shortVoiceName(voice)' in note and "NO_ARABIC_VOICE" in note
    assert "setText(voiceArabicNote," in note
    # Redrawn with the list of voices (also when it arrives late) and when the switch is pressed.
    choices = body(script, "function drawVoiceChoices()")
    assert choices.index("drawArabicVoiceNote();") < choices.index("if (!speech) return;")
    assert "drawVoiceChoices();" in body(script, "function voicesChanged()")
    assert "drawArabicVoiceNote();" in body(script, "function voiceLanguageChanged()")
    assert ".voice-arabic-note {" in script and "font-size: 0.8rem" in script


# ---- listening -----------------------------------------------------------------------------------------


def test_the_browser_listens_for_arabic_while_the_switch_says_arabic():
    page = page_text()
    start = between(page, "function startListening(", "function beginHearing(")
    assert ('recognizer.lang = "en-US";\n    if (answerLanguage() === "ar") recognizer.lang = "ar-EG";') in start
    # Hosted listening is unchanged: it names the language in its address.
    assert '"/api/voice/hear?language=" + encodeURIComponent(answerLanguage())' in body(page, "async function sendRecording(")


def arabic_stop_words(page):
    words = re.search(r"const ARABIC_STOP_WORDS = \[(.+)\];", page).group(1)
    return re.findall(r'"([^"]+)"', words)


LRM, RLM, ALM = chr(0x200E), chr(0x200F), chr(0x061C)    # invisible direction marks


def is_arabic_stop(page, said):
    """What isStopWords does for Arabic, with the page's own list, vowel marks and direction marks."""
    vowels = re.compile(re.search(r"const ARABIC_VOWEL_MARKS = /(.+)/g;", page).group(1))
    directions = re.compile(re.search(r"const DIRECTION_MARKS = /(.+)/g;", page).group(1))
    cleaned = " ".join(said.lower().split())

    def bare(text):
        return re.sub(r"[.!؟،]+$", "", directions.sub("", vowels.sub("", text)).strip()).strip()

    return any(bare(words) == bare(cleaned) for words in arabic_stop_words(page))


def test_the_arabic_stop_words_are_the_ones_asked_for():
    assert arabic_stop_words(page_text()) == ["توقف", "قف", "خلاص", "كفاية", "شكراً", "شكرا", "مع السلامة"]
    stop = body(page_text(), "function isStopWords(")
    assert "return STOP_WORDS.test(cleaned) || isArabicStopWords(cleaned);" in stop
    arabic = body(page_text(), "function isArabicStopWords(")
    assert ".replace(ARABIC_VOWEL_MARKS, \"\")" in arabic and '.replace(DIRECTION_MARKS, "")' in arabic
    assert '.replace(/[.!؟،]+$/, "")' in arabic
    assert r"const DIRECTION_MARKS = /[\u200E\u200F\u061C]/g;" in page_text()   # written as escapes, not invisible


@pytest.mark.parametrize("said", ["توقف", "قف", "خلاص", "كفاية", "شكراً", "شكرا", "مع السلامة",
                                  "شكراً!", "خلاص.", "توقف؟", "شُكْرًا", "مع  السلامة.", " قف ", "شكرا،",
                                  RLM + "شكراً" + RLM, LRM + "خلاص،", "توقف" + ALM + "!", "كفاية" + RLM + "."])
def test_an_arabic_stop_word_alone_ends_the_conversation(said):
    assert is_arabic_stop(page_text(), said)


@pytest.mark.parametrize("said", ["توقف السيارة", "شكرا على المساعدة", "ما هذا؟", "قف هنا من فضلك",
                                  "كفاية كلام عن السيارة", "مع"])
def test_an_arabic_stop_word_inside_a_question_does_not_end_it(said):
    assert not is_arabic_stop(page_text(), said)


# ---- the language switch stops speaking and listening ------------------------------------------------


def test_the_switch_stops_speaking_and_listening_at_once():
    page = page_text()
    click = between(page, 'languageButton.addEventListener("click"', "\n});\n")
    assert click.index("showLanguage();") < click.index("voiceLanguageChanged();")
    changed = body(page, "function voiceLanguageChanged()")
    assert 'if (conversation) conversationNext(conversation, "stop");' in changed
    assert "cancelListening();" in changed and "hush();" in changed


def test_nothing_new_is_spoken_by_itself():
    page = page_text()
    for function in ("function speakArabicPiece(", "function fetchNextHostedPiece(", "function voiceLanguageChanged()",
                     "function drawArabicVoiceNote()", "function testSentence()", "function isArabicStopWords(",
                     "function whenArabicVoicesMayCome(", "function isArabicPiece("):
        code = body(page, function)
        assert "say(" not in code and "speak(" not in code, function
    # The only new call to speak() is the Test button's, which a click starts.
    assert page.count("speak(testSentence()") == 1
