"""M18: bold, italic and code in the assistant's answers are shown, not their marks.

The page's script cannot run here (no JavaScript engine), so its rules are checked by the shape
of the script; the rules themselves were run once in a real browser engine (headless Edge) on a
table of cases, as the M18 spec describes.
"""

import json
import re

import pytest

from test_app import TEMPLATE, between, css_rule


def page_text():
    return TEMPLATE.read_text(encoding="utf-8")


def emphasis_section(page):
    """The code of the emphasis rules: from their heading comment to answerNodes, which joins them."""
    return between(page, "// ---- bold, italic and code", "function answerNodes(")


def js_pattern(page, name):
    """A regular expression constant of the page as a Python pattern (the syntaxes agree for these)."""
    match = re.search(r"const " + name + r" = /(.+)/;", page)
    assert match, name
    return re.compile(match.group(1))


def js_string(page, name):
    match = re.search(r"const " + name + r' = ("(?:[^"\\]|\\.)*");', page)
    assert match, name
    return json.loads(match.group(1))


def test_answer_nodes_joins_the_emphasis_parts_and_the_link_parts():
    page = page_text()
    nodes = between(page, "function answerNodes(", "\n}\n")
    assert "for (const part of emphasisParts(text))" in nodes
    assert "linkedNodes(words)" in nodes
    assert "linkParts(parts.map((part) => part.text).join(\"\"))" in between(page, "function linkedNodes(", "\n}\n")
    # A code span is never looked at for links: it is made as it is, in its <code>.
    assert "styledNode(document.createTextNode(part.text), part)" in nodes
    # answerNodes stays the one function for an answer: the chat, the panel's chat, a summary.
    assert "paragraph.append(...answerNodes(text));" in between(page, "function messageText(", "\n}\n")
    assert "summary.append(...answerNodes(asText(info.summary)));" in between(page, "function showInfo(", "\n}\n")


def test_emphasis_parts_is_pure_and_has_no_page_in_it():
    page = page_text()
    rules = between(page, "// ---- bold, italic and code", "const plainWords")
    for function in ("function emphasisParts(", "function lineParts(", "function lineTokens(", "function addMark(",
                     "function styledParts(", "function markOpens(", "function markCloses(", "function codeEnd("):
        assert function in rules, function
    for word in ("document", "make(", "append", "textContent"):
        assert word not in rules, word
    assert 'asText(text).split("\\n")' in between(page, "function emphasisParts(", "\n}\n")   # no mark crosses a line


def test_nodes_are_made_with_create_element_and_text_only():
    page = page_text()
    section = emphasis_section(page) + between(page, "function answerNodes(", "\n}\n")
    assert "const box = make(tag);" in between(page, "function styledNode(", "\n}\n")   # make: createElement
    assert "document.createTextNode(" in between(page, "function wordNodes(", "\n}\n")
    assert "linkNode(piece)" in between(page, "function linkedNodes(", "\n}\n")         # every link rule kept
    for word in ("outerHTML", "innerText", "DOMParser", "createRange", ".replace("):
        assert word not in section, word


@pytest.mark.parametrize("word", ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"])
def test_the_page_never_writes_html(word):
    assert word not in page_text()


def test_link_parts_is_unchanged_for_speech_and_the_caption():
    page = page_text()
    parts = between(page, "function linkParts(", "\n}\n")
    assert "parts.push({ text: whole, url, label, markdown });" in parts
    assert "if (url === null) continue;" in parts
    assert "linkParts(text).map((part) => (part.url ? part.label : part.text))" in between(page, "function earText(", "\n}\n")
    assert "linkParts(asText(text).replaceAll(LINK_MARK, \"\"))" in between(page, "function captionNodes(", "\n}\n")
    # Speech and the caption drop the marks without a space where they were; "_" stays.
    lines = between(page, "function readableLines(", "\n}\n")
    assert r"line.replace(/^\s*(#{1,6}|>|[-*+]|\d+[.)])\s+/, " in lines
    assert '.replace(/[*`~]+/g, "")' in lines
    assert r'.replace(/(?<=[\p{L}\p{N}])[*`~]+(?=[\p{L}\p{N}])/gu, " ")' in lines   # "5*4*3" is not "543"
    assert '.replace(/[|#]+/g, " ")' in lines
    assert "_" not in "".join(re.findall(r"\.replace\(/([^/]*)/g", lines))


def test_a_bare_address_ends_before_stars_and_marks_do_not_hide_a_colon():
    page = page_text()
    assert "*" in js_string(page, "ADDRESS_END")
    says = re.search(r"const SAYS_WHAT_IT_IS = /(.+)/i;", page)
    rule = re.compile(says.group(1), re.IGNORECASE)
    for before in ("**Profile:** ", "*Profile:* ", "`Profile:` ", "LinkedIn: ", "His profile is **"):
        assert rule.search(before), before
    for before in ("**Profile** ", "The code is on **"):
        assert not rule.search(before), before


def test_trim_address_counts_the_brackets_once():
    trim = between(page_text(), "function trimAddress(", "\n}\n")
    assert trim.count(".split(") == 2                       # once each, before the loop
    assert trim.index(".split(") < trim.index("while (")
    assert 'if (address[end - 1] === ")") closed -= 1;' in trim


def test_code_may_sit_inside_bold_and_a_long_line_is_searched_once():
    page = page_text()
    tokens = between(page, "function lineTokens(", "\n}\n")
    assert 'const opens = markOpens(line, at, at + 1, "*");' in tokens
    assert "end = opens && codeCanClose ? codeEnd(line, at) : -1;" in tokens
    assert "if (opens && end < 0) codeCanClose = false;" in tokens
    assert 'markCloses(line, end, end + 1, "*")' in between(page, "function codeEnd(", "\n}\n")
    head = between(page, "// ---- links in answers", "const LINK_LABEL_CHARS")
    assert "a code span inside a label is shown as written" in head


def test_a_mark_opens_and_closes_only_beside_the_characters_the_spec_names():
    page = page_text()
    assert set(js_string(page, "MARK_OPEN_AFTER")) == set("([{\"':-–")
    assert set(js_string(page, "MARK_CLOSE_BEFORE")) == set(".,;:!?)]}\"'")
    opens = between(page, "function markOpens(", "\n}\n")
    assert "start === 0 || isSpace(before) || MARK_OPEN_AFTER.includes(before) || also.includes(before)" in opens
    assert "after !== undefined && !isSpace(after)" in opens
    closes = between(page, "function markCloses(", "\n}\n")
    assert "start > 0 && !isSpace(before)" in closes
    assert "after === undefined || isSpace(after) || MARK_CLOSE_BEFORE.includes(after) || also.includes(after)" in closes
    # Only a backtick passes "*" as `also`: stars themselves keep the two lists.
    assert "markOpens(line, at, end), closes = markCloses(line, at, end)" in between(page, "function lineTokens(", "\n}\n")


def test_only_stars_and_backticks_are_marks_and_long_runs_stay():
    page = page_text()
    tokens = between(page, "function lineTokens(", "\n}\n")
    assert 'line[end] !== "*" && line[end] !== "`"' in tokens          # underscores are never marks
    assert "if (run.length > 3) tokens.push({ text: run });" in tokens   # "****" and 5000 stars stay
    # A mark without a partner stays in the text as it was written.
    mark = between(page, "function addMark(", "\n}\n")
    assert "const partner = same.pop();" in mark
    assert "while (other.length && other[other.length - 1] > partner) other.pop();" in mark
    assert "if (token.role)" in between(page, "function styledParts(", "\n}\n")


@pytest.mark.parametrize("line, found", [
    ("# Title", True), ("###### Title", True), ("  ## Title", True),
    ("####### Title", False), ("#3 is the laptop", False), ("#", False), ("Title # not", False),
])
def test_heading_pattern(line, found):
    assert bool(js_pattern(page_text(), "HEADING").search(line)) is found


@pytest.mark.parametrize("line, found", [
    ("- item", True), ("* item", True), ("+ item", True), ("    - item", True),
    ("1. item", False), ("**bold** word", False), ("*italic*", False), ("-5 degrees", False), ("---", False),
])
def test_bullet_pattern(line, found):
    assert bool(js_pattern(page_text(), "BULLET").search(line)) is found


def test_a_heading_is_bold_and_a_bullet_keeps_its_indentation():
    page = page_text()
    line = between(page, "function lineParts(", "\n}\n")
    assert "styledParts(lineTokens(line.slice(heading[0].length)), true)" in line   # the title in bold
    assert 'plainPart(bullet[1] + "•")' in line                                    # the indentation kept


def test_the_specs_list_shows_short_labels_and_values():
    page = page_text()
    info = between(page, "function showInfo(", "\n}\n")
    assert "value.append(...answerNodes(asText(spec.value)));" in info
    assert 'make("dt", "", plainWords(asText(spec.name)))' in info
    assert "const plainWords = (text) => emphasisParts(text).map((part) => part.text).join(\"\");" in page


def test_the_css_for_bold_italic_and_code_follows_the_link_rules():
    page = page_text()
    words = css_rule(page, ".msg strong, .msg em, .details-content strong, .details-content em")
    assert "color: inherit" in words
    code = css_rule(page, ".msg code, .details-content code")
    for part in ("monospace", "background: var(--hairline)", "border-radius"):
        assert part in code, part
    assert "border:" not in code and "border-width" not in code
    # Directly after the .answer-link rules, before the question form.
    after_links = page.index(".caption .answer-link:focus-visible")
    assert after_links < page.index(".msg strong, .msg em") < page.index(".ask-form {")


def test_the_links_comment_says_which_part_of_markdown_is_shown():
    page = page_text()
    assert "this is not a Markdown renderer" not in page
    head = between(page, "// ---- links in answers", "const LINK_LABEL_CHARS")
    assert "A small, fixed part of Markdown is shown as well (emphasisParts, below)" in head
    assert "else stays plain text, exactly as written." in head
