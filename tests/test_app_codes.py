"""M16 Part D: the row of codes under the In view pills, and the card with a code's text.

The page's script cannot run here (no JavaScript engine), so its rules are checked by the
shape of the template, as tests/test_app_discover.py does.
"""

import pytest

from test_app import TEMPLATE, between, client_for, css_rule, ids_around

PRIVACY = ("QR codes and barcodes are read on this laptop, and the program sends their text nowhere. "
           "A picture sent for a click, for Describe or for Auto-discover can still show a code.")
ALTERED = "This text was shortened or cleaned, so it may not be exactly what the code holds."


@pytest.fixture(scope="module")
def page():
    return TEMPLATE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rendered():
    return client_for().get("/").get_data(as_text=True)


@pytest.fixture(scope="module")
def section(page):
    """The script's codes section."""
    return between(page, "// ---- codes: QR codes and barcodes in view", "// ---- the chat")


# ---- the HTML --------------------------------------------------------------------------


def test_the_row_and_the_card_are_under_the_in_view_pills(rendered):
    inview = between(rendered, '<div class="inview"', '<p class="privacy" id="privacy">')
    assert inview.index('id="things"') < inview.index('id="nothing"') < inview.index('id="codeRow"')
    assert inview.index('id="codeRow"') < inview.index('id="codeCard"')
    assert '<ul class="things code-row" id="codeRow" aria-label="Codes in view" hidden></ul>' in rendered
    around = ids_around(rendered)
    for inside in ("codeRow", "codeCard"):
        assert "watch" in around[inside] and "codes" in around[inside], inside
    for inside in ("codeKind", "codeText", "codeOpen", "codeCopy", "codeClose"):
        assert "codeCard" in around[inside], inside


def test_the_card_has_its_kind_text_buttons_and_close_button(rendered):
    card = between(rendered, 'id="codeCard"', '<p class="privacy"')
    assert 'class="code-card glass"' in rendered and "hidden>" in card[:120]
    assert '<a class="pill primary small" id="codeOpen" target="_blank" rel="noopener noreferrer" hidden>Open link</a>' in card
    assert ('<button class="pill secondary small" id="codeCopy" type="button">'
            '<span id="codeCopyWord" aria-live="polite">Copy</span></button>') in card
    assert '<p class="code-text" id="codeText" tabindex="-1"></p>' in card       # focusable by the script only
    assert f'<p class="code-note" id="codeNote" hidden>{ALTERED}</p>' in card
    assert 'id="codeClose" type="button" title="Close (Esc)"' in card
    assert 'class="round"' in card                              # the same round button as the details panel
    assert "href=" not in card                                  # no address until a code with a link is opened


def test_the_privacy_paragraph_names_the_codes(rendered):
    privacy = between(rendered, '<p class="privacy" id="privacy">', "</p>")
    assert privacy.endswith(PRIVACY)


# ---- the script --------------------------------------------------------------------------


def test_poll_draws_the_codes_right_after_the_objects(page):
    poll = between(page, "async function poll()", "\n}\n")
    assert "drawObjects(status);\n    drawCodes(status);\n" in poll


def test_the_section_sits_before_the_chat(page):
    assert page.index("// ---- codes: QR codes and barcodes in view") < page.index("// ---- the chat")
    assert page.index("function drawObjects(") < page.index("function drawCodes(")


def test_the_text_goes_in_only_through_text_content(section):
    assert "setText(codeText, code.text);" in section
    assert "setText(codeKind, codeKindName(code));" in section
    assert 'make("button", "thing")' in section and 'make("span", "thing-name")' in section
    assert "setText(item.firstChild.lastChild, code.preview);" in section
    for word in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "createContextualFragment"):
        assert word not in section, word


def test_the_text_is_never_spoken_or_sent(section):
    for word in ("speak(", "say(", "fetch(", "postJson(", "sendJson(", "/api/", "sendPrompt", "localStorage"):
        assert word not in section, word


def test_a_link_needs_the_status_and_web_address_and_is_never_opened_by_itself(section):
    assert "const codeUrl = (code) => (code.link && !code.altered ? webAddress(code.link) : null);" in section
    assert "altered: raw.altered === true," in section
    opening = between(section, "function openCodeCard(", "\n}\n")
    assert "codeNote.hidden = !code.altered;" in opening
    assert "codeOpen.hidden = url === null;" in opening
    assert "codeOpen.href = url.href;" in opening and "codeOpen.title = url.href;" in opening
    assert 'codeOpen.removeAttribute("href");' in opening
    for word in ("window.open", "location", ".click()"):
        assert word not in section, word


def test_a_pill_shows_the_site_name_of_a_link_or_the_first_28_characters(section):
    assert "const CODE_PREVIEW_CHARS = 28;" in section
    preview = between(section, "function codePreview(", "\n}\n")
    assert "return cutPreview(url ? siteName(url) : code.text);" in preview    # a site name is cut too
    cut = between(section, "function cutPreview(", "\n}\n")
    assert 'letters.slice(0, CODE_PREVIEW_CHARS).join("") + "…"' in cut
    assert 'const CODE_NAMES = { qr: "QR code", barcode: "Barcode" };' in section
    assert 'code.format.replace(/_/g, "-")' in section and '" · "' in section   # Barcode · EAN-13


def test_copy_says_copied_or_could_not_copy_for_two_seconds(section):
    assert "const COPY_NOTE_MS = 2000;" in section
    copy = between(section, "async function copyCode(", "\n}\n")
    assert "navigator.clipboard.writeText(code.text)" in copy
    assert 'showCopyWord(copied ? "Copied" : "Could not copy");' in copy
    assert 'setTimeout(() => showCopyWord("Copy"), COPY_NOTE_MS)' in copy
    assert 'codeCopy.addEventListener("click", copyCode);' in section
    assert "setText(codeCopyWord, word);" in between(section, "function showCopyWord(", "\n}\n")


def test_the_card_keeps_its_code_until_it_is_closed_or_another_is_opened(section):
    draw = between(section, "function drawCodes(", "\n}\n")
    assert "codeCard" not in draw and "openCode =" not in draw      # a new status never closes the card
    assert "if (!changed(\"codes\", shown)) return;" in draw          # redrawn only when it changed
    assert "codeRow.hidden = items.length === 0;" in draw


def test_the_pills_are_reused_and_put_in_again_only_when_the_list_differs(section):
    draw = between(section, "function drawCodes(", "\n}\n")
    assert "let item = codeItems.get(code.key);" in draw
    assert ("const same = items.length === codeRow.children.length && "
            "items.every((item, i) => codeRow.children[i] === item);") in draw
    assert "if (!same) codeRow.replaceChildren(...items);" in draw
    assert draw.count("replaceChildren") == 1


def test_the_pills_and_the_card_work_with_a_keyboard_and_a_screen_reader(section):
    item = between(section, "function codeItem(", "\n}\n")
    assert 'button.setAttribute("aria-controls", "codeCard");' in item
    assert 'button.setAttribute("aria-expanded", "false");' in item
    mark = between(section, "function markOpenCode(", "\n}\n")
    assert 'item.firstChild.setAttribute("aria-expanded", open ? "true" : "false");' in mark
    assert "codeText.focus({ preventScroll: true });" in between(section, "function openCodeCard(", "\n}\n")
    closing = between(section, "function closeCodeCard(", "\n}\n")
    assert "item.firstChild.focus({ preventScroll: true });" in closing     # back to its pill ...
    assert "else video.focus({ preventScroll: true });" in closing          # ... or, when it left, the video
    assert "item.isConnected" in closing


def test_the_close_button_and_escape_close_the_card(page, section):
    assert 'codeClose.addEventListener("click", () => closeCodeCard(true));' in section
    topmost = between(page, "function closeTopmost()", 'document.addEventListener("keydown"')
    assert topmost.index("closeVoiceBox(true)") < topmost.index("if (closeCodeCard(true)) return;")
    assert topmost.index("if (closeCodeCard(true)) return;") < topmost.index("leaveFullView()")
    assert "closeCodeCard(false);" in between(section, "function clearCodes(", "\n}\n")   # no focus move


def test_a_click_on_a_code_in_the_video_opens_its_card_and_sends_nothing(page, section):
    handler = between(page, 'video.addEventListener("click"', "});")
    assert handler.index("if (point && openCodeAt(point)) return;") < handler.index("inspect(point, null, video);")
    finding = between(section, "function openCodeAt(", "\n}\n")
    assert "codesInView" in finding and "openCodeCard(best);" in finding
    assert "codesInView = withCodeKeys(" in section                 # from the newest status


def test_the_codes_are_cleared_when_the_program_is_not_running(page, section):
    offline = between(page, "function setOnline(", "\n}\n")
    assert "clearCodes();" in offline
    clear = between(section, "function clearCodes(", "\n}\n")
    assert "closeCodeCard(false);" in clear and "codeRow.hidden = true;" in clear


def test_the_section_has_no_endless_animation(section):
    for word in ("requestAnimationFrame", "setInterval"):
        assert word not in section, word


# ---- the CSS ---------------------------------------------------------------------------


def test_the_css_follows_the_pill_rules_and_comes_before_the_full_view(page):
    assert page.index(".thing-tag {") < page.index(".code-row {") < page.index("html.full-view, html.full-view body")
    for selector in (".code-row", ".code-card", ".code-text", ".code-actions", ".stage.is-full .codes",
                     ".stage.is-full .code-card"):
        rule = css_rule(page, selector)
        assert "infinite" not in rule, selector


def test_in_the_full_view_the_card_is_solid_and_can_be_clicked(page):
    full_card = css_rule(page, ".stage.is-full .code-card")
    assert "backdrop-filter: none" in full_card and "var(--glass-solid)" in full_card
    assert "pointer-events: auto" in full_card
    assert "position: absolute" in css_rule(page, ".stage.is-full .codes")
    assert "backdrop-filter" not in css_rule(page, ".code-row .thing-name::before")
