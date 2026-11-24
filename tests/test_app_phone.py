"""M22: the page on a phone.

The layout itself is checked by eye in a real browser; these tests check the template's text:
the viewport meta, the one phone block at the end of the style sheet and what it must hold,
and that nothing the page had before is gone (no id, no CSS rule above the block)."""

import re
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "car_vision" / "templates" / "index.html"
PHONE_HEADING = "/* ---- phone: up to 640 px wide ----"
UPRIGHT = "@media (max-width: 640px)"
ON_ITS_SIDE = "@media (max-height: 450px) and (orientation: landscape)"

# Every id in the template at the base commit of M22 (e978e1c). The script and the tests find
# elements by id, so none of them may go.
BASE_IDS = """
addPerson ask askForm askMic brushButton brushCanvas brushClear brushTools cameraAddress cameraBox
cameraButton cameraConnect cameraForm cameraHelp cameraOff cameraOn cameraPower cameraResult caption
captionAnswer captionQuestion chat clear closeDetails codeCard codeClose codeCopy codeCopyWord
codeKind codeNote codeOpen codeRow codeText codes conversationRow conversationSwitch delay
delayClock delaySwitch detailContent detailMeta detailPhoto detailRole detailTitle details
detailsBody detailsGrip detailsScroll device discoverPrivacy discoverRow discoverState
discoverSwitch faces findChips finder finderFind finderForm finderStop finderText followDone
followForm followMessages followMic followPart followQuestion followSend followUp fps frame
fullButton greetings heroText hint liveText messages nav notes nothing offline people peopleEmpty
peopleLink peopleList peopleNote peopleResult personCancel personForm personFormTitle personLines
personLinesHelp personName personResult personRole personSave personStory photoBox photoCancel
photoCapsule photoCount photoHint photoNotes photoProgress photoResult photoWho pictureChoice
pictureHelp pictureHint pictureNow pictureRow pictureShown powerProblem privacy question sceneButton
send source stageMic status themeButton things timeline timelineBox timelineClear timelineClearNo
timelineClearYes timelineConfirm timelineEmpty timelineFilter timelineList timelineResult top video
viewMeta voiceBox voiceButton voiceChoice voiceHelp voiceSwitch voiceTest waiting watch
""".split()

# The selector of every CSS rule (the text before its "{") in the style sheet at the base
# commit. The phone block only overrides them, so each must still be there above the block.
BASE_SELECTORS = [line for line in """
:root, :root[data-theme="light"]
:root[data-theme="dark"]
:root
.theme-fade, .theme-fade *, .theme-fade *::before, .theme-fade *::after
*
[hidden]
html
body
a
:focus-visible
.visually-hidden
.glass
@supports ((-webkit-backdrop-filter: blur(1px)) or (backdrop-filter: blur(1px)))
.backdrop
.blob
.blob.one
.blob.two
.blob.three
@supports (animation-timeline: scroll())
@keyframes drift-one
@keyframes drift-two
@keyframes drift-three
.nav
.nav.is-compact
.brand
.nav-links
.nav-links a
.nav-links a:hover
.status-pill
.dot
.dot::after
.is-live .dot
.is-live .dot::after
.is-down .dot
.is-offline .dot, .is-stopped .dot
.is-ended .dot
@keyframes ping
.status-state
.is-stopped .status-state
.status-item
.status-item::before
.fps
.is-offline .status-item, .is-stopped .status-item
.nav.over-video
.pill
.pill:hover
.pill:active
.pill.primary
.pill.secondary
.pill:disabled
.pill.small
.theme-button svg
.theme-button .moon
:root[data-theme="dark"] .theme-button .sun
:root[data-theme="dark"] .theme-button .moon
.nav .round
.voice-button svg
.voice-button .voice-waves
.voice-button.is-on
.voice-button.is-on .voice-waves
.voice-button.is-on .voice-off
.voice-button.is-speaking .voice-waves
@keyframes speak-pulse
.glass.voice-box
.voice-label
.voice-box .voice-line select
.mic-button
.mic-button svg
.ask-form .mic-button
.mic-button[aria-pressed="true"], .frame-button[aria-pressed="true"]
.voice-privacy
.voice-box .voice-conversation
.mic-button[data-talk], .frame-button[data-talk]
.mic-button[data-talk="listening"], .frame-button[data-talk="listening"]
.mic-button[data-talk="thinking"], .frame-button[data-talk="thinking"]
.mic-button[data-talk="speaking"], .frame-button[data-talk="speaking"]
.glass.camera-box
@keyframes box-in
.field-label
.camera-row
.camera-help
.camera-result
.camera-result.problem
.camera-box hr
.picture-row
.picture-line
.picture-line select
.picture-line select:focus-visible
.picture-now
.switch
.switch input
.switch-track
.switch-track::after
.switch input:checked + .switch-track
.switch input:checked + .switch-track::after
.switch input:focus-visible + .switch-track
main
.section
.section.wide
.section-head
.section-head h2
.section-head p
.hero
.hero-title
.hero-line
.hero-actions
.js .reveal
.js .reveal.is-in
.stage
.shot
.frame
.frame img
.frame img.is-clickable
.frame img:focus-visible
.waiting, .offline
.waiting
.offline
.offline small
.banners
.capsule
.greetings
.greeting
@keyframes capsule-in
.frame-tools
.frame-button
.frame-button:hover
.frame-button:active
.frame-button svg
.frame-button .shrink
.frame-button.power-button
.frame-button.power-button[aria-pressed="true"]
.frame-button.power-button[aria-pressed="false"]
.power-button .power-strike
.power-button[aria-pressed="false"] .power-strike
.frame-button.power-button.is-busy
.capsule.power-problem
.is-off .dot
.camera-off
.capsule.camera-off-capsule
.frame.is-camera-off .waiting
.stage:not(.is-full) .stage-mic
.frame-button.scene-button
.scene-button .dots
@container (max-width: 700px)
.scene-label
.scene-button.is-working svg
.scene-button:disabled
.photo-capsule
.caption
.stage:not(.is-full) .caption
.caption p
.caption-question
.caption-answer
.caption-answer.problem
.finder
.finder > *
.stage.is-full .finder
.finder-form
.finder-form input
.finder-form input::placeholder
.finder-form input:focus-visible
.finder-button
.finder-button:hover
.finder-button:active
.finder-button.stop
.finder-button:focus-visible
.find-chips
.find-chip
.find-chip-name, .find-chip-state
.find-chip-name
.find-chip-state
.find-chip.is-found
.find-chip.is-found .find-chip-state
.find-chip.is-found::before
.find-chip.is-failed .find-chip-state
@container (max-width: 520px)
.is-full .frame-button .expand
.is-full .frame-button .shrink
.delay
.delay-clock
.delay-help
.view-meta
.notes
.note::before
.stopped
.inview
.inview h3
.inview-head
.discover-switch
.discover-state
.discover-state strong
.discover-state.problem
.discover-privacy
.things
.things li
@keyframes pill-in
.thing
.thing.is-gone
.thing:hover
.thing:active
.thing-name
.thing-detail
.thing.is-known .thing-detail
.thing.is-known::before
.thing.is-open
.thing.is-found
.thing.is-found .thing-name::before
.thing-tag
.code-row
.code-row .thing-name::before
.code-card
.code-card-top
.code-text
.code-text:focus-visible
.code-note
.code-actions
.stage.is-full .codes
.stage.is-full .code-row
.stage.is-full .code-row > li:first-child
.stage.is-full .code-row > li:last-child
.stage.is-full .code-card
.quiet
.privacy
.details
.details-scroll
@keyframes sheet-in
@media (min-width: 1280px)
.stage.has-details
@media (max-width: 1279.98px)
.details-grip
.details-grip::before
.details-grip:hover::before, .details-grip:focus-visible::before, .is-resizing .details-grip::before
.details-grip:focus-visible
html.is-resizing, html.is-resizing *
html.full-view, html.full-view body
.stage.is-full
.stage::backdrop
.stage.is-full.has-details
.stage.is-full .shot
.stage.is-full .frame
.stage.is-full .frame img:focus-visible
.stage.is-full #nothing, .stage.is-full .delay-help
.stage.is-full .inview
.stage.is-full .things
.stage.is-full .things > li:first-child
.stage.is-full .things > li:last-child
.stage.is-full .thing
.stage.is-full .delay
.stage.is-full .delay-clock
.stage.is-full .details
.details-top
.details-label
.round
.round:hover
.round:active
.round svg
.photo
.photo-box
.brush-canvas
.photo-box.is-brushing, .photo-box.is-brushing .photo
.brush-tools
.brush-tools svg
.brush-button[aria-pressed="true"]
.follow-form .part-button
.tag
.details-title
.details-meta
.details-role
.details-content
.details-content p
.details-content ul
.details-content li + li
.details-content .intro
.details-content .ask-more
.looking
.problem
.specs
.specs dt, .specs dd
.specs dt
.follow
.follow-messages
.follow-messages:empty
.follow .msg
.follow-form
.follow-form .pill
.chat
.chat-top
.chat-title
.messages
.hint
.msg
@keyframes msg-in
.msg .text
.msg.question
.msg.answer
.msg.error
.msg.thinking
.dots
.dots i
.dots i:nth-child(2)
.dots i:nth-child(3)
@keyframes blink
.web
.web-label
.source
.source:hover
.more
.answer-link
.answer-link:hover
.answer-link:focus-visible
.caption .answer-link
.caption .answer-link:focus-visible
.msg strong, .msg em, .details-content strong, .details-content em
.msg code, .details-content code
.ask-form
.ask-form input, .camera-row input, .follow-form input
.camera-row input, .follow-form input
.ask-form input::placeholder, .camera-row input::placeholder, .follow-form input::placeholder
.ask-form input:focus-visible, .camera-row input:focus-visible, .follow-form input:focus-visible
.follow-form input:disabled
.follow-form input
.people
.people > *
.people > #addPerson
.people-list
.person-card
.person-card.is-editing
.person-card .person-form
.person-name
.person-role
.person-meta
.person-tools, .confirm-buttons
.confirm-row
.confirm-row p
.form-result
.form-result.problem
.form-result .warn-text
.person-form, .photo-progress
.person-form
.form-title
.person-form .field-label
.person-form input, .person-form textarea
.person-form textarea
.person-form input:focus-visible, .person-form textarea:focus-visible
.person-form .camera-help
.form-row
.form-row .form-result
.photo-progress
.photo-progress p
.photo-who
.photo-hint
.photo-count
.photo-notes
.photo-notes:empty
.timeline
.timeline-tools
.timeline-tools input
.timeline-tools input::placeholder
.timeline-tools input:focus-visible
.timeline .confirm-row
.timeline > .quiet
.timeline-list
.timeline-list:empty
.moment
.moment-picture, .moment-mark
.moment-picture
.moment-mark
.moment-name
.moment-detail
.footer
.footer p
@media (max-width: 1179.98px)
@media (max-width: 760px)
.status-item.wide
@media (prefers-reduced-motion: reduce)
*, *::before, *::after
.hero-text
""".splitlines() if line]


def page():
    return TEMPLATE.read_text(encoding="utf-8")


def style_sheet():
    text = page()
    return text[text.index("<style>") + len("<style>"):text.index("</style>")]


def phone_block():
    sheet = style_sheet()
    return sheet[sheet.index(PHONE_HEADING):]


def rule_body(css, selector):
    """The body of the first rule in css whose selector is exactly this one."""
    match = re.search(r"^\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", css, re.MULTILINE)
    assert match, selector
    return match.group(1)


def selectors(css):
    return {line.split("{")[0].strip() for line in css.splitlines() if "{" in line}


def test_the_viewport_meta_covers_the_notch():
    head = page()[:page().index("</head>")]
    metas = re.findall(r'<meta name="viewport" content="([^"]*)">', head)
    assert len(metas) == 1
    parts = [part.strip() for part in metas[0].split(",")]
    assert parts == ["width=device-width", "initial-scale=1", "viewport-fit=cover"]


def test_the_phone_block_is_the_last_thing_in_the_style_sheet():
    sheet = style_sheet()
    assert sheet.count(PHONE_HEADING) == 1
    block = phone_block()
    assert UPRIGHT + " {" in block and ON_ITS_SIDE + " {" in block
    # Nothing after it: no section heading of the sheet comes later, and the block ends the sheet.
    later_headings = [heading for heading in re.findall(r"/\* ---- [^\n]*", block) if heading != PHONE_HEADING]
    assert later_headings == []
    assert sheet.rstrip().endswith("}")


def test_the_phone_block_holds_both_ways_of_holding_the_phone():
    block = phone_block()
    shared = block.index("@media (max-width: 640px), (max-height: 450px) and (orientation: landscape) {")
    assert shared < block.index(UPRIGHT + " {") < block.index(ON_ITS_SIDE + " {")   # shared, upright, on its side


def test_fields_are_at_least_16_pixels_so_iphone_safari_does_not_zoom():
    body = rule_body(phone_block(), "input, select, textarea")
    assert "font-size: max(16px, 1rem) !important" in body


def test_the_section_links_are_hidden_and_the_bar_keeps_its_buttons():
    block = phone_block()
    upright = block[block.index(UPRIGHT + " {"):]
    assert "display: none" in rule_body(upright, ".brand, .nav-links")
    assert "display: none" in rule_body(upright, ".status-item")     # source, fps and device
    assert "width: 44px; height: 44px" in rule_body(block, ".round")  # the voice and theme buttons
    assert "min-height: 44px" in rule_body(block, ".pill")            # the Camera button and every pill


def test_the_bar_never_wraps_and_the_status_gives_way_before_any_button():
    block = phone_block()
    upright = block[block.index(UPRIGHT + " {"):]
    assert "flex-wrap: nowrap" in rule_body(upright, ".nav")
    assert "flex: none" in rule_body(upright, ".nav > *")             # no button shrinks
    pill = rule_body(upright, ".nav > .status-pill")
    assert "flex: 0 1 auto" in pill and "min-width: 0" in pill and "overflow: hidden" in pill
    state = rule_body(upright, ".status-state, .is-stopped .status-state")
    assert "text-overflow: ellipsis" in state and "white-space: nowrap" in state and "min-width: 0" in state
    assert "liveText.title = text" in page()                            # the whole state word is in its title


def test_the_details_panel_is_a_bottom_sheet_not_a_side_column():
    block = phone_block()
    upright = block[block.index(UPRIGHT + " {"):block.index(ON_ITS_SIDE + " {")]
    sheet = rule_body(upright, ".details")
    for part in ("position: fixed", "top: auto", "left: 0; right: 0; bottom: 0", "width: auto", "max-width: none",
                 "max-height: 62svh", "background: var(--surface)", "backdrop-filter: none", "animation: phone-sheet-in"):
        assert part in sheet, part
    assert "env(safe-area-inset-bottom)" in rule_body(upright, ".details-scroll")
    assert "padding-bottom: 62svh" in rule_body(upright, "body:has(#details:not([hidden]))")   # room under the page
    assert "display: none" in rule_body(upright, ".details-grip")
    # In the full view too: one column, and the panel a sheet over the lower half.
    assert "grid-template-columns: minmax(0, 1fr)" in rule_body(upright, ".stage.is-full.has-details")
    full = rule_body(upright, ".stage.is-full .details")
    assert "position: absolute" in full and "top: 50%" in full and "max-width: 100%" in full


def test_the_safe_areas_of_a_notch_are_respected():
    block = phone_block()
    for side in ("top", "bottom", "left", "right"):
        assert "env(safe-area-inset-" + side + ")" in block, side
    assert "env(safe-area-inset-bottom)" in rule_body(block, ".stage.is-full .inview")


def test_the_finder_sits_under_the_video_and_the_boxes_scroll():
    block = phone_block()
    finder = rule_body(block, ".stage:not(.is-full) .finder")
    assert "top: calc(100% + 12px)" in finder and "bottom: auto" in finder
    boxes = rule_body(block, ".glass.camera-box, .glass.voice-box")
    assert "overflow-y: auto" in boxes and "max-height:" in boxes


def test_the_phone_block_adds_no_endless_animation_and_no_blur_over_the_video():
    block = phone_block()
    assert "infinite" not in block
    upright = block[block.index(UPRIGHT + " {"):]
    assert "backdrop-filter" not in rule_body(upright, ".stage.is-full .details")   # the sheet over the video
    assert "background: var(--surface)" in rule_body(upright, ".stage.is-full .details")


def test_no_id_that_existed_before_is_gone():
    text = page()
    missing = [name for name in BASE_IDS if 'id="' + name + '"' not in text]
    assert missing == []


def test_the_full_view_column_is_not_shifted_by_the_sheet_offsets():
    """The one change above the phone block: in the full view the details column kept the top
    and right offsets of the sheet over the page, so position: relative moved it about 90 px
    down (and 16 px left below 1280 px) and its follow-up field off the screen."""
    sheet = style_sheet()
    column = rule_body(sheet[:sheet.index(PHONE_HEADING)], ".stage.is-full .details")
    assert "position: relative; inset: auto; width: auto; max-width: none; height: 100%; max-height: none;" in column


def test_no_rule_above_the_phone_block_was_removed():
    sheet = style_sheet()
    above = selectors(sheet[:sheet.index(PHONE_HEADING)])
    missing = [selector for selector in BASE_SELECTORS if selector not in above]
    assert missing == []
