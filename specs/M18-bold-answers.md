# M18 spec: bold text in answers

The assistant sometimes writes Markdown, and the page shows it raw: `**Battery:** 5000 mAh`. After this milestone the page shows the bold as bold and never shows the marks. This is a change to `car_vision/templates/index.html` only (and its tests); the assistant and the server are untouched.

Read `CLAUDE.md`, then the section `// ---- links in answers` of `index.html` (`linkParts`, `linkNode`, `answerNodes`) and `captionNodes`, which already shows how to keep links apart while the text around them is worked on.

## What is rendered

`answerNodes(text)` stays the one function that turns an answer's text into nodes, and it is used wherever it is used today (the main chat, the panel's follow-up chat, an object's summary). It now understands a small, fixed part of Markdown. Everything else stays plain text exactly as written.

Inside a line:

| Written | Shown |
| --- | --- |
| `**text**` | **text** in a `<strong>` |
| `*text*` | *text* in an `<em>` |
| `` `text` `` | text in a `<code>` (nothing inside it is rendered further) |
| `***text***` | bold and italic |

- A mark opens only when it is at the start of the text or follows a space, a line break or one of `( [ { " ' : - –`, and the character after it is not a space. It closes only before a space, a line break, the end, or one of `. , ; : ! ? ) ] } " '`, and the character before it is not a space. So `2 * 3 * 4`, `5*4*3`, `a ** b` and a lone `**` stay as written.
- Underscores are never marks: `file_name_here` and `__init__` stay as written.
- Emphasis does not cross a line break. A mark with no partner on its line stays as written.
- Links keep working inside and around emphasis: `**[LinkedIn](https://...)**` is a bold link; `**Profile:** https://...` shows `Profile:` bold and the word `link`, as today's rules for a bare address decide. Every safety rule of the links section stays (http and https only, no user name, `rel`, `title`, the other-site check).
- Bold may hold italic and the other way round; no deeper nesting is needed.

At the start of a line:

| Written | Shown |
| --- | --- |
| `# Title` to `###### Title` | the title in bold, the `#` marks gone (no heading element) |
| `- item`, `* item`, `+ item` | `• item` (the mark becomes a bullet; indentation is kept) |
| `1. item` | as written |

`#3 is the laptop` (no space after `#`) stays as written.

## How

- Add `emphasisParts` (or a name you find plainer) beside `linkParts`, as a pure function from text to a list of plain objects, with no DOM in it, so its rules can be read and checked on their own. `answerNodes` joins the two. Do not change what `linkParts` returns: `earText` and `captionNodes` use it.
- Nodes are made with `make()`, `document.createElement` and `textContent` only. Never `innerHTML`, never a regular expression replace that builds HTML.
- Unfinished or odd input must never throw and never lose a character of the words: for every input, the text content of the nodes equals the input with only the rendered marks removed (and `- ` turned into `• `).
- The specifications list in the details panel (`showInfo`: the `dt` and `dd` of each spec) shows short labels and values: render each value with `answerNodes` too, and strip the marks from each name (a small `plainWords(text)` helper built on the same parts).
- Speech and the caption already drop Markdown signs (`readableLines`). Check that a spoken answer and the caption do not say or show the marks for the inputs in the table above; change them only if they do.
- CSS: `strong` and `em` inside a message inherit the colour; `code` gets a quiet monospace look that fits the glass style in both themes (a faint background, small radius, no border). Put the rules directly after the `.answer-link` rules.
- Update the comment at the top of the links section: it no longer says "this is not a Markdown renderer"; it says which small part is rendered and that everything else stays as written.
- Other changes are going on in this file in parallel. Touch only the links section, `showInfo`'s specs loop, the CSS place named above, and (only if needed) `readableLines`.

## Checking the JavaScript for real

The project has no Node. The test suite checks the page by reading the template's text, which cannot run the rules. So before you report, run them once in a real browser engine yourself: write a scratch HTML file outside the project (your temporary folder) that holds the functions copied from the template and a table of cases, prints `PASS` or `FAIL` per case into the document, and run it with headless Edge:

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless --disable-gpu --dump-dom "file:///<path>/check.html"
```

Cases to cover at least: every row of the two tables; every "stays as written" example; bold around a Markdown link; bold label then a bare address; an unclosed `**`; a code span holding `**`; an answer of several lines with a heading, bullets and a bold word; an empty string; 5000 characters of `*`. Put the table of cases and their results in your report. Do not commit the scratch file. If Edge is not at that path or will not run headless, say so in the report and the orchestrator checks the cases live.

## Tests: `tests/test_app_bold.py` (new)

String checks on the template, in the style of the link tests near the end of `tests/test_app.py`: the new function exists and is used by `answerNodes`; nodes come from `createElement`/`make` and `textContent`; `innerHTML`, `insertAdjacentHTML` and `document.write` appear nowhere in the template; the specs loop uses `answerNodes`; the CSS rules exist. Existing tests that quote the old lines of `answerNodes`, `showInfo` or the old comment are updated in place to the new lines.

## Exit test

The whole test suite passes, and the browser check above passes for every case.
