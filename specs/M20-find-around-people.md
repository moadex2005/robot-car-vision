# M20 spec: a Find that works around people

The owner sat in front of the laptop camera, typed `window` into **Look for** and got "I can't see a window right now", with the window filling the picture behind them. Then they typed their own name, with their face recognised in view, and got "I can't see a Mohamed Adel right now". Find looked broken. It is not broken; three rules make it useless whenever a person is in the picture, which at a demo is always.

Read `CLAUDE.md`, then the parts "Looking for things by a typed name" and "A named find on a person" of the docstring of `car_vision/pipeline.py`, then `merge_named`, `on_a_person`, `search_status`, `find_target` and `_remember` there.

## What was measured (the real model, one live frame, the owner in front of a window)

| Name typed | Score | Its box | What the program did |
| --- | --- | --- | --- |
| `window` | 0.26 | most of the picture, the person inside it | "not in view": below `open_confidence` (0.35), and "on a person" |
| `glasses` | 0.75 | on the face | dropped: on a person |
| `wall` | 0.35 | the whole picture | dropped: on a person |
| `curtain` | 0.70 | beside the person | found |
| the person's own name | none | | given to the object finder, which cannot know people |

## Part A: what "on a person" means from now on (`pipeline.py`)

Today a named find is dropped when its centre lies in a person box or it overlaps one by more than 0.3. That drops the window behind a person and the keys in their hand. The rule exists so that a person is never treated as an object: no picture of them kept in the memory, nothing of them sent when clicked. Keep that promise, drop less:

1. **The find is the person**: its box and an everyday person box overlap by more than `SAME_AS_PERSON = 0.6` (intersection over union). Dropped, as today: the name is only another word for that person.
2. **The find is on a person**: at least `ON_PERSON_SHARE = 0.5` of the find's own box lies inside one person box (glasses, a shirt, keys held in front of the body). It is **kept and shown**: drawn, listed, counted as found. But it is never a thing of its own:
   - a click on it, in the video or on its pill, is a click on that person: `target()` gives the person target (`kind "person"`), so nothing is sent, exactly as today for a click on a person;
   - the memory is not told about it (no sighting, no picture);
   - its entry in `status()["objects"]` has `"on_person": true`.
3. **Anything else** (a window or a wall behind a person, a thing beside them): an ordinary find, as if no person were there.

`on_a_person` is replaced by small pure functions with these names: `same_as_person(find, people)`, `share_inside(box, other)` and `on_person(find, people)`. The `View` gains `on_person: tuple[Detection, ...] = ()`, so `target()` can tell. `state.py` is not changed. The same rules hold for names from auto-discover.

## Part B: "maybe" for the visitor's own names (`detector.py`, `pipeline.py`, `draw.py`)

New setting, already in `config.yaml`: `detector.open_maybe_confidence: 0.20` (more than 0, at most `open_confidence`, else `ConfigError`). `tests/test_pipeline.py` `CONFIG` has it too.

- `OpenDetector` runs the model with `open_maybe_confidence` as its limit and returns everything it gets; it reads the new setting itself.
- The pipeline sorts the finds of each of the visitor's **own** names: those at or above `open_confidence` are **found**, as today. A name with no such find, but with candidates at or above the maybe limit, gets its single most confident candidate as a **maybe**. Never more than one maybe per name. Names from auto-discover never get a maybe: their finds below `open_confidence` are dropped.
- A maybe is shown and can be clicked like any find (the assistant then says what it really is). It is not told to the memory. Parts A's rules apply to it.
- `status()["objects"]` entries of a maybe have `"found": true, "maybe": true`. `status()["search"]["found"]` entries gain `"sure": true | false`; a name with only a maybe is in `found` with `"sure": false`, not in `missing`.
- Drawing: `draw_detections` gains `unsure: Iterable[int] = ()` (track numbers, like `wanted`). A maybe is drawn in `WANTED_COLOR` with the ordinary line thickness, no tint inside its outline, and a `?` after its label in the caption (`window? 0.26 #100003`). Existing calls and tests are unaffected.
- The tracker starts a new track only above its own limit (0.25 with `bytetrack.yaml`), so a candidate between 0.20 and 0.25 may not reach the page until it once scores 0.25. That is accepted; say in your report what you saw in the code about it.

## Part C: a known person's name in Look for (`pipeline.py`, and `app.py` only if needed)

- A typed name **is a person's** when, ignoring case and extra spaces, it equals an enrolled person's display name, their folder name (underscores as spaces), or the first word of their display name. A first name shared by several enrolled people means all of them.
- Such a name is never given to the open detector. Its state comes from the recognised faces: found when that person's face is recognised in the newest fresh frame, with the place of their person box (their face box when there is none).
- `status()["search"]["names"]` and the `found` and `missing` entries show the person's display name as it was enrolled, whatever was typed. A found entry of a person has `"person": true, "sure": true`. With several people of one first name the entry's `count` says how many are in view.
- In the video the found person's box is drawn in `WANTED_COLOR` with `WANTED_THICKNESS`; their face box and name stay as they are.
- Their entry in `status()["objects"]` gets `"found": true`. A click on them is a click on a person, as always.
- People are still only ever recognised by their enrolled face. Words for people (`person`, `man`, `face`, ...) are refused as today. A name of somebody who is not enrolled is an ordinary name, as today. With recognition off, every name is an ordinary name.
- A search made only of people's names never loads the open detector: its state is `ready` at once.
- When people are added or removed from the page while a search runs, the next frame uses the new list.

## Part D: the page (`templates/index.html`: only the finder's code, the In view pills and their CSS)

- Chips: a sure find says `found on the left` as today; a maybe says `maybe, on the left` and looks quieter (class `is-maybe`); a person says `in view, on the left`.
- Pills under In view: a maybe gets a small `maybe` tag (like the `discovered` tag) and the title `Not sure: it may be something else`. A pill with `on_person` has the title `On a person: a click shows the person`.
- The sentences the page writes and speaks (`findSentences` and the ones around it): for a person `I can see Mohamed Adel, on the left.` and `I can't see Mohamed Adel right now. I'll keep looking.`, with no `a` before the name; for a maybe `I think I can see a window, in the middle, but I am not sure.`; mixed results are joined as today. A maybe counts as an answer: the page does not keep waiting for a sure find before it speaks.
- Other work is going on in this file in parallel (the camera box, the voice code, the top bar, the chat's requests, and new CSS at the very end of the style block). Touch only the finder's section (`// ---- find by name` and what follows it up to auto-discover), `readObject`, `thingText`, `thingItem`, `drawObjects`, and the CSS of `.thing` and `.find-chips`.

## Tests (new files; edit existing ones only where an assertion must change)

- `tests/test_find_people.py`: the three cases of Part A with boxes taken from the table above (window behind a person kept, glasses on a face kept and marked, a box that is the person dropped); `share_inside` on touching, nested and apart boxes; a click on an on-person find gives the person target and no photo is sent; the memory is not told; `status()` marks it. Part C with a fake recogniser: full name, folder name, first name, wrong case; two people with one first name; not enrolled; recognition off; people changed while searching; the open detector is never given a person's name and never loaded for people-only searches; the wanted colour on the person's box.
- `tests/test_find_maybe.py`: the settings check; found against maybe against missing; one maybe per name; none for discovered names; `sure` in the status; the memory not told; `draw_detections(unsure=...)` (colour, thin line, no tint, the `?`).
- `tests/test_app_find_words.py`: string checks on the template for the new chip words, tags and sentences, in the style of `tests/test_app_find.py`.
- Existing tests that pin the old rule (a find whose centre is in a person box is dropped) are changed to the new rule in place; list each one in your report.

## Exit test

The whole test suite passes. The orchestrator checks it live with the real model and a person in front of a window: `window`, `glasses`, `curtain` and the person's own name.
