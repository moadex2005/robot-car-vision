# M25 spec: watch for

"Tell me when a bottle appears." The visitor names a thing, goes on with something else, and when the thing comes into view the page shows a quiet notice and writes a line in the Timeline. It never speaks by itself.

This builds on M20 (a Find that works around people): read `specs/M20-find-around-people.md` and the code it left in `car_vision/pipeline.py` first (found, maybe, on a person, a known person's name), then `CLAUDE.md`, `specs/M13-find-and-remember.md` and `car_vision/memory.py`.

## Settings (already in `config.yaml`, added by the orchestrator)

```yaml
watch:
  max_names: 4              # watch for at most this many things at once
  again_after_seconds: 60   # the same thing is announced again only after it was away this long
```

## What "appears" means

- A watched name is **in view** on a frame when: it is one of the everyday detector's labels and that detector sees it; or it is a known person's name (M20, Part C) and that person's face is recognised; or the find-by-name detector finds it **for sure** (a maybe never counts; a find on a person does count: keys in a hand have appeared).
- It **appears** when it has been in view for `APPEAR_SECONDS = 1.0` without a break longer than `NAMED_HOLD_SECONDS`, after not being in view. One event per appearance.
- After an event the same name gives no new event until it has been out of view for `watch.again_after_seconds`, however often it flickers.
- When a watch starts while the thing is already in view, there is no event: the answer to the visitor says it is in view now (see the page), and the first event comes when it appears again after being away.
- A camera switch, the camera turned off and on, or a new frame size start the watching afresh without an event: what is in view on the first second of the new picture did not "appear".
- While the camera is off nothing appears. The watch list is kept.

## Part A: `car_vision/pipeline.py`

```python
class Pipeline:
    def watch(self, names: Sequence[str]) -> None: ...      # replaces the list; [] stops watching
```

- Names are checked exactly as `search()` checks them (`clean_names`), with `watch.max_names` as the most; words for people are refused with the same message; a known person's name is allowed and shown as enrolled.
- Everyday labels and people's names cost nothing extra. Any other watched name is given to the find-by-name detector, after the visitor's own search names and before discovered names, six names in all as today (when there is no room, the watch for that name waits and `status()` says so). Their finds are drawn like discovered ones (not in the search's yellow), and are not "found" in `status()["search"]`.
- `status()` gains `"watch"`, directly after the `"search"` line:

  ```python
  {"names": [{"name": "bottle", "state": "watching" | "loading" | "waiting" | "failed", "in_view": false, "last_at": 1795510000.0 | None}],
   "events": [{"id": 7, "name": "bottle", "place": "left", "at": 1795510000.0, "person": false}]}   # the newest 10 of this run, oldest first
  ```

  Event ids go up by one and never repeat in a run. `at` is wall-clock seconds.
- Each event is also told to the memory once (Part B). Nothing about a watch is spoken, and no reaction is made of it.
- Everything is decided on the loop thread from the frame's results; `watch()` from a web thread only swaps the list under the lock and starts a finder round when the finder's names change, as `search()` does. Put the new public method directly after `discover`, new fields in `__init__` directly after `self._discovery = ...`, and the loop's part in a section of its own, `# ---- watching for things (the loop thread) ----`, directly before the codes section.

## Part B: `car_vision/memory.py` and the Timeline

- The sighting log learns one more kind of entry: an event `watch` with the name, the place and the time, kept and deleted by the same rules as sightings (`memory.keep_days`), never with a picture. Add it with the smallest change to the table and the queries that you can find; existing files from before this milestone must still open (add the column or table if it is missing; never drop data).
- `GET /api/memory` lists these entries among the others in time order, marked `"kind": "watch"`. The Timeline shows one as `Bottle appeared, on the left` with its time and a small bell, and its filter field finds it by name.
- The lines the assistant gets about what the camera remembers include it (`10:42 the watched bottle appeared, on the left`), so "when did the bottle show up?" has an answer.
- With the memory off (`memory.file` empty) the notices still work; there is just no Timeline entry.

## Part C: `car_vision/app.py`

`POST /api/watch` with `{"names": ["bottle", "keys"]}` (a `ValueError` from the pipeline is a 400 with its message); `DELETE /api/watch` stops. Place both directly after the search routes.

## Part D: `car_vision/templates/index.html`

- **Starting a watch.** Beside the Look for field's Find button, a second small button with a bell: it watches for the names typed in the field instead of searching for them. And by words, typed in the chat or spoken: `tell me when a bottle appears`, `tell me when you see my keys`, `let me know when Mohamed arrives`, `watch for the remote`, `stop watching`. Recognise these the way `findRequest` recognises find sentences: strictly (the whole sentence, 1 to 4 names, the same little words dropped), so an ordinary question is never taken for a watch. A sentence that is a watch request does not go to the assistant.
- **The answer to starting one** is the page's own sentence, shown where a find's answer is shown and spoken only if the voice is on (it answers what the visitor just did): `I'll tell you when I see a bottle.`; when it is in view already: `A bottle is in view right now, on the left. I'll tell you when it appears again.`; for a person, no `a` before the name.
- **Watched names** are shown as quiet chips with a small bell beside the search chips (in the full view too), each with a button to stop watching that one. A chip says `watching`, `in view`, `getting ready` (loading) or `waiting for room` (more than six names).
- **The notice.** When `status.watch.events` holds an event this page has not shown yet (a larger id than the last one shown; on a fresh page load only events from the last 10 seconds count, so a reload does not replay old ones), a capsule appears among the video's banners: a small bell and `Bottle appeared, on the left`. It fades in once, stays 8 seconds, fades out. Several events queue, one capsule each. It is never spoken, not even with the voice on and not in conversation mode; `aria-live="polite"` lets a screen reader say it. The browser tab's title gets a `(1)` in front while the page is in the background and loses it when the page is looked at.
- The page rules, all of them: text only through `textContent` and `make()`, no outside resource, no endless animation (the bell does not swing for ever), no blur over the video, redraw only what changed, cleared when the program is not running.
- Touch only the finder's sections, the banners, the Timeline's item code, and their CSS.

## Tests (new files; edit existing ones only where an assertion must change)

- `tests/test_pipeline_watch.py`, with the fakes of `tests/test_pipeline.py` and an injected clock: appears after one second in view and not before; a flicker gives one event; again only after `again_after_seconds` away; already in view at the start gives no event and `in_view` true; an everyday label needs no finder; a person's name uses the faces; another name goes to the finder after the own names; a maybe never triggers; a find on a person does; the camera switched, turned off and on, and a new frame size give no event; the six-name limit and `waiting`; ids rise and at most 10 events are kept; the memory is told once per event; nothing is spoken and no reaction is made; `watch([])` clears; wrong names raise.
- `tests/test_memory_watch.py`: the new entry is stored, listed in order, deleted with the others after `keep_days`, cleared by Clear memory, never has a picture, and a memory file made by the code before this milestone still opens and gains what it needs (build such a file in the test from the old table layout).
- `tests/test_app_watch.py`: the two routes, and string checks on the template for the bell button, the chips, the capsule, the sentences, the title counter, and that no call to `say(` or `speak(` is reachable from the event's notice.

## Exit test

The whole test suite passes. The orchestrator checks live on the recorded clip and the laptop camera: a watch for something that comes into view gives one capsule and one Timeline line and no sound.
