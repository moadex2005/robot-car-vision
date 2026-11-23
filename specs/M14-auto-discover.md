# M14 spec: auto-discover

The everyday detector knows 80 kinds of object and cannot tell that something it has no name for is in view. The find-by-name detector (M13) can find anything, but only when someone types the name. Auto-discover closes the gap with a third, cheap model in the cloud:

1. Every few seconds a small vision-language model **looks at the picture and names what is there**.
2. Those names are given to the find-by-name detector, which **draws live boxes** for them.
3. Claude still writes every answer when the visitor clicks or asks. Nothing about that changes.

The project owner supplied a Hugging Face key for this and wants hosted models, not more local ones. Read `CLAUDE.md`, then `specs/M13-find-and-remember.md`.

## What was measured before this was written

The key reaches Hugging Face's Inference Providers through one OpenAI-style address. Checked with the owner's key on a 512 x 288 frame of the recorded clip:

```
POST https://router.huggingface.co/v1/chat/completions
Authorization: Bearer <HF_TOKEN>
{"model": "google/gemma-3-4b-it:parasail", "max_tokens": 80, "temperature": 0,
 "messages": [{"role": "user", "content": [
     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
     {"type": "text", "text": "<the question>"}]}]}
-> {"choices": [{"message": {"content": "TV, Monitor, Router, Cushion, Curtains, Table, Cables, Pillow"}}],
    "usage": {"prompt_tokens": 312, "completion_tokens": 17}}
```

- `google/gemma-3-4b-it` through the provider `parasail`: 1.2 seconds, 312 tokens in and 17 out. At its listed price (0.04 dollars per million tokens in, 0.05 out) one look costs about 0.000013 dollars; one look every 10 seconds is about half a cent an hour.
- `google/gemma-3-12b-it` through `deepinfra` gave much the same list for more money. A model id with a provider that does not serve it answers HTTP 404 with a JSON error.
- `GET https://router.huggingface.co/v1/models` lists the models with their providers and prices.

## New settings (already in `config.yaml`, added by the orchestrator)

```yaml
lookout:
  model: "google/gemma-3-4b-it:parasail"   # a small hosted model that looks at pictures
  every_seconds: 10
  picture_width: 512            # the picture sent is shrunk to this width
  max_names: 6
  daily_budget_usd: 0.50        # looking stops for the day when the estimated cost reaches this
  input_price_per_million: 0.04   # the model's listed prices, used for the estimate
  output_price_per_million: 0.05
  spend_file: data/lookout.json   # today's count and estimated cost, kept across restarts
```

The key is `HF_TOKEN` in `.env`, beside the Claude key. The same rules hold for it: never printed, logged, committed or put in a test.

## Part A: `car_vision/lookout.py` (new; standard library, plus `dotenv` to read `.env` as `assistant.py` does)

```python
class LookoutError(Exception): ...            # a plain message for the page
class Lookout:
    def __init__(self, settings: dict, fetch=None, clock=None) -> None: ...
    def names(self, jpeg: bytes) -> list[str]: ...
    @property
    def looks_today(self) -> int: ...
    @property
    def spent_today(self) -> float: ...        # estimated dollars
    @property
    def budget(self) -> float: ...
```

- `names` sends one request and returns the object names from the answer, cleaned: split on commas and line breaks; bullets, numbers and quotes removed; lower case; each 1 to 40 characters of letters, digits, spaces, hyphens and apostrophes; no duplicates; at most `max_names`; words for people left out (person, people, man, woman, boy, girl, child, face, hand, and the like, as one named list). An answer with no usable name gives an empty list.
- The question asks for the distinct physical objects in view, most noticeable first, as short names separated by commas and nothing else, and says not to list people. It is a module constant.
- `fetch(url, headers, body_bytes, timeout) -> (status, body_text)` can be replaced in tests; the real one uses `urllib` with a 20 second timeout and never follows a redirect to another host.
- Cost: after each answer the estimate grows by the usage the answer reports times the two prices (when usage is missing, by a fixed 400 tokens in and 30 out). The day's looks and estimate are kept in `spend_file` (written atomically; a new day by the local date starts from zero; an unreadable file starts from zero with one log line). Before sending, `names` raises `LookoutError("Today's budget for auto-discover is used up ...")` when the estimate has reached `daily_budget_usd`.
- Errors, each a `LookoutError` with a plain message and nothing leaked: no `HF_TOKEN` (says to add it to `.env`); HTTP 401 or 403 (the key was refused); 402 (the Hugging Face account has no credit left); 404 (the model or provider in `lookout.model` is not available); 429 and 5xx (busy, try later); a timeout or no network; an answer that is not the expected JSON. The key is never part of a message or a log line, also when the server echoes it.
- `jpeg` must start with the JPEG marker and be at most 200 kB, else `ValueError`.
- Nothing about the picture or the names is logged; counts and errors only.
- Tests with a fake `fetch` and clock (no network, no real key): the request's address, headers and body shape; the cleaning rules on awkward answers (numbered lists, sentences, "a red cup", duplicates, people words, non-English names, 20 names, an empty answer); each error; the key never in an error or the log; the cost estimate with and without usage; the budget stop; the spend file across a restart, a new day, and a broken file; settings validation.

## Part B: `car_vision/pipeline.py`

```python
class Pipeline:
    def discover(self, on: bool) -> None: ...
```

- Off at start-up, always. With it on, one background thread wakes every `lookout.every_seconds`, takes the clean frame shrunk to `picture_width` (no boxes drawn), and asks the `Lookout` for names. It never runs on the loop thread and never holds the pipeline lock across the request. `Lookout` is built on first use and can be handed in for tests (`lookout=None`).
- From the names it keeps a set of **discovered names**: a name stays for 60 seconds after the lookout last mentioned it; names equal to a label the everyday detector knows (its 80 class names, and their plain variants such as `television` for `tv`, kept as one small table) are dropped, because that detector already boxes them.
- The find-by-name detector is given the visitor's own search names first, then discovered names up to 6 in all. `set_names` is called only when that list really changed, and at most once per look. Everything in M13 about loading, the lock, the rounds and the merge applies unchanged.
- Results for discovered names are ordinary named detections, but they are **not** "found": `status()["objects"][i]["found"]` stays for the visitor's own search, and such entries get `"discovered": true` instead; they are drawn like everyday detections, not in the search's yellow. `status()["search"]` reports only the visitor's own names.
- `status()` gains `"discover"`: `{"on": bool, "state": "off" | "looking" | "paused", "names": [...], "note": "<why paused, in plain words>", "looks": <today>, "spent_usd": <today, estimated>, "budget_usd": <setting>}`. A `LookoutError` pauses it with the message as the note: after three failures in a row it waits 60 seconds before the next look; a used-up budget, a refused key and a missing key pause it until it is switched off and on again.
- Turning it off stops the looks at once, forgets the discovered names and gives the find-by-name detector only the visitor's own names again. `use_phone`, a new frame size and `stop()` behave sensibly (the thread ends with the pipeline; no look uses a picture from the camera that was replaced).
- Tests with a fake lookout and the fake open detector: off by default; the thread's rhythm with an injected clock or a short interval; names kept 60 s then dropped; everyday labels dropped; own names first and the cap of 6; `set_names` only on change; `discovered` against `found`; each pause reason; off clears; no look after `stop()`.

## Part C: `car_vision/app.py`, `templates/index.html`

- `POST /api/discover` with `{"on": true | false}`; anything else is a 400.
- The page: a switch **Auto-discover** beside the In view heading, off by default and never remembered between visits. Beside it, small: `looking every 10 s · 14 looks today · about $0.0002` or the pause note. Discovered things appear in the video and in the In view list like any other, with a small mark on their pills that says they were discovered.
- The first time it is switched on in a visit, a short line appears under the switch and stays while it is on: `Auto-discover sends a small picture of the whole view to a Hugging Face model every few seconds. Anyone in view is in that picture.` The privacy line in the footer gains the same sentence.
- It is silent: nothing about discovery is ever spoken.
- All page rules hold (text only via `textContent`, no outside resources, no endless animation, no blur over the video, the full view's inert page).

## What to tell the reader plainly in the README (the orchestrator writes this)

- Three models work together: a fast local detector on every frame, a slower local one that finds things by name, and a small hosted one that suggests the names. Claude answers questions.
- Auto-discover costs about half a cent an hour and stops at a daily limit. It sends pictures of the whole view to a third party, so it is off until switched on.
- The hosted model's names can be wrong, and the local detector then looks for things that are not there.
