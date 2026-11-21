# M6 spec: person briefs and "what is this?"

Two requests from the project owner, added after M5 was specified:

1. Each person is described by a Markdown brief instead of an entry in `people.json`. The owner supplies a folder per person holding their face photos and one `about.md`.
2. When the camera sees something the detector cannot name, the assistant can look at the camera picture and say what it is, using the same hosted API.

The work is split in parts so that each lands in the module that owns it. Read `CLAUDE.md` first. The second request is now specified in `specs/M7-click-to-inspect.md`, which wins where the two files differ.

## The person folder

```
data/people/
  sara/
    about.md
    photo_1.jpg
    photo_2.jpg
```

The folder name is the person id (small letters, digits, underscores). `about.md` is written by hand, so reading it must be forgiving:

```markdown
# Sara Ahmed
Role: Team lead

- Built the car chassis
- Fourth-year student

Sara joined the project in the first week and wrote the motor control code.
```

- The first `# ` heading is the display name. Without one, the folder name is the display name.
- A line starting with `Role:` (any letter case) gives the role. Optional.
- Lines starting with `- ` or `* ` are the card lines; the info card shows the first three. Optional.
- Everything in the file, as written, is what the assistant may know about the person.
- A missing `about.md` is fine: the person is known by their folder name and has no card text.
- Files are read as UTF-8, accepting a byte-order mark; a file in another encoding gives a readable error naming the file.

`data/people.json` is no longer read anywhere.

## Part A: `car_vision/assistant.py`

- New setting `assistant.people_dir` (already in `config.yaml`). At start-up read every `<people_dir>/<id>/about.md`, in sorted id order, and add them to the system prompt under a heading that tells the model these are the people the camera can recognise, each introduced by its id and display name. The system prompt stays a pure function of the notes file and these briefs: nothing per-question goes in. A missing or empty folder adds nothing and is not an error. HTML comments are stripped as for the notes.
- The model is told: when the camera line names a person, use that person's brief to answer questions such as "who is this?"; never guess who someone is from a picture.
- New context line for the user turn: when a `WorldState` was given and it holds detections newer than `seen_within_seconds`, add one sentence listing the distinct object labels the detector currently reports (not `person`), after the people line.
- The chat takes no pictures: `ask(question)` keeps its signature, and the chat's system prompt says nothing about attached pictures (remove that paragraph if it is there). Pictures are handled by `describe`, specified in `specs/M7-click-to-inspect.md`. The helpers `check_image` and `user_turn` stay for it: a picture larger than 1.5 MB, or bytes that do not start with the JPEG marker, raise `ValueError` before anything is sent, and the image block has the shape given in the API reference.
- `main` (the typed chat) is unchanged; it has no camera.
- Tests: briefs reach the system prompt in sorted order and it stays identical between questions; missing folder; a brief in a wrong encoding; the objects line (present with fresh detections, absent with stale ones or with only people in view).

## Part B: `car_vision/faces.py`, `recognizer.py`, `enroll.py`

- `load_people(photos_dir: Path) -> dict[str, Person]` now builds the people from the folders: one `Person` per sub-folder, parsed from `about.md` as described above. `Person` gains `brief: str = ""` (the whole file text). Folder names that are not valid ids are skipped with one warning each.
- `recognizer.people_file` is removed from `config.yaml`; nothing reads `people.json`. `data/people.example.json` is replaced by `data/people.example/` holding one `about.md` with a made-up person and no photos (it sits outside `photos_dir`, so it is never enrolled).
- `enroll` takes the list of people from the folders: every sub-folder of `photos_dir` is a person. A folder with an `about.md` but no photos is reported as not enrolled.
- Tests updated to match, including: no heading, no role, no bullets, more than three bullets, a byte-order mark, a wrong encoding, an invalid folder name.

## Part C

Replaced by `specs/M7-click-to-inspect.md`: instead of a "What is this?" button, the visitor clicks an object on the page.
