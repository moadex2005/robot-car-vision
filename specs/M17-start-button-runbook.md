# M17 spec: one start button and a demo runbook

On demo day the owner should not type a command. A double-click on a desktop shortcut starts the site and opens the page, and one printed page says what to show and what to do when something fails.

Read `CLAUDE.md`, `README.md` and `HANDOFF.md`. The page's features are described in `README.md`; four more are being built in parallel and are described in `specs/M15-outlines.md`, `specs/M16-codes.md`, `specs/M18-bold-answers.md` and `specs/M19-camera-power.md`. The runbook may mention those four as the specs describe them. Do not invent anything the README or those specs do not say.

## Part A: `start_demo.cmd` (new, project root)

A Windows command file that:

1. sets the window title to `Robot Car Vision - close this window to stop`;
2. changes into its own folder (`cd /d "%~dp0"`), so it works from a shortcut and from any drive;
3. when `.venv\Scripts\python.exe` is missing, says in one plain line that the environment is not installed and that the README's Install section says how, waits for a key and ends with code 1;
4. runs `.venv\Scripts\python.exe -m car_vision.app --open %*` (extra words such as `--source phone` are passed on);
5. when the program ends with an error code, waits for a key so the message can be read; when it ends normally (Ctrl+C or a closed page), closes.

No path of this laptop is written into it (no `D:\`), no key, no name. ASCII only, CRLF line ends.

## Part B: `car_vision/app.py`, a second start

Today a second start prints "The demo page is already running ..." and ends with code 1. With `--open` it now also opens that address in the browser first, and ends with code 0, so a second double-click on the shortcut simply shows the page. Without `--open` nothing changes. No model is loaded in either case (the check stays before `Pipeline(config)`). Change only `main()` for this.

## Part C: `tools/make_shortcut.ps1` (new) and `tools/robot_car.ico` (new)

```powershell
tools\make_shortcut.ps1 [-Folder <where to put the shortcut>]
```

- Makes `Robot Car Vision.lnk` in `-Folder`, by default the current user's Desktop (`[Environment]::GetFolderPath('Desktop')`). Target: `start_demo.cmd` in the project root, found from the script's own place (`$PSScriptRoot\..`), never a written-out path. Start in: the project root. Icon: `tools\robot_car.ico`. Description: `Starts the Robot Car Vision demo page`.
- Running it again replaces the shortcut. It prints one line saying where the shortcut is. It works in Windows PowerShell 5.1 (no `&&`, no ternary).
- `tools/robot_car.ico`: a simple icon, a round blue camera lens on a dark rounded square, with sizes 16 to 256. Make it once with a short Python script using Pillow (already installed with Ultralytics) and commit only the `.ico`; if Pillow is not installed, leave the icon out, let the shortcut use the default icon and say so in your report.
- You test it only with `-Folder` pointing at a temporary folder. Never write to the real Desktop: the orchestrator runs it there after the merge.

## Part D: `RUNBOOK.md` (new, project root)

One page: at most 70 lines of Markdown, short sentences, plain words (the reader is a student presenting to an audience). Sections, in this order:

1. **Ten minutes before**: laptop on power; close other programs and browser tabs (8 GB of RAM); internet on (the assistant, the natural voices and auto-discover need it); phone charged, IP Webcam installed, phone and laptop on the same network, with the phone's hotspot as the simple way; the phone overheats, so start IP Webcam last.
2. **Start**: double-click **Robot Car Vision** on the desktop; a black window opens and the page follows in about a minute; keep the black window open; closing it stops the demo. Keep one browser tab on the page.
3. **Switch to the phone**: Camera, type the address IP Webcam shows, Connect.
4. **What to show**, a numbered list of at most 9 steps in an order that tells a story, one line each with what to do and what the audience sees: live boxes and track numbers; click an object; ask a follow-up; a known person is greeted and a click shows their line; Look for a thing by name (it is drawn with its outline); a QR code held to the camera; Describe the scene; ask by voice; the Timeline. Mention the camera on and off button where it fits.
5. **If something fails**, a table of symptom and what to do, at most 10 rows: the page does not open; "Program not running"; the video is black or says Reconnecting; the phone does not connect; the video is slow or late (several tabs, large picture size, auto-discover on); the assistant says it is not available (internet, key in `.env`); no voice (browser, the speaker button, Edge for the natural voices); a person is not recognised (light, distance, enrol again); find-by-name finds nothing or the first search is slow; everything is stuck: close the black window and start again.
6. **After the demo**: close the black window; replace the two keys in `.env` (they were once pasted into a chat).

No key, no token, no e-mail address, no person's name, no path of this laptop.

## Tests: `tests/test_start_demo.py` (new)

- `start_demo.cmd`: has the venv Python, `-m car_vision.app`, `--open`, `%*`, `cd /d "%~dp0"`; is ASCII; has no drive-letter path.
- `tools/make_shortcut.ps1`: refers to `start_demo.cmd` and `$PSScriptRoot`, takes `-Folder`, has no drive-letter path.
- `RUNBOOK.md`: exists, at most 70 lines, has the six section headings, and holds nothing that looks like a key (`sk-`, `hf_`), an e-mail address or a drive-letter path.
- `app.main(["--open"])` when something already serves (replace `already_serving` and `webbrowser.open` with fakes): the browser is opened at the page's address, the return code is 0, and `Pipeline` is never built. Without `--open`: no browser, return code 1, as today.

## Exit test

The whole test suite passes. `tools\make_shortcut.ps1 -Folder <temp folder>` makes a shortcut whose target is the `start_demo.cmd` beside it (read the `.lnk` back with `WScript.Shell` to check). You do not start the site: port 5000, the camera and the models belong to the orchestrator.
