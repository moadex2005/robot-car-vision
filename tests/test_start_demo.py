"""The start button (start_demo.cmd, tools/make_shortcut.ps1), the demo runbook, and a second start.

Nothing here starts the site, a camera or a model. The script tests run copies of
the scripts with cmd.exe and powershell.exe in a temporary folder, against a stub
car_vision package; they are skipped where those programs do not exist.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import car_vision.app as app_module
from car_vision.app import STILL_STARTING, main, take_start_lock
from car_vision.config import ConfigError

ROOT = Path(__file__).resolve().parent.parent
START = ROOT / "start_demo.cmd"
SHORTCUT = ROOT / "tools" / "make_shortcut.ps1"
ICON = ROOT / "tools" / "robot_car.ico"
RUNBOOK = ROOT / "RUNBOOK.md"

DRIVE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")        # C:\... or D:/..., not the "p:/" in http://
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
RUNBOOK_SECTIONS = ["## Ten minutes before", "## Start", "## Switch to the phone",
                    "## What to show", "## If something fails", "## After the demo"]
MAX_RUNBOOK_LINES = 70
ICON_SIZES = {16, 24, 32, 48, 64, 128, 256}
RUN_SECONDS = 60                                   # no script run may hang a test longer than this
WAITED = "Press any key"                           # what cmd's pause prints
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe, powershell.exe and msvcrt are Windows only")


def text_of(path: Path) -> str:
    return path.read_bytes().decode("ascii")       # raises when a file is not plain ASCII


# ---- start_demo.cmd as text ------------------------------------------------------------


def test_start_demo_runs_the_venv_python_with_open_and_passes_extra_words_on():
    text = text_of(START)
    assert r".venv\Scripts\python.exe -m car_vision.app --open %*" in text


def test_start_demo_keeps_its_variables_and_changes_into_its_own_folder_first():
    text = text_of(START)
    assert text.splitlines()[:2] == ["@echo off", "setlocal"]
    assert 'cd /d "%~dp0"' in text
    assert text.index('cd /d "%~dp0"') < text.index("-m car_vision.app")


def test_start_demo_sets_the_window_title():
    assert "title Robot Car Vision - close this window to stop" in text_of(START)


def test_start_demo_is_ascii_with_no_path_of_this_laptop():
    text = text_of(START)
    assert DRIVE_PATH.search(text) is None


# ---- start_demo.cmd for real, in a folder with awkward characters -----------------------


def venv_launcher() -> Path | None:
    """The interpreter launcher of the venv running the tests (small), or None outside a venv."""
    config = Path(sys.prefix) / "pyvenv.cfg"
    return Path(sys.executable) if sys.prefix != sys.base_prefix and config.is_file() else None


@pytest.fixture
def demo_copy(tmp_path):
    """A copy of start_demo.cmd in "demo & copy (1)", with a stub car_vision.app and no .venv yet.

    The stub prints its arguments and ends with the number given as its last word.
    """
    folder = tmp_path / "demo & copy (1)"
    (folder / "car_vision").mkdir(parents=True)
    shutil.copy(START, folder)
    (folder / "car_vision" / "__init__.py").write_text("", encoding="ascii")
    (folder / "car_vision" / "app.py").write_text(
        "import sys\nprint('stub app:', *sys.argv[1:])\nsys.exit(int(sys.argv[-1]))\n", encoding="ascii")
    return folder


def add_venv(folder: Path) -> None:
    launcher = venv_launcher()
    if launcher is None:
        pytest.skip("the tests do not run in a venv, so there is no small launcher to copy")
    (folder / ".venv" / "Scripts").mkdir(parents=True)
    shutil.copy(Path(sys.prefix) / "pyvenv.cfg", folder / ".venv")
    shutil.copy(launcher, folder / ".venv" / "Scripts" / "python.exe")


def run_start(folder: Path, *words: str) -> subprocess.CompletedProcess:
    """cmd /s /c ""<folder>\\start_demo.cmd" words", from another folder, with an empty stdin for pause."""
    line = f'""{folder / "start_demo.cmd"}" {" ".join(words)}"'
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return subprocess.run(f"cmd.exe /d /s /c {line}", stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, errors="replace", timeout=RUN_SECONDS, cwd=folder.parent, env=env)


@WINDOWS_ONLY
def test_start_demo_closes_without_waiting_when_the_program_ends_normally(demo_copy):
    add_venv(demo_copy)
    run = run_start(demo_copy, "--source", "phone", "0")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "stub app: --open --source phone 0" in run.stdout
    assert WAITED not in run.stdout


@WINDOWS_ONLY
def test_start_demo_passes_an_error_code_on_and_waits_for_a_key(demo_copy):
    add_venv(demo_copy)
    run = run_start(demo_copy, "3")
    assert run.returncode == 3, run.stdout + run.stderr
    assert "stopped with an error" in run.stdout and WAITED in run.stdout


@WINDOWS_ONLY
def test_start_demo_without_the_environment_says_so_and_ends_with_1(demo_copy):
    run = run_start(demo_copy, "0")
    assert run.returncode == 1, run.stdout + run.stderr
    assert "not installed" in run.stdout and "README" in run.stdout and WAITED in run.stdout
    assert "stub app" not in run.stdout


# ---- tools/make_shortcut.ps1 and the icon ---------------------------------------------------


def test_make_shortcut_finds_start_demo_from_its_own_place_with_no_path_of_this_laptop():
    text = text_of(SHORTCUT)
    assert "start_demo.cmd" in text and "$PSScriptRoot" in text and "robot_car.ico" in text
    assert "[Environment]::GetFolderPath('Desktop')" in text and "$Folder" in text
    assert DRIVE_PATH.search(text) is None
    assert "&&" not in text and "??" not in text       # Windows PowerShell 5.1 has neither


@WINDOWS_ONLY
def test_make_shortcut_makes_a_shortcut_to_start_demo_in_the_folder(tmp_path):
    powershell = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass"]
    for _ in range(2):                                 # the second run replaces the shortcut
        made = subprocess.run([*powershell, "-File", str(SHORTCUT), "-Folder", str(tmp_path)],
                              capture_output=True, text=True, errors="replace", timeout=RUN_SECONDS)
        assert made.returncode == 0, made.stdout + made.stderr
    link = tmp_path / "Robot Car Vision.lnk"
    read = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:SHORTCUT_TO_READ); "
            "$s.TargetPath; $s.WorkingDirectory; $s.IconLocation; $s.Description")
    back = subprocess.run([*powershell, "-Command", read], capture_output=True, text=True, errors="replace",
                          timeout=RUN_SECONDS, env={**os.environ, "SHORTCUT_TO_READ": str(link)})
    target, start_in, icon, description = back.stdout.splitlines()[:4]
    assert os.path.normcase(target) == os.path.normcase(str(START))
    assert os.path.normcase(start_in) == os.path.normcase(str(ROOT))
    assert os.path.normcase(icon) == os.path.normcase(f"{ICON},0")
    assert description == "Starts the Robot Car Vision demo page"
    assert [path.name for path in tmp_path.iterdir()] == ["Robot Car Vision.lnk"]


def test_the_shortcut_icon_holds_every_size_from_16_to_256():
    data = ICON.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind) == (0, 1)                  # an icon file, not a cursor
    sizes = set()
    for index in range(count):
        width, height = data[6 + 16 * index], data[7 + 16 * index]   # 0 means 256
        assert width == height
        sizes.add(width or 256)
    assert sizes == ICON_SIZES


# ---- RUNBOOK.md ------------------------------------------------------------------------


def test_runbook_fits_on_one_page():
    lines = RUNBOOK.read_text(encoding="utf-8").splitlines()
    assert 0 < len(lines) <= MAX_RUNBOOK_LINES


def test_runbook_has_the_six_sections_in_order():
    lines = RUNBOOK.read_text(encoding="utf-8").splitlines()
    headings = [line.strip() for line in lines if line.startswith("## ")]
    assert headings == RUNBOOK_SECTIONS


def test_runbook_holds_no_key_address_or_path():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "sk-" not in text and "hf_" not in text
    assert EMAIL.search(text) is None
    assert DRIVE_PATH.search(text) is None


def test_runbook_lists_at_most_nine_steps_and_ten_problems():
    text = RUNBOOK.read_text(encoding="utf-8")
    show = text.split("## What to show", 1)[1].split("## ", 1)[0]
    steps = [line for line in show.splitlines() if re.match(r"\d+\. ", line)]
    assert 0 < len(steps) <= 9
    fails = text.split("## If something fails", 1)[1].split("## ", 1)[0]
    rows = [line for line in fails.splitlines() if line.startswith("|")]
    assert 0 < len(rows) - 2 <= 10                        # minus the header and its rule


# ---- the start lock --------------------------------------------------------------------


@WINDOWS_ONLY
def test_only_one_copy_holds_the_start_lock_and_it_is_free_again_when_let_go(tmp_path):
    path = tmp_path / "logs" / "car_vision.lock"         # the folder is made
    first = take_start_lock(path)
    try:
        assert first is not None
        assert take_start_lock(path) is None
    finally:
        first.close()
    again = take_start_lock(path)
    assert again is not None
    again.close()


# ---- a second start: the lock and the port ------------------------------------------------


CONFIG = {
    "stream": {"source": "webcam", "file_path": "data/samples/car_clip.mp4"},
    "assistant": {"notes_file": "data/does-not-exist/knowledge.md"},
    "app": {"host": "127.0.0.1", "port": 5000, "jpeg_quality": 80, "log_file": "logs/car_vision.log"},
}
URL = "http://127.0.0.1:5000/"


class StopAtPipeline:
    """Stands in for Pipeline: counts how often it is built, then stops main() with a plain error."""

    built = 0

    def __init__(self, *args, **kwargs) -> None:
        StopAtPipeline.built += 1
        raise ConfigError("the test stops here, before any model")


@pytest.fixture
def start(monkeypatch, tmp_path):
    """main() with the lock and the port replaced: set start["lock"] and start["serving"] first."""
    state = {"lock": object(), "serving": False, "opened": []}
    StopAtPipeline.built = 0
    monkeypatch.setattr(app_module, "load_config", lambda: CONFIG)
    monkeypatch.setattr(app_module, "take_start_lock", lambda path: state["lock"])
    monkeypatch.setattr(app_module, "already_serving", lambda host, port: state["serving"])
    monkeypatch.setattr(app_module, "Pipeline", StopAtPipeline)
    monkeypatch.setattr(app_module.webbrowser, "open", lambda url: state["opened"].append(url))
    monkeypatch.setattr(app_module, "setup_logging", lambda log_file: None)          # no log file is made
    monkeypatch.setattr(app_module, "CAMERA_FILE", str(tmp_path / "camera.json"))    # never the real one
    return state


@pytest.mark.parametrize("argv", [[], ["--open"]])
def test_with_the_lock_and_a_free_port_the_program_starts_as_today(start, argv):
    assert main(argv) == 1                                # the stand-in Pipeline stops it
    assert StopAtPipeline.built == 1
    assert start["opened"] == []


@pytest.mark.parametrize("argv", [[], ["--open"]])
def test_with_the_lock_but_another_program_on_the_port_nothing_starts_or_opens(start, argv, capsys):
    start["serving"] = True
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert "already running" in out and "or another program uses port 5000" in out
    assert start["opened"] == [] and StopAtPipeline.built == 0


def test_a_second_start_with_open_shows_the_running_page_and_ends_with_0(start, capsys):
    start["lock"], start["serving"] = None, True
    assert main(["--open", "--source", "phone"]) == 0
    assert start["opened"] == [URL]
    assert "already running" in capsys.readouterr().out
    assert StopAtPipeline.built == 0


def test_a_second_start_without_open_only_says_so_and_ends_with_1(start, capsys):
    start["lock"], start["serving"] = None, True
    assert main([]) == 1
    out = capsys.readouterr().out
    assert "already running" in out and URL in out
    assert start["opened"] == [] and StopAtPipeline.built == 0


@pytest.mark.parametrize("argv", [[], ["--open"]])
def test_a_second_start_while_the_first_still_loads_asks_to_wait(start, argv, capsys):
    start["lock"], start["serving"] = None, False
    assert main(argv) == 1
    assert STILL_STARTING in capsys.readouterr().out
    assert start["opened"] == [] and StopAtPipeline.built == 0
