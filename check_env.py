"""Checks that this laptop is ready to run the car vision program.

    .venv\\Scripts\\python.exe check_env.py
    .venv\\Scripts\\python.exe check_env.py --camera     (also opens the laptop webcam)

Every line starts with OK, NOTE or FAIL. NOTE means "not needed yet".
The script ends with exit code 1 if anything failed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
failures: list[str] = []


def ok(message: str) -> None:
    print(f"OK    {message}")


def note(message: str) -> None:
    print(f"NOTE  {message}")


def fail(message: str) -> None:
    print(f"FAIL  {message}")
    failures.append(message)


def check_python() -> None:
    version = sys.version_info
    text = f"Python {version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) == (3, 11):
        ok(text)
    else:
        fail(f"{text} - this project is built for Python 3.11")
    if sys.prefix == sys.base_prefix:
        fail("Not running inside the virtual environment; use .venv\\Scripts\\python.exe")
    else:
        ok("Running inside the virtual environment")


def check_import(module: str, label: str | None = None, package: str | None = None) -> object | None:
    """Import a module and print its installed version. `package` is the pip name when it differs."""
    try:
        loaded = importlib.import_module(module)
    except Exception as error:                       # a broken install can raise anything
        fail(f"{label or module} does not import: {type(error).__name__}: {error}")
        return None
    try:
        version = importlib.metadata.version(package or module)
    except importlib.metadata.PackageNotFoundError:
        version = ""
    ok(f"{label or module} {version}".rstrip())
    return loaded


def check_torch() -> None:
    torch = check_import("torch", "PyTorch")
    if torch is None:
        return
    if not torch.cuda.is_available():
        fail("PyTorch cannot see the graphics card; object detection would run on the processor")
        return
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    try:
        total = float((torch.ones(8, device="cuda") * 2).sum().item())
    except Exception as error:
        fail(f"PyTorch sees {name} but could not compute on it: {error}")
        return
    if total == 16.0:
        ok(f"PyTorch runs on the graphics card: {name}, {memory:.1f} GB")
    else:
        fail(f"PyTorch computed a wrong result on {name}")


def check_onnxruntime() -> None:
    """Build a tiny model and run it on the graphics card, the way the face models will."""
    onnxruntime = check_import("onnxruntime", "ONNX Runtime", "onnxruntime-gpu")
    if onnxruntime is None:
        return
    try:
        import numpy
        import onnx
        from onnx import TensorProto, helper

        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()               # borrow the CUDA libraries PyTorch ships
        node = helper.make_node("Add", ["x", "x"], ["y"])
        graph = helper.make_graph(
            [node], "double",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        session = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        result = session.run(None, {"x": numpy.array([1.0, 2.0], dtype=numpy.float32)})[0]
        active = session.get_providers()[0]
    except Exception as error:
        fail(f"ONNX Runtime could not run a test model: {type(error).__name__}: {error}")
        return
    if list(result) != [2.0, 4.0]:
        fail("ONNX Runtime computed a wrong result")
    elif active == "CUDAExecutionProvider":
        ok("ONNX Runtime runs on the graphics card")
    else:
        fail(f"ONNX Runtime fell back to {active}; face recognition would run on the processor")


def check_project_files() -> None:
    try:
        from car_vision.config import load_config, resolve_path, section

        config = load_config()
        for name in ("stream", "detector", "recognizer", "assistant", "reactions", "app"):
            section(config, name)
        ok("config.yaml loads and has every section")
    except Exception as error:
        fail(f"config.yaml problem: {error}")
        return

    clip = resolve_path(config["stream"]["file_path"])
    if clip.is_file():
        ok(f"Sample clip found: {clip.name}")
    else:
        note(f"No sample clip yet at {clip.relative_to(ROOT)} (recorded in milestone 1)")

    notes = resolve_path(config["assistant"]["notes_file"])
    if notes.is_file():
        ok(f"Notes file found: {notes.name}")
    else:
        note(f"No notes file yet at {notes.relative_to(ROOT)} (needed for the assistant)")


def check_api_key() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        ok("An API key for the assistant is set (its value is not shown)")
    else:
        note("No API key in .env yet (needed for the assistant; copy .env.example to .env)")


def check_camera() -> None:
    try:
        import cv2

        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        opened, frame = capture.read()
        capture.release()
    except Exception as error:
        fail(f"Laptop webcam error: {error}")
        return
    if opened and frame is not None:
        ok(f"Laptop webcam gives frames of {frame.shape[1]} x {frame.shape[0]}")
    else:
        fail("Laptop webcam did not give a frame (is another program using it?)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that the laptop is ready.")
    parser.add_argument("--camera", action="store_true", help="also open the laptop webcam")
    args = parser.parse_args()

    check_python()
    check_import("numpy", "NumPy")
    check_import("cv2", "OpenCV", "opencv-python")
    check_import("yaml", "PyYAML", "pyyaml")
    check_torch()
    check_import("ultralytics", "Ultralytics")
    check_import("lap", "lapx (tracker matching)", "lapx")
    check_onnxruntime()
    check_import("insightface", "InsightFace")
    check_import("anthropic", "Anthropic SDK")
    check_import("flask", "Flask")
    check_project_files()
    check_api_key()
    if args.camera:
        check_camera()

    print()
    if failures:
        print(f"{len(failures)} problem(s) found. Fix the FAIL lines above.")
        return 1
    print("Everything needed is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
