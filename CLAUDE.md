# ESP robot car: vision and assistant

A phone on a small robot car streams its camera over WiFi to this Windows laptop. The laptop runs object detection, face recognition with an info card, and an assistant. Nothing is sent to the car.

## Running things

- Always use the project's Python: `.venv\Scripts\python.exe`. Never install globally.
- Tests: `.venv\Scripts\python.exe -m pytest`. Tests marked `hardware` are skipped by default.
- Environment check: `.venv\Scripts\python.exe check_env.py`.
- Modules run as `.venv\Scripts\python.exe -m car_vision.<module>`.
- Shell is PowerShell on Windows 11. The laptop has a GTX 1660 Ti (6 GB) and 8 GB of RAM: load one heavy model at a time in tests.

## Rules that hold everywhere

- `car_vision/state.py` is the only thing shared between the vision side and the assistant side. Its dataclasses and `WorldState` methods are a fixed interface; do not change them without the orchestrator.
- All settings come from `config.yaml` through `car_vision/config.py`. The keys are already defined for every milestone. Read them; do not rename, add or remove keys without the orchestrator.
- Each third-party library stays behind one module: `ultralytics` only in `detector.py`, `insightface` and `onnxruntime` only in `recognizer.py` and `enroll.py`, `anthropic` only in `assistant.py`, `flask` only in `app.py`.
- The API key lives in `.env`. Never print it, log it, commit it or put it in a test.
- Face photos (`data/people/`) and `data/embeddings.npz` are private and git-ignored. Do not copy or print them.
- The phone camera is tested only by the user. Agents use `stream.source: webcam` or `file`. Do not change `stream.source` in `config.yaml`; pass `--source` on the command line instead.
- Unit tests use fakes and must pass without a camera, network or graphics card.
- Code is read by students: plain names, short functions, a docstring that says what a module is for.
