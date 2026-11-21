# ESP robot car: vision and assistant

A phone on the car streams its camera to this laptop. The laptop labels objects, recognises enrolled people and shows their info card, and runs an assistant that answers from your notes. Nothing is sent to the car.

## Install (Windows 11)

Already done on this laptop. To repeat it on another one:

1. Install Python 3.11 (64-bit) and Git.
2. Open PowerShell in the project folder and create the environment:

   ```powershell
   py -3.11 -m venv .venv
   ```

3. Install PyTorch with graphics-card support, then everything else:

   ```powershell
   .venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
   ```

   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Repair ONNX Runtime. InsightFace installs a processor-only build over the graphics-card one, so remove both and put the right one back:

   ```powershell
   .venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu
   ```

   ```powershell
   .venv\Scripts\python.exe -m pip install --no-deps onnxruntime-gpu==1.32.0
   ```

5. Check the laptop is ready:

   ```powershell
   .venv\Scripts\python.exe check_env.py
   ```

   Every line should start with `OK` or `NOTE`. A `FAIL` line says what to fix.

## Settings

Everything you can change is in `config.yaml`, with a comment beside each value. The one you will change most is `stream.source`: `webcam` (the laptop's camera), `phone`, or `file` (a recorded clip).

## Connect the phone

1. Install the **IP Webcam** app on the Android phone.
2. Put the phone and the laptop on the same network. The simplest way is to turn on the phone's own hotspot and join it from the laptop.
3. In IP Webcam, set the video resolution to 640 x 480, scroll down and tap **Start server**. The app shows an address such as `http://192.168.43.1:8080`.
4. Put that address plus `/video` into `stream.phone_url` in `config.yaml`.
5. Test it:

   ```powershell
   .venv\Scripts\python.exe -m car_vision.stream --source phone
   ```

   Press `c` to show a millisecond clock. Point the phone at the clock: the clock shows now, the video of the clock shows then, and the difference is the delay. Press `q` to quit.

## What each part does

| Command | What you see |
| --- | --- |
| `.venv\Scripts\python.exe -m car_vision.stream` | The plain camera view with its FPS |
| `.venv\Scripts\python.exe -m car_vision.detector` | Objects labelled with boxes and track numbers |
| `.venv\Scripts\python.exe -m car_vision.enroll` | Turns the photos in `data/people/` into saved face data |
| `.venv\Scripts\python.exe -m car_vision.recognizer` | Names and info cards on the people it knows |
| `.venv\Scripts\python.exe -m car_vision.assistant` | A typed chat with the assistant |
| `.venv\Scripts\python.exe -m car_vision.app` | The full demo page in the browser |

Add `--source phone`, `--source webcam` or `--source file` to any camera command to pick the camera for that run.

## What only you can provide

- **Face photos.** For each person, with their permission, make a folder `data/people/<id>/` holding 3 to 5 clear photos, then describe them in `data/people.json`. Run the enroll command after any change.
- **Notes.** Write what the assistant should know into `data/knowledge.md`.
- **An API key.** Copy `.env.example` to `.env` and put your own Anthropic API key in it. The key stays on this laptop; never paste it into a chat.
- **A sample clip.** Record about 2 minutes from the car's camera height and save it as `data/samples/car_clip.mp4`, so the program can be tested without the phone.

Face photos, the saved face data and `.env` are never committed to Git.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

These run without a camera, the network or the graphics card.

## Licences worth knowing

The YOLO weights and code from Ultralytics are AGPL-3.0, and the InsightFace pretrained face models are for non-commercial research only. Both are fine for a university project; neither may go into a commercial product as is.
