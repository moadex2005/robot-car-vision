# ESP robot car: vision and assistant

A phone on the car streams its camera to this laptop. The laptop labels objects, recognises enrolled people, and serves a web page with the live video. A click on anything in the video opens a panel that says what it is. An assistant answers questions from your notes. Nothing is sent to the car.

## The demo page

```powershell
.venv\Scripts\python.exe -m car_vision.app --open
```

- **Live video** with a box, a label and a track number on every object the detector finds.
- **In view**: a list of those objects. Click one in the list, or click it in the video.
- **Details panel**: a small photo of what you clicked, what it is, a short description and its specifications. For a person the camera recognises, it shows what you wrote about them in their `about.md` instead.
- **Ask about this**: a small chat at the bottom of the details panel, for more questions about the thing you clicked. Each thing keeps its own short conversation (up to 10 questions).
- **Chat**: type a question; the assistant answers from `data/knowledge.md` and searches the web when the notes do not have the answer.
- **Mark a part**: under the photo in the details panel. Switch the brush on, paint over a part of the photo, then ask about it or press **What is this part?**. The assistant gets the same cut-out with your marked area outlined. Not offered for people.
- **Describe**: the eye button on the video. The assistant looks at the whole picture and says what it sees in a few sentences. This sends the whole picture to the API.
- **Voice**: the speaker button in the top bar opens a small box. Switch on **Speak the answers** and the page reads out what answers something you just did: a chat answer, the description of a thing you clicked, a follow-up answer, a scene description. It never speaks on its own. Choose among the English voices your browser has and press **Test** to hear one. The most natural voices are the "Natural" ones in Microsoft Edge, which need the internet. The microphone buttons (beside the chat, in the details panel, and on the video in the full view) let you ask by speaking, in Edge and Chrome.
- **People**: a section further down the page. **Add a person** takes a name, a role and a few lines; **Take photos** then takes five photos from the live camera, telling the person what to do, and the person is recognised from then on, without a restart. **Remove** moves the person's folder to `data/people.removed/` instead of deleting it.
- **Full screen**: the button in the corner of the video, or the `F` key. The live view fills the whole browser window at once, and the whole screen too where the browser allows it. The list and the details panel work there. `F`, the button or `Esc` goes back.
- **Camera**: the button in the top bar. Type the address IP Webcam shows on the phone (like `192.168.1.7:8080`) and press Connect; the page switches to that phone at once and remembers it for next time. Under the address, **Picture from the phone** chooses the picture size. **Automatic** asks IP Webcam for a small picture (640 x 360 for a wide phone): the least delay and the fastest detection. Picking a size makes the phone send it and the page show it, up to 1920 x 1080: sharper video, sharper cut-outs for the assistant, but later and slower. Measured on the demo laptop with the page closed: about 25 frames a second at 640 x 360, 22 at 1280 x 720, 17 at 1920 x 1080. The same box has a **Delay test** switch: it shows a clock with milliseconds, so pointing the phone at the screen shows how far the video is behind.
- **Light and dark**: the sun and moon button in the top bar. Until you choose, the page follows Windows.

Limits to know before the demo:

- The detector knows 80 everyday kinds of object (person, chair, laptop, bottle and so on). Anything else gets no box. Click it anyway: the assistant's API looks at that part of the picture and searches the web if it has to.
- The description and specifications come from a language model looking at a small picture. For an ordinary object they are typical values, and they can be wrong.
- A click on an object sends a small cut-out of the picture to the API, costs one API call and takes a few seconds. A second click on the same tracked object is answered from memory. Each follow-up question sends the same cut-out again and costs one more call. A click on a person sends nothing.
- Asking by voice sends the sound of the question to the browser's speech service (Microsoft or Google) to be turned into text, and a "Natural" or online voice sends the text of the answer there to be spoken. The Claude API itself has no speech: Claude writes the answers, the browser does the listening and the speaking.
- Take photos of a person only with their permission. Photos, face data and what you write about people stay on this laptop.
- Keep one browser tab of the page open, not several: every open tab pulls the video and draws the page, and that slows the detection down.
- The Camera box accepts only addresses on the local network (`192.168.x.x`, `10.x.x.x`, `172.16` to `172.31`), without a login.

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
3. In IP Webcam, scroll down and tap **Start server**. The app shows an address such as `http://192.168.43.1:8080`. You do not need to set the video resolution: the program asks the phone for a small picture itself (`stream.phone_picture: auto` in `config.yaml`).
4. Start the demo page and type that address into the **Camera** box. (Or put the address plus `/video` into `stream.phone_url` in `config.yaml`; an address typed on the page is remembered in `data/camera.json` and wins at the next start.)
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
| `.venv\Scripts\python.exe -m car_vision.app` | The full demo page in the browser (add `--open` to open it) |

Add `--source phone`, `--source webcam` or `--source file` to any camera command to pick the camera for that run.

## What only you can provide

- **People.** The easy way is the **People** section of the demo page. By hand: for each person, with their permission, make a folder `data/people/<id>/` (small letters, digits and underscores) holding 3 to 5 clear photos and an `about.md` that describes them. `data/people.example/about.md` shows the layout: a `# Name` heading, a `Role:` line, a few `- ` bullet lines for the card, then anything else you want the assistant to know. Run the enroll command after adding or changing photos. A folder with only an `about.md` is fine for now; the person is recognised once photos are added and enrolled.
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
