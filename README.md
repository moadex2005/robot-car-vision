# ESP robot car: vision and assistant

A phone on the car streams its camera to this laptop. The laptop labels objects, recognises enrolled people, and serves a web page with the live video. A click on anything in the video opens a panel that says what it is. An assistant answers questions from your notes. Nothing is sent to the car.

## The demo page

```powershell
.venv\Scripts\python.exe -m car_vision.app --open
```

Or double-click the **Robot Car Vision** shortcut on the desktop: see "One start button" below. [RUNBOOK.md](RUNBOOK.md) is one page of what to show at a demo and what to do when something fails.

- **Live video** with a box, a label and a track number on every object the detector finds.
- **In view**: a list of those objects. Click one in the list, or click it in the video.
- **Details panel**: a small photo of what you clicked, what it is, a short description and its specifications. For a person the camera recognises, it shows one short line that introduces them instead.
- **Camera off and on**: the round camera button at the top right of the video. Off really lets go of the camera: the laptop camera's light goes out, or the phone's stream is closed, and no model runs. The page says `Camera off`. Press it again, or **Turn on**, to start the same camera again. The camera is always on when the program starts.
- **QR codes and barcodes**: a code in view is marked in the video and gets a pill under the In view list. Click the pill, or the code in the video, to read what it says; **Copy** copies it, and **Open link** appears when it is a plain web address. Nothing is opened by itself.
- **Ask about this**: a small chat at the bottom of the details panel, for more questions about the thing you clicked. Each thing keeps its own short conversation (up to 10 questions).
- **Chat**: type a question; the assistant answers from `data/knowledge.md` and searches the web when the notes do not have the answer. Bold, italic, code words and bullet lists in an answer are shown as such, not as `**` marks.
- **Mark a part**: under the photo in the details panel. Switch the brush on, paint over a part of the photo, then ask about it or press **What is this part?**. The assistant gets the same cut-out with your marked area outlined. Not offered for people.
- **Look for**: the field on the video. Type what to find, for example `power bank, keys`, and press Find. The camera then looks for those things by name, whatever they are: found ones are drawn in bright yellow with their own outline, not a box, and each word says `found on the right`, `maybe, on the right` (the camera is not sure) or `not in view`. Typing the name of a person the camera knows, like `Sara`, finds that person by their face. You can also type or say `find my keys` or `where is the remote` in the chat or into a microphone, and the page answers `I can see the keys, on the left.` Press Stop to end the search.
- **Watch for**: "tell me when a bottle appears". Type a name in the Look for field and press the bell beside Find, or type or say `tell me when a bottle appears`, `watch for the remote`, `let me know when Sara arrives`; `stop watching` ends it. When the thing comes into view, a quiet notice appears over the video and the Timeline gets a line. The notice is never spoken. A thing that was already in view is announced once it has been away for a few seconds and comes back; after a notice the same thing stays quiet until it has been away for a minute. At most 4 things at once (`watch` in `config.yaml`).
- **Auto-discover**: a switch beside the In view heading, off until you turn it on. Every 10 seconds a small hosted model (through a Hugging Face token) looks at the picture and names what is there; the find-by-name detector then draws live boxes for those names. So the camera labels things beyond its 80 built-in kinds by itself. The switch shows how many looks were made today and what they cost.
- **Timeline**: a section further down the page. The program remembers what it saw and when, with a small picture of each object (never of people). The chat can use it too: `where did you last see the bottle?`, `who came by today?`
- **Describe**: the eye button on the video. The assistant looks at the whole picture and says what it sees in a few sentences. This sends the whole picture to the API.
- **Voice**: the speaker button in the top bar opens a small box. Switch on **Speak the answers** and the page reads out what answers something you just did: a chat answer, the description of a thing you clicked, a follow-up answer, a scene description. It never speaks on its own. Choose among the English voices your browser has and press **Test** to hear one. The most natural voices are the "Natural" ones in Microsoft Edge, which need the internet. The microphone buttons (beside the chat, in the details panel, and on the video in the full view) let you ask by speaking, in Edge and Chrome.
- **Hosted voice and hosted listening**: two switches in the same voice box, off until you switch them on. **Hosted voice** speaks the answers with one natural voice that sounds the same in every browser; **Hosted listening** turns a spoken question into words in every browser, also where the browser cannot listen by itself. Both go through your Hugging Face token (`HF_TOKEN` in `.env`) and stop for the day at `voice.daily_budget_usd` (25 cents; one spoken answer costs about 0.03 cents, one question about 0.002 cents). The hosted voice speaks English only.
- **Arabic**: the `EN` / `ع` button in the top bar. With `ع` the assistant answers in Arabic (Modern Standard; `assistant.arabic_style: egyptian` in `config.yaml` gives everyday Egyptian Arabic), shown right to left, and the microphone listens for Arabic. An Arabic answer is spoken by the browser's own Arabic voice (Microsoft Edge has good ones; the hosted voice speaks English only). A browser with no Arabic voice shows the answer and stays silent, and the voice box says so. The page's own buttons and sentences stay English, and Look for and Watch for take English names.
- **On a phone**: the page fits a phone's screen (one column, the details as a sheet at the bottom). See "Open the page on your phone" below.
- **People**: a section further down the page. **Add a person** takes a name, a role and a few lines; **Take photos** then takes five photos from the live camera, telling the person what to do, and the person is recognised from then on, without a restart. **Remove** moves the person's folder to `data/people.removed/` instead of deleting it.
- **Full screen**: the button in the corner of the video, or the `F` key. The live view fills the whole browser window at once, and the whole screen too where the browser allows it. The list and the details panel work there. `F`, the button or `Esc` goes back.
- **Camera**: the button in the top bar. Type the address IP Webcam shows on the phone (like `192.168.1.7:8080`) and press Connect; the page switches to that phone at once and remembers it for next time. Under the address, **Picture from the phone** chooses the picture size. **Automatic** asks IP Webcam for a small picture (640 x 360 for a wide phone): the least delay and the fastest detection. Picking a size makes the phone send it and the page show it, up to 1920 x 1080: sharper video, sharper cut-outs for the assistant, but later and slower. Measured on the demo laptop with the page closed: about 25 frames a second at 640 x 360, 22 at 1280 x 720, 17 at 1920 x 1080. The same box has a **Delay test** switch: it shows a clock with milliseconds, so pointing the phone at the screen shows how far the video is behind.
- **Light and dark**: the sun and moon button in the top bar. Until you choose, the page follows Windows.

Limits to know before the demo:

- The everyday detector knows 80 kinds of object (person, chair, laptop, bottle and so on). Anything else gets no box unless you look for it by name. You can also click it: the assistant's API looks at that part of the picture and searches the web if it has to.
- The description and specifications come from a language model looking at a small picture. For an ordinary object they are typical values, and they can be wrong.
- A click on an object sends a small cut-out of the picture to the API, costs one API call and takes a few seconds. A second click on the same tracked object is answered from memory. Each follow-up question sends the same cut-out again and costs one more call. A click on a person sends nothing.
- Looking for something by name uses a second, slower model (YOLOE) beside the everyday one. It is off until you ask, loads on first use (a few seconds; the very first time it needs the internet to fetch about 250 MB), and while it runs the video is a little slower. It finds what looks like the words: it can be wrong, it may mark similar-looking things too, and it cannot tell your keys from someone else's.
- Three models work together on the picture: a fast local detector on every frame, a slower local one that finds things by name, and (with Auto-discover on) a small hosted one that suggests the names. Claude answers the questions.
- Auto-discover sends a small picture of the whole view, with anyone in it, to a Hugging Face inference provider every few seconds. It costs about half a cent an hour and stops for the day at `lookout.daily_budget_usd` (50 cents). Its names can be wrong; the local detector then looks for things that are not there.
- A person is never treated as a thing. Words for people (`person`, `man`, `face`) are refused in Look for; a known person's own name is answered by face recognition, never by the object finder. A thing on a person (glasses, keys in a hand) is found and shown, but it stays part of that person: a click on it shows the person and sends nothing, and the memory keeps no picture of it. A thing behind or around a person (a window, a wall) is an ordinary find.
- Faces are covered by a grey block in every cut-out that is sent to the assistant for a click, in the photo the details panel shows, and in the small pictures the memory keeps. Only faces the camera has found are covered: with face recognition off, none are. Describe the scene and Auto-discover still send the whole picture as it is.
- A watch costs nothing by itself: an everyday thing (one of the 80 kinds) or a known person's name needs no extra model, and any other name uses the find-by-name model, as Look for does. A "maybe" never sets off a notice.
- QR codes and barcodes are read on this laptop by OpenCV, with no model and no internet (`codes` in `config.yaml`; `enabled: false` switches it off). The program shows a code's text to you and does nothing else with it: it is not logged, remembered, spoken or given to the assistant. A picture sent for a click, for Describe or for Auto-discover can still show a code. Hold a code close and flat: at the usual picture size a QR code is read once it fills about half the picture's height, and a barcode when it is about a third to two thirds of the picture wide. A larger picture from the phone reads from farther away. `tools\make_codes_clip.py` makes a short test clip with both kinds of code, to try this without a camera.
- The memory is a file on this computer (`data/memory.sqlite3`). It records labels, the names of enrolled people, times and small pictures of objects, and deletes what is older than one day (`memory.keep_days`); **Clear memory** in the Timeline deletes everything at once. A chat question sends its text lines, without pictures, to the assistant.
- With a hosted switch on, the text of each answer (hosted voice) or the recording of each question (hosted listening) is sent to a Hugging Face inference provider. Nothing is recorded until you press a microphone button, and the microphone is let go as soon as the question ends. A conversation pauses by itself after 20 turns, or after 3 turns in a room that is never quiet.
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

## One start button

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\make_shortcut.ps1
```

This puts a **Robot Car Vision** shortcut on your desktop. A double-click runs `start_demo.cmd`: a black window opens, the models load, and the page opens in the browser after about a minute. Closing the black window stops the program. A second double-click while it is still starting is refused with a line that says so; once the page runs, it just opens the page. `start_demo.cmd --source phone` passes extra words on to the program.

## Open the page on your phone

The laptop's own page (`http://127.0.0.1:5000/`) opens only on the laptop, with no password, and that never changes. A phone gets a second door of its own, which is off until you switch it on:

1. Add a line `PAGE_PASSWORD=...` to `.env`: a password of your own, 8 characters or more. It stays on this laptop.
2. In `config.yaml` set `phone_door.enabled: true`.
3. Start the program. It prints a line like `Phone door: https://192.168.43.105:5443/`. The first time, Windows Firewall asks whether Python may be reached from the network: allow it for private networks.
4. Put the phone on the same network as the laptop (the phone's own hotspot works), then open that address on the phone. The **Camera** box on the laptop's page shows the address and a QR code to scan.
5. The phone warns that it does not know the certificate. That is expected: the laptop made the certificate itself (in `data/door/`, with the OpenSSL that comes with Git). Choose to continue, then type the password.

What to know: the door speaks HTTPS, so the password and the video cannot be read by others on the network, and the phone's browser allows the microphone. A phone stays signed in for 12 hours. After 5 wrong passwords from one phone it must wait 5 minutes, and after 30 from all phones together everyone must. When something is missing (no password, no OpenSSL, no network) the program says so in one line and runs on without the door. `app.host` must stay `127.0.0.1`: the program refuses to start with any other value, because that would open the page to the network without a password.

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
- **Notes.** Copy `data/knowledge.example.md` to `data/knowledge.md` and write into it what the assistant should know about your project and team. The real file is not committed, because it names people.
- **An API key.** Copy `.env.example` to `.env` and put your own Anthropic API key in it. The key stays on this laptop; never paste it into a chat. For Auto-discover, the hosted voice and hosted listening, add a Hugging Face token as `HF_TOKEN` in the same file (optional). For the phone door, add a password of your own as `PAGE_PASSWORD` (optional).
- **A sample clip.** Record about 2 minutes from the car's camera height and save it as `data/samples/car_clip.mp4`, so the program can be tested without the phone.

Face photos, the saved face data and `.env` are never committed to Git.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

These run without a camera, the network or the graphics card.

## Licence

This project is released under the GNU Affero General Public License v3.0; see [LICENSE](LICENSE). That matches the object detector it is built on: the YOLO weights and code from Ultralytics are AGPL-3.0.

The InsightFace pretrained face models are for non-commercial research only. That is fine for a university project; the face recognition part may not go into a commercial product as is.

Model weights, face photos, face data, recorded clips and the API key are not part of this repository. The models are downloaded on the first run; everything about people stays on the computer that runs the program.
