# Every feature, in detail

[Back to the README](../README.md)

The numbers in the headings match the [sketch of the page](images/page.svg).

## The live view

- **Live video (2)**: a box, a label and a track number on every object the detector finds.
- **In view (5)**: a list of those objects. Click one in the list, or click it in the video.
- **Details panel (6)**: a small photo of what you clicked, what it is, a short description and its specifications. For a person the camera recognises, it shows one short line that introduces them instead.
- **Ask about this (7)**: a small chat at the bottom of the details panel, for more questions about the thing you clicked. Each thing keeps its own short conversation (up to 10 questions).
- **Mark a part**: under the photo in the details panel. Switch the brush on, paint over a part of the photo, then ask about it or press **What is this part?**. The assistant gets the same cut-out with your marked area outlined. Not offered for people.
- **Describe (4)**: the eye button on the video. The assistant looks at the whole picture and says what it sees in a few sentences. This sends the whole picture to the API.
- **Camera off and on (4)**: the round camera button at the top right of the video. Off really lets go of the camera: the laptop camera's light goes out, or the phone's stream is closed, and no model runs. The page says `Camera off`. Press it again, or **Turn on**, to start the same camera again. The camera is always on when the program starts.
- **Full screen (4)**: the button in the corner of the video, or the `F` key. The live view fills the whole browser window at once, and the whole screen too where the browser allows it. The list and the details panel work there. `F`, the button or `Esc` goes back.

## Finding things

- **Look for (3)**: the field on the video. Type what to find, for example `power bank, keys`, and press Find. The camera then looks for those things by name, whatever they are: found ones are drawn in bright yellow with their own outline, not a box, and each word says `found on the right`, `maybe, on the right` (the camera is not sure) or `not in view`. Typing the name of a person the camera knows, like `Sara`, finds that person by their face. You can also type or say `find my keys` or `where is the remote` in the chat or into a microphone, and the page answers `I can see the keys, on the left.` Press Stop to end the search.
- **Watch for (3)**: "tell me when a bottle appears". Type a name in the Look for field and press the bell beside Find, or type or say `tell me when a bottle appears`, `watch for the remote`, `let me know when Sara arrives`; `stop watching` ends it. When the thing comes into view, a quiet notice appears over the video and the Timeline gets a line. The notice is never spoken. A thing that was already in view is announced once it has been away for a few seconds and comes back; after a notice the same thing stays quiet until it has been away for a minute. At most 4 things at once (`watch` in `config.yaml`).
- **Auto-discover**: a switch beside the In view heading, off until you turn it on. Every 10 seconds a small hosted model (through a Hugging Face token) looks at the picture and names what is there; the find-by-name detector then draws live boxes for those names. So the camera labels things beyond its 80 built-in kinds by itself. The switch shows how many looks were made today and what they cost.
- **QR codes and barcodes**: a code in view is marked in the video and gets a pill under the In view list. Click the pill, or the code in the video, to read what it says; **Copy** copies it, and **Open link** appears when it is a plain web address. Nothing is opened by itself.

## Asking

- **Chat (8)**: type a question; the assistant answers from `data/knowledge.md` and searches the web when the notes do not have the answer. Bold, italic, code words and bullet lists in an answer are shown as such, not as `**` marks.
- **Voice (1)**: the speaker button in the top bar opens a small box. Switch on **Speak the answers** and the page reads out what answers something you just did: a chat answer, the description of a thing you clicked, a follow-up answer, a scene description. It never speaks on its own. Choose among the English voices your browser has and press **Test** to hear one. The most natural voices are the "Natural" ones in Microsoft Edge, which need the internet. The microphone buttons (beside the chat, in the details panel, and on the video in the full view) let you ask by speaking, in Edge and Chrome.
- **Hosted voice and hosted listening**: two switches in the same voice box, off until you switch them on. **Hosted voice** speaks the answers with one natural voice that sounds the same in every browser; **Hosted listening** turns a spoken question into words in every browser, also where the browser cannot listen by itself. Both go through your Hugging Face token (`HF_TOKEN` in `.env`) and stop for the day at `voice.daily_budget_usd` (25 cents; one spoken answer costs about 0.03 cents, one question about 0.002 cents). The hosted voice speaks English only.
- **Arabic (1)**: the `EN` / `ع` button in the top bar. With `ع` the assistant answers in Arabic (Modern Standard; `assistant.arabic_style: egyptian` in `config.yaml` gives everyday Egyptian Arabic), shown right to left, and the microphone listens for Arabic. An Arabic answer is spoken by the browser's own Arabic voice (Microsoft Edge has good ones; the hosted voice speaks English only). A browser with no Arabic voice shows the answer and stays silent, and the voice box says so. The page's own buttons and sentences stay English, and Look for and Watch for take English names.

## People and memory

- **People**: a section further down the page. **Add a person** takes a name, a role and a few lines; **Take photos** then takes five photos from the live camera, telling the person what to do, and the person is recognised from then on, without a restart. **Remove** moves the person's folder to `data/people.removed/` instead of deleting it.
- **Timeline (9)**: a section further down the page. The program remembers what it saw and when, with a small picture of each object (never of people). The chat can use it too: `where did you last see the bottle?`, `who came by today?`

## Camera and screen

- **Camera (1)**: the button in the top bar. Type the address IP Webcam shows on the phone (like `192.168.1.7:8080`) and press Connect; the page switches to that phone at once and remembers it for next time. Under the address, **Picture from the phone** chooses the picture size. **Automatic** asks IP Webcam for a small picture (640 x 360 for a wide phone): the least delay and the fastest detection. Picking a size makes the phone send it and the page show it, up to 1920 x 1080: sharper video, sharper cut-outs for the assistant, but later and slower. Measured on the demo laptop with the page closed: about 25 frames a second at 640 x 360, 22 at 1280 x 720, 17 at 1920 x 1080. The same box has a **Delay test** switch: it shows a clock with milliseconds, so pointing the phone at the screen shows how far the video is behind.
- **On a phone**: the page fits a phone's screen (one column, the details as a sheet at the bottom). See [Use a phone](phone.md).
- **Light and dark (1)**: the sun and moon button in the top bar. Until you choose, the page follows Windows.
