# Demo runbook: Robot Car Vision

## Ten minutes before

- Plug the laptop into power.
- Close other programs and browser tabs. The laptop has only 8 GB of memory.
- Turn the internet on. The assistant, the natural voices and Auto-discover need it.
- Charge the phone. Check that IP Webcam is installed.
- Put the phone and the laptop on the same network. The simple way: turn on the phone's hotspot and join it from the laptop.
- With the hotspot, the laptop's internet uses the phone's mobile data, and the phone gets warmer.
- The phone gets hot. Start IP Webcam last, just before you switch to it.
- Everyone you will show must already be added in the **People** section of the page, with their permission.

## Start

1. Double-click **Robot Car Vision** on the desktop.
2. A black window opens. The page opens in the browser after about a minute.
3. Keep the black window open. Closing it stops the demo.
4. Do not double-click again while it starts. If a second tab opens, close it.
5. Keep one browser tab on the page.

The page starts with the laptop camera.

## Switch to the phone

1. In IP Webcam, tap **Start server**. It shows an address like `192.168.43.1:8080`.
2. On the page, press **Camera** in the top bar.
3. Type that address and press **Connect**.

## What to show

1. Point the camera at a desk with a bottle on it: each object gets a box, a label and a track number. The **In view** list names them.
2. Click an object: after a few seconds the details panel says what it is, with a short description and specifications.
3. Ask a follow-up in **Ask about this**, like "What is it made of?". The answer is about that object.
4. A person the camera knows steps in: the page says "Hello" with their name. A click on them shows one line: their name and role.
5. Type a thing in **Look for** on the video, like `keys` or `glasses`, and press **Find**: it is drawn with a yellow outline of its shape. Type a known person's name: they are found by their face.
6. Hold a QR code close to the camera, filling about half the picture: the video marks it. Click its pill under **In view** to read what it says.
7. Press the eye button on the video: the assistant describes the whole scene in a few sentences.
8. Press the microphone beside the chat (Edge or Chrome; the browser asks for permission the first time) and ask "Where did you last see the bottle?". To hear the answer, press the speaker button in the top bar: a small box opens with the **Speak the answers** switch.
9. Scroll down to the **Timeline**: what the camera saw and when, with small pictures of objects, never of people.

The round camera button at the top right of the video turns the camera off. The page then says "Camera off". Press it again to turn the camera on.

More to show if there is time: type `bottle` in **Look for** and press the bell (or say "tell me when a bottle appears"), take the bottle away and bring it back: a quiet notice appears, with no sound. Press **EN** in the top bar to get **ع**: the assistant answers in Arabic.

## If something fails

| What you see | What to do |
| --- | --- |
| The page does not open | Wait one minute. Then open the address the black window shows. If the window shows an error, read it. |
| "Program not running" | Start it again with the **Robot Car Vision** shortcut. Close the old tab. |
| The video is black or says "Reconnecting" | Check IP Webcam still runs and the phone is on the same network. Press **Connect** again. |
| The phone does not connect | Same network for both? Type the address exactly as IP Webcam shows it. |
| The video is slow or late | Close extra tabs of the page. Set **Picture from the phone** to **Automatic**. Switch **Auto-discover** off. |
| The assistant says it is not available | Check the internet. Check the API key is in `.env`, then start again. |
| No voice | Open the speaker button, switch on **Speak the answers**, press **Test**. Natural voices need Edge and the internet; in another browser switch on **Hosted voice** there. |
| A person is not recognised | More light, face the camera, come closer. Take their photos again in **People**. |
| Look for finds nothing, or the first search is slow | The first search loads a second model: wait a few seconds. Try a plainer word, or move closer. |
| Everything is stuck | Close the black window. Double-click **Robot Car Vision** again. |

## After the demo

- Close the black window.
- Replace the two keys in `.env` (the Anthropic key and the Hugging Face token) with new ones. They were once pasted into a chat.
