# Limits and privacy

[Back to the README](../README.md)

What the program cannot do, what it sends away and when, and what it never does with a person.

## What the models can and cannot do

- The everyday detector knows 80 kinds of object (person, chair, laptop, bottle and so on). Anything else gets no box unless you look for it by name. You can also click it: the assistant's API looks at that part of the picture and searches the web if it has to.
- The description and specifications come from a language model looking at a small picture. For an ordinary object they are typical values, and they can be wrong.
- Looking for something by name uses a second, slower model (YOLOE) beside the everyday one. It is off until you ask, loads on first use (a few seconds; the very first time it needs the internet to fetch about 250 MB), and while it runs the video is a little slower. It finds what looks like the words: it can be wrong, it may mark similar-looking things too, and it cannot tell your keys from someone else's.
- Three models work together on the picture: a fast local detector on every frame, a slower local one that finds things by name, and (with Auto-discover on) a small hosted one that suggests the names. Claude answers the questions.
- A watch costs nothing by itself: an everyday thing (one of the 80 kinds) or a known person's name needs no extra model, and any other name uses the find-by-name model, as Look for does. A "maybe" never sets off a notice.
- QR codes and barcodes are read on this laptop by OpenCV, with no model and no internet (`codes` in `config.yaml`; `enabled: false` switches it off). Hold a code close and flat: at the usual picture size a QR code is read once it fills about half the picture's height, and a barcode when it is about a third to two thirds of the picture wide. A larger picture from the phone reads from farther away. `tools\make_codes_clip.py` makes a short test clip with both kinds of code, to try this without a camera.

## What is sent away, and when

- A click on an object sends a small cut-out of the picture to the API, costs one API call and takes a few seconds. A second click on the same tracked object is answered from memory. Each follow-up question sends the same cut-out again and costs one more call. A click on a person sends nothing.
- Auto-discover sends a small picture of the whole view, with anyone in it, to a Hugging Face inference provider every few seconds. It costs about half a cent an hour and stops for the day at `lookout.daily_budget_usd` (50 cents). Its names can be wrong; the local detector then looks for things that are not there.
- With a hosted switch on, the text of each answer (hosted voice) or the recording of each question (hosted listening) is sent to a Hugging Face inference provider. Nothing is recorded until you press a microphone button, and the microphone is let go as soon as the question ends. A conversation pauses by itself after 20 turns, or after 3 turns in a room that is never quiet.
- Asking by voice sends the sound of the question to the browser's speech service (Microsoft or Google) to be turned into text, and a "Natural" or online voice sends the text of the answer there to be spoken. The Claude API itself has no speech: Claude writes the answers, the browser does the listening and the speaking.
- The program shows a code's text to you and does nothing else with it: it is not logged, remembered, spoken or given to the assistant. A picture sent for a click, for Describe or for Auto-discover can still show a code.
- The memory is a file on this computer (`data/memory.sqlite3`). It records labels, the names of enrolled people, times and small pictures of objects, and deletes what is older than one day (`memory.keep_days`); **Clear memory** in the Timeline deletes everything at once. A chat question sends its text lines, without pictures, to the assistant.

## People

- A person is never treated as a thing. Words for people (`person`, `man`, `face`) are refused in Look for; a known person's own name is answered by face recognition, never by the object finder. A thing on a person (glasses, keys in a hand) is found and shown, but it stays part of that person: a click on it shows the person and sends nothing, and the memory keeps no picture of it. A thing behind or around a person (a window, a wall) is an ordinary find.
- Faces are covered by a grey block in every cut-out that is sent to the assistant for a click, in the photo the details panel shows, and in the small pictures the memory keeps. Only faces the camera has found are covered: with face recognition off, none are. Describe the scene and Auto-discover still send the whole picture as it is.
- Take photos of a person only with their permission. Photos, face data and what you write about people stay on this laptop.

## Practical

- Keep one browser tab of the page open, not several: every open tab pulls the video and draws the page, and that slows the detection down.
- The Camera box accepts only addresses on the local network (`192.168.x.x`, `10.x.x.x`, `172.16` to `172.31`), without a login.
