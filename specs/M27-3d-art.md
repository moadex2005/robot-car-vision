# M27 brief: 3D art for the page, made in Blender

This is a design brief for a 3D artist who works by script. You build a small set of 3D objects in Blender and render them to transparent picture sequences. A web page will later play those pictures with plain JavaScript (turning an object as the visitor scrolls or drags, tilting it under the pointer). Nothing is drawn in 3D inside the browser, so the pictures are the product: they must look finished.

You work alone in this folder. Everything you need is written here.

## The page they are for

"Robot Car Vision" is a student mechatronics project: a phone rides on a small robot car and streams its camera to a laptop, which finds objects, recognises people and answers questions about what it sees. Its web page is styled like a phone maker's product page: light, spacious, calm, "liquid glass" panels, small motion. Never a dark gamer dashboard.

- Light theme: page background `#f5f5f7`, text `#1d1d1f`, accent blue `#0066cc`.
- Dark theme: page background `#000000`, text `#f5f5f7`, accent blue `#2997ff`.
- The same pictures are shown on both themes, so every picture has a transparent background and must look right on both.

## The look

A product shot, not a game asset. Soft studio light, a clear highlight, gentle shadow on the object itself. Few materials and few colours:

- **ceramic white** (`#f2f2f4`, slightly glossy, a soft clear-coat feel) for bodies and hubs;
- **graphite** (`#2b2b2f`, satin) for tyres, the circuit board and dark parts;
- **smoked glass** (grey, see-through, with a bright edge) for the car's decks and any glass part;
- **one accent blue** (`#2997ff`) used sparingly: a lens ring, a small light, a thin stripe. A small glow on a light is welcome;
- **brushed metal** only for very small parts (standoffs, screws, a shield can).

Every edge is rounded: no razor CG edges anywhere (bevel everything; smooth shading with sensible auto-smooth or weighted normals). No text, no logos, no brand shapes, no numbers, no real faces. No noisy textures: surfaces are clean; a faint roughness variation is the most that is allowed.

Colour management: choose the view transform that keeps the accent blue recognisable as `#2997ff` on a lit surface and the white from turning grey. Check it in a render, not by assumption.

## The objects

### 1. The hero: the robot car

A small four-wheeled robot car, about 20 cm long in real life. Stylised but believable as something a student built well:

- two decks of smoked glass with rounded corners, the upper one a little shorter, held apart by four small metal standoffs;
- four wheels: graphite tyres with a simple tread hint, ceramic white hubs, on short axles from small motor blocks under the lower deck;
- on the upper deck, towards the back: a microcontroller board (a small graphite board with a metal shield can, two rows of pin headers, and one tiny accent-blue light that glows);
- between the decks: a compact battery pack (two rounded cells in a white holder);
- at the front: a phone standing upright in a slim white holder, leaning back about 10 degrees, its back towards the front of the car so its camera looks forward. The camera is a round lens with an accent-blue ring, in the upper corner of the phone's back. The phone is a plain rounded slab: graphite back, darker glass front, no notch details, no logo;
- one thin accent-blue stripe along the edge of the upper deck or the phone holder. No loose wires, or at most two short tidy ones.

Proportions matter more than detail: the phone is the tallest thing and the "face" of the car; the wheels are chunky; the whole car reads clearly as a silhouette at 200 pixels wide.

Deliver for the car:

- **`turn`**: a turntable of 60 frames, one full turn about the vertical axis in equal steps of 6 degrees, a seamless loop. Frame 0 is the most flattering view: three-quarter front, the phone's lens towards the viewer and slightly to the left. The camera looks down at about 18 degrees with mild perspective (a long lens, roughly 70 mm on a full-frame sensor). The car stays centred and the same size through the whole turn. Frame size 960 x 720.
- **`poster`**: frame 0 again at 1920 x 1440, as a PNG.
- **`open`** (second priority, do it after everything else works): 36 frames in which, seen from the frame-0 view, the car's layers lift apart vertically into an exploded view (lower deck with wheels, battery, upper deck, board, phone and holder), easing in and out, so the page can scrub it while the visitor scrolls. Frame 0 equals `turn` frame 0 exactly. Frame size 960 x 720.

### 2. Four section objects

One small object for each section of the page. Each is a single, simple, chunky form in the same materials, readable at 96 pixels:

- **`live`** (the Live view section): the phone from the car by itself, seen from its back at three-quarter view, with one thin glassy accent-blue frame of four rounded corner brackets floating a little in front of its lens (the mark a camera draws around a thing it has found).
- **`ask`** (the Ask section): a speech bubble made of thick smoked glass with a ceramic white rim, with three soft white beads in a row inside it.
- **`people`** (the People section): an identity badge: a ceramic white card with rounded corners, a simple head-and-shoulders shape in relief on it (no face features at all), a slot at the top with a small metal clip, and a small accent-blue dot.
- **`timeline`** (the Timeline section): a thick ceramic white ring like a watch bezel, a smoked glass face, one graphite hand, and three small beads at different points around the ring, one of them accent blue.

Deliver for each:

- **`arc`**: 25 frames in which the object swings about the vertical axis from 30 degrees left to 30 degrees right in equal steps of 2.5 degrees; frame 12 is the straight front view (the object's best three-quarter "front"). The page will pick a frame from where the pointer is. Frame size 384 x 384.
- **`poster`**: frame 12 at 768 x 768, as a PNG.

### 3. Three more section objects (third priority, only if the rest is done and good)

`find` (a glass magnifier with an accent ring), `watch` (a small ceramic bell with a soft accent glow dot), `codes` (a rounded ceramic tile with a pattern of raised graphite squares that looks like a code but is not a real one). Same `arc` and `poster` as above.

## Technical rules

- **Blender**: `C:\Program Files\Blender Foundation\Blender 5.3\blender.exe`, run without a window: `blender.exe --background --factory-startup --python <script> -- <arguments>`. Find out the real names of this version's render engine and settings from the program itself (`bpy.app.version`, the enum items) before you rely on them; do not assume names from older versions.
- **Everything is built by script.** No hand work, no file you did not generate. One command rebuilds all of it: `blender.exe --background --factory-startup --python build_all.py`. Scripts are plain, commented Python that a student can read: one module for materials, one for lights and camera, one per object, one for rendering, one for checking.
- **No outside files and no network**: no downloaded models, textures, fonts, HDR pictures or Python packages. Use only Blender and the Python and numpy that come inside it. Light the scene with lights and a plain or gradient world that you make in the script.
- **Write only inside this folder.** Before starting Blender set `TEMP` and `TMP` to a `tmp` folder here, and `BLENDER_USER_RESOURCES` to a `blender_user` folder here, so Blender writes nothing elsewhere. Do not read or change anything outside this folder, except to run Blender itself.
- **Render engine**: the fast real-time engine (Eevee in this version), with a transparent film. It must finish everything in well under 20 minutes on a GeForce GTX 1660 Ti (6 GB) with 8 GB of system memory, so keep the scenes light: the car under 150 000 triangles, each small object under 30 000. One Blender process at a time. If the graphics card cannot be used from where you run, say so plainly in your report rather than switching to a slow engine without telling.
- **Transparency**: no ground, no backdrop and no cast shadow on a floor in any frame (the page draws its own soft shadow). The object keeps its own shading and contact shadows between its parts. Edges must be clean on both a white and a black page: no dark or light fringe. Check this (see below).
- **Output files** (all under `out/`):
  - `out/hero/turn/000.webp` to `059.webp`, `out/hero/open/000.webp` to `035.webp`, `out/hero/poster.png`
  - `out/icons/<name>/arc/00.webp` to `24.webp`, `out/icons/<name>/poster.png`
  - WebP with an alpha channel, quality about 85. Budgets: the 60 turntable frames together at most 3 MB; each small object's 25 frames together at most 450 kB. If Blender cannot write WebP with alpha directly, write PNG and convert with what Blender's Python offers; say which.
  - `out/models/<name>.glb` for each object (under 3 MB each) and `out/blend/<name>.blend`, for later use.
  - `out/manifest.json`: for each sequence its name, frame count, frame size, the angle of each frame, the file names and their sizes in bytes.
- **Contact sheets** for looking at the work: for each sequence, `out/contact/<name>_light.png` and `out/contact/<name>_dark.png`: every fourth frame of the hero (every third of a small object) laid in a grid, composited over the page's light background `#f5f5f7` and over black `#000000`.

## Check your own work before you report

Write `check.py` (run inside Blender's Python or plain Python with Blender's numpy) and `out/report.md`. For every frame measure, from its alpha channel: the box around the object; that the box keeps at least 4 % of the frame free on every side (nothing cut off); that the centre of the box moves less than 2 % of the frame width through a turntable; that no frame is empty or fully opaque; that file sizes are inside the budgets; that the step between the last and the first turntable frame is the same as every other step. For the fringe check, composite frame 0 over white and over black and measure the brightness of the pixels just inside the edge against the pixels further in: report the numbers.

Then look at the contact sheets yourself, as an art director would, and fix what is weak before you stop: flat or dull light, a grey look where white was asked for, parts that pass through each other, a silhouette that does not read, proportions that look like a toy block, an accent colour that came out purple or pale. Render, look, improve, at least twice for the car. If you cannot view pictures where you run, say so in the report and describe what the numbers tell you instead.

## Your report

End with a short report, also saved as `out/report.md`: what you built (each object in one line), the exact command that rebuilds everything and how long it took, the table of checks, the sizes against the budgets, what you could not do or are not happy with, and anything in this brief you departed from. Plain sentences.
