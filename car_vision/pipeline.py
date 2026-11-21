"""The see-decide-show loop: camera frames in, annotated JPEG pictures and a status out.

One background thread takes every new frame from the camera and
1. finds the objects in it (the detector),
2. checks the faces on every few frames (the recognizer) and keeps them on
   screen in between (FaceMemory),
3. writes what it saw into the shared WorldState, which the assistant reads,
4. asks the ReactionEngine which greetings and warnings to show,
5. draws the boxes and names and keeps the picture as JPEG bytes for the page.

The info cards are not drawn on the video: the demo page shows them.

Next to the drawn picture the loop keeps one View: a clean copy of the same
frame (nothing drawn on it) with that frame's detections and faces. When a
visitor clicks the video, target() finds what is under the click in that view
and cuts a small photo of it out of the clean frame.

use_phone() switches the live source to another phone while the loop runs
(the camera box on the page). Each switch starts a new "generation": the
loop notices it, forgets everything about the old picture, and drops any
result it was still working out for the old camera.

Whenever a phone is started, a short thread of its own asks the phone which
picture sizes it offers and, with stream.phone_picture set to auto, asks for
the smallest one that is still sharp enough: smaller pictures arrive sooner.
set_phone_picture() changes that choice for this run (the camera box again).

Every part can be handed in from outside, so the tests run the loop with
fakes and need no camera, model or graphics card.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from car_vision import draw
from car_vision.config import ConfigError, require, section
from car_vision.detector import Detector
from car_vision.faces import FaceMemory, Person
from car_vision.phone import (Fetch, Size, area, ask_for_size, best_size, parse_size, read_picture_sizes,
                              size_text, without_login)
from car_vision.reactions import Reaction, ReactionEngine
from car_vision.recognizer import Recognizer, load_known_faces
from car_vision.state import Box, Detection, RecognizedFace, WorldSnapshot, WorldState
from car_vision.stream import FpsCounter, FrameSource

log = logging.getLogger(__name__)

IDLE_WAIT_SECONDS = 0.005      # pause when no new frame has arrived yet
STOP_WAIT_SECONDS = 2.0        # stop() waits at most this long for the loop to end
MAX_FAILURES_IN_A_ROW = 10     # the loop gives up after this many broken frames in a row
STALE_SECONDS = 2.0            # status() leaves out objects, people and reactions older than this
NOBODY_ENROLLED = "Nobody is enrolled yet, so every face is shown as Unknown."
NO_INFO_CARDS = "No about.md was found for the enrolled people, so recognised faces show only a name."

PHOTO_MARGIN = 0.15            # the photo of a target is its box grown by 15 % on each side
PHOTO_MIN_PIXELS = 160         # ... and at least this wide and high, where the frame allows
CLIP_FINISHED = "Clip finished"   # drawn and shown when a clip has played to its end
AREA_SHARE = 0.5               # a click on no box takes half the frame wide and half the frame high
FACE_LABEL = "face"            # the label of a target that is a face box with no person box around it
DARK_LEVEL = 24                # the plain dark picture shown while a new camera connects

PICTURE_WORDS = ("auto", "keep")   # stream.phone_picture: these two words, or a size like 1280x720
BAD_PICTURE_SETTING = "stream.phone_picture must be auto, keep or a size like 1280x720, got {!r}"
NOT_A_PICTURE_CHOICE = "Choose auto, keep or one of the sizes the phone offers, like 640x360."
READ_BACK_TRIES = 2            # after a change the size is read back from the phone, once more if it lags
READ_BACK_WAIT_SECONDS = 1.0   # ... this long later


@dataclass(frozen=True)
class Target:
    """What a click on the video points at, with a small photo of it."""

    kind: str                    # "object", "person" or "area"
    label: str                   # the detector's label; "" for an area
    confidence: float            # 0.0 for an area
    track_id: int | None
    photo: bytes                 # JPEG cut from the clean frame
    person: Person | None = None # the recognised person, for kind "person"


@dataclass(frozen=True)
class View:
    """One frame as the camera gave it (nothing drawn on it) and what was found in it.

    The loop replaces the whole view at once and never writes into its frame
    afterwards, so a web-server thread can read it without holding the lock.
    """

    frame: np.ndarray
    detections: tuple[Detection, ...]
    faces: tuple[RecognizedFace, ...]

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the frame in pixels."""
        height, width = self.frame.shape[:2]
        return width, height


@dataclass(frozen=True)
class PhonePicture:
    """What the phone said about its picture: the size it sends now and the sizes it offers."""

    current: Size
    offered: tuple[Size, ...]


SourceFactory = Callable[[dict], Any]     # stream settings in, a FrameSource (or a fake) out


class Pipeline:
    """Runs the camera, the models and the reactions in one background thread."""

    def __init__(self, config: dict, source=None, detector=None, recognizer=None,
                 reactions=None, state: WorldState | None = None,
                 source_factory: SourceFactory | None = None,
                 phone_fetch: Fetch | None = None) -> None:
        recognizer_settings = section(config, "recognizer")
        self.every_n_frames: int = require(recognizer_settings, "every_n_frames", int)
        if self.every_n_frames < 1:
            raise ConfigError("recognizer.every_n_frames must be 1 or more")
        self.jpeg_quality: int = require(section(config, "app"), "jpeg_quality", int)
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError(f"app.jpeg_quality must be from 1 to 100, got {self.jpeg_quality}")

        self._notes: list[str] = []
        self._stream_settings = dict(section(config, "stream"))
        phone_url = self._stream_settings.get("phone_url")
        self._phone_url: str | None = phone_url if isinstance(phone_url, str) else None
        self._picture_choice: str = configured_picture_choice(self._stream_settings)
        self._box: Size = (require(self._stream_settings, "width", int),
                           require(self._stream_settings, "height", int))
        self._fetch = phone_fetch              # None: really ask the phone (tests pass a fake)
        self._make_source: SourceFactory = source_factory if source_factory is not None else FrameSource
        # Cheap parts first, so a wrong setting or a missing clip shows before the slow model loads.
        self._source = source if source is not None else self._make_source(self._stream_settings)
        self._reactions = reactions if reactions is not None else ReactionEngine(section(config, "reactions"))
        self._state = state if state is not None else WorldState()
        # A detector that cannot load is a real failure: its error is not caught here.
        self._detector = detector if detector is not None else Detector(section(config, "detector"))
        self._recognizer = recognizer if recognizer is not None else self._build_recognizer(recognizer_settings)
        for note in self._notes:
            log.warning("%s", note)

        self._memory = FaceMemory()
        self._fps = FpsCounter()
        self._lock = threading.Lock()          # guards the values below that both threads use
        self._jpeg: bytes | None = None
        self._view: View | None = None
        self._on_screen: list[Reaction] = []
        self._error: str | None = None
        self._generation = 0                   # goes up by one on every use_phone()
        self._switched_at = 0.0                # sightings from before this moment are old
        self._started = False                  # start() was called
        self._loop_running = False             # the loop thread runs (it ends after too many failures)

        self._picture: PhonePicture | None = None   # None: not a phone, or it did not say
        self._picture_round = 0                # goes up for every new phone or choice; the newest wins

        self._switch_lock = threading.Lock()   # one use_phone() at a time
        self._picture_lock = threading.Lock()  # one talk with the phone about its picture at a time
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Only the loop thread uses these.
        self._frames = 0                       # frames handed to the models so far
        self._failures = 0                     # broken frames in a row
        self._last_drawn: np.ndarray | None = None
        self._lost_line: str | None = None     # the line already drawn on the last frame, if any

    # ---- public ------------------------------------------------------------

    @property
    def state(self) -> WorldState:
        """What the camera sees right now. The assistant reads the same object."""
        return self._state

    def start(self) -> "Pipeline":
        with self._switch_lock:
            with self._lock:
                if self._started:
                    return self
                self._started = True
                self._loop_running = True
                source = self._source
            self._stop.clear()
            source.start()
            self._start_loop()
            self._check_phone_picture()         # only does something when the source is the phone
        return self

    def stop(self) -> None:
        """End the loop (within two seconds), then let go of the camera."""
        self._stop.set()
        with self._switch_lock:                 # a switch under way finishes first
            with self._lock:
                self._started = False
                self._picture_round += 1        # a talk with the phone still under way is ignored
                source = self._source
            thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=STOP_WAIT_SECONDS)
            if thread.is_alive():
                log.warning("The video loop did not stop within %.0f seconds", STOP_WAIT_SECONDS)
        source.stop()

    def use_phone(self, url: str) -> None:
        """Switch the live source to the phone at this address, now.

        Called from a web-server thread while the loop runs. Returns at once:
        the old camera is let go in a thread of its own, because stopping it
        can take seconds while it waits on a phone that is gone. Two switches
        at the same time are done one after the other; the last one wins.
        """
        settings = {**self._stream_settings, "source": "phone", "phone_url": url}
        with self._switch_lock:
            new_source = self._make_source(settings)
            with self._lock:
                old_source = self._source
                self._source = new_source
                self._phone_url = url
                self._forget_old_picture()
                live = self._started and not self._stop.is_set()
                restart = live and not self._loop_running   # the loop had stopped after too many failures
                if restart:
                    self._loop_running = True
            if live:
                new_source.start()
            if restart:
                self._start_loop()
            self._check_phone_picture()
        log.info("Switched the camera to the phone at %s", url)
        let_go_in_background(old_source)

    def set_phone_picture(self, choice: str) -> None:
        """Choose the phone's picture size for this run: "auto", "keep" or a size like "640x360".

        config.yaml is not written. The phone is asked in a thread of its own,
        so this returns at once. Raises ValueError with a plain message for
        anything else, and for a size the phone does not offer.
        """
        wanted = picture_choice(choice)
        if wanted is None:
            raise ValueError(NOT_A_PICTURE_CHOICE)
        size = parse_size(wanted)
        with self._lock:
            offered = self._picture.offered if self._picture is not None else ()
            if size is not None and size not in offered:
                raise ValueError(f"The phone does not offer {size[0]} x {size[1]}. "
                                 "Choose one of the sizes in the list.")
            self._picture_choice = wanted
        log.info("Phone picture: %s", wanted)
        self._check_phone_picture()

    def latest_jpeg(self) -> bytes | None:
        """The newest annotated frame as JPEG bytes; None until the first frame.

        Every new picture is a new bytes object, so a caller can tell a new
        frame from one it has already sent by comparing with `is`.
        """
        with self._lock:
            return self._jpeg

    def status(self) -> dict[str, Any]:
        """Everything the page shows besides the video, as plain JSON-ready values."""
        snapshot = self._state.snapshot()
        with self._lock:
            on_screen = list(self._on_screen)
            error = self._error
            source = self._source
            phone_url = self._phone_url
            switched_at = self._switched_at        # 0.0 until the first switch
            picture, choice = self._picture, self._picture_choice
        # While the stream is down nothing new is seen, so old sightings are left out,
        # and so is everything seen by the camera used before the last switch.
        seen_at = snapshot.detections_at
        fresh = seen_at > switched_at and time.monotonic() - seen_at <= STALE_SECONDS
        return {
            "source": source.source,
            "connected": bool(source.connected) and error is None,   # a stopped loop is not live
            "ended": bool(source.ended),              # a clip played to its end (loop_file: false)
            "phone_url": without_login(phone_url) if phone_url else None,   # in use or remembered; no password
            "phone_picture": picture_status(picture, choice) if source.source == "phone" else None,
            "fps": round(self._fps.value(), 1),
            "device": self._detector.device,
            "recognition": self._recognizer is not None,
            "notes": list(self._notes),
            "people": self._people(snapshot) if fresh else [],
            "objects": objects_in_view(snapshot, self._person_card) if fresh else [],
            "reactions": [{"kind": r.kind, "text": r.text} for r in on_screen] if fresh else [],
            "error": error,
        }

    def target(self, x: float | None = None, y: float | None = None,
               track_id: int | None = None, label: str | None = None) -> Target | None:
        """What a click points at, with a photo cut from the clean frame; None when it is gone.

        x and y are the click as fractions of the picture (0 to 1 from the
        top-left corner). With a track_id the point is not used. Called from
        the web server's threads while the loop runs.
        """
        with self._lock:
            view = self._view                   # the view is never changed, only replaced
        if view is None:
            return None
        found = find_target(view, x, y, track_id, label, self._person_card)
        if found is None:
            return None
        target, box = found
        photo = self._encode(cut_out(view.frame, photo_box(box, view.size)))
        return replace(target, photo=photo)

    # ---- start-up --------------------------------------------------------------

    def _build_recognizer(self, settings: dict) -> Recognizer | None:
        """The real face recogniser, or None when it cannot start. Never stops the program.

        Missing people or a missing embeddings file are handled the way the
        recognizer's own live view handles them: a note, and everyone is Unknown.
        """
        try:
            people, matcher = load_known_faces(settings)
            recognizer = Recognizer(settings, people=people, matcher=matcher)
        except Exception as error:          # a broken model pack or graphics card setup raises many kinds
            log.exception("Face recognition could not start")
            self._notes.append(f"Face recognition is off: {first_line(error)}")
            return None
        if not matcher.person_ids:
            self._notes.append(NOBODY_ENROLLED)
        elif not people:
            self._notes.append(NO_INFO_CARDS)
        return recognizer

    # ---- the phone's picture size (a short thread of its own) ------------------
    #
    # Each call of _check_phone_picture() starts a new "round". The thread of an
    # older round finds out that it is no longer the newest before it writes or
    # asks anything, and stops. Only one thread talks to the phone at a time.

    def _check_phone_picture(self) -> None:
        """Read the phone's picture sizes and ask for the chosen one, off the video loop.

        Does nothing unless the running source is the phone.
        """
        with self._lock:
            self._picture_round += 1
            work = self._picture_round
            live = self._started and not self._stop.is_set()
            url = self._phone_url if self._source.source == "phone" else None
        if live and url is not None:
            threading.Thread(target=self._tune_phone_picture, args=(url, work),
                             name="phone-picture", daemon=True).start()

    def _tune_phone_picture(self, url: str, work: int) -> None:
        """Read the sizes, ask for the chosen one when it differs, then read the size back.

        Every step gives up quietly: a phone that is off or another camera app
        just leaves status()["phone_picture"] at None.
        """
        with self._picture_lock:
            if not self._is_newest(work):
                return
            current, offered = read_picture_sizes(url, self._fetch)
            if current is None or not self._keep_picture(work, current, offered):
                return
            wanted = self._wanted_size(current, offered)
            if wanted is None or not self._is_newest(work):
                return
            log.info("Asking the phone for %s pictures instead of %s", size_text(wanted), size_text(current))
            if ask_for_size(url, wanted, self._fetch):
                self._read_back(url, work, wanted, offered)

    def _read_back(self, url: str, work: int, wanted: Size, offered: list[Size]) -> None:
        """Keep what the phone says it sends after the change; once more a moment later if it lags.

        Only the current size is taken from these answers. The list of offered
        sizes stays the one read before the change: while its camera restarts,
        the real phone lists just one size for a moment.
        """
        for attempt in range(READ_BACK_TRIES):
            if attempt > 0:
                self._stop.wait(READ_BACK_WAIT_SECONDS)
            if not self._is_newest(work):
                return
            current, _listed_during_restart = read_picture_sizes(url, self._fetch)
            if current is None or not self._keep_picture(work, current, offered) or current == wanted:
                return

    def _wanted_size(self, current: Size, offered: list[Size]) -> Size | None:
        """The size to ask the phone for under the present choice; None to leave it as it is."""
        with self._lock:
            choice = self._picture_choice
        if choice == "keep":
            return None
        if choice == "auto":
            return best_size(offered, current, *self._box)
        size = parse_size(choice)
        if size not in offered:
            log.warning("stream.phone_picture is %s, but the phone does not offer that size; "
                        "the phone is left as it is", choice)
            return None
        return None if size == current else size

    def _is_newest(self, work: int) -> bool:
        with self._lock:
            return work == self._picture_round

    def _keep_picture(self, work: int, current: Size, offered: list[Size]) -> bool:
        """Keep what the phone said, unless a newer round began meanwhile (then False)."""
        with self._lock:
            if work != self._picture_round:
                return False
            self._picture = PhonePicture(current, tuple(offered))
            return True

    # ---- the loop thread -----------------------------------------------------

    def _start_loop(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def _forget_old_picture(self) -> None:
        """What use_phone() clears, under the lock: a click must never hit the old camera's
        picture, and the assistant must not be told about people that camera saw."""
        self._generation += 1
        self._switched_at = time.monotonic()
        self._picture = None                    # the new phone has not said yet
        self._picture_round += 1                # what the old phone still says is ignored
        self._view = None
        self._error = None
        self._on_screen = []
        frame_size = self._state.snapshot().frame_size
        self._state.update_detections([], frame_size, self._switched_at)
        self._state.update_faces([], self._switched_at)

    def _run(self) -> None:
        generation = None                       # the camera this loop is working with
        last_sequence = 0
        while not self._stop.is_set():
            with self._lock:                    # the source and its generation, read together
                source, current = self._source, self._generation
            if current != generation:
                switched = generation is not None
                generation, last_sequence = current, 0
                self._start_afresh(source, switched)
            frame, sequence = source.read()
            if frame is None or sequence == last_sequence:
                if not source.connected:
                    self._show_stream_lost(source)
                self._stop.wait(IDLE_WAIT_SECONDS)
                continue
            last_sequence = sequence
            try:
                self._process(frame, generation)
            except Exception as error:      # one bad frame must not end the demo
                if self._too_many_failures(error, generation):
                    return
            else:
                self._failures = 0

    def _start_afresh(self, source: Any, switched: bool) -> None:
        """A new camera: forget the faces, failures, frame count, FPS and lost line of the old one.

        Only the loop thread uses these, so the loop clears them itself. After a
        switch the old picture is replaced at once by a plain dark one of the same
        size with "Connecting to the phone ...", so nothing of the old camera stays
        on screen.
        """
        self._frames = 0
        self._failures = 0
        self._memory = FaceMemory()
        self._fps = FpsCounter()
        self._lost_line = None
        if switched and self._last_drawn is not None:
            self._last_drawn = np.full_like(self._last_drawn, DARK_LEVEL)
            self._show_stream_lost(source)          # no frame yet, so: "Connecting to the phone ..."

    def _process(self, frame: np.ndarray, generation: int) -> None:
        """See, decide and show for one frame. Nothing is written until the models are done,
        and nothing at all when the camera was switched while the models worked."""
        now = time.monotonic()
        index = self._frames
        self._frames += 1

        detections = self._detector.detect(frame)
        if self._recognizer is not None and index % self.every_n_frames == 0:
            found = self._recognizer.recognize(frame, detections)
            self._memory.remember(found, detections, now)
        faces = self._memory.current(detections, now)

        # The boxes are drawn on a copy: the camera's own frame goes into the view untouched.
        drawn = frame.copy()
        draw.draw_detections(drawn, detections)
        draw.draw_faces(drawn, faces)
        jpeg = self._encode(drawn)
        view = View(frame, tuple(detections), tuple(faces))
        height, width = frame.shape[:2]
        with self._lock:                        # state, picture, view and reactions change together
            if self._generation != generation:
                return                          # this frame came from the camera used before the switch
            self._state.update_detections(detections, (width, height), now)
            self._state.update_faces(faces, now)
            on_screen = self._reactions.update(self._state.snapshot(), now)
            self._jpeg = jpeg
            self._view = view
            self._on_screen = list(on_screen)
        self._last_drawn = drawn
        self._lost_line = None
        self._fps.tick(now)

    def _too_many_failures(self, error: Exception, generation: int) -> bool:
        """Log a skipped frame. True when too many failed in a row and the loop must stop."""
        self._failures += 1
        if self._failures == 1:
            log.exception("Skipped a frame because processing it failed")
        else:
            log.warning("Skipped another frame (%d in a row): %s: %s",
                        self._failures, type(error).__name__, error)
        if self._failures < MAX_FAILURES_IN_A_ROW:
            return False
        message = (f"The video loop stopped after {MAX_FAILURES_IN_A_ROW} frames in a row failed "
                   f"({type(error).__name__}: {first_line(error)})")
        jpeg = self._picture_with_line(f"Stopped: {MAX_FAILURES_IN_A_ROW} frames in a row failed")
        with self._lock:                        # decided under the lock, so use_phone() can restart the loop
            if self._generation != generation:
                return False                    # another camera was chosen meanwhile: carry on with it
            self._error = message
            if jpeg is not None:
                self._jpeg = jpeg
            self._loop_running = False
        log.error("%s", message)
        return True

    def _show_stream_lost(self, source: Any) -> None:
        """Keep serving the last picture, with a line on it saying why nothing new comes.

        Each line is drawn once; it is drawn again only when the reason changes,
        e.g. from reconnecting to "Clip finished" when a clip has played to its end.
        """
        if source.ended:
            text = CLIP_FINISHED
        elif self._frames == 0:                 # no picture from this camera yet (just switched)
            text = f"Connecting to the {source.source} ..."
        else:
            text = f"{source.source} stream lost - reconnecting"
        if self._lost_line == text:
            return
        self._lost_line = text
        jpeg = self._picture_with_line(text)
        if jpeg is not None:
            with self._lock:
                self._jpeg = jpeg

    def _picture_with_line(self, text: str) -> bytes | None:
        """The last drawn picture with one line of text on it, as JPEG; None before the first picture."""
        if self._last_drawn is None:
            return None
        frame = self._last_drawn.copy()
        draw.draw_status(frame, text)
        return self._encode(frame)

    def _encode(self, frame: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("OpenCV could not encode the frame as JPEG")
        return buffer.tobytes()

    # ---- status pieces ---------------------------------------------------------

    def _people(self, snapshot: WorldSnapshot) -> list[dict[str, Any]]:
        """One info card per recognised person in view."""
        people: list[dict[str, Any]] = []
        seen: set[str] = set()
        for face in snapshot.known_faces():
            if face.person_id in seen:
                continue
            seen.add(face.person_id)
            card = self._person_card(face)
            people.append({
                "id": face.person_id,
                "name": card.name,
                "role": card.role,
                "lines": list(card.lines),
                "similarity": round(float(face.similarity), 2),
            })
        return people

    def _person_card(self, face: RecognizedFace) -> Person:
        """The person a known face belongs to; just the name when they have no about.md."""
        card = self._recognizer.person(face.person_id) if self._recognizer is not None else None
        return card if card is not None else Person(face.person_id, face.name)


def let_go_in_background(source: Any) -> None:
    """Stop a camera that is no longer used, in a thread of its own.

    FrameSource.stop() waits for its reader thread, which may be stuck waiting
    on a phone that is gone; nobody else has to wait for that.
    """
    def let_go() -> None:
        try:
            source.stop()
        except Exception:                       # a camera driver that fails to close must not matter
            log.exception("Letting go of the old camera failed")

    threading.Thread(target=let_go, name="old-camera", daemon=True).start()


# ---- the phone's picture size: the setting and the status --------------------------------


def picture_choice(text: Any) -> str | None:
    """"auto", "keep" or a size written like "640x360"; None for anything else."""
    if not isinstance(text, str):
        return None
    word = text.strip().lower()
    if word in PICTURE_WORDS:
        return word
    size = parse_size(word)
    return size_text(size) if size is not None else None


def configured_picture_choice(settings: dict[str, Any]) -> str:
    """stream.phone_picture from config.yaml, checked. Raises ConfigError when it is missing or wrong."""
    if "phone_picture" not in settings:
        require(settings, "phone_picture", str)     # raises the usual "missing the setting" message
    value = settings["phone_picture"]
    choice = picture_choice(value)
    if choice is None:
        raise ConfigError(BAD_PICTURE_SETTING.format(value))
    return choice


def picture_status(picture: PhonePicture | None, choice: str) -> dict[str, Any] | None:
    """What the page shows about the phone's picture, e.g.
    {"current": "1920x1080", "offered": ["640x360", ...], "choice": "auto"}; None when the phone did not say."""
    if picture is None:
        return None
    offered = sorted(picture.offered, key=lambda size: (area(size), size[0]))
    return {"current": size_text(picture.current),
            "offered": [size_text(size) for size in offered],
            "choice": choice}


CardOf = Callable[[RecognizedFace], Person]     # the person a known face belongs to


# ---- the objects list on the page --------------------------------------------------


def objects_in_view(snapshot: WorldSnapshot, card_of: CardOf) -> list[dict[str, Any]]:
    """One entry per detection: people first, then by label, then by track number."""
    owners = face_owners(snapshot.detections, snapshot.faces)
    order = sorted(range(len(snapshot.detections)), key=lambda i: list_order(snapshot.detections[i]))
    return [object_entry(snapshot.detections[i], owners.get(i), snapshot.frame_size, card_of) for i in order]


def list_order(detection: Detection) -> tuple[bool, str, bool, int]:
    track = detection.track_id
    return detection.label != "person", detection.label, track is None, track if track is not None else 0


def object_entry(detection: Detection, face: RecognizedFace | None, size: tuple[int, int],
                 card_of: CardOf) -> dict[str, Any]:
    """One entry, e.g. {"track_id": 7, "label": "laptop", "confidence": 0.83, "box": [...], "name": None}.

    `face` is the known face that belongs to this box (see face_owners), or None.
    """
    return {
        "track_id": int(detection.track_id) if detection.track_id is not None else None,
        "label": detection.label,
        "confidence": round(float(detection.confidence), 2),
        "box": box_fractions(detection.box, size),
        "name": card_of(face).name if face is not None else None,
    }


def box_fractions(box: Box, size: tuple[int, int]) -> list[float]:
    """A pixel box as [x1, y1, x2, y2] fractions of the frame, each from 0 to 1."""
    width, height = size
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = box
    return [round(min(max(value / whole, 0.0), 1.0), 3)
            for value, whole in ((x1, width), (y1, height), (x2, width), (y2, height))]


# ---- what a click points at ----------------------------------------------------------


def find_target(view: View, x: float | None, y: float | None, track_id: int | None,
                label: str | None, card_of: CardOf) -> tuple[Target, Box] | None:
    """The target (its photo still empty) and the box to cut it from, or None.

    1. A track id: that detection (with that label, when one is given).
    2. Else the smallest detection or face box that holds the point.
    3. Else, when no label was asked for, the area around the point.
    """
    if track_id is not None:
        index = by_track(view.detections, track_id, label)
        return detection_target(view, index, card_of) if index is not None else None
    if x is None or y is None:
        return None
    width, height = view.size
    px, py = x * width, y * height
    hit = smallest_box_at(view, px, py, label)
    if isinstance(hit, RecognizedFace):
        return face_target(hit, view, card_of)
    if isinstance(hit, Detection):
        return detection_target(view, index_of(view.detections, hit), card_of)
    if label is not None:
        return None                             # the thing asked for is no longer there
    return Target("area", "", 0.0, None, b""), area_box(view.size, px, py)


def by_track(detections: Sequence[Detection], track_id: int, label: str | None) -> int | None:
    """The position of the detection with that track number (and label), or None."""
    for index, detection in enumerate(detections):
        if detection.track_id == track_id and (label is None or detection.label == label):
            return index
    return None


def index_of(detections: Sequence[Detection], detection: Detection) -> int:
    """The position of this very detection (two detections can be equal, so `is` is used)."""
    return next(index for index, item in enumerate(detections) if item is detection)


def smallest_box_at(view: View, px: float, py: float,
                    label: str | None) -> Detection | RecognizedFace | None:
    """The smallest box holding the point, so a phone in a hand beats the person holding it.

    Only detections with `label` count when one is given; face boxes count
    when no label is given or the label is "person".
    """
    boxes: list[Detection | RecognizedFace] = [
        d for d in view.detections if (label is None or d.label == label) and contains(d.box, px, py)]
    if label is None or label == "person":
        boxes += [f for f in view.faces if contains(f.box, px, py)]
    return min(boxes, key=lambda item: box_area(item.box), default=None)


def detection_target(view: View, index: int, card_of: CardOf) -> tuple[Target, Box]:
    """An object, or a person (recognised when a known face belongs to their box)."""
    detection = view.detections[index]
    confidence = float(detection.confidence)
    if detection.label != "person":
        return Target("object", detection.label, confidence, detection.track_id, b""), detection.box
    face = face_owners(view.detections, view.faces).get(index)
    person = card_of(face) if face is not None else None
    return Target("person", detection.label, confidence, detection.track_id, b"", person), detection.box


def face_target(face: RecognizedFace, view: View, card_of: CardOf) -> tuple[Target, Box]:
    """A click on a face means that person: their person box when there is one, else the face box."""
    index = body_of(face, view.detections, view.faces)
    if index is not None:
        return detection_target(view, index, card_of)
    person = card_of(face) if face.is_known else None
    return Target("person", FACE_LABEL, 0.0, face.track_id, b"", person), face.box


# ---- which face belongs to which person box ------------------------------------------
#
# The rules, so a stranger standing behind a recognised person never gets their name:
# - a face with a track number belongs only to the person box with that track number;
# - a face without one belongs to the smallest person box around its centre that has
#   no face of its own;
# - one person is never given to two boxes in the same view.


def face_owners(detections: Sequence[Detection], faces: Sequence[RecognizedFace]) -> dict[int, RecognizedFace]:
    """The known face of each person box, by the box's position in `detections`."""
    owners: dict[int, RecognizedFace] = {}
    given: set[str] = set()                     # person ids already given to a box
    known = [face for face in faces if face.is_known]
    for face in known:                          # matching track numbers come first and win
        index = body_of(face, detections, faces) if face.track_id is not None else None
        if index is not None and index not in owners and face.person_id not in given:
            owners[index] = face
            given.add(face.person_id)
    untracked = [(boxes_around(face, detections, faces), face) for face in known if face.track_id is None]
    untracked.sort(key=lambda item: box_area(detections[item[0][0]].box) if item[0] else 0)
    for boxes, face in untracked:               # then the smallest box around the face
        free = [index for index in boxes if index not in owners]
        if free and face.person_id not in given:
            owners[free[0]] = face
            given.add(face.person_id)
    return owners


def body_of(face: RecognizedFace, detections: Sequence[Detection],
            faces: Sequence[RecognizedFace]) -> int | None:
    """The position of the person box a face belongs to, or None."""
    if face.track_id is not None:
        return next((index for index, d in enumerate(detections)
                     if d.label == "person" and d.track_id == face.track_id), None)
    boxes = boxes_around(face, detections, faces)
    return boxes[0] if boxes else None


def boxes_around(face: RecognizedFace, detections: Sequence[Detection],
                 faces: Sequence[RecognizedFace]) -> list[int]:
    """Person boxes with no face of their own that hold the face's centre, smallest first."""
    tracks_with_a_face = {f.track_id for f in faces if f.track_id is not None}
    middle = centre(face.box)
    found = [index for index, d in enumerate(detections)
             if d.label == "person" and d.track_id not in tracks_with_a_face and contains(d.box, *middle)]
    return sorted(found, key=lambda index: box_area(detections[index].box))


def contains(box: Box, px: float, py: float) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def centre(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


# ---- the photo of a target -----------------------------------------------------------


def area_box(size: tuple[int, int], px: float, py: float) -> Box:
    """Half the frame wide and half the frame high, centred on the point, moved to stay inside."""
    width, height = size
    box_width, box_height = round(width * AREA_SHARE), round(height * AREA_SHARE)
    x1 = slide_inside(round(px - box_width / 2), box_width, width)
    y1 = slide_inside(round(py - box_height / 2), box_height, height)
    return x1, y1, x1 + box_width, y1 + box_height


def slide_inside(start: float, length: float, limit: int) -> float:
    """Move a stretch of `length` so it starts at 0 or later and ends at `limit` or earlier."""
    return max(0, min(start, limit - length))


def photo_box(box: Box, size: tuple[int, int]) -> Box:
    """The part of the frame to cut: the box grown by 15 %, at least 160 pixels, inside the frame."""
    width, height = size
    x1, x2 = photo_span(box[0], box[2], width)
    y1, y2 = photo_span(box[1], box[3], height)
    return x1, y1, x2, y2


def photo_span(start: float, end: float, limit: int) -> tuple[int, int]:
    """One side of photo_box, from left to right or from top to bottom."""
    margin = (end - start) * PHOTO_MARGIN
    start, end = start - margin, end + margin
    short = min(PHOTO_MIN_PIXELS, limit) - (end - start)
    if short > 0:                               # too small: widen it, staying inside the frame
        start = slide_inside(start - short / 2, end - start + short, limit)
        end = start + min(PHOTO_MIN_PIXELS, limit)
    begin = min(max(0, round(start)), limit - 1)
    finish = max(min(limit, round(end)), begin + 1)
    return begin, finish


def cut_out(frame: np.ndarray, box: Box) -> np.ndarray:
    """The pixels inside the box, at their own size (never enlarged)."""
    x1, y1, x2, y2 = box
    return np.ascontiguousarray(frame[y1:y2, x1:x2])


def first_line(error: Exception) -> str:
    """The first line of an error message, or the error's type when it has no message."""
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__
