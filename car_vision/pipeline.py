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

set_camera() turns the camera off and on from the page (the round button on
the video). Off is really off: the live source is let go like an old camera
at a switch, a stand-in (CameraOff) takes its place, and the loop handles no
frames, so no model runs; the memory still writes. On makes and starts the
same camera again. The camera is on at every start.

Whenever a phone is started, a short thread of its own asks the phone which
picture sizes it offers and, with stream.phone_picture set to auto, asks for
the smallest one that is still sharp enough: smaller pictures arrive sooner.
set_phone_picture() changes that choice for this run (the camera box again).

The choice also decides how large the pictures are that the loop works on.
With auto or keep, the phone's pictures are shrunk into the stream.width x
stream.height box from config.yaml, as before. With a chosen size the phone
offers, the box becomes that size (at most 1920 x 1080), so the detector, the
recogniser, the photos for the assistant and the video on the page all get
the sharper picture. A clip or the webcam always keeps the config box.

People can be added from the page without a restart:
- start_photos() begins a photo session for a person folder. On its face-check
  frames the loop looks at the faces the recogniser found and saves the clean
  frame as photo_<n>.jpg when exactly one face is in view, large and sharp
  enough, a second after the last photo. photo_progress() tells the page how
  it goes and what the person should do.
- reload_people() enrols everyone again with the face model that is already
  loaded and swaps the result into the recogniser. A finished photo session
  calls it in a thread of its own. The loop and enrolment take turns with the
  face model one picture at a time (recognizer.SharedFaceModel), so the video
  waits at most for one photo to be read.
- scene_photo() gives the newest clean frame for "Describe the scene".

Looking for things by a typed name (search()):
- A second, slower detector (detector.OpenDetector) finds things by the names
  the visitor typed. It is built and loaded on the first search, never at
  start-up. Loading it and setting the names happen in a thread of its own,
  in "rounds" like the phone's picture size: a newer search replaces an older
  one, and an older round that is still busy gives up as soon as it notices.
- The open detector is used by that thread (load, set_names) and by the loop
  (detect), never by both at once: whoever uses it holds self._open_lock. The
  background thread waits for the lock; the loop only tries it, and when the
  background thread has it the loop simply skips the named detection.
- While names are set, the loop runs the open detector on every few frames
  and keeps its finds on screen in between, for at most a second. merge_named()
  adds them to the everyday detections; what it returns is what the whole page
  sees (the list, the clicks, the photos, the assistant and the memory).

A named find on a person (M20): a person is never treated as an object.
- A find that is the person (same_as_person) is dropped: the name is only
  another word for them.
- A find on a person (on_person: glasses, a shirt, keys held in front of the
  body) is shown, listed and counted as found, but it is never a thing of its
  own: a click on it is a click on that person (nothing is sent), and the
  memory is not told about it.
- Anything else, like a window behind a person, is an ordinary find; but a
  named find that touches a person or face box is remembered without a picture.
Every face box, known or unknown, counts as a person box too (someone close to
the camera often has a face and no person box). Being the person or on a
person sticks to the find's track (PersonGuard), so a frame where the person
detector misses them never turns the find into an object. Every cut-out of an
object or an area has the faces in it covered with grey (cover_faces) before
it is shown or sent.
Words for people are refused as names: a person keeps their own box, name and rules.

Sure and maybe (M20): the open detector gives everything at or above
detector.open_maybe_confidence. A find at or above open_confidence is found,
and its track stays sure while it goes on. A name of the visitor's own with no
sure find gets its single most confident other find as a "maybe": shown with
a "?", never told to the memory. Discovered names never get a maybe.

A known person's name in Look for (M20): a typed name that is an enrolled
person's (their name, their folder name or their first name, see
people_among) is never given to the open detector. That person is found when
their face is recognised in view; their person box is drawn in the wanted
colour. People are still only ever recognised by their enrolled face.

Auto-discover (discover()): a look thread of its own asks the lookout
(lookout.Lookout, a small hosted model) every few seconds what is in view,
using the newest clean frame. The names it gives, minus what the everyday
detector boxes already and minus words for people, are given to the open
detector after the visitor's own names, 6 in all, through the same rounds
as a search. Their finds are marked "discovered", never "found", and are
drawn like everyday ones.

Watching for things (watch(), M25): "tell me when a bottle appears". A watched
name is in view when the everyday detector sees it (one of its labels), when
that person's face is recognised (an enrolled person's name), or when the open
detector surely finds it (any other name: given to it after the visitor's own
names and before discovered ones). It appears when it has been in view for
APPEAR_SECONDS after it was away (see Watcher). Each appearance is one event in
status()["watch"] and one entry in the memory. Nothing about a watch is spoken,
and no reaction is made of it.

The memory (memory.SightingLog): every tracked thing of every processed frame
is told to the sighting log, which decides what is worth keeping. A small
picture is cut for an object only now and then (PictureTimer), never for a
person. The loop calls the log's tick() on every turn, so a thing that left
is written although nothing else comes. A thread of its own puts the log's
summary into the WorldState every few seconds, where the assistant reads it.
stop() closes the log, which writes what is still open.

QR codes and barcodes (codes.CodeReader): on every few frames the loop reads
the codes in the clean frame, keeps them on screen for a moment, draws their
outline and lists them in status()["codes"]. Their text goes to the page only.

Every part can be handed in from outside, so the tests run the loop with
fakes and need no camera, model or graphics card.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Collection, Hashable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from car_vision import draw
from car_vision.codes import Code, CodeReader, checked_settings, web_link
from car_vision.config import ConfigError, require, resolve_path, section
from car_vision.detector import Detector, open_limits
from car_vision.enroll import encode_people, save_enrolment
from car_vision.faces import FaceMatcher, FaceMemory, Person
from car_vision.people import person_folder
from car_vision.phone import (Fetch, Size, area, ask_for_size, best_size, parse_size, read_picture_sizes,
                              size_text, without_login)
from car_vision.reactions import Reaction, ReactionEngine
from car_vision.recognizer import Recognizer, load_known_faces
from car_vision.state import Box, Detection, RecognizedFace, WorldSnapshot, WorldState
from car_vision.stream import FpsCounter, FrameSource, fitted_size

if TYPE_CHECKING:                       # only for the type hints: memory.py is imported when it is used
    from car_vision.memory import SightingLog

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
CAMERA_OFF = "Camera off"         # the line on the still picture, and nothing else is drawn
AREA_SHARE = 0.5               # a click on no box takes half the frame wide and half the frame high
FACE_LABEL = "face"            # the label of a target that is a face box with no person box around it
DARK_LEVEL = 24                # the plain dark picture shown while a new camera connects

PICTURE_WORDS = ("auto", "keep")   # stream.phone_picture: these two words, or a size like 1280x720
BAD_PICTURE_SETTING = "stream.phone_picture must be auto, keep or a size like 1280x720, got {!r}"
NOT_A_PICTURE_CHOICE = "Choose auto, keep or one of the sizes the phone offers, like 640x360."
READ_BACK_TRIES = 2            # after a change the size is read back from the phone, once more if it lags
READ_BACK_WAIT_SECONDS = 1.0   # ... this long later
STREAM_SETTLE_SECONDS = 0.5    # the phone's stream needs this long to switch to a new picture size
MAX_SHOWN_SIZE = (1920, 1080)  # a chosen picture size is shown at most this large
MAX_ASKS = 3                   # at most this many size requests to one phone ...
ASK_WINDOW_SECONDS = 60.0      # ... within this long; then the phone is left as it is
PAUSE_STEP_SECONDS = 0.02      # a waiting phone thread checks this often whether it must give way
TOO_MANY_ASKS = ("The phone changed its picture size back {} times within a minute; it is left as it is "
                 "until the next Connect or another choice in the Camera box.")

RECOGNITION_OFF = "Face recognition is off."   # replaced by the start-up note that says why
FEWEST_PHOTOS, MOST_PHOTOS = 3, 10       # a photo session takes this many photos ...
PHOTO_TIMEOUT_SECONDS = 60.0             # ... within this long
PHOTO_GAP_SECONDS = 2.0                  # between two photos: time to read the hint and change pose
# What the person is asked to do for each photo in turn. The pose is not checked; it is a request.
PHOTO_POSES = ("Look straight at the camera", "Turn your head a little to the left",
               "Turn your head a little to the right", "Lift your chin a little",
               "Lower your chin a little")
PHOTO_MIN_FACE_WIDTH = 80                # the face must be at least this many pixels wide
SHARPNESS_WIDTH = 112                    # faces are measured at the size the face model reads them
SHARPNESS_THRESHOLD = 30.0               # variance of the Laplacian; see photo_hint and its tests
PHOTO_JPEG_QUALITY = 95                  # photos for enrolment are saved with little compression
PHOTO_NAME = re.compile(r"photo_(\d+)", re.IGNORECASE)
HINT_NO_FACE = "Stand in front of the camera"
HINT_ONE_PERSON = "Only one person in the picture, please"
HINT_CLOSER = "Come closer"
HINT_STILL = "Hold still"
HINT_CANCELLED = "Cancelled"
HINT_SWITCHED = "Cancelled: another camera was chosen"
HINT_CAMERA_OFF = "Cancelled: the camera was switched off"

MOST_NAMES = 6                           # search(): at most this many names at once ...
LONGEST_NAME = 40                        # ... each at most this many characters
TOO_MANY_NAMES = "Look for at most {} things at once."
NAME_MARKS = " -'’"                 # besides letters and digits: space, hyphen, apostrophes (' and ’)
NAMED_ID_OFFSET = 100_000                # named track numbers are kept apart from the everyday ones
NAMED_HOLD_SECONDS = 1.0                 # a named find stays on screen this long without being seen again
REPLACE_OVERLAP = 0.6                    # a named find replaces an everyday box it overlaps more than this
PLACES = ("left", "middle", "right")     # the thirds of the picture
SAME_AS_PERSON = 0.6                     # a named find overlapping a person box more than this ...
MOSTLY_INSIDE = 0.8                      # ... with this share of its box inside it, is that person
ON_PERSON_SHARE = 0.5                    # a named find with this share of its box inside a person box is on them
FACE_COVER_MARGIN = 0.2                  # a face in a cut-out is covered by its box grown 20 % on each side ...
FACE_COVER_GREY = 128                    # ... filled with this plain mid-grey
NO_PEOPLE = ("Looking for people by name is not supported. "
             "People the camera knows are recognised by their face.")
# Words for people and parts of people, one list: a name with one of them is refused, because
# a find on a person would turn them into an "object" (a picture of them kept, a click sent out).
PEOPLE_WORDS = frozenset({
    "person", "persons", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "child", "children", "kid", "kids", "baby", "babies", "face", "faces", "head", "heads",
    "hand", "hands", "body", "bodies", "human", "humans", "someone", "somebody", "anybody", "anyone",
})

# Auto-discover (M14): a hosted model names what is in view now and then.
DISCOVERED_KEEP_SECONDS = 60.0           # a discovered name stays this long after it was last mentioned
FAILURES_BEFORE_WAIT = 3                 # after this many failed looks in a row ...
FAILURE_WAIT_SECONDS = 60.0              # ... the next look waits this long
LASTING_PROBLEMS = frozenset({"budget", "no_key", "key_refused", "no_credit", "not_available"})
NOT_A_FAILURE = "still_looking"          # the lookout's previous look is still under way: just skip
LOOK_FAILED = "Auto-discover could not look at the picture."
# The everyday detector's 80 labels, and the plain words for some of them that a hosted
# model tends to use. A discovered name on this list is dropped: that detector boxes it already.
EVERYDAY_LABELS = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
})
EVERYDAY_VARIANTS = frozenset({
    "television", "tv screen", "monitor", "computer monitor", "sofa", "phone", "mobile phone",
    "smartphone", "cellphone", "telephone", "bike", "motorbike", "plane", "aeroplane", "table",
    "plant", "houseplant", "purse", "bag", "ball", "doughnut", "fridge", "remote control",
    "computer mouse", "laptop computer", "mug", "hair dryer", "teddy", "glass", "notebook computer",
})

# Watching for things (M25): a quiet event when a watched thing comes into view.
APPEAR_SECONDS = 1.0                     # a watched thing appears once in view this long after being away
MOST_EVENTS = 10                         # status()["watch"]["events"] keeps the newest this many
START_AWAY_SECONDS = 3.0                 # a thing there at the start is announced once away this long, then back
TOO_MANY_WATCHED = "Watch for at most {} things at once."

MEMORY_SHARE_SECONDS = 5.0               # the memory's summary goes into the WorldState this often
PICTURE_EVERY_SECONDS = 5.0              # the sighting log is given a new picture of a thing at most this often
PICTURE_FORGET_SECONDS = 10.0            # ... and a thing not seen for this long is forgotten by the timer
PICTURE_MAX_SIDE = 160                   # the picture's longer side, in pixels

MOST_CODES = 8                           # status()["codes"] lists at most this many, the largest first


@dataclass(frozen=True)
class Target:
    """What a click on the video points at, with a small photo of it."""

    kind: str                    # "object", "person" or "area"
    label: str                   # the detector's label; "" for an area
    confidence: float            # 0.0 for an area
    track_id: int | None
    photo: bytes                 # JPEG cut from the clean frame
    person: Person | None = None # the recognised person, for kind "person"
    epoch: int = 0               # goes up whenever track numbers start again (new camera, new frame size)


@dataclass(frozen=True)
class View:
    """One frame as the camera gave it (nothing drawn on it) and what was found in it.

    The loop replaces the whole view at once and never writes into its frame
    afterwards, so a web-server thread can read it without holding the lock.
    """

    frame: np.ndarray
    detections: tuple[Detection, ...]
    faces: tuple[RecognizedFace, ...]
    epoch: int = 0                      # the track-number epoch this frame's track numbers belong to
    on_person: tuple[Detection, ...] = ()   # the named finds on a person: a click on one is a click on them
    maybes: tuple[Detection, ...] = ()      # the named finds the finder is not sure of

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


@dataclass
class PhotoSession:
    """One run of taking photos of a person. Changed only while the pipeline's lock is held."""

    person_id: str
    folder: Path
    wanted: int
    started_at: float
    generation: int                    # the camera the session began with
    taken: int = 0
    last_photo_at: float = float("-inf")
    state: str = "taking"              # "taking", "enrolling", "done" or "failed"
    hint: str = HINT_NO_FACE
    notes: list[str] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.state in ("taking", "enrolling")

    def progress(self) -> dict[str, Any]:
        """What the page shows, as plain JSON-ready values."""
        return {"person_id": self.person_id, "taken": self.taken, "wanted": self.wanted,
                "state": self.state, "hint": self.hint, "notes": list(self.notes)}


@dataclass(frozen=True)
class PersonName:
    """A name the visitor typed that is an enrolled person's: who it means and how it is shown."""

    typed: str                         # as typed (cleaned), e.g. "mohamed"
    shown: str                         # as enrolled, e.g. "Mohamed Adel"
    person_ids: tuple[str, ...]        # everyone it means: several people can share a first name


@dataclass(frozen=True)
class Search:
    """What the open detector is asked to find: the visitor's own names, then discovered ones.

    Replaced as a whole, under the pipeline's lock. Every new list of names is
    a new round; the newest round wins. `state` is that of the open detector
    for this round's names. The own names that are enrolled people's (`people`)
    are never given to the open detector: the recognised faces answer them.
    """

    names: tuple[str, ...] = ()        # the visitor's own names (status()["search"] reports only these)
    state: str = "off"                 # "off", "loading", "ready" or "failed"
    note: str = ""                     # why it failed
    round: int = 0                     # goes up with every new list of names; the newest round wins
    discovered: tuple[str, ...] = ()   # discovered names given to the open detector after the own ones
    own_ready: bool = False            # the own names were ready already (only discovered ones changed)
    people: tuple[PersonName, ...] = ()    # the own names that are enrolled people's
    # How each own name is shown, fixed when the visitor searched: enrolling someone later may
    # change how a name is answered, never its text (the page knows the search by it).
    shown: tuple[str, ...] = ()
    watched: tuple[str, ...] = ()      # watched names given to the open detector after the own ones (M25)
    watch_ready: bool = False          # the watched names were ready already (only discovered ones changed)

    @property
    def object_names(self) -> tuple[str, ...]:
        """The own names that are not people's: these go to the open detector."""
        typed = {person.typed for person in self.people}
        return tuple(name for name in self.names if name not in typed)

    @property
    def finder_names(self) -> tuple[str, ...]:
        """Every name the open detector is given: the own ones first, then the watched, then the discovered."""
        return self.object_names + self.watched + self.discovered


@dataclass(frozen=True)
class Discovery:
    """Auto-discover: on or off, and how its looks go. Replaced as a whole, under the pipeline's lock."""

    on: bool = False
    state: str = "off"                 # "off", "looking" or "paused"
    note: str = ""                     # why it is paused, in plain words
    round: int = 0                     # goes up with every discover(); a look thread of an older round ends


@dataclass(frozen=True)
class Watch:
    """The things to watch for (watch()). Replaced as a whole, under the pipeline's lock."""

    names: tuple[str, ...] = ()            # as cleaned (clean_names)
    people: tuple[PersonName, ...] = ()    # the names that are enrolled people's: their faces answer them
    shown: tuple[str, ...] = ()            # how each name is shown: a person's name as enrolled

    def person(self, name: str) -> PersonName | None:
        """The person a watched name means, or None for a thing."""
        return next((person for person in self.people if person.typed == name), None)

    def text(self, name: str) -> str:
        """How a watched name is shown, e.g. "bottle" or "Mohamed Adel"."""
        return self.shown[self.names.index(name)] if name in self.names else name

    @property
    def finder_names(self) -> tuple[str, ...]:
        """The names only the open detector can answer: not people's, not everyday labels."""
        return tuple(name for name in self.names if self.person(name) is None and not everyday_label(name))


@dataclass(frozen=True)
class NamedFinds:
    """What the named search found, kept on screen between its runs (see merge_named)."""

    search_round: int = 0
    finds: tuple[tuple[Detection, float], ...] = ()     # each find and when it was last seen

    @property
    def detections(self) -> tuple[Detection, ...]:
        return tuple(detection for detection, _seen_at in self.finds)


class PictureTimer:
    """Decides when the sighting log is given a small picture of a thing.

    Cutting and encoding a picture costs more than everything else the memory
    does, so it is not done for every thing on every frame. A thing gets its
    first picture once it has been in view for `first_after` seconds (when the
    log starts keeping it), then a new one every PICTURE_EVERY_SECONDS. Only
    the loop thread uses it.
    """

    def __init__(self, first_after: float, every: float = PICTURE_EVERY_SECONDS,
                 forget_after: float = PICTURE_FORGET_SECONDS) -> None:
        self.first_after = first_after
        self.every = every
        self.forget_after = forget_after
        self._times: dict[Hashable, list[float]] = {}    # key -> [first seen, last seen, last picture]

    def due(self, key: Hashable, now: float) -> bool:
        """True when this thing should get a picture now; counts it as given."""
        times = self._times.setdefault(key, [now, now, float("-inf")])
        first_seen, _last_seen, given_at = times
        times[1] = now
        if now - first_seen < self.first_after or now - given_at < self.every:
            return False
        times[2] = now
        return True

    def forget_old(self, now: float) -> None:
        """Drop the things not seen for forget_after seconds."""
        old = [key for key, times in self._times.items() if now - times[1] > self.forget_after]
        for key in old:
            del self._times[key]

    def __len__(self) -> int:
        return len(self._times)


SourceFactory = Callable[[dict], Any]     # stream settings in, a FrameSource (or a fake) out


class Pipeline:
    """Runs the camera, the models and the reactions in one background thread."""

    def __init__(self, config: dict, source=None, detector=None, recognizer=None,
                 reactions=None, state: WorldState | None = None,
                 source_factory: SourceFactory | None = None,
                 phone_fetch: Fetch | None = None,
                 open_detector=None, sightings=None, lookout=None, code_reader=None) -> None:
        recognizer_settings = section(config, "recognizer")
        self.every_n_frames: int = require(recognizer_settings, "every_n_frames", int)
        if self.every_n_frames < 1:
            raise ConfigError("recognizer.every_n_frames must be 1 or more")
        self.jpeg_quality: int = require(section(config, "app"), "jpeg_quality", int)
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError(f"app.jpeg_quality must be from 1 to 100, got {self.jpeg_quality}")
        # Where enrolling again reads the photos and saves the face vectors.
        self._photos_dir = resolve_path(require(recognizer_settings, "photos_dir", str))
        self._embeddings_path = resolve_path(require(recognizer_settings, "embeddings_file", str))
        self._match_threshold = float(require(recognizer_settings, "match_threshold", (int, float)))
        self._detector_settings = section(config, "detector")
        self.open_every_n_frames: int = require(self._detector_settings, "open_every_n_frames", int)
        if self.open_every_n_frames < 1:
            raise ConfigError("detector.open_every_n_frames must be 1 or more")
        # A named find at or above this is sure; the open detector checks and uses the maybe limit.
        self.sure_confidence, _maybe = open_limits(self._detector_settings)
        memory_settings = section(config, "memory")
        memory_file = require(memory_settings, "file", (str, type(None)))     # empty: no memory
        keep_days = float(require(memory_settings, "keep_days", (int, float)))
        min_seconds = float(require(memory_settings, "min_seconds", (int, float)))
        keep_pictures = require(memory_settings, "pictures", bool)
        # QR codes and barcodes (M16): read with OpenCV alone. Off: no reader, nothing is read.
        code_settings = section(config, "codes")
        self.codes_on: bool = require(code_settings, "enabled", bool)
        self.code_every_n_frames, _max_width, self.code_hold_seconds = checked_settings(code_settings)
        self._code_reader = None
        if self.codes_on:
            self._code_reader = code_reader if code_reader is not None else CodeReader(code_settings)
        # Auto-discover: the pipeline reads only what it needs itself; Lookout checks the rest.
        self._lookout_settings = section(config, "lookout")
        self.look_every_seconds = float(require(self._lookout_settings, "every_seconds", (int, float)))
        if self.look_every_seconds <= 0:
            raise ConfigError("lookout.every_seconds must be more than 0")
        self.look_picture_width: int = require(self._lookout_settings, "picture_width", int)
        if self.look_picture_width < 32:
            raise ConfigError("lookout.picture_width must be 32 pixels or more")
        self.look_budget = float(require(self._lookout_settings, "daily_budget_usd", (int, float)))
        # Watching for things (M25): read here, so a wrong setting shows before the models load.
        watch_most, watch_again_after = watch_settings(section(config, "watch"))

        self._notes: list[str] = []
        self._recognition_off = RECOGNITION_OFF
        self._stream_settings = dict(section(config, "stream"))
        phone_url = self._stream_settings.get("phone_url")
        self._phone_url: str | None = phone_url if isinstance(phone_url, str) else None
        self._picture_choice: str = configured_picture_choice(self._stream_settings)
        # The box from config.yaml: used for auto and keep, and by auto to pick a size.
        self._config_box: Size = (require(self._stream_settings, "width", int),
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
        # The open detector is only built on the first search (see _open_detector_ready),
        # the lookout on the first look (see _lookout_ready).
        self._open_detector = open_detector
        self._lookout = lookout
        self._sightings = (sightings if sightings is not None
                           else self._build_sightings(memory_file, keep_days, min_seconds))
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

        self._epoch = 0                        # goes up whenever track numbers start again
        # For the memory's keys, the two kinds of track numbers are counted apart, so a new
        # search does not split an everyday object that stayed in view into two sightings.
        self._everyday_epoch = 0               # the everyday tracks started again (new camera or frame size)
        self._named_epoch = 0                  # the named tracks started again (also: new names)
        self._picture: PhonePicture | None = None   # None: not a phone, or it did not say
        self._picture_round = 0                # goes up for every new phone or choice; the newest wins
        self._stream_size: Size | None = None  # what the phone sent when its stream was last opened
        self._asked_at: list[float] = []       # when this phone was asked for a size (this connect, this choice)
        self._gave_up_asking = False           # too many asks: leave the phone as it is

        self._photos: PhotoSession | None = None    # the newest photo session, under self._lock
        self._enrol_lock = threading.Lock()    # one enrolment at a time
        self._faces_changed = threading.Event()     # set after enrolling: the loop forgets the faces it keeps

        self._search = Search()                # the newest search, under self._lock
        self._found: tuple[Detection, ...] = ()     # finds of the own names in the last frame shown, under self._lock
        self._discovered: tuple[Detection, ...] = ()    # ... and of discovered names
        self._looked_for: tuple[str, ...] | None = None  # the own names a named detect last ran for
        self._open_lock = threading.Lock()     # held by whoever uses the open detector (see the module docstring)

        self._discovery = Discovery()          # auto-discover, under self._lock
        # Watching for things (M25). Under self._lock: the list, what the loop last decided, the events.
        self.watch_most: int = watch_most      # watch.max_names
        self._watch = Watch()
        self._watch_seen: dict[str, bool] = {}     # judged name -> in view, from the last frame shown
        self._watch_events: list[dict[str, Any]] = []   # the newest MOST_EVENTS events, oldest first
        self._watch_last_at: dict[str, float] = {}  # name -> wall-clock time of its last event
        self._watch_count = 0                  # events so far in this run: the next event's id is one more
        # Only the loop thread uses these.
        self._watcher = Watcher(watch_again_after)
        self._finder_looked: set[str] = set()  # names (folded) the open detector looked for in this round
        self._looked_round = 0                 # ... that round
        self._discovered_names: dict[str, tuple[float, int]] = {}   # name -> (last mentioned, its place in that look)
        self._look_lock = threading.Lock()     # one look thread talks to the lookout at a time

        self._switch_lock = threading.Lock()   # one use_phone() at a time
        self._camera_on = True                 # set_camera(): on at every start, under self._lock
        self._phone_locks: dict[str, threading.Lock] = {}   # one talk with each phone at a time
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Only the loop thread uses these.
        self._frames = 0                       # frames handed to the models so far
        self._failures = 0                     # broken frames in a row
        self._last_drawn: np.ndarray | None = None
        self._drawn_from: int | None = None    # the camera generation _last_drawn belongs to
        self._lost_line: str | None = None     # the line already drawn on the last frame, if any
        self._frame_size: tuple[int, int] | None = None   # (width, height) of the last frame worked on
        self._named = NamedFinds()             # the named finds kept on screen between runs
        self._guard = PersonGuard()            # which named tracks are on a person, or are one
        self._sure_tracks: set[int] = set()    # named tracks that once reached open_confidence
        self._named_tick = 0                   # ready frames of this search round so far
        self._open_tracks_stale = False        # the open detector must start its tracks again first
        self._pictures = PictureTimer(min_seconds) if keep_pictures else None
        self._memory_failed = False            # a memory error was logged already
        self._codes: dict[tuple[str, str], tuple[Code, float]] = {}   # (kind, text) -> newest read, when read
        self._codes_failed = False             # a code reader error was logged already
        self._code_reads: list[float] = []     # when the reader last ran, the two newest times
        # Under self._lock: the codes drawn on the last frame shown, and that frame's (width, height).
        self._shown_codes: tuple[tuple[Code, ...], tuple[int, int]] = ((), (0, 0))

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
            self._start_memory_summary()
            self._check_phone_picture()         # only does something when the source is the phone
        return self

    def stop(self) -> None:
        """End the loop (within two seconds), then let go of the camera and close the memory.

        A search or auto-discover under way ends too: a search still loading is
        never marked ready afterwards, and no look happens after this.
        """
        self._stop.set()
        with self._switch_lock:                 # a switch under way finishes first
            with self._lock:
                self._started = False
                self._picture_round += 1        # a talk with the phone still under way is ignored
                self._search = Search(round=self._search.round + 1)        # an older round gives up
                self._discovery = Discovery(round=self._discovery.round + 1)
                self._discovered_names = {}
                source = self._source
            thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=STOP_WAIT_SECONDS)
            if thread.is_alive():
                log.warning("The video loop did not stop within %.0f seconds", STOP_WAIT_SECONDS)
        source.stop()
        self._close_memory()

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
                self._camera_on = True          # Connect means "use this camera", also while it is off
                self._forget_old_picture()
                cancelled = self._cancel_photos_locked(HINT_SWITCHED)
                live = self._started and not self._stop.is_set()
                restart = live and not self._loop_running   # the loop had stopped after too many failures
                if restart:
                    self._loop_running = True
            if live:
                new_source.start()
            if restart:
                self._start_loop()
            self._check_phone_picture()
            self._resume_search()
        log.info("Switched the camera to the phone at %s", url)
        if cancelled is not None:
            log.info("Photos of %s cancelled by the camera switch after %d", cancelled.person_id, cancelled.taken)
        let_go_in_background(old_source)

    def set_camera(self, on: bool) -> None:
        """Turn the camera off (False) or on again (True), now. It is on at every start.

        Off: the live source is let go in a thread of its own (the webcam's
        light goes out, the phone's stream is closed) and a CameraOff stand-in
        takes its place. Everything the old picture showed is forgotten as at a
        camera switch, a photo session is cancelled, and the loop handles no
        frames, so no model runs; the memory still writes. A frame already in
        the models when the camera goes off finishes at most the one model call
        that was under way, and nothing of it is kept. On: a new source is made
        from the settings in use now (see settings_to_turn_on) and started.
        Returns at once, like use_phone(). The value it already has does nothing.
        ConfigError when the source cannot be made (a clip that was deleted);
        the camera then stays off.
        """
        if not isinstance(on, bool):
            raise ValueError("The camera is switched with true or false.")
        with self._switch_lock:
            with self._lock:
                if on == self._camera_on:
                    return
                kind = self._source.source
            if on:
                new_source = self._make_source(settings_to_turn_on(self._stream_settings, kind, self._phone_url))
            else:
                new_source = CameraOff(kind)
            with self._lock:
                old_source = self._source
                self._source = new_source
                self._camera_on = on
                self._forget_old_picture()
                cancelled = None if on else self._cancel_photos_locked(HINT_CAMERA_OFF)
                live = self._started and not self._stop.is_set()
                restart = live and not self._loop_running   # the loop had stopped after too many failures
                if restart:
                    self._loop_running = True
            if live:
                new_source.start()
            if restart:
                self._start_loop()
            self._check_phone_picture()         # only does something when the camera is on and a phone
            if on:
                self._resume_search()
        log.info("The camera was turned %s", "on" if on else "off")
        if cancelled is not None:
            log.info("Photos of %s cancelled by turning the camera off after %d", cancelled.person_id, cancelled.taken)
        let_go_in_background(old_source)

    def set_phone_picture(self, choice: str) -> None:
        """Choose the phone's picture size for this run: "auto", "keep" or a size like "640x360".

        config.yaml is not written. The box the phone's pictures are fitted into
        changes at once (see shown_box); the phone is asked in a thread of its
        own, so this returns at once. Raises ValueError with a plain message for
        anything else, for a size the phone does not offer, and for any size
        while the camera is off (auto and keep are only stored then).
        """
        wanted = picture_choice(choice)
        if wanted is None:
            raise ValueError(NOT_A_PICTURE_CHOICE)
        size = parse_size(wanted)
        with self._lock:
            if size is not None and not self._camera_on:     # the phone has said nothing since it went off
                raise ValueError("Turn the camera on first to choose a picture size.")
            offered =self._picture.offered if self._picture is not None else ()
            if size is not None and size not in offered:
                raise ValueError(f"The phone does not offer {size[0]} x {size[1]}. "
                                 "Choose one of the sizes in the list.")
            self._picture_choice = wanted
            self._set_phone_box()
            self._forget_asks()                 # a new choice may ask the phone again
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
        with self._lock:
            # The loop writes the detections under this lock too, so the entries and what marks
            # them (found, maybe, on a person) always come from the same frame.
            snapshot = self._state.snapshot()
            on_screen = list(self._on_screen)
            error = self._error
            source = self._source
            camera_on = self._camera_on
            phone_url = self._phone_url
            switched_at = self._switched_at        # 0.0 until the first switch
            picture, choice = self._picture, self._picture_choice
            search, found, discovered = self._search, self._found, self._discovered
            looked = search.names and self._looked_for == search.names
            discovery, discovered_names = self._discovery, self._discovered_list_locked()
            codes, code_frame_size = self._shown_codes
            view = self._view
            watch, watch_seen = self._watch, dict(self._watch_seen)
            watch_events, watch_last_at = list(self._watch_events), dict(self._watch_last_at)
        on_person, maybes = (view.on_person, view.maybes) if view is not None else ((), ())
        # While the stream is down nothing new is seen, so old sightings are left out,
        # and so is everything seen by the camera used before the last switch.
        seen_at = snapshot.detections_at
        fresh = seen_at > switched_at and time.monotonic() - seen_at <= STALE_SECONDS
        found, discovered = (found, discovered) if fresh else ((), ())
        wanted_people = {person_id for person in search.people for person_id in person.person_ids}
        people_seen = people_in_view(search.people, snapshot.detections, snapshot.faces) if fresh else {}
        return {
            "source": source.source,
            "camera": {"on": camera_on},
            "connected":bool(source.connected) and error is None,   # a stopped loop is not live
            "ended": bool(source.ended),              # a clip played to its end (loop_file: false)
            "phone_url": without_login(phone_url) if phone_url else None,   # in use or remembered; no password
            "phone_picture": picture_status(picture, choice) if source.source == "phone" else None,
            "fps": round(self._fps.value(), 1) if camera_on else 0.0,
            "device": self._detector.device,
            "recognition": self._recognizer is not None,
            "notes": list(self._notes),
            "people": self._people(snapshot) if fresh else [],
            "objects": objects_in_view(snapshot, self._person_card, found, discovered, maybes=maybes,
                                       on_person=on_person, wanted_people=wanted_people) if fresh else [],
            "codes": codes_in_view(codes, code_frame_size) if fresh else [],
            "reactions": [{"kind": r.kind, "text": r.text} for r in on_screen] if fresh else [],
            "error": error,
            "search": search_status(search, found, snapshot.frame_size[0], bool(looked and fresh),
                                    maybes, people_seen),
            "watch": watch_status(watch, search, watch_seen, fresh, camera_on, watch_events, watch_last_at),
            "discover": self._discover_status(discovery, discovered_names, search),
        }

    def search(self, names: Sequence[str]) -> tuple[str, ...]:
        """Look for these things by name from now on; [] stops looking.

        1 to 6 names, each 1 to 40 letters, digits, spaces, hyphens or
        apostrophes, and not words for people; the same name twice (in any
        case) counts once. Anything else raises ValueError with a plain
        message. Returns at once: loading the open detector and setting the
        names happen in a thread of its own (status()["search"]["state"] says
        how far it is), and a newer search replaces an older one that has not
        finished. The finds of names no longer looked for vanish at once.

        An enrolled person's name is answered by their recognised face, never
        by the open detector. Returns the names as status()["search"]["names"]
        shows them: a person's name as they were enrolled.
        """
        wanted = clean_names(names)
        with self._lock:
            if wanted != self._search.names:
                self._looked_for = None         # the page waits until the finder has looked for these
            work = self._new_finder_round_locked(wanted, self._search.discovered, always=True)
            shown = shown_names(self._search)
        log.info("Looking for %d thing(s)", len(wanted))     # the names themselves stay off the log
        self._prepare_in_background(work)
        return shown

    def discover(self, on: bool) -> None:
        """Switch auto-discover on or off (it is off at start-up).

        On: a thread of its own asks the lookout every lookout.every_seconds
        what is in view, and the open detector looks for those names too. Off:
        the looks stop at once, the discovered names are forgotten and the open
        detector is given only the visitor's own names again.
        """
        if not isinstance(on, bool):
            raise ValueError("Auto-discover is switched with true or false.")
        with self._lock:
            if on == self._discovery.on:
                return
            work = self._discovery.round + 1
            self._discovery = Discovery(on, "looking" if on else "off", "", work)
            self._discovered_names = {}
            finder = self._new_finder_round_locked(self._search.names, ())
        log.info("Auto-discover %s", "on" if on else "off")
        self._prepare_in_background(finder)
        if on:
            threading.Thread(target=self._look_now_and_then, args=(work,), name="lookout", daemon=True).start()

    def watch(self, names: Sequence[str]) -> None:
        """Watch for these things from now on (it replaces the list); [] stops watching.

        The names are checked as search() checks them, with at most
        watch.max_names; anything else raises ValueError with a plain message.
        An enrolled person's name is allowed (their face answers it) and shown as
        enrolled. Returns at once: the loop decides on each frame what is in view
        and what appeared (status()["watch"]). A name only the open detector can
        answer starts a finder round, as a search does.
        """
        wanted = clean_names(names, self.watch_most, TOO_MANY_WATCHED)
        with self._lock:
            self._watch = watch_of(wanted, self._enrolled_people())
            work = self._new_finder_round_locked(self._search.names, self._discovered_list_locked())
        log.info("Watching for %d thing(s)", len(wanted))     # the names themselves stay off the log
        self._prepare_in_background(work)

    def sightings(self) -> SightingLog | None:
        """The camera's memory, for the web routes; None when the memory is off."""
        return self._sightings

    def target(self, x: float | None = None, y: float | None = None,
               track_id: int | None = None, label: str | None = None) -> Target | None:
        """What a click points at, with a photo cut from the clean frame; None when it is gone.

        x and y are the click as fractions of the picture (0 to 1 from the
        top-left corner). With a track_id the point is not used. Called from
        the web server's threads while the loop runs. The photo of an object or
        an area may go to the assistant, so every face in it is covered first
        (see cover_faces); the page shows that same covered photo.
        """
        with self._lock:
            view = self._view                   # the view is never changed, only replaced
        if view is None:
            return None
        found = find_target(view, x, y, track_id, label, self._person_card)
        if found is None:
            return None
        target, box = found
        frame = view.frame if target.kind == "person" else cover_faces(view.frame, [face.box for face in view.faces])
        photo = self._encode(cut_out(frame, photo_box(box, view.size)))
        return replace(target, photo=photo, epoch=view.epoch)

    def scene_photo(self, max_side: int = 960) -> bytes | None:
        """The newest clean frame as JPEG, its longer side at most max_side (never enlarged).

        None before the first frame and while the view is cleared after a camera switch.
        """
        if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 1:
            raise ValueError("max_side must be a whole number of pixels, 1 or more")
        with self._lock:
            view = self._view                   # the view is never changed, only replaced
        if view is None:
            return None
        return self._encode(shrunk(view.frame, max_side))

    # ---- people: enrolling again and taking photos --------------------------------

    def enrolled_ids(self) -> set[str]:
        """The ids of everyone the recogniser has saved faces for; empty with recognition off."""
        return self._recognizer.enrolled_ids() if self._recognizer is not None else set()

    def reload_people(self) -> list[str]:
        """Enrol every person folder again with the loaded face model, and use the result at once.

        Saves embeddings.npz, swaps the people and their faces into the
        recogniser, and makes the loop forget the faces it keeps on screen.
        Returns enrolment's notes (people without photos, skipped photos, ...).
        Call it off the loop thread: it takes as long as reading every photo,
        while the video goes on. RuntimeError when face recognition is off.
        """
        recognizer = self._recognizer
        if recognizer is None:
            raise RuntimeError(self._recognition_off)
        with self._enrol_lock:
            enrolment = encode_people(self._photos_dir, lambda: recognizer.face_model)
            notes = enrolment.notes + save_enrolment(enrolment, self._photos_dir, self._embeddings_path)
            matcher = FaceMatcher(enrolment.ids, enrolment.vectors, self._match_threshold)
            recognizer.reload_known(enrolment.people, matcher)
            self._faces_changed.set()
            self._update_enrolment_notes(enrolment.people, matcher)
            self._search_with_new_people()
        log.info("Enrolled again: %d face vectors for %d people",
                 len(enrolment.ids), len(matcher.person_ids))
        return notes

    def start_photos(self, person_id: str, count: int = 5) -> None:
        """Begin taking `count` photos of an existing person from the live camera.

        KeyError for an unknown person. ValueError with a plain message when
        face recognition is off, the camera is turned off, another session is running,
        or count is not 3 to 10.
        """
        if self._recognizer is None:
            raise ValueError(f"{self._recognition_off} Photos cannot be taken without it.")
        if isinstance(count, bool) or not isinstance(count, int) or not FEWEST_PHOTOS <= count <= MOST_PHOTOS:
            raise ValueError(f"Take {FEWEST_PHOTOS} to {MOST_PHOTOS} photos.")
        folder = person_folder(self._photos_dir, person_id)
        with self._lock:
            if not self._camera_on:
                raise ValueError("Turn the camera on first to take photos.")
            if self._photos is not None and self._photos.running:
                raise ValueError("Photos are being taken already. Wait until they are done, or cancel them.")
            self._photos = PhotoSession(person_id, folder, count, time.monotonic(), self._generation)
        log.info("Taking %d photos of %s", count, person_id)

    def photo_progress(self) -> dict | None:
        """How the newest photo session goes; None when no session ran yet.

        {"person_id", "taken", "wanted", "state", "hint", "notes"}, see PhotoSession.
        """
        self._end_photos_if_late(time.monotonic())
        with self._lock:
            return self._photos.progress() if self._photos is not None else None

    def cancel_photos(self) -> None:
        """End a session that is taking photos as failed, with the hint Cancelled.

        The photos taken so far stay in the folder. Once enrolment has begun it
        cannot be stopped halfway, so a session that is enrolling goes on.
        """
        with self._lock:
            cancelled = self._cancel_photos_locked(HINT_CANCELLED)
        if cancelled is not None:
            log.info("Photos of %s cancelled after %d", cancelled.person_id, cancelled.taken)

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
            self._recognition_off = f"Face recognition is off: {first_line(error)}"
            self._notes.append(self._recognition_off)
            return None
        if not matcher.person_ids:
            self._notes.append(NOBODY_ENROLLED)
        elif not people:
            self._notes.append(NO_INFO_CARDS)
        return recognizer

    def _build_sightings(self, file: str | None, keep_days: float, min_seconds: float) -> SightingLog | None:
        """The camera's memory, or None when memory.file is empty or the log cannot start.

        The sighting log carries on by itself when its file breaks later; this
        only covers a log that cannot even be made. The video runs either way.
        """
        if not file:
            return None
        try:
            from car_vision.memory import SightingLog     # standard library only: cheap to import
            return SightingLog(resolve_path(file), keep_days=keep_days, min_seconds=min_seconds)
        except Exception as error:          # a locked or broken file must not stop the video
            log.warning("The camera's memory is off: %s", first_line(error))
            self._notes.append(f"The camera's memory is off: {first_line(error)}")
            return None

    # ---- looking for things by name (a thread of its own) ----------------------
    #
    # Each new list of names for the open detector (a search(), or a change in
    # the discovered names) starts a new round. Its thread waits for the open
    # detector (self._open_lock), checks that its round is still the newest
    # before each slow step, and gives up as soon as it is not. Only the newest
    # round may mark the search ready or failed; stop() starts a round of its
    # own, so nothing is marked ready after it.

    def _new_finder_round_locked(self, own: tuple[str, ...], discovered: Sequence[str],
                                 always: bool = False) -> int | None:
        """Start a new round when the open detector's names change (always, for search()).

        Call with self._lock held. The finds of names no longer looked for
        vanish at once. Returns the round to prepare in the background, or
        None when there is nothing to prepare.
        """
        old = self._search
        # Watched names come after the own ones and before the discovered ones, 6 in all (M25).
        watched = names_after(own, self._watch.finder_names)
        # The order of names already given is kept, so the open detector is not set again
        # (and its tracks restarted) just because the lookout named things in another order.
        extra = in_earlier_order(names_after(own + watched, discovered), old.discovered)
        people = people_among(own, self._enrolled_people())
        if not always and (own, watched, extra, people) == (old.names, old.watched, old.discovered, old.people):
            return None
        new = Search(own, "off", "", old.round + 1, extra, people=people, watched=watched)
        keep_text = not always and own == old.names and len(old.shown) == len(own)
        new = replace(new, shown=old.shown if keep_text else tuple(text_of(new, name) for name in own))
        objects = new.object_names
        # People's names alone need no open detector: such a search is ready at once.
        own_ready = bool(own) and (not objects or (objects == old.object_names and (
            old.state == "ready" or (old.state == "loading" and old.own_ready))))
        watch_ready = bool(watched) and watched == old.watched and (
            old.state == "ready" or (old.state == "loading" and old.watch_ready))
        finder = new.finder_names
        state = "loading" if finder else ("ready" if own else "off")
        self._search = replace(new, state=state, own_ready=own_ready, watch_ready=watch_ready)
        self._drop_finds_not_wanted_locked(finder)
        return self._search.round if finder else None

    def _enrolled_people(self) -> dict[str, str]:
        """Everyone the recogniser knows by face: person id -> their name as enrolled.
        Empty with recognition off."""
        if self._recognizer is None:
            return {}
        people = {}
        for person_id in sorted(self._recognizer.enrolled_ids()):
            card = self._recognizer.person(person_id)
            people[person_id] = card.name if card is not None else person_id   # as the video shows them
        return people

    def _search_with_new_people(self) -> None:
        """After enrolling: the next frame answers the own names with the new list of people."""
        with self._lock:
            objects = self._search.object_names
            self._watch = with_new_people(self._watch, self._enrolled_people())    # a watched name may be a person now
            work = self._new_finder_round_locked(self._search.names, self._search.discovered)
            if self._search.object_names != objects:
                self._looked_for = None         # a name went back to the finder: the page waits until it looked
        self._prepare_in_background(work)

    def _prepare_in_background(self, work: int | None) -> None:
        """Start the thread that gives the open detector the names of round `work`."""
        if work is None:
            return
        with self._lock:
            names = self._search.finder_names if self._search.round == work else ()
        if names:
            threading.Thread(target=self._prepare_search, args=(names, work),
                             name="open-detector", daemon=True).start()

    def _drop_finds_not_wanted_locked(self, names: Sequence[str]) -> None:
        """Take the finds of names no longer looked for out of the view and the WorldState at once,
        also when no new frame comes. Call with self._lock held."""
        unwanted = [find for find in self._found + self._discovered if not label_among(find.label, names)]
        if not unwanted:
            return
        def kept(detections: Sequence[Detection]) -> tuple[Detection, ...]:
            return tuple(d for d in detections if not any(d is find for find in unwanted))
        self._found, self._discovered = kept(self._found), kept(self._discovered)
        if self._view is not None:
            self._view = replace(self._view, detections=kept(self._view.detections))
        snapshot = self._state.snapshot()
        self._state.update_detections(list(kept(snapshot.detections)), snapshot.frame_size,
                                      snapshot.detections_at)    # the same moment: nothing new was seen

    def _prepare_search(self, names: tuple[str, ...], work: int) -> None:
        """Load the open detector if needed, give it the names and warm it up; then the search is ready."""
        with self._open_lock:
            if not self._may_prepare(work):
                return
            try:
                detector = self._open_detector_ready()
                if not self._may_prepare(work):
                    return
                before = tuple(detector.names)
                detector.set_names(list(names))
                changed = tuple(detector.names) != before
                if changed:
                    # Its first run after new names is slow: done here, not on the video loop.
                    detector.warm_up(self._warm_up_size())
            except Exception as error:      # no internet on the first use, a broken model, ...
                log.warning("Looking for things failed (%s: %s)", type(error).__name__, first_line(error))
                self._search_failed(work, first_line(error))
                return
            with self._lock:
                if changed:
                    self._epoch += 1        # its track numbers started again
                    self._named_epoch += 1  # ... only the named ones
                if self._search.round == work:
                    self._search = replace(self._search, state="ready")
        log.info("Looking for things: ready")

    def _warm_up_size(self) -> tuple[int, int]:
        """The size of the frames the open detector will see: the last one's, else the config box."""
        width, height = self._state.snapshot().frame_size
        return (width, height) if width > 0 and height > 0 else self._config_box

    def _open_detector_ready(self) -> Any:
        """The open detector, built and loaded. Call with self._open_lock held: it is slow the first time."""
        if self._open_detector is None:
            from car_vision.detector import OpenDetector     # built on the first search, never at start-up
            self._open_detector = OpenDetector(self._detector_settings)
        if not self._open_detector.loaded:
            self._open_detector.load()
        return self._open_detector

    def _may_prepare(self, work: int) -> bool:
        """True when round `work` is the newest and the camera is on: no model runs while it is off.
        A round that gives up because the camera is off stays "loading" (see _resume_search)."""
        with self._lock:
            return self._search.round == work and self._camera_on

    def _resume_search(self) -> None:
        """After the camera is turned on: prepare the names that were asked for while it was off."""
        with self._lock:
            waiting = self._search.round if self._search.state == "loading" else None
        self._prepare_in_background(waiting)

    def _search_failed(self, work: int, note: str) -> None:
        """Mark the search as failed with this plain reason, if it is still the newest round."""
        with self._lock:
            if self._search.round == work:
                self._search = replace(self._search, state="failed", note=note)
                self._found = self._discovered = ()

    # ---- auto-discover (a thread of its own) -----------------------------------
    #
    # discover(True) starts a look thread for a new discovery round. It wakes
    # every lookout.every_seconds, asks the lookout about the newest clean
    # frame, and gives way at once to stop() or a newer round, also while it
    # waits. The request itself (up to 20 s) runs with no pipeline lock held;
    # its answer is thrown away when the round or the camera changed meanwhile.
    # Only one thread talks to the lookout at a time (self._look_lock).

    def _look_now_and_then(self, work: int) -> None:
        """The look thread of discovery round `work`: one look, then one every every_seconds."""
        failures = 0
        with self._look_lock:                   # an older round's request finishes first
            while self._is_discovering(work):
                failures = self._look_once(work, failures)
                if failures is None:
                    return                      # a lasting problem: paused until off and on again
                wait = FAILURE_WAIT_SECONDS if failures >= FAILURES_BEFORE_WAIT else self.look_every_seconds
                if not self._wait_while_discovering(work, wait):
                    return

    def _look_once(self, work: int, failures: int) -> int | None:
        """One look. Returns the failures in a row afterwards; None after a lasting problem.

        The open detector gets a new list at most once per look: the new names
        and the forgotten old ones are passed on together.
        """
        picture = self._picture_to_look_at()
        if picture is None:
            self._take_discovered(work, None, (), time.monotonic())    # no frame yet: only forget old names
            return failures
        jpeg, generation = picture
        try:
            names = self._lookout_ready().names(jpeg)
        except Exception as error:              # a LookoutError has a plain message and a kind
            failures_now = self._look_failed(work, error, failures)
            if failures_now is not None:
                self._take_discovered(work, None, (), time.monotonic())
            return failures_now
        self._take_discovered(work, generation, names, time.monotonic())
        return 0

    def _lookout_ready(self) -> Any:
        """The lookout, built on the first look (never at start-up)."""
        if self._lookout is None:
            from car_vision.lookout import Lookout     # reads .env only when it looks
            self._lookout = Lookout(self._lookout_settings)
        return self._lookout

    def _picture_to_look_at(self) -> tuple[bytes, int] | None:
        """The newest clean frame as a JPEG picture_width wide (no boxes), with its camera generation."""
        with self._lock:
            view, generation = self._view, self._generation
        if view is None:
            return None
        return self._encode(shrunk_to_width(view.frame, self.look_picture_width)), generation

    def _look_failed(self, work: int, error: Exception, failures: int) -> int | None:
        """Pause with the lookout's plain message. Returns the failures in a row; None when it lasts."""
        kind = getattr(error, "kind", None)
        if kind == NOT_A_FAILURE:
            return failures                     # the previous look is still under way: skip this turn
        note = str(error) if kind is not None else LOOK_FAILED
        if kind is None:
            log.warning("Auto-discover: a look failed (%s)", type(error).__name__)
        lasting = kind in LASTING_PROBLEMS
        with self._lock:
            if self._discovery.round != work:
                return None
            self._discovery = replace(self._discovery, state="paused", note=note)
            if lasting:                         # nothing more is looked at: forget what was
                self._discovered_names = {}
                finder = self._new_finder_round_locked(self._search.names, ())
        if lasting:
            self._prepare_in_background(finder)
            return None
        return failures + 1

    def _take_discovered(self, work: int, generation: int | None, names: Sequence[str], now: float) -> None:
        """After a look: forget the names not mentioned for DISCOVERED_KEEP_SECONDS, keep the new
        ones, and give the open detector the new list when it changed.

        `generation` is the camera the picture came from (None: no answer this
        time). The answer about a camera that was replaced meanwhile is dropped.
        """
        with self._lock:
            if self._discovery.round != work:
                return
            self._discovered_names = {name: seen for name, seen in self._discovered_names.items()
                                      if now - seen[0] <= DISCOVERED_KEEP_SECONDS}
            if generation is not None and generation == self._generation:
                self._discovery = replace(self._discovery, state="looking", note="")
                for place, name in enumerate(discoverable(names)):
                    self._discovered_names[name] = (now, place)
            finder = self._new_finder_round_locked(self._search.names, self._discovered_list_locked())
        self._prepare_in_background(finder)

    def _discovered_list_locked(self) -> list[str]:
        """The discovered names, the most recently mentioned first. Call with self._lock held."""
        order = sorted(self._discovered_names.items(), key=lambda item: (-item[1][0], item[1][1]))
        return [name for name, _seen in order]

    def _is_discovering(self, work: int) -> bool:
        with self._lock:
            return self._discovery.round == work and not self._stop.is_set()

    def _wait_while_discovering(self, work: int, seconds: float) -> bool:
        """Wait, but give way at once to stop() or a newer round. True when still discovering."""
        deadline = time.monotonic() + seconds
        while self._is_discovering(work):
            left = deadline - time.monotonic()
            if left <= 0:
                return True
            self._stop.wait(min(left, PAUSE_STEP_SECONDS))
        return False

    def _discover_status(self, discovery: Discovery, names: list[str], search: Search) -> dict[str, Any]:
        """status()["discover"]. A failed open detector for discovered names pauses it too."""
        state, note = discovery.state, discovery.note
        if discovery.on and state == "looking" and search.state == "failed" and search.discovered:
            state, note = "paused", search.note
        lookout = self._lookout
        return {"on": discovery.on, "state": state, "names": names if discovery.on else [], "note": note,
                "looks": int(lookout.looks_today) if lookout is not None else 0,
                "spent_usd": round(float(lookout.spent_today), 6) if lookout is not None else 0.0,
                "budget_usd": self.look_budget}

    # ---- the phone's picture size (a short thread of its own) ------------------
    #
    # Each call of _check_phone_picture() starts a new "round". The thread of an
    # older round checks that it is still the newest before every request to the
    # phone and while it waits, and stops as soon as it is not. Only one thread
    # talks to the same phone at a time; a phone that does not answer never holds
    # up a round for another phone.
    #
    # The stream is opened again when the phone sends another size than it did
    # when the stream was opened (see _open_stream_again). That is decided from
    # what the phone says, not from which round asked, so a change asked for by
    # a round that a newer one replaced is still followed by a reopen.

    def _check_phone_picture(self) -> None:
        """Read the phone's picture sizes and ask for the chosen one, off the video loop.

        Does nothing unless the running source is the phone (never while the camera is off).
        """
        with self._lock:
            self._picture_round += 1
            work = self._picture_round
            live = self._started and not self._stop.is_set() and self._camera_on
            url = self._phone_url if self._source.source == "phone" else None
        if live and url is not None:
            threading.Thread(target=self._tune_phone_picture, args=(url, work),
                             name="phone-picture", daemon=True).start()

    def _tune_phone_picture(self, url: str, work: int) -> None:
        """Read the sizes, ask for the chosen one when it differs, then read the size back.

        Every step gives up quietly: a phone that is off or another camera app
        just leaves status()["phone_picture"] at None.
        """
        with self._phone_lock(url):
            if not self._is_newest(work):
                return
            current, offered = read_picture_sizes(url, self._fetch)
            picture = self._keep_picture(work, current, offered) if current is not None else None
            if picture is None:
                return
            self._open_stream_again(work, current)    # a change an older round asked for
            wanted = self._wanted_size(current, list(picture.offered))
            if wanted is None or not self._may_ask(work):
                return
            log.info("Asking the phone for %s pictures instead of %s", size_text(wanted), size_text(current))
            if ask_for_size(url, wanted, self._fetch):
                now_sent = self._read_back(url, work, wanted, picture.offered)
                if now_sent is not None:
                    self._open_stream_again(work, now_sent)

    def _read_back(self, url: str, work: int, wanted: Size, offered: Sequence[Size]) -> Size | None:
        """Keep what the phone says it sends after the change; once more a moment later if it lags.
        Returns the size the phone said it sends last; None when it stopped answering or a
        newer round began.

        Only the current size is taken from these answers. The list of offered
        sizes stays the one read before the change: while its camera restarts,
        the real phone lists just one size for a moment.
        """
        current = None
        for attempt in range(READ_BACK_TRIES):
            if attempt > 0 and not self._pause(work, READ_BACK_WAIT_SECONDS):
                return None
            if not self._is_newest(work):
                return None
            current, _listed_during_restart = read_picture_sizes(url, self._fetch)
            if current is None or self._keep_picture(work, current, offered) is None:
                return None
            if current == wanted:
                break
        return current

    def _open_stream_again(self, work: int, now_sent: Size) -> None:
        """Read the phone's stream afresh when the phone now sends another size than it did
        when the stream was opened. The first size heard from a phone is taken as that size.

        Seen on the real phone: the stream reader fixes the frame size when it
        opens the stream, so without this the frames stay the old size however
        large the phone's pictures are now. The short wait lets the phone send
        its first pictures in the new size before the stream is opened again.
        """
        with self._lock:
            if work != self._picture_round or self._stream_size == now_sent:
                return
            if self._stream_size is None:
                self._stream_size = now_sent
                return
        if not self._pause(work, STREAM_SETTLE_SECONDS):
            return                              # the newer round decides again from what it hears
        with self._lock:
            if work != self._picture_round or self._source.source != "phone":
                return
            source = self._source
            self._stream_size = now_sent
        source.reopen()

    def _phone_lock(self, url: str) -> threading.Lock:
        """The lock that keeps the talk with the phone at this address to one thread at a time."""
        with self._lock:
            return self._phone_locks.setdefault(url, threading.Lock())

    def _pause(self, work: int, seconds: float) -> bool:
        """Wait, but give way at once to a newer round or to stop(). True when still the newest."""
        deadline = time.monotonic() + seconds
        while not self._stop.is_set() and self._is_newest(work):
            left = deadline - time.monotonic()
            if left <= 0:
                return True
            self._stop.wait(min(left, PAUSE_STEP_SECONDS))
        return False

    def _may_ask(self, work: int) -> bool:
        """True when this round may ask the phone for a size now, and counts the ask.

        A phone that changes its size back each time would otherwise be asked
        for ever: after 3 asks within a minute it is left as it is, with one
        warning, until the next Connect or another choice.
        """
        now = time.monotonic()
        with self._lock:
            if work != self._picture_round or self._gave_up_asking:
                return False
            self._asked_at = [at for at in self._asked_at if now - at < ASK_WINDOW_SECONDS]
            if len(self._asked_at) >= MAX_ASKS:
                self._gave_up_asking = True
                log.warning(TOO_MANY_ASKS.format(MAX_ASKS))
                return False
            self._asked_at.append(now)
            return True

    def _forget_asks(self) -> None:
        """Start counting asks afresh (a new phone or a new choice). Call with self._lock held."""
        self._asked_at = []
        self._gave_up_asking = False

    def _wanted_size(self, current: Size, offered: list[Size]) -> Size | None:
        """The size to ask the phone for under the present choice; None to leave it as it is."""
        with self._lock:
            choice = self._picture_choice
        if choice == "keep":
            return None
        if choice == "auto":
            return best_size(offered, current, *self._config_box)
        size = parse_size(choice)
        if size not in offered:
            log.warning("stream.phone_picture is %s, but the phone does not offer that size; "
                        "the phone is left as it is", choice)
            return None
        return None if size == current else size

    def _is_newest(self, work: int) -> bool:
        with self._lock:
            return work == self._picture_round

    def _keep_picture(self, work: int, current: Size, offered: Sequence[Size]) -> PhonePicture | None:
        """Keep what the phone said and return it, unless a newer round began meanwhile (then None).

        A shorter list than this phone gave before is not kept (see
        offered_to_keep). Now that the offered sizes are known, a chosen size can
        take effect: the phone source gets its box here.
        """
        with self._lock:
            if work != self._picture_round:
                return None
            self._picture = PhonePicture(current, offered_to_keep(self._picture, offered))
            self._set_phone_box()
            return self._picture

    def _set_phone_box(self) -> None:
        """Give the phone source the box for the present choice. Call with self._lock held.

        Holding the lock means two threads can never leave an older box behind.
        A clip or the webcam is left alone: it keeps the box from config.yaml.
        """
        if self._source.source != "phone":
            return
        box = shown_box(self._picture_choice, self._picture, self._config_box)
        self._source.set_box(*box)

    # ---- the photo session (the loop takes the photos) ---------------------------------
    #
    # The session is one PhotoSession object, changed only under self._lock. The
    # loop decides and saves a photo outside the lock, then counts it under the
    # lock if the session is still the same one and still taking photos. The
    # timeout is checked on every turn of the loop, also while no frames come
    # (a phone that dropped out), and by photo_progress().

    def _take_photo_if_ready(self, frame: np.ndarray, boxes: Sequence[Box], now: float,
                             generation: int) -> None:
        """On a face-check frame: save the clean frame as the next photo when every rule holds."""
        with self._lock:
            session = self._photos
            if session is None or session.state != "taking" or session.generation != generation:
                return
            last_photo_at = session.last_photo_at
        hint = photo_hint(frame, boxes)
        ready = not hint and now - last_photo_at >= PHOTO_GAP_SECONDS
        if ready:
            try:
                save_photo(session.folder, frame)
            except (OSError, RuntimeError) as error:
                self._photos_failed(session, f"Could not save the photo: {first_line(error)}")
                return
        with self._lock:
            if self._photos is not session or session.state != "taking":
                return                          # cancelled meanwhile; a saved photo stays in the folder
            if ready:
                session.taken += 1
                session.last_photo_at = now
            taken = session.taken
            # When nothing is wrong, the hint asks for the pose of the next photo: five
            # photos of the very same pose teach the recogniser less than five different ones.
            session.hint = hint or pose_for(taken)
            enough = taken >= session.wanted
            if enough:
                session.state = "enrolling"
        if ready:
            log.info("Photo %d of %d of %s saved", taken, session.wanted, session.person_id)
        if enough:
            self._enrol_in_background(session)

    def _end_photos_if_late(self, now: float) -> None:
        """After PHOTO_TIMEOUT_SECONDS: enrol with 3 or more photos, else end the session as failed."""
        with self._lock:
            session = self._photos
            if session is None or session.state != "taking" or now - session.started_at < PHOTO_TIMEOUT_SECONDS:
                return
            seconds = round(PHOTO_TIMEOUT_SECONDS)
            enough = session.taken >= FEWEST_PHOTOS
            if enough:
                session.state, session.hint = "enrolling", ""
                session.notes = [f"Only {session.taken} of {session.wanted} photos were taken "
                                 f"within {seconds} seconds."]
            else:
                session.state = "failed"
                session.hint = (f"Only {session.taken} clear photo(s) in {seconds} seconds; at least "
                                f"{FEWEST_PHOTOS} are needed. Try again with more light, facing the camera.")
        log.info("Photos of %s: time is up with %d of %d", session.person_id, session.taken, session.wanted)
        if enough:
            self._enrol_in_background(session)

    def _enrol_in_background(self, session: PhotoSession) -> None:
        threading.Thread(target=self._enrol_for, args=(session,), name="enrol", daemon=True).start()

    def _enrol_for(self, session: PhotoSession) -> None:
        """Enrol everyone after a photo session, then end the session as done (or failed)."""
        try:
            notes = self.reload_people()
        except Exception as error:              # a full disk or a broken model must not end the program
            log.warning("Enrolling after the photos of %s failed (%s)", session.person_id, type(error).__name__)
            self._photos_failed(session, f"Enrolling failed: {first_line(error)}", state="enrolling")
            return
        with self._lock:
            if session.state == "enrolling":
                session.state, session.hint = "done", ""
                session.notes = session.notes + notes
        log.info("Photos of %s: %d taken and enrolled", session.person_id, session.taken)

    def _photos_failed(self, session: PhotoSession, hint: str, state: str = "taking") -> None:
        """End the session as failed with this hint, if it is still in `state`."""
        with self._lock:
            if session.state == state:
                session.state, session.hint = "failed", hint

    def _cancel_photos_locked(self, hint: str) -> PhotoSession | None:
        """End a session that is taking photos as failed. Call with self._lock held.
        Returns the session when one was cancelled."""
        session = self._photos
        if session is None or session.state != "taking":
            return None
        session.state, session.hint = "failed", hint
        return session

    def _update_enrolment_notes(self, people: dict[str, Person], matcher: FaceMatcher) -> None:
        """Bring the start-up notes about enrolment up to date after enrolling again."""
        notes = [note for note in self._notes if note not in (NOBODY_ENROLLED, NO_INFO_CARDS)]
        if not matcher.person_ids:
            notes.append(NOBODY_ENROLLED)
        elif not people:
            notes.append(NO_INFO_CARDS)
        with self._lock:
            self._notes = notes

    # ---- the loop thread -----------------------------------------------------

    def _start_loop(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def _forget_old_picture(self) -> None:
        """What use_phone() and set_camera() clear, under the lock: a click must never hit the old camera's
        picture, and the assistant must not be told about people that camera saw."""
        self._generation += 1
        self._switched_at = time.monotonic()
        # The new phone has not said yet. Its source starts with the config box,
        # and gets the chosen box once it says it offers that size (_keep_picture).
        self._picture = None
        self._picture_round += 1                # what the old phone still says is ignored
        self._stream_size = None                # the new stream's size is learned from the new phone
        self._forget_asks()
        self._epoch += 1                        # track numbers of the new camera start again
        self._everyday_epoch += 1
        self._view = None
        self._found = self._discovered = ()
        self._watch_seen = {}                   # the new picture is watched afresh (the list is kept)
        self._shown_codes = ((), (0, 0))
        self._looked_for = None                 # the finder has not looked at the new camera yet
        self._error = None
        self._on_screen = []
        frame_size = self._state.snapshot().frame_size
        self._state.update_detections([], frame_size, self._switched_at)
        self._state.update_faces([], self._switched_at)

    def _run(self) -> None:
        generation = None                       # the camera this loop is working with
        last_sequence = 0
        while not self._stop.is_set():
            self._end_photos_if_late(time.monotonic())   # also while no frames come
            self._tick_memory()                          # so is the memory's writing
            with self._lock:                    # the source, its generation and the search, read together
                source, current, search = self._source, self._generation, self._search
            if current != generation:
                generation, last_sequence = current, 0
                self._start_afresh(source, current)
            frame, sequence = source.read()
            if frame is None or sequence == last_sequence:
                if not source.connected:
                    self._show_stream_lost(source)
                self._stop.wait(IDLE_WAIT_SECONDS)
                continue
            last_sequence = sequence
            try:
                self._process(frame, generation, search)
            except Exception as error:      # one bad frame must not end the demo
                if self._too_many_failures(error, generation):
                    return
            else:
                self._failures = 0

    def _start_afresh(self, source: Any, generation: int) -> None:
        """A new camera: forget the faces, failures, frame count, FPS and lost line of the old one.

        Only the loop thread uses these, so the loop clears them itself. When the
        kept picture belongs to another camera generation (a switch, also one that
        restarted a loop stopped after too many failures), it is replaced at once by
        a plain dark one of the same size with "Connecting to the phone ..." (or
        "Camera off"), so nothing of the old camera stays on screen.
        """
        self._frames = 0
        self._failures = 0
        self._memory = FaceMemory()
        self._fps = FpsCounter()
        self._lost_line = None
        self._frame_size = None
        self._forget_named_finds()
        self._forget_codes()
        self._watch_afresh()
        if self._last_drawn is not None and self._drawn_from != generation:
            self._last_drawn = np.full_like(self._last_drawn, DARK_LEVEL)
            self._drawn_from = generation
            self._show_stream_lost(source)          # no frame yet, so: "Connecting to the phone ..."

    def _process(self, frame: np.ndarray, generation: int, search: Search) -> None:
        """See, decide and show for one frame. Nothing is written until the models are done,
        and nothing at all when the camera was switched while the models worked, or when
        the names searched for changed while this frame's named finds were worked out."""
        now = time.monotonic()
        index = self._frames
        self._frames += 1

        resized = self._note_frame_size(frame)      # before detect(): the detector restarts its tracks too
        enrolled_again = self._faces_changed.is_set()
        if enrolled_again:                          # the remembered names may be out of date
            self._faces_changed.clear()
            self._memory = FaceMemory()
        detections = self._detector.detect(frame)
        if self._camera_changed_since(generation):
            return                                  # switched or turned off meanwhile: no more models for it
        # The faces are matched to the everyday person boxes, as before.
        if self._recognizer is not None and (index % self.every_n_frames == 0 or resized or enrolled_again):
            check = self._recognizer.check_faces(frame, detections)
            self._memory.remember(check.faces, detections, now)
            self._take_photo_if_ready(frame, check.boxes, now, generation)
        faces = self._memory.current(detections, now)
        shown, named, finder_ran = self._add_named_finds(frame, detections, faces, search, now)
        looked = finder_ran or (bool(search.names) and not search.object_names)   # people's names: the faces answer
        # The visitor's own finds stand out in yellow; discovered (and watched) ones are drawn like everyday ones.
        found = tuple(find for find in named if label_among(find.label, search.names))
        discovered = tuple(find for find in named if label_among(find.label, search.discovered))
        maybes = tuple(find for find in found if not is_sure(find, self.sure_confidence, self._sure_tracks))
        on_people = tuple(find for find in named if self._guard.on_a_person(find, now))
        codes = self._codes_to_show(frame, index, now)

        # The boxes are drawn on a copy: the camera's own frame goes into the view untouched.
        drawn = frame.copy()
        draw.draw_detections(drawn, shown, wanted=wanted_tracks(found, maybes, shown, faces, search),
                             unsure=maybes)
        draw.draw_faces(drawn, faces)
        draw.draw_codes(drawn, codes)
        jpeg = self._encode(drawn)
        height, width = frame.shape[:2]
        with self._lock:                        # state, picture, view and reactions change together
            if self._generation != generation:
                return                          # this frame came from the camera used before the switch
            current = self._search
            if any(not label_among(find.label, current.finder_names) for find in named) or (
                    found and current.names != search.names):
                return                          # finds of names replaced while this frame was worked on
            self._state.update_detections(shown, (width, height), now)
            self._state.update_faces(faces, now)
            on_screen = self._reactions.update(self._state.snapshot(), now)
            self._jpeg = jpeg
            epoch = self._epoch
            memory_epochs = (self._everyday_epoch, self._named_epoch)
            self._view = View(frame, tuple(shown), tuple(faces), epoch, on_people, maybes)
            self._found, self._discovered = found, discovered
            self._shown_codes = (codes, (width, height))
            if looked and current.names == search.names:
                self._looked_for = search.names     # the page may answer now: the finder has looked
            self._on_screen = list(on_screen)
            phone_not_heard = self._picture is None
        self._remember(frame, shown, faces, named, memory_epochs, untold=maybes + on_people,
                       people=self._guard.recent_boxes(now))
        self._watch_for_things(detections, faces, named, maybes, search, finder_ran, generation, width, now)
        self._last_drawn = drawn
        self._drawn_from = generation
        came_back = self._lost_line is not None and index > 0    # this camera had dropped out and returned
        self._lost_line = None
        self._fps.tick(now)
        if came_back:
            # Seen on the real phone: when its camera app restarts, it sends its own
            # picture size again (1920 x 1080), so the chosen size is asked for once more.
            # Its stream was opened again by the reader, at the size the phone sends now.
            self._forget_stream_size()
            self._check_phone_picture()
        elif index == 0 and phone_not_heard:
            # The first picture from a phone that has not said anything about its sizes
            # yet: its camera app was not serving when it was asked at start-up or at
            # Connect. Now it is, so ask again. (Does nothing for a clip or the webcam.)
            self._check_phone_picture()

    def _camera_changed_since(self, generation: int) -> bool:
        """True when the camera was switched or turned off after this generation's frame was read."""
        with self._lock:
            return self._generation != generation

    def _forget_stream_size(self) -> None:
        """The reader opened the stream again by itself: its size is learned afresh from the phone."""
        with self._lock:
            self._stream_size = None

    def _note_frame_size(self, frame: np.ndarray) -> bool:
        """True when this frame is not the size of the one before from this camera.

        That happens when another picture size is chosen or the phone changes
        its own. The remembered faces are then forgotten, because their boxes are
        in the old size's pixels; the caller checks the faces again at once.
        The detector starts its track numbers again by itself, so a new track
        epoch begins: a track number from before means something else now.
        """
        height, width = frame.shape[:2]
        resized = self._frame_size is not None and (width, height) != self._frame_size
        self._frame_size = (width, height)
        if resized:
            self._memory = FaceMemory()
            self._forget_named_finds()              # their boxes are in the old size's pixels too
            self._forget_codes()                    # ... and so are the codes'
            self._watch_afresh()                    # what is in view at first in the new size did not appear
            with self._lock:
                self._epoch += 1
                self._everyday_epoch += 1
        return resized

    # ---- the named finds (the loop thread) -------------------------------------

    def _add_named_finds(self, frame: np.ndarray, detections: list[Detection], faces: Sequence[RecognizedFace],
                         search: Search, now: float) -> tuple[list[Detection], tuple[Detection, ...], bool]:
        """The frame's detections with the named finds added, the named finds alone, and
        whether the open detector ran on this frame.

        Every person box and every face box (known or unknown) counts as a person
        (see PersonGuard). With no names to look for, the detections themselves
        and no finds: nothing is copied.
        """
        self._guard.see(people_boxes(detections) + [face.box for face in faces], now)
        names = search.finder_names
        if not names:
            if self._named.finds:
                self._named = NamedFinds()
            return detections, (), False
        if self._named.search_round != search.round:
            self._named_tick = 0                    # a new list of names runs on its first ready frame
        if search.state == "loading" and search.own_ready:
            self._named = held_own_finds(self._named, search.names, now)
        fresh = self._detect_named(frame, search)
        shown, self._named = merge_named(detections, self._named, fresh, search.round, now, names,
                                         own=search.names, sure=self.sure_confidence, sure_tracks=self._sure_tracks,
                                         is_person=lambda find: self._guard.is_the_person(find, now))
        self._sure_tracks = sure_tracks_now(self._named.detections, self._sure_tracks, self.sure_confidence)
        self._guard.forget_ended(self._named.detections, now)
        return shown, self._named.detections, fresh is not None

    def _detect_named(self, frame: np.ndarray, search: Search) -> list[Detection] | None:
        """What the open detector finds on this frame; None when it is not due or is busy.

        Runs on every open_every_n_frames-th ready frame. When the background
        thread holds the open detector (loading it or setting names), the loop
        does not wait: it skips, and tries again on the next frame.
        """
        if search.state != "ready":
            return None
        if self._named_tick % self.open_every_n_frames != 0:
            self._named_tick += 1
            return None
        if not self._open_lock.acquire(blocking=False):
            return None
        try:
            if self._open_tracks_stale:             # a new camera or frame size: track numbers start again
                self._open_detector.reset_tracks()
                self._open_tracks_stale = False
            found = self._open_detector.detect(frame)
        except Exception as error:                  # a broken search must not stop the video
            log.warning("Looking for things failed on a frame (%s: %s)", type(error).__name__, first_line(error))
            self._search_failed(search.round, first_line(error))
            return None
        finally:
            self._open_lock.release()
        self._named_tick += 1
        return found

    def _forget_named_finds(self) -> None:
        """A new camera or frame size: the finds kept on screen and the open detector's tracks are old."""
        self._named = NamedFinds()
        self._guard = PersonGuard()                 # the boxes and tracks it knows are in the old pixels
        self._sure_tracks = set()
        self._named_tick = 0
        self._open_tracks_stale = True

    # ---- watching for things (the loop thread) ----
    #
    # watch() only swaps the list under the lock. Everything else is decided here, from each
    # frame's results: which watched things are in view, and which of them appeared (Watcher).
    # Each appearance becomes one event in status()["watch"] and one entry in the memory.
    # Nothing is spoken and no reaction is made of it: the page shows a quiet notice.

    def _watch_for_things(self, detections: Sequence[Detection], faces: Sequence[RecognizedFace],
                          named: Sequence[Detection], maybes: Sequence[Detection], search: Search,
                          finder_ran: bool, generation: int, width: int, now: float) -> None:
        """Decide for this frame which watched things are in view and which of them appeared."""
        with self._lock:
            watch = self._watch
        self._watcher.keep(watch.names)
        looked = self._finder_looked_for(search, finder_ran)
        people = people_in_view(watch.people, detections, faces)
        appeared = []
        for name in watch.names:
            judged, place = watched_place(name, watch, search, looked, detections, people, named, maybes, width)
            if judged and self._watcher.look(name, place is not None, now):
                appeared.append((name, place))
        self._publish_watch(watch, appeared, generation, now)

    def _finder_looked_for(self, search: Search, finder_ran: bool) -> set[str]:
        """The names (folded) the open detector has looked for in this search round and picture.

        A finder name is judged only after that: while the open detector loads or
        is given new names, its things are neither in view nor away.
        """
        if search.round != self._looked_round:
            self._looked_round, self._finder_looked = search.round, set()
        if finder_ran:
            self._finder_looked |= {name.casefold() for name in search.finder_names}
        return self._finder_looked

    def _publish_watch(self, watch: Watch, appeared: Sequence[tuple[str, str]], generation: int,
                       now: float) -> None:
        """Put what was decided into the status, and tell the memory about each event once."""
        wall_clock = time.time()
        events = []
        with self._lock:
            if self._generation != generation:
                return                              # another camera meanwhile: it is watched afresh
            self._watch_seen = {name: self._watcher.in_view(name, now)
                                for name in watch.names if self._watcher.judged(name)}
            for name, place in appeared:
                self._watch_count += 1
                events.append({"id": self._watch_count, "name": watch.text(name), "place": place,
                               "at": wall_clock, "person": watch.person(name) is not None})
                self._watch_last_at[name] = wall_clock
            self._watch_events = (self._watch_events + events)[-MOST_EVENTS:]
        for event in events:
            log.info("Watch event %d: a watched thing appeared", event["id"])   # its name stays off the log
            self._remember_appeared(event)

    def _remember_appeared(self, event: dict[str, Any]) -> None:
        """Tell the memory that a watched thing appeared (never with a picture). Errors are logged once."""
        if self._sightings is None:
            return
        label, name = ("person", event["name"]) if event["person"] else (event["name"], None)
        try:
            self._sightings.appeared(label, event["place"], event["at"], name)
        except Exception:
            self._memory_error("Telling the memory that a watched thing appeared failed")

    def _watch_afresh(self) -> None:
        """A new camera or frame size: what is in view at first did not appear (see Watcher)."""
        self._watcher.start_afresh()
        self._finder_looked = set()
        with self._lock:
            self._watch_seen = {}

    # ---- codes (the loop thread) -------------------------------------------------
    #
    # A code's text was written by a stranger: it goes to the page (status()["codes"])
    # and nowhere else. It is never logged, never put into the WorldState (so the
    # assistant never hears of it) and never told to the memory.

    def _codes_to_show(self, frame: np.ndarray, index: int, now: float) -> tuple[Code, ...]:
        """The codes to draw on this frame, read from the clean frame on every
        code_every_n_frames-th frame and kept for a while after their last read (see _code_keep_seconds)."""
        if self._code_reader is None:
            return ()
        if index % self.code_every_n_frames == 0:
            self._code_reads = (self._code_reads + [now])[-2:]
            for code in self._read_codes(frame):
                self._codes[(code.kind, code.text)] = (code, now)    # the newest box wins
        keep = self._code_keep_seconds()
        self._codes = {key: (code, read_at) for key, (code, read_at) in self._codes.items()
                       if now - read_at <= keep}
        return tuple(code for code, _read_at in self._codes.values())

    def _code_keep_seconds(self) -> float:
        """How long a code stays after its last read: code_hold_seconds, or twice the time
        between the reader's last two runs when that is longer.

        At a low frame rate the reader runs seldom (5 frames at 3 frames a second
        take 1.7 s); without this a code held still would vanish between two reads.
        """
        if len(self._code_reads) < 2:
            return self.code_hold_seconds
        return max(self.code_hold_seconds, 2 * (self._code_reads[1] - self._code_reads[0]))

    def _forget_codes(self) -> None:
        """A new camera or frame size: the codes kept on screen and the reader's rhythm are old."""
        self._codes = {}
        self._code_reads = []

    def _read_codes(self, frame: np.ndarray) -> list[Code]:
        """What the reader finds; [] when it raises (logged once, without any text: the video goes on)."""
        try:
            return list(self._code_reader.read(frame))
        except Exception as error:
            if not self._codes_failed:
                self._codes_failed = True
                log.warning("Reading the codes failed (%s); the video goes on without them", type(error).__name__)
            return []

    # ---- the memory (the loop thread) --------------------------------------------

    def _remember(self, frame: np.ndarray, detections: Sequence[Detection],
                  faces: Sequence[RecognizedFace], named: Sequence[Detection],
                  epochs: tuple[int, int], untold: Sequence[Detection] = (),
                  people: Sequence[Box] = ()) -> None:
        """Tell the sighting log about every tracked detection of this frame.

        `named` are the named finds among the detections; `epochs` is
        (everyday epoch, named epoch). See memory_key for why they are kept apart.
        `untold` are never told: the maybes, and the finds on a person (a person
        is never an object, so nothing on them gets a picture). A named find
        whose box touches any of the person and face boxes in `people` (a window
        behind someone) is remembered without a picture.
        A memory error is logged once and the video goes on.
        """
        sightings = self._sightings
        if sightings is None or not detections:
            return
        wall_clock, now = time.time(), time.monotonic()
        width = frame.shape[1]
        try:
            owners = face_owners(detections, faces)
            for index, detection in enumerate(detections):
                if detection.track_id is None:
                    continue                        # untracked: it cannot be followed, so not remembered
                if any(detection is find for find in untold):
                    continue
                is_named = any(detection is find for find in named)
                key = memory_key(detection, is_named, *epochs)
                person = detection.label == "person"
                no_picture = person or (is_named and touches_any(detection.box, people))
                face = owners.get(index)
                sightings.seen(key, "person" if person else "object", detection.label,
                               self._person_card(face).name if face is not None else None,
                               place_of(detection.box, width), float(detection.confidence), wall_clock,
                               None if no_picture else self._picture_for(key, detection.box, frame, now, faces))
            if self._pictures is not None:
                self._pictures.forget_old(now)
        except Exception:
            self._memory_error("Telling the memory what was seen failed")

    def _picture_for(self, key: Hashable, box: Box, frame: np.ndarray, now: float,
                     faces: Sequence[RecognizedFace] = ()) -> bytes | None:
        """A small JPEG of the thing for the sighting log, when one is due (see PictureTimer); else None.

        Every face of the view in it is covered with grey first (cover_faces), so a kept
        picture never shows a face: a phone held beside a cheek shows the phone only.
        """
        if self._pictures is None or box_area(box) == 0 or not self._pictures.due(key, now):
            return None
        covered = cover_faces(frame, [face.box for face in faces])
        return self._encode(shrunk(cut_out(covered, box), PICTURE_MAX_SIDE))

    def _tick_memory(self) -> None:
        """On every turn of the loop, also with nothing in view: the log's periodic work (writing)."""
        if self._sightings is None:
            return
        try:
            self._sightings.tick(time.time())
        except Exception:
            self._memory_error("The memory's periodic work failed")

    def _start_memory_summary(self) -> None:
        """Start the thread that puts the memory's summary into the WorldState (not the loop's work)."""
        if self._sightings is not None:
            threading.Thread(target=self._share_memory_now_and_then, name="memory-summary", daemon=True).start()

    def _share_memory_now_and_then(self) -> None:
        """Every MEMORY_SHARE_SECONDS until stop(): the summary, read off the video loop, for the assistant."""
        while not self._stop.is_set():
            try:
                self._state.set_memory(self._sightings.summary(time.time()))
            except Exception:
                self._memory_error("Sharing the memory's summary failed")
            self._stop.wait(MEMORY_SHARE_SECONDS)

    def _close_memory(self) -> None:
        """At stop(): write the memory's tail and close its file (also a log that was handed in)."""
        if self._sightings is None:
            return
        try:
            self._sightings.close()
        except Exception:
            self._memory_error("Closing the memory failed")

    def _memory_error(self, what: str) -> None:
        """Log a memory problem in full the first time, then only briefly."""
        if self._memory_failed:
            log.debug("%s again", what)
            return
        self._memory_failed = True
        log.exception("%s; the video goes on", what)

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
        While the camera is off the line is "Camera off" (on the plain dark picture).
        """
        if isinstance(source, CameraOff):
            text = CAMERA_OFF
        elif source.ended:
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


class CameraOff:
    """Stands in for the live source while the camera is turned off (set_camera(False)).

    It never gives a frame, so the loop handles none and no model runs. `source`
    names the camera that will be used again, so status() still says it.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.connected = False
        self.ended = False

    def read(self) -> tuple[None, int]:
        return None, 0

    def start(self) -> "CameraOff":
        return self

    def stop(self) -> None:
        pass

    def set_box(self, max_width: int, max_height: int) -> None:
        pass

    def reopen(self) -> None:
        pass


def settings_to_turn_on(stream_settings: dict[str, Any], kind: str, phone_url: str | None) -> dict[str, Any]:
    """The stream settings for set_camera(True): the phone at the address in use when the
    camera was a phone, otherwise the source the program started with."""
    if kind == "phone" and phone_url is not None:
        return {**stream_settings, "source": "phone", "phone_url": phone_url}
    return dict(stream_settings)


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


def shown_box(choice: str, picture: PhonePicture | None, config_box: Size) -> Size:
    """The box the phone's pictures are fitted into, as (width, height).

    A chosen size the phone offers gives that size, at most 1920 x 1080 with
    its shape kept (a 3840 x 2160 choice gives 1920 x 1080). Everything else
    gives the box from config.yaml: auto, keep, a size the phone does not
    offer, and any size before the phone has said what it offers.
    """
    size = parse_size(choice)
    if size is None or picture is None or size not in picture.offered:
        return config_box
    return fitted_size(size[0], size[1], *MAX_SHOWN_SIZE)


def offered_to_keep(earlier: PhonePicture | None, offered: Sequence[Size]) -> tuple[Size, ...]:
    """The offered sizes to keep for this phone: the new list, unless it lacks a size the earlier one had.

    Seen on the real phone: while its camera restarts it lists just the one
    size it sends. Taking that list would make a chosen size "not offered".
    The earlier list belongs to the same phone, because a switch forgets it.
    """
    if earlier is not None and not set(earlier.offered) <= set(offered):
        return earlier.offered
    return tuple(offered)


CardOf = Callable[[RecognizedFace], Person]     # the person a known face belongs to


# ---- the objects list on the page --------------------------------------------------


def objects_in_view(snapshot: WorldSnapshot, card_of: CardOf, found: Sequence[Detection] = (),
                    discovered: Sequence[Detection] = (), maybes: Sequence[Detection] = (),
                    on_person: Sequence[Detection] = (), wanted_people: Collection[str] = ()) -> list[dict[str, Any]]:
    """One entry per detection: people first, then by label, then by track number.

    The entries of the detections in `found` (the visitor's own search) also
    get "found": true, those in `discovered` (auto-discover) "discovered": true.
    A find in `maybes` also gets "maybe": true, one in `on_person` "on_person": true.
    A person box whose recognised face is one of `wanted_people` (ids the
    visitor typed a name of) gets "found": true too.
    """
    owners = face_owners(snapshot.detections, snapshot.faces)
    order = sorted(range(len(snapshot.detections)), key=lambda i: list_order(snapshot.detections[i]))
    entries = []
    for i in order:
        detection, face = snapshot.detections[i], owners.get(i)
        entry = object_entry(detection, face, snapshot.frame_size, card_of)
        if any(detection is find for find in found):
            entry["found"] = True
            if any(detection is find for find in maybes):
                entry["maybe"] = True
        elif any(detection is find for find in discovered):
            entry["discovered"] = True
        elif face is not None and face.person_id in wanted_people:
            entry["found"] = True
        if any(detection is find for find in on_person):
            entry["on_person"] = True
        entries.append(entry)
    return entries


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


def codes_in_view(codes: Sequence[Code], size: tuple[int, int]) -> list[dict[str, Any]]:
    """status()["codes"]: at most MOST_CODES codes, the largest box first, e.g.
    {"kind": "qr", "format": "", "text": "...", "link": "..." or None, "altered": False,
     "box": [x1, y1, x2, y2] fractions}.

    A text that cleaning changed ("altered") is never a link: the address shown
    might not be the one the code holds.
    """
    largest_first = sorted(codes, key=lambda code: box_area(code.box), reverse=True)
    return [{"kind": code.kind, "format": code.format, "text": code.text,
             "link": None if code.altered else web_link(code.text), "altered": code.altered,
             "box": box_fractions(code.box, size)}
            for code in largest_first[:MOST_CODES]]


# ---- looking for things by name --------------------------------------------------------


def clean_names(names: Sequence[str], most: int = MOST_NAMES, too_many: str = TOO_MANY_NAMES) -> tuple[str, ...]:
    """The names to look for, trimmed, with repeats dropped (in any case); () stops looking.

    Raises ValueError with a plain message for anything but a list of 1 to `most`
    names (6 for a search), each 1 to 40 letters (any script), digits, spaces,
    hyphens or apostrophes with at least one letter or digit, and none a word
    for people. `too_many` is the message for too many, with {} for `most`.
    """
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise ValueError("Give the names as a list, like [\"power bank\", \"keys\"].")
    kept: list[str] = []
    for name in names:
        name = clean_name(name)
        if all(name.casefold() != other.casefold() for other in kept):
            kept.append(name)
    if len(kept) > most:
        raise ValueError(too_many.format(most))
    return tuple(kept)


def clean_name(name: Any) -> str:
    """One name in one Unicode form (NFC), trimmed and with single spaces. ValueError when it breaks a rule."""
    if not isinstance(name, str):
        raise ValueError("Each name must be text.")
    name = re.sub(" +", " ", unicodedata.normalize("NFC", name).strip())
    if not 1 <= len(name) <= LONGEST_NAME:
        raise ValueError(f"Each name must be 1 to {LONGEST_NAME} characters long.")
    if not all(allowed_in_name(letter) for letter in name):
        raise ValueError("A name may only have letters, digits, spaces, hyphens and apostrophes.")
    if not any(unicodedata.category(letter)[0] == "L" or unicodedata.category(letter) == "Nd"
               for letter in name):
        raise ValueError("A name needs at least one letter or digit.")
    if about_people(name):
        raise ValueError(NO_PEOPLE)
    return name


def allowed_in_name(letter: str) -> bool:
    """Letters and their accents in any script, digits, and the few marks in NAME_MARKS."""
    kind = unicodedata.category(letter)
    return kind[0] in "LM" or kind == "Nd" or letter in NAME_MARKS


def about_people(name: str) -> bool:
    """True when a word of the name is a word for people (PEOPLE_WORDS), e.g. "man" or "old man's"."""
    for word in name.casefold().replace("-", " ").split():
        if word.removesuffix("'s").removesuffix("’s").strip("'’") in PEOPLE_WORDS:
            return True
    return False


def label_among(label: str, names: Sequence[str]) -> bool:
    """True when the label is one of the names, in any case."""
    folded = label.casefold()
    return any(folded == name.casefold() for name in names)


def names_after(own: Sequence[str], discovered: Sequence[str]) -> tuple[str, ...]:
    """The discovered names that go to the open detector after the own ones: no repeats, 6 in all."""
    room = max(0, MOST_NAMES - len(own))
    extra: list[str] = []
    for name in discovered:
        if len(extra) >= room:
            break
        if not label_among(name, own) and not label_among(name, extra):
            extra.append(name)
    return tuple(extra)


def in_earlier_order(names: Sequence[str], earlier: Sequence[str]) -> tuple[str, ...]:
    """The names, those already in `earlier` first and in its order, then the new ones."""
    return tuple([name for name in earlier if name in names] + [name for name in names if name not in earlier])


def discoverable(names: Sequence[str]) -> list[str]:
    """The lookout's names worth looking for: plain names, no people, nothing the everyday detector boxes."""
    kept = []
    for name in names:
        try:
            name = clean_name(name)
        except ValueError:
            continue                            # a word for people, or not a plain name
        if not boxed_every_day(name) and not label_among(name, kept):
            kept.append(name)
    return kept


def boxed_every_day(name: str) -> bool:
    """True for a name the everyday detector has a label for (or a plain word for one), also plural."""
    folded = name.casefold()
    words = EVERYDAY_LABELS | EVERYDAY_VARIANTS
    return folded in words or (folded.endswith("s") and folded[:-1] in words) or (
        folded.endswith("es") and folded[:-2] in words)


# ---- a named find and the people in view (M20) ----------------------------------------
#
# A person is never treated as an object: nothing that is a person or on a person gets a
# picture in the memory, and a click on it never sends a cut-out anywhere.


def people_boxes(detections: Sequence[Detection]) -> list[Box]:
    """The boxes of the everyday person detections."""
    return [detection.box for detection in detections if detection.label == "person"]


def share_inside(box: Box, other: Box) -> float:
    """How much of `box` lies inside `other`: 0.0 apart (or only touching), 1.0 wholly inside."""
    inside = box_area((max(box[0], other[0]), max(box[1], other[1]), min(box[2], other[2]), min(box[3], other[3])))
    whole = box_area(box)
    return inside / whole if whole > 0 else 0.0


def touches_any(box: Box, others: Sequence[Box]) -> bool:
    """True when the box overlaps any of the others at all (only touching edges is not overlapping)."""
    return any(share_inside(box, other) > 0 for other in others)


def same_as_person(find: Detection, people: Sequence[Box]) -> bool:
    """True when the find is a person: it overlaps a person (or face) box by more than
    SAME_AS_PERSON and at least MOSTLY_INSIDE of the find lies inside that box.
    Such a find is dropped: its name is only another word for that person. A wall
    that holds a person is not the person (see on_person)."""
    return any(overlap(find.box, box) > SAME_AS_PERSON and share_inside(find.box, box) >= MOSTLY_INSIDE
               for box in people)


def on_person(find: Detection, people: Sequence[Box]) -> bool:
    """True when at least ON_PERSON_SHARE of the find's box lies inside one person (or face) box
    (glasses, a shirt, keys held in front of the body). It is shown, but never a thing of
    its own: a click on it is a click on that person, and the memory is not told about it."""
    return any(share_inside(find.box, box) >= ON_PERSON_SHARE for box in people)


def wearer_of(find: Detection, detections: Sequence[Detection],
              faces: Sequence[RecognizedFace] = ()) -> int | None:
    """The position of the person box the find is on, or None.

    Inside two overlapping people it belongs to the one whose face box holds
    the find's centre; with no such face, to the one holding most of it, and
    of those that tie, the smallest box.
    """
    shares = {index: share_inside(find.box, d.box) for index, d in enumerate(detections) if d.label == "person"}
    candidates = [index for index, share in shares.items() if share >= ON_PERSON_SHARE]
    if not candidates:
        return None
    middle = centre(find.box)
    by_face = [body_of(face, detections, faces) for face in faces if contains(face.box, *middle)]
    by_face = [index for index in by_face if index in candidates]
    best = max(shares[index] for index in candidates)
    tied = by_face or [index for index in candidates if shares[index] == best]
    return min(tied, key=lambda index: box_area(detections[index].box))


def face_under(find: Detection, faces: Sequence[RecognizedFace]) -> RecognizedFace | None:
    """The face the find is on (its box holds the find's centre, or most of the find), the smallest; or None."""
    middle = centre(find.box)
    under = [face for face in faces
             if contains(face.box, *middle) or share_inside(find.box, face.box) >= ON_PERSON_SHARE]
    return min(under, key=lambda face: box_area(face.box), default=None)


class PersonGuard:
    """Keeps a person from turning into an object on a frame where the detector misses them.

    Only the loop thread uses it. It remembers the person and face boxes of the
    last NAMED_HOLD_SECONDS, and which named tracks have been the person or on
    a person. Such a track stays so until it ends or until NAMED_HOLD_SECONDS
    have passed since a person or face box was last under it, whichever is later.

    A find without a track number cannot be followed, so it is judged against
    the boxes of the last NAMED_HOLD_SECONDS, and conservatively: it is on a
    person when it overlaps any of those boxes at all.
    """

    def __init__(self) -> None:
        self._boxes: list[tuple[Box, float]] = []           # (person or face box, when it was seen)
        self._tracks: dict[int, tuple[str, float]] = {}     # track -> ("is" or "on", when a box was last under it)

    def see(self, boxes: Sequence[Box], now: float) -> None:
        """This frame's person and face boxes."""
        self._boxes = [(box, seen) for box, seen in self._boxes if now - seen <= NAMED_HOLD_SECONDS]
        self._boxes += [(box, now) for box in boxes]

    def recent_boxes(self, now: float) -> list[Box]:
        """The person and face boxes seen in the last NAMED_HOLD_SECONDS."""
        return [box for box, seen in self._boxes if now - seen <= NAMED_HOLD_SECONDS]

    def is_the_person(self, find: Detection, now: float) -> bool:
        """True when the find is a person (see same_as_person), now or earlier in its track."""
        if same_as_person(find, self.recent_boxes(now)):
            self._mark(find, "is", now)
            return True
        return self._kind(find) == "is"

    def on_a_person(self, find: Detection, now: float) -> bool:
        """True when the find is on a person (see on_person), now or earlier in its track."""
        boxes = self.recent_boxes(now)
        if find.track_id is None:
            return touches_any(find.box, boxes)
        if on_person(find, boxes):
            self._mark(find, "on", now)
            return True
        return self._kind(find) is not None

    def forget_ended(self, live: Sequence[Detection], now: float) -> None:
        """Forget the marks of tracks that ended (not among `live`) and had no person under them for a while."""
        tracks = {find.track_id for find in live}
        self._tracks = {track: mark for track, mark in self._tracks.items()
                        if track in tracks or now - mark[1] <= NAMED_HOLD_SECONDS}

    def _kind(self, find: Detection) -> str | None:
        mark = self._tracks.get(find.track_id) if find.track_id is not None else None
        return mark[0] if mark is not None else None

    def _mark(self, find: Detection, kind: str, now: float) -> None:
        if find.track_id is None:
            return
        if self._kind(find) == "is":
            kind = "is"                             # once the person, always the person
        self._tracks[find.track_id] = (kind, now)


def is_sure(find: Detection, sure: float, sure_tracks: Collection[int] = ()) -> bool:
    """True when the find is at or above `sure`, or its track once was (a dip does not make it a maybe)."""
    return find.confidence >= sure or (find.track_id is not None and find.track_id in sure_tracks)


def sure_tracks_now(finds: Sequence[Detection], earlier: Collection[int], sure: float) -> set[int]:
    """The tracks among `finds` that are sure now or were before; the others ended."""
    return {find.track_id for find in finds if find.track_id is not None and is_sure(find, sure, earlier)}


def sure_or_maybe(finds: Sequence[tuple[Detection, float]], own: Sequence[str], sure: float,
                  sure_tracks: Collection[int] = ()) -> tuple[tuple[Detection, float], ...]:
    """The finds to show: every sure one (see is_sure), and for each of the visitor's `own`
    names with no sure find its single most confident other find (a "maybe").

    The rest are dropped, so a discovered name never gets a maybe.
    """
    maybes = []
    for name in own:
        mine = [item for item in finds if label_among(item[0].label, [name])]
        if mine and not any(is_sure(item[0], sure, sure_tracks) for item in mine):
            maybes.append(max(mine, key=lambda item: item[0].confidence))
    return tuple(item for item in finds
                 if is_sure(item[0], sure, sure_tracks) or any(item is maybe for maybe in maybes))


def merge_named(everyday: Sequence[Detection], kept: NamedFinds, fresh: Sequence[Detection] | None,
                search_round: int, now: float, names: Sequence[str] = (), own: Sequence[str] = (),
                sure: float = 0.0, sure_tracks: Collection[int] = (),
                is_person: Callable[[Detection], bool] | None = None) -> tuple[list[Detection], NamedFinds]:
    """Add the named finds to the frame's everyday detections. What it returns is what
    the whole page sees: the list, the clicks, the photos, the assistant and the memory.

    `fresh` is what the open detector found on this frame, or None when it did
    not run; `names` are the names it is given in this round, `own` the
    visitor's own names among them, `sure` the open_confidence and
    `sure_tracks` the tracks that once reached it. `is_person` says whether a
    find is a person (the pipeline's PersonGuard); without it, same_as_person
    on this frame's person boxes. The rules:
    - named track numbers are kept apart from the everyday ones (NAMED_ID_OFFSET is added);
    - a find stays on screen until it has not been seen for NAMED_HOLD_SECONDS;
    - from an earlier round only the finds of names still looked for are kept;
    - a find that is a person is dropped; a find on a person stays (the pipeline marks it);
    - below `sure` only one "maybe" per own name stays (see sure_or_maybe);
    - a find replaces an everyday box it overlaps by more than REPLACE_OVERLAP,
      but never a person box: a wall behind someone does not hide them.

    Returns the detections to show and the finds to keep for the next frame.
    """
    finds = kept.finds if kept.search_round == search_round else tuple(
        (find, seen_at) for find, seen_at in kept.finds if label_among(find.label, names))
    if fresh is not None:
        finds = renewed_finds(finds, [named_apart(detection) for detection in fresh], now)
    people = people_boxes(everyday)

    def dropped(find: Detection) -> bool:
        return is_person(find) if is_person is not None else same_as_person(find, people)

    finds = tuple((find, seen_at) for find, seen_at in finds
                  if now - seen_at <= NAMED_HOLD_SECONDS and not dropped(find))
    finds = sure_or_maybe(finds, own, sure, sure_tracks)
    named = [find for find, _seen_at in finds]
    shown = [detection for detection in everyday if detection.label == "person"
             or all(overlap(detection.box, find.box) <= REPLACE_OVERLAP for find in named)]
    return shown + named, NamedFinds(search_round, finds)


def held_own_finds(kept: NamedFinds, own: Sequence[str], now: float) -> NamedFinds:
    """The finds of the visitor's own names, counted as seen now (M25).

    Used while the open detector is given new names (a watch or auto-discover
    changed them) but the own names stayed the same: the open detector cannot
    look meanwhile, and the search's answer must hold, so its finds stay until
    the new round has looked instead of vanishing after NAMED_HOLD_SECONDS.
    """
    return replace(kept, finds=tuple((find, now if label_among(find.label, own) else seen_at)
                                     for find, seen_at in kept.finds))


def renewed_finds(earlier: Sequence[tuple[Detection, float]], fresh: Sequence[Detection],
                  now: float) -> tuple[tuple[Detection, float], ...]:
    """The finds after a run: all fresh ones, plus the earlier tracked ones it missed this time.

    An earlier find is dropped when its track was found again, when it has no
    track number (it cannot be told apart from a fresh one), or when a fresh
    find covers it (its track got a new number).
    """
    fresh_tracks = {detection.track_id for detection in fresh}
    missed = tuple((detection, seen_at) for detection, seen_at in earlier
                   if detection.track_id is not None and detection.track_id not in fresh_tracks
                   and all(overlap(detection.box, find.box) <= REPLACE_OVERLAP for find in fresh))
    return tuple((detection, now) for detection in fresh) + missed


def named_apart(detection: Detection) -> Detection:
    """A named find with NAMED_ID_OFFSET added to its track number, so it never meets an everyday one."""
    if detection.track_id is None:
        return detection
    return replace(detection, track_id=detection.track_id + NAMED_ID_OFFSET)


def memory_key(detection: Detection, is_named: bool, everyday_epoch: int, named_epoch: int) -> tuple:
    """The sighting log's key for a tracked detection: it changes exactly when its track restarts.

    Everyday tracks start again only with a new camera or frame size; named
    tracks also start again when the names change. So an everyday object that
    stays in view keeps its key however often the visitor searches:
    (everyday epoch, track number, label), and for a named find
    (everyday epoch, named epoch, track number, label).
    """
    if is_named:
        return everyday_epoch, named_epoch, detection.track_id, detection.label
    return everyday_epoch, detection.track_id, detection.label


def overlap(a: Box, b: Box) -> float:
    """Intersection over union of two boxes: 0.0 apart, 1.0 the same box."""
    inside = box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = box_area(a) + box_area(b) - inside
    return inside / union if union > 0 else 0.0


def place_of(box: Box, width: int) -> str:
    """Which third of the picture the box centre is in: "left", "middle" or "right"."""
    if width <= 0:
        return "middle"
    third = min(int(3 * centre(box)[0] / width), 2)
    return PLACES[max(third, 0)]


def search_status(search: Search, found: Sequence[Detection], width: int, seen: bool = True,
                  maybes: Sequence[Detection] = (), people: dict[str, list[Box]] | None = None) -> dict[str, Any]:
    """status()["search"]: the visitor's own names, their state, and which are in view and where.

    Discovered names are not reported here. `seen` is true once the finder
    has looked at a fresh frame for these names; until then found and missing
    are both empty, so the page holds its answer. For several finds of one
    name the place is that of the most confident one. A found entry has
    "sure": false when its only find is one of `maybes`.

    `people` gives, for each typed name that is a person's, the boxes of those
    people in view (see people_in_view). Their entries have "person": true;
    "people" lists those names, and is there only when some names are
    people's. Every name is shown with its text as first answered (text_of).
    """
    state = own_state(search)
    looked = seen and state == "ready"
    in_view, missing = [], []
    if looked:
        listed: set[str] = set()                # "mohamed" and "Mohamed Adel" are one person: listed once
        for name in search.names:
            person, text = person_named(search, name), text_of(search, name)
            if text.casefold() in listed:
                continue
            listed.add(text.casefold())
            entry = (person_entry(text, (people or {}).get(name, []), width) if person is not None
                     else find_entry(name, text, found, width, maybes))
            if entry is None:
                missing.append(text)
            else:
                in_view.append(entry)
    status = {"names": list(shown_names(search)), "state": state, "note": search.note if state == "failed" else "",
              "found": in_view, "missing": missing, "seen": looked}
    if search.people:
        status["people"] = unique([text_of(search, person.typed) for person in search.people])
    return status


def find_entry(name: str, text: str, found: Sequence[Detection], width: int,
               maybes: Sequence[Detection]) -> dict[str, Any] | None:
    """A found entry (shown as `text`) for one name the open detector looks for, or None when it is not in view."""
    mine = [find for find in found if find.label.casefold() == name.casefold()]
    if not mine:
        return None
    best = max(mine, key=lambda find: find.confidence)
    sure = any(not any(find is maybe for maybe in maybes) for find in mine)
    return {"name": text, "count": len(mine), "place": place_of(best.box, width), "sure": sure}


def person_entry(text: str, boxes: Sequence[Box], width: int) -> dict[str, Any] | None:
    """A found entry (shown as `text`) for a person's name, or None when none of the people it means is in view."""
    if not boxes:
        return None
    return {"name": text, "count": len(boxes), "place": place_of(boxes[0], width), "person": True, "sure": True}


def own_state(search: Search) -> str:
    """The state of the visitor's own search: "off" without own names; still "ready" while
    only the discovered names given after them are being changed."""
    if not search.names:
        return "off"
    if search.state == "loading" and search.own_ready:
        return "ready"
    return search.state


# ---- a known person's name in Look for (M20) -------------------------------------------
#
# People are only ever recognised by their enrolled face. A typed name that is an enrolled
# person's is answered by the recognised faces, never by the open detector.


def plain(text: str) -> str:
    """The text in one Unicode form, with single spaces, straight apostrophes and in small
    letters, for comparing names: "sean o'brien" is "Sean O’Brien"."""
    return " ".join(unicodedata.normalize("NFC", text).replace("’", "'").split()).casefold()


def names_person(typed: str, person_id: str, name: str) -> bool:
    """True when the typed name is this person's: their name as enrolled, their folder name
    (underscores as spaces) or the first word of their name, in any case."""
    wanted, full = plain(typed), plain(name)
    first = full.split(" ")[0] if full else ""
    return wanted in (full, plain(person_id.replace("_", " ")), first)


def people_among(names: Sequence[str], enrolled: dict[str, str]) -> tuple[PersonName, ...]:
    """The typed names that are enrolled people's (enrolled: person id -> name as enrolled).

    A first name shared by several people means all of them. A name of
    somebody who is not enrolled is not among them: it is an ordinary name.
    """
    people = []
    for name in names:
        ids = tuple(person_id for person_id, full in enrolled.items() if names_person(name, person_id, full))
        if ids:
            people.append(PersonName(name, shown_name(name, [enrolled[person_id] for person_id in ids]), ids))
    return tuple(people)


def shown_name(typed: str, names: Sequence[str]) -> str:
    """How a person's name is shown: their name as enrolled. For several people of one
    first name, that first name as enrolled (the typed name when they differ more)."""
    if len({plain(name) for name in names}) == 1:
        return names[0]
    first = names[0].split()[0] if names[0].split() else typed
    return first if plain(first) == plain(typed) else typed


def person_named(search: Search, name: str) -> PersonName | None:
    """The person a typed name of this search means, or None for an ordinary name."""
    return next((person for person in search.people if person.typed == name), None)


def text_of(search: Search, name: str) -> str:
    """How one own name is shown: its text as first answered for this search when that is
    known (Search.shown), else a person's name as enrolled, else the name itself."""
    if len(search.shown) == len(search.names) and name in search.names:
        return search.shown[search.names.index(name)]
    person = person_named(search, name)
    return person.shown if person is not None else name


def shown_names(search: Search) -> tuple[str, ...]:
    """The own names as the page shows them (see text_of), each once."""
    return tuple(unique([text_of(search, name) for name in search.names]))


def unique(names: Sequence[str]) -> list[str]:
    """The names without repeats (in any case), the first one kept."""
    kept: list[str] = []
    for name in names:
        if not label_among(name, kept):
            kept.append(name)
    return kept


def people_in_view(people: Sequence[PersonName], detections: Sequence[Detection],
                   faces: Sequence[RecognizedFace]) -> dict[str, list[Box]]:
    """For each typed name that is a person's: where those people are, the surest face first.

    Each one is found by their recognised face; the place is their person box,
    or their face box when there is none.
    """
    owners = face_owners(detections, faces)
    bodies = {face.person_id: detections[index].box for index, face in owners.items()}
    surest: dict[str, RecognizedFace] = {}      # person id -> their surest face in view
    for face in faces:
        if face.is_known and (face.person_id not in surest or face.similarity > surest[face.person_id].similarity):
            surest[face.person_id] = face
    order = sorted(surest.values(), key=lambda face: -face.similarity)
    return {person.typed: [bodies.get(face.person_id, face.box) for face in order
                           if face.person_id in person.person_ids]
            for person in people}


def wanted_tracks(found: Sequence[Detection], maybes: Sequence[Detection], shown: Sequence[Detection],
                  faces: Sequence[RecognizedFace], search: Search) -> list[int]:
    """The track numbers drawn in the wanted colour: the sure finds of the own names, and the
    person boxes of the people whose names were typed (their face box and name stay as they are)."""
    sure = [find.track_id for find in found if not any(find is maybe for maybe in maybes)]
    wanted_people = {person_id for person in search.people for person_id in person.person_ids}
    people = [shown[index].track_id for index, face in face_owners(shown, faces).items()
              if face.person_id in wanted_people]
    return [track for track in sure + people if track is not None]


# ---- watching for things (M25) -----------------------------------------------------------
#
# "Tell me when a bottle appears." The Watcher decides on the loop thread when a watched
# thing appears; the functions around it say what is in view and what the status shows.


def watch_settings(settings: dict[str, Any]) -> tuple[int, float]:
    """(watch.max_names, watch.again_after_seconds), checked. Raises ConfigError when one is wrong."""
    most = require(settings, "max_names", int)
    if not 1 <= most <= MOST_NAMES:
        raise ConfigError(f"watch.max_names must be from 1 to {MOST_NAMES}, got {most}")
    again_after = float(require(settings, "again_after_seconds", (int, float)))
    if again_after < 0:
        raise ConfigError("watch.again_after_seconds must be 0 or more")
    return most, again_after


def everyday_label(name: str) -> bool:
    """True when the name is one of the everyday detector's own labels, in any case ("Bottle")."""
    return name.casefold() in EVERYDAY_LABELS


def watch_of(names: Sequence[str], enrolled: dict[str, str]) -> Watch:
    """The Watch for these cleaned names (enrolled: person id -> name as enrolled).

    A name of an enrolled person is shown as enrolled; a second name for the same
    person ("mohamed" and "Mohamed Adel") is dropped, so nobody is listed twice.
    """
    people = people_among(names, enrolled)
    kept: list[str] = []
    shown: list[str] = []
    for name in names:
        person = next((person for person in people if person.typed == name), None)
        text = person.shown if person is not None else name
        if not label_among(text, shown):
            kept.append(name)
            shown.append(text)
    return Watch(tuple(kept), tuple(person for person in people if person.typed in kept), tuple(shown))


def with_new_people(watch: Watch, enrolled: dict[str, str]) -> Watch:
    """The same watch after enrolling: which names are people's is decided again; the texts stay."""
    return replace(watch, people=people_among(watch.names, enrolled))


def watched_place(name: str, watch: Watch, search: Search, looked: Collection[str],
                  detections: Sequence[Detection], people: dict[str, list[Box]],
                  named: Sequence[Detection], maybes: Sequence[Detection], width: int) -> tuple[bool, str | None]:
    """Whether a watched name can be judged on this frame, and its place when it is in view (else None).

    - An enrolled person's name: in view when their face is recognised (`people`, see people_in_view).
    - An everyday label: in view when the everyday detector sees it.
    - Any other name: in view when the open detector surely finds it; a maybe never counts, a
      find on a person does (keys in a hand). It is judged only once the open detector has
      looked for it in this round (`looked`, folded names); without room it is not judged at all.
    The place is that of the surest one.
    """
    if watch.person(name) is not None:
        boxes = people.get(name, [])
        return True, place_of(boxes[0], width) if boxes else None
    if everyday_label(name):
        mine = [detection for detection in detections if label_among(detection.label, [name])]
    elif label_among(name, search.finder_names) and name.casefold() in looked:
        mine = [find for find in named if label_among(find.label, [name]) and not any(find is m for m in maybes)]
    else:
        return False, None
    if not mine:
        return True, None
    return True, place_of(max(mine, key=lambda detection: detection.confidence).box, width)


def watch_state(name: str, watch: Watch, search: Search, judged: bool) -> str:
    """One watched name's state: "watching", "loading", "waiting" (no room in the open
    detector) or "failed". "loading" also until the loop has judged it once (`judged`)."""
    if watch.person(name) is not None or everyday_label(name):
        state = "ready"
    elif label_among(name, search.names):
        state = own_state(search)                   # the visitor searches for it too
    elif label_among(name, search.watched):
        state = "ready" if search.state == "loading" and search.watch_ready else search.state
    else:
        return "waiting"
    if state == "failed":
        return "failed"
    return "watching" if state == "ready" and judged else "loading"


def watch_status(watch: Watch, search: Search, seen: dict[str, bool], fresh: bool, camera_on: bool,
                 events: Sequence[dict[str, Any]], last_at: dict[str, float]) -> dict[str, Any]:
    """status()["watch"]: each watched name with its state, and the newest events, oldest first.

    `seen` is what the loop last decided (judged name -> in view). While the camera
    is off nothing is judged, so a name is "watching" then, never "getting ready".
    """
    names = []
    for name in watch.names:
        judged = name in seen or not camera_on
        names.append({"name": watch.text(name), "state": watch_state(name, watch, search, judged),
                      "in_view": bool(fresh and camera_on and seen.get(name, False)),
                      "last_at": last_at.get(name), "person": watch.person(name) is not None})
    return {"names": names, "events": [dict(event) for event in events]}


@dataclass
class WatchedThing:
    """What the Watcher knows about one watched name in the present picture (monotonic seconds)."""

    started_at: float                  # its first judged frame
    since: float | None = None         # when the stretch it is in view now began
    last_seen: float | None = None     # the last frame it was in view
    quiet: bool = False                # it appeared: no event until away again_after seconds
    at_start: bool = False             # it was there at the start: no event until away START_AWAY_SECONDS


class Watcher:
    """Decides when a watched thing appears. Only the loop thread uses it; every time is handed in.

    A thing appears when it has been in view for APPEAR_SECONDS without a break
    longer than NAMED_HOLD_SECONDS, after not being in view. One event per
    appearance. After it, the thing stays quiet until it has been out of view
    for `again_after` seconds, however often it flickers. A thing in view
    within APPEAR_SECONDS of its first judged frame (when the watch began, or
    the first second of a new picture) was there already: it gives no event
    until it has been out of view for START_AWAY_SECONDS (a bottle taken away
    for a moment and put back is announced then).
    """

    def __init__(self, again_after: float) -> None:
        self.again_after = again_after
        self._things: dict[str, WatchedThing] = {}

    def look(self, name: str, seen: bool, now: float) -> bool:
        """One judged frame for this name. True when it appeared on this frame."""
        thing = self._things.setdefault(name, WatchedThing(started_at=now))
        if not seen:
            return False
        away = now - thing.last_seen if thing.last_seen is not None else math.inf
        if away > NAMED_HOLD_SECONDS:               # back after a real break: a new stretch in view
            thing.since = now
            if away >= self.again_after:
                thing.quiet = False
            if away >= START_AWAY_SECONDS:
                thing.at_start = False
        thing.last_seen = now
        if thing.since - thing.started_at < APPEAR_SECONDS:
            thing.at_start = True                   # in view from the start: it did not appear
        if thing.quiet or thing.at_start or now - thing.since < APPEAR_SECONDS:
            return False
        thing.quiet = True
        return True

    def in_view(self, name: str, now: float) -> bool:
        """True when the thing was in view within the last NAMED_HOLD_SECONDS."""
        thing = self._things.get(name)
        return thing is not None and thing.last_seen is not None and now - thing.last_seen <= NAMED_HOLD_SECONDS

    def judged(self, name: str) -> bool:
        """True once the name was judged on a frame of the present picture."""
        return name in self._things

    def keep(self, names: Collection[str]) -> None:
        """Forget the names no longer watched; a name watched again starts afresh."""
        self._things = {name: thing for name, thing in self._things.items() if name in names}

    def start_afresh(self) -> None:
        """A new camera or frame size: every name is judged afresh, as when the watch began."""
        self._things = {}


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
    """An object, or a person (recognised when a known face belongs to their box).

    A named find on a person is never a thing of its own: it means that person
    (their person box; on a face without one, what a click on that face means).
    """
    detection = view.detections[index]
    if any(detection is find for find in view.on_person):
        wearer = wearer_of(detection, view.detections, view.faces)
        if wearer is not None:
            return detection_target(view, wearer, card_of)
        face = face_under(detection, view.faces)
        if face is not None:
            return face_target(face, view, card_of)
        return Target("person", "person", 0.0, None, b""), detection.box   # never an object, even alone
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


def cover_faces(frame: np.ndarray, faces: Sequence[Box]) -> np.ndarray:
    """A copy of the frame with every face covered by a plain mid-grey block (FACE_COVER_GREY):
    the face box grown by FACE_COVER_MARGIN on each side, kept inside the frame.

    Every cut-out that may go to the assistant is cut from this copy, so no face leaves.
    """
    covered = frame.copy()
    height, width = frame.shape[:2]
    for x1, y1, x2, y2 in faces:
        grow_x, grow_y = (x2 - x1) * FACE_COVER_MARGIN, (y2 - y1) * FACE_COVER_MARGIN
        left, top = max(0, math.floor(x1 - grow_x)), max(0, math.floor(y1 - grow_y))
        right, bottom = min(width, math.ceil(x2 + grow_x)), min(height, math.ceil(y2 + grow_y))
        covered[top:bottom, left:right] = FACE_COVER_GREY
    return covered


def cut_out(frame: np.ndarray, box: Box) -> np.ndarray:
    """The pixels inside the box, at their own size (never enlarged)."""
    x1, y1, x2, y2 = box
    return np.ascontiguousarray(frame[y1:y2, x1:x2])


def shrunk(frame: np.ndarray, max_side: int) -> np.ndarray:
    """The frame made smaller so its longer side is at most max_side; never enlarged."""
    height, width = frame.shape[:2]
    longer = max(width, height)
    if longer <= max_side:
        return frame
    scale = max_side / longer
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def shrunk_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """The frame made at most max_width wide, its shape kept; never enlarged."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    size = (max_width, max(1, round(height * max_width / width)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


# ---- photos for enrolment --------------------------------------------------------------


def pose_for(photos_taken: int) -> str:
    """What to ask the person for before the next photo, e.g. after 1 photo: turn a little left."""
    return PHOTO_POSES[photos_taken % len(PHOTO_POSES)]


def photo_hint(frame: np.ndarray, boxes: Sequence[Box]) -> str:
    """What the person should do before a photo can be taken; "" when the frame is good.

    `boxes` are all the faces found in the frame, however small. The rules:
    exactly one face, at least PHOTO_MIN_FACE_WIDTH pixels wide, and sharp.

    Sharp means the variance of the Laplacian of the face, measured at
    SHARPNESS_WIDTH pixels wide (the size the face model reads a face), is above
    SHARPNESS_THRESHOLD. The test images in tests/test_pipeline.py show the
    scale: a drawn face scores several hundred, the same face blurred by 2
    pixels (what a moving head or a camera out of focus gives) scores below 20,
    and an even, featureless patch scores about 0. 30 refuses clearly blurred
    faces but lets slightly soft ones through, because real faces have less
    fine detail than a drawing and a session that never takes a photo helps nobody.
    """
    if not boxes:
        return HINT_NO_FACE
    if len(boxes) > 1:
        return HINT_ONE_PERSON
    box = boxes[0]
    if box[2] - box[0] < PHOTO_MIN_FACE_WIDTH:
        return HINT_CLOSER
    if sharpness(cut_out(frame, box)) <= SHARPNESS_THRESHOLD:
        return HINT_STILL
    return ""


def sharpness(face: np.ndarray) -> float:
    """How crisp a face cut-out is: the variance of its Laplacian at SHARPNESS_WIDTH pixels wide.
    0.0 for an empty cut-out."""
    height, width = face.shape[:2]
    if width == 0 or height == 0:
        return 0.0
    size = (SHARPNESS_WIDTH, max(1, round(height * SHARPNESS_WIDTH / width)))
    resized = cv2.resize(face, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def next_photo_number(folder: Path) -> int:
    """One more than the highest n of the photo_<n> files in the folder; 1 when there are none."""
    numbers = [int(match.group(1)) for path in folder.iterdir()
               if (match := PHOTO_NAME.fullmatch(path.stem))]
    return max(numbers, default=0) + 1


def save_photo(folder: Path, frame: np.ndarray) -> Path:
    """Save the frame as the next free photo_<n>.jpg in the folder, never over another file."""
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), PHOTO_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("OpenCV could not encode the photo as JPEG")
    number = next_photo_number(folder)
    while True:
        path = folder / f"photo_{number}.jpg"
        try:
            with open(path, "xb") as file:      # "x": fails instead of writing over a file
                file.write(buffer.tobytes())
            return path
        except FileExistsError:
            number += 1


def first_line(error: Exception) -> str:
    """The first line of an error message, or the error's type when it has no message."""
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__
