"""Simple reactions to what the camera sees: greet known people, warn when someone is close.

Pure logic with no camera, no model and no clock of its own. The pipeline
calls ReactionEngine.update once per processed frame and passes the time, so
the tests can play any sequence of frames at any speed.

- Greeting: a known face gets "Hello, <name>!" for message_seconds, at most
  once every greet_cooldown_seconds per person.
- Close warning: shown when a person box has been at least
  close_box_height_ratio of the frame height for close_hold_seconds without
  a break; gone on the first frame in which nobody is that close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from car_vision.config import ConfigError, require
from car_vision.state import WorldSnapshot

log = logging.getLogger(__name__)

WARNING_TEXT = "Person close in front of the car"


@dataclass(frozen=True)
class Reaction:
    """One message to show over the video."""

    kind: str                 # "greeting" or "warning"
    text: str
    person_id: str | None = None


WARNING = Reaction("warning", WARNING_TEXT)


def tallest_person_ratio(snapshot: WorldSnapshot) -> float:
    """Height of the tallest person box as a share of the frame height.

    0.0 when there is no person, or when the frame size is zero (not known yet).
    """
    width, height = snapshot.frame_size
    if width <= 0 or height <= 0:
        return 0.0
    heights = [d.box[3] - d.box[1] for d in snapshot.people()]
    return max(heights, default=0) / height


class ReactionEngine:
    """Decides, frame by frame, which greetings and warnings are on screen."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.greet_cooldown_seconds = _seconds(settings, "greet_cooldown_seconds")
        self.close_box_height_ratio = float(require(settings, "close_box_height_ratio", (int, float)))
        self.close_hold_seconds = _seconds(settings, "close_hold_seconds")
        self.message_seconds = _seconds(settings, "message_seconds")
        if not 0.0 < self.close_box_height_ratio <= 1.0:
            raise ConfigError(
                "reactions.close_box_height_ratio must be above 0 and at most 1, "
                f"got {self.close_box_height_ratio}"
            )

        self._greeted_at: dict[str, float] = {}                    # person id -> when last greeted
        self._greetings: dict[str, tuple[Reaction, float]] = {}    # on screen: id -> (greeting, ends at)
        self._close_since: float | None = None                     # when the current close spell began
        self._warning_on = False

    def update(self, snapshot: WorldSnapshot, now: float) -> list[Reaction]:
        """The reactions that should be on screen at `now`: the warning first, then greetings."""
        self._update_greetings(snapshot, now)
        self._update_warning(snapshot, now)
        on_screen = [WARNING] if self._warning_on else []
        on_screen += [greeting for greeting, _ends_at in self._greetings.values()]
        return on_screen

    # ---- greetings ---------------------------------------------------------

    def _update_greetings(self, snapshot: WorldSnapshot, now: float) -> None:
        for person_id, (greeting, ends_at) in list(self._greetings.items()):
            if now >= ends_at:
                del self._greetings[person_id]
                log.info("Greeting ended: %s", greeting.text)

        for face in snapshot.known_faces():
            person_id = face.person_id
            last = self._greeted_at.get(person_id)
            if last is not None and now - last < self.greet_cooldown_seconds:
                continue                          # greeted recently: once per cooldown
            greeting = Reaction("greeting", f"Hello, {face.name}!", person_id)
            self._greetings[person_id] = (greeting, now + self.message_seconds)
            self._greeted_at[person_id] = now
            log.info("Greeting started: %s is in view (%s)", face.name, person_id)

    # ---- the close warning -------------------------------------------------

    def _update_warning(self, snapshot: WorldSnapshot, now: float) -> None:
        ratio = tallest_person_ratio(snapshot)
        if ratio < self.close_box_height_ratio:
            self._close_since = None              # a break: the hold starts again next time
            if self._warning_on:
                self._warning_on = False
                log.info("Close warning ended: the tallest person box is %.2f of the frame height", ratio)
            return

        if self._close_since is None:
            self._close_since = now
        if not self._warning_on and now - self._close_since >= self.close_hold_seconds:
            self._warning_on = True
            log.info("Close warning started: a person box is %.2f of the frame height (limit %.2f)",
                     ratio, self.close_box_height_ratio)


def _seconds(settings: dict[str, Any], key: str) -> float:
    value = float(require(settings, key, (int, float)))
    if value < 0:
        raise ConfigError(f"reactions.{key} must be 0 or more, got {value}")
    return value
