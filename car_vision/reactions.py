"""Simple reactions to what the camera sees: greet the people it knows.

Pure logic with no camera, no model and no clock of its own. The pipeline
calls ReactionEngine.update once per processed frame and passes the time, so
the tests can play any sequence of frames at any speed.

- Greeting: a known face gets "Hello, <name>!" for message_seconds, at most
  once every greet_cooldown_seconds per person.

(An earlier version also showed a red warning when a person was close in front
of the car. The project owner did not want it, so it was taken out.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from car_vision.config import ConfigError, require
from car_vision.state import WorldSnapshot

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reaction:
    """One message to show over the video."""

    kind: str                 # "greeting"
    text: str
    person_id: str | None = None


class ReactionEngine:
    """Decides, frame by frame, which greetings are on screen."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.greet_cooldown_seconds = _seconds(settings, "greet_cooldown_seconds")
        self.message_seconds = _seconds(settings, "message_seconds")

        self._greeted_at: dict[str, float] = {}                    # person id -> when last greeted
        self._greetings: dict[str, tuple[Reaction, float]] = {}    # on screen: id -> (greeting, ends at)

    def update(self, snapshot: WorldSnapshot, now: float) -> list[Reaction]:
        """The greetings that should be on screen at `now`."""
        self._update_greetings(snapshot, now)
        return [greeting for greeting, _ends_at in self._greetings.values()]

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


def _seconds(settings: dict[str, Any], key: str) -> float:
    value = float(require(settings, key, (int, float)))
    if value < 0:
        raise ConfigError(f"reactions.{key} must be 0 or more, got {value}")
    return value
