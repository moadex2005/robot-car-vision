"""Loads config.yaml once and gives every module the same settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(Exception):
    """A setting is missing or has the wrong kind of value."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML settings file and return it as a dictionary."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Settings file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path} must contain named sections, e.g. 'stream:'")
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one named section, with a clear error when it is missing."""
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"config.yaml has no '{name}:' section")
    return value


def require(settings: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    """Return settings[key], checking it exists and has the expected type."""
    if key not in settings:
        raise ConfigError(f"config.yaml is missing the setting '{key}'")
    value = settings[key]
    # bool is a subclass of int in Python; do not accept true/false as a number.
    if isinstance(value, bool) and kind is not bool and bool not in (kind if isinstance(kind, tuple) else (kind,)):
        raise ConfigError(f"Setting '{key}' must be {kind}, got true/false")
    if not isinstance(value, kind):
        raise ConfigError(f"Setting '{key}' has the wrong kind of value: {value!r}")
    return value


def resolve_path(relative: str | Path) -> Path:
    """Turn a path from config.yaml into a full path under the project folder."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
