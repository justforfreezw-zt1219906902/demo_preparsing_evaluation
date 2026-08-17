from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = files("openscan_eval").joinpath("default.yaml")


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without overriding existing environment variables."""
    dotenv = Path(path)
    if not dotenv.is_file():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults and recursively apply an optional YAML override."""
    load_dotenv()
    defaults = _read_yaml(DEFAULT_CONFIG)
    return defaults if path is None else _merge(defaults, _read_yaml(Path(path)))
