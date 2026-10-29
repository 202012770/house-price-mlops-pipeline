"""YAML-backed configuration with path resolution."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml


def _detect_project_root() -> Path:
    """Resolve the repository root for both editable and wheel installs.

    Editable installs live at ``<root>/src/house_price/config.py``; wheel installs
    live in site-packages, so fall back to ``HPP_PROJECT_ROOT`` or the working
    directory (``/app`` in the container).
    """
    env_root = os.environ.get("HPP_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs").exists():
        return candidate
    return Path.cwd()


PROJECT_ROOT = _detect_project_root()
CONFIG_DIR = Path(os.environ.get("HPP_CONFIG_DIR", PROJECT_ROOT / "configs"))


class Config(dict):
    """Dict with recursive attribute access and path resolution."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def path(self, key: str) -> Path:
        value: Any = self
        for part in key.split("."):
            value = value[part]
        return resolve_path(value)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@cache
def load_config(name: str) -> Config:
    """Load ``configs/<name>.yaml`` once per process."""
    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle) or {})


@cache
def data_config() -> Config:
    return load_config("data")


@cache
def training_config() -> Config:
    return load_config("training")
