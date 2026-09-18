"""Configuration loader for MoEInfra.

Reads ``config.yaml`` (or another YAML file) with PyYAML, validates that all
required top-level sections are present, and returns a plain Python ``dict``.

Typical usage::

    from config_loader import load_config, get_section

    cfg = load_config()
    model_cfg = get_section(cfg, "model")
    cache_cfg = get_section(cfg, "cache")
"""
from __future__ import annotations

import os
from typing import Any

import yaml

_REQUIRED_SECTIONS: list[str] = [
    "model",
    "cache",
    "transfer",
    "logging",
    "metrics",
]


def load_config(path: str = "config.yaml") -> dict:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML config file.  Relative paths are resolved from
            the current working directory.

    Returns:
        A ``dict`` containing all configuration values from the YAML file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a required top-level section is missing.
        yaml.YAMLError: If the file is not valid YAML.
    """
    abs_path: str = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Configuration file not found: {abs_path!r}"
        )

    with open(abs_path, "r", encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh) or {}

    _validate(config)
    return config


def _validate(config: dict) -> None:
    """Assert all required top-level sections exist.

    Args:
        config: Parsed configuration dict.

    Raises:
        ValueError: If any required section is absent.
    """
    missing: list[str] = [
        section for section in _REQUIRED_SECTIONS if section not in config
    ]
    if missing:
        raise ValueError(
            f"Configuration is missing required sections: {missing}"
        )


def get_section(config: dict, section: str) -> dict:
    """Retrieve a named section from a configuration dict.

    Args:
        config: Full configuration dict as returned by :func:`load_config`.
        section: The top-level key to retrieve (e.g. ``"cache"``).

    Returns:
        The sub-dict for *section*.

    Raises:
        KeyError: If *section* is not present in *config*.
    """
    if section not in config:
        raise KeyError(
            f"Section {section!r} not found in configuration. "
            f"Available sections: {list(config.keys())}"
        )
    return config[section]
