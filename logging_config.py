"""Logging configuration for MoEInfra.

Provides helpers to initialise a rotating-file + stream logging setup driven
by values read from ``config.yaml`` (via :mod:`config_loader`).

Typical usage::

    from config_loader import load_config
    from logging_config import setup_logging, get_logger

    cfg = load_config()
    setup_logging(cfg)
    log = get_logger(__name__)
    log.info("Ready.")
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from typing import Optional


_DEFAULT_LEVEL: str = "INFO"
_DEFAULT_LOG_FILE: str = "logs/moeinfra.log"
_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT: int = 5
_LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S"


def setup_logging(config: dict) -> None:
    """Configure the root logger from *config*.

    Sets up both a :class:`logging.handlers.RotatingFileHandler` (writing to
    the path specified in ``config['logging']['file']``) and a
    :class:`logging.StreamHandler` for console output.

    Args:
        config: Full configuration dict as returned by
            :func:`config_loader.load_config`.  Must contain a ``logging``
            sub-dict with optional keys ``level`` and ``file``.

    Returns:
        None
    """
    log_cfg: dict = config.get("logging", {})
    level_name: str = log_cfg.get("level", _DEFAULT_LEVEL).upper()
    log_file: str = log_cfg.get("file", _DEFAULT_LOG_FILE)

    level: int = getattr(logging, level_name, logging.INFO)

    log_dir: str = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance for *name*.
    """
    return logging.getLogger(name)
