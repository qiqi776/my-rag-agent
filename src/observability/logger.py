"""Logging helpers."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "minimal-rag", log_level: str = "INFO") -> logging.Logger:
    """Return a stderr-backed logger."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    return logging.getLogger(name)

