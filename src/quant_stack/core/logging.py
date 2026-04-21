"""Logging setup using loguru, driven by config/logging.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from loguru import logger


def setup_logging(config_path: str | Path = "config/logging.yaml") -> None:
    """Remove default loguru handler and install handlers from YAML config."""
    logger.remove()

    path = Path(config_path)
    if not path.exists():
        # Minimal fallback when config file is absent
        logger.add(sys.stderr, level="INFO")
        return

    with path.open() as f:
        cfg = yaml.safe_load(f)

    for handler in cfg.get("handlers", []):
        sink = handler.pop("sink", "stderr")
        # Resolve built-in sinks
        if sink == "stderr":
            sink = sys.stderr
        elif sink == "stdout":
            sink = sys.stdout
        else:
            # File sink: ensure parent directory exists
            Path(sink).parent.mkdir(parents=True, exist_ok=True)
        logger.add(sink, **handler)


def get_logger(name: str) -> "logger":  # type: ignore[name-defined]
    """Return a loguru logger bound with a module name."""
    return logger.bind(name=name)
