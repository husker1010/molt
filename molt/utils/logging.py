"""Console and file logging built on :mod:`rich`."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["get_logger", "setup_logging", "console"]

console = Console(stderr=True)
"""Shared stderr console used by progress bars and log output."""

_FMT_FILE = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_configured = False


def setup_logging(
    level: int | str | None = None,
    log_file: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``molt`` logger hierarchy.

    Args:
        level: Logging level; defaults to ``$MOLT_LOG_LEVEL`` or ``INFO``.
        log_file: Optional file that receives plain-text log records as well.
        force: Reconfigure even if logging was already set up.

    Returns:
        The root ``molt`` logger.
    """
    global _configured
    logger = logging.getLogger("molt")
    if _configured and not force and log_file is None:
        return logger
    level = level or os.environ.get("MOLT_LOG_LEVEL", "INFO")
    logger.setLevel(level if isinstance(level, int) else str(level).upper())
    if force or not _configured:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        rich_handler = RichHandler(
            console=console, show_path=False, rich_tracebacks=True, markup=False
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(rich_handler)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT_FILE))
        logger.addHandler(file_handler)
    logger.propagate = False
    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``molt`` namespace, configuring logging on first use."""
    setup_logging()
    if not name or name == "molt":
        return logging.getLogger("molt")
    return logging.getLogger(name if name.startswith("molt") else f"molt.{name}")
