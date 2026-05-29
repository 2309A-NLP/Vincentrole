# -*- coding: utf-8 -*-
import logging
import os


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_logging(default_level: str = "INFO") -> None:
    """Configure application logging once for CLI and API entrypoints."""
    level_name = os.getenv("ROLEPLAY_LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format=os.getenv("ROLEPLAY_LOG_FORMAT", DEFAULT_LOG_FORMAT),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
