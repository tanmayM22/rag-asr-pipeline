import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

def get_logger(name: str) -> logging.Logger:
    """Return a named logger that writes only to file, never to terminal."""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # rotate after 5MB, keep last 3 log files
    handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.addHandler(handler)
    logger.propagate = False  # don't bubble up to root logger (keeps terminal clean)

    return logger