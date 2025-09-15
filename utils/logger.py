import logging
import os
from logging.handlers import RotatingFileHandler

_LOGGERS = {}


def get_logger(name: str = "security_data", level: int = logging.INFO) -> logging.Logger:
    """
    Create or get a named logger that logs to console and to logs/security_data.log
    with rotation. Thread-safe via stdlib caching by name.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(ch_fmt)
        logger.addHandler(ch)

        # Rotating file handler
        fh = RotatingFileHandler(
            filename=os.path.join("logs", "security_data.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh_fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fh_fmt)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger
