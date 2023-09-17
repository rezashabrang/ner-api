"""Logger Module."""
from typing import Optional

import logging
import os

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    None: logging.NOTSET,
}


class LoggerSetup:
    def __init__(self, level: Optional[str] = "debug") -> None:
        """Initialize"""
        logging.root.setLevel(LEVELS[level])

    def get_colored(self, name):
        """Colored logger"""
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())

        return self.create_logger(name=name, handler=handler)

    def get_minimal(self, name):
        """Simple logger"""
        # Creating handler
        handler = logging.StreamHandler()

        # Create formatters and add it to handlers
        s_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(s_format)

        return self.create_logger(name=name, handler=handler)

    def get_detailed(self, name):
        """Getting detailed logger.
        Usually used for debug & error levels."""
        # Create handler
        handler = logging.StreamHandler()

        # Create formatters and add it to handlers
        s_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(lineno)d - %(message)s"
        )
        handler.setFormatter(s_format)

        return self.create_logger(name=name, handler=handler)

    def file_log(self, name, file_path):
        """Logging into a file"""
        # Create handler
        f_handler = logging.FileHandler(file_path)

        # Create formatters and add it to handlers
        f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(f_format)

        return self.create_logger(name=name, handler=f_handler)

    def create_logger(self, name, handler):
        """Creating logger based on handler."""
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        return logger


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    log_format = "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    # (%(filename)s:%(lineno)d)

    FORMATS = {
        logging.DEBUG: grey + log_format + reset,
        logging.INFO: blue + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record):
        """Formatter function."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


LOGGER_SETUP = LoggerSetup(level=str(os.getenv("LOG_LEVEL")).lower())

# ------------------------- Loggers ----------------------------
LOGGER = LoggerSetup().get_colored("ner_api")
