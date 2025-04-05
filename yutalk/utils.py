import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output"""

    # ANSI escape sequences for colors
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",  # Reset color
    }

    def format(self, record: logging.LogRecord) -> str:
        # Add color to the level name
        level_name = record.levelname
        if level_name in self.COLORS:
            colored_level_name = (
                f"{self.COLORS[level_name]}{level_name}{self.COLORS['RESET']}"
            )
            record.levelname = colored_level_name

        return super().format(record)


def setup_logger(
    name: str = __name__,
    level: int = logging.DEBUG,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with colored console output and optional file output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = (
            "[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s: %(message)s"
        )

    # Custom datefmt format with period instead of comma
    datefmt = "%Y-%m-%d %H:%M:%S"  # Period as separator for microseconds

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(format_string, datefmt=datefmt)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(format_string, datefmt=datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger