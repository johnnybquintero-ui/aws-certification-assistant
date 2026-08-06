import logging
from pathlib import Path

# Find the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Store chatbot history inside the project's logs directory.
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "chatbot_history.log"

# Create a logger specifically for this module.
logger = logging.getLogger(__name__)


def configure_chatbot_logging(
    log_path: Path = DEFAULT_LOG_PATH,
) -> logging.Logger:
    """Configure and return the chatbot's file-only logger."""

    # Create the logs directory if it does not already exist.
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Record INFO messages, warnings, errors and critical errors.
    logger.setLevel(logging.DEBUG)

    # Prevent records from being passed to the root logger.
    # This keeps chatbot history out of the terminal.
    logger.propagate = False

    # Python remembers loggers during the current process.
    # Only attach a handler if one has not already been added,
    # preventing each record from being written multiple times.
    if not logger.handlers:

        # A handler determines where log records are sent.
        # FileHandler writes them to chatbot_history.log.
        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )

        # Define how each line in the history file should appear.
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Apply the format to the file handler.
        file_handler.setFormatter(formatter)

        # Connect the file handler to the chatbot logger.
        logger.addHandler(file_handler)

    return logger
