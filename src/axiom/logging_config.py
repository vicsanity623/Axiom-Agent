import logging
import sys


def setup_logging():
    """
    Configure the root logger for the entire Axiom application.

    This function sets up logging to output to both the console (stdout)
    and a file (`axiom.log`). It ensures a consistent format and silences
    noisy third-party libraries. This should be called once at the start
    of any application entry point.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler("axiom.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    logging.info("--- Logging initialized for Axiom Agent ---")
