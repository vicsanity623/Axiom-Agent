import logging
import os
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

log_level_str = os.getenv("AXIOM_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, log_level_str, logging.INFO)


custom_theme = Theme(
    {
        "info": "bold cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "time": "dim white",
        "border": "bright_black",
    }
)
console = Console(theme=custom_theme)


def setup_logging():
    """
    Configure rich-enhanced logging for the Axiom Agent project.
    This setup provides two handlers:
    1. A rich, colored console handler for interactive development.
    2. A plain-text, rotating file handler for persistent logging, which is
       used by the MetacognitiveEngine for self-analysis.
    """
    logging.captureWarnings(True)

    axiom_logger = logging.getLogger("axiom")

    if axiom_logger.hasHandlers():
        axiom_logger.handlers.clear()

    axiom_logger.setLevel(LOG_LEVEL)
    axiom_logger.propagate = False

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        log_time_format="[%H:%M:%S]",
        show_level=True,
        show_path=False,
        omit_repeated_times=False,
        keywords=["Axiom", "Agent", "Study Cycle", "Goal", "Metacognition"],
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    axiom_logger.addHandler(rich_handler)

    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file_path = os.getenv("AXIOM_LOG_FILE", "axiom.log")
    file_handler = RotatingFileHandler(
        log_file_path,
        mode="a",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    axiom_logger.addHandler(file_handler)

    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)

    axiom_logger.info(
        "[success]Logging successfully initialized for Axiom Agent.[/success]"
    )
    console.rule("[bold cyan]Axiom Agent Online[/bold cyan]", style="border")
