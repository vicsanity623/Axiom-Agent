import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

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
    Configure rich-enhanced logging for the entire Axiom Agent project.
    Provides colored, neatly wrapped, and bordered log output for better readability.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        log_time_format="[%H:%M:%S]",
        show_level=True,
        show_path=False,
        omit_repeated_times=False,
        keywords=["Axiom", "Agent", "Study Cycle", "Goal"],
    )

    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler("axiom.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(rich_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    console.rule(
        "[bold cyan]Axiom Agent Logging Initialized[/bold cyan]", style="border"
    )

    logging.info("[success]Logging successfully initialized for Axiom Agent.[/success]")
