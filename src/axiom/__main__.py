"""
Axiom Metacognitive Engine Entrypoint

Provides a command-line interface for running self-analysis, optimization,
and other metacognitive tasks. This replaces the previous basic script with
a robust CLI using Typer.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

# Use relative imports as this is part of the axiom package
from .cognitive_agent import CognitiveAgent
from .config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE, GEMINI_API_KEY
from .metacognitive_engine import MetacognitiveEngine, PerformanceMonitor

# Create a Typer app for a clean, self-documenting CLI experience
app = typer.Typer(
    name="axiom-meta",
    help="Axiom Metacognitive Engine: Tools for self-analysis and improvement.",
    add_completion=False,
)


@app.command()
def analyze_logs(
    log_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the log file to analyze.",
        ),
    ] = Path("axiom.log"),
) -> None:
    """
    Analyzes performance logs to find the highest-priority optimization target.
    """
    typer.echo(f"🔍 Analyzing logs from '{log_path}'...")
    monitor = PerformanceMonitor()
    result = monitor.find_optimization_target(log_path)

    if result:
        typer.secho("✅ Found an optimization target:", fg=typer.colors.GREEN)
        result_dict = {
            "file_path": str(result.file_path),
            "target_name": result.target_name,
            "issue_description": result.issue_description,
            "relevant_logs": result.relevant_logs,
        }
        typer.echo(json.dumps(result_dict, indent=2))
    else:
        typer.secho(
            "✅ Analysis complete. No high-priority optimization targets found.",
            fg=typer.colors.GREEN,
        )


@app.command()
def cycle_now(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force a new cycle even if a suggestion file exists.",
        ),
    ] = False,
) -> None:
    """
    (Experimental) Runs one full, on-demand metacognitive cycle.
    """
    if force:
        suggestion_file = Path("code_suggestion.json")
        if suggestion_file.exists():
            typer.echo(
                f"🗑️ --force specified. Removing existing suggestion file: {suggestion_file}"
            )
            suggestion_file.unlink()

    typer.echo("🚀 Initializing agent for a single metacognitive cycle...")
    try:
        # We need a live agent instance for the engine to access its components
        agent = CognitiveAgent(
            brain_file=DEFAULT_BRAIN_FILE,
            state_file=DEFAULT_STATE_FILE,
            inference_mode=True,  # No autonomous learning during this cycle
        )
        engine = MetacognitiveEngine(agent=agent, gemini_api_key=GEMINI_API_KEY)

        typer.echo("⚙️ Running introspection cycle...")
        engine.run_introspection_cycle()
        typer.echo("🏁 Cycle complete.")

    except Exception as e:
        typer.secho(
            f"❌ CRITICAL ERROR during metacognitive cycle: {e}", fg=typer.colors.RED
        )
        raise typer.Abort()


@app.command()
def introspect() -> None:
    """(Coming soon) Interactively inspect agent cognitive modules."""
    typer.secho(
        "Feature not yet implemented: Inspecting agent modules.", fg=typer.colors.YELLOW
    )
    raise typer.Exit()


@app.command()
def self_modify() -> None:
    """(Coming soon) Apply the latest verified patch automatically."""
    typer.secho(
        "Feature not yet implemented: Applying verified patches.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit()


if __name__ == "__main__":
    app()
