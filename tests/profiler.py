from __future__ import annotations

# tests/profiler.py
import cProfile
import io
import pstats
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester

if TYPE_CHECKING:
    from types import TracebackType


def run_profiling() -> None:
    """Initialize the agent and profile its core autonomous learning cycles.

    This script runs a performance analysis on the `study_cycle` and
    `discover_cycle` methods of the KnowledgeHarvester. It uses Python's
    built-in cProfile module to measure execution time and identify
    potential bottlenecks in the autonomous learning logic.

    The final, sorted statistics are saved to `profile_results.txt` for
    detailed analysis.
    """
    print("--- [Profiler Started] ---")

    dummy_lock = threading.Lock()
    print("Initializing agent for profiling...")
    agent = CognitiveAgent(
        brain_file=Path("brain/my_agent_brain.json"),
        state_file=Path("brain/my_agent_state.json"),
    )
    harvester = KnowledgeHarvester(agent=agent, lock=dummy_lock)

    pr = cProfile.Profile()

    print("\n--- Profiling Study Cycle ---")
    pr.enable()
    try:
        harvester.study_cycle()
    except Exception as e:
        print(f"Error during Study Cycle profiling: {e}")
    pr.disable()
    print("--- Study Cycle Profiling Complete ---")

    print("\n--- Profiling Discovery Cycle ---")
    pr.enable()
    try:
        harvester.discover_cycle()
    except Exception as e:
        print(f"Error during Discovery Cycle profiling: {e}")
    pr.disable()
    print("--- Discovery Cycle Profiling Complete ---")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()

    with open("profile_results.txt", "w") as f:
        f.write(s.getvalue())

    print("\n--- [Profiler Finished] ---")
    print("Profiling results have been saved to 'profile_results.txt'.")


if __name__ == "__main__":
    run_profiling()