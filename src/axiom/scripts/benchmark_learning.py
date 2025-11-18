from __future__ import annotations

import json
import re
from pathlib import Path

# --- Configuration ---
BRAIN_FILE = Path("src/axiom/brain/my_agent_brain.json")
LOG_FILE = Path("axiom.log")
TRAINING_DURATION_MINUTES = 60  # The standard duration of the benchmark run


def analyze_log_file(log_path: Path) -> dict[str, int]:
    """Parses the log file to extract key metrics."""
    if not log_path.exists():
        return {"high_priority_tasks_created": 0}

    content = log_path.read_text()

    # Count how many times a new high-priority task was created
    hp_tasks_created = len(
        re.findall(r"Autonomously creating HIGH-PRIORITY task", content)
    )

    return {"high_priority_tasks_created": hp_tasks_created}


def analyze_brain_file(brain_path: Path) -> dict[str, int | float]:
    """Analyzes the final brain file to score its structure."""
    if not brain_path.exists():
        return {"concepts": 0, "relations": 0, "connectivity_ratio": 0.0}

    data = json.loads(brain_path.read_text())
    num_nodes = len(data.get("nodes", []))
    num_edges = len(data.get("edges", []))

    connectivity = (num_edges / num_nodes) if num_nodes > 0 else 0.0

    return {
        "concepts": num_nodes,
        "relations": num_edges,
        "connectivity_ratio": round(connectivity, 3),
    }


def main():
    """Runs the full automated benchmark analysis."""
    print(
        "--- Axiom Learning Quality Benchmark (ALQB) - Part 1: Automated Analysis ---"
    )

    log_metrics = analyze_log_file(LOG_FILE)
    brain_metrics = analyze_brain_file(BRAIN_FILE)

    # --- Calculate Final Scores ---

    # How many new concepts did we learn per minute? (Higher is better)
    acquisition_rate = round(brain_metrics["concepts"] / TRAINING_DURATION_MINUTES, 2)

    # How many blocker tasks did we need to create? (Lower is better)
    # This is the inverse of the "knowledge explosion" problem.
    vocabulary_efficiency = log_metrics["high_priority_tasks_created"]

    # How well-connected is the final graph? (Higher is generally better)
    connectivity = brain_metrics["connectivity_ratio"]

    # --- Print Report ---
    print("\n--- BENCHMARK REPORT ---")
    print(f"Training Duration: {TRAINING_DURATION_MINUTES} minutes")
    print("\n[ Efficiency Metrics ]")
    print(f"  - Concepts Acquired: {brain_metrics['concepts']}")
    print(f"  - Acquisition Rate: {acquisition_rate} concepts/min")
    print(
        f"  - Vocabulary Efficiency (Blockers Created): {vocabulary_efficiency} (Lower is better)"
    )

    print("\n[ Knowledge Structure Metrics ]")
    print(f"  - Total Relations: {brain_metrics['relations']}")
    print(f"  - Connectivity Ratio (relations/concept): {connectivity}")

    # A simple, combined score. The weights can be tuned.
    # We want high acquisition and connectivity, and low blocker creation.
    final_score = (
        (acquisition_rate * 10) + (connectivity * 20) - (vocabulary_efficiency / 5)
    )
    print("\n--------------------------")
    print(f"  FINAL EFFICIENCY SCORE: {round(final_score, 2)}")
    print("--------------------------\n")


if __name__ == "__main__":
    main()
