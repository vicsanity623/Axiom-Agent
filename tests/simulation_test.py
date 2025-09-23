from __future__ import annotations

# tests/simulation_test.py
import threading
from pathlib import Path

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester


def run_simulation() -> None:
    """Run a rapid, end-to-end simulation of the agent's core lifecycle.

    This script provides a non-interactive test to verify the agent's
    entire learning pipeline. It simulates a typical sequence of events:
    1.  Initializes the agent with a clean brain.
    2.  Simulates a user teaching several facts via the `chat` method.
    3.  Forces a single `discover_cycle` to run.
    4.  Forces a single `study_cycle` to run.

    This allows for quick, repeatable verification of the core logic
    without requiring manual input or long wait times.
    """
    print("--- [SIMULATION]: Starting agent initialization... ---")

    agent_interaction_lock = threading.Lock()

    sim_brain_file = Path("brain/simulation_brain.json")
    sim_state_file = Path("brain/simulation_state.json")
    if sim_brain_file.exists():
        sim_brain_file.unlink()
    if sim_state_file.exists():
        sim_state_file.unlink()

    axiom_agent = CognitiveAgent(
        brain_file=sim_brain_file,
        state_file=sim_state_file,
    )
    harvester = KnowledgeHarvester(agent=axiom_agent, lock=agent_interaction_lock)

    print("\n--- [SIMULATION]: Agent initialized. Brain is ready. ---")

    print("\n--- [SIMULATION]: Starting manual learning simulation... ---")
    manual_facts = [
        "The sun is a star.",
        "A tiger is a feline.",
        "Water boils at 100 degrees Celsius.",
    ]
    for fact in manual_facts:
        print(f"\n--- Simulating user teaching: '{fact}' ---")
        with agent_interaction_lock:
            axiom_agent.chat(fact)

    print("\n--- [SIMULATION]: Manual learning simulation complete. ---")

    print("\n--- [SIMULATION]: Forcing one Discovery Cycle to run... ---")
    harvester.discover_cycle()
    print("\n--- [SIMULATION]: Discovery Cycle complete. ---")

    print("\n--- [SIMULATION]: Forcing one Study Cycle to run... ---")
    harvester.study_cycle()
    print("\n--- [SIMULATION]: Study Cycle complete. ---")

    print("\n\n--- [SIMULATION]: ALL TASKS FINISHED. ---")
    print(
        "The agent's brain should now contain all seeded, manual, and",
        "discovered knowledge.",
    )
    print("You can now inspect the 'brain/simulation_brain.json' file.")


if __name__ == "__main__":
    run_simulation()
