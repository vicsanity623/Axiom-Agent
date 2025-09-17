# simulation_test.py

import threading

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester


def run_simulation():
    """
    A script to rapidly simulate the agent's learning lifecycle for testing.
    """
    print("--- [SIMULATION]: Starting agent initialization... ---")

    # We must provide the same lock the harvester and chat methods expect
    agent_interaction_lock = threading.Lock()

    # Initialize the core components
    axiom_agent = CognitiveAgent(
        brain_file="my_agent_brain.json",
        state_file="my_agent_state.json",
    )
    harvester = KnowledgeHarvester(agent=axiom_agent, lock=agent_interaction_lock)

    print("\n--- [SIMULATION]: Agent initialized. Brain is ready. ---")

    # --- 1. SIMULATE MANUAL USER LESSONS ---
    print("\n--- [SIMULATION]: Starting manual learning simulation... ---")
    manual_facts = [
        "The sun is a star.",
        "A tiger is a feline.",
        "Water boils at 100 degrees Celsius.",
    ]
    for fact in manual_facts:
        print(f"\n--- Simulating user teaching: '{fact}' ---")
        with agent_interaction_lock:
            # We call the agent's chat method directly
            axiom_agent.chat(fact)

    print("\n--- [SIMULATION]: Manual learning simulation complete. ---")

    # --- 2. SIMULATE A DISCOVERY CYCLE ---
    print("\n--- [SIMULATION]: Forcing one Discovery Cycle to run... ---")
    # We call the harvester's method directly, bypassing the scheduler
    harvester.discover_new_topic_and_learn()
    print("\n--- [SIMULATION]: Discovery Cycle complete. ---")

    # --- 3. SIMULATE A STUDY CYCLE ---
    print("\n--- [SIMULATION]: Forcing one Study Cycle to run... ---")
    # This will pick a fact learned above and try to expand on it
    harvester.study_existing_concept()
    print("\n--- [SIMULATION]: Study Cycle complete. ---")

    print("\n\n--- [SIMULATION]: ALL TASKS FINISHED. ---")
    print(
        "The agent's brain should now contain all seeded, manual, and discovered knowledge.",
    )
    print("You can now inspect the 'my_agent_brain.json' file.")


if __name__ == "__main__":
    run_simulation()
