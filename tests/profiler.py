# profiler.py

import cProfile
import pstats
import io
from threading import Lock
from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester


def run_profiling():
    """
    Initializes the agent and runs a focused performance profile on its
    most computationally expensive autonomous functions.
    """
    print("--- [Profiler Started] ---")

    # 1. Initialize the agent and its components, just like in app.py
    # We create a dummy lock because we are not running in a multi-threaded web server context.
    dummy_lock = Lock()
    print("Initializing agent for profiling...")
    agent = CognitiveAgent(load_from_file=True)
    harvester = KnowledgeHarvester(agent, dummy_lock)

    # We'll use an in-memory string stream to capture the profiler's output
    pr = cProfile.Profile()

    # 2. Profile the Study Cycle
    print("\n--- Profiling Study Cycle ---")
    pr.enable()
    try:
        harvester.study_existing_concept()
    except Exception as e:
        print(f"Error during Study Cycle profiling: {e}")
    pr.disable()
    print("--- Study Cycle Profiling Complete ---")

    # 3. Profile the Discovery Cycle
    print("\n--- Profiling Discovery Cycle ---")
    pr.enable()
    try:
        harvester.discover_new_topic_and_learn()
    except Exception as e:
        print(f"Error during Discovery Cycle profiling: {e}")
    pr.disable()
    print("--- Discovery Cycle Profiling Complete ---")

    # 4. Format and save the results to a file for analysis
    s = io.StringIO()
    # Sort the stats by cumulative time spent in functions
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()

    with open("profile_results.txt", "w") as f:
        f.write(s.getvalue())

    print("\n--- [Profiler Finished] ---")
    print("Profiling results have been saved to 'profile_results.txt'")
    print("You can now review this file to find performance bottlenecks.")


if __name__ == "__main__":
    run_profiling()
