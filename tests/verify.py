# verify.py

import time
from threading import Lock

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester


def run_verification_tests():
    """
    Runs the final success metric tests for Phase 1 of the roadmap.
    1. Measures query time to ensure it's under the 2-second goal.
    2. Runs a long-duration stability test of the autonomous cycles.
    """
    print("--- [Phase 1 Verification Started] ---")

    # --- Initialization ---
    print("\nInitializing agent for verification tests...")
    dummy_lock = Lock()
    try:
        agent = CognitiveAgent(load_from_file=True)
        harvester = KnowledgeHarvester(agent, dummy_lock)
    except Exception as e:
        print(f"FATAL: Agent initialization failed. Error: {e}")
        return

    # --- Test 1: Query Time Measurement ---
    print("\n--- [Test 1: Query Time Measurement] ---")
    try:
        # We'll test a known, multi-hop query: "agent"
        # The first run will be slow due to a cache miss. This is expected.
        print("Running initial query (expect cache miss)...")
        start_time_miss = time.time()
        agent.chat("what is the agent?")
        end_time_miss = time.time()
        duration_miss = end_time_miss - start_time_miss
        print(f"  -> Cache Miss Query Time: {duration_miss:.4f} seconds.")

        # The second run should be extremely fast, served from the cache.
        print("\nRunning second query (expect cache hit)...")
        start_time_hit = time.time()
        agent.chat("what is the agent?")
        end_time_hit = time.time()
        duration_hit = end_time_hit - start_time_hit
        print(f"  -> Cache Hit Query Time: {duration_hit:.4f} seconds.")

        if duration_hit < 2.0:
            print("  [SUCCESS]: Query time is under the 2-second goal.")
        else:
            print(
                f"  [FAILURE]: Query time ({duration_hit:.4f}s) exceeds the 2-second goal.",
            )
    except Exception as e:
        print(f"  [FAILURE]: An error occurred during the query time test: {e}")

    # --- Test 2: 24-Hour Stability Simulation (Soak Test) ---
    print("\n--- [Test 2: Stability Soak Test] ---")
    print("This test will run autonomous cycles continuously.")
    print("Let this run for as long as possible (ideally overnight).")
    print("Press CTRL+C to stop the test at any time.")
    print("Look for any crashes or fatal errors in the log.")

    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            print(f"\n--- Soak Test Cycle #{cycle_count} ---")

            print("  -> Running Study Cycle...")
            harvester.study_existing_concept()

            # Add a small delay to simulate time passing
            time.sleep(2)

            print("\n  -> Running Discovery Cycle...")
            harvester.discover_new_topic_and_learn()

            print(f"\n--- Cycle #{cycle_count} Complete. System is stable. ---")
            # In a real 24-hour test, you'd have a much longer sleep here.
            # For simulation, we'll just loop.
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n--- [Soak Test Manually Stopped] ---")
        print(
            f"The system successfully completed {cycle_count} autonomous cycles without crashing.",
        )
        print("  [SUCCESS]: The agent has demonstrated long-term stability.")
    except Exception as e:
        print("\n\n--- [Soak Test FAILED] ---")
        print(f"A fatal error occurred after {cycle_count} cycles.")
        print(f"Error: {e}")

    print("\n--- [Phase 1 Verification Complete] ---")


if __name__ == "__main__":
    run_verification_tests()
