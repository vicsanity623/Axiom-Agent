from __future__ import annotations

# tests/verify.py
import threading
import time
from pathlib import Path

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester


def run_verification_suite() -> None:
    """Run a suite of tests to verify the agent's core capabilities.

    This script executes a series of automated checks to validate the
    performance, stability, and learning capabilities of the agent. It is
    designed to be run as a final "seal of approval" after major changes.
    """
    print("--- [Verification Suite Started] ---")

    print("\nInitializing agent for verification tests...")
    dummy_lock = threading.Lock()
    try:
        agent = CognitiveAgent(
            brain_file=Path("brain/my_agent_brain.json"),
            state_file=Path("brain/my_agent_state.json"),
        )
        harvester = KnowledgeHarvester(agent=agent, lock=dummy_lock)
    except Exception as e:
        print(f"FATAL: Agent initialization failed. Error: {e}")
        return

    _test_query_performance(agent)

    _test_long_term_stability(harvester)

    print("\n--- [Verification Suite Complete] ---")


def _test_query_performance(agent: CognitiveAgent) -> None:
    """Test the multi-hop reasoning cache performance."""
    print("\n--- [Test 1: Query Time Measurement] ---")
    try:
        query = "what is the agent?"
        print(f"Running initial query (expect cache miss): '{query}'")
        start_time_miss = time.perf_counter()
        agent.chat(query)
        end_time_miss = time.perf_counter()
        duration_miss = end_time_miss - start_time_miss
        print(f"  -> Cache Miss Query Time: {duration_miss:.4f} seconds.")

        print(f"\nRunning second query (expect cache hit): '{query}'")
        start_time_hit = time.perf_counter()
        agent.chat(query)
        end_time_hit = time.perf_counter()
        duration_hit = end_time_hit - start_time_hit
        print(f"  -> Cache Hit Query Time: {duration_hit:.4f} seconds.")

        if duration_hit < 0.01:
            print(
                f"  ✅ [SUCCESS]: Cache Hit Query Time ({duration_hit:.4f}s) is excellent.",
            )
        else:
            print(
                f"  ⚠️ [WARNING]: Cache Hit Query Time ({duration_hit:.4f}s) is slower than expected.",
            )
    except Exception as e:
        print(f"  ❌ [FAILURE]: An error occurred during the query time test: {e}")


def _test_long_term_stability(harvester: KnowledgeHarvester) -> None:
    """Run the autonomous learning cycles in a loop to test for stability."""
    print("\n--- [Test 2: Stability Soak Test] ---")
    print("This test will run autonomous cycles continuously.")
    print("Let this run for as long as possible (ideally overnight).")
    print("Press CTRL+C to stop the test at any time.")

    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            print(f"\n--- Soak Test Cycle #{cycle_count} ---")

            print("  -> Running Study Cycle...")
            harvester.study_cycle()
            time.sleep(2)

            print("\n  -> Running Discovery Cycle...")
            harvester.discover_cycle()
            print(f"\n--- Cycle #{cycle_count} Complete. System is stable. ---")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n--- [Soak Test Manually Stopped] ---")
        print(
            f"The system successfully completed {cycle_count} autonomous "
            "cycles without crashing.",
        )
        print("  ✅ [SUCCESS]: The agent has demonstrated long-term stability.")
    except Exception as e:
        print("\n\n--- [Soak Test FAILED] ---")
        print(f"A fatal error occurred after {cycle_count} cycles.")
        print(f"Error: {e}")


if __name__ == "__main__":
    run_verification_suite()
