from __future__ import annotations

import json
from pathlib import Path

from axiom.cognitive_agent import CognitiveAgent

# --- Configuration ---
BRAIN_FILE = Path("src/axiom/brain/my_agent_brain.json")
EVALUATION_OUTPUT_FILE = Path("evaluation_results.json")

# This is our "Golden Question Set" that we will use to score every version of the agent.
GOLDEN_QUESTIONS = [
    # --- Direct Fact Retrieval ---
    "What is an apple?",
    "What is a dog?",
    "What is a chair?",
    # --- Simple Inference (Properties) ---
    "What color is a banana?",
    "What sound does a cat make?",
    # --- Categorization / Hierarchy ---
    "Is a dog an animal?",
    "Is a chair a type of fruit?",  # (Negative test)
    # --- Simple Reasoning ---
    "What is a house used for?",
    "Why do people wear shoes?",
]


def main():
    """
    Runs the HITL evaluation by asking the agent the Golden Questions and saving its answers.
    """
    print(
        "--- Axiom Learning Quality Benchmark (ALQB) - Part 2: Reasoning Evaluation ---"
    )

    if not BRAIN_FILE.exists():
        print(
            f"Error: Brain file not found at {BRAIN_FILE}. Please run a training session first."
        )
        return

    # Load the agent in inference mode with the trained brain
    agent = CognitiveAgent(brain_file=BRAIN_FILE, inference_mode=True)

    results = []

    print(f"\nAsking {len(GOLDEN_QUESTIONS)} golden questions...")
    for i, question in enumerate(GOLDEN_QUESTIONS):
        print(f"  ({i + 1}/{len(GOLDEN_QUESTIONS)}) Asking: {question}")
        answer = agent.chat(question)
        print(f"      -> Answer: {answer}")

        results.append(
            {
                "question": question,
                "answer": answer,
                "score": None,  # To be filled in by a human
                "notes": "",  # For human feedback
            }
        )

    EVALUATION_OUTPUT_FILE.write_text(json.dumps(results, indent=4))

    print("\n--- EVALUATION COMPLETE ---")
    print(f"Answers saved to '{EVALUATION_OUTPUT_FILE}'.")
    print("Please open this file and score each answer on a scale of 0-2:")
    print("  - 0: Incorrect, irrelevant, or 'I don't know'.")
    print("  - 1: Partially correct, or technically correct but unhelpful.")
    print("  - 2: Correct, concise, and helpful.")


if __name__ == "__main__":
    main()
