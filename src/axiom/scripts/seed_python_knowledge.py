import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from axiom.cognitive_agent import CognitiveAgent
from axiom.config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("axiom")


def seed_python_knowledge():
    print("Initializing Cognitive Agent...")
    # IMPORTANT: Set inference_mode=False to allow writing to the graph
    agent = CognitiveAgent(
        brain_file=DEFAULT_BRAIN_FILE,
        state_file=DEFAULT_STATE_FILE,
        inference_mode=False,
    )
    agent.load_state()

    # List of facts to teach the agent
    # Format: (Subject, Relation, Object)
    facts = [
        ("Python", "is_a", "programming language"),
        ("Python", "has_property", "dynamic typing"),
        ("Python", "has_property", "garbage collection"),
        ("Python", "supports", "object-oriented programming"),
        ("Python", "supports", "functional programming"),
        ("Python", "uses", "indentation for scoping"),
        ("list", "is_a", "data structure"),
        ("list", "is_a", "sequence"),
        ("list", "is_mutable", "true"),
        ("tuple", "is_a", "data structure"),
        ("tuple", "is_a", "sequence"),
        ("tuple", "is_immutable", "true"),
        ("dictionary", "is_a", "data structure"),
        ("dictionary", "stores", "key-value pairs"),
        ("function", "is_a", "callable object"),
        ("class", "defines", "objects"),
        ("pip", "is_a", "package manager"),
        ("pip", "installs", "packages"),
        ("Flask", "is_a", "web framework"),
        ("Django", "is_a", "web framework"),
        ("NumPy", "is_a", "library for numerical computing"),
        ("Pandas", "is_a", "library for data analysis"),
    ]

    print(f"Seeding {len(facts)} facts about Python...")

    for subject, relation, obj in facts:
        try:
            # We use the internal knowledge manager API to bypass the parser/interpreter
            # and inject the facts directly as if they were "learned".

            # 1. Get or create nodes
            sub_node = agent.knowledge_manager.get_or_create_concept(subject)
            obj_node = agent.knowledge_manager.get_or_create_concept(obj)

            if sub_node and obj_node:
                # 2. Add the edge
                agent.graph.add_edge(sub_node, obj_node, relation, weight=1.0)
                print(f"  [+] Added: {subject} --[{relation}]--> {obj}")
            else:
                print(f"  [-] Failed to create nodes for: {subject} -> {obj}")

        except Exception as e:
            print(f"  [!] Error adding fact {subject} -> {obj}: {e}")

    # Save the brain
    print("Saving knowledge graph...")
    agent.save_brain()
    print("Done! Python knowledge injected.")


if __name__ == "__main__":
    seed_python_knowledge()
