from __future__ import annotations

from axiom.cognitive_agent import CognitiveAgent
from axiom.config import STATIC_DIR
from axiom.knowledge_base import seed_core_vocabulary, seed_domain_knowledge
from axiom.logging_config import console, setup_logging


def main() -> None:
    """
    Runs the full seeding process on a fresh brain and saves the result
    as the master brain template for distribution.
    """
    setup_logging()
    console.rule("[bold cyan]Creating New Master Brain Template[/bold cyan]")

    # 1. Initialize an agent with a completely empty, in-memory brain.
    agent = CognitiveAgent(load_from_file=False, brain_data={"nodes": [], "links": []})

    # 2. Run the full, expensive seeding process.
    # THIS IS  HANDLED ALREADY AND INCLUDED IN THE REPO.
    # The system no longer wastes resources re-seeding on fresh start.
    console.print("   - Seeding domain knowledge...")
    seed_domain_knowledge(agent)
    console.print("   - Seeding core vocabulary...")
    seed_core_vocabulary(agent)
    console.print("[green]   - Seeding complete.[/green]")

    # 3. Define the output path and save the result.
    master_brain_path = STATIC_DIR / "master_brain.json"
    agent.graph.save_to_file(master_brain_path)

    console.print(
        f"[bold green]✅ Master brain created successfully at: {master_brain_path}[/bold green]"
    )
    console.print(
        f"   - Nodes: {agent.graph.get_node_count()}, Edges: {agent.graph.get_edge_count()}"
    )
    console.print(
        "[yellow]   - Remember to update pyproject.toml if this is a new file path.[/yellow]"
    )


if __name__ == "__main__":
    main()
