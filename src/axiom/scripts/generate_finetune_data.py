import json
import random
from pathlib import Path
from typing import Any

from ..cognitive_agent import CognitiveAgent
from ..config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE
from ..universal_interpreter import (
    InterpretData,
)
from ..universal_interpreter import (
    log_llm_interaction as original_log_llm_interaction,
)

LLM_INTERACTIONS_LOG = Path("llm_interactions.jsonl")
SEEN_PROMPTS: set[str] = set()


def log_llm_interaction(prompt: str, ideal_response: str) -> bool:
    """A wrapper that first checks for duplicates before logging."""
    if prompt in SEEN_PROMPTS:
        return False
    SEEN_PROMPTS.add(prompt)
    original_log_llm_interaction(prompt, ideal_response)
    return True


# ==============================================================================
# == SELF-AUGMENTING DATASET GENERATOR
# ==============================================================================


class DatasetGenerator:
    """
    A dynamic, self-augmenting engine for generating high-quality fine-tuning data
    by leveraging the agent's own knowledge graph and cognitive functions.
    """

    def __init__(self, num_examples: int):
        print("--- Initializing Dataset Generator ---")
        print("Loading Cognitive Agent (this may take a moment)...")
        self.agent = CognitiveAgent(
            brain_file=DEFAULT_BRAIN_FILE,
            state_file=DEFAULT_STATE_FILE,
            enable_llm=True,
        )
        self.num_examples = num_examples
        self.generated_count = 0

        all_edges = self.agent.graph.get_all_edges()
        node_ids = {edge.source for edge in all_edges} | {
            edge.target for edge in all_edges
        }
        all_nodes = [self.agent.graph.get_node_by_id(node_id) for node_id in node_ids]

        self.concepts = [
            node.name
            for node in all_nodes
            if node is not None
            and "noun" in self.agent.lexicon.get_promoted_pos(node.name)
        ]

        if not self.concepts:
            raise ValueError(
                "Agent brain lacks sufficient promoted nouns to use as seeds. Please run axiom-train for longer."
            )

        print(
            f"Loaded agent with {len(self.concepts)} high-quality 'noun' concepts for generation."
        )

    def _generate_and_log(
        self, prompt: str, ideal_response: str | list | dict[Any, Any] | InterpretData
    ) -> bool:
        """Logs a generated example if it's unique and we still need more."""
        if self.generated_count >= self.num_examples:
            return False

        response_str = (
            json.dumps(ideal_response)
            if not isinstance(ideal_response, str)
            else ideal_response
        )

        if log_llm_interaction(prompt, response_str):
            self.generated_count += 1
            return True
        return False

    def generate_all(self) -> None:
        """Runs the full generation pipeline until the target number of examples is reached."""
        if LLM_INTERACTIONS_LOG.exists():
            LLM_INTERACTIONS_LOG.unlink()

        print(f"\n--- Generating ~{self.num_examples} unique examples ---")

        while self.generated_count < self.num_examples:
            self.run_generation_chain()
            print(
                f"  Generated {self.generated_count}/{self.num_examples} examples...",
                end="\r",
            )

        print("\n--- ✅ Dataset generation complete! ---")
        with LLM_INTERACTIONS_LOG.open("r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(f"Total unique examples generated: {line_count}")

    def run_generation_chain(self) -> None:
        """
        Executes a single, chained sequence of generation tasks to create
        interconnected, high-quality data.
        """
        topic = random.choice(self.concepts)
        node = self.agent.graph.get_node_by_name(topic)
        if not node:
            return

        edges = self.agent.graph.get_edges_from_node(node.id)
        if not edges:
            return

        fact_edge = random.choice(edges)
        target_node = self.agent.graph.get_node_by_id(fact_edge.target)
        if not target_node:
            return

        fact_sentence = (
            f"{topic} {fact_edge.type.replace('_', ' ')} {target_node.name}."
        )

        ideal_questions = self.agent.interpreter.generate_curious_questions(
            topic, fact_sentence
        )
        if ideal_questions:
            prompt = self.agent.interpreter.get_prompt_for_test(
                "question_generation", topic=topic, known_fact=fact_sentence
            )
            self._generate_and_log(prompt, "\n".join(f"- {q}" for q in ideal_questions))

            new_question = random.choice(ideal_questions)
            interpret_prompt = self.agent.interpreter.get_prompt_for_test(
                "interpret", user_input=new_question
            )
            ideal_interpret = self.agent.interpreter.interpret(new_question)

            if ideal_interpret is not None:
                self._generate_and_log(interpret_prompt, ideal_interpret)

        if len(edges) > 1:
            facts_to_synth = []
            for edge in random.sample(edges, min(len(edges), 2)):
                tn = self.agent.graph.get_node_by_id(edge.target)
                if tn:
                    facts_to_synth.append(
                        f"{topic} {edge.type.replace('_', ' ')} {tn.name}."
                    )

            if len(facts_to_synth) > 1:
                synthesis_prompt = self.agent.interpreter.get_prompt_for_test(
                    "synthesis_statement", facts_to_rephrase=facts_to_synth
                )
                ideal_synthesis = self.agent.interpreter.synthesize(
                    structured_facts=facts_to_synth, mode="statement"
                )
                if ideal_synthesis:
                    self._generate_and_log(synthesis_prompt, ideal_synthesis)

                    decomposition_prompt = self.agent.interpreter.get_prompt_for_test(
                        "decomposition", text=ideal_synthesis, main_topic=topic
                    )
                    ideal_decomposition = (
                        self.agent.interpreter.decompose_sentence_to_relations(
                            ideal_synthesis, main_topic=topic
                        )
                    )
                    if ideal_decomposition:
                        self._generate_and_log(
                            decomposition_prompt, ideal_decomposition
                        )


def main():
    """Main function to generate the full, large dataset."""
    try:
        if not hasattr(
            CognitiveAgent(enable_llm=False).interpreter, "get_prompt_for_test"
        ):
            raise NotImplementedError(
                "The UniversalInterpreter needs a 'get_prompt_for_test' method to expose its internal prompts for data generation."
            )

        generator = DatasetGenerator(num_examples=1000)
        generator.generate_all()
    except (ValueError, FileNotFoundError, NotImplementedError) as e:
        print(f"\n[ERROR] Could not generate dataset: {e}")
        print(
            "Please ensure you have run 'axiom-train' at least once to create a brain file, and that the UniversalInterpreter has the required 'get_prompt_for_test' helper method."
        )
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
