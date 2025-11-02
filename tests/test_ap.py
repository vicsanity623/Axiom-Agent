from __future__ import annotations

from typing import TYPE_CHECKING

from axiom.universal_interpreter import InterpretData

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent


class TestResolveReferences:
    """Tests for the _resolve_references method in CognitiveAgent."""

    def test_agent_resolves_pronoun_from_history_as_fallback(
        self, agent: CognitiveAgent
    ) -> None:
        """
        Given: A conversation history where the last subject was "raven".
        When: _resolve_references is called with a sentence that has a pronoun
              but NO local nouns.
        Then: The agent correctly resolves the pronoun using the historical context.
        """
        # ARRANGE: Clear history and set up a historical context.
        agent.structured_history.clear()
        fake_interpretation = InterpretData(
            intent="statement_of_fact",
            entities=[{"name": "raven", "type": "CONCEPT"}],
            relation={"subject": "raven", "verb": "is_a", "object": "bird"},
            key_topics=["raven", "bird"],
            full_text_rephrased="a raven is a bird",
        )
        agent.structured_history.append(("user", [fake_interpretation]))

        pronoun_sentence = "what color is it?"

        # ACT
        resolved_sentence = agent._resolve_references(pronoun_sentence)

        # ASSERT
        assert "raven" in resolved_sentence
        assert "what color is" in resolved_sentence

    def test_agent_prioritizes_local_context_for_pronouns(
        self, agent: CognitiveAgent
    ) -> None:
        """
        Given: A conversation history with "sky" and a new sentence with "raven".
        When: _resolve_references is called on the new sentence containing a pronoun.
        Then: The agent correctly resolves the pronoun to "raven" (the local noun),
              ignoring the historical context of "sky".
        """
        # ARRANGE: Set up a "distractor" in the history.
        agent.structured_history.clear()
        fake_history_interpretation = InterpretData(
            intent="statement_of_fact",
            entities=[{"name": "sky", "type": "CONCEPT"}],
            relation={"subject": "sky", "verb": "is", "object": "blue"},
            key_topics=["sky", "blue"],
            full_text_rephrased="the sky is blue",
        )
        agent.structured_history.append(("user", [fake_history_interpretation]))

        # --- FIX: Explicitly teach the agent that "raven" is a noun for this test. ---
        # This ensures that the local context search will find it.
        agent.lexicon._promote_word_for_test("raven", "noun")

        # This sentence has a local noun ("raven") and a pronoun ("it").
        pronoun_sentence = "I saw a raven, what color is it?"

        # ACT
        resolved_sentence = agent._resolve_references(pronoun_sentence)

        # ASSERT: The pronoun should resolve to the local noun, not the historical one.
        assert "raven" in resolved_sentence
        assert "sky" not in resolved_sentence
        assert "I saw a" in resolved_sentence

    def test_agent_does_not_change_sentence_with_no_pronouns(
        self, agent: CognitiveAgent
    ) -> None:
        """
        Given: A sentence with no pronouns.
        When: _resolve_references is called.
        Then: The sentence is returned unchanged.
        """
        # ARRANGE
        no_pronoun_sentence = "what is a bird?"

        # ACT
        unresolved_sentence = agent._resolve_references(no_pronoun_sentence)

        # ASSERT
        assert unresolved_sentence == no_pronoun_sentence
