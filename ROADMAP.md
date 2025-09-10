# Axiom Agent: Development Roadmap

This document outlines the current status, planned features, and architectural improvements for the Axiom Agent project.

## ✅ Core Architecture (Complete & Stable)

This foundational phase is complete. The agent is a fully autonomous, reasoning, and self-correcting learning entity.

- **✅ Universal Interpreter & Synthesizer:** Uses a local LLM (`Mistral-7B`) to reliably translate natural language into structured data and back into fluent responses.
- **✅ Verifiable Knowledge Graph:** The agent's long-term memory is a persistent `ConceptGraph`, preventing hallucinations and ensuring logical consistency.
- **✅ Multi-Hop Logical Reasoning:** The agent can answer complex questions by traversing its knowledge graph and connecting multiple facts to infer new conclusions.
- **✅ The Curiosity Engine (Active Learning Loop):**
    - **Contradiction Detection:** Actively identifies when new information conflicts with its existing world model.
    - **Active Inquiry:** Autonomously formulates and asks clarifying questions to resolve its own confusion.
    - **Knowledge Reconciliation:** Uses user feedback to correct its own brain, reinforcing correct facts and punishing incorrect ones.
- **✅ Autonomous Knowledge Harvester:**
    - **Multi-Source Topic Finding:** Uses a robust, multi-stage fallback system (NYT API -> Wikipedia) to find relevant and varied topics.
    - **Self-Aware Learning:** Intelligently checks its own memory to avoid re-learning topics, forcing it to broaden its knowledge base.
    - **Intelligent Fact Sourcing:** After finding a topic, it uses a multi-stage fallback (Wikipedia -> DuckDuckGo) to find a high-quality, simple, and learnable fact.
- **✅ Performance & Stability:**
    - **Persistent Caching:** Caches interpretations to disk for instantaneous responses to repeated queries.
    - **Anticipatory Caching:** Proactively pre-warms the cache for newly learned topics during idle time.
    - **Thread-Safe Operation:** A global lock protects the agent from memory corruption during simultaneous user interaction and autonomous learning.
- **✅ Vast Knowledge Base Seeding:** The agent begins its life with a large, pre-seeded set of foundational knowledge about itself, the world, and abstract concepts.

---

## Phase 1: Deepening Nuance & Interaction

*Focus: Enhance the agent's ability to understand more complex, real-world contexts like time and conversational flow.*

### 1. Relationship Properties & Nuance (`#1 Priority`)
- **Status:** **`Partially Implemented`**. The agent's brain and learning logic can store facts with properties. The next step is to implement the reasoning logic to use them.
- **Goal:** Allow facts to have context, such as time, to enable more precise answers.
- **Example:**
  - **Teach:** `As of September 9, 2025, Donald Trump is the President.`
  - **Ask:** `who is the president now`
- **Implementation Plan:**
  1.  Enhance the `UniversalInterpreter` prompt to reliably extract temporal properties (like dates) from statements.
  2.  Add specific logic to the `chat` method to handle questions containing temporal words (e.g., "now", "currently", "in 2020").
  3.  This logic will retrieve relevant facts and use Python's `datetime` library to compare the `effective_date` property on each fact to the current date to determine the correct answer.

### 2. Conversational Context (Short-Term Memory)
- **Status:** **`On Hold`**. Initial implementation proved unstable. This feature needs to be re-architected.
- **Goal:** Enable the agent to understand and answer follow-up questions with pronouns.
- **Example:**
  - `what is an apple` -> Agent answers.
  - `what color is it` -> Agent knows "it" refers to the apple.
- **New Implementation Plan:**
  1.  In `CognitiveAgent`, re-introduce a `conversation_history` queue.
  2.  Create a separate, specialized LLM call (`resolve_pronouns`) that is only triggered when a pronoun is detected.
  3.  This function will take the history and the new input and return the resolved entity (e.g., "apple").
  4.  The main `interpret` call will then use this resolved entity, keeping its primary prompt simple and reliable.

---

## Phase 2: User-Driven Growth & Scalability

*Focus: Make learning more direct and ensure the knowledge base can scale gracefully.*

### 3. User-Driven Correction Mechanism
- **Status:** Partially addressed by the Curiosity Engine, but a direct correction command is still needed.
- **Goal:** Allow users to directly correct the agent's mistakes without needing to trigger a contradiction.
- **Example:**
  - **Agent:** `The sky is green.`
  - **User:** `correction: the sky is blue`
- **Implementation Plan:**
  1.  Upgrade the `UniversalInterpreter` prompt to recognize a new intent: `statement_of_correction`.
  2.  In `CognitiveAgent`, add logic to handle this intent. It should find and remove/punish the old fact and learn the new one.

### 4. Long-Term Memory & Fact Salience
- **Goal:** Make the agent's recall more relevant over time by "remembering" frequently accessed facts more strongly.
- **Concept:** Track the usage of nodes and edges to determine their importance.
- **Implementation Plan:**
  1.  In `graph_core.py`, add an `access_count: int` property to `RelationshipEdge`.
  2.  In `CognitiveAgent`'s `_gather_facts_multihop` method, increment the `access_count` of every edge used to form an answer.
  3.  When displaying facts, prioritize showing a limited number of facts with the highest `access_count`, making the agent's responses more concise and relevant.