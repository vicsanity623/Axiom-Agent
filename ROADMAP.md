# Axiom Agent: Development Roadmap

This document outlines the current status, planned features, and architectural improvements for the Axiom Agent project.

## ✅ Core Architecture (Complete)

This foundational phase is complete and stable. The agent is a fully autonomous, learning entity.

- **✅ Universal Interpreter:** The agent uses a Mini LLM (`Mistral-7B`) to reliably interpret natural language into structured intents, relations, and commands.
- **✅ Verifiable Knowledge Graph:** The agent's long-term memory is a `ConceptGraph` that stores facts persistently, preventing LLM-style hallucinations.
- **✅ Fluent Synthesizer:** The agent uses the Mini LLM to convert its structured knowledge into natural, grammatically correct responses.
- **✅ Self-Identity & Recall:** The agent can be taught a rich identity (name, purpose, capabilities) and can answer questions about itself fluently by synthesizing its known facts.
- **✅ Alias Merging:** The agent can synthesize knowledge about a single entity from multiple sources (e.g., "Eminem" and "Marshall Mathers").
- **✅ Autonomous Learning (`KnowledgeHarvester`):**
    - **Multi-Source Topic Finding:** Uses a robust, three-stage fallback system (NYT API -> Wikipedia -> DuckDuckGo) to find relevant and varied topics.
    - **Self-Awareness:** Intelligently checks its own memory to avoid re-learning topics it already knows, forcing it to broaden its knowledge base.
    - **Intellectual Caution:** Correctly rejects complex or ambiguous sentences that cannot be distilled into high-quality, atomic facts.
- **✅ Persistent Caching:** The agent caches interpretations and synthesized responses to disk, providing instant replies to repeated queries across server restarts.
- **✅ System Commands:** The agent understands and executes built-in commands, such as `show all facts`, for transparent inspection of its knowledge base.
- **✅ Thread-Safe Operation:** A global lock prevents race conditions and crashes between user interactions and the autonomous learning cycle.
- **✅ PWA Web Interface:** The agent is served via a Flask web application and is fully configured as an installable Progressive Web App.

---

## Phase 1: Deepening Reasoning & Understanding

*Focus: Enhance the agent's ability to reason with and understand the nuances of the knowledge it acquires.*

### 1. Multi-Hop Reasoning (`#1 Priority`)
- **Goal:** Enable the agent to answer questions by connecting multiple facts in its knowledge graph.
- **Concept:** Move from simple fact retrieval to basic logical inference.
- **Example:**
  - **Teach:** `Paris is located in France.`
  - **Teach:** `France is a country in Europe.`
  - **Ask:** `is Paris in Europe`
- **Implementation Plan:**
  1.  Modify the fact retrieval logic in `cognitive_agent.py`'s `chat` method.
  2.  After retrieving direct (1-hop) facts for an entity, perform a second query on the results of the first hop (e.g., find where "France" is located).
  3.  Aggregate both 1-hop and 2-hop facts into the `structured_response`.
  4.  Pass the aggregated facts to the `synthesize` method with the original question, allowing the LLM to form the final inferential answer.

### 2. Relationship Properties & Nuance
- **Status:** **`Partially Implemented`**. The agent's brain (`graph_core.py`) and learning logic (`cognitive_agent.py`) are now fully capable of storing and retrieving facts with properties. The next step is to implement the reasoning logic to use them.
- **Goal:** Allow facts to have additional context, such as time, to enable more precise answers.
- **Example:**
  - **Teach:** `As of September 9, 2025, Donald Trump is the President.`
  - **Ask:** `who is the president now`
- **Implementation Plan:**
  1.  Enhance the `UniversalInterpreter` prompt to reliably extract temporal properties (like dates) from statements.
  2.  Add specific logic to the `chat` method to handle questions containing temporal words (e.g., "now", "currently", "in 2020").
  3.  This logic will retrieve all relevant facts and use Python's `datetime` library to compare the `effective_date` property on each fact to the current date to determine the correct answer.

---

## Phase 2: Enhancing Interaction & Usability

*Focus: Make the agent more conversational, adaptable, and user-friendly.*

### 3. Conversational Context (Short-Term Memory)
- **Status:** **`On Hold`**. Initial implementation proved to be unstable. This feature needs to be re-architected.
- **Goal:** Enable the agent to understand and answer follow-up questions with pronouns.
- **Example:**
  - `what is an apple` -> Agent answers.
  - `what color is it` -> Agent knows "it" refers to the apple.
- **New Implementation Plan:**
  1.  In `CognitiveAgent`, re-introduce a `conversation_history` queue.
  2.  Create a separate, specialized LLM call (`resolve_pronouns`) that is only triggered when a pronoun is detected.
  3.  This function will take the history and the new input and return the resolved entity (e.g., "apple").
  4.  The main `interpret` call will then use this resolved entity, keeping its primary prompt simple and reliable.

### 4. User-Driven Correction Mechanism
- **Goal:** Allow users to directly correct the agent's mistakes.
- **Concept:** Introduce a "correction" intent that modifies existing knowledge.
- **Example:**
  - **Agent:** `The sky is green.`
  - **User:** `no, the sky is blue`
- **Implementation Plan:**
  1.  Upgrade the `UniversalInterpreter` prompt to recognize a new intent: `statement_of_correction`.
  2.  In `CognitiveAgent`, add logic to handle this intent. It should find the incorrect fact in its graph and either delete it or lower its confidence, then learn the new, correct fact.

---

## Phase 3: Scaling Knowledge & Long-Term Growth

*Focus: Ensure the agent's knowledge can grow massively while remaining manageable and relevant.*

### 5. Vast Knowledge Base Seeding (`✅ COMPLETE`)
- **Status:** **`Complete`**. The `knowledge_base.py` file has been significantly expanded to seed the agent with a large, high-quality set of foundational knowledge about itself, the world, and abstract concepts upon its first run.