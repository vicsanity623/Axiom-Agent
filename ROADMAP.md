# Axiom Agent: Development Roadmap

This document outlines the current status, planned features, and architectural improvements for the Axiom Agent project.

## ✅ Core Architecture & Capabilities (Complete & Stable)

This foundational phase is complete. The agent is a fully autonomous, reasoning, and self-correcting learning entity with a robust user interface.

- **✅ Hybrid Cognitive Architecture:** The agent's mind is a hybrid of a symbolic **Knowledge Graph** (for verifiable, persistent memory) and a neural **Universal Interpreter** (for flexible language understanding), giving it the strengths of both paradigms.
- **✅ Multi-Hop Logical Reasoning:** The agent can answer complex questions by traversing its knowledge graph and connecting multiple facts to infer new conclusions.
- **✅ Temporal Reasoning:** The agent can perceive, learn, and reason with time-based facts. It understands the context of words like "now" or "currently" and can determine the most relevant fact from a series of historical events.
- **✅ The Curiosity Engine (Active Learning Loop):**
    - **Contradiction Detection:** Actively identifies when new information conflicts with its existing world model.
    - **Active Inquiry:** Autonomously formulates and asks clarifying questions to resolve its own confusion.
    - **Knowledge Reconciliation:** Uses user feedback to correct its own brain, reinforcing correct facts and punishing incorrect ones.
- **✅ Autonomous Knowledge Harvester:**
    - **Self-Aware Learning:** Intelligently checks its own memory to avoid re-learning topics, forcing it to broaden its knowledge base.
    - **Intelligent Fact Sourcing:** Uses a multi-stage fallback (NYT API -> Wikipedia -> DuckDuckGo) to find high-quality, simple, and learnable facts from the real world.
- **✅ Vast Knowledge Base Seeding:** The agent begins its life with a large, pre-seeded set of foundational knowledge about itself, the world, and abstract concepts.

## ✅ User Interface & Experience (Complete & Stable)

- **✅ Multi-Conversation Management:** The UI supports multiple, persistent chat sessions, which are saved to the browser's local storage.
- **✅ Chat History Sidebar:** Users can easily navigate, review, and delete previous conversations.
- **✅ Rich Message Interaction:** Individual chat bubbles feature a menu with a cross-browser compatible "Copy to Clipboard" function.
- **✅ PWA (Progressive Web App):** The agent is fully configured as an installable web application with a service worker and custom icons, allowing for a native-like experience on both desktop and mobile.
- **✅ Performance & Stability:**
    - **Persistent Caching:** Caches interpretations to disk for instantaneous responses to repeated queries.
    - **Anticipatory Caching:** Proactively pre-warms the cache for newly learned topics during the agent's idle time.
    - **Thread-Safe Operation:** A global lock protects the agent from memory corruption during simultaneous user interaction and autonomous learning.

---

## Phase 1: Deepening Interaction & Scalability

*Focus: Enhance the agent's conversational abilities and ensure its knowledge base can scale gracefully.*

### 1. Conversational Context (Short-Term Memory) (`#1 Priority`)
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

### 2. User-Driven Correction Mechanism
- **Status:** Partially addressed by the Curiosity Engine, but a direct correction command is still needed.
- **Goal:** Allow users to directly correct the agent's mistakes without needing to trigger a contradiction.
- **Example:**
  - **Agent:** `The sky is green.`
  - **User:** `correction: the sky is blue`
- **Implementation Plan:**
  1.  Upgrade the `UniversalInterpreter` prompt to recognize a new intent: `statement_of_correction`.
  2.  In `CognitiveAgent`, add logic to handle this intent. It should find and remove/punish the old fact and learn the new one.

### 3. Long-Term Memory & Fact Salience
- **Goal:** Make the agent's recall more relevant over time by "remembering" frequently accessed facts more strongly.
- **Concept:** Track the usage of nodes and edges to determine their importance.
- **Implementation Plan:**
  1.  In `graph_core.py`, add an `access_count: int` property to `RelationshipEdge`.
  2.  In `CognitiveAgent`'s `_gather_facts_multihop` method, increment the `access_count` of every edge used to form an answer.
  3.  When displaying facts, prioritize showing a limited number of facts with the highest `access_count`, making the agent's responses more concise and relevant.