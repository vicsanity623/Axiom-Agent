# Axiom Agent: Development Roadmap

This document outlines the current status and future architectural direction for the Axiom Agent project.

## ✅ Phase 1: The Genesis Engine (Complete)

This foundational phase is **complete**. It established a stable, high-performance cognitive engine with a professional toolchain, a scalable knowledge graph, and a complete training-to-deployment workflow running on local CPU hardware.

## ✅ Phase 2: The Intelligent Learner (Complete)

This crucial phase is **complete**. It transformed the agent's autonomous learning from a random process into an intelligent, goal-driven system.

### ✅ **Core Achievements of this Phase:**
- **Symbolic-First Cognitive Flow:** The agent now attempts to understand input using its native `SymbolicParser` **before** falling back to the LLM.
- **Autonomous Vocabulary Expansion:** The agent can identify unknown words and research their definitions using a high-precision **Dictionary API**.
- **Intelligent Topic Discovery:** The `Discovery Cycle` now explores curated **core subjects** and uses a **popularity heuristic** to find relevant new topics.
- **Productive "Deepen Knowledge" Cycle:** The agent can now proactively research concepts it already knows to discover and learn new, related facts, enriching its understanding.

## ✅ Phase 3: The Conversational & Introspective Mind (Complete)

This phase is now **complete**. It focused on giving the agent the foundational capabilities to understand more complex language, the flow of a conversation, and its own internal knowledge.

### ✅ **Core Achievements of this Phase:**
- **Multi-Stage Parser (The Chunker):** The `SymbolicParser` has been evolved into a multi-stage pipeline, beginning with a "Chunker" that can split complex, multi-sentence text into individual, parsable clauses.
- **Expanded Parser Grammar:** The parser's pattern-matching engine was upgraded to understand **adjectives** and to correctly identify **questions**.
- **Coreference Resolution:** The agent now possesses a deterministic, symbolic mechanism for **short-term memory**. It can resolve basic pronouns (e.g., "it," "they") by referencing the subject of the previous conversational turn.
- **Introspective Learning & Refinement:** The agent now has two forms of introspection:
    - **Introspective Learning:** It can **learn from its own synthesized output**. If the LLM synthesizer "leaks" a new fact, the agent parses its own response and integrates that new knowledge.
    - **Introspective Refinement:** A dedicated `Refinement Cycle` now allows the agent to find "chunky," complex facts in its own brain and use the LLM to decompose them into smaller, more precise, atomic facts, thereby improving the quality of its knowledge.
- **Phased Cognitive Architecture:** A `CycleManager` switches the agent between a "Learning Phase" (broadening knowledge) and a "Refinement Phase" (deepening and cleaning knowledge).

---

## The Path Forward: Hardening the Mind

With the agent now capable of stable, conversational, and introspective learning, the focus shifts to **hardening its foundational understanding of language** to make its reasoning more powerful and resilient.

### **Phase 4: The Hardened Mind (Foundational Nuance & Robustness) (Current Focus)**
- **Goal:** To address the subtle but critical limitations discovered during long-term autonomous runs and to make the agent's knowledge base more unified.
- **Status:** **Current Focus.**
- **Key Steps / To-Do List:**
    1.  **Implement a Lemmatization Layer:** (Problem: "factory" vs "factories"). The single biggest barrier to deeper reasoning is that the agent treats singular and plural words as separate concepts. The next step is to integrate a lemmatizer (e.g., from `nltk`) into `_clean_phrase` to reduce all words to their dictionary root form.
    2.  **Expand Parser Grammar (Prepositions):** Teach the parser to handle prepositional phrases (e.g., "The book is **on the table**," "Paris is **in France**") to dramatically increase the complexity of facts it can learn.
    3.  **Establish Semantic Equivalence:** (Problem: `"zero"` vs `"0"`). Begin seeding a new relationship, `is_equivalent_to`, to allow for abstract reasoning.

### **Phase 5: The Autonomous Scholar (Advanced Curriculum Learning)**
- **Goal:** Evolve the agent's learning from simple goal fulfillment to a strategic, goal-oriented "curriculum" driven by the gaps in its own understanding.
- **Status:** **Not Started.**

### **Phase 6: The Distributed Mind (Knowledge Scalability)**
- **Goal:** Overcome local RAM/storage limits by migrating the knowledge graph to a free-tier cloud database.
- **Status:** **Not Started.**