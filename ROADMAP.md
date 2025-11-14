# Axiom Agent — Evolving Mind Roadmap

This roadmap documents the evolution of the Axiom Agent, from its foundational architecture to its future as a sophisticated, autonomous reasoner. It's organized by **phases of evolution**. Each phase represents a significant leap in capability and includes **goals**, **deliverables**, and **key features**.

***

The Axiom Agent has successfully evolved through its foundational phases. It began with the **"Genesis Engine,"** establishing a stable knowledge graph and basic parsing. It then became an **"Intelligent Learner"** with autonomous discovery, a **"Conversational Mind"** with contextual understanding, and underwent a critical **"Stabilization & Hardening"** phase to build a resilient and testable core.

Following this, the agent has now achieved a new level of intelligence in the **"Autonomous Scholar"** phase. It has moved beyond simple learning to **strategic, goal-oriented research**, driven by a formal `GoalManager`. Most significantly, it now possesses a **`MetacognitiveEngine`**, allowing it to analyze its own performance logs, suggest improvements to its own source code, and verify those fixes in a secure sandbox. This suite of capabilities marks a transition to true autonomy, setting the stage for the next leaps into deeper semantic understanding and procedural reasoning.

---

## How to use this roadmap

*   Treat each phase as a major strategic goal.
*   The features within each phase can be broken down into discrete tickets/issues.
*   For each completed phase, its summary serves as a record of the agent's growth.

---

# Phase 0 — Genesis Engine (COMPLETE)

**Goal:** Establish the core symbolic-first architecture and knowledge representation.

### Key Features
*   `ConceptGraph` implemented using NetworkX for in-memory knowledge storage.
*   Basic `SymbolicParser` for simple sentence structures.
*   Initial brain seeding process to provide foundational knowledge.
*   Core `CognitiveAgent` class to orchestrate the basic chat loop.

---

# Phase 1 — Intelligent Learner (COMPLETE)

**Goal:** Enable autonomous, self-directed learning and knowledge expansion.

### Key Features
*   `KnowledgeHarvester` implementation for autonomous study and discovery cycles.
*   `CycleManager` to switch between learning and refinement phases.
*   Integration with WordNet to enrich the agent's vocabulary and conceptual understanding.
*   Curiosity mechanism to generate follow-up questions for study.

---

# Phase 2 — Conversational & Introspective Mind (COMPLETE)

**Goal:** Enhance conversational fluency and the ability to learn from interaction.

### Key Features
*   `UniversalInterpreter` integration, using an LLM for complex language understanding.
*   Coreference resolution to handle pronouns and maintain conversational context.
*   Introspection loop allowing the agent to parse its own synthesized responses to learn new facts.
*   Conflict detection and clarification mechanism for exclusive relationships.

---

# Phase 3 — Stabilization & Hardening (COMPLETE)

**Goal:** Solidify the foundation, improve user experience, and make the agent more robust and resilient.

### Key Features
*   **Centralized & Robust Architecture:** Refactored the entire project to use a central `config.py` for all file paths.
*   **LLM-Powered Fact Verification:** Implemented an LLM-driven "quality gate" (`verify_and_reframe_fact`) that critically evaluates facts found on the web.
*   **Intelligent Lexicon Promotion:** Created a formal "promotion" system for the lexicon, allowing the agent to distinguish between observing a word and trusting its definition.
*   **Comprehensive Test Suite:** Achieved a fully passing, clean CI/CD pipeline (Ruff, MyPy, Pytest).
*   **Streamlined User Experience:** Refactored developer scripts into professional command-line entry points (`axiom-train`, `axiom-webui`, `axiom-llm`).

---

# Phase 4 — Autonomous Scholar & Metacognitive Mind (COMPLETE)

**Goal:** Elevate the agent from a simple learner to a strategic, goal-oriented entity capable of introspection, self-correction, and persistent, stateful learning.

### Key Features
*   **Hierarchical & Creative Goal Planning:** The `GoalManager` and `UniversalInterpreter` were upgraded to support randomized, creative curriculum generation.
*   **Event-Driven Cognitive Cycles:** The `CycleManager` was re-architected from a static timer to a responsive, event-driven model, dramatically increasing the agent's operational tempo.
*   **Intelligent Learning Filters:** The core learning pipeline was enhanced with a "triviality filter" to reject semantically useless facts.
*   **Hardened Refinement & Metacognition:** The `Refinement Cycle` was made transactional, and the `MetacognitiveEngine`'s prioritization logic was fixed.
*   **Robust Development Environment:** The project's dependency management was hardened, and a high-coverage End-to-End test was created to validate the entire `axiom-train` lifecycle.

---

# Phase 5 — Foundational Bootstrapping & Belief Systems (COMPLETE)

**Goal:** Dramatically accelerate the agent's initial learning, establish a robust belief system, and handle linguistic ambiguity.

### Key Features
*   **Recursive Self-Unblocking Study Cycle:** The `study_cycle` was re-architected into a powerful, recursive loop. It can now autonomously identify its own knowledge gaps ("blockers") and create high-priority tasks to resolve them, allowing it to solve deep dependency chains in a single, persistent cycle.
*   **Innate Foundational Knowledge:** The slow, active "foundational curriculum" was replaced with a much more efficient "innate" knowledge system. Hundreds of fundamental concepts are now pre-loaded directly into the `seed_core_vocabulary`, making the agent's bootstrapping process orders of magnitude faster.
*   **Knowledge Provenance & Confidence:** The `PropertyData` structure and `validate_and_add_relation` function were enhanced to track the `provenance` (source) and `confidence` of every learned fact.
*   **Belief Revision System:** A formal belief revision system was implemented. The `_resolve_exclusive_conflict` method now uses `PROVENANCE_RANK` to intelligently decide whether to accept a new fact, reject it, or ask for clarification, allowing the agent to update its beliefs based on more trustworthy information.

---

# Phase 6 — Semantic Mastery

**Goal:** Deepen the agent's understanding of language nuance, context, and complex relationships.

### Key Features
*   **Homonym & Polysemy Disambiguation:** Enhance the `_add_or_update_concept` method and `ConceptGraph` to create and manage distinct "sense" nodes for words with multiple meanings (e.g., "one" the number vs. "one" the pronoun).
*   **Advanced Relationship Extraction:** Enhance the `UniversalInterpreter` to extract and the `ConceptGraph` to represent multi-part, qualified relationships (e.g., `(person, born_in, city, on_date, date)`), moving beyond simple S-V-O.
*   **Sentiment and Tone Analysis:** Implement a mechanism to track the sentiment of the user's input and adjust the tone of synthesized responses accordingly, making conversation feel more empathetic and natural.
*   **Temporal Reasoning:** Upgrade the `_filter_facts_for_temporal_query` to handle a wider range of date/time formats and relative queries (e.g., "before 2020," "last year").

---

# Phase 7 — The Logical Reasoner (Procedural Thought)

**Goal:** Evolve beyond semantic knowledge to incorporate procedural and logical reasoning by introducing "Tools."

### Key Features
*   **CAUTION: This phase introduces significant security risks (e.g., prompt injection, arbitrary code execution). Proceed with extreme caution and a focus on sandboxing.**
*   **Tool Use Framework:** Create a `ToolManager` and a formal "Tool" interface that the agent can use. Each tool will be a specialized function for tasks the knowledge graph cannot perform.
*   **Mathematical Capability (First Tool):** A `MathTool` that integrates a symbolic math library like `SymPy`.
*   **Real-Time Data (Second Tool):** A `WebSearchTool` and a `CurrentDateTool` to fetch live data from the internet.
*   **Code Execution (Advanced Tool):** A sandboxed `PythonInterpreterTool` that can execute simple Python code to answer complex procedural questions.

---

# Phase 8 — The Autonomous Operator

**Goal:** Grant the agent the agency to act upon its own conclusions, evolving from a "Verified Advisor" to a "Human-in-the-Loop Operator."

### Key Features
*   **Action Proposal System:** Enhance the `MetacognitiveEngine` to not only suggest code changes but also to generate `git patch` files and open draft Pull Requests in a repository.
*   **Interactive Review:** Develop a conversational interface (CLI or web) where a human developer can review a proposed action (like applying a patch) and give a simple "approve" or "deny" command.
*   **Automated Execution:** Upon approval, the agent will be granted the ability to execute the verified action (e.g., `git apply patch.diff`, `git commit`, `git push`).
*   **Rollback Capability:** Implement a safety protocol where the agent can automatically revert its last action if post-deployment monitoring detects a critical failure.

---

# Phase 9 — Distributed Mind

**Goal:** Prepare the agent for massive scalability and collaborative learning.

### Key Features
*   **Pluggable Graph Backends:** Refactor `ConceptGraph` to be an interface with options for graph databases (e.g., Neo4j).
*   **Agent Federation:** Design a protocol that allows multiple Axiom Agents to query each other's knowledge and collaborate.