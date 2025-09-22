# Axiom Agent: Development Roadmap

This document outlines the current status and future architectural direction for the Axiom Agent project.

## ✅ Phase 1: The Genesis Engine (Complete & Stable)

This foundational phase is **complete**. The project has successfully evolved from a prototype into a stable, high-performance cognitive engine. This phase established a robust foundation for all future development, including a professional toolchain, a scalable knowledge graph, and a complete training-to-deployment workflow running on local CPU hardware.

### ✅ **Core Achievements of this Phase:**
- **Symbolic Knowledge Graph:** A verifiable, persistent memory powered by `NetworkX` that prevents hallucinations.
- **LLM as a Tool:** A successful proof-of-concept using a local LLM as a fallback "universal interpreter" for language tasks.
- **Autonomous Learning Cycles:** A stable "Cognitive Scheduler" that enables the agent to learn and study 24/7.
- **Professional Development Environment:** A full suite of quality assurance tools (`Ruff`, `MyPy`, `Pytest`), a modern `pyproject.toml` configuration, and a clean `src` layout.
- **Complete Deployment Workflow:** A robust `Train -> Render -> Deploy` cycle with dedicated scripts for each stage.

---

## The Path Forward: Achieving True Understanding

With the core engine stable, the strategic focus now pivots from *using an LLM as a tool* to **systematically replacing it**. The following phases are designed to build a truly **Symbolic-First Architecture**, where the agent develops its own grounded understanding of language from the ground up, moving from a hybrid system to a truly autonomous cognitive entity.

### **Phase 2: The Symbolic Interpreter (Current Focus)**
- **Goal:** To methodically replace the probabilistic LLM interpreter with a deterministic, graph-native **Axiom Lexicon & Parser**, enabling the agent to build a truly grounded understanding of language.
- **Status:** **In Progress.** The foundational components are complete.
- **Completed Steps & Successes:**
    1.  **Lexicon Manager:** A dedicated module for managing the agent's internal dictionary of known words is complete and integrated.
    2.  **"Unknown Word" Reflex:** The agent can now successfully identify words it doesn't know, create an "INVESTIGATE" goal, and defer processing—a primitive form of metacognitive awareness.
    3.  **Targeted Researcher:** The `KnowledgeHarvester` has been successfully refactored. Its primary mission is now to fulfill "INVESTIGATE" goals by performing targeted web searches for definitions, **without any LLM dependency.**
    4.  **Symbolic Parser (v1):** The initial version of the parser is complete. It can successfully parse simple `Subject-Verb-Object` sentences and create a structured interpretation, allowing the agent to learn facts **without calling the LLM.**
- **Next Steps:**
    1.  **Expand Parser Grammar:** Incrementally add rules to the `SymbolicParser` to handle more complex sentence structures (e.g., prepositional phrases, adjectives, compound sentences).
    2.  **Develop Question Parsing:** Teach the parser to recognize and deconstruct questions (e.g., "What is a dog?") into a query that can be run against the Knowledge Graph.
    3.  **Implement Coreference Resolution:** Build a simple, deterministic mechanism to resolve basic pronouns (e.g., "it," "they") by looking at the immediate conversation history.
- **Success Metrics:** Achieve a measurable reduction in LLM fallback calls. The agent can answer simple questions it has learned the answer to, entirely through its own symbolic logic.

---

### **Phase 3: The Distributed Mind (Knowledge Scalability)**
- **Goal:** Overcome local RAM/storage limits by migrating the knowledge graph to a free-tier cloud database, enabling the agent's memory to scale to hundreds of thousands of concepts.
- **Milestone:** The agent's brain lives in a persistent, scalable cloud database (e.g., Neo4j AuraDB Free, Redis Cloud), separating the "mind" from the "machine."
- **Key Steps:**
    1.  **Migrate Graph Storage:** Replace `NetworkX` file I/O in `graph_core.py` with a connector to a cloud graph database.
    2.  **Refactor Harvester:** Update the `KnowledgeHarvester` to perform queries and writes directly against the cloud database.
    3.  **Implement Efficient Caching:** Develop a mechanism to cache relevant sub-graphs from the cloud for fast local reasoning.
- **Success Metrics:** The agent's knowledge base can grow beyond the limits of a local machine without performance degradation.

---

### **Phase 4: The Autonomous Scholar (Advanced Curriculum Learning)**
- **Goal:** Evolve the agent's learning from simple goal fulfillment to a strategic, goal-oriented "curriculum" driven by the gaps in its own understanding.
- **Milestone:** The agent can autonomously set and pursue multi-step learning goals to comprehend complex topics.
- **Key Steps:**
    1.  **Develop a Goal System:** Implement a mechanism for the agent to set high-level learning goals (e.g., "Understand photosynthesis").
    2.  **Curriculum Generation:** When a goal is set, the agent will use its parser and lexicon to generate a curriculum of prerequisite topics. Before it can understand "photosynthesis," it must first create and resolve "INVESTIGATE" goals for "plant," "sunlight," "chlorophyll," etc.
    3.  **Implement Reinforcement Heuristics:** Create a simple system that "rewards" the agent for learning facts related to its current curriculum goal, guiding its study process.
- **Success Metrics:** The agent demonstrates the ability to learn a complex topic by systematically exploring its foundational concepts first. The knowledge graph shows dense, interconnected clusters of knowledge around specific domains.
