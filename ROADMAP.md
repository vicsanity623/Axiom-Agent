# Axiom Agent: Development Roadmap

This document outlines the current status and future architectural direction for the Axiom Agent project.

## ✅ Phase 1: The Genesis Engine (Complete)

This foundational phase is **complete**. It established a stable, high-performance cognitive engine with a professional toolchain, a scalable knowledge graph, and a complete training-to-deployment workflow running on local CPU hardware.

---

## ✅ Phase 2: The Symbolic Interpreter & Intelligent Learner (Complete)

This crucial phase is **complete**. The project has successfully pivoted to a **Symbolic-First Architecture**, systematically replacing the original LLM dependency with a more robust, deterministic, and intelligent system for learning and understanding language.

### ✅ **Core Achievements of this Phase:**
- **Symbolic-First Cognitive Flow:** The agent now attempts to understand input using its native `SymbolicParser` **before** falling back to the LLM, marking a fundamental shift in its architecture.
- **Autonomous Vocabulary Expansion:** The agent can identify unknown words, create "INVESTIGATE" goals, and autonomously research their definitions using a high-precision **Dictionary API**, with a web-scraping fallback for resilience.
- **Intelligent Topic Discovery:** The `Discovery Cycle` is no longer random. It now explores curated **core subjects** and uses a **popularity heuristic** to find relevant new topics, leading to a more useful learning path.
- **Productive Study Cycle:** The agent now has a **"Deepen Knowledge"** routine where it proactively researches concepts it already knows to discover and learn new, related facts, enriching its understanding.
- **Expanded Parser Grammar (Adjectives & Questions):** The `SymbolicParser` has been upgraded to understand adjectives (learning `has_property` relationships) and to correctly identify and parse simple questions.

---

## ✅ Phase 3: Advanced Symbolic Reasoning (Complete)

This phase is now **complete**. The agent has successfully moved beyond simple fact recall to true symbolic reasoning. Its understanding of grammar and context has been significantly deepened, and it can now infer knowledge by combining facts it has learned.

### ✅ **Core Achievements of this Phase:**
- **Expanded Parser Grammar (Prepositions):** The `SymbolicParser` was upgraded to understand prepositional phrases (e.g., "The book is **on the table**," "Paris is **in France**"), allowing it to learn more complex and precise factual relationships.
- **Deterministic Coreference Resolution:** A non-LLM mechanism for resolving pronouns (`it`, `they`) was implemented. The agent now uses the immediate conversation history to understand simple contextual follow-up questions.
- **Introspective Knowledge Refinement:** A new autonomous `refinement_cycle` was created. This allows the agent to proactively review its own "chunky," un-atomic facts and use the LLM to break them down into smaller, higher-quality, atomic facts, thereby improving the integrity of its knowledge base.
- **Multi-Hop Symbolic Reasoning:** The agent is now capable of **Multi-Hop Symbolic Reasoning**. It can answer a question by logically combining multiple, separate facts from its knowledge base without LLM assistance (e.g., answering "Is Socrates a mortal?" by combining the facts `Socrates is a human` and `A human is a mortal`).

---

## The Path Forward: Scalability and Strategy

With the core symbolic learning and reasoning engine now in place, the strategic focus shifts from *intelligence* to *scalability* and *strategic learning*.

### **Phase 4: The Distributed Mind (Knowledge Scalability) (Current Focus)**
- **Goal:** Overcome local RAM/storage limits by migrating the knowledge graph to a free-tier cloud database, enabling the agent's memory to scale to hundreds of thousands of concepts.
- **Status:** **Current Focus.**
- **Next Steps:**
    - **Milestone:** The agent's brain lives in a persistent, scalable cloud database (e.g., Neo4j AuraDB Free, Redis Cloud), separating the "mind" from the "machine."
- **Success Metrics:** The agent's knowledge base can grow beyond the limits of a local machine without performance degradation. The autonomous training can run on one machine while the interactive agent runs on another, both connected to the same cloud-based brain.

### **Phase 5: The Autonomous Scholar (Advanced Curriculum Learning)**
- **Goal:** Evolve the agent's learning from simple goal fulfillment to a strategic, goal-oriented "curriculum" driven by the gaps in its own understanding.
- **Status:** **Not Started.**
- **Next Steps:**
    1.  **Curriculum Generation:** When a high-level goal is set (e.g., "Understand photosynthesis"), the agent will use its parser and lexicon to generate a curriculum of prerequisite topics ("plant," "sunlight," "chlorophyll," etc.) and create `INVESTIGATE` goals for them.
    2.  **Implement Reinforcement Heuristics:** Create a simple system that "rewards" the agent for learning facts related to its current curriculum goal, guiding its study process.
- **Success Metrics:** The agent demonstrates the ability to learn a complex topic by systematically exploring its foundational concepts first. The knowledge graph shows dense, interconnected clusters of knowledge around specific domains.
