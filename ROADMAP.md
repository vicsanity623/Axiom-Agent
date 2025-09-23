# Axiom Agent: Development Roadmap

This document outlines the current status and future architectural direction for the Axiom Agent project.

## ✅ Phase 1: The Genesis Engine (Complete)

This foundational phase is **complete**. It established a stable, high-performance cognitive engine with a professional toolchain, a scalable knowledge graph, and a complete training-to-deployment workflow running on local CPU hardware.

---

## ✅ Phase 2: The Symbolic Interpreter & Intelligent Learner (Complete)

This crucial phase is now **complete**. The project has successfully pivoted to a **Symbolic-First Architecture**, systematically replacing the original LLM dependency with a more robust, deterministic, and intelligent system for learning and understanding language.

### ✅ **Core Achievements of this Phase:**
- **Symbolic-First Cognitive Flow:** The agent now attempts to understand input using its native `SymbolicParser` **before** falling back to the LLM, marking a fundamental shift in its architecture.
- **Autonomous Vocabulary Expansion:** The agent can identify unknown words, create "INVESTIGATE" goals, and autonomously research their definitions using a high-precision **Dictionary API**, with a web-scraping fallback for resilience.
- **Intelligent Topic Discovery:** The `Discovery Cycle` is no longer random. It now explores curated **core subjects** and uses a **popularity heuristic** to find relevant new topics, leading to a more useful learning path.
- **Productive Study Cycle:** The agent now has a **"Deepen Knowledge"** routine where it proactively researches concepts it already knows to discover and learn new, related facts, enriching its understanding.
- **Expanded Parser Grammar (Adjectives & Questions):** The `SymbolicParser` has been upgraded to understand adjectives (learning `has_property` relationships) and to correctly identify and parse simple questions.

---

## The Path Forward: Deepening Semantic Understanding

With the agent now capable of stable, intelligent, and symbolic-first learning, the strategic focus shifts to deepening its semantic understanding and scaling its knowledge base.

### **Phase 3: Advanced Symbolic Reasoning (Current Focus)**
- **Goal:** To expand the `SymbolicParser`'s grammatical capabilities and the agent's reasoning logic, allowing it to understand and answer more complex queries without LLM assistance.
- **Status:** **In Progress.**
- **Next Steps:**
    1.  **Expand Parser Grammar (Prepositions):** The next major grammatical challenge is to teach the parser to handle prepositional phrases (e.g., "The book is **on the table**," "Paris is **in France**"), which will dramatically increase the complexity of facts it can learn.
    2.  **Implement Coreference Resolution:** Build a simple, deterministic mechanism to resolve basic pronouns (e.g., "it," "they") by looking at the immediate conversation history.
    3.  **Develop Introspective Refinement:** Create a new autonomous cycle where the agent reviews its own "chunky" facts (long, un-atomic definitions) and attempts to break them down into smaller, more precise facts.
- **Success Metrics:** Achieve a measurable reduction in LLM fallback calls. The agent can answer multi-part questions by combining facts it has learned symbolically.

### **Phase 4: The Distributed Mind (Knowledge Scalability)**
- **Goal:** Overcome local RAM/storage limits by migrating the knowledge graph to a free-tier cloud database, enabling the agent's memory to scale to hundreds of thousands of concepts.
- **Milestone:** The agent's brain lives in a persistent, scalable cloud database (e.g., Neo4j AuraDB Free, Redis Cloud), separating the "mind" from the "machine."
- **Success Metrics:** The agent's knowledge base can grow beyond the limits of a local machine without performance degradation.

### **Phase 5: The Autonomous Scholar (Advanced Curriculum Learning)**
- **Goal:** Evolve the agent's learning from simple goal fulfillment to a strategic, goal-oriented "curriculum" driven by the gaps in its own understanding.
- **Milestone:** The agent can autonomously set and pursue multi-step learning goals to comprehend complex topics.
- **Key Steps:**
    1.  **Curriculum Generation:** When a high-level goal is set (e.g., "Understand photosynthesis"), the agent will use its parser and lexicon to generate a curriculum of prerequisite topics ("plant," "sunlight," "chlorophyll," etc.) and create `INVESTIGATE` goals for them.
    2.  **Implement Reinforcement Heuristics:** Create a simple system that "rewards" the agent for learning facts related to its current curriculum goal, guiding its study process.
- **Success Metrics:** The agent demonstrates the ability to learn a complex topic by systematically exploring its foundational concepts first. The knowledge graph shows dense, interconnected clusters of knowledge around specific domains.
