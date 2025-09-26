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

### **Phase 6: The Hardened Mind (Foundational Nuance & Robustness)**

-   **Goal:** To address the subtle but critical limitations discovered during long-term autonomous runs. This phase will focus on hardening the agent's foundational understanding of language and making its internal logic more resilient to the messy nature of real-world data.

-   **Status:** **Not Started.**

-   **Key Steps / To-Do List:**

    1.  **Implement a Lemmatization Layer:**
        *   **Problem:** The agent treats singular and plural words (e.g., "factory" vs. "factories") as completely separate concepts. This is the single biggest barrier to deeper reasoning.
        *   **Solution:** Integrate a lemmatizer (e.g., from `nltk`) into the `_clean_phrase` method. This will reduce all concepts to their dictionary root form before they are stored or queried, unifying the agent's knowledge graph.

    2.  **Expand the Pre-processing Pipeline:**
        *   **Problem:** The agent doesn't understand common linguistic variations. It fails on contractions ("what's" vs. "what is") and its self-reference replacement is too naive (incorrectly changing "You" in a song title).
        *   **Solution:** Create a dedicated pre-processing pipeline that runs early in the `chat` method. This will include a **contraction expander** and a smarter, **context-aware self-reference normalizer.**

    3.  **Establish Semantic Equivalence:**
        *   **Problem:** The agent has no concept that different strings can represent the same idea (e.g., `"zero"` vs. `"0"`, `"plus"` vs. `"+"`). This makes abstract reasoning, like mathematics, impossible.
        *   **Solution:** Begin seeding a new type of relationship, `is_equivalent_to`, in the knowledge base. The pre-processing pipeline can then use this knowledge to normalize concepts (e.g., converting all instances of "zero" to "0") before the main cognitive engine sees the input.

    4.  **Debug and Harden the `Refinement Cycle`:**
        *   **Problem:** The `_find_chunky_fact` method is failing to identify obvious candidates for refinement, and the LLM interpreter sometimes fails to parse the decomposed sentences.
        *   **Solution:** Perform a focused debugging session on `_find_chunky_fact`. Add more sophisticated heuristics to improve its accuracy. Harden the `UniversalInterpreter` to be more resilient to malformed JSON from the LLM.

    5.  **Improve Discovery Topic Filtering:**
        *   **Problem:** The `discover_cycle` still occasionally selects low-quality meta-pages like "(disambiguation)".
        *   **Solution:** Enhance the `reject_keywords` filter in `_find_new_topic` to correctly parse and reject topics containing these parenthetical meta-tags.

-   **Success Metrics:**
    *   The agent can successfully reason across singular and plural concepts.
    *   The agent correctly understands questions containing common contractions.
    *   The agent can be taught a simple mathematical fact using symbols (`0 + 1 = 1`) and correctly answer a question about it using words (`"What is zero plus one?"`).
    *   The `Refinement Cycle` reliably finds and refines at least one "chunky" fact per day during autonomous runs.

---
