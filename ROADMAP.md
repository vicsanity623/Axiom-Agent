# Axiom Agent: Development Roadmap

This document outlines the current status and future architectural direction for the Axiom Agent project.

## ✅ Phase 1: The Genesis Engine (Complete)

This foundational phase is **complete**. It established a stable, high-performance cognitive engine with a professional toolchain, a scalable knowledge graph, and a complete training-to-deployment workflow running on local CPU hardware.

## ✅ Phase 2: The Symbolic Interpreter & Intelligent Learner (Complete)

This crucial phase is **complete**. It transformed the agent's autonomous learning from a random process into an intelligent, goal-driven system.

### ✅ **Core Achievements of this Phase:**
- **Symbolic-First Cognitive Flow:** The agent now attempts to understand input using its native `SymbolicParser` **before** falling back to the LLM.
- **Autonomous Vocabulary Expansion:** The agent can identify unknown words, create "INVESTIGATE" goals, and autonomously research their definitions using a high-precision **Dictionary API**.
- **Intelligent Topic Discovery:** The `Discovery Cycle` now explores curated **core subjects** and uses a **popularity heuristic** to find relevant new topics.
- **Productive Study Cycle:** The agent now has a **"Deepen Knowledge"** routine where it proactively researches concepts it already knows to discover and learn new, related facts.

---

## The Path Forward: Deepening Semantic Understanding

With the agent now capable of stable, intelligent, and symbolic-first learning, the strategic focus shifts to deepening its semantic understanding and scaling its knowledge base.

### **Phase 3: The Multi-Stage Symbolic Parser (Current Focus)**
- **Goal:** To evolve the `SymbolicParser` from a simple, monolithic function into a sophisticated, multi-stage pipeline, enabling it to deconstruct complex sentences and extract multiple atomic facts.
- **Status:** **In Progress.**
- **Completed Steps & Successes:**
    1.  **The Chunker (Clause Splitter):** The `SymbolicParser` has been successfully refactored into a pipeline. It can now split complex, multi-sentence text into individual, parsable clauses.
    2.  **Expanded Parser Grammar:** The parser's pattern-matching engine has been upgraded to understand **adjectives** (learning `has_property` relationships) and to correctly identify **questions**.
    3.  **Coreference Resolution:** A deterministic, symbolic mechanism for **short-term memory** is complete. The agent can now resolve basic pronouns (e.g., "it," "they") by referencing the subject of the previous conversational turn.
    4.  **Introspective Learning:** The agent can now **learn from its own synthesized output**. If the LLM synthesizer "leaks" a new fact, the agent can parse its own response and integrate that new knowledge into its symbolic brain.
- **Next Steps:**
    i. **Implement the Semantic Mapper:** Develop the first version of the contextual mapper with a small set of rules for common prepositions to handle more complex clauses (e.g., "lying between Africa and Asia").
    ii. **Develop Introspective Refinement:** Create a new autonomous cycle where the agent reviews its own "chunky" facts (long, un-atomic definitions) and attempts to break them down into smaller, more precise facts.
- **Success Metrics:** The agent can successfully parse a complex sentence (e.g., "The Red Sea is an inlet of the Indian Ocean, which lies between Africa and Asia") and learn **multiple, distinct, atomic facts** from it, all without LLM assistance.

---

## The Path Forward: Foundational Refinements & Scalability

With the core reasoning engine now in place, the immediate focus is on refining the agent's fundamental understanding of language to make its knowledge base more unified and its reasoning more powerful.

### **Phase 4: The Hardened Mind (Foundational Nuance & Robustness) (Current Focus)**
- **Goal:** To address the subtle but critical limitations discovered during long-term autonomous runs. This phase will focus on hardening the agent's foundational understanding of language and making its internal logic more resilient to the messy nature of real-world data.
- **Status:** **Current Focus.**
- **Key Steps / To-Do List:**
    1.  **Implement a Lemmatization Layer:** (Problem: "factory" vs "factories"). Solution: Integrate a lemmatizer into `_clean_phrase`.
    2.  **Expand the Pre-processing Pipeline:** (Problem: "what's" vs "what is"). Solution: Add a contraction expander and a smarter self-reference normalizer.
    3.  **Establish Semantic Equivalence:** (Problem: `"zero"` vs `"0"`). Solution: Seed an `is_equivalent_to` relationship and normalize concepts in the pre-processor.
    4.  **Harden the LLM Interpreter:** The interpreter methods currently crash if the LLM is disabled. Add null checks (`if self.llm is None:`) to all methods that call the LLM, allowing them to fail gracefully and providing a true symbolic-only mode for development.
    5.  **Debug and Harden the `Refinement Cycle`:** (Problem: `_find_chunky_fact` is missing candidates). Solution: Perform a focused debugging session and add more sophisticated heuristics.
    6.  **Improve Discovery Topic Filtering:** (Problem: "(disambiguation)" pages). Solution: Enhance the `reject_keywords` filter to handle parenthetical meta-tags.

- **Success Metrics:**
    *   The agent can successfully reason across singular and plural concepts.
    *   The agent correctly understands questions containing common contractions.
    *   The agent can be taught a simple mathematical fact using symbols (`0 + 1 = 1`) and correctly answer a question about it using words (`"What is zero plus one?"`).
    *   The `Refinement Cycle` reliably finds and refines at least one "chunky" fact per day during autonomous runs.

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
