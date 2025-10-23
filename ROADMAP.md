# Axiom Agent — Evolving Mind Roadmap

This roadmap documents the evolution of the Axiom Agent, from its foundational architecture to its future as a sophisticated, autonomous reasoner. It's organized by **phases of evolution**. Each phase represents a significant leap in capability and includes **goals**, **deliverables**, and **key features**.

***

The Axiom Agent has successfully evolved through its foundational phases. It began with the **"Genesis Engine,"** establishing a stable knowledge graph. The **"Intelligent Learner"** phase enabled autonomous discovery, while the **"Conversational Mind"** gave it the ability to parse complex sentences and learn from its own responses.

Following this rapid innovation, the Axiom Agent has now completed a critical **"Stabilization & Hardening"** phase. This engineering-focused effort addressed technical debt, dramatically improved the user experience, and fortified the agent's core logic with a comprehensive test suite and more intelligent learning mechanisms. This essential work has created a robust and maintainable platform, paving the way for the agent's next major functional leaps: achieving true **"Semantic Mastery"** with a deeper understanding of language, becoming an **"Autonomous Scholar"** with strategic learning, and finally, evolving into a **"Logical Reasoner"** capable of procedural thought and tool use.

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
*   **Centralized & Robust Architecture:** Refactored the entire project to use a central `config.py` for all file paths, eliminating dozens of bugs. The core learning logic was fortified with a robust deferral system for facts with untrusted words.
*   **LLM-Powered Fact Verification:** Implemented an LLM-driven "quality gate" (`verify_and_reframe_fact`) that critically evaluates facts found on the web. It rejects irrelevant information and reframes good facts into simple, learnable sentences, fixing major intelligence flaws (e.g., synonym/redirect bugs).
*   **Real-Time Interactive Learning:** The agent's "Cognitive Reflex" can now perform real-time research. When it encounters an unknown word during conversation, it immediately attempts to define it, enabling a dynamic, interactive learning loop.
*   **Intelligent Lexicon Promotion:** Created a formal "promotion" system for the lexicon. The agent now distinguishes between knowing a word's definition and trusting it, and automatically promotes words learned from high-confidence sources (like dictionaries or LLM verification).
*   **Comprehensive Test Suite:** Achieved a fully passing, clean CI/CD pipeline (Ruff, MyPy, Pytest). The test suite was completely overhauled to validate the new, more sophisticated learning and deferral behaviors.
*   **Streamlined User Experience:** Refactored developer scripts into professional command-line entry points (`axiom-train`, `axiom-teach`, etc.) and automated the LLM download process (`axiom-llm`).

---

# Phase 4 — Semantic Mastery

**Goal:** Deepen the agent's understanding of language nuance, context, and complex relationships.

### Key Features
*   **Knowledge Provenance & Confidence Scoring:** Add metadata to all learned facts, tracking their source (`user`, `llm_verified`, `dictionary`), timestamp, and confidence score. This is a prerequisite for belief revision.
*   **Advanced Relationship Extraction:** Enhance the `UniversalInterpreter` to extract and represent multi-part, qualified relationships (e.g., `(person, born_in, city, on_date, date)`), moving beyond simple S-V-O.
*   **Belief Revision System:** Implement a strategy for resolving conflicting facts. When a contradiction is found, the agent will use provenance and confidence scores to decide whether to update its belief, ask for clarification, or maintain both possibilities.
*   **Sentiment and Tone Analysis:** Implement a mechanism to track the sentiment of the user's input and adjust the tone of synthesized responses accordingly, making conversation feel more empathetic and natural.

---

# Phase 5 — Autonomous Scholar

**Goal:** Transform the agent from a passive learner into a strategic researcher with long-term goals.

### Key Features
*   **Goal-Oriented Learning:** Introduce a `GoalManager` that allows the agent to be given high-level learning objectives (e.g., "Become an expert on ancient Rome").
*   **Strategic Study Plans:** The `KnowledgeHarvester` will use its learning goals to generate and execute multi-step study plans (e.g., find a Wikipedia page, read it, identify key entities, generate curious questions for each entity, then research those questions).
*   **Dynamic Knowledge Pruning:** Implement a "forgetting" mechanism where facts that are consistently low-activation and low-confidence are periodically removed to keep the graph relevant.

---

# Phase 6 — The Logical Reasoner (Procedural Thought)

**Goal:** Evolve beyond semantic knowledge to incorporate procedural and logical reasoning by introducing "Tools."

### Key Features
*   **Tool Use Framework:** Create a `ToolManager` and a formal "Tool" interface that the agent can use. Each tool will be a specialized function for tasks the knowledge graph cannot perform.
*   **Mathematical Capability (First Tool):**
    *   **Deliverable:** A `MathTool` that integrates a symbolic math library like `SymPy`.
    *   **Logic:** The `SymbolicParser` and `UniversalInterpreter` will be trained to recognize a new `question_math` intent.
    *   **Flow:** When this intent is detected, `_process_intent` will route the mathematical expression to the `MathTool` for execution, and the result will be returned to the user.
*   **Real-Time Data (Second Tool):**
    *   **Deliverable:** A `WebSearchTool` (e.g., using a Google/DuckDuckGo API) and a `CurrentDateTool`.
    *   **Logic:** The agent will be taught to recognize questions it cannot answer from its static knowledge (e.g., "What is the weather in New York?", "Who won the game last night?").
    *   **Flow:** The agent will learn to route these questions to the appropriate tool to fetch live data from the internet.
*   **Code Execution (Advanced Tool):**
    *   **Deliverable:** A sandboxed `PythonInterpreterTool` that can execute simple Python code to answer complex procedural questions.
    *   **Flow:** For questions like "What are the first 10 prime numbers?", the agent will learn to write and execute a small script instead of trying to find the answer in its knowledge graph.

---

# Phase 7 — Distributed Mind

**Goal:** Prepare the agent for massive scalability and collaborative learning.

### Key Features
*   **Pluggable Graph Backends:** Refactor `ConceptGraph` to be an interface, with the current in-memory NetworkX implementation as the default and a new implementation for a graph database (e.g., Neo4j, RedisGraph) as an option for large-scale deployments.
*   **Agent Federation:** Design a protocol that allows multiple Axiom Agents to query each other's knowledge, share facts, and collaborate on learning goals.