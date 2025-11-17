<p align="center">
  <img src="https://raw.githubusercontent.com/vicsanity623/Axiom-Agent/main/src/axiom/static/Axiom.png" alt="Axiom Agent Banner" width="80%">
</p>

Axiom is a **cognitive architecture**—a blueprint for a new type of artificial intelligence. Its mission is to achieve **Intellectual Escape Velocity**: a state of genuine, self-directed learning where its understanding grows faster than the complexity of the problems it encounters.

This project’s core philosophy is that true intelligence cannot be achieved by statistical mimicry alone. It must be built on a **verifiable, symbolic core** that allows for genuine reasoning. **Axiom is an experiment to build that core.**

---

## 🧠 The Architecture: Symbolic-First, LLM-Assisted

Axiom’s design is a hybrid model that fuses the deterministic logic of symbolic AI with the linguistic fluency of modern LLMs. It operates on a **Symbolic-First** principle: the agent always attempts to "think for itself" using its fast, internal logic before resorting to the slower, more expensive LLM.

1.  **The Symbolic Core (The "Fast Path"):**
    *   **Lexicon & Parser:** The agent's `LexiconManager` and `SymbolicParser` form a high-speed engine for understanding language. By analyzing grammar first (`POS-First`), the parser can deconstruct simple sentences into structured, logical facts in milliseconds, **with zero LLM calls.**
    *   **Knowledge Graph:** Axiom’s `ConceptGraph` is its long-term memory. This structured map of concepts and relationships (e.g., `(cat) --[is_a]--> (animal)`) grounds the agent’s understanding in verifiable facts, enabling true reasoning and preventing hallucinations.

2.  **The LLM as a Tool (The "Slow Path"):**
    *   When the Symbolic Core encounters a sentence that is too complex or ambiguous, it gracefully fails and the agent falls back to its `UniversalInterpreter`. This component uses a local LLM as a powerful tool to translate messy human language into the structured, logical data that the Symbolic Core can understand. **The LLM is a tool, not the mind.**

This two-tiered system allows Axiom to be both fast and robust, combining the speed of symbolic logic with the flexibility of neural networks.

---

## 🧬 The Cognitive Loop: How the Agent Thinks and Grows

The agent runs in a continuous, multi-phase loop of learning and self-improvement, orchestrated by the `CycleManager`.

### High-Level Cognitive Phases

```mermaid
%%{init: { "theme": "forest" }}%%
graph TD
    A["Phase 1: Learning<br/>(Acquire new knowledge from curriculum)"]
    B["Phase 2: Discovery<br/>(Explore the web for new topics)"]
    C["Phase 3: Refinement<br/>(Decompose complex facts)"]
    D["Phase 4: Metacognition<br/>(Analyze and improve its own code)"]

    A --> B --> C --> D --> A
```

---

### The Core Learning Process: The "Symbolic First" Pipeline

This diagram shows how the agent processes a new piece of information, always prioritizing its fast, internal logic.

```mermaid
%%{init: { "theme": "forest" }}%%
flowchart TD
    A["Input Sentence<br/>(e.g., 'A cat is an animal')"] --> B{Attempt Symbolic Parse}

    B -- Success --> C["Symbolic Success<br/>• Fact understood in milliseconds<br/>• No LLM call needed"]
    B -- Failure --> D["Symbolic Failure<br/>(Sentence too complex)"]

    D --> E["Fallback to LLM<br/>(Universal Interpreter)"]
    E --> F["Structured Data<br/>(Relation Object)"]

    C --> G["Knowledge Graph"]
    F --> G["Knowledge Graph"]
```

---

## ✅ Key Capabilities

*   **Symbolic-First Architecture:** Prioritizes fast, deterministic, and low-cost internal reasoning.
*   **Hybrid Intelligence:** Combines a rule-based `SymbolicParser` with an LLM-based `UniversalInterpreter`.
*   **Autonomous Learning Cycles:** Continuously runs through Learning, Discovery, and Refinement phases to expand its knowledge.
*   **Metacognitive Self-Improvement:** Can analyze its own performance logs and suggest improvements to its source code.
*   **Long-Term Memory:** All learned knowledge is stored persistently in a structured knowledge graph.

---

## ⚖️ How Axiom Compares to a Standard LLM

```mermaid
%%{init: { "theme": "forest" }}%%
flowchart LR

    subgraph Axiom ["Axiom Agent"]
        direction TB
        A1["Mind:<br/>Symbolic Knowledge Graph"]
        A2["Senses:<br/>Symbolic Parser (Primary)<br/>LLM (Fallback)"]
        A3["Learning:<br/>Continuous & Autonomous"]
        A4["Reliability:<br/>Verifiable & Explainable"]
    end

    subgraph LLM ["Traditional LLM"]
        direction TB
        L1["Mind:<br/>Neural Network Weights"]
        L2["Senses:<br/>Statistical Token Patterns"]
        L3["Learning:<br/>Offline & Static"]
        L4["Reliability:<br/>Probabilistic & Opaque"]
    end

    classDef axiom fill:#DFF6FF,stroke:#007ACC,stroke-width:2px
    classDef llm fill:#FFEFD5,stroke:#FF6600,stroke-width:2px

    class A1,A2,A3,A4 axiom
    class L1,L2,L3,L4 llm
```

---

## 🚀 The Vision: The Path to Apex AI

The long-term goal of Axiom is to create a system that achieves **Intellectual Escape Velocity**. This is a state where the agent's ability to acquire and integrate new knowledge outpaces the growth of complexity in its environment.

By building a powerful and efficient symbolic core, we reduce the agent's reliance on slow, expensive LLMs. This allows it to learn, reason, and adapt at an ever-increasing rate, creating a virtuous cycle of intelligence.

Future expansions include:
*   A fully self-improving `SymbolicParser` that learns new grammatical rules.
*   A framework for tool use and procedural knowledge (learning "how" to do things).
*   Advanced reasoning capabilities, such as analogical and causal inference.

---

## 🔬 Getting Started & Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for a full guide on how to set up your development environment, run the agent, and contribute to the project.

---

## 🗺️ Project Roadmap

See **[ROADMAP.md](ROADMAP.md)** for complete milestones and plans.