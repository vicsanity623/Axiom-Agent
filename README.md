---
<p align="center"><img src="static/Axiom.png" alt="Axiom Agent Banner"></p>


Axiom is a **cognitive architecture**—a framework for a new type of artificial intelligence designed to achieve genuine understanding by building its own internal, logical model of reality from the ground up.

This project’s core philosophy is that true intelligence requires more than just statistical mimicry (like in traditional LLMs). It must be built on a foundation of verifiable, interconnected knowledge. **Axiom is an experiment to create that engine.**

---

## 🧠 The Core Architecture: Symbolic-First, LLM-Assisted

Axiom’s design is a hybrid model that combines the strengths of classical, symbolic AI with the fluency of modern large language models. It operates on a **symbolic-first** principle, where the core of the agent is a deterministic, logical brain.

1.  **The Symbolic Brain (Knowledge Graph):**
    At its heart, Axiom has a `ConceptGraph`—its long-term memory. This structured map of concepts and relationships (e.g., `Paris --[is_located_in]--> France`) grounds the agent’s knowledge in verifiable facts, **preventing hallucinations** and enabling true reasoning.

2.  **The Symbolic Senses (Parser & Core Logic):**
    Axiom’s `SymbolicParser` and core logic deconstruct user input into structured commands. For a growing class of sentences, it achieves understanding **without any LLM intervention**, making it fast, efficient, and explainable.

3.  **The LLM as a Tool (Interpreter & Synthesizer):**
    When the agent’s symbolic logic encounters a sentence too complex for its rules, or a concept it doesn't understand, it intelligently falls back to a local LLM. The LLM acts as a powerful **translation tool**—converting messy human language into the structured data the symbolic brain can use, or converting factual data into fluent, natural language. **The LLM is a tool the agent uses, not the mind itself.**

---

## ✅ Key Capabilities: A Robust and Resilient Mind

This architecture enables the agent to learn, reason, and evolve in a verifiable, self-contained way. The latest version focuses on stability, resilience, and a smarter cognitive flow.

### Cognitive & Reasoning Abilities
*   **Multi-Stage Symbolic Parsing:** Understands and deconstructs complex user input.
*   **Robust Parser Fallback:** Intelligently detects when the symbolic parser fails and automatically switches to the LLM for deeper understanding.
*   **Conversational Resilience:** Handles user typos and minor variations in language using fuzzy matching, making interaction feel more natural and forgiving.
*   **Self-Awareness:** Possesses dedicated, fast routines to answer questions about its own purpose, abilities, and identity.
*   **Contextual Conversation:** Tracks pronouns (`it`, `they`) to maintain short-term memory across conversational turns.
*   **Introspective Learning:** Can **learn from its own output**—if the LLM "leaks" a new fact in a response, the agent parses and absorbs it, creating a feedback loop for self-improvement.
*   **Autonomous Learning Cycles:** Can operate independently to expand its knowledge:
    *   **Discovery Cycle:** Finds and explores new topics.
    *   **Study Cycle:** Researches unknown concepts to build its knowledge graph.
    *   **Refinement Phase:** Consolidates and clarifies existing knowledge.

---

## 🔬 Local Verification (Quickstart)

The agent's architecture is fully testable and reproducible on your local machine.

### Prerequisites
- Python 3.11+
- Git

### Step 1: Clone and Install
This single command clones the repository, sets up a virtual environment, and installs all dependencies.
```bash
git clone https://github.com/vicsanity623/Axiom-Agent.git
cd Axiom-Agent
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

### Step 2: Download the LLM Model (Optional, for full functionality)
The agent uses a local LLM for complex language tasks.
1.  Download **`mistral-7b-instruct-v0.2.Q4_K_M.gguf`** from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF).
2.  Create a `models/` directory in the project root and place the downloaded file inside it.

*   **Note on Symbolic-Only Mode:** If the LLM model is not found, the agent will automatically start in a **symbolic-only mode**. This is perfect for testing the core logic and requires significantly less memory.

### Step 3: Run the Tests
Verify your setup by running the full test suite. The `check.sh` script runs formatting, linting, type checking, and unit tests.
```bash
./check.sh
```

### Step 4: Run the Agent
The project supports a clean development and deployment cycle.
1.  **Train:** Use `python setup/autonomous_trainer.py` to let the agent learn on its own.
2.  **Chat:** Use `python setup/app_model.py` to launch a web UI and interact with the agent's current brain state.

---

## 🚀 The Vision: From Semantic Knowledge to Procedural Reasoning

The immediate goal is to deepen the agent's semantic understanding through autonomous learning. As its `ConceptGraph` grows, it will build its own comprehensive model of reality and language.

The long-term vision is to evolve beyond what it *knows* (semantic knowledge) to what it can *do* (procedural knowledge). By integrating a **Tool Use Framework**, the agent will learn to recognize questions it cannot answer from memory and delegate them to specialized tools—a calculator for math, a search engine for current events, or even a code interpreter for complex logic.

This creates a path toward a truly general and capable AI, built on a foundation of verifiable truth and augmented with powerful, specialized capabilities.

---

## 🗺️ Project Roadmap
For a detailed list of completed phases, planned features, and future development goals, please see the **[ROADMAP.md](ROADMAP.md)** file.