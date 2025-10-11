<p align="center"><img src="static/Axiom.png" alt="Axiom Agent Banner"></p>

# Axiom Agent

Axiom is a **cognitive architecture**—a framework for a new type of artificial intelligence designed to achieve genuine understanding by building its own internal, logical model of reality from the ground up.

This project's core philosophy is that true intelligence cannot be achieved by statistical mimicry (like in traditional LLMs). It must be built on a foundation of verifiable, interconnected knowledge. Axiom is an experiment to cultivate such a mind.

---

## The Core Architecture: Symbolic-First, LLM-Assisted

Axiom's design is a radical departure from LLM-centric AI. It operates on a **symbolic-first** principle, where the core of the agent is a deterministic, logical brain.

1.  **The Symbolic Brain (The Knowledge Graph):** At its heart, Axiom has a structured `ConceptGraph`—its long-term memory. This is a logical map of concepts and their relationships (`Paris --[is_located_in]--> France`). This architecture **prevents hallucinations** by ensuring the agent's knowledge is grounded in verifiable facts it has learned.

2.  **The Symbolic Senses (The Multi-Stage Parser):** Axiom has its own native ability to understand language. It uses a `SymbolicParser` that operates as a multi-stage pipeline, first splitting complex sentences into simpler clauses, then applying a series of grammatical rules to extract atomic facts. For a growing number of sentences, it can achieve understanding **without any external models.**

3.  **The LLM as a Tool (The Fallback Interpreter):** When the agent's native parser encounters a sentence too complex for its current understanding, it uses a local LLM as a fallback tool. The LLM's only job is to translate the complex sentence into a structured format that the agent's symbolic brain can then process. **The LLM is a temporary crutch, not the mind itself.**

---

## ✅ Key Capabilities: A New Path to Understanding

This architecture allows the agent to learn, reason, and grow in a way that is fundamentally different from static models.

### Cognitive & Reasoning Abilities
*   **Multi-Stage Symbolic Parsing:** The agent's `SymbolicParser` can deconstruct complex, multi-sentence text into individual clauses and learn multiple, distinct facts from a single input.
*   **Contextual Conversation:** The agent possesses a deterministic **coreference resolution** mechanism for short-term memory, allowing it to understand what pronouns like `it` and `they` refer to based on the recent conversation history.
*   **Introspective Learning:** The agent can **learn from its own output**. If its LLM synthesizer "leaks" a new fact while answering a question, the agent parses its own response and integrates that new knowledge into its symbolic brain, creating a powerful self-improvement loop.
*   **Intelligent Autonomous Learning:** The agent operates 24/7 with two focused learning cycles:
    -   **The Discovery Cycle:** Explores curated subjects and uses a popularity heuristic to find relevant new topics.
    -   **The Study Cycle:** Prioritizes researching unknown words from its learning queue using a high-precision **Dictionary API**. When the queue is empty, it proactively **deepens its knowledge** by researching concepts it already knows to find new, related facts.
    -   **Refinement Phase** Agent spends hours introspectively consolidate and improve what it has learned.

<br/>

### Professional Development & Deployment
*   **Automated Quality Assurance (`./check.sh`):** A full suite of tools (Ruff, MyPy, Pytest) ensures the codebase is clean, consistent, and reliable.
*   **Modern Python Packaging (`pyproject.toml`):** The project uses the latest standards for managing dependencies and configuration.
*   **Robust Train -> Render -> Deploy Workflow:** A professional toolchain separates offline training (in `cnt.py` or `autonomous_trainer.py`) from online, read-only deployment (in `app_model.py`), ensuring stability.

---

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.11+
- Git

### Step 1: Clone and Install
This single command clones the repository, sets up a virtual environment, and installs all project and development dependencies.
```bash
git clone https://github.com/vicsanity623/Axiom-Agent.git
cd Axiom-Agent
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

### Step 2: Download the LLM Model (Optional, for full functionality)
The agent uses a local LLM for complex sentences and introspective learning.
1.  Download **`mistral-7b-instruct-v0.2.Q4_K_M.gguf`** from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF).
2.  Create a `models/` directory in the project root and place the downloaded file inside it.

<br/>

*   **Note on Symbolic-Only Mode:** If the LLM model is not found, the agent will automatically start in a **symbolic-only mode**. This is perfect for testing the core logic and requires significantly less memory, though some fallback and creative features will be disabled.

<br/>

### Step 3: Run the Agent
The project is designed around a clean development cycle.
1.  **Train:** Use `python setup/cnt.py` to interactively teach the agent
2.  or `python setup/autonomous_trainer.py` to let it learn on its own.
3.  **Render:** Use `python setup/render_model.py` to package the trained brain into a stable `.axm` model.
4.  **Deploy:** Use `python setup/app_model.py` to launch the web UI, which will serve the latest rendered model.

---

## 🚀 The Vision: Intellectual Escape Velocity

The goal is to continue expanding the sophistication of the `SymbolicParser` until the LLM fallback is no longer needed. As the agent's internal, verifiable `ConceptGraph` and `Lexicon` grow, it will build its own comprehensive, high-fidelity model of reality and language. This creates a path toward a truly autonomous cognitive entity built on a foundation of verifiable truth, not just probabilistic mimicry.

---

## 🗺️ Project Roadmap
For a detailed list of planned features and future development goals, please see the **[ROADMAP.md](ROADMAP.md)** file.
