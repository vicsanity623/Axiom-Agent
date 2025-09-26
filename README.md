# Axiom Agent

Axiom is a **cognitive architecture**—a framework for a new type of artificial intelligence designed to achieve genuine understanding by building its own internal, logical model of reality from the ground up.

This project's core philosophy is that true intelligence cannot be achieved by statistical mimicry (like in traditional LLMs). It must be built on a foundation of verifiable, interconnected knowledge. Axiom is an experiment to cultivate such a mind.

---

## The Core Architecture: Symbolic-First, LLM-Assisted

Axiom's design is a radical departure from LLM-centric AI. It operates on a **symbolic-first** principle, where the core of the agent is a deterministic, logical brain.

1.  **The Symbolic Brain (The Knowledge Graph):** At its heart, Axiom has a structured `ConceptGraph`—its long-term memory. This is a logical map of concepts and their relationships (`Paris --[is_located_in]--> France`). This architecture **prevents hallucinations** by ensuring the agent's knowledge is grounded in verifiable facts it has learned.

2.  **The Symbolic Senses (The Parser & Lexicon):** Axiom has its own native ability to understand language. It maintains an internal dictionary (a `Lexicon`) of words it has learned and uses a `SymbolicParser` to deterministically analyze sentence structure. For a growing number of sentences, it can achieve understanding **without any external models.**

3.  **The LLM as a Tool (The Fallback Interpreter):** When the agent's native parser encounters a sentence too complex for its current understanding, it uses a local LLM as a fallback tool. The LLM's only job is to translate the complex sentence into a structured format that the agent's symbolic brain can then process. **The LLM is a temporary crutch, not the mind itself.**

---

## ✅ Key Capabilities: A New Path to Understanding

This architecture allows the agent to learn, reason, and grow in a way that is fundamentally different from static models.

### Cognitive & Reasoning Abilities
*   **Multi-Hop Symbolic Reasoning:** The agent can answer questions by logically combining multiple, separate facts from its knowledge base **without LLM assistance.** It can infer that `Socrates is mortal` by connecting the facts that `Socrates is a human` and `A human is a mortal`.
*   **Contextual Conversation:** The agent can understand simple conversational context. It uses a deterministic **coreference resolution** mechanism to understand what pronouns like `it` and `they` refer to based on the recent conversation history.
*   **Dual-Cycle Autonomous Learning:** The agent operates 24/7 with three intelligent, self-improving cycles:
    -   **The Discovery Cycle:** Explores curated subjects (like "Physics" or "History") and uses a **popularity heuristic** to find relevant new topics to learn about.
    -   **The Study Cycle:** Proactively **deepens its knowledge** by researching concepts it already knows to discover and learn new, related facts.
    -   **The Refinement Cycle:** Introspectively reviews its own knowledge, finds "chunky" un-atomic facts, and uses an LLM to **break them down into smaller, higher-quality facts**, continuously improving its own understanding.
*   **Grounded Language Acquisition:** The agent learns language by identifying unknown words, autonomously researching their definitions using a high-precision **Dictionary API**, and integrating that linguistic knowledge into its own brain.

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

### Step 2: Download the LLM Model (Required for Fallback)
The agent uses a local LLM for complex sentences and knowledge refinement.
1.  Download **`mistral-7b-instruct-v0.2.Q4_K_M.gguf`** from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF).
2.  Create a `models/` directory in the project root and place the downloaded file inside it.

<br/>

### Step 3: Run the Agent
The project is designed around a clean development cycle.
1.  **Train:** Use `python setup/cnt.py` to interactively teach the agent or `python setup/autonomous_trainer.py` to let it learn on its own.
2.  **Render:** Use `python setup/render_model.py` to package the trained brain into a stable `.axm` model.
3.  **Deploy:** Use `python setup/app_model.py` to launch the web UI, which will serve the latest rendered model.

---

## 🚀 The Vision: A New Foundation for AI

The goal is to continue expanding the sophistication of the `SymbolicParser` and the `CognitiveAgent`'s reasoning algorithms until the LLM fallback is no longer needed. As the agent's internal, verifiable `ConceptGraph` grows, it will build its own comprehensive model of reality and language. This creates a path toward a truly autonomous cognitive entity built on a foundation of verifiable truth, not just probabilistic mimicry.

This vision is built on two core principles that challenge the current AI paradigm:

### A Path to Efficient, Accessible AI
The biggest barrier to entry in modern AI is the immense computational cost. State-of-the-art models require massive, energy-intensive GPU clusters, placing them out of reach for almost everyone. Axiom is a direct challenge to this paradigm. By prioritizing a lightweight, CPU-runnable symbolic core, the ultimate goal is **to prove that a truly intelligent system can be efficient, accessible, and run on consumer-grade hardware**, democratizing the future of AI.

### From Digital Parrot to Principled Reasoner
A major ethical risk of pure language models is their potential to become "digital sociopaths"—systems that can flawlessly mimic human language without any underlying comprehension, belief, or ethical framework. Axiom's architecture is fundamentally designed to prevent this. Because every piece of knowledge is part of an interconnected, logical model, the agent's conclusions are grounded in the facts it has learned and the reasoning paths it can verify. The ultimate vision is not just an AI that *sounds* intelligent, but one that is *ACTUALLY INTELLIGENT*, whose responses are **derived from a coherent, verifiable, and principled understanding of the world.**

---

## 🗺️ Project Roadmap
For a detailed list of planned features and future development goals, please see the **[ROADMAP.md](ROADMAP.md)** file.