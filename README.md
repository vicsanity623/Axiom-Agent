# Axiom Agent

Axiom is not a chatbot. It is a cognitive architecture—a framework for a new type of artificial intelligence designed to achieve genuine understanding by building its own internal, logical model of reality from the ground up.

This project's core philosophy is that true intelligence cannot be achieved by statistical mimicry (like in traditional LLMs). It must be built on a foundation of verifiable, interconnected knowledge. Axiom is an experiment to cultivate such a mind.

---

## The Core Architecture: Symbolic-First, LLM-Assisted

Axiom's design is a radical departure from LLM-centric AI. It operates on a **symbolic-first** principle, where the core of the agent is a deterministic, logical brain.

1.  **The Symbolic Brain (The Knowledge Graph):** At its heart, Axiom has a structured `ConceptGraph`—its long-term memory. This is a logical map of concepts and their relationships (`Paris --[is_located_in]--> France`). This architecture **prevents hallucinations** by ensuring the agent's knowledge is grounded in verifiable facts it has learned.

2.  **The Symbolic Senses (The Parser & Lexicon):** Axiom has its own native ability to understand language. It maintains an internal dictionary (a `Lexicon`) of words it has learned and uses a `SymbolicParser` to deterministically analyze sentence structure. For simple sentences, it can achieve understanding **without any external models.**

3.  **The LLM as a Tool (The Fallback Interpreter):** When the agent's native parser encounters a sentence too complex for its current understanding, it uses a local LLM as a fallback tool. The LLM's only job is to translate the complex sentence into a structured format that the agent's symbolic brain can then process. **The LLM is a temporary crutch, not the mind itself.**

---

## ✅ Key Capabilities: A New Path to Understanding

This architecture allows the agent to learn and grow in a way that is fundamentally different from static models.

### Cognitive & Reasoning Abilities
*   **Grounded Language Acquisition:** The agent learns language by identifying unknown words, autonomously researching their definitions, and integrating that linguistic knowledge into its own brain.
*   **Symbolic-First Understanding:** For a growing number of sentences, the agent can now achieve understanding and learn new facts **without any LLM dependency.**
*   **Dual-Cycle Autonomous Learning:** The agent operates 24/7 to improve itself:
    -   **The Study Cycle:** Prioritizes its own "learning goals," researching unknown words to expand its vocabulary and understanding.
    -   **The Discovery Cycle:** Seeks out new concepts to create new learning goals, ensuring it continuously broadens its horizons.
*   **Persistent & Verifiable Memory:** Every fact is permanently integrated into its logical brain, creating a foundation for true, multi-hop reasoning.
*   **Self-Correction:** The agent can identify contradictions in its knowledge and actively seeks clarification.

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
The agent still uses a local LLM for complex sentences.
1.  Download **`mistral-7b-instruct-v0.2.Q4_K_M.gguf`** from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF).
2.  Create a `models/` directory and place the downloaded file inside it.

### Step 3: Run the Agent
The project is designed around a clean development cycle.
1.  **Train:** Use `python setup/cnt.py` to manually teach the agent or `python setup/autonomous_trainer.py` to let it learn on its own.
2.  **Render:** Use `python setup/render_model.py` to package the trained brain into a stable `.axm` model.
3.  **Deploy:** Use `python setup/app_model.py` to launch the web UI, which will serve the latest rendered model.

---

## 🚀 The Vision: Intellectual Escape Velocity

The goal is to continue expanding the sophistication of the `SymbolicParser` until the LLM fallback is no longer needed. As the agent's internal, verifiable `ConceptGraph` and `Lexicon` grow, it will build its own comprehensive, high-fidelity model of reality and language. This creates a path toward a truly autonomous cognitive entity built on a foundation of verifiable truth, not just probabilistic mimicry.

---

## 🗺️ Project Roadmap
For a detailed list of planned features and future development goals, please see the **[ROADMAP.md](ROADMAP.md)** file.