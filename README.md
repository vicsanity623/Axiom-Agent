<p align="center"><img src="static/Axiom.png" alt="Axiom Agent Banner"></p>

# Axiom Agent

Axiom is a **cognitive architecture**—a framework for a new type of artificial intelligence designed to achieve genuine understanding by building its own internal, logical model of reality from the ground up.

This project’s core philosophy is that true intelligence cannot be achieved by statistical mimicry (like in traditional LLMs). It must be built on a foundation of verifiable, interconnected knowledge. **Axiom is an experiment to cultivate such a mind.**

---

## 🧠 The Core Architecture: Symbolic-First, LLM-Assisted

Axiom’s design is a radical departure from LLM-centric AI.  
It operates on a **symbolic-first** principle, where the core of the agent is a deterministic, logical brain.

1. **The Symbolic Brain (Knowledge Graph):**  
   At its heart, Axiom has a structured `ConceptGraph` — its long-term memory.  
   This logical map of concepts and relationships (e.g. `Paris --[is_located_in]--> France`) grounds the agent’s knowledge in verifiable facts and **prevents hallucinations**.

2. **The Symbolic Senses (Multi-Stage Parser):**  
   Axiom’s `SymbolicParser` deconstructs complex text into atomic logical statements, extracting multiple facts per sentence.  
   For a growing class of sentences, it achieves understanding **without any external model.**

3. **The LLM as a Tool (Fallback Interpreter):**  
   When the agent’s parser encounters a sentence too complex for its rules, it calls a local LLM **as a translation tool**.  
   The model’s only job is to produce a structured representation that the symbolic brain can then absorb.  
   The **LLM is a temporary crutch, not the mind itself.**

---

## ✅ Key Capabilities: A New Path to Understanding

This architecture enables the agent to learn, reason, and evolve in a verifiable, self-contained way.

### Cognitive & Reasoning Abilities
* **Multi-Stage Symbolic Parsing:** Understands multi-clause input and extracts multiple distinct facts.
* **Contextual Conversation:** Tracks pronouns (`it`, `they`) via a deterministic short-term memory.
* **Introspective Learning:** Can **learn from its own output**—if the LLM “leaks” a new fact, the agent parses and stores it, creating a feedback loop of self-improvement.
* **Intelligent Autonomous Learning:** Operates in cycles:
  - **Discovery Cycle:** Finds and explores new topics.
  - **Study Cycle:** Researches unknown or related concepts via a high-precision Dictionary API.
  - **Refinement Phase:** Consolidates and clarifies existing knowledge.

---

## 🔬 Evidence, Tests, and Verification

The claims of this architecture are **reproducibly testable** through automated experiments.

### 🧪 Verified Demonstrations

| Test | Description | Runs in CI? |
|------|--------------|-------------|
| [`tests/test_golden_path.py`](tests/test_golden_path.py) | The **original golden-path** demonstration. Shows full learn → query → introspect → recall loop using a real LLM. | ⚙️ Optional (requires model) |
| [`tests/test_golden_path_mocked.py`](tests/test_golden_path_mocked.py) | **Deterministic CI-safe test**. Mocks the LLM to verify the same loop without requiring a model. | ✅ Always |
| [`tests/test_introspection_suite.py`](tests/test_introspection_suite.py) | **Parameterized introspection suite**. Tests multiple concept–property pairs and collects success metrics. | ⚙️ Runs when a model is available |

The continuous-integration workflow (`.github/workflows/ci.yml`) automatically:
- Runs all static analysis and lint checks via `./check.sh`
- Executes the deterministic test suite (`pytest -m "not introspection"`)
- Optionally runs full introspection tests if a model is detected
- Uploads coverage and introspection reports as build artifacts

### 📊 Local Verification (Quickstart)

To reproduce the evidence locally:

```bash
# 1. Run the deterministic, CI-safe suite
pytest -q -m "not introspection" --disable-warnings

# 2. (Optional) Run the full introspection suite with a local model
pytest -q -m introspection --disable-warnings

# 3. View coverage results
pytest --cov=axiom --cov-report=term-missing
```

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
