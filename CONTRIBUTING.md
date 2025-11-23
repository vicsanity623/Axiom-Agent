# 🧠 Contributing to the Axiom Agent

Thank you for your interest in contributing to **Axiom Agent**! This is not just another chatbot; it’s an exploration into a new kind of **symbolic-first cognitive architecture** designed for genuine, self-directed learning. By contributing, you are helping to cultivate an artificial mind and push the boundaries of what AI can become.

We welcome all contributions, from documentation and bug fixes to deep architectural enhancements.

---

## 📜 Code of Conduct

This project and everyone participating in it are governed by our **Code of Conduct**. By participating, you agree to uphold this code.

➡️ Please read our [**CODE_OF_CONDUCT.md**](CODE_OF_CONDUCT.md) before contributing.

---

## ⚙️ Development Environment Setup

The project uses **[uv](https://github.com/astral-sh/uv)** for fast and reliable environment and dependency management.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vicsanity623/Axiom-Agent.git
cd Axiom-Agent
```

### 2️⃣ Create and Sync the Environment

Use `uv` to create a virtual environment and install all necessary dependencies in one command:

```bash
uv sync --extra dev
```

This command automatically creates a `.venv` directory and performs two critical actions:
1.  Installs all core and development dependencies (`ruff`, `mypy`, `pytest`).
2.  Installs the `axiom` package itself in **editable mode** (`-e`), which means changes you make in the `src/` directory are immediately available when you run any scripts.

To activate the environment in a new terminal:
```bash
source .venv/bin/activate
```

### 3️⃣ Download Required NLP Data

The agent's `SymbolicParser` relies on data packages from the `nltk` library. A helper script is provided to download and install these resources correctly into your virtual environment.

**Run this script once after setting up your environment:**

```bash
./download_resources.sh
```

### 4️⃣ Download the Local LLM

The agent's `UniversalInterpreter` (the "slow path" fallback) relies on a local GGUF-formatted language model. You can download the recommended model using the built-in CLI command:

```bash
axiom-llm
```
This will download the model to the `models/` directory.

---

## 🧩 The Development Workflow: "Train, Observe, Improve"

Axiom follows a **Check → Code → Check → Submit** workflow.

### ✅ 1. Run the Quality Gate

Before committing any changes, and after you've finished your work, always run the master quality script. This is the same script our CI pipeline uses.

```bash
./check.sh
```
This script ensures your code is compliant by running:
*   `ruff format --check` → Code Formatting
*   `ruff check` → Linting & Static Analysis
*   `mypy` → Strict Static Type Checking
*   `pytest` → The Full Unit & Integration Test Suite

> **A passing `./check.sh` is a mandatory requirement for all pull requests.**

### 🧠 2. Run the Agent & Observe its Behavior

The primary way to develop and test Axiom is to run its autonomous training process and observe its "thought process" in the logs.

| Command | Description |
|---|---|
| **Core Workflow** | |
| `axiom-studio` | **(Primary Workflow)** Launches the Integrated Development Environment. |
| `axiom-train` | Starts the agent's autonomous learning cycles (CLI mode). |
| `axiom-train --duration 1.5` | Runs the agent for a fixed duration (e.g., 1.5 hours), essential for benchmarking. |
| `axiom-teach` | (Continuous Narrative Training) Teach the agent facts directly via the terminal. |
| **Analysis & Debugging** | |
| `axiom-reset` | **(Soft Reset)** Wipes the agent's state and caches but **preserves its learned knowledge**. Use this to start a new curriculum. |
| `axiom-factory-reset` | **(Hard Reset)** Wipes **everything**, including the agent's brain, returning it to its factory state. |
| `axiom-bench` | Runs an automated analysis on the agent's brain and logs after a training run to score its learning efficiency. |
| `axiom-evaluate` | Runs a "golden question" set against the trained agent to score its reasoning ability. |

A full list of commands can be found in [**pyproject.toml**](pyproject.toml).

**Typical Development Session:**

1.  **Start with a clean slate:**
    ```bash
    # This performs a HARD reset, wiping the brain, state, and logs.
    axiom-factory-reset
    ```
2.  **Launch Axiom Studio:**
    ```bash
    axiom-studio
    ```
3.  **In Studio:**
    *   Use the **Integrated Terminal** to run `axiom-train`.
    *   Watch the **Live Graph** to observe learning progress.
    *   Use the **AI Chat** to ask questions or debug code.
4.  **Make your code changes** in the Studio editor or your preferred IDE.
5.  **Restart the process** as needed.

---

## 🚀 How to Contribute

We are currently in **Phase 6: Symbolic Core Mastery**. Our focus is on making the `SymbolicParser` and `LexiconManager` as intelligent and robust as possible to minimize reliance on the LLM.

### Your First Contribution: Enhance the `SymbolicParser`

A great first contribution is to improve the agent's "fast path" thinking. The `SymbolicParser` in `symbolic_parser.py` is responsible for understanding simple sentences. It currently handles basic SVO ("Subject-Verb-Object") patterns and questions but could be much more powerful.

**Steps:**

1.  Fork the repository and create a branch:
    ```bash
    git checkout -b feature/parser-handles-adverbs
    ```
2.  Locate the `_handle_svo_statement` method in `src/axiom/symbolic_parser.py`.
3.  **Add new logic** to handle a new grammatical pattern. For example, you could try to identify adverbs (`RB` tag from `nltk`) and add them as a `property` to the extracted `RelationData`.
4.  **Add a new test** to a relevant test file (e.g., `tests/test_symbolic_parser.py`) that specifically validates your new parsing logic.
5.  Run `./check.sh` to ensure all checks pass.
6.  Submit a Pull Request with a clear description of the improvement.

---

## 🪜 General Contribution Process

1.  **Read the Roadmap:** See the current phase in [**ROADMAP.md**](ROADMAP.md).
2.  **Find or Propose a Task:** Choose an issue or open a new one to discuss your idea.
3.  **Fork and Branch:** Use a descriptive branch name (e.g., `fix/memory-leak`, `feature/concept-merging`).
4.  **Code & Test:** Make your changes and ensure you add or update tests to cover them.
5.  **Check & Submit:** Run `./check.sh` and then submit a Pull Request with details on what you changed and why.

---

## ❤️ Thank You

Every contribution helps Axiom grow toward its goal: a system capable of autonomous, interpretable, and self-directed learning. Together, we’re building something truly extraordinary.