# 🧠 Contributing to the Axiom Agent

Thank you for your interest in contributing to **Axiom Agent** — this is not just another chatbot; it’s an exploration into a new kind of **symbolic-first cognitive architecture**. By contributing, you’re helping to cultivate a mind and push the boundaries of what AI can become.

We welcome all contributions — from bug fixes and documentation to deep architectural work.

---

## 📜 Code of Conduct

This project and everyone participating in it are governed by a standard **Code of Conduct**. By participating, you agree to uphold this code.

➡️ Please read our [**CODE_OF_CONDUCT.md**](CODE_OF_CONDUCT.md) before contributing.

---

## ⚙️ Development Environment Setup

The project now uses **[uv](https://github.com/astral-sh/uv)** for environment and dependency management, replacing traditional `pip` and `setup.sh` workflows.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vicsanity623/Axiom-Agent.git
cd Axiom-Agent
```

### 2️⃣ Create and Sync the Environment

Use `uv` to create the virtual environment and install dependencies in one step:

```bash
uv sync --extra dev
```

This automatically creates a `.venv` in the project root and installs:
*   Core runtime dependencies (Axiom itself as an editable package).
*   Development tools (`ruff`, `mypy`, `pytest`, etc.).

> 💡 You no longer need to run `setup.sh` — `uv sync` handles everything.

To activate the environment manually if you open a new terminal:
```bash
source .venv/bin/activate
```

### 3️⃣ Verify Installation

You can confirm that the Axiom CLI is working by checking the version:
```bash
axiom --version
```

---

## 🧩 Development Workflow

Axiom follows a **Check → Code → Check → Submit** workflow.

### ✅ 1. Run Quality Checks

Before and after making changes, run the master check script:
```bash
./check.sh
```
This script runs:
*   `ruff format` → Automatic code formatting
*   `ruff check` → Linting and static analysis
*   `mypy` → Static type checking
*   `pytest` → The full unit test suite with coverage

> If all checks pass, your environment and code are good to go.

### 💻 2. Develop and Test Using the CLI

Axiom exposes a unified command-line interface for all primary operations.

| Command                  | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `axiom train`            | Run the agent’s autonomous learning cycles.                  |
| `axiom webui`            | Launch the Gradio-based interactive web UI.                  |
| `axiom teach`            | Teach the agent facts directly via the terminal.             |
| `axiom visualize`        | Generate an interactive HTML visualization of the brain.     |
| `axiom llm`              | Download the recommended local LLM model.                    |
| `rm -f src/axiom/brain/*`| **Deletes the current brain** for a clean testing session.   |

For a clean testing session, you can reset the symbolic brain:
```bash
rm -f src/axiom/brain/*
```

---

## 🚀 How to Contribute

We’re currently in **Phase 5: Semantic Mastery** — improving how the agent represents, reasons about, and verifies knowledge.

### Your First Contribution: Adding Knowledge Provenance

A great first contribution is implementing **knowledge provenance**, tracking *where* a fact came from. This is the first goal of the current roadmap phase.

**Steps:**

1.  Fork the repository and create a branch:
    ```bash
    git checkout -b feature/knowledge-provenance
    ```
2.  Modify the `RelationshipEdge` in `src/axiom/graph_core.py` to include `provenance: str` in its properties.
3.  Update methods like `validate_and_add_relation()` and `learn_new_fact_autonomously()` to accept and store this provenance value (e.g., `"dictionary_api"`, `"llm_decomposition"`).
4.  Add or update tests under `tests/` to verify that provenance is correctly saved and retrieved.
5.  Run `./check.sh` and ensure all checks pass.
6.  Submit a Pull Request with a clear description of your work.

---

## 🪜 General Contribution Process

1.  **Read the Roadmap:** See the current phase in [**ROADMAP.md**](ROADMAP.md).
2.  **Find or Propose a Task:** Choose an issue or open a new one.
3.  **Fork and Branch:** Use a descriptive branch name, e.g., `fix/memory-leak` or `feature/concept-merging`.
4.  **Test & Check:** Run all local checks and tests.
5.  **Submit a Pull Request:** Include details on what and why you changed.

---

## ❤️ Thank You

Every contribution — no matter the size — helps Axiom grow toward its goal: a system capable of autonomous, interpretable, self-directed learning.

Together, we’re building something truly extraordinary.