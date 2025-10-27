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

> 💡 **Important:** You no longer need `setup.sh` or manual `sys.path` modifications in your tests. The editable install handles everything.

To activate the environment in a new terminal:
```bash
source .venv/bin/activate
```

### 3️⃣ Download the Local LLM

The agent's `UniversalInterpreter` relies on a local GGUF-formatted language model (large context window). You can download the recommended model using the built-in CLI command:

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
| `axiom-train` | **(Primary Workflow)** Starts the agent's autonomous learning cycles. |
| `axiom-webui` | Launches an interactive web UI to chat with the agent. |
| `axiom-teach` | Teach the agent facts directly via the terminal. |
| `axiom-render` | Generate a snapshot of the brain (axiom.axm)for inference chat used in axiom-webui. |

**Typical Debugging Session:**

1.  **Start with a clean slate:**
    ```bash
    # This command safely removes only the agent's brain and state files.
    rm -f src/axiom/brain/* data/*
    ```
2.  **Run the trainer and watch the logs:**
    ```bash
    axiom-train
    ```
3.  **Make your code changes** in a separate terminal.
4.  **Restart the trainer** to see how your changes affect the agent's learning behavior.

### 🤖 3. Working with the Metacognitive Engine

Axiom's most advanced feature is its ability to analyze its own performance and suggest improvements. After a `MetacognitiveEngine` cycle, it generates a `code_suggestion.json` file.

To apply a suggestion:
1.  **Extract the code:** A utility script is provided to cleanly extract the suggested code.
    ```bash
    ./extract_suggestion.py
    ```
2.  **Review the diff:** This creates a `code.py` file. Use your IDE or `diff` to compare it with the original source file.
3.  **Apply and commit:** If you approve (CODE REVIEW the `suggested_solution` is a MUST!! as LLMs can and will usually make mistakes), manually copy the new code into the project, run `./check.sh` one last time, restart `axiom-train`, verify stability and improvements and finally commit the change.

---

## 🚀 How to Contribute

We are currently in **Phase 5: Semantic Mastery**. Our focus is on deepening the agent's understanding of language nuance and complex relationships.

### Your First Contribution: Enhance the `_clean_phrase` function

A great first contribution is to improve the agent's ability to normalize concepts. The `_clean_phrase` method in `cognitive_agent.py` is responsible for cleaning text before it becomes a concept in the knowledge graph. It currently handles parentheses but could be much more robust.

**Steps:**

1.  Fork the repository and create a branch:
    ```bash
    git checkout -b feature/robust-phrase-cleaning
    ```
2.  Locate the `_clean_phrase` method in `src/axiom/cognitive_agent.py`.
3.  **Add new logic** to handle other cases, such as removing possessives (e.g., `"word's"` → `"word"`) or stripping leading/trailing articles more effectively.
4.  **Add a new test** to `tests/test_core_behavior.py` that specifically validates your new cleaning logic.
5.  Run `./check.sh` to ensure all checks pass.
6.  Submit a Pull Request with a clear description of the improvement.

---

## 🪜 General Contribution Process

1.  **Read the Roadmap:** See the current phase in [**ROADMAP.md**](ROADMAP.md).
2.  **Find or Propose a Task:** Choose an issue or open a new one to discuss your idea.
3.  **Fork and Branch:** Use a descriptive branch name (e.g., `fix/memory-leak`, `feature/concept-merging`).
4.  **Code & Test:** Make your changes and ensure you add or update tests to cover them. (source code currently at `60%+ total coverage` as of `Oct, 27, 2025`)
5.  **Check & Submit:** Run `./check.sh` and then submit a Pull Request with details on what you changed and why.

---

## ❤️ Thank You

Every contribution helps Axiom grow toward its goal: a system capable of autonomous, interpretable, and self-directed learning. Together, we’re building something truly extraordinary.