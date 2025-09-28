# Contributing to the Axiom Agent

Thank you for your interest in contributing to the Axiom Agent project! This is not just another chatbot; it's an exploration into a new kind of cognitive architecture. By contributing, you are helping to cultivate a mind and push the boundaries of what AI can be.

We welcome contributions of all kinds, from bug fixes and performance improvements to new features and documentation enhancements.

## Code of Conduct

This project and everyone participating in it is governed by a standard Code of Conduct. By participating, you are expected to uphold this code.

## Development Environment Setup

The following steps will guide you through setting up a complete and stable development environment using modern Python standards.

### Step 1: Clone the Repository & Setup Environment
First, get the code, create an isolated virtual environment, and activate it.
```bash
git clone https://github.com/vicsanity623/Axiom-Agent.git
cd Axiom-Agent
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install All Dependencies
This project uses a `pyproject.toml` file to manage all dependencies. This single command installs the core application libraries *and* all the development tools (like Ruff, MyPy, and Pytest) needed for contributing.
```bash
pip install -e '.[dev]'
```
The quotes are important to prevent errors in some shells like Zsh


### Step 3: Download the LLM Model (Optional, for Full Functionality)
The agent uses a local LLM as a fallback for complex sentences and for its autonomous "Refinement" cycle.

1.  Download the **`mistral-7b-instruct-v0.2.Q4_K_M.gguf`** model from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF).
2.  Create a `models/` directory in the root of the project.
3.  Place the downloaded `.gguf` file inside the `models/` directory.

---

## The Core Development Workflow

The project is structured around a professional **Check -> Code/Train -> Check -> Submit** cycle.

### 1. Run Quality Checks (The First and Last Step)
**Before you start coding and before you commit, always run the master check script.** This ensures your environment is working and your code adheres to project standards.
```bash
./check.sh
```
If the script reports formatting or linting issues, you can often fix them automatically with these commands:
```bash
ruff format .

ruff check . --fix
```
Auto-format all code
Auto-fix all simple linting issues

Run `./check.sh` again to confirm everything passes.

### 2. Train and Develop (Offline)
This is where you'll do your work. The agent's memory is stored in the `brain/` directory. For most development, you'll want a clean slate to test new features.
```bash
rm -rf brain/
```
To start fresh for a test session

-   **For direct, interactive testing:** `python setup/cnt.py`
-   **For testing autonomous cycles:** `python setup/autonomous_trainer.py`

### 3. LLM-Optional Development (For Fast, Symbolic-Only Testing)
We are committed to supporting development on lower-spec machines. The training scripts (`cnt.py`, `app.py`) will **automatically detect if the LLM model file is missing** and start the agent in a symbolic-only mode.

This is the perfect way to work on the `SymbolicParser` or the core `CognitiveAgent` logic without the overhead of loading and running the LLM. Some features, like the `refinement_cycle`, will be disabled, but you will be able to test the core symbolic engine at full speed.

---

## How to Contribute

We are thrilled that you want to help build this new kind of mind. The current focus of the project is on **Phase 6: The Hardened Mind**. Our goal is to address the subtle but critical limitations discovered during long-term autonomous runs to make the agent's understanding of language more robust and nuanced.

### Your First Contribution: The "Plural Problem" (Lemmatization)

The best way to get started is to help the agent overcome one of its biggest current limitations: it treats singular and plural words (like "factory" and "factories") as completely different concepts.

**The Challenge:**
1.  Fork the repository and create a new branch (e.g., `feat/lemmatization-layer`).
2.  Your mission is to integrate a lemmatizer (like `WordNetLemmatizer` from the `nltk` library, which is already a dependency) into the agent's core logic. The best place for this is likely within the `_clean_phrase` method in `src/axiom/cognitive_agent.py`.
3.  The goal is to ensure that all concepts are reduced to their dictionary root form (their "lemma") before being stored in or queried from the knowledge graph. This will unify the agent's understanding and make its reasoning much more powerful.
4.  Add a test case to the `tests/` directory to verify your new functionality.
5.  Ensure all checks in `./check.sh` are passing.
6.  Submit a Pull Request with a clear description of your changes!

### General Contribution Steps
1.  **Read the Roadmap:** For the big picture, please review the **[ROADMAP.md](ROADMAP.md)** file.
2.  **Find a Task:** Find a task in the "Current Focus" phase that interests you, or propose a new one by opening an issue.
3.  **Fork and Branch:** Fork the repository and create a new feature branch for your work.
4.  **Submit a Pull Request:** When your feature is complete and all checks are passing, submit a pull request with a clear description of the changes you've made.

Thank you again for your interest. Together, we can build something truly extraordinary.
