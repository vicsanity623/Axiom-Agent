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
# The quotes are important to prevent errors in some shells like Zsh
pip install -e '.[dev]'
```

### Step 3: Download the LLM Model (Optional, for Full Functionality)
The agent uses a local LLM as a fallback for complex sentences and for its introspective learning loop.
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
# Auto-format all code
ruff format .

# Auto-fix all simple linting issues
ruff check . --fix
```
Run `./check.sh` again to confirm everything passes.

### 2. Train and Develop (Offline)
This is where you'll do your work. The agent's memory is stored in the `brain/` directory. For most development, you'll want a clean slate to test new features.
```bash
# To start fresh for a test session
rm -f brain/*
```
-   **For direct, interactive testing:** `python setup/cnt.py`
-   **For testing autonomous cycles:** `python setup/autonomous_trainer.py`

---

## How to Contribute

We are thrilled that you want to help build this new kind of mind. The current focus of the project is on **Phase 4: Advanced Symbolic Reasoning**. Our goal is to expand the agent's native language understanding to handle more complex grammar and make its knowledge base more unified.

### Your First Contribution: Expanding the Parser's Grammar (Prepositions)

The best way to get started is to help the agent overcome one of its next big limitations: it doesn't understand prepositional phrases.

**The Challenge:**
1.  Fork the repository and create a new branch (e.g., `feature/parser-prepositions`).
2.  Your mission is to upgrade the `SymbolicParser` in `src/axiom/symbolic_parser.py` to correctly parse a sentence like **"Paris is in France"**.
3.  The parser should recognize "in" as a preposition and create a semantic relationship like `Paris --[is_located_in]--> France`. This will likely require creating a mapping from common prepositions to relationship types in the parser (e.g., "in" -> "is_located_in", "of" -> "is_part_of").
4.  You will need to update the `knowledge_base.py` to seed the agent with knowledge of common prepositions.
5.  Add a test case to the `tests/` directory to verify your new functionality.
6.  Ensure all checks in `./check.sh` are passing.
7.  Submit a Pull Request with a clear description of your changes!

### General Contribution Steps
1.  **Read the Roadmap:** For the big picture, please review the **[ROADMAP.md](ROADMAP.md)** file.
2.  **Find a Task:** Find a task in the "Current Focus" phase that interests you, or propose a new one by opening an issue.
3.  **Fork and Branch:** Fork the repository and create a new feature branch for your work.
4.  **Submit a Pull Request:** When your feature is complete and all checks are passing, submit a pull request with a clear description of the changes you've made.

Thank you again for your interest. Together, we can build something truly extraordinary.