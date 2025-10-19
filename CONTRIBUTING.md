## Contributing to the Axiom Agent

Thank you for your interest in contributing to the Axiom Agent project! This is not just another chatbot; it's an exploration into a new kind of cognitive architecture. By contributing, you are helping to cultivate a mind and push the boundaries of what AI can be.

We welcome contributions of all kinds, from bug fixes and documentation enhancements to major new features.

## Code of Conduct

This project and everyone participating in it is governed by a standard Code of Conduct. By participating, you are expected to uphold this code. Please read our **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)**.

## Development Environment Setup

The following steps will guide you through setting up a complete and stable development environment.

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
# The quotes are important to prevent errors in some shells like Zsh
```bash
pip install -e '.[dev]'
```

### Step 3: Download the LLM Model (Optional, for Full Functionality)
The agent uses a local LLM as a fallback for complex sentences.
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
This script runs formatting, linting, static type checking (MyPy), and the full unit test suite. If it passes, you can be confident your changes are high-quality.

### 2. Train and Develop
The agent's memory is stored in the `brain/` directory. For most development, you'll want a clean slate to test new features.
```bash
# To start fresh for a test session
rm -f brain/*
```
*   **For manual training and chat:** `python setup/cnt.py`
*   **For rendering a chat model:** `python setup/render_model.py`
*   **For interactive chat and testing on latest chat model:** `python setup/app_model.py`
*   **For testing autonomous cycles / endless learning:** `python setup/autonomous_trainer.py`

---

## How to Contribute

We are thrilled that you want to help build this new kind of mind. The current focus of the project is on **Phase 4: Semantic Mastery**. Our goal is to deepen the agent's understanding of language and make its knowledge base more robust and verifiable.

### Your First Contribution: Adding Knowledge Provenance

A great way to get started is to help the agent track *where* its knowledge comes from. This is a critical feature called "provenance."

**The Challenge:**
1.  Fork the repository and create a new branch (e.g., `feature/knowledge-provenance`).
2.  Your mission is to upgrade the `RelationshipEdge` class in `src/axiom/graph_core.py`. Add a new field to its `__init__` method and `to_dict` method, such as `source_of_knowledge: str`.
3.  Modify the `CognitiveAgent.add_knowledge` method (and similar learning methods) to accept a `source` argument (e.g., `"user_input"`, `"wordnet"`, `"introspection"`). This source should be passed down and stored on the `RelationshipEdge` when it's created.
4.  Update the `_get_all_facts_as_string` method in `CognitiveAgent` to include the source when it prints out facts (e.g., `(sky --[has_property]--> blue) [Source: seeded_knowledge]`).
5.  Add or update a test case in the `tests/` directory to verify that the source is correctly saved and retrieved.
6.  Ensure all checks in `./check.sh` are passing.
7.  Submit a Pull Request with a clear description of your changes!

### General Contribution Steps
1.  **Read the Roadmap:** For the big picture, please review the **[ROADMAP.md](ROADMAP.md)** file.
2.  **Find a Task:** Find a task in the "Current Focus" phase that interests you, or propose a new one by opening an issue.
3.  **Fork and Branch:** Fork the repository and create a new feature branch for your work.
4.  **Submit a Pull Request:** When your feature is complete and all checks are passing, submit a pull request with a clear description of the changes you've made.

Thank you again for your interest. Together, we can build something truly extraordinary.