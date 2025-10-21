## Contributing to the Axiom Agent

Thank you for your interest in contributing to the Axiom Agent project! This is not just another chatbot; it's an exploration into a new kind of cognitive architecture. By contributing, you are helping to cultivate a mind and push the boundaries of what AI can be.

We welcome contributions of all kinds, from bug fixes and documentation enhancements to major new features.

## Code of Conduct

This project and everyone participating in it is governed by a standard Code of Conduct. By participating, you are expected to uphold this code. Please read our **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)**.

## Development Environment Setup

The following steps will guide you through setting up a complete and stable development environment.

### Step 1: Clone the Repository & Setup Environment
First, get the code, create an isolated virtual environment, and activate it. (TIP: the ./setup.sh is to make sure nothing is forgotten during setup, it needs to be `executable` before you run  it)
```bash
git clone https://github.com/vicsanity623/Axiom-Agent.git
cd Axiom-Agent
./setup.sh
```
make sure venv is activated (it should activate with ./setup.sh)


### Step 2: Download the LLM Model (Optional, for full functionality)
The agent uses a local LLM for many of its advanced features. You can download the recommended model automatically by running this command from your project's root directory:
```bash
axiom-llm
```

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
rm -f src/axiom/brain/*
```
*   **For manual training and chat:** ```axiom-teach``` (manually teach inside the terminal)
*   **For rendering a chat model:** ```axiom-render``` (creates a .axm model for chat)
*   **For interactive chat and testing on latest chat model:** ```axiom-webui_app``` (can learn form user input)
*   **For interactive READ ONLY chat and testing on latest chat model:** ```axiom-webui``` (CANNOT LEARN / READ ONLY)
*   **For testing autonomous cycles / endless learning:** ```axiom-train``` (24/7 autonomous learning)

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