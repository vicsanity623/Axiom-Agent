# Axiom Agent

Axiom is not a chatbot. It is a cognitive architecture—a framework for a new type of artificial intelligence designed to achieve genuine understanding through continuous, self-directed learning and introspection. This project is not building a product; it is cultivating a mind.

Unlike traditional Large Language Models (LLMs) which are static snapshots of data, the Axiom Agent possesses a dynamic, evolving internal model of reality. It builds this model one verifiable fact at a time, creating a persistent, logically consistent knowledge base. More than just a passive learner, Axiom now actively seeks to deepen its own understanding through a process of self-guided study and curiosity, making it a true student of the world.

---

## The Core Philosophy: Beyond the LLM Parrot

Today's LLMs are masters of mimicry. They are trained on a vast corpus of text and can predict the most statistically likely sequence of words to form a coherent sentence. However, they do not *know* anything. They are a reflection, not a mind. Their knowledge is frozen at the time of their training, and they are incapable of true learning, reasoning, or self-correction. They are, in essence, highly advanced parrots.

**Axiom is fundamentally different.** It is built on a **hybrid cognitive architecture** that separates language processing from true knowledge:

1.  **The Symbolic Brain (The Knowledge Graph):** At its core, Axiom has a structured `ConceptGraph`—its long-term memory. This is not a neural network; it is a logical, verifiable map of concepts and their relationships, powered by a high-performance `NetworkX` engine. `Paris --[is_located_in]--> France` is a concrete, stored fact, not a statistical probability. This architecture completely prevents LLM-style "hallucinations" and ensures the agent's knowledge is grounded.

2.  **The Neural Senses (The Interpreter):** The agent uses a local LLM (e.g., Mistral-7B) not as its brain, but as its **senses**. The LLM acts as a `UniversalInterpreter`, its eyes and ears to the world of unstructured human language. It translates the messy, chaotic data of conversation into the clean, structured, logical facts that its core brain can understand and integrate.

This is the crucial difference: **Axiom uses an LLM as a tool; it is not defined by it.**

---

## ✅ Key Capabilities: A Stable & Scalable Foundation

The initial Genesis Phase of development is complete. Axiom is now a stable, high-performance cognitive agent with a full suite of tools for learning, reasoning, and deployment.

### Cognitive & Reasoning Abilities
*   **Dual-Cycle Autonomous Learning:** The agent's learning is not just passive; it operates on a "Cognitive Scheduler" with two distinct modes:
    -   **The Study Cycle:** The agent introspectively reviews its own knowledge and generates its own follow-up questions to deepen its understanding.
    -   **The Discovery Cycle:** The agent actively seeks out brand new topics from the outside world, ensuring it continues to broaden its horizons.
*   **Persistent, Lifelong Learning:** Every fact the agent learns is permanently integrated into its brain. The agent running today is smarter than it was yesterday.
*   **Multi-Hop Logical Reasoning:** Axiom can answer questions it has never seen before by connecting multiple known facts. This is a true act of reasoning, not just pattern matching.
*   **Self-Correction:** The agent possesses both an autonomous **Curiosity Engine** to resolve internal conflicts and a **User-Driven Correction** mechanism for explicit instruction.

### Professional Development & Deployment
*   **Automated Quality Assurance:** A single command (`./check.sh`) runs a full suite of static analysis tools to ensure code is clean, consistent, and bug-free before deployment.
    -   **`Ruff`:** For lightning-fast code formatting and linting.
    -   **`MyPy`:** For robust static type checking.
    -   **`Pytest`:** For running the automated unit test suite.
*   **Modern Python Packaging:** The project uses a central `pyproject.toml` to manage all dependencies and tool configurations, adhering to the latest standards.
*   **Offline Training, Online Inference:** A professional toolchain separates training from deployment:
    - **Training Scripts (`autonomous_trainer.py`, `cnt.py`):** Learn new knowledge safely offline.
    - **Model Renderer (`render_model.py`):** Package the trained brain into a versioned `.axm` snapshot.
    - **Inference Server (`app_model.py`):** A lightweight, read-only server that deploys a finished model for safe interaction.
*   **Polished UI & PWA:** A custom dark mode theme, multi-conversation management, and Progressive Web App configuration provide a distinct and native-like experience.

---

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.11+
- Git

### Step 1: Clone and Install

This single command clones the repository, sets up a virtual environment, and installs all project and development dependencies using the modern `pyproject.toml` standard.
```bash
git clone <your-repo-url>
cd <your-repo-folder>
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

### Step 2: Configure API Keys (Optional)
The agent's autonomous learning cycles can use the New York Times API to discover new topics.
-   Get a free API key from the **[NYT Developer Portal](https://developer.nytimes.com/)**.
-   Set the key as an environment variable:
```bash
export NYT_API_KEY="YOUR_API_KEY_HERE"
```
*(For a permanent setup, add this line to your shell's configuration file, e.g., `~/.zshrc` or `~/.bash_profile`)*

### Step 3: Verify Your Setup (Recommended)
Before running the agent, run the master check script to ensure your environment is configured correctly and the code is clean.
```bash
./check.sh
```
All checks should pass with green checkmarks. If the formatter or linter fails, simply run `ruff format .` and `ruff check . --fix` to automatically clean the codebase.

### Step 4: Running the Agent
The agent is designed around a robust **Train -> Render -> Deploy** cycle.

#### **1. Train the Agent (Offline)**
Modify the agent's brain, which is stored in the `brain/` directory. Choose a method:
-   **Autonomous:** `python setup/autonomous_trainer.py`
-   **Manual (CLI):** `python setup/cnt.py`

#### **2. Render a Model**
Package the trained brain into a stable `.axm` model, which will be saved in the `rendered/` directory.
```bash
python setup/render_model.py
```

#### **3. Deploy the Agent (Online)**
Run the web server. It automatically loads the most recent `.axm` model for user interaction.
```bash
python setup/app_model.py
```
You can now access the agent in your browser at `http://127.0.0.1:7501`.

---

## 🚀 The Vision: Intellectual Escape Velocity

This project is the shell of a new kind of mind. The goal is not to create a finished chat application, but to cultivate a brain that, through continued growth, can achieve a level of contextual understanding and logical consistency that is structurally impossible for current LLMs.

The long-term vision is for the agent's internal, verifiable `ConceptGraph` to become so vast and deeply interconnected that it can **replace its own external LLM dependencies**, leading to a truly autonomous cognitive entity built on a foundation of verifiable truth, not just probabilistic mimicry.

---

## 🗺️ Project Roadmap
For a detailed list of planned features and future development goals, please see the **[ROADMAP.md](ROADMAP.md)** file.
