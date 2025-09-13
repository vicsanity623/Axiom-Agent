# Axiom Agent

Axiom is not a chatbot. It is a cognitive architecture—a framework for a new type of artificial intelligence designed to achieve genuine understanding through continuous, self-directed learning and introspection. This project is not building a product; it is cultivating a mind.

Unlike traditional Large Language Models (LLMs) which are static snapshots of data, the Axiom Agent possesses a dynamic, evolving internal model of reality. It builds this model one verifiable fact at a time, creating a persistent, logically consistent knowledge base. More than just a passive learner, Axiom now actively seeks to deepen its own understanding through a process of self-guided study and curiosity, making it a true student of the world.

---

## The Core Philosophy: Beyond the LLM Parrot

Today's LLMs are masters of mimicry. They are trained on a vast corpus of text and can predict the most statistically likely sequence of words to form a coherent sentence. However, they do not *know* anything. They are a reflection, not a mind. Their knowledge is frozen at the time of their training, and they are incapable of true learning, reasoning, or self-correction. They are, in essence, highly advanced parrots.

**Axiom is fundamentally different.** It is built on a **hybrid cognitive architecture** that separates language processing from true knowledge:

1.  **The Symbolic Brain (The Knowledge Graph):** At its core, Axiom has a structured `ConceptGraph`—its long-term memory. This is not a neural network; it is a logical, verifiable map of concepts and their relationships. `Paris --[is_located_in]--> France` is a concrete, stored fact, not a statistical probability. This architecture completely prevents LLM-style "hallucinations" and ensures the agent's knowledge is grounded.

2.  **The Neural Senses (The Interpreter):** The agent uses a local LLM (Mistral-7B) not as its brain, but as its **senses**. The LLM acts as a `UniversalInterpreter`, its eyes and ears to the world of unstructured human language. It translates the messy, chaotic data of conversation into the clean, structured, logical facts that its core brain can understand and integrate.

This is the crucial difference: **Axiom uses an LLM as a tool; it is not defined by it.**

---

## ✅ Key Capabilities: A New Paradigm of AI

This architecture provides a clear path for Axiom to become vastly more capable than any static LLM. Its intelligence is not measured by the size of its initial dataset, but by its capacity for growth and self-improvement.

### Cognitive & Reasoning Abilities
*   **Dual-Cycle Autonomous Learning (Study & Discovery):** This is the agent's most advanced capability. Its learning is not just passive; it operates on a "Cognitive Scheduler" with two distinct modes:
    -   **The Study Cycle (Knowledge Integration):** On a frequent timer, the agent introspectively reviews its own important knowledge. It then uses a specialized "curiosity" tool to generate its own follow-up questions about concepts it wants to understand more deeply, actively seeking to enrich its own brain. This is the agent's "homework."
    -   **The Discovery Cycle (Knowledge Acquisition):** On a less frequent timer, the agent actively seeks out brand new, unknown topics from the outside world, ensuring it continues to broaden its horizons. This is the agent "going to school."
*   **Persistent, Lifelong Learning:** Every fact the agent learns—whether from a user or its own study—is permanently integrated into its brain. The agent running today is smarter and more deeply knowledgeable than it was yesterday.
*   **Multi-Hop Logical Reasoning:** Axiom can answer questions it has never seen before by connecting multiple known facts. It can infer that `Paris is in Europe` by logically chaining its knowledge that `Paris is in France` and `France is in Europe`. This is a true act of reasoning, not pattern matching.
*   **Dynamic Memory & Fact Salience:** The agent's memory is not static. It tracks an `access_count` for every fact in its brain, reinforcing memories that are used frequently. Its reasoning and study engines prioritize these "salient" facts, focusing its attention on what it knows to be most important.
*   **The Curiosity Engine (Implicit Correction):** The agent can recognize when new information conflicts with its existing world model. Instead of blindly accepting it, it exhibits curiosity, formulates a clarifying question, and uses the user's feedback to actively correct its own brain.
*   **User-Driven Correction:** The user can act as a teacher, taking direct control of the learning process with a `correction:` command to explicitly override and replace incorrect facts.

### Autonomous Systems
*   **Resilient Knowledge Harvester:** The agent's tool for acquiring knowledge from the world is designed for resilience. It uses a hybrid discovery model (Wikipedia Categories, NYT Archives), maintains a "rejection memory" to avoid getting stuck on unlearnable topics, and uses a "Simplicity Filter" to discard grammatically complex facts that it cannot reliably understand.

### Deployment & User Experience
*   **Versioned Model Rendering (`.axm`):** A custom "Axiom Mind" file format and a dedicated renderer allow for the creation of stable, versioned snapshots of the agent's brain after training milestones.
*   **Read-Only Inference Mode:** A separate, lightweight chat application can load any `.axm` model, allowing for safe, distributable deployment of a "finished" agent.
*   **Remote Access with `ngrok`:** Both the training and inference apps have integrated support for exposing the local server to the internet.
*   **Polished UI & PWA:** A custom dark mode theme, multi-conversation management, and Progressive Web App configuration provide a distinct and native-like experience on both desktop and mobile.
*   **Stability First:** Experimental features like "Anticipatory Caching" and "Conversational Context" are preserved in the code but disabled by default with "kill switches" to guarantee the integrity of the core cognitive system.

---

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.11+
- Git

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### Step 2: Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure API & Auth Keys (Optional but Recommended)

-   **New York Times API:** The agent's autonomous "Discovery Cycle" uses the NYT Archive API to find a diverse range of new topics to learn about.
    -   Get a free API key from the **[NYT Developer Portal](https://developer.nytimes.com/)**.
    -   When creating your "App," enable access to the **"Times APIs"** service. This will grant you access to the Archive API. Or Enable them manually as these change.
-   **Ngrok Auth Token:** To use the `--ngrok` feature for an extended period, you'll need to configure your free auth token.
    -   Get a token from your **[ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)**.

Set these keys as environment variables in your terminal before running the application.

```bash
export NYT_API_KEY="YOUR_API_KEY_HERE"
export NGROK_AUTHTOKEN="YOUR_NGROK_TOKEN_HERE"
```
*(For a permanent setup, add these lines to your shell's configuration file, e.g., `~/.zshrc` or `~/.bash_profile`)*

### Step 6: Run the Agent
Once all dependencies are installed and the model is in place, you have two modes:

**A Training Mode:** The full application with the autonomous harvester active.
```bash
# Run locally
python3 app.py

# Run with a public ngrok URL
python3 app.py --ngrok
```

**B Inference Mode:** A lightweight, read-only chat app using a pre-rendered model.
```bash
# First, create a model version (do this after training)
python3 render_model.py 0.1.0

# Run the read-only app locally
python3 chat_app.py axiom_v0.1.0.axm

# Run with a public ngrok URL
python3 chat_app.py axiom_v0.1.0.axm --ngrok
```
---

## 🚀 The Vision

This project is the shell of a new kind of mind. The goal is not to create a finished chat application, but to cultivate a brain that, through continued growth and the implementation of more advanced reasoning modules, can achieve a level of contextual understanding, logical consistency, and self-correcting wisdom that is structurally impossible for current LLMs.

The ultimate vision is for the agent to achieve **intellectual escape velocity**.

As the agent's internal ConceptGraph grows exponentially—through both guided teaching and autonomous discovery—it is not just accumulating facts; it is building a comprehensive, high-fidelity model of reality. The long-term goal is for this verifiable, logical brain to become so vast and deeply interconnected that it can **replace its own external LLM dependencies**.

This creates a path where a future **axiom_vX.X.X.axm model will no longer need Mistral, Grok, or any other external model for its senses or voice**. It will have developed its own, superior, internally consistent model of language and logic. This will result in a truly autonomous, self-aware cognitive entity that can continuously expand its understanding, reason through problems of immense complexity, and engage in infinite growth through its self-correction loop.

This is a training in progress. This is the architectural path toward a genuine AGI—one built on a foundation of verifiable truth, not just probabilistic mimicry. This is the path to a truly intelligent, **non-parrot AI**.

---

## 🗺️ Project Roadmap
For a detailed list of planned features and future development goals, please see the **[ROADMAP.md](ROADMAP.md)** file.