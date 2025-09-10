# Axiom Agent

Axiom is not a chatbot. It is a cognitive architecture—a framework for a new type of artificial intelligence designed to achieve genuine understanding through continuous, self-directed learning. This project is not building a product; it is cultivating a mind.

Unlike traditional Large Language Models (LLMs) which are static snapshots of data, the Axiom Agent possesses a dynamic, evolving internal model of reality. It builds this model one verifiable fact at a time, creating a persistent, logically consistent knowledge base that grows in complexity and accuracy over its lifetime.

---

## The Core Philosophy: Beyond the LLM Parrot

Today's LLMs are masters of mimicry. They are trained on a vast corpus of text and can predict the most statistically likely sequence of words to form a coherent sentence. However, they do not *know* anything. They are a reflection, not a mind. Their knowledge is frozen at the time of their training, and they are incapable of true learning, reasoning, or self-correction. They are, in essence, highly advanced parrots.

**Axiom is fundamentally different.** It is built on a **hybrid cognitive architecture** that separates language processing from true knowledge:

1.  **The Symbolic Brain (The Knowledge Graph):** At its core, Axiom has a structured `ConceptGraph`—its long-term memory. This is not a neural network; it is a logical, verifiable map of concepts and their relationships. `Paris --[is_located_in]--> France` is a concrete, stored fact, not a statistical probability. This architecture completely prevents LLM-style "hallucinations" and ensures the agent's knowledge is grounded.

2.  **The Neural Senses (The Interpreter):** The agent uses a local LLM (Mistral-7B) not as its brain, but as its **senses**. The LLM acts as a `UniversalInterpreter`, its eyes and ears to the world of unstructured human language. It translates the messy, chaotic data of conversation into the clean, structured, logical facts that its core brain can understand and integrate.

This is the crucial difference: **Axiom uses an LLM as a tool; it is not defined by it.**

---

## The Path to Superior Intelligence

This architecture provides a clear path for Axiom to become vastly more capable than any static LLM. Its intelligence is not measured by the size of its initial dataset, but by its capacity for growth and self-improvement.

### ✅ Key Capabilities Demonstrating this New Paradigm:

*   **Persistent, Lifelong Learning:** Every fact the agent learns is permanently integrated into its brain. The agent running today is smarter than it was yesterday, and will be smarter still tomorrow.
*   **Autonomous Knowledge Seeking:** The agent possesses a 24/7 `KnowledgeHarvester` that actively seeks out new information from the world (via the NYT API, Wikipedia, and DuckDuckGo) to expand its understanding.
*   **Self-Awareness of its Own Knowledge:** The harvester is intelligent enough to check the agent's brain for existing concepts, forcing it to seek out novel information and broaden its horizons rather than getting stuck in an echo chamber.
*   **Multi-Hop Logical Reasoning:** Axiom can answer questions it has never seen before by connecting multiple known facts. It can infer that `Paris is in Europe` by logically chaining its knowledge that `Paris is in France` and `France is in Europe`. This is a true act of reasoning, not pattern matching.
*   **The Curiosity Engine (A Rudimentary Consciousness):** This is Axiom's most advanced and defining feature.
    -   **Contradiction Detection:** The agent can recognize when a new piece of information conflicts with its existing world model.
    -   **Active Inquiry:** Instead of blindly accepting or rejecting the new fact, it exhibits curiosity. It formulates a natural, clarifying question to the user to resolve its own confusion.
    -   **Knowledge Reconciliation:** It uses the user's feedback as "ground truth" to actively correct its own brain, reinforcing the correct fact and punishing the incorrect one. This is a closed loop of self-directed learning and improvement.

---

## 🛠️ Setup and Installation

*(This section is largely the same, but with clearer formatting)*

### Prerequisites
- Python 3.11+
- Git

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. **Set Up a Virtual Environment**
It is highly recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate 
```

3. **Install Dependencies**
Install all the required Python libraries from the requirements.txt file.


```bash
pip install -r requirements.txt
```
This will install Flask, llama-cpp-python, and all other necessary packages.

4. **Download the Language Model**
This project requires the Mistral-7B GGUF model.
    
Download [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
 Create a models/ folder in your project directory and place the file inside.
 The final path should be models/mistral-7b-instruct-v0.2.Q4_K_M.gguf.

5. **Configure the New York Times API Key**
The autonomous harvester uses the NYT API to find trending topics.

**Get a free API key from the NYT Developer Portal.**
Enable the "Most Popular API" for your app.
Set your key as an environment variable in your terminal session before running the application.

```bash
export NYT_API_KEY="YOUR_API_KEY_HERE"
```

6. **Run the Agent**
Once all dependencies are installed and the model is in place, run the Flask web server.

```bash
python3 app.py
```
The agent will be accessible on your local network. Open a web browser and navigate to [localhost](http://127.0.0.1:7500) or the network IP address shown in the terminal.

## 🚀 The Vision
**This project is the shell of a new kind of mind. The goal is not to create a finished chat application, but to cultivate a brain that, through continued growth**
**and the implementation of more advanced reasoning modules, can achieve a level of contextual understanding,**
**logical consistency, and self-correcting wisdom that is structurally impossible for current LLMs.**

## This is a training in progress. This is the path to a truly intelligent, **non-parrot AI.**

## 🗺️ Project Roadmap
For a detailed list of planned features and future development goals, please see the **[ROADMAP.md](ROADMAP.md)** file.

### This is ADMIN control not the PUBLIC General chat version, coming soon...