# Axiom Agent

Axiom is an autonomous, learning cognitive agent with a hybrid AI architecture. It's designed to build a verifiable knowledge base from natural language and grow its understanding of the world through both user interaction and autonomous exploration.

---

## Core Concepts & Philosophy

The Axiom Agent is not just a chatbot. It's built on a **hybrid cognitive architecture** that combines the strengths of two AI paradigms:

1.  **A Symbolic Core (The Knowledge Graph):** The agent's long-term memory is a structured `ConceptGraph` (`my_agent_brain.json`). Every fact is verifiable and logically connected. This completely prevents LLM-style "hallucinations" and ensures the agent's knowledge is grounded and reliable.
2.  **A Neural Front-End (The Interpreter):** The agent uses a local Large Language Model (Mistral-7B) as its interface to the world. The LLM acts as a powerful `UniversalInterpreter`, translating unstructured natural language into the structured facts that its logical brain can learn and reason with.

This hybrid approach allows the agent to be both flexible in its understanding and rigorous in its knowledge.

---

## ✅ Key Capabilities

### Core Functionality
- **Natural Language Learning:** Teach the agent new facts, concepts, and relationships in simple English.
- **Verifiable Memory:** All learned knowledge is stored persistently and can be inspected with the `show all facts` command.
- **Fluent Response Synthesis:** The agent uses its LLM to convert its structured facts into grammatically correct, natural-sounding answers.
- **Alias Merging:** Understands that different names (e.g., "Eminem" and "Marshall Mathers") can refer to the same entity and combines knowledge from both.
- **Self-Identity:** Can be taught its own name and purpose and answer questions like `who are you`.

### Autonomous Systems
- **Knowledge Harvester:** A 24/7 background process that autonomously seeks out new knowledge.
- **Multi-Source Topic Finding:** Uses a robust fallback system (New York Times API -> Wikipedia) to find interesting topics.
- **Self-Aware Learning:** Intelligently checks its own memory to avoid re-learning topics, forcing it to explore new areas of knowledge.
- **Intelligent Fact Sourcing:** After finding a topic, it uses a multi-stage fallback (Wikipedia -> DuckDuckGo) to find a high-quality, simple, and learnable fact.
- **Intellectual Caution:** Automatically rejects sentences that are too complex or ambiguous, preventing its knowledge graph from being polluted with low-quality information.

### Performance & Stability
- **Persistent Caching:** Caches the results of slow LLM interpretations and syntheses to disk, providing instantaneous responses to repeated questions, even after a server restart.
- **Anticipatory Caching:** After autonomously learning a new fact, the agent proactively "thinks" about the most likely follow-up question and pre-warms the cache, making the first user interaction with new knowledge feel instant.
- **Thread-Safe Operation:** A global lock protects the agent from memory corruption and crashes when user interactions and the autonomous learning cycle run simultaneously.

---

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.11+
- Git

### 1. **Clone the Repository**
Clone this project to your local machine.
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
    
Download the file: mistral-7b-instruct-v0.2.Q4_K_M.gguf
can be found on huggingface here [TheBloke](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
Place it in the correct directory: Create a folder named models inside your project directory and place the downloaded .gguf file inside it. The final path should be models/mistral-7b-instruct-v0.2.Q4_K_M.gguf.

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
The agent will be accessible on your local network. Open a web browser and navigate to http://127.0.0.1:7500 or the network IP address shown in the terminal.

## 🚀 How to Use
**Teach the Agent:** State a simple, atomic fact (e.g., Paris is a city in France).
**Ask Questions:** Ask questions about concepts it knows (e.g., who is Eminem).
**Inspect Memory:** Use the built-in command show all facts to see everything the agent has learned.

## 🗺️ Project Roadmap
For a detailed list of planned features and future development goals, please see the **[ROADMAP.md](ROADMAP.md)** file.

### This is ADMIN control not the PUBLIC General chat version, coming soon...