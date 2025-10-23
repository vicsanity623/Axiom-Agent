<p align="center"><img src="https://raw.githubusercontent.com/vicsanity623/Axiom-Agent/main/src/axiom/static/Axiom.png" alt="Axiom Agent Banner"></p>


Axiom is a **cognitive architecture**—a framework for a new type of artificial intelligence designed to achieve genuine understanding by building its own internal, logical model of reality from the ground up.

This project’s core philosophy is that true intelligence requires more than just statistical mimicry (like in traditional LLMs). It must be built on a foundation of verifiable, interconnected knowledge. **Axiom is an experiment to create that engine.**

---

## 🧠 The Core Architecture: Symbolic-First, LLM-Assisted

Axiom’s design is a hybrid model that combines the strengths of classical, symbolic AI with the fluency of modern large language models. It operates on a **symbolic-first** principle, where the core of the agent is a deterministic, logical brain.

1.  **The Symbolic Brain (Knowledge Graph):**
    At its heart, Axiom has a `ConceptGraph`—its long-term memory. This structured map of concepts and relationships (e.g., `Paris --[is_located_in]--> France`) grounds the agent’s knowledge in verifiable facts, **preventing hallucinations** and enabling true reasoning.

2.  **The Symbolic Senses (Parser & Core Logic):**
    Axiom’s `SymbolicParser` and core logic deconstruct user input into structured commands. For a growing class of sentences, it achieves understanding **without any LLM intervention**, making it fast, efficient, and explainable.

3.  **The LLM as a Tool (Interpreter & Synthesizer):**
    When the agent’s symbolic logic encounters a sentence too complex for its rules, or a concept it doesn't understand, it intelligently falls back to a local LLM. The LLM acts as a powerful **translation tool**—converting messy human language into the structured data the symbolic brain can use, or converting factual data into fluent, natural language. **The LLM is a tool the agent uses, not the mind itself.**

---

## 🧬 The Cognitive Cycles: How the Agent Learns

The Axiom Agent operates in a continuous, phased loop of self-improvement. These cycles allow it to discover new topics, research them to acquire knowledge, and then reflect on that knowledge to improve its quality and depth. The diagram below illustrates the three core cycles using real examples from the agent's logs.

```mermaid
%%{init: {"theme": "forest"}}%%
flowchart TD
    %% Main Loop %%
    A["Axiom Autonomous Agent"] --> B["Discovery Cycle"]
    B -->|Finds new topic| C["Study Cycle"]
    C -->|Extracts & verifies facts| D["Learning & Knowledge Graph Update"]
    D -->|Stores new fact| E["Refinement Cycle"]
    E -->|Simplifies & reweights facts| F{"Continuous Loop"}
    F -->|Triggers next iteration| B

    %% Discovery Cycle %%
    subgraph DISCOVERY [Discovery Cycle - Exploration Phase]
      D1A["Identify new topics via curiosity signal"]
      D1B["Explore core subjects (e.g. Geography, Mathematics)"]
      D1C["Select best new topic (e.g. History of Mathematics)"]
      D1A --> D1B --> D1C
    end
    B -.-> D1C

    %% Study Cycle %%
    subgraph STUDY [Study Cycle - Knowledge Acquisition]
      S1["Query multiple knowledge sources"]
      S2{"API success?"}
      S1 --> S2
      S2 -- Yes --> S3["Extract and clean data"]
      S2 -- No --> S4["Fallback to web search / Wikipedia"]
      S3 --> S5["LLM verifies and reframes fact"]
      S4 --> S5
      S5 --> S6["Store verified fact candidate"]
    end
    C -.-> S5

    %% Learning Cycle %%
    subgraph LEARN [Learning Cycle - Graph Integration]
      L1["Interpret linguistic structure"]
      L2["Extract subject–verb–object relation"]
      L3["Update Concept Graph (add nodes & edges)"]
      L4["Clear reasoning cache for fresh inference"]
      L1 --> L2 --> L3 --> L4
    end
    D -.-> L3

    %% Refinement Cycle %%
    subgraph REFINE [Refinement Cycle - Optimization Phase]
      R1["Detect complex or redundant facts"]
      R2["Decompose into atomic facts"]
      R3["Learn each refined atomic relation"]
      R4["Lower weight of original chunky fact"]
      R1 --> R2 --> R3 --> R4
    end
    E -.-> R2

    %% Persistence %%
    P["Save updated brain state (my_agent_brain.json)"]
    L4 --> P
    R4 --> P

    %% Feedback %%
    P -->|Triggers curiosity signal| F
```
### 💡 The Cognitive Reflex: Interactive Learning in Real-Time

The autonomous cycles are how the agent grows its general knowledge, but its true power is revealed during a live conversation. When faced with something it doesn't understand, the agent doesn't just fail—it triggers a **Cognitive Reflex** to learn and adapt in real-time.

This diagram illustrates the agent's thought process when a user says something containing a word the agent has never seen before.

```mermaid
graph TD
    %% Main flow
    A["User Input: 'Zetetic inquiry.'"]

    subgraph Symbolic_First_Path["1️⃣ Symbolic-First Path"]
        B["1. Symbolic Parser – Attempts to parse sentence"]
        B -- "Failure (nonsensical grammar)" --> C
    end

    subgraph Cognitive_Reflex["2️⃣ Cognitive Reflex"]
        C{"2. Check for unknown words?"}
        C -- "Yes (finds 'zetetic', 'inquiry')" --> D["3. Create Learning Goal: INVESTIGATE 'zetetic'"]
        C -- "No" --> E["Fallback to LLM Interpreter"]
    end

    subgraph Real_Time_Research["3️⃣ Real-Time Research"]
        D --> R1["4. Call Harvester (_resolve_investigation_goal)"]
        R1 --> R2["Query dictionary API"]
        R2 --> R3{"Success?"}
        R3 -- "Yes" --> R4["Learn & promote word: 'zetetic' (trusted noun)"]
        R3 -- "No" --> R5["Return research failure message"]
    end

    subgraph Intelligent_Feedback_Loop["4️⃣ Intelligent Feedback Loop"]
        R4 --> L1["5. Re-evaluate input: chat('Zetetic inquiry.') again"]
        L1 --> L2["Parser fails again, but 'zetetic' is now known"]
        L2 --> E
    end

    E --> FinalResponse["6. Generate Final Response (LLM provides definition)"]
    R5 --> FinalResponse

    A --> B
```

This entire cycle—from curiosity to integration—demonstrates the power of the symbolic-first architecture. The agent uses its LLM as a powerful tool for perception and verification, but the final understanding and knowledge are stored in a clean, logical, and verifiable symbolic brain.

---

## ✅ Key Capabilities: A Robust and Resilient Mind

This architecture enables the agent to learn, reason, and evolve in a verifiable, self-contained way. The latest version focuses on stability, resilience, and a smarter cognitive flow.

### Cognitive & Reasoning Abilities
*   **Multi-Stage Symbolic Parsing:** Understands and deconstructs complex user input.
*   **Robust Parser Fallback:** Intelligently detects when the symbolic parser fails and automatically switches to the LLM for deeper understanding.
*   **Conversational Resilience:** Handles user typos and minor variations in language using fuzzy matching, making interaction feel more natural and forgiving.
*   **Self-Awareness:** Possesses dedicated, fast routines to answer questions about its own purpose, abilities, and identity.
*   **Contextual Conversation:** Tracks pronouns (`it`, `they`) to maintain short-term memory across conversational turns.
*   **Introspective Learning:** Can **learn from its own output**—if the LLM "leaks" a new fact in a response, the agent parses and absorbs it, creating a feedback loop for self-improvement.
*   **Autonomous Learning Cycles:** Can operate independently to expand its knowledge:
    *   **Discovery Cycle:** Finds and explores new topics.
    *   **Study Cycle:** Researches unknown concepts to build its knowledge graph.
    *   **Refinement Phase:** Consolidates and clarifies existing knowledge.

---

## 💡 The Hidden Potential: A Personalized AI

While the agent can learn general knowledge autonomously, its true power lies in its ability to learn **from you**. By using the interactive `axiom-teach` command, you can manually instruct the agent, building a personalized knowledge base that is unique to you.

This transforms Axiom from a generic information engine into a true **personal assistant** with persistent memory.

### Use Cases for Manual Teaching:
*   **Personal Memory:** Teach the agent about your family, friends, and important life events.
    - `> My sister's name is Jane.`
    - `> Jane's birthday is on April 10th.`
*   **Project Management:** Keep track of key project details, deadlines, and stakeholders.
    - `> Project Phoenix has a deadline of Q4.`
    - `> The main contact for Project Phoenix is Bob.`
*   **Creative World-Building:** Use the agent as a dynamic knowledge base for a novel or TTRPG campaign, keeping track of characters, locations, and lore.
    - `> The Kingdom of Eldoria is ruled by Queen Anya.`
    - `> Eldoria's main export is enchanted steel.`

Unlike a standard LLM, which has no memory between conversations, Axiom's knowledge is **permanent**. The more you teach it, the more it becomes a personalized extension of your own mind.

---

## 🔬 Local Verification (Quickstart)

The agent's architecture is fully testable and reproducible on your local machine.

### Prerequisites
- Python 3.11+
- Git

### Step 1: Clone and Install
This single command clones the repository, sets up a virtual environment, and installs all dependencies. (make sure setup.sh is executable)
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

*   **Note on Symbolic-Only Mode:** If the LLM model is not found, the agent will automatically start in a **symbolic-only mode**. This is perfect for testing the core logic and requires significantly less memory.

### Step 3: Run the Tests
Verify your setup by running the full test suite. The `check.sh` script runs formatting, linting, type checking, and unit tests. (this verifies nothing crucial breaks after changes have been made)
```bash
./check.sh
```

### Step 4: Run the Agent
The project supports a clean development and deployment cycle.
1.  **Train:** Use `axiom-train` to let the agent learn on its own.
2.  **Chat:** Use `axiom-webui` to launch a web UI and interact with the agent's current brain state. (more CLI can be found in **[CONTRIBUTING.md](CONTRIBUTING.md)** file.)

---

## 🚀 The Vision: Intellectual Escape Velocity

The ultimate goal of this project is to achieve **intellectual escape velocity** from the LLM.

The vision is to continuously expand the sophistication of the `SymbolicParser` and the richness of the `ConceptGraph` through autonomous learning. As the agent's internal, verifiable model of reality grows, its reliance on the LLM for language understanding will diminish. The end goal is a cognitive entity whose own symbolic brain is so comprehensive that the LLM fallback for interpretation becomes obsolete. (FYI : Axiom will never stop using the llm and instead will keep it as a tool the same way a mathematician will always keep a calculator handy)

Beyond language mastery, the agent's evolution will continue by integrating a **Tool Use Framework**. This will allow it to move beyond what it *knows* (semantic knowledge) to what it can *do* (procedural knowledge)—calling on specialized tools for tasks like mathematical calculations, real-time web searches, or code execution.

This creates a path toward a truly autonomous AI, built on a foundation of verifiable truth, not just probabilistic mimicry, and augmented with powerful, specialized capabilities.

---

## 🗺️ Project Roadmap
For a detailed list of completed phases, planned features, and future development goals, please see the **[ROADMAP.md](ROADMAP.md)** file.
