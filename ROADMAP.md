# Axiom Agent: Development Roadmap

This document outlines the current status, planned features, and architectural improvements for the Axiom Agent project.

## ✅ Core Architecture & Capabilities (Complete & Stable)

This foundational phase is complete. The agent is a fully autonomous, reasoning, and self-correcting learning entity with a robust user interface.

- **✅ Hybrid Cognitive Architecture:** The agent's mind is a hybrid of a symbolic **Knowledge Graph** (for verifiable, persistent memory) and a neural **Universal Interpreter** (for flexible language understanding), giving it the strengths of both paradigms.
- **✅ Multi-Hop Logical Reasoning:** The agent can answer complex questions by traversing its knowledge graph and connecting multiple facts to infer new conclusions.
- **✅ Dynamic Memory & Fact Salience:** The agent's memory is not static. It tracks an `access_count` for every fact in its brain, reinforcing memories that are used frequently. Its reasoning engine prioritizes these "salient" facts to provide more concise and relevant answers.
- **✅ Temporal Reasoning:** The agent can perceive, learn, and reason with time-based facts. It understands the context of words like "now" or "currently" and can determine the most relevant fact from a series of historical events.
- **✅ Dual-Mode Learning System:** The agent possesses two distinct but complementary mechanisms for learning and self-correction.
    - **The Curiosity Engine (Implicit Correction):** Actively identifies when new information conflicts with its existing world model and autonomously asks clarifying questions to resolve its own confusion.
    - **User-Driven Correction (Explicit Correction):** Allows a user to take direct control of the learning process by using a `correction:` command to override, punish, and replace incorrect facts in the agent's brain.
- **✅ Autonomous Knowledge Harvester:**
    - **Diverse & Novel Topic Discovery:** The harvester is not limited to current events. It uses a hybrid discovery model, randomly exploring both Wikipedia's vast category structure and the New York Times' historical archives. This ensures a constant stream of diverse and novel topics, transforming the agent from a "trending topic" follower into a true "historical explorer."
    - **Self-Aware & Resilient Learning:** Intelligently checks its own memory to avoid re-learning topics. It maintains a short-term "rejection memory" to avoid getting stuck on unlearnable topics, and uses a "Simplicity Filter" to discard facts that are too grammatically complex to reliably understand.
- **✅ Vast Knowledge Base Seeding:** The agent begins its life with a large, pre-seeded set of foundational knowledge about itself, the world, and abstract concepts.

## ✅ Model Rendering & Deployment (Complete & Stable)

- **✅ Versioned Model Format (`.axm`):** A custom "Axiom Mind" file format packages the agent's brain and cache into a single, portable, versioned snapshot.
- **✅ Model Renderer:** A dedicated script (`render_model.py`) allows for the easy creation of new model versions after training milestones.
- **✅ Read-Only Inference App:** A separate, lightweight chat application (`chat_app.py`) can load any `.axm` model, allowing for safe, distributable deployment of a "finished" agent that cannot learn or change its mind.
- **✅ Remote Access with `ngrok`:** Both the training and inference apps have integrated support for exposing the local server to the internet via an optional `--ngrok` flag.

## ✅ User Interface & Experience (Complete & Stable)

- **✅ Multi-Conversation Management:** The UI supports multiple, persistent chat sessions, which are saved to the browser's local storage.
- **✅ Chat History Sidebar:** Users can easily navigate, review, and delete previous conversations.
- **✅ Rich Message Interaction:** Individual chat bubbles feature a menu with a cross-browser compatible "Copy to Clipboard" function.
- **✅ PWA (Progressive Web App):** The agent is fully configured as an installable web application with a service worker and custom icons, allowing for a native-like experience on both desktop and mobile.
- **✅ Performance & Stability:**
    - **Persistent Caching:** Caches interpretations to disk for instantaneous responses to repeated queries.
    - **Anticipatory Caching:** The code is in place to proactively pre-warm the cache after autonomous learning, but is **disabled by default** due to persistent stability issues.
    - **Thread-Safe Operation:** A global lock protects the agent from memory corruption during simultaneous user interaction and autonomous learning.

---

## 🛑 On Hold / Deprecated Features

*Features that have been implemented but are currently too unstable for active use. The code remains in the project but is disabled by a "kill switch."*

### 1. Conversational Context (Short-Term Memory)
- **Status:** **`On Hold / Disabled`**. The architectural approach of using a separate LLM call to resolve context has proven to be unreliable and is a frequent source of bugs. The feature has been disabled to ensure core system stability.
- **Goal:** Enable the agent to reliably understand and answer follow-up questions with pronouns.
- **Path Forward:** This feature requires a complete architectural rethink. A future solution might involve a more powerful base model or a different logical approach entirely.

---

## Phase 2: Multimodal Interface (Voice)

*Focus: Evolve the agent from a text-based entity to a voice-interactive companion. This is the next major development priority.*

### 1. Custom Voice Cloning
- **Goal:** Create a unique, high-quality voice for the agent using a custom audio dataset.
- **Technology:** Utilize open-source tools like **Coqui TTS**.
- **Implementation Plan:**
  1.  **Data Preparation:** Meticulously prepare a dataset of high-quality audio clips (5-15 seconds, complete thoughts) and their exact transcriptions.
  2.  **Training:** Use a cloud GPU service (like Google Colab) to fine-tune a TTS model on the custom dataset.
  3.  **Export:** The final output will be a unique voice model that defines the Axiom Agent's speech.

### 2. Speech-to-Text (STT) and Text-to-Speech (TTS) Integration
- **Goal:** Build the full, end-to-end voice interface.
- **Technology:** **Whisper** for STT and the custom **Coqui TTS** model for TTS.
- **Implementation Plan:**
  1.  Integrate the custom voice model into the application to convert the agent's text responses into speech.
  2.  Integrate Whisper to transcribe user microphone input into text.
  3.  Upgrade the UI with microphone controls and audio playback capabilities to create a seamless, voice-first conversational experience.