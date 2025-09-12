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
    - **Self-Aware & Resilient Learning:** Intelligently checks its own memory to avoid re-learning topics. It maintains a short-term "rejection memory" to avoid getting stuck on unlearnable topics during a session.
    - **Intelligent Fact Sourcing:** Uses a multi-stage fallback (NYT API -> Wikipedia -> DuckDuckGo) and a "Simplicity Filter" to find high-quality, simple, and learnable facts from the real world.
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
    - **Anticipatory Caching:** Proactively pre-warms the cache for newly learned topics, with safety checks to prevent caching failed responses.
    - **Thread-Safe Operation:** A global lock protects the agent from memory corruption during simultaneous user interaction and autonomous learning.

---

## Phase 2: Advanced Interaction

*Focus: Stabilize and deepen the agent's conversational abilities.*

### 1. Stabilize Conversational Context (`#1 Priority`)
- **Status:** **`In Progress / Buggy`**. The core code is implemented but failed verification. The agent is not correctly resolving pronouns in follow-up questions.
- **Goal:** Enable the agent to reliably understand and answer follow-up questions.
- **Example:**
  - `what is an apple` -> Agent answers.
  - `what color is it` -> Agent should know "it" refers to the apple.
- **Next Steps:**
  1.  Re-run the test scenario to replicate the failure.
  2.  Analyze the terminal logs to pinpoint the exact point of failure in the new `resolve_context` logic.
  3.  Refine the `resolve_context` prompt or the `CognitiveAgent`'s logic to ensure robust performance.

---

## Phase 3: Multimodal Interface (Voice)

*Focus: Evolve the agent from a text-based entity to a voice-interactive companion.*

### 1. Custom Voice Cloning
- **Goal:** Create a unique, high-quality voice for the agent using a custom audio dataset.
- **Technology:** Utilize open-source tools like **Coqui TTS**.
- **Implementation Plan:**
  1.  **Data Preparation:** Meticulously prepare a dataset of high-quality audio clips (5-15 seconds, complete sentences) and their exact transcriptions.
  2.  **Training:** Use a cloud GPU service (like Google Colab) to fine-tune a TTS model on the custom dataset.
  3.  **Export:** The final output will be a unique voice model that defines the Axiom Agent's speech.

### 2. Speech-to-Text (STT) and Text-to-Speech (TTS) Integration
- **Goal:** Build the full, end-to-end voice interface.
- **Technology:** **Whisper** for STT and the custom **Coqui TTS** model for TTS.
- **Implementation Plan:**
  1.  Integrate the custom voice model into the application to convert the agent's text responses into speech.
  2.  Integrate Whisper to transcribe user microphone input into text.
  3.  Upgrade the UI with microphone controls and audio playback capabilities to create a seamless, voice-first conversational experience.