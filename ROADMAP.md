# Axiom Agent: Development Roadmap

This document outlines the current status, planned features, and architectural improvements for the Axiom Agent project.

## ✅ Core Architecture & Capabilities (Complete & Stable)

This foundational genesis phase is complete. The agent is a fully autonomous, reasoning, and self-correcting learning entity with a robust user interface.

- **✅ Hybrid Cognitive Architecture:** The agent's mind is a hybrid of a symbolic **Knowledge Graph** (for verifiable, persistent memory) and a neural **Universal Interpreter** (for flexible language understanding), giving it the strengths of both paradigms.
- **✅ Multi-Hop Logical Reasoning:** The agent can answer complex questions by traversing its knowledge graph and connecting multiple facts to infer new conclusions.
- **✅ Dual-Cycle Autonomous Learning (Study & Discovery):** The agent's learning is not just passive. It operates on a sophisticated "Cognitive Scheduler" with two distinct modes:
    - **The Study Cycle (Knowledge Integration):** On a frequent timer, the agent introspectively reviews its own knowledge. It then uses a specialized "curiosity" tool to generate its own follow-up questions about concepts it wants to understand more deeply, actively seeking to enrich its own brain.
    - **The Discovery Cycle (Knowledge Acquisition):** On a less frequent timer, the agent actively seeks out brand new, unknown topics from the outside world, ensuring it continues to broaden its horizons.
- **✅ Temporal Reasoning:** The agent can perceive, learn, and reason with time-based facts. It understands the context of words like "now" or "currently" and can determine the most relevant fact from a series of historical events.
- **✅ Dual-Mode Learning System:** The agent possesses two distinct but complementary mechanisms for learning and self-correction.
    - **The Curiosity Engine (Implicit Correction):** Actively identifies when new information conflicts with its existing world model and autonomously asks clarifying questions to resolve its own confusion.
    - **User-Driven Correction (Explicit Correction):** Allows a user to take direct control of the learning process by using a `correction:` command to override, punish, and replace incorrect facts in the agent's brain.
- **✅ Resilient Knowledge Harvester:** The agent's tool for acquiring knowledge is designed for resilience. It uses a hybrid discovery model (Wikipedia Categories, NYT Archives), maintains a "rejection memory" to avoid getting stuck on unlearnable topics, and uses a "Simplicity Filter" to discard grammatically complex facts.
- **✅ Vast Knowledge Base Seeding:** The agent begins its life with a large, pre-seeded set of foundational knowledge about itself, the world, and abstract concepts.

## ✅ Scalability & Performance Optimization (Genesis Phase Complete)

This critical architectural overhaul is complete. The agent's core has been re-engineered for long-term stability and high-performance operation on local CPU hardware.

- **✅ Knowledge Graph Engine Upgrade:** The original, custom Python-based graph has been successfully replaced with a **`NetworkX`** engine. This provides a battle-tested, C-optimized foundation that eliminates traversal bottlenecks and ensures the agent can scale to tens of thousands of facts without performance degradation.
- **✅ Multi-Layer Caching System:** A sophisticated, multi-layer caching system has been implemented to dramatically reduce query latency:
    - **Reasoning Cache (`lru_cache`):** The most computationally expensive function—multi-hop graph traversal—is now cached in memory. This prevents the agent from re-thinking the same logical paths, resulting in an exponential performance increase for repeated queries.
    - **Interpreter & Synthesizer Cache:** All calls to the underlying LLM for both interpretation and sentence synthesis are cached to disk, ensuring that identical language tasks are instantaneous.
- **✅ Bombproof Stability:** The notoriously unstable **Discovery Cycle** has been permanently fixed. A new "pre-filtering" and sanitization guardrail now protects the core LLM from malformed, real-world data, eliminating the segmentation faults that caused frequent crashes.
- **✅ Verified Performance:** The success of this phase has been empirically verified using **`cProfile`** and custom timing tests. The results confirmed that all Python-level bottlenecks have been eliminated and that cached query times are now under the **0.01-second** mark, far exceeding the initial 2-second goal.

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
- **✅ Thread-Safe Operation:** A global lock protects the agent from memory corruption during simultaneous user interaction and autonomous learning cycles.

---

## 🛑 On Hold / Deprecated Features

*Features that have been implemented but are currently unstable or have been superseded by architectural changes.*

### 1. Dynamic Fact Salience (`access_count`)
- **Status:** **`On Hold / Partially Deprecated`**. The mechanism for tracking the `access_count` of facts during queries was found to be incompatible with the high-performance `lru_cache` system.
- **Path Forward:** This feature has been temporarily disabled in favor of massive performance gains. It can be re-introduced in a future phase with a more sophisticated, cache-aware architecture if needed.

### 2. Conversational Context (Short-Term Memory)
- **Status:** **`On Hold / Disabled`**. The architectural approach of using a separate LLM call to resolve context has proven to be unreliable. The feature has been disabled to ensure core system stability.
- **Path Forward:** This feature requires a complete architectural rethink. A future solution might involve a more powerful base model or a different logical approach entirely.

---

## Phase 1: Local CPU Optimization & Stability (Completed)

This foundational phase focused on transforming the agent from a fragile prototype into a stable, efficient, and scalable system capable of running reliably on local CPU hardware (iMac i5). All objectives for this phase have been successfully met.

### Flaw Addressed: Scalability of the Knowledge Graph

-   **Milestone Achieved:** The agent can now handle a large-scale knowledge graph with fast query times, laying the groundwork for massive, lifelong learning.

-   **Completed Steps & Successes:**
    1.  **Replaced Core Graph Engine:** The original, slow, custom dictionary-based graph in `graph_core.py` was successfully replaced with a high-performance `NetworkX` `MultiDiGraph`. This provides C-optimized traversals and robust, industry-standard data structures.
    2.  **Implemented Multi-Layer Caching:** To ensure high performance, a two-layer caching system was implemented in `cognitive_agent.py` and `universal_interpreter.py`:
        -   **Reasoning Cache (`lru_cache`):** The most expensive graph traversal function, `_gather_facts_multihop`, is now cached.
        -   **Synthesis Cache (Dictionary):** The final LLM-based language generation step in `synthesize` is also cached.
        -   **Bombproof Invalidation:** The caches are intelligently and automatically cleared the moment the agent learns a new fact, preventing the use of stale data.
    3.  **Performance Profiling:** Using Python's `cProfile`, we empirically proved that over 95% of the system's runtime was spent in the LLM. This confirms that our Python logic for graph traversal, caching, and autonomous cycles is now highly efficient.
    4.  **Achieved Success Metrics:**
        -   **Query Speed:** The `verify.py` script confirmed that cached queries now resolve in **~0.0005 seconds**, vastly exceeding the sub-2-second goal.
        -   **System Stability:** All known critical bugs, including fatal segmentation faults, amnesia on restart, and Study Cycle errors, have been resolved. The agent's autonomous learning cycles have been proven stable.

---

## Future Development Roadmap (Next 12 Months)

This section outlines the strategic, multi-phase plan for the agent's continued development. The plan is structured around addressing core architectural flaws in stages, leveraging free-tier cloud resources to achieve massive scale and intelligence without incurring costs.

### Free Cloud GPU/Resource Strategy

-   **Google Colab**: Free T4 GPU (up to 16GB VRAM), 12-hour session limits, idle timeouts after 90 minutes. You can run Jupyter notebooks for code execution, model inference, or simulations. Restart sessions as needed; use Google Drive (free) for persistent storage.
-   **Kaggle**: Free P100/T4 GPUs, 9-hour kernels, 20GB persistent datasets. Ideal for data-intensive tasks like knowledge sourcing or benchmarking; kernels can be scheduled or run interactively.
-   **AWS SageMaker Studio Lab**: Completely free ML environment with GPU access (T4-equivalent), 15GB storage, no time limits per session but overall usage quotas (e.g., 4 hours GPU/day). Supports full Python setups for fine-tuning or long runs.
-   **Hugging Face Spaces/Lightning.ai**: Free GPU for model hosting/demos (limited runtime, e.g., 1-2 hours) or student/hobbyist plans; useful for quick inference bursts.

These resources cover all future GPU needs in the roadmap. Limitations will be managed by saving frequent checkpoints and rotating services if quotas are hit.

---

## Flaw 1: Scalability of the Knowledge Graph

### Phase 1: Short-Term Fixes (0-3 Months) - CPU-Efficient Optimization
- **Status:** **COMPLETED**

### Phase 2: Medium-Term Enhancements (3-6 Months) - Free Cloud Database Integration
- **Milestone Explanation**: Scale to 500k nodes by storing the graph in free cloud databases, offloading storage/querying from the local iMac's limited RAM.
- **Steps Explanation**:
  1. **Migrate to free graph DB**: Use Neo4j Aura free tier or RedisGraph (via Redis Cloud free).
     - Why: Local JSON files bloat; clouds handle large data without crashing the local machine.
     - How: Sign up for free accounts; use Python drivers to connect.
  2. **Partition graph**: Label nodes by domains (e.g., "science" vs. "history") for loading only needed parts.
     - Why: Reduces memory load during syncs.
     - How: Add labels in queries (e.g., Neo4j Cypher: `MATCH (n:Science)`).
  3. **Salience tweaks**: Add decay (e.g., reduce `access_count` by 1% daily) to prune unused edges.
     - Why: Keeps graph lean, preventing bloat.
     - How: Simple math in code.
  4. **Test in Colab**: Upload the project to Colab and run simulations with larger data.
     - Why: Colab's free GPU/RAM tests scalability beyond the iMac.
- **Files Modified or Added**:
  - Modify `graph_core.py`: Add cloud connection code.
  - Modify `knowledge_harvester.py`: Update cycles to query the cloud.
  - Add `cloud_utils.py`: New file for helper functions like syncing.
- **Resources/Tools**: Neo4j Aura/Redis Cloud, Google Colab.
- **Success Metrics**: Graph syncs to cloud without errors. Pruning reduces size by 15%.

### Phase 3: Long-Term Optimizations (6-12+ Months) - Hybrid Scaling
- **Milestone Explanation**: Enable "near-unlimited" growth by distributing across free services.
- **Steps Explanation**:
  1. **Distributed free storage**: Use Kaggle datasets for backups; shard graph across multiple free accounts.
  2. **Offload traversals**: Run complex queries in SageMaker Studio Lab (free GPU).
  3. **Compression**: Use `zlib` to zip graph data.
  4. **Benchmark**: Compare reasoning to LLMs using cloud runs.
- **Files Modified or Added**:
  - Modify `graph_core.py`: Add sharding logic.
  - Add `benchmark.py`: New script for tests against LLMs.
- **Resources/Tools**: Kaggle, SageMaker, zlib.
- **Success Metrics**: Handle 1M+ nodes. Cloud queries 5x faster than local.

---

## Flaw 2: LLM Dependency and Interpretation Risks

### Phase 1: Short-Term Fixes (0-3 Months) - CPU Prompt Optimization
- **Milestone Explanation**: Reduce parsing errors by 40% on the iMac.
- **Steps Explanation**:
  1. **Refine prompts**: Add few-shot examples to LLM calls.
  2. **Validation layers**: Use Pydantic to check JSON output.
  3. **Simpler inputs**: Break complex sentences in the filter.
  4. **Test**: Create a local `test_dataset.txt` and score accuracy.
- **Files Modified or Added**:
  - Modify `universal_interpreter.py`: Update `interpret()` prompts and add validation.
  - Add `test_dataset.txt`.
- **Resources/Tools**: Pydantic.
- **Success Metrics**: Manual scoring on dataset shows <15% errors.

### Phase 2: Medium-Term Enhancements (3-6 Months) - Free GPU Ensemble
- **Milestone Explanation**: Achieve 85% accuracy by combining models, using GPU bursts.
- **Steps Explanation**:
  1. **Multi-model**: Add a smaller model (e.g., Phi-2 GGUF) for voting.
  2. **Bias checks**: Implement rule-based flags.
  3. **Cloud bursts**: Run tough interpretations in Colab GPU.
  4. **User loop**: Batch interpretations for user review.
- **Files Modified or Added**:
  - Modify `universal_interpreter.py`: Add ensemble logic.
  - Add `cloud_inference.py`: For GPU offloading.
- **Resources/Tools**: Colab/Kaggle, Hugging Face.
- **Success Metrics**: Tested handling of complexity; 3x speed improvement.

### Phase 3: Long-Term Optimizations (6-12+ Months) - Reduced Dependency
- **Milestone Explanation**: Cut LLM use for interpretation, targeting >90% accuracy.
- **Steps Explanation**:
  1. **Custom quantization**: Lower model bit-rate for speed.
  2. **Rule-based hybrids**: Use `spaCy` for basic entity recognition.
  3. **Self-data**: Generate and fine-tune a smaller model in SageMaker.
  4. **Evaluate**: Cloud benchmarks.
- **Files Modified or Added**:
  - Modify `universal_interpreter.py`: Add hybrid parsers.
  - Add `fine_tune_script.py`.
- **Resources/Tools**: llama.cpp, SageMaker, spaCy.
- **Success Metrics**: 60% fewer LLM calls for interpretation; high accuracy.

---

## Flaw 3: Autonomous Learning Loops

### Phase 1: Short-Term Fixes (0-3 Months) - CPU-Adaptive Loops
- **Milestone**: Stable, faster cycles with adaptive timing.
- **Steps**: Implement persistent memory for rejected topics; use `psutil` to create dynamic timers based on CPU load; local caches for sources; simulate.
- **Files**: Modify `knowledge_harvester.py`.
- **Resources**: `psutil` library.
- **Metrics**: No cycle hangs or crashes.

### Phase 2: Medium-Term Enhancements (3-6 Months) - Smarter Curation
- **Milestone**: Denser, more relevant knowledge acquisition.
- **Steps**: Use templates for question generation; re-introduce salience for topic selection; use Kaggle APIs for data sourcing; integrate.
- **Files**: Modify `knowledge_harvester.py`.
- **Resources**: Kaggle APIs.
- **Metrics**: Lower rate of failed learning attempts.

### Phase 3: Long-Term Optimizations (6-12+ Months) - Autonomous Scaling
- **Milestone**: An efficient, self-driven learning curriculum.
- **Steps**: Develop heuristics for reinforcement learning (rewarding successful topics); fuse multiple APIs; set configurable learning goals; compare.
- **Files**: Add `rl_heuristics.py`.
- **Resources**: SageMaker.
- **Metrics**: Achieves learning efficiency comparable to a baseline LLM.

---

## Flaw 4: User Reliance and Self-Correction

### Phase 1: Short-Term Fixes (0-3 Months) - Basic Rules
- **Milestone**: Partial self-resolution of knowledge conflicts.
- **Steps**: Implement rule-based conflict detection; queue conflicts for later review; simulate.
- **Files**: Modify `cognitive_agent.py`.
- **Resources**: Local.
- **Metrics**: Fewer conversation stalls due to contradictions.

### Phase 2: Medium-Term Enhancements (3-6 Months) - Probabilistic Tools
- **Milestone**: High degree of autonomous error correction.
- **Steps**: Add probabilities to facts; use web searches in Colab to verify conflicting facts; stabilize context memory.
- **Files**: Modify `graph_core.py`.
- **Resources**: Colab.
- **Metrics**: Increased independence from user correction.

### Phase 3: Long-Term Optimizations (6-12+ Months) - Advanced Awareness
- **Milestone**: Minimal reliance on the user for knowledge integrity.
- **Steps**: Implement meta-reasoning about knowledge confidence; use cloud services for large-scale verification; long-duration runs.
- **Files**: Add `meta_reasoning.py`.
- **Resources**: Kaggle.
- **Metrics**: High adaptability to new, potentially incorrect information.

---

## Flaw 5: Security and Stability

### Phase 1: Short-Term Fixes (0-3 Months) - Local Safeguards
- **Milestone**: High uptime and resilience to deadlocks.
- **Steps**: Implement thread timeouts; enhance logging for easier debugging; stress test.
- **Files**: Modify `app.py`.
- **Resources**: Local.
- **Metrics**: Zero crashes during extended local runs.

### Phase 2: Medium-Term Enhancements (3-6 Months) - Feature Stability
- **Milestone**: All features can be enabled without compromising stability.
- **Steps**: Optimize and stabilize context memory; add feature toggles/controls.
- **Files**: Modify `universal_interpreter.py`.
- **Resources**: Local.
- **Metrics**: Full feature set is stable.

### Phase 3: Long-Term Optimizations (6-12+ Months) - Cloud Robustness
- **Milestone**: Agent is ready for load testing and potential deployment.
- **Steps**: Containerize the application with Docker; perform code reviews for security.
- **Files**: Add `Dockerfile`.
- **Resources**: Docker.
- **Metrics**: Can handle bursts of simulated requests.

---

## Flaw 6: General Overheads

### Phase 1: Short-Term Fixes (0-3 Months) - Efficiency Tweaks
- **Milestone**: Reduced CPU and memory footprint.
- **Steps**: Experiment with 4-bit quantization for the LLM; optimize all data caches.
- **Files**: Modify `universal_interpreter.py`.
- **Resources**: llama.cpp.
- **Metrics**: Faster startup and cycle execution times.

### Phase 2: Medium-Term Enhancements (3-6 Months) - Cloud Migration
- **Milestone**: A hybrid system capable of 24/7 operation.
- **Steps**: Migrate autonomous cycles to run in Colab; implement rules for when to use local vs. cloud resources.
- **Files**: Add `cloud_migration.py`.
- **Resources**: Free cloud services.
- **Metrics**: Can sustain learning sessions for 24 hours.

### Phase 3: Long-Term Optimizations (6-12+ Months) - Sustainable Scaling
- **Milestone**: A fully scalable, sustainable, and powerful learning architecture.
- **Steps**: Create automated restart/recovery scripts for cloud sessions; implement ethical safeguards for autonomous learning.
- **Files**: Modify `app.py`, add deployment scripts.
- **Resources**: Multiple cloud accounts.
- **Metrics**: Competes with baseline benchmarks for knowledge acquisition.