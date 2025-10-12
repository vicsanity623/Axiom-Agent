# Axiom‑Agent — Refactor & Recovery Roadmap

This roadmap turns the static‑analysis findings into a concrete, phased plan you (or contributors) can act on. It’s organized by **phases** (quick wins → deep refactor → hardening → launch/maintain). Each item includes **deliverables**, **acceptance criteria**, and **practical tasks** (files to add or change). Use this as a living plan: mark checklist items done, attach PR links, and iterate. 

***

The Axiom Agent has successfully evolved through its first three foundational phases, transforming from a basic cognitive engine into a sophisticated, conversational learner. It began with the **"Genesis Engine,"** establishing a stable core for knowledge. Next, the **"Intelligent Learner"** phase enabled the agent to learn with purpose, autonomously expanding its vocabulary and discovering new topics. Most recently, the **"Conversational & Introspective Mind"** gave it the ability to parse complex sentences, use short-term memory, and even learn from its own responses to continuously refine its knowledge base.

This rapid progress has proven the agent's core concepts, but now it is time to solidify the foundation for future growth. The following **"Refactor & Recovery Roadmap"** is a detailed engineering plan to address the technical debt incurred during this period of rapid innovation. By focusing on repository hygiene, testing, and targeted refactoring, we are building a more stable, robust, and maintainable platform. This essential work will pave the way for the agent's next major functional leaps: achieving a **"Hardened Mind"** with a deeper understanding of language, becoming an **"Autonomous Scholar"** with strategic learning, and finally, evolving into a **"Distributed Mind"** for massive scalability.

---

## How to use this roadmap

* Treat each bullet as a discrete ticket/issue. Keep PRs small and focused (one major refactor or 3–5 small fixes per PR).
* Each phase has *must‑do* items (required before moving on) and *nice‑to‑have* items (optional, high ROI).
* For each completed item, add a one‑line note with a link to the PR that resolves it.

---

# Phase 0 — Repository hygiene (must do first)

**Goal:** make the repo safe, reproducible, and instrumented so all later work is testable and visible.

### Deliverables

* `.github/workflows/ci.yml` that runs tests, linters, mypy, radon and uploads coverage artifacts.
* `Makefile` or `setup.sh` with `make setup`, `make test`, `make lint`, `make report` targets.
* `setup/download_model.sh` that supports `AXIOM_MODEL_PATH` env var and verifies checksums.
* `SECURITY.md`, `CODE_OF_CONDUCT.md`, and issue/PR templates.

### Tasks

* Create CI workflow file at `.github/workflows/ci.yml` (include steps to run `./check.sh` and `pytest --cov`).
* Add `Makefile` with common targets. Example: `setup`, `lint`, `test`, `coverage`, `format`.
* Implement `setup/download_model.sh` and document `AXIOM_MODEL_PATH` in `README.md`.
* Add `SECURITY.md` and templates in `.github/ISSUE_TEMPLATE/` and `.github/PULL_REQUEST_TEMPLATE.md`.

### Acceptance criteria

* CI runs on every push/PR and shows pass/fail for lint/tests.
* `make setup` boots dev environment (venv) and `make test` runs without fatal errors (tests may fail but command exists).
* Project has clear contribution/security guidelines.

---

# Phase 1 — Quick wins (low effort, high impact)

**Goal:** remove noise and reduce immediate technical debt so you can make bigger changes with confidence.

### Must‑do

* Run formatting: `black` + `ruff --fix` and commit as a single PR.
* Fix trivial `pylint` issues: add missing module docstrings, shorten long lines, and add small helper functions for repeated code.
* Remove or annotate dead code found by `vulture` (either delete or add `# pragma: no cover` + TODO comment).
* Add basic smoke tests and an `examples/symbolic_demo.sh` that boots symbolic-only mode and runs a short scenario.

### Deliverables

* `tools/format_and_lint.sh` (idempotent) and a PR applying formatting.
* `examples/symbolic_demo.sh` and `examples/README.md` showing how to run the demo.

### Acceptance criteria

* `ruff` and `black` report no diffs in CI run.
* `examples/symbolic_demo.sh` runs end‑to‑end in symbolic‑only mode on a fresh checkout (document any env setup required).

---

# Phase 2 — Tests, coverage, and core stabilization

**Goal:** raise confidence by adding tests and reducing core module complexity so regressions are caught early.

### Must‑do

* Add focused unit tests for these core modules (start here):

  * `src/axiom/symbolic_parser.py`
  * `src/axiom/graph_core.py`
  * `src/axiom/cognitive_agent.py` (high-level flows only, mock external deps)
  * `src/axiom/dictionary_utils.py` (after initial refactor into helpers)
* Add `conftest.py` fixtures to mock LLM calls and model IO.
* Configure coverage threshold in CI (baseline: require coverage gating for new code; overall target to be achieved later).

### Deliverables

* `tests/test_parser.py`, `tests/test_graph.py`, `tests/test_agent_flow.py` (mocked).
* `conftest.py` with fixtures: `tmp_cache`, `mock_llm`, `minimal_knowledge_base`.
* Coverage configuration in `pyproject.toml` or `coverage.rc`.

### Acceptance criteria

* Tests run in CI and create `coverage.xml` artifact.
* New tests prove the basic learn/query flows and reduce flakiness in developer runs.

---

# Phase 3 — Refactor hotspots (high‑impact code restructuring)

**Goal:** reduce cyclomatic complexity and split large modules into smaller, testable components.

### Top targets (from analysis)

* `src/axiom/dictionary_utils.py::get_word_info_from_wordnet` → **extract helpers**: `_fetch`, `_normalize`, `_extract_relations`, `_assemble`.
* `src/axiom/cognitive_agent.py` → **split into smaller modules**: `agent/preprocessing.py`, `agent/intent.py`, `agent/learning.py`, `agent/query.py`, `agent/dialog.py`.
* `src/axiom/lexicon_manager.py::add_linguistic_knowledge` → split responsibilities: parsing vs persistence.

### Strategy & tasks

* For each hotspot: create a branch `refactor/<module>-split` and open a PR with:

  1. small extraction commits (only helpers, no behavior change),
  2. unit tests for each helper,
  3. integration tests ensuring original behavior still holds.
* Add type hints to any newly extracted helper functions; run `mypy` on them.
* Replace `if/elif` chains with table-driven dispatch, strategy pattern, or small classes where appropriate.

### Deliverables

* PRs: `refactor/dictionary-utils`, `refactor/cognitive-agent-split-1`, `refactor/cognitive-agent-split-2` etc.
* Test coverage for refactored parts and CI passing.

### Acceptance criteria

* Each refactor PR reduces CC for targeted function(s) below threshold (CC < 10 preferred), and tests ensure no behavior regression.

---

# Phase 4 — Observability, provenance, and conflict handling

**Goal:** make the system debuggable and safe: reasoning traces, fact provenance, and deterministic conflict resolution.

### Must‑do

* Add structured reasoning traces (JSON) and an optional `--trace` flag to queries.
* Add provenance metadata to facts: `{source, timestamp, confidence, version}`.
* Implement a conflict detection and lightweight belief revision strategy (e.g., confidence scores, timestamps, source precedence).
* Add an exporter: `tools/export_graph.py` → Graphviz + JSON formats.

### Deliverables

* `src/axiom/tracing.py` + integration points in agent and graph operations.
* `tools/export_graph.py` and a demo script `examples/visualize_graph.sh`.

### Acceptance criteria

* Running a query with `--trace` emits a JSON trace that can be consumed to visualize reasoning paths.
* Conflicting facts create a human‑review queue entry (or clear resolution action) and are recorded with provenance.

---

# Phase 5 — Performance, scalability, and optional graph backends

**Goal:** ensure the concept graph and reasoning scale and remain responsive.

### Investigation steps (do before heavy migration)

* Add benchmarks: `bench/graph_insert.py` and `bench/graph_query.py` to test insert/query at scale (10k/100k facts). Use synthetic data.
* Profile hot paths using `pyinstrument` or `cProfile` and gather flamegraphs.

### If needed

* Introduce an optional graph DB backend (Neo4j, RedisGraph, or SQLite-based adjacency indices) behind an adapter interface. Keep native in‑memory implementation for small deployments.

### Acceptance criteria

* Benchmarks and profiling show where to optimize. If a DB backend is added, it should be pluggable and optional.

---

# Phase 6 — Hardening, docs, and release

**Goal:** ship a stable, documented release and create a repeating maintenance plan.

### Deliverables

* `CHANGELOG.md` and a `v1.0.0` tag when core features are stable.
* `docs/` (mkdocs) with architecture diagram, developer guide, and contributor onboarding.
* Release checklist in `.github/RELEASE_CHECKLIST.md`.

### Acceptance criteria

* A tagged release with a release note that documents what changed, known limitations, and migration notes.

---

# Ongoing & maintenance

* Weekly/biweekly: triage issues, run smoke tests, review PRs for complexity (radon check in CI to block big increases).
* Monthly: run performance benchmarks and update `ROADMAP.md` with new priorities.
* Keep a `TECH_DEBT.md` file listing known technical debt items and status.

---

# Quick PR / task templates (copy/paste)

**Refactor PR template**

```
Title: refactor: <module> — split <function> into helpers
Summary: what changed and why
Files changed: list
Testing: tests added/modified (list)
Checklist:
- [ ] Unit tests added
- [ ] Mypy passes
- [ ] Radon CC reduced for target function
- [ ] CI passes
```

**Bug / fix PR template**

```
Title: fix: <short description>
Summary: what broke and how fixed
Testing: tests added/modified (list)
Notes: any migration or compatibility notes
```

---

# Ready‑to‑run next actions (pick any and mark it done here)

* DONE `.github/workflows/ci.yml` and commit a first version.
* Create `examples/symbolic_demo.sh` and add to README quickstart.
* Open `refactor/dictionary-utils` branch and extract helpers for `get_word_info_from_wordnet`.

---
