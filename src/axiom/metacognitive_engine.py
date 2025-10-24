from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import google.generativeai as genai

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight configuration for metacognitive persistence and optional features
# ---------------------------------------------------------------------------
BASELINE_STORE_PATH = Path(".metacog_baseline.json")
EMBEDDING_MODEL_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDING_MODEL_AVAILABLE = True
    _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _EMB_MODEL = None
    EMBEDDING_MODEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class OptimizationTarget(NamedTuple):
    """Represents a specific function or method identified for optimization."""

    file_path: Path
    target_name: str
    issue_description: str
    relevant_logs: str


@dataclass
class BaselineStats:
    """
    Persisted baseline stats used to make thresholds adaptive over time.
    Stored per-function:
      - error_count_ewma: exponentially-weighted moving average of error counts
      - mean_time: rolling mean execution time
      - var_time: rolling variance (for computing stdev)
      - samples: number of samples observed for timing
      - last_observed_time: most recent timing observed (new)
    """

    error_count_ewma: float = 0.0
    mean_time: float = 0.0
    var_time: float = 0.0
    samples: int = 0
    last_observed_time: float = 0.0

    def update_time(self, new_val: float, alpha: float = 0.2):
        """Update timing stats for this function."""
        if self.samples == 0:
            self.mean_time = new_val
            self.var_time = 0.0
            self.samples = 1
        else:
            old_mean = self.mean_time
            self.mean_time = (1 - alpha) * self.mean_time + alpha * new_val
            self.var_time = (1 - alpha) * self.var_time + alpha * (
                (new_val - old_mean) ** 2
            )
            self.samples += 1

        self.last_observed_time = new_val

    def update_error_count(self, new_count: int, alpha: float = 0.3):
        self.error_count_ewma = (1 - alpha) * self.error_count_ewma + alpha * new_count


# ---------------------------------------------------------------------------
# PerformanceMonitor - enhanced & adaptive
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """Analyzes logs to find performance bottlenecks or recurring errors using
    statistical, semantic, and contextual heuristics. This version is adaptive:
    it persists baselines, supports optional semantic clustering via embeddings,
    correlates anomalies with recent git commits, and updates learned thresholds.
    """

    def __init__(self, baseline_store: Path = BASELINE_STORE_PATH):
        self.baseline_store = baseline_store
        self.baselines: dict[str, BaselineStats] = self._load_baselines()

    # ----------------------
    # persistence for adaptivity
    # ----------------------
    def _load_baselines(self) -> dict[str, BaselineStats]:
        if not self.baseline_store.exists():
            return {}
        try:
            raw = json.loads(self.baseline_store.read_text(encoding="utf-8"))
            return {k: BaselineStats(**v) for k, v in raw.items()}
        except Exception as e:
            logger.warning("[Metacognition]: Failed to load baseline store: %s", e)
            return {}

    def _save_baselines(self):
        try:
            payload = {k: asdict(v) for k, v in self.baselines.items()}
            self.baseline_store.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("[Metacognition]: Failed to save baseline store: %s", e)

    # ----------------------
    # embeddings & semantic clustering (optional, best-effort)
    # ----------------------
    def _compute_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """
        Compute embeddings for a list of text snippets if sentence-transformers is installed.

        Returns None if embeddings are not available.
        """
        if not EMBEDDING_MODEL_AVAILABLE or _EMB_MODEL is None:
            return None
        try:
            embs = _EMB_MODEL.encode(texts, show_progress_bar=False)
            return cast(
                "list[list[float]]",
                embs.tolist() if hasattr(embs, "tolist") else embs,
            )
        except Exception as e:
            logger.warning("[Metacognition]: Embedding computation failed: %s", e)
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-12)

    def _semantic_cluster_counts(self, messages: list[str]) -> dict[int, int]:
        """
        If embeddings are available, cluster messages by simple greedy affinity
        (single pass) and return cluster sizes. This is lightweight but effective
        for small sets. Fall back to hashing-based pseudo-clusters otherwise.
        """
        if not messages:
            return {}

        embs = self._compute_embeddings(messages)
        if embs:
            clusters: list[list[float]] = []
            cluster_sizes: list[int] = []
            for emb in embs:
                placed = False
                for idx, centroid in enumerate(clusters):
                    sim = self._cosine_similarity(emb, centroid)
                    if sim > 0.80:
                        clusters[idx] = [
                            (c * cluster_sizes[idx] + e) / (cluster_sizes[idx] + 1)
                            for c, e in zip(centroid, emb)
                        ]
                        cluster_sizes[idx] += 1
                        placed = True
                        break
                if not placed:
                    clusters.append(emb[:])
                    cluster_sizes.append(1)
            return dict(enumerate(cluster_sizes))
        cnt = Counter(sha256(m.encode("utf-8")).hexdigest()[:8] for m in messages)
        return dict(enumerate(cnt.values()))

    # ----------------------
    # simple git correlation (causal hinting)
    # ----------------------
    def _recent_git_commits(self, since_seconds: int = 3600 * 24) -> list[dict]:
        """
        Return recent git commits as dicts with 'hexsha', 'author', 'timestamp', 'message'.
        If git is not available or repo not present, returns empty list.
        """
        try:
            since_dt = int(time.time()) - since_seconds
            cmd = [
                "git",
                "log",
                f"--since={since_dt}",
                "--pretty=format:%H%x1f%an%x1f%at%x1f%s%x1e",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = result.stdout.strip("\n\x1e")
            commits: list[dict[str, int | str]] = []
            if not out:
                return commits
            for raw in out.split("\x1e"):
                parts = raw.strip().split("\x1f")
                if len(parts) < 4:
                    continue
                commits.append(
                    {
                        "hexsha": parts[0],
                        "author": parts[1],
                        "timestamp": int(parts[2]),
                        "message": parts[3],
                    },
                )
            return commits
        except Exception:
            return []

    def _correlate_with_commits(
        self,
        anomaly_time: int,
        window_seconds: int = 3600 * 6,
    ) -> list[dict]:
        """
        Correlate an anomaly timestamp with nearby commits within a +/- window.
        Returns a list of commits that fall into that window (heuristic causal hints).
        """
        commits = self._recent_git_commits(since_seconds=3600 * 24 * 7)  # last week
        return [
            c for c in commits if abs(c["timestamp"] - anomaly_time) <= window_seconds
        ]

    # ----------------------
    # main analysis routine
    # ----------------------
    def find_optimization_target(self, log_file: Path) -> OptimizationTarget | None:
        """
        Advanced analysis of logs to detect performance bottlenecks or recurring errors.

        Enhancements:
        - Tracks error & warning frequencies per function and maintains EWMA baselines.
        - Detects performance anomalies via timing statistics and compares against learned baselines.
        - Uses optional semantic clustering of log messages to surface concentrated issues.
        - Correlates anomalies with recent git commits to provide causal hints.
        - Persists updated baselines (adaptive system).
        - Detects deferred learning entries and flags them for review.
        """
        if not log_file.exists():
            logger.warning(
                "[Metacognition]: Log file not found at %s. Cannot analyze performance.",
                log_file,
            )
            return None

        lines = log_file.read_text(encoding="utf-8").splitlines()

        log_pattern = re.compile(
            r"\[(?P<module>[^\]]+)\]:\s*(?P<message>.+?)(?:\s*\(in\s*(?P<function>[\w\.]+)\))?$",
        )
        timing_pattern = re.compile(
            r"Execution time[:=]?\s*(?P<seconds>\d+\.\d+)s",
            re.IGNORECASE,
        )

        errors_by_func = defaultdict(list)
        warnings_by_func = defaultdict(list)
        timings_by_func = defaultdict(list)
        first_seen_time_by_func = {}

        ts_pattern = re.compile(r"^(?P<iso>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
        deferred_fact_re = re.compile(
            r"Deferred learning for\s+(?P<fact>[\w\-\._]+)",
            re.IGNORECASE,
        )

        for idx, line in enumerate(lines):
            if "Deferred learning" in line:
                m = deferred_fact_re.search(line)
                fact_id = m.group("fact") if m else f"{idx}"
                func = f"DeferredLearning_{fact_id}"
                message = line.strip()

                try:
                    self._save_baselines()
                except Exception as e:
                    logger.warning(
                        "[Metacognition]: Failed to save baselines before returning deferred target: %s",
                        e,
                    )

                logger.info(
                    "[Metacognition]: Detected deferred learning entry for '%s' (line %d). Returning as optimization target.",
                    fact_id,
                    idx,
                )

                return OptimizationTarget(
                    file_path=Path("src/axiom/cognitive_agent.py"),
                    target_name=func,
                    issue_description=f"Deferred learning detected for `{fact_id}`: {message}",
                    relevant_logs=message,
                )

            match = log_pattern.search(line)
            if not match:
                continue

            message = (match.group("message") or "").strip()
            func = (match.group("function") or "unknown_function").strip()

            ts_match = ts_pattern.search(line)
            if ts_match:
                try:
                    t_struct = time.strptime(ts_match.group("iso"), "%Y-%m-%dT%H:%M:%S")
                    ts = int(time.mktime(t_struct))
                except Exception:
                    ts = int(time.time())
            else:
                ts = int(time.time())

            if func not in first_seen_time_by_func:
                first_seen_time_by_func[func] = ts

            upper = line.upper()
            if "ERROR" in upper:
                errors_by_func[func].append(message)
            elif "WARNING" in upper:
                warnings_by_func[func].append(message)

            t_match = timing_pattern.search(message)
            if t_match:
                try:
                    timings_by_func[func].append(float(t_match.group("seconds")))
                except Exception:
                    pass

        for func, msgs in errors_by_func.items():
            observed_err_count = len(msgs)
            if func not in self.baselines:
                self.baselines[func] = BaselineStats()
            self.baselines[func].update_error_count(observed_err_count)

        for func, times in timings_by_func.items():
            if func not in self.baselines:
                self.baselines[func] = BaselineStats()
            for t in times:
                self.baselines[func].update_time(t)

        candidate_scores: dict[str, float] = {}
        candidate_reasons: dict[str, list[str]] = {}
        now = int(time.time())

        for func, msgs in {**errors_by_func, **warnings_by_func}.items():
            cluster_counts = self._semantic_cluster_counts(msgs)
            largest_cluster = max(cluster_counts.values()) if cluster_counts else 0
            sem_score = math.log1p(largest_cluster) if largest_cluster > 0 else 0.0

            baseline = self.baselines.get(func, BaselineStats())
            observed_err = len(errors_by_func.get(func, []))
            err_delta = observed_err - baseline.error_count_ewma
            err_score = (
                (err_delta / (1.0 + baseline.error_count_ewma))
                if baseline.error_count_ewma > 0
                else float(observed_err)
            )

            timing_score = 0.0
            if timings_by_func.get(func):
                observed_mean = statistics.mean(timings_by_func[func])
                if baseline.samples > 1:
                    baseline_stdev = math.sqrt(baseline.var_time)
                    timing_score = (observed_mean - baseline.mean_time) / (
                        baseline_stdev + 1e-6
                    )
                else:
                    timing_score = observed_mean / (1.0 + observed_mean)

            score = (
                1.5 * max(0.0, err_score)
                + 1.0 * max(0.0, timing_score)
                + 0.6 * sem_score
            )
            if score > 0:
                candidate_scores[func] = score
                reasons = []
                if err_score > 0.5:
                    reasons.append(
                        f"error recurrence high (observed={observed_err}, baseline_ewma={baseline.error_count_ewma:.2f})",
                    )
                if timing_score > 2.0:
                    reasons.append(
                        f"timing anomaly (mean={observed_mean:.3f}s vs baseline={baseline.mean_time:.3f}s)",
                    )
                if sem_score > 0.5:
                    reasons.append(
                        f"semantic cluster concentration (largest={largest_cluster})",
                    )
                candidate_reasons[func] = reasons or [
                    "elevated composite anomaly score",
                ]

        if candidate_scores:
            func, top_score = max(candidate_scores.items(), key=lambda kv: kv[1])
            baseline = self.baselines.get(func, BaselineStats())
            adaptive_threshold = 2.0 + 0.5 * baseline.error_count_ewma
            logger.debug(
                "[Metacognition]: Candidate '%s' score=%.3f threshold=%.3f",
                func,
                top_score,
                adaptive_threshold,
            )

            if top_score >= adaptive_threshold:
                anomaly_time = first_seen_time_by_func.get(func, now)
                correlated_commits = self._correlate_with_commits(anomaly_time)
                commit_hint = ""
                if correlated_commits:
                    commit_hint = "Recent commits within anomaly window:\n" + "\n".join(
                        f"- {c['hexsha'][:8]} {c['author']}: {c['message']}"
                        for c in correlated_commits
                    )

                sample_logs = "\n".join(
                    list(
                        dict.fromkeys(
                            errors_by_func.get(func, [])
                            + warnings_by_func.get(func, []),
                        ),
                    )[:8],
                )
                issue_description = (
                    f"Automated analysis flagged `{func}` with anomaly score {top_score:.2f}. "
                    f"Contributing factors: {', '.join(candidate_reasons.get(func, []))}."
                )
                if commit_hint:
                    issue_description += f"\n\n{commit_hint}"

                self._save_baselines()

                target = OptimizationTarget(
                    file_path=self._guess_source_file(func),
                    target_name=func,
                    issue_description=issue_description,
                    relevant_logs=sample_logs or "(no sample logs captured)",
                )
                logger.info("[Metacognition]: Identified optimization target: %s", func)
                return target

        self._save_baselines()
        logger.info(
            "[Metacognition]: No clear optimization targets found in logs this cycle.",
        )
        return None

    # ----------------------
    # helper: guess source file (can be extended to symbol-table / AST index)
    # ----------------------
    def _guess_source_file(self, func_name: str) -> Path:
        """
        Attempts to infer the likely source file based on the function name.
        (In production this would query a real symbol table or index.)
        """
        mapping = {
            "_parse_single_clause": "src/axiom/symbolic_parser.py",
            "process_input": "src/axiom/core_agent.py",
            "update_graph": "src/axiom/graph_core.py",
        }
        base = func_name.split(".")[-1]
        return Path(
            mapping.get(func_name, mapping.get(base, "src/axiom/unknown_module.py")),
        )


# ---------------------------------------------------------------------------
# CodeIntrospector (unchanged except minor robustness)
# ---------------------------------------------------------------------------


class CodeIntrospector:
    """Reads and extracts specific code blocks from the agent's own source."""

    def get_function_source(self, file_path: Path, function_name: str) -> str | None:
        """
        Use AST to extract the full source code of a specific function or method.
        """

        if not file_path.exists():
            logger.error("[Metacognition]: Source file not found: %s", file_path)
            return None

        try:
            from typing import cast

            source_code = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source_code)

            target_parts = function_name.split(".")
            target_node = None

            if len(target_parts) > 1:
                class_name = target_parts[0]
                method_name = target_parts[1]
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        for sub_node in node.body:
                            if (
                                isinstance(sub_node, ast.FunctionDef)
                                and sub_node.name == method_name
                            ):
                                target_node = sub_node
                                break
                        break
            else:
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == function_name:
                        target_node = node
                        break

            if target_node:
                return cast("str", ast.unparse(target_node))

            logger.warning(
                "[Metacognition]: Could not find function '%s' in '%s'.",
                function_name,
                file_path,
            )
            return None

        except Exception as e:
            logger.error(
                "[Metacognition]: Failed to parse source code from '%s'. Error: %s",
                file_path,
                e,
            )
            return None


# ---------------------------------------------------------------------------
# ExternalAnalysisBridge, SandboxVerifier, SelfModifier, MetacognitiveEngine
# (kept unchanged except minor typing clarifications)
# ---------------------------------------------------------------------------


class ExternalAnalysisBridge:
    """Manages secure interaction with a powerful external LLM for code analysis."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def get_code_suggestion(
        self,
        problematic_code: str,
        issue_description: str,
        relevant_logs: str,
    ) -> str | None:
        """Constructs a detailed prompt and securely calls the Gemini Pro API."""

        prompt = f"""
        **ROLE:**
        You are an expert Python software refactoring engine. Your sole purpose is to improve Python code for clarity, performance, and correctness.

        **CONTEXT:**
        I am an autonomous AI agent analyzing my own source code. I have identified a problematic function based on my operational logs.

        **ISSUE DESCRIPTION:**
        {issue_description}

        **RELEVANT LOGS:**
        ```
        {relevant_logs}
        ```

        **PROBLEMATIC SOURCE CODE:**
        ```python
        {problematic_code}
        ```

        **YOUR TASK:**
        Rewrite the entire Python function provided above to address the described issue.

        **STRICT RULES:**
        1.  Your output MUST be ONLY the complete, new Python code for the function.
        2.  You MUST include the function signature (the `def` line) and the complete docstring.
        3.  DO NOT include any explanation, commentary, or markdown code fences (```).
        4.  The code must be syntactically correct and production-ready.
        """
        try:
            logger.info(
                "[Metacognition]: Sending code analysis request to external LLM...",
            )
            response = self.model.generate_content(prompt)

            suggested_code = cast("str", response.text).strip()

            if not suggested_code.startswith("def "):
                logger.warning(
                    "[Metacognition]: External LLM response was not valid Python code. Response: %s",
                    suggested_code,
                )
                return None

            logger.info("[Metacognition]: Received code suggestion from external LLM.")
            return suggested_code
        except Exception as e:
            logger.error(
                "[Metacognition]: Failed to get code suggestion from external LLM: %s",
                e,
            )
            return None


class SandboxVerifier:
    """Verifies that a suggested code change is safe and effective."""

    def verify_change(
        self,
        original_file_path: Path,
        original_function_name: str,
        new_function_code: str,
    ) -> bool:
        """
        Create a temporary, sandboxed version of a module and apply the new code.

        Then run the project's own test suite against it.
        """

        logger.info(
            "[Metacognition]: Beginning sandboxed verification of suggested code change...",
        )

        try:
            original_source = original_file_path.read_text(encoding="utf-8")
            tree = ast.parse(original_source)

            class FunctionReplacer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if node.name == original_function_name:
                        logger.info(
                            "  [Sandbox]: Found target function '%s'. Replacing node in AST.",
                            original_function_name,
                        )
                        new_function_node = ast.parse(new_function_code).body[0]
                        ast.fix_missing_locations(new_function_node)
                        return new_function_node
                    return node

            transformer = FunctionReplacer()
            new_tree = transformer.visit(tree)
            modified_source = ast.unparse(new_tree)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                sandboxed_src_path = tmp_path / "src" / "axiom"
                sandboxed_src_path.mkdir(parents=True, exist_ok=True)

                for file in Path("src/axiom").glob("*.py"):
                    if file.name != original_file_path.name:
                        shutil.copy(file, sandboxed_src_path)

                sandboxed_file_path = sandboxed_src_path / original_file_path.name
                sandboxed_file_path.write_text(modified_source, encoding="utf-8")

                pytest_executable = shutil.which("pytest")
                assert pytest_executable is not None, (
                    "pytest executable not found in PATH"
                )

                logger.info(
                    "  [Sandbox]: Running full pytest suite against modified code...",
                )

                process = subprocess.run(
                    [
                        pytest_executable,
                        "tests",
                        "--rootdir",
                        str(tmp_path),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                    env={**os.environ, "PYTHONPATH": f"{tmp_path}/src"},
                )

                if process.returncode == 0:
                    logger.info(
                        "  [Sandbox]: ✅ All tests passed against the sandboxed code.",
                    )
                    return True
                logger.error(
                    "  [Sandbox]: ❌ Tests FAILED against the sandboxed code.",
                )
                logger.error("  [Sandbox]: Pytest stdout:\n%s", process.stdout)
                logger.error("  [Sandbox]: Pytest stderr:\n%s", process.stderr)
                return False

        except Exception as e:
            logger.critical(
                "[Metacognition]: A critical error occurred during sandboxed verification: %s",
                e,
            )
            return False


class SelfModifier:
    """Applies a verified code change to the live source code."""

    def apply_live_patch(
        self,
        file_path: Path,
        target_name: str,
        new_function_code: str,
    ):
        """Safely overwrites a function in a source file, with backups."""
        if not file_path.exists():
            logger.error("[SelfModifier]: Target file does not exist: %s", file_path)
            return

        logger.warning(
            "!!! [SelfModifier]: Applying LIVE code patch to '%s' !!!",
            file_path,
        )

        backup_path = file_path.with_suffix(f".{int(time.time())}.bak")
        shutil.copy(file_path, backup_path)
        logger.info("  [SelfModifier]: Created backup at '%s'", backup_path)

        try:
            original_source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(original_source)

            class FunctionReplacer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if node.name == target_name:
                        new_node = ast.parse(new_function_code).body[0]
                        ast.fix_missing_locations(new_node)
                        return new_node
                    return node

            transformer = FunctionReplacer()
            new_tree = transformer.visit(tree)
            modified_source = ast.unparse(new_tree)

            file_path.write_text(modified_source, encoding="utf-8")
            logger.info(
                "  [SelfModifier]: ✅ Successfully applied patch to source file.",
            )
            print("\n" + "=" * 80)
            print("=== METACOGNITIVE SELF-MODIFICATION COMPLETE ===")
            print(
                f"The function '{target_name}' in '{file_path.name}' has been updated.",
            )
            print("A restart is required for the changes to take effect.")
            print("=" * 80 + "\n")

        except Exception as e:
            logger.critical(
                "  [SelfModifier]: ❌ FAILED to apply live patch! Restoring from backup. Error: %s",
                e,
            )
            shutil.move(backup_path, file_path)
            logger.info("  [SelfModifier]: Restored original file from backup.")


class MetacognitiveEngine:
    """Perform the self-introspection and modification cycle."""

    analysis_bridge: ExternalAnalysisBridge | None

    def __init__(self, agent: CognitiveAgent, gemini_api_key: str | None):
        self.agent = agent
        self.performance_monitor = PerformanceMonitor()
        self.code_introspector = CodeIntrospector()
        self.sandbox_verifier = SandboxVerifier()
        self.self_modifier = SelfModifier()

        if gemini_api_key:
            self.analysis_bridge = ExternalAnalysisBridge(api_key=gemini_api_key)
        else:
            self.analysis_bridge = None
            logger.warning(
                "[Metacognition]: Gemini API key not found. Self-modification is disabled.",
            )

    def run_introspection_cycle(self):
        """Executes one full 24-hour self-analysis and modification cycle."""
        if not self.analysis_bridge:
            return

        logger.info("=" * 80)
        logger.info("--- [METACOGNITIVE CYCLE STARTED] ---")
        logger.info("=" * 80)

        log_file = Path("axiom.log")
        target = self.performance_monitor.find_optimization_target(log_file)

        if not target:
            logger.info("--- [METACOGNITIVE CYCLE FINISHED]: No targets found. ---")
            return

        function_name = target.target_name.split(".")[-1]
        problematic_code = self.code_introspector.get_function_source(
            target.file_path,
            function_name,
        )

        if not problematic_code:
            logger.error(
                "--- [METACOGNITIVE CYCLE FAILED]: Could not retrieve source code for target. ---",
            )
            return

        suggested_code = self.analysis_bridge.get_code_suggestion(
            problematic_code,
            target.issue_description,
            target.relevant_logs,
        )

        if not suggested_code:
            logger.warning(
                "--- [METACOGNITIVE CYCLE FINISHED]: External LLM provided no suggestion. ---",
            )
            return

        is_safe = self.sandbox_verifier.verify_change(
            target.file_path,
            function_name,
            suggested_code,
        )

        if not is_safe:
            logger.error(
                "--- [METACOGNITIVE CYCLE FAILED]: Sandbox verification failed. Discarding change. ---",
            )
            return

        self.self_modifier.apply_live_patch(
            target.file_path,
            function_name,
            suggested_code,
        )

        logger.info("=" * 80)
        logger.info("--- [METACOGNITIVE CYCLE FINISHED SUCCESSFULLY] ---")
        logger.info("=" * 80)
