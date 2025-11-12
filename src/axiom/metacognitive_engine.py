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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, NamedTuple, cast

import google.generativeai as genai

from .config import SRC_ROOT

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
except ImportError:
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
    """Persisted baseline stats used to make thresholds adaptive over time."""

    error_count_ewma: float = 0.0
    mean_time: float = 0.0
    var_time: float = 0.0
    samples: int = 0
    last_observed_time: float = 0.0

    def update_time(self, new_val: float, alpha: float = 0.2):
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


@dataclass
class _LogEntry:
    """Structured representation of a single parsed log line."""

    function_name: str
    message: str
    timestamp: int
    log_level: str
    execution_time: float | None = None
    traceback: str | None = None


@dataclass
class _LogAnalysisData:
    entries_by_func: dict[str, list[_LogEntry]] = field(
        default_factory=lambda: defaultdict(list)
    )
    deferred_targets: list[OptimizationTarget] = field(default_factory=list)


class PerformanceMonitor:
    """Analyzes logs to find performance bottlenecks, errors, and learning inefficiencies."""

    _LOG_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"""^\[(?P<ts>\d{2}:\d{2}:\d{2})\]\s+
            (?P<level>\w+)\s+
            (?P<message>.*?)\s*
            (?:\(in\s+(?P<function>[\w\.]+)\))?$
            """,
        re.VERBOSE,
    )
    _TIMING_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"Execution time[:=]?\s*(?P<seconds>\d+\.\d+)\s*s?", re.IGNORECASE
    )
    _LEARNING_LOG_RE: ClassVar[re.Pattern] = re.compile(
        r"Learned new fact: (?P<sub>.+) --\[(?P<verb>.+)\]--> (?P<obj>.+) \(status=(?P<status>\w+)\)"
    )
    _SCORING_WEIGHTS: ClassVar[dict[str, float]] = {
        "error_recurrence": 1.5,
        "timing_anomaly": 1.0,
        "semantic_cluster": 0.6,
        "learning_inefficiency": 1.2,
    }

    def __init__(self, baseline_store: Path = BASELINE_STORE_PATH):
        self.baseline_store = baseline_store
        self.baselines: dict[str, BaselineStats] = self._load_baselines()

    def _load_baselines(self) -> dict[str, BaselineStats]:
        if not self.baseline_store.exists():
            return {}
        try:
            raw = json.loads(self.baseline_store.read_text(encoding="utf-8"))
            return {k: BaselineStats(**v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError) as e:
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

    def _compute_embeddings(self, texts: list[str]) -> list[list[float]] | None:
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
        if not messages:
            return {}
        embs = self._compute_embeddings(messages)
        if embs:
            clusters: list[list[float]] = []
            cluster_sizes: list[int] = []
            for emb in embs:
                placed = False
                for idx, centroid in enumerate(clusters):
                    if self._cosine_similarity(emb, centroid) > 0.80:
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

    def _recent_git_commits(self, since_seconds: int = 3600 * 24) -> list[dict]:
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
            if not out:
                return []
            commits = []
            for raw in out.split("\x1e"):
                parts = raw.strip().split("\x1f")
                if len(parts) >= 4:
                    commits.append(
                        {
                            "hexsha": parts[0],
                            "author": parts[1],
                            "timestamp": int(parts[2]),
                            "message": parts[3],
                        },
                    )
            return commits
        except (FileNotFoundError, ValueError):
            return []

    def _correlate_with_commits(
        self,
        anomaly_time: int,
        window_seconds: int = 3600 * 6,
    ) -> list[dict]:
        commits = self._recent_git_commits(since_seconds=3600 * 24 * 7)
        return [
            c
            for c in commits
            if abs(int(c["timestamp"]) - anomaly_time) <= window_seconds
        ]

    def find_optimization_target(
        self, log_file: Path, introspector: CodeIntrospector
    ) -> OptimizationTarget | None:
        if not log_file.exists():
            logger.warning("[Metacognition]: Log file not found at %s.", log_file)
            return None
        analysis_data = self._parse_log_file(log_file)
        scored_candidates = self._score_anomalies(analysis_data)
        best_target = self._select_best_target(
            analysis_data, scored_candidates, introspector
        )
        self._save_baselines()
        if best_target:
            logger.info(
                "[Metacognition]: Identified optimization target: %s",
                best_target.target_name,
            )
        else:
            logger.info(
                "[Metacognition]: No clear optimization targets found this cycle."
            )
        return best_target

    def _parse_log_file(self, log_file: Path) -> _LogAnalysisData:
        """Parses a log file, including high-priority heuristics for critical issues."""
        lines = log_file.read_text(encoding="utf-8").splitlines()
        analysis_data = _LogAnalysisData()
        i = 0
        while i < len(lines):
            line = lines[i]

            # --- RESTORED: High-priority heuristic from original version ---
            if "Deferred learning" in line or "Consolidation incomplete" in line:
                message = line.strip()
                target = OptimizationTarget(
                    file_path=Path("src/axiom/knowledge_harvester.py"),
                    target_name="KnowledgeHarvester.study_cycle",
                    issue_description=(
                        "High-priority systemic issue: The learning or consolidation process is failing, "
                        "leading to deferred facts or incomplete cycles. This suggests a flaw in the core learning loop."
                    ),
                    relevant_logs=message,
                )
                analysis_data.deferred_targets.append(target)
                i += 1
                continue

            match = self._LOG_PATTERN.search(line)
            if not match:
                i += 1
                continue

            groups = match.groupdict()
            message = (groups.get("message") or "").strip()
            func = (groups.get("function") or "unknown_function").strip()
            log_level = (groups.get("level") or "INFO").upper()
            ts = int(time.time())
            timing_match = self._TIMING_PATTERN.search(message)
            exec_time = float(timing_match.group("seconds")) if timing_match else None
            traceback_str = None
            i += 1

            # Preserve the robust, multi-line traceback parsing logic
            if log_level in ("ERROR", "CRITICAL") or "traceback" in message.lower():
                if i < len(lines) and "Traceback (most recent call last):" in lines[i]:
                    tb_lines = [lines[i]]  # Start with the "Traceback..." line
                    i += 1
                    while i < len(lines) and (
                        lines[i].startswith("  ") or "Error:" in lines[i]
                    ):
                        tb_lines.append(lines[i].rstrip())
                        i += 1

                    if tb_lines:
                        traceback_str = "\n".join(tb_lines)
                        # Extract function name from the last "File" line
                        for tb_line in reversed(tb_lines):
                            if tb_line.strip().startswith("File "):
                                tb_match = re.search(
                                    r"in\s+<module>|in\s+([a-zA-Z0-9_]+)", tb_line
                                )
                                if tb_match and tb_match.group(1):
                                    func = tb_match.group(1)
                                    break
                else:
                    # Rewind to ensure the next line is processed if it wasn't part of a traceback
                    i -= 1

            entry = _LogEntry(func, message, ts, log_level, exec_time, traceback_str)
            analysis_data.entries_by_func[func].append(entry)
        return analysis_data

    def _score_anomalies(
        self, analysis_data: _LogAnalysisData
    ) -> dict[str, tuple[float, list[str]]]:
        candidate_scores = {}
        candidate_reasons = defaultdict(list)
        for func, entries in analysis_data.entries_by_func.items():
            errors = [e.message for e in entries if e.log_level == "ERROR"]
            warnings = [e.message for e in entries if e.log_level == "WARNING"]
            timings = [
                e.execution_time for e in entries if e.execution_time is not None
            ]
            if func not in self.baselines:
                self.baselines[func] = BaselineStats()
            baseline = self.baselines[func]
            observed_err_count = len(errors)
            baseline.update_error_count(observed_err_count)
            err_delta = observed_err_count - baseline.error_count_ewma
            err_score = (
                (err_delta / (1.0 + baseline.error_count_ewma))
                if baseline.error_count_ewma > 0
                else float(observed_err_count)
            )
            if err_score > 0.5:
                candidate_reasons[func].append(
                    f"error recurrence high (observed={observed_err_count}, baseline_ewma={baseline.error_count_ewma:.2f})",
                )
            timing_score = 0.0
            if timings:
                observed_mean = statistics.mean(timings)
                for t in timings:
                    baseline.update_time(t)
                if baseline.samples > 1:
                    baseline_stdev = math.sqrt(baseline.var_time)
                    timing_score = (observed_mean - baseline.mean_time) / (
                        baseline_stdev + 1e-6
                    )
                else:
                    timing_score = observed_mean / (1.0 + observed_mean)
                if timing_score > 2.0:
                    candidate_reasons[func].append(
                        f"timing anomaly (mean={observed_mean:.3f}s vs baseline={baseline.mean_time:.3f}s)",
                    )
            cluster_counts = self._semantic_cluster_counts(errors + warnings)
            largest_cluster = max(cluster_counts.values()) if cluster_counts else 0
            sem_score = math.log1p(largest_cluster)
            if sem_score > 0.5:
                candidate_reasons[func].append(
                    f"semantic cluster concentration (largest={largest_cluster})",
                )

            composite_score = (
                PerformanceMonitor._SCORING_WEIGHTS["error_recurrence"]
                * max(0.0, err_score)
                + PerformanceMonitor._SCORING_WEIGHTS["timing_anomaly"]
                * max(0.0, timing_score)
                + PerformanceMonitor._SCORING_WEIGHTS["semantic_cluster"] * sem_score
            )

            if composite_score > 0:
                candidate_scores[func] = (composite_score, candidate_reasons[func])

        learning_score, learning_reasons, responsible_func = (
            self._score_learning_quality(analysis_data)
        )
        if learning_score > 0 and responsible_func:
            weighted_score = (
                self._SCORING_WEIGHTS["learning_inefficiency"] * learning_score
            )
            if responsible_func in candidate_scores:
                existing_score, existing_reasons = candidate_scores[responsible_func]
                candidate_scores[responsible_func] = (
                    existing_score + weighted_score,
                    existing_reasons + learning_reasons,
                )
            else:
                candidate_scores[responsible_func] = (weighted_score, learning_reasons)
        return candidate_scores

    def _score_learning_quality(
        self,
        analysis_data: _LogAnalysisData,
    ) -> tuple[float, list[str], str | None]:
        """
        Analyzes the quality of the agent's learning process.
        Returns the inefficiency score, reasons, and the responsible "meta-function".
        """
        learned_facts = []
        all_concepts = set()

        # The learning logs are now correctly attributed by the parser,
        # but we can still look them up by the known function name.
        learning_entries = analysis_data.entries_by_func.get(
            "CognitiveAgent._add_new_fact", []
        )
        for entry in learning_entries:
            match = self._LEARNING_LOG_RE.search(entry.message)
            if match:
                subj, obj = match.group("sub").strip(), match.group("obj").strip()
                learned_facts.append((subj, obj))
                all_concepts.add(subj.lower())
                all_concepts.add(obj.lower())

        if not learned_facts:
            return 0.0, [], None

        responsible_func = "CognitiveAgent.LearningProcess"
        total_facts = len(learned_facts)
        simple_facts_count = sum(
            1 for s, o in learned_facts if len(s.split()) + len(o.split()) <= 3
        )
        simple_fact_ratio = simple_facts_count / total_facts
        fact_counts = Counter(f"{s.lower()}::{o.lower()}" for s, o in learned_facts)
        duplicate_events = sum(count - 1 for count in fact_counts.values())
        duplication_ratio = duplicate_events / total_facts
        discovery_ratio = len(all_concepts) / total_facts if total_facts > 0 else 0
        low_discovery_score = max(0, 1.0 - discovery_ratio)
        inefficiency_score = simple_fact_ratio + duplication_ratio + low_discovery_score
        reasons = []
        if simple_fact_ratio > 0.5:
            reasons.append(f"high ratio of simple facts ({simple_fact_ratio:.2f})")
        if duplication_ratio > 0.2:
            reasons.append(f"high ratio of duplicate facts ({duplication_ratio:.2f})")
        if low_discovery_score > 0.3:
            reasons.append(
                f"low new concept discovery rate (ratio={discovery_ratio:.2f})",
            )

        return inefficiency_score, reasons, responsible_func

    def _select_best_target(
        self,
        analysis_data: _LogAnalysisData,
        scored_candidates: dict[str, tuple[float, list[str]]],
        introspector: CodeIntrospector,
    ) -> OptimizationTarget | None:
        """Selects the best target, restoring the logic for high-priority alerts and meta-functions."""
        if analysis_data.deferred_targets:
            logger.warning(
                "[Metacognition]: Prioritizing 'Deferred learning' issue over other anomalies."
            )
            return analysis_data.deferred_targets[0]
        for func, entries in analysis_data.entries_by_func.items():
            first_error_entry = next((e for e in entries if e.traceback), None)
            if first_error_entry and first_error_entry.traceback:
                file_path = introspector.get_source_file(func)
                if not file_path:
                    logger.error(
                        "[Metacognition]: Could not find source file for traceback in function '%s'.",
                        func,
                    )
                    continue
                logger.warning(
                    "[Metacognition]: Prioritizing critical error (traceback) in '%s'.",
                    func,
                )
                return OptimizationTarget(
                    file_path=file_path,
                    target_name=func,
                    issue_description=f"A critical error with a traceback occurred in `{func}`.",
                    relevant_logs=first_error_entry.traceback,
                )

        if not scored_candidates:
            logger.info(
                "[Metacognition]: No scored candidates after parsing; nothing to act on."
            )
            return None

        func, (top_score, reasons) = max(
            scored_candidates.items(), key=lambda item: item[1][0]
        )

        if func == "CognitiveAgent.LearningProcess":
            threshold = 1.0  # A fixed, sensitive threshold for systemic issues
            final_target_name = "CognitiveAgent._process_statement_for_learning"
            file_path = introspector.get_source_file(final_target_name)
        else:
            default_threshold = 1.5
            env_thresh = float(os.getenv("METACOG_THRESHOLD", str(default_threshold)))
            baseline = self.baselines.get(func, BaselineStats())
            threshold = env_thresh + 0.5 * baseline.error_count_ewma
            final_target_name = func
            file_path = introspector.get_source_file(func)

        if not file_path:
            logger.error(
                f"[Metacognition]: Could not find source file for top-scored candidate '{func}'."
            )
            return None

        logger.info(
            f"[Metacognition]: Candidate '{func}' score={top_score:.3f} threshold={threshold:.3f} reasons={reasons}"
        )
        if top_score < threshold:
            logger.info(
                f"[Metacognition]: Top candidate '{func}' below threshold; skipping."
            )
            return None

        entries = analysis_data.entries_by_func.get(func, [])
        anomaly_time = entries[0].timestamp if entries else int(time.time())
        correlated_commits = self._correlate_with_commits(anomaly_time)
        commit_hint = ""
        if correlated_commits:
            commit_hint = "Recent commits within anomaly window:\n" + "\n".join(
                f"- {c['hexsha'][:8]} {c['author']}: {c['message']}"
                for c in correlated_commits
            )

        relevant_log_messages = [
            e.message for e in entries if e.log_level in ("ERROR", "WARNING")
        ]
        if "high ratio" in " ".join(reasons):
            relevant_log_messages.extend(
                [e.message for e in entries if self._LEARNING_LOG_RE.search(e.message)]
            )
        sample_logs = "\n".join(list(dict.fromkeys(relevant_log_messages))[:8])
        issue_description = f"Automated analysis flagged `{func}` with anomaly score {top_score:.2f}. Contributing factors: {', '.join(reasons or ['elevated composite score'])}."
        if commit_hint:
            issue_description += f"\n\n{commit_hint}"

        return OptimizationTarget(
            file_path=file_path,
            target_name=func,
            issue_description=issue_description,
            relevant_logs=sample_logs or "(no sample logs captured)",
        )


# ---------------------------------------------------------------------------
# CodeIntrospector, ExternalAnalysisBridge, SandboxVerifier, SelfModifier
# ---------------------------------------------------------------------------


class CodeIntrospector:
    """Dynamically builds a map of all functions in the project for easy source retrieval."""

    def __init__(self) -> None:
        self.function_map: dict[str, Path] = self._build_function_map()

    @cached_property
    def function_map(self) -> dict[str, Path]:
        return self._build_function_map()

    def _build_function_map(self) -> dict[str, Path]:
        """Builds a map of all function/method names to file paths by walking the AST."""
        logger.info("[Metacognition]: Building source code function map...")
        function_map: dict[str, Path] = {}

        for file_path in Path(SRC_ROOT).rglob("*.py"):
            if "tests" in file_path.parts:
                continue
            try:
                source_code = file_path.read_text(encoding="utf-8")
                tree = ast.parse(source_code)

                for node in ast.walk(tree):
                    for child in ast.iter_child_nodes(node):
                        setattr(child, "parent", node)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        parent = getattr(node, "parent", None)
                        if isinstance(parent, ast.ClassDef):
                            qualified_name = f"{parent.name}.{node.name}"
                            function_map[qualified_name] = file_path
                        elif isinstance(
                            parent, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            pass
                        else:
                            function_map[node.name] = file_path
            except Exception as e:
                logger.warning("[Metacognition]: Failed to parse %s: %s", file_path, e)

        logger.info(
            "[Metacognition]: Function map built with %d entries.", len(function_map)
        )
        return function_map

    def get_source_file(self, function_name: str) -> Path | None:
        """Get the source file path for a function using a multi-pass lookup."""
        if function_name in self.function_map:
            return self.function_map[function_name]
        short_name = function_name.split(".")[-1]
        if short_name in self.function_map:
            return self.function_map[short_name]
        for key, path in self.function_map.items():
            if key.endswith(f".{short_name}"):
                return path
        return None

    def get_function_source(self, function_name: str) -> tuple[Path | None, str | None]:
        """Retrieves the full source code of a function using the pre-built map."""
        file_path = self.get_source_file(function_name)
        if not file_path or not file_path.exists():
            logger.error(
                "[Metacognition]: Source file not found for function: %s", function_name
            )
            return None, None

        try:
            source_code = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source_code)

            target_parts = function_name.split(".")
            target_name = target_parts[-1]
            class_name = target_parts[0] if len(target_parts) > 1 else None

            if class_name:
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        for sub_node in node.body:
                            if (
                                isinstance(
                                    sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)
                                )
                                and sub_node.name == target_name
                            ):
                                return file_path, ast.get_source_segment(
                                    source_code, sub_node
                                )
            else:
                for node in ast.walk(tree):
                    if (
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == target_name
                    ):
                        return file_path, ast.get_source_segment(source_code, node)

        except Exception as e:
            logger.error(
                "[Metacognition]: Error reading source for %s: %s", function_name, e
            )

        logger.warning(
            "Could not find a specific match for '%s' in '%s'.",
            function_name,
            file_path,
        )
        return file_path, None


class ExternalAnalysisBridge:
    """Manages secure interaction with the Gemini API for code analysis and refactoring."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initializes the bridge, configures the API key, and detects the best model."""
        api_key_final = api_key or os.getenv("GEMINI_API_KEY")
        if api_key_final is None:
            raise ValueError(
                "Gemini API key is required. Pass it to the constructor or set the GEMINI_API_KEY environment variable.",
            )

        genai.configure(api_key=api_key_final)
        self.model_name = self._detect_best_model()
        self.model = genai.GenerativeModel(self.model_name)
        logger.warning(
            "[warning]   - Initialized with model: %s[/warning]", self.model_name
        )

    def _detect_best_model(self) -> str:
        """Detects the best available Gemini model that supports content generation."""
        try:
            logger.warning(
                "[warning]   - Detecting best available model for PerformanceMonitor...[/warning]"
            )
            available_models = genai.list_models()
            supported_models = [
                m.name
                for m in available_models
                if "generateContent" in m.supported_generation_methods
            ]
            if "models/gemini-2.5-pro" in supported_models:
                return "gemini-2.5-pro"
            if "models/gemini-2.5-flash" in supported_models:
                return "gemini-2.5-flash"
            logger.warning(
                "[yellow][ExternalAnalysisBridge] Preferred models not found. Falling back to default.[/yellow]"
            )
        except Exception as e:
            logger.warning(
                "[error][ExternalAnalysisBridge] Model detection failed (%s). Using fallback.[/error]",
                e,
            )
        return "gemini-1.5-pro-latest"

    def _extract_python_code_from_markdown(self, markdown_text: str) -> str:
        """
        Finds and extracts the first Python code block from a markdown string.
        If no block is found, assumes the entire string is code.
        """
        match = re.search(r"```python\n(.*?)```", markdown_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return markdown_text.strip().strip("`")

    def get_code_suggestion(
        self,
        problematic_code: str,
        issue_description: str,
        relevant_logs: str,
    ) -> str | None:
        """Constructs a strictly constrained prompt and retrieves a type-safe, compliant code suggestion."""
        traceback_section = ""
        if "Traceback (most recent call last):" in relevant_logs:
            traceback_section = f"**TRACEBACK:**\n```\n{relevant_logs}\n```"
            relevant_logs = "(See traceback above for primary context.)"

        prompt = f"""
        **ROLE:** You are a senior Python software refactoring engine operating under enterprise-grade static analysis rules.

        **CONTEXT:** This code is part of an autonomous cognitive agent system. The verification pipeline enforces Ruff formatting and `mypy --strict`. All generated code must be valid and type-safe.

        **CRITICAL TYPE HINTING RULE:** The codebase uses `typing.TypedDict` for data structures like `RelationData` and `PropertyData`. When you create a dictionary literal that should conform to one of these types, you MUST use `typing.cast` to ensure type safety.
        Example: `my_var = cast("RelationData", {{\"subject\": "a", "verb": "is", "object": "b\"}})`

        **ISSUE DESCRIPTION:** {issue_description}
        {traceback_section}

        **RELEVANT LOGS:**
        ```
        {relevant_logs}
        ```

        **PROBLEMATIC SOURCE CODE:**
        ```python
        {problematic_code}
        ```

        **CODING STANDARDS & CONSTRAINTS (MANDATORY):**
        1. Code must pass `mypy --strict` with no warnings or errors.
        2. Adhere to the CRITICAL TYPE HINTING RULE regarding `TypedDict` and `cast`.
        3. Code must be Ruff-compliant and well-formatted.
        4. All returns must match declared types.
        5. Include complete docstrings.

        **YOUR TASK:** Rewrite the function above while strictly adhering to all standards.
        **OUTPUT RULES:** Return ONLY the function code with full docstring, no markdown or commentary.
        """

        try:
            logger.info(
                "[Metacognition]: Sending code analysis request to external LLM with strict static rules..."
            )
            response = self.model.generate_content(prompt)
            suggested_code_raw = getattr(response, "text", None)

            if suggested_code_raw is None:
                logger.error("[Metacognition]: Gemini returned no text response.")
                return None

            clean_code = self._extract_python_code_from_markdown(suggested_code_raw)

            if not clean_code.startswith("def "):
                logger.warning(
                    "[Metacognition]: Gemini response was not valid Python code after cleaning. Response: %s",
                    clean_code,
                )
                return None

            logger.info(
                "[Metacognition]: Received and successfully parsed code suggestion from Gemini."
            )
            return clean_code
        except Exception as e:
            logger.error(
                "[Metacognition]: Code suggestion request failed: %s", e, exc_info=True
            )
            return None


class SandboxVerifier:
    """Verifies that a suggested code change is safe and effective by running

    the project's own 'check.sh' script in a fully isolated clone."""

    def verify_change(
        self,
        original_file_path: Path,
        original_function_name: str,
        new_function_code: str,
    ) -> bool:
        """
        Creates a hermetically sealed copy of the project, applies the new code,
        automatically formats it, and then runs the 'check.sh' quality gate
        from within the isolated environment.
        """
        logger.info(
            "[Metacognition]: Beginning fully isolated sandboxed verification...",
        )

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sandbox_root = Path(tmpdir)
                project_root = Path.cwd()

                logger.info("  [Sandbox]: Cloning project into isolated environment...")
                ignore_patterns = shutil.ignore_patterns(
                    ".venv",
                    ".git",
                    ".idea",
                    "__pycache__",
                    "*.pyc",
                    "*.log",
                    "*.bak",
                    ".metacog_baseline.json",
                    "code_suggestion.json",
                    "models",
                    "visualizations",
                )
                shutil.copytree(
                    project_root,
                    sandbox_root,
                    dirs_exist_ok=True,
                    ignore=ignore_patterns,
                )

                sandboxed_file_to_patch = sandbox_root / original_file_path
                logger.info(
                    "  [Sandbox]: Applying patch to '%s' in sandbox.",
                    sandboxed_file_to_patch,
                )

                original_source = sandboxed_file_to_patch.read_text(encoding="utf-8")
                tree = ast.parse(original_source)

                class FunctionReplacer(ast.NodeTransformer):
                    def visit_FunctionDef(self, node):
                        if node.name == original_function_name:
                            new_function_node = ast.parse(new_function_code).body[0]
                            ast.fix_missing_locations(new_function_node)
                            return new_function_node
                        return node

                new_tree = FunctionReplacer().visit(tree)
                sandboxed_file_to_patch.write_text(
                    ast.unparse(new_tree),
                    encoding="utf-8",
                )

                logger.info("  [Sandbox]: Running multi-pass auto-fixer (Ruff)...")

                subprocess.run(
                    ["ruff", "check", "--select", "F821", "--add-noqa", "."],
                    capture_output=True,
                    text=True,
                    cwd=sandbox_root,
                )

                subprocess.run(
                    ["ruff", "format", "."],
                    capture_output=True,
                    text=True,
                    cwd=sandbox_root,
                )

                fix_process = subprocess.run(
                    ["ruff", "check", "--fix", "."],
                    capture_output=True,
                    text=True,
                    cwd=sandbox_root,
                )

                if fix_process.returncode != 0:
                    logger.error(
                        "  [Sandbox]: ❌ Ruff auto-fix FAILED. The suggestion is likely unrecoverable."
                    )
                    logger.error("  [Sandbox]: Ruff stderr:\n%s", fix_process.stderr)
                    return False

                logger.info("  [Sandbox]: ✅ Auto-fixer applied successfully.")

                check_script_path = sandbox_root / "check.sh"
                if not check_script_path.exists():
                    logger.critical(
                        "[Metacognition]: 'check.sh' not found in sandbox. Cannot verify.",
                    )
                    return False

                logger.info(
                    "  [Sandbox]: Running 'check.sh' quality gate in sandbox...",
                )
                process = subprocess.run(
                    [
                        "bash",
                        str(check_script_path),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=sandbox_root,
                )

                if process.returncode == 0:
                    logger.info(
                        "  [Sandbox]: ✅ 'check.sh' passed in the isolated environment.",
                    )
                    return True

                logger.error(
                    "  [Sandbox]: ❌ 'check.sh' FAILED in the isolated environment.",
                )
                logger.error("  [Sandbox]: Script stdout:\n%s", process.stdout)
                logger.error("  [Sandbox]: Script stderr:\n%s", process.stderr)
                return False

        except Exception as e:
            logger.critical(
                "[Metacognition]: A critical error occurred during sandboxed verification: %s",
                e,
                exc_info=True,
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

            modified_source = ast.unparse(FunctionReplacer().visit(tree))
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
    """Performs a self-introspection cycle to generate and verify code improvement suggestions."""

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

    def _save_suggestion_report(
        self,
        target: OptimizationTarget,
        suggested_code: str,
        verification_passed: bool,
    ):
        """Saves a JSON report of the introspection cycle's findings, including verification status."""
        report_path = Path("code_suggestion.json")
        logger.info("[Metacognition]: Saving suggestion report to %s", report_path)
        try:
            target_dict = target._asdict()
            target_dict["file_path"] = str(target_dict["file_path"])

            status_message = (
                "Sandbox Verification PASSED. Ready for human review."
                if verification_passed
                else "Sandbox Verification FAILED. Suggestion should be discarded."
            )

            report_data = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "identified_target": target_dict,
                "suggested_solution": {
                    "code": suggested_code,
                    "verification_passed": verification_passed,
                    "status": status_message,
                },
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4)
        except Exception as e:
            logger.error(
                "[Metacognition]: Failed to save suggestion report. Error: %s",
                e,
            )

    def run_introspection_cycle(self) -> None:
        """Executes one full self-analysis cycle with improved visibility."""
        suggestion_file = Path("code_suggestion.json")
        if suggestion_file.exists() and suggestion_file.stat().st_size > 0:
            logger.info(
                "[Metacognition]: Skipping cycle: pending suggestion exists at '%s'. Remove or archive this file to resume cycles.",
                suggestion_file,
            )
            return
        if not self.analysis_bridge:
            logger.warning(
                "[Metacognition]: Self-modification disabled (no Gemini key). To enable, set GEMINI_API_KEY."
            )
            return

        logger.info("=" * 80)
        logger.info("--- [METACOGNITIVE CYCLE STARTED (Verified Advisory Mode)] ---")
        logger.info("=" * 80)

        try:
            log_file = Path("axiom.log")
            if not log_file.exists():
                logger.info(
                    "[Metacognition]: No axiom.log found at %s. Nothing to do.",
                    log_file,
                )
                return

            target = self.performance_monitor.find_optimization_target(
                log_file, self.code_introspector
            )
            if not target:
                return

            # 1. Get the tuple and unpack it immediately. We trust the type hint that it's always a tuple.
            file_path, problematic_code = self.code_introspector.get_function_source(
                target.target_name
            )

            # 2. Perform a simple, clear guard clause on each potentially None variable.
            if file_path is None:
                logger.error(
                    f"[Metacognition]: Could not determine file path for target {target.target_name}"
                )
                return

            if problematic_code is None:
                logger.error(
                    f"[Metacognition]: Could not extract source code for {target.target_name} in {file_path}"
                )
                return

            # By this point, mypy is guaranteed that file_path is a Path and problematic_code is a str.
            suggested_code = self.analysis_bridge.get_code_suggestion(
                problematic_code, target.issue_description, target.relevant_logs
            )
            if not suggested_code:
                logger.error(
                    "[Metacognition]: External analysis bridge returned no suggested code."
                )
                return

            base_function_name = target.target_name.split(".")[-1]
            is_safe = self.sandbox_verifier.verify_change(
                file_path, base_function_name, suggested_code
            )
            self._save_suggestion_report(target, suggested_code, is_safe)

        except Exception as e:
            logger.error(
                "--- [METACOGNITIVE CYCLE FAILED]: An unexpected error occurred: %s ---",
                e,
                exc_info=True,
            )
        finally:
            logger.info("=" * 80)
            logger.info("--- [METACOGNITIVE CYCLE FINISHED] ---")
            logger.info("=" * 80)
