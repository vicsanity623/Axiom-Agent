"""
Axiom Metacognitive Engine Entrypoint

Allows running self-analysis and optimization directly via:
    python -m axiom --analyze-logs axiom.log
"""

import json
import sys
from pathlib import Path

from axiom.metacognitive_engine import PerformanceMonitor


def analyze_logs_for_cycle(log_path="axiom.log"):
    """Run PerformanceMonitor and return structured result."""
    monitor = PerformanceMonitor()
    result = monitor.find_optimization_target(Path(log_path))
    if result:
        return {
            "file_path": str(result.file_path),
            "target_name": result.target_name,
            "issue_description": result.issue_description,
            "relevant_logs": result.relevant_logs,
        }
    return None


def main():
    if "--analyze-logs" in sys.argv:
        idx = sys.argv.index("--analyze-logs")
        log_file = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "axiom.log"
        result = analyze_logs_for_cycle(log_file)
        print(json.dumps(result, indent=2))
        return

    print(
        "Axiom Metacognitive Engine\n"
        "Usage:\n"
        "  python -m axiom --analyze-logs <path_to_log>\n"
        "\n"
        "Coming soon:\n"
        "  --cycle-now       Run full metacognitive self-modification cycle\n"
        "  --introspect      Inspect agent cognitive modules\n"
        "  --self-modify     Apply verified patches automatically\n",
    )


if __name__ == "__main__":
    main()
