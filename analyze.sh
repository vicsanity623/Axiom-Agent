#!/usr/bin/env bash
# before using - make this file executable.
set -euo pipefail

WORKDIR="${PWD}"
VENV="${WORKDIR}/venv"
REPORTS_DIR="${WORKDIR}/analysis_reports"

echo "=== preparing workspace ==="
mkdir -p "$REPORTS_DIR"

if [ -z "${VIRTUAL_ENV-}" ]; then
    echo "=== creating venv (if not already active) ==="
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
fi

echo "=== installing analysis tools ==="
pip install --upgrade pip
pip install radon pylint ruff vulture

echo "=== installing project dependencies ==="
pip install -e '.[dev]'

echo "=== running radon cyclomatic complexity ==="
radon cc -s -a src > "${REPORTS_DIR}/radon_cc.txt"
radon cc -s src > "${REPORTS_DIR}/radon_cc_per_file.txt"

echo "=== running radon maintainability index ==="
radon mi src > "${REPORTS_DIR}/radon_mi.txt"

echo "=== running pylint ==="
pylint src --output-format=text > "${REPORTS_DIR}/pylint.txt" || true

echo "=== running ruff (fast linter) ==="
ruff check src --quiet || true
ruff check src -- --format=github > "${REPORTS_DIR}/ruff.txt" || true

echo "=== running mypy ==="
mypy src > "${REPORTS_DIR}/mypy.txt" || true

echo "=== running vulture (dead code finder) ==="
vulture src > "${REports_DIR}/vulture.txt" || true

echo "=== packaging results ==="
cd "$REPORTS_DIR"
tar czf ../analysis_reports.tar.gz .
cd ..
echo "===> Done. Reports saved to: ${WORKDIR}/analysis_reports.tar.gz"
echo "You can now inspect the text files in the '${REPORTS_DIR}' folder."