#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- 🚀 Starting Code Quality and Test Suite ---"

# --- Code Formatting Check ---
echo -e "\n[1/4] Checking code formatting with Ruff..."
ruff format --check .
echo "✅ Formatting looks good."

# --- Linting ---
echo -e "\n[2/4] Linting with Ruff..."
ruff check .
echo "✅ Linter found no issues."

# --- Static Type Checking ---
echo -e "\n[3/4] Static type checking with MyPy..."
mypy
echo "✅ Type checking passed."

# --- Unit & Integration Tests ---
echo -e "\n[4/4] Running unit tests with Pytest and Coverage..."

# Step A: Erase any old coverage data.
echo -e "\nErasing old coverage data..."
python -m coverage erase

# Step B: Run pytest in parallel. change the cores to 1 instead of 4 if you need.
pytest -q --cov=axiom --cov-append

# This checks if there are any files matching the .coverage.* pattern.
if ls .coverage.* 1> /dev/null 2>&1; then
    echo -e "\nCombining coverage reports from all processes..."
    python -m coverage combine
else
    echo -e "\nNo parallel coverage data to combine. Skipping."
fi

# Step D: Generate the final, combined reports.
echo -e "\nGenerating final coverage report..."
python -m coverage report --show-missing --fail-under=0
python -m coverage xml
python -m coverage html

echo "✅ All tests passed."

echo -e "\n--- 🎉 All checks passed successfully! ---"