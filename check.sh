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

pytest --cov=src/axiom \
       --cov-report=term-missing \
       --cov-report=xml \
       --cov-fail-under=0

echo "✅ All tests passed."

echo -e "\n--- 🎉 All checks passed successfully! ---"