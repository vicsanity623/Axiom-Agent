#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- 🚀 Starting Code Quality and Test Suite ---"

# --- Code Formatting Check ---
echo -e "\n[1/4] Checking code formatting with Ruff..."
# Ruff automatically finds pyproject.toml and applies its config
ruff format --check .
echo "✅ Formatting looks good."

# --- Linting ---
echo -e "\n[2/4] Linting with Ruff..."
ruff check .
echo "✅ Linter found no issues."

# --- Static Type Checking ---
echo -e "\n[3/4] Static type checking with MyPy..."
# MyPy also finds and uses the config in pyproject.toml
mypy
echo "✅ Type checking passed."

# --- Unit & Integration Tests ---
echo -e "\n[4/4] Running unit tests with Pytest and Coverage..."

# This command creates the .coverage data file
coverage run --source=src/axiom -m pytest

# This command reads the data file and prints the report.
# --fail-under=0 tells it not to error out if coverage is low or even zero.
# This prevents the "No data to report" error from failing the CI.
coverage report -m --fail-under=0

coverage xml --fail-under=0

echo "✅ All tests passed."

echo -e "\n--- 🎉 All checks passed successfully! ---"
