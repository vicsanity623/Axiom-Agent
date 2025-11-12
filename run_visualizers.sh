#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "---"
echo "Running visualizer scripts at $(date)"
echo "---"

# --- FIX: Use the CLI entry points defined in pyproject.toml ---
# This is the most robust method. It does not depend on file paths and
# automatically uses the Python from the activated virtual environment.

echo "Generating main brain visualization..."
axiom-visualize

echo "Generating Fibonacci galaxy visualization..."
axiom-galaxy1

echo "Generating Seed of Life visualization..."
axiom-galaxy2

echo "---"
echo "All visualization scripts finished successfully."
echo "---"