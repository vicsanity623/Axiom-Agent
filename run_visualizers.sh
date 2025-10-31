#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# This makes the script behave like your '&&' chain.
# If you want scripts to run independently, remove this line.
set -e

# Define variables for easier maintenance
PROJECT_DIR="/PATH/TO/YOUR/Axiom-Agent"
PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python3" # PATH TO YOUR .VENV

# Echo the current timestamp to the log for better debugging
echo "---"
echo "Running visualizer scripts at $(date)"

# Change to the project directory
cd "$PROJECT_DIR"

# Run the scripts
echo "Running visualize_brain.py..."
"$PYTHON_EXEC" src/axiom/scripts/visualize_brain.py

echo "Running visualize_fibonacci_galaxy.py..."
"$PYTHON_EXEC" src/axiom/scripts/visualize_fibonacci_galaxy.py

echo "Running visualize_seed_of_life.py..."
"$PYTHON_EXEC" src/axiom/scripts/visualize_seed_of_life.py

echo "All scripts finished successfully."
echo "---"