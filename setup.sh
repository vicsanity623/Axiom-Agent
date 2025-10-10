#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Setting up Axiom-Agent ---"

# 1. Check if python3 is available
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

# 2. Create a virtual environment named 'venv' if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 3. Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 4. Install project dependencies
echo "Installing dependencies from pyproject.toml..."
pip install -e ".[dev]"

# 5. Download NLTK data
echo "Downloading NLTK 'wordnet' data..."
python -c "import nltk; nltk.download('wordnet')"

echo ""
echo "--- Setup complete! ---"
echo "You can now run the agent. The virtual environment is active."