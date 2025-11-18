#!/bin/bash

# This script MANUALLY downloads and unzips all necessary NLTK resources
# directly into the project's virtual environment (.venv), preserving the
# required directory structure.

set -e # Exit immediately if any command fails

VENV_DIR="./.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment directory '$VENV_DIR' not found."
    exit 1
fi
NLTK_DATA_DIR="$VENV_DIR/nltk_data"
BASE_URL="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"

echo "--- Manually Downloading and Installing NLTK Resources ---"
echo "Target directory: $NLTK_DATA_DIR"

download_and_unzip() {
    local category=$1
    local package=$2
    local step=$3
    
    echo "[$step] Downloading '$package'..."
    
    local target_dir="$NLTK_DATA_DIR/$category"
    mkdir -p "$target_dir"
    
    local temp_zip_file=$(mktemp)
    curl -L "$BASE_URL/$category/$package.zip" -o "$temp_zip_file"
    
    echo "Unzipping '$package' to '$target_dir'..."
    unzip -o "$temp_zip_file" -d "$target_dir"
    
    rm "$temp_zip_file"
}

# --- Download all required packages ---
download_and_unzip "tokenizers" "punkt" "1/5"
download_and_unzip "tokenizers" "punkt_tab" "2/5"
download_and_unzip "taggers" "averaged_perceptron_tagger" "3/5"
download_and_unzip "taggers" "averaged_perceptron_tagger_eng" "4/5" # <-- FIX: Add the missing package
download_and_unzip "corpora" "wordnet" "5/5"

echo "--- ✅ All necessary NLTK resources have been correctly installed into the .venv. ---"