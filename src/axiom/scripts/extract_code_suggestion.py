"""
Utility script to extract code suggestions from 'code_suggestion.json'
into a clean, readable 'code.py' file for human review.
"""

import json
import sys
from pathlib import Path

# Configuration
INPUT_FILE = Path("code_suggestion.json")
OUTPUT_FILE = Path("code.py")


def main():
    # 1. Check if the input file exists
    if not INPUT_FILE.exists():
        print(f"❌ Error: Input file '{INPUT_FILE}' not found.")
        sys.exit(1)

    # 2. Read and parse the JSON
    try:
        print(f"🔍 Reading '{INPUT_FILE}'...")
        content = INPUT_FILE.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to parse JSON. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred while reading the file: {e}")
        sys.exit(1)

    # 3. Extract the code string
    try:
        # Navigate the JSON structure safely
        suggested_solution = data.get("suggested_solution", {})
        code_string = suggested_solution.get("code")

        if not code_string:
            print("❌ Error: Could not find any code in 'suggested_solution.code'.")
            sys.exit(1)

        # 4. Write to the output file
        OUTPUT_FILE.write_text(code_string, encoding="utf-8")

        print(f"✅ Success! Extracted code to '{OUTPUT_FILE}'.")
        print("   You can now review 'code.py' in your editor.")

    except Exception as e:
        print(f"❌ An error occurred during extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
