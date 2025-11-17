import nltk


def main():
    """
    Downloads the necessary NLTK data packages in a non-interactive way,
    using the correct package IDs.
    """
    # --- FIX: Use the correct, full package IDs ---
    packages = {
        "punkt": "tokenizers/punkt",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "wordnet": "corpora/wordnet",  # Good to have for lemmatization
    }

    print("--- Verifying NLTK Resources ---")
    for pkg_id, path in packages.items():
        try:
            nltk.data.find(path)
            print(f" [✅] '{pkg_id}' is already downloaded.")
        except LookupError:
            print(f" [⏳] Downloading '{pkg_id}'...")
            # Use the package ID directly for the download
            nltk.download(pkg_id, quiet=True)
            print(f" [✅] '{pkg_id}' downloaded successfully.")

    print("\nAll necessary NLTK resources are available.")


if __name__ == "__main__":
    main()
