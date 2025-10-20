from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm

MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODELS_FOLDER = Path("models")
MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DESTINATION_PATH = MODELS_FOLDER / MODEL_NAME


def download_model_with_progress(url: str, destination: Path) -> None:
    """Downloads a file from a URL to a destination, showing a progress bar."""
    print("--- 🚀 Starting Axiom LLM Downloader ---")
    print(f"Source: {url}")
    print(f"Destination: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            block_size = 1024

            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            with open(destination, "wb") as f:
                for data in r.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("❌ ERROR: Download failed. File might be incomplete.")
                if destination.exists():
                    destination.unlink()
            else:
                print(f"\n✅ Model downloaded successfully to: {destination}")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ CRITICAL ERROR: Failed to download the model. Error: {e}")
        print("   Please check your internet connection or the URL.")
        if destination.exists():
            destination.unlink()


def main() -> None:
    """Entry point for the axiom-llm command."""
    if DESTINATION_PATH.exists():
        print(f"✅ Model already exists at: {DESTINATION_PATH}")
        user_input = input("Do you want to re-download it? (y/N): ").lower().strip()
        if user_input != "y":
            print("Skipping download.")
            return

    download_model_with_progress(MODEL_URL, DESTINATION_PATH)


if __name__ == "__main__":
    main()
