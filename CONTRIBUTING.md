# Contributing to the Axiom Agent

Thank you for your interest in contributing to the Axiom Agent project! This is not just another chatbot; it's an exploration into a new kind of cognitive architecture. By contributing, you are helping to cultivate a mind and push the boundaries of what AI can be.

We welcome contributions of all kinds, from bug fixes and performance improvements to new features and documentation enhancements.

## Code of Conduct

This project and everyone participating in it is governed by a standard Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior.

## Development Environment Setup

The following steps will guide you through setting up a complete and stable development environment using modern Python standards.

### Step 1: Clone the Repository & Setup Environment
First, get the code, create an isolated virtual environment, and activate it.
```bash
git clone <your-repo-url>
cd <your-repo-folder>
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install All Dependencies
This project uses a `pyproject.toml` file to manage all dependencies. This single command installs the core application libraries *and* all the development tools (like Ruff, MyPy, and Pytest) needed for contributing.
```bash
# The quotes are important to prevent errors in some shells like Zsh
pip install -e '.[dev]'
```

### Step 3: Download the LLM Model
The agent uses a local LLM as its "senses." You need to download the specific GGUF model file.
1.  Download the **`mistral-7b-instruct-v0.2.Q4_K_M.gguf`** model from Hugging Face.
    -   **Direct Link:** [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)
2.  Create a `models/` directory in the root of the project.
3.  Place the downloaded `.gguf` file inside the `models/` directory.

### Step 4: Configure Environment Variables (Optional)
To enable the agent's autonomous discovery capabilities, you can provide a free API key from the New York Times.
1.  Get a key from the [NYT Developer Portal](https://developer.nytimes.com/).
2.  Set it as an environment variable in your terminal.
```bash
export NYT_API_KEY="YOUR_API_KEY_HERE"
```
*For a permanent setup, add this line to your shell's configuration file (e.g., `~/.zshrc` or `~/.bash_profile`) and restart your terminal.*

---

## The Core Development Workflow

The project is structured around a professional **Check -> Code/Train -> Check -> Render -> Test** cycle.

### 1. Run Quality Checks (The First and Last Step)
**Before you start coding and before you commit, always run the master check script.** This ensures your environment is working and your code adheres to project standards.
```bash
./check.sh
```
If the script reports formatting or linting issues, you can often fix them automatically with these commands:
```bash
# Auto-format all code
ruff format .

# Auto-fix all simple linting issues
ruff check . --fix
```
Run `./check.sh` again to confirm everything passes.

### 2. Train and Develop (Offline)
This is where you'll do your work. Modify the agent's brain, which is stored in the `brain/` directory.
-   **For long-running autonomous tests:** `python setup/autonomous_trainer.py`
-   **For direct, interactive fact-checking:** `python setup/cnt.py`

### 3. Render a Model Snapshot
After training, package the brain into a stable `.axm` model. This creates a versioned snapshot in the `rendered/` folder.
```bash
python setup/render_model.py
```

### 4. Test the Deployed Agent
Run the read-only inference server to test the final user experience with the model you just created.
```bash
python setup/app_model.py
```
You can now open your browser to `http://127.0.0.1:7501` to interact with the deployed agent.

---

## How to Contribute

We are thrilled that you want to help build this new kind of mind.

1.  **Read the Roadmap:** Please review the **[ROADMAP.md](ROADMAP.md)** file. It contains a detailed, multi-phase plan for the agent's evolution.
2.  **Pick a Task:** Find a task in an upcoming phase that interests you.
3.  **Fork and Branch:** Fork the repository and create a new feature branch for your work (`git checkout -b feature/my-new-feature`).
4.  **Run the Checks:** As you code, frequently run `./check.sh` to maintain code quality. **All checks must pass before a pull request will be reviewed.**
5.  **Submit a Pull Request:** When your feature is complete and all checks are passing, submit a pull request with a clear description of the changes you've made and why.

Thank you again for your interest. Together, we can build something truly extraordinary.
