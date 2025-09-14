# Contributing to the Axiom Agent

Thank you for your interest in contributing to the Axiom Agent project! This is not just another chatbot; it's an exploration into a new kind of cognitive architecture. By contributing, you are helping to cultivate a mind and push the boundaries of what AI can be.

We welcome contributions of all kinds, from bug fixes and performance improvements to new features and documentation enhancements. This guide will walk you through the entire process of setting up your development environment for maximum stability.

## Code of Conduct

This project and everyone participating in it is governed by a standard Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior. (We can add a link to a formal Code of Conduct file later if the project grows).

## Setting Up Your Development Environment

The following steps will guide you through a complete setup, including the final, critical step of running the agent as a persistent, self-recovering service on macOS.

### Step 1: Clone the Repository

First, get the code onto your local machine.

```bash
git clone <your-repo-url>
cd Axiom-Agent
```

### Step 2: Set Up a Python Virtual Environment

It is crucial to use a virtual environment to keep the project's dependencies isolated.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 4: Download the LLM Model

The agent uses a local LLM as its "senses." You need to download the specific GGUF model file.

1.  Download the **`mistral-7b-instruct-v0.2.Q4_K_M.gguf`** model from Hugging Face.
    -   **Direct Link:** [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)
2.  Create a `models/` directory in the root of the project.
3.  Place the downloaded `.gguf` file inside the `models/` directory.

### Step 5: Configure Environment Variables (Optional but Recommended)

To unlock the agent's full autonomous capabilities, you need a free API key from the New York Times.

1.  Get a key from the [NYT Developer Portal](https://developer.nytimes.com/).
2.  Set it as an environment variable in your terminal.

```bash
export NYT_API_KEY="YOUR_API_KEY_HERE"
```
*For a permanent setup, add this line to your shell's configuration file (e.g., `~/.zshrc` or `~/.bash_profile`) and restart your terminal.*

---

## Final Setup: Running as a Self-Recovering Service (macOS)

This is the most important step for long-term development and stability. Running the agent as a service ensures it will automatically restart after any crash (like a segmentation fault) or a system reboot.

### Step 1: Find Your Absolute Paths

Services require full, absolute paths. Run these two commands in your terminal **while your `venv` is activated**:

1.  **Find your Python executable path:**
    ```bash
    which python3
    ```
    *(Copy the output, e.g., `/Users/yourname/Projects/Axiom-Agent/venv/bin/python3`)*

2.  **Find your project directory path:**
    ```bash
    pwd
    ```
    *(Copy the output, e.g., `/Users/yourname/Projects/Axiom-Agent`)*

### Step 2: Create the Service File

Create a new file at this exact location: `~/Library/LaunchAgents/com.axiomagent.app.plist`

Paste the following XML code into it. **You must replace the placeholder paths with the real paths you just copied.**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.axiomagent.app</string>
    
    <key>ProgramArguments</key>
    <array>
        <!-- IMPORTANT: REPLACE with the path from 'which python3' -->
        <string>/Users/yourname/Projects/Axiom-Agent/venv/bin/python3</string>
        
        <!-- IMPORTANT: REPLACE with the path to app.py -->
        <string>/Users/yourname/Projects/Axiom-Agent/app.py</string>
    </array>
    
    <key>WorkingDirectory</key>
    <!-- IMPORTANT: REPLACE with the path from 'pwd' -->
    <string>/Users/yourname/Projects/Axiom-Agent</string>
    
    <!-- This key is the magic: it tells macOS to restart the agent if it ever crashes -->
    <key>KeepAlive</key>
    <true/>
    
    <!-- Redirect all output and errors to log files in your project directory -->
    <key>StandardOutPath</key>
    <string>/Users/yourname/Projects/Axiom-Agent/axiom_agent_output.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/yourname/Projects/Axiom-Agent/axiom_agent_error.log</string>
</dict>
</plist>
```

### Step 3: Load and Start the Service

Run these commands in your terminal to activate the service:

```bash
# Load the service into the macOS launch system
launchctl load ~/Library/LaunchAgents/com.axiomagent.app.plist

# Start the agent for the first time
launchctl start com.axiomagent.app
```

The agent is now running as a persistent background process!

### Step 4: Monitor the Agent

You can now "watch" the agent's mind in real-time. Open two separate terminal windows and run these commands:

```bash
# In Terminal 1: Watch the main output log
tail -f axiom_agent_output.log

# In Terminal 2: Watch for any errors
tail -f axiom_agent_error.log
```

You can now interact with the agent at `http://127.0.0.1:7500` and it will run continuously, recovering from any potential crashes.

---

## How to Contribute

We are thrilled that you want to help build this new kind of mind. The best place to start is by looking at our official plan for the future.

1.  **Read the Roadmap:** Please review the **[ROADMAP.md](ROADMAP.md)** file. It contains a detailed, multi-phase plan for the agent's evolution, broken down by architectural flaws.
2.  **Pick a Task:** Find a task in an upcoming phase that interests you. The roadmap is structured to be modular, so you can tackle anything from improving LLM prompts to integrating new cloud services.
3.  **Fork and Branch:** Fork the repository, create a new feature branch for your work, and start coding!
4.  **Submit a Pull Request:** When your feature is complete, submit a pull request with a clear description of the changes you've made and why.

Thank you again for your interest. Together, we can build something truly extraordinary.
