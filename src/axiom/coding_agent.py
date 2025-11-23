from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import PROJECT_ROOT

if TYPE_CHECKING:
    from .universal_interpreter import UniversalInterpreter

logger = logging.getLogger(__name__)


class CodingAgent:
    """
    A specialized agent component for handling coding tasks.
    It uses the UniversalInterpreter (Mistral) to generate code and perform file operations.
    """

    def __init__(self, interpreter: UniversalInterpreter):
        self.interpreter = interpreter
        self.working_dir = PROJECT_ROOT
        self.chat_history: list[dict[str, str]] = []

    def clear_history(self):
        """Clears the conversation history."""
        self.chat_history = []

    def process_request(self, user_input: str, context: dict | None = None) -> str:
        """
        Processes a coding request from the user.
        Parses the input for tool calls or generates code.
        """
        # Simple tool parsing for now. In a real system, we'd use the LLM to decide tools.
        # Format: /code [tool] [args]

        parts = user_input.strip().split(maxsplit=2)
        # parts[0] is '/code'

        if len(parts) < 2:
            return self._generate_code_response(user_input, context)

        action = parts[1].lower()
        args = parts[2] if len(parts) > 2 else ""

        if action == "read":
            return self._tool_read_file(args)
        if action == "write":
            # Expect args format: "path content..."
            # This is a naive parser; for robustness we'd use a structured format
            write_parts = args.split(maxsplit=1)
            if len(write_parts) == 2:
                return self._tool_write_file(write_parts[0], write_parts[1])
            return "Usage: /code write <path> <content>"
        if action == "list":
            return self._tool_list_dir(args)
        # Treat as a general coding question
        return self._generate_code_response(
            user_input.replace("/code", "").strip(), context
        )

    def _tool_read_file(self, path_str: str) -> str:
        try:
            path = (self.working_dir / path_str).resolve()
            if not path.is_relative_to(self.working_dir):
                return f"Error: Access denied to {path}"
            if not path.exists():
                return f"Error: File not found: {path}"

            content = path.read_text(encoding="utf-8")
            return f"### File: {path.name}\n```python\n{content}\n```"  # Defaulting to python highlighting
        except Exception as e:
            return f"Error reading file: {e}"

    def _tool_write_file(self, path_str: str, content: str) -> str:
        try:
            path = (self.working_dir / path_str).resolve()
            if not path.is_relative_to(self.working_dir):
                return f"Error: Access denied to {path}"

            path.parent.mkdir(parents=True, exist_ok=True)

            # Clean up content (remove wrapping backticks if present)
            content = content.strip()
            if content.startswith("```") and content.endswith("```"):
                lines = content.splitlines()
                if len(lines) >= 2:
                    content = "\n".join(lines[1:-1])

            path.write_text(content, encoding="utf-8")
            return f"Successfully wrote to `{path.name}`."
        except Exception as e:
            return f"Error writing file: {e}"

    def _tool_list_dir(self, path_str: str) -> str:
        try:
            path = (
                (self.working_dir / path_str).resolve()
                if path_str
                else self.working_dir
            )
            if not path.is_relative_to(self.working_dir):
                return f"Error: Access denied to {path}"

            if not path.exists():
                return f"Error: Directory not found: {path}"

            items = [p.name + ("/" if p.is_dir() else "") for p in path.iterdir()]
            return f"### Directory: {path.name}\n" + "\n".join(sorted(items))
        except Exception as e:
            return f"Error listing directory: {e}"

    def _generate_code_response(self, prompt: str, context: dict | None = None) -> str:
        """Uses the LLM to answer a coding question with conversation history."""
        if not self.interpreter.llm:
            return "Error: LLM is not loaded."

        # Construct System Prompt
        system_prompt = (
            "You are an expert coding assistant named Axiom (Gemma 3). "
            "Your task is to help the user write, debug, and understand code. "
            "Provide clear, concise explanations and correct code snippets."
        )

        # Inject Context (Active File)
        context_str = ""
        if context and "active_file" in context:
            f = context["active_file"]
            context_str = (
                f"\n\n[CONTEXT]\n"
                f"The user is currently looking at this file:\n"
                f"File: {f['name']}\n"
                f"```\n{f['content']}\n```\n"
                f"[/CONTEXT]\n"
            )

        # Build Prompt with History (Gemma Format)
        # <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>
        full_prompt = f"<start_of_turn>user\n{system_prompt}{context_str}\n\n"

        # Add History
        for turn in self.chat_history:
            full_prompt += f"{turn['user']}<end_of_turn>\n<start_of_turn>model\n{turn['model']}<end_of_turn>\n<start_of_turn>user\n"

        # Add Current User Input
        full_prompt += f"{prompt}<end_of_turn>\n<start_of_turn>model\n"

        try:
            response = self.interpreter._run_llm_completion_with_retries(
                full_prompt,
                max_tokens=4096,
                stop=["<end_of_turn>"],  # Allow full response generation
            )

            # Update History
            self.chat_history.append({"user": prompt, "model": response})

            # Keep history manageable (last 10 turns)
            if len(self.chat_history) > 10:
                self.chat_history.pop(0)

            return response
        except Exception as e:
            return f"Error generating response: {e}"
