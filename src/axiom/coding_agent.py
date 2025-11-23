from __future__ import annotations

import json
import logging
import re
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
        self.stop_generation = False  # Flag for emergency stop
        self.pending_edit: dict | None = None  # Store proposed edit

    def cancel_generation(self):
        """Cancels the current generation."""
        self.stop_generation = True
        logger.info("[CodingAgent]: Generation cancelled by user")

    def clear_history(self):
        """Clears the conversation history."""
        self.chat_history = []

    def process_request(
        self, user_input: str, context: dict | None = None
    ) -> str | dict:
        """
        Processes a coding request from the user.
        Returns either a string response or a dict for structured edits.
        """
        # Check if this is an edit request with an active file
        if (
            context
            and "active_file" in context
            and self._detect_code_edit_intent(user_input, context)
        ):
            edit_result = self._generate_code_edit(user_input, context)
            if edit_result:
                return edit_result  # Return structured dict

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

    def _detect_code_edit_intent(
        self, prompt: str, context: dict | None = None
    ) -> bool:
        """
        Detects if the user is requesting a file modification.
        Made very conservative to avoid performance issues.
        """
        # Only trigger on VERY explicit edit commands to avoid slow LLM generation
        # Users can still get code suggestions via regular streaming responses
        explicit_edit_patterns = [
            r"^(please\s+)?apply\s+",  # "apply this change"
            r"^(please\s+)?write\s+this\s+to\s+",  # "write this to file"
            r"^(please\s+)?save\s+this\s+",  # "save this change"
        ]

        prompt_lower = prompt.lower().strip()
        for pattern in explicit_edit_patterns:
            if re.search(pattern, prompt_lower):
                return True

        # Otherwise, use streaming for everything else (including "add", "fix", etc.)
        return False

    def _generate_code_edit(
        self, prompt: str, context: dict | None = None
    ) -> dict | None:
        """
        Generates a structured code edit using the LLM.
        Returns a dict with type, file, original, modified, and explanation.
        """
        if not self.interpreter.llm or not context or "active_file" not in context:
            return None

        active_file = context["active_file"]
        file_name = active_file["name"]
        original_content = active_file["content"]

        # Build a specialized prompt for code editing
        edit_prompt = f"""You are a code editor assistant. The user is viewing this file:

File: {file_name}
```
{original_content}
```

User Request: {prompt}

Your task:
1. Generate the COMPLETE modified file content that addresses the user's request
2. Preserve all code that doesn't need to change
3. Only modify the necessary parts
4. Maintain the original formatting and style

Respond ONLY with a JSON object in this exact format:
{{
  "explanation": "Brief explanation of what you changed and why",
  "modified_content": "THE COMPLETE FILE CONTENT WITH YOUR CHANGES"
}}

Do not include any other text before or after the JSON."""

        try:
            response = self.interpreter._run_llm_completion_with_retries(
                f"<start_of_turn>user\n{edit_prompt}<end_of_turn>\n<start_of_turn>model\n",
                max_tokens=8192,
                stop=["<end_of_turn>"],
            )

            # Parse the JSON response
            # Try to extract JSON from the response (in case there's extra text)
            json_match = re.search(
                r'\{[^{}]*"explanation"[^{}]*"modified_content"[^{}]*\}',
                response,
                re.DOTALL,
            )
            if not json_match:
                logger.warning(
                    f"Failed to extract JSON from LLM response: {response[:200]}"
                )
                return None

            parsed = json.loads(json_match.group())
            modified_content = parsed.get("modified_content", "").strip()
            explanation = parsed.get("explanation", "Code modification")

            # Check if the content actually changed
            if modified_content == original_content.strip():
                # No changes detected, return a text response instead
                return None

            # Return structured edit
            return {
                "type": "edit",
                "file": file_name,
                "original": original_content,
                "modified": modified_content,
                "explanation": explanation,
            }

        except Exception as e:
            logger.error(f"Error generating code edit: {e}")
            return None

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

        # Reset stop flag at start of new request
        self.stop_generation = False

        try:
            # Check if cancelled before starting LLM
            if self.stop_generation:
                return "[Response cancelled]"

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

    def _generate_code_response_streaming(
        self, prompt: str, context: dict | None = None
    ):
        """Streaming version of _generate_code_response using generator."""
        if not self.interpreter.llm:
            yield 'data: {"token": "LLM is not available."}\n\n'
            yield 'data: {"done": true}\n\n'
            return

        # Build Context
        context_summary = ""
        if context and "active_file" in context:
            file_name = context["active_file"]["name"]
            file_content = context["active_file"]["content"]
            context_summary = f"Active File: {file_name}\n```\n{file_content}\n```\n\n"

        # Build Conversation History
        history_summary = ""
        for turn in self.chat_history[-3:]:  # Last 3 turns
            history_summary += f"User: {turn['user']}\nAssistant: {turn['model']}\n\n"

        # Build Prompt
        full_prompt = f"""<start_of_turn>user
{context_summary}{history_summary}User: {prompt}
<end_of_turn>
<start_of_turn>model
"""

        # Stream response
        accumulated_response = ""
        try:
            for sse_chunk in self.interpreter._run_llm_completion_streaming(
                full_prompt,
                max_tokens=4096,
                stop=["<end_of_turn>"],
            ):
                # Extract token from SSE format
                if sse_chunk.startswith("data: "):
                    import json

                    try:
                        data = json.loads(sse_chunk[6:])  # Remove "data: " prefix
                        if "token" in data:
                            accumulated_response += data["token"]
                        yield sse_chunk  # Forward SSE chunk to client
                    except json.JSONDecodeError:
                        continue

            # Update history with full response
            self.chat_history.append({"user": prompt, "model": accumulated_response})
            if len(self.chat_history) > 10:
                self.chat_history.pop(0)

        except Exception as e:
            import json

            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield 'data: {"done": true}\n\n'
