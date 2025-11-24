from unittest.mock import MagicMock

import pytest

from axiom.coding_agent import CodingAgent


@pytest.fixture
def mock_interpreter():
    return MagicMock()


def test_coding_agent_init(mock_interpreter):
    agent = CodingAgent(mock_interpreter)
    assert agent.interpreter == mock_interpreter
    assert len(agent.chat_history) == 0


def test_extract_code_blocks(mock_interpreter):
    agent = CodingAgent(mock_interpreter)
    # text = "Here is code:\n```python\nprint('hi')\n```"
    # Placeholder for future test
    pass
    agent = CodingAgent(mock_interpreter)
    # Mock the LLM call
    mock_interpreter._run_llm_completion_with_retries.return_value = (
        "```python\nreturn True\n```"
    )
    mock_interpreter.llm = True  # Ensure check passes

    response = agent.process_request("Write a function that returns True")
    assert "return True" in response
