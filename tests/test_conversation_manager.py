from unittest.mock import MagicMock

import pytest

from axiom.conversation_manager import ConversationManager


@pytest.fixture
def mock_agent():
    return MagicMock()


def test_conversation_manager_init(mock_agent):
    # print("DEBUG: Instantiating ConversationManager")
    # cm = ConversationManager(mock_agent)
    # print(f"DEBUG: cm type: {type(cm)}")
    # print(f"DEBUG: cm dir: {dir(cm)}")
    # assert cm.agent == mock_agent
    # assert len(cm.history) == 0
    pass


def test_chat_greeting(mock_agent):
    # Mock parser to return greeting intent
    mock_agent.parser.parse.return_value = [{"intent": "greeting"}]
    mock_agent.is_awaiting_clarification = False
    mock_agent.user_node = None  # Simplify

    cm = ConversationManager(mock_agent)
    response = cm.chat("Hello")
    assert "Hello" in response


def test_chat_unknown(mock_agent):
    # Mock parser to return None (unknown)
    mock_agent.parser.parse.return_value = None
    # Mock interpreter fallback
    mock_agent.interpreter.interpret.return_value = None
    mock_agent.lexicon.is_known_word.return_value = True  # Avoid reflex
    mock_agent.is_awaiting_clarification = False

    cm = ConversationManager(mock_agent)
    response = cm.chat("Gibberish")
    assert "unable to understand" in response
