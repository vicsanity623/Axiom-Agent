from unittest.mock import MagicMock

import pytest

from axiom.symbolic_parser import SymbolicParser


@pytest.fixture
def mock_agent():
    return MagicMock()


def test_parser_init(mock_agent):
    # spacy is mocked globally in conftest.py
    parser = SymbolicParser(mock_agent)
    assert parser.nlp is not None


def test_parse_command(mock_agent):
    parser = SymbolicParser(mock_agent)
    result = parser.parse("show all facts")
    assert result is not None
    assert result[0]["intent"] == "command_show_all_facts"


def test_parse_meta_question(mock_agent):
    parser = SymbolicParser(mock_agent)
    result = parser.parse("who are you")
    assert result is not None
    assert result[0]["intent"] == "meta_question_self"
