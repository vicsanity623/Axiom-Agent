from axiom.universal_interpreter import UniversalInterpreter


def test_interpreter_initialization():
    # Initialize without loading LLM for speed
    interpreter = UniversalInterpreter(load_llm=False)
    assert interpreter.llm is None


def test_parse_simple_sentence():
    # interpreter = UniversalInterpreter(load_llm=False)

    # Test methods that don't require LLM if any
    # decompose_sentence_to_relations uses LLM, so we'd need to mock it
    pass


def test_extract_properties():
    # Assuming there's a method for this
    pass
