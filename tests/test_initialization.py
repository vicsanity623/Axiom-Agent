# tests/test_initialization.py

# Add these lines to fix the import path

from axiom.cognitive_agent import CognitiveAgent


def test_agent_initialization():
    """
    Tests if the CognitiveAgent can be created in inference mode without a brain file.
    This is a basic "smoke test" to ensure the core components load.
    """
    # We use inference_mode and pass dummy data to prevent file I/O in a test
    agent = CognitiveAgent(
        load_from_file=False,
        brain_data={"nodes": [], "links": []},
        cache_data={"interpretations": [], "synthesis": []},
        inference_mode=True,
    )
    # The test passes if the agent is created successfully
    assert agent is not None
    assert agent.inference_mode is True
