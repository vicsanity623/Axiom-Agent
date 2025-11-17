from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from axiom.scripts.visualize_seed_of_life import (
    seed_of_life_positions,
    visualize_seed_of_life,
)


def test_seed_of_life_positions() -> None:
    """
    Given: A specific number of nodes.
    When: The seed_of_life_positions function is called.
    Then: It should return a dictionary of positions with the correct length and structure.
    """
    # Arrange
    total_nodes = 7  # Center + first layer of 6

    # Act
    positions = seed_of_life_positions(total_nodes)

    # Assert
    assert isinstance(positions, dict)
    assert len(positions) == total_nodes
    assert 0 in positions
    assert positions[0] == {"x": 0, "y": 0}  # Center node is always at origin

    for i in range(1, total_nodes):
        assert i in positions
        assert "x" in positions[i]
        assert "y" in positions[i]
        assert isinstance(positions[i]["x"], int)
        assert isinstance(positions[i]["y"], int)


def test_visualize_seed_of_life_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: A valid mock brain file.
    When: The visualize_seed_of_life function is called.
    Then: It should create an HTML visualization file with the correct content.
    """
    # --- GIVEN ---
    # 1. Create a fake brain file in a temporary directory
    fake_brain_file = tmp_path / "mock_brain.json"
    mock_brain_data: dict[str, Any] = {
        "nodes": [
            {"id": "n1", "name": "Node 1"},
            {"id": "n2", "name": "Node 2"},
            {"id": "n3", "name": "Node 3"},
        ],
        "links": [
            {"source": "n1", "target": "n2", "type": "connects_to"},
            {"source": "n2", "target": "n3", "type": "is_related"},
        ],
    }
    fake_brain_file.write_text(json.dumps(mock_brain_data))

    # 2. Patch the script's DEFAULT_BRAIN_FILE to point to our fake file
    monkeypatch.setattr(
        "axiom.scripts.visualize_seed_of_life.DEFAULT_BRAIN_FILE", fake_brain_file
    )

    # 3. Change the current working directory to the temporary path to contain output
    monkeypatch.chdir(tmp_path)

    # --- WHEN ---
    visualize_seed_of_life()

    # --- THEN ---
    # 1. Check that the output file was created in the correct location
    output_file = tmp_path / "visualizations" / "axiom_seed_of_life.html"
    assert output_file.exists()

    # 2. Check that the content of the file is correct
    html_content = output_file.read_text()
    assert "<title>Axiom Seed of Life (Dense)</title>" in html_content
    assert "cytoscape.min.js" in html_content

    # 3. Verify that our specific node and edge data is embedded in the HTML
    # --- FIX: Use double quotes to match the JSON output format ---
    assert '"id": "n1", "label": "Node 1"' in html_content
    assert '"id": "n3", "label": "Node 3"' in html_content
    assert '"source": "n1", "target": "n2", "label": "connects_to"' in html_content


def test_visualize_brain_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Given: A path to a brain file that does not exist.
    When: The visualize_seed_of_life function is called.
    Then: It should raise a FileNotFoundError.
    """
    # --- GIVEN ---
    non_existent_file = Path("/tmp/this/file/does/not/exist.json")
    monkeypatch.setattr(
        "axiom.scripts.visualize_seed_of_life.DEFAULT_BRAIN_FILE", non_existent_file
    )

    # --- WHEN / THEN ---
    with pytest.raises(FileNotFoundError, match="Brain file not found"):
        visualize_seed_of_life()


def test_visualize_no_nodes_in_brain_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: A brain file that is valid JSON but contains no nodes.
    When: The visualize_seed_of_life function is called.
    Then: It should raise a ValueError.
    """
    # --- GIVEN ---
    empty_brain_file = tmp_path / "empty_brain.json"
    empty_brain_data: dict[str, Any] = {"nodes": [], "links": []}
    empty_brain_file.write_text(json.dumps(empty_brain_data))

    monkeypatch.setattr(
        "axiom.scripts.visualize_seed_of_life.DEFAULT_BRAIN_FILE", empty_brain_file
    )

    # --- WHEN / THEN ---
    with pytest.raises(ValueError, match="No nodes found in brain file"):
        visualize_seed_of_life()
