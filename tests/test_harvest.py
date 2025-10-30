from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, Any

import axiom.knowledge_harvester as kh_mod
from axiom.knowledge_harvester import KnowledgeHarvester

if TYPE_CHECKING:
    from pathlib import Path

    from axiom.cognitive_agent import CognitiveAgent


def test_mark_and_load_research_cache(
    tmp_path: Path, agent: CognitiveAgent, monkeypatch: Any
) -> None:
    """_mark_as_researched writes cache; _load_research_cache reads it back."""
    lock: threading.Lock = threading.Lock()
    h: KnowledgeHarvester = KnowledgeHarvester(agent, lock)

    cache_file: Path = tmp_path / "research_cache.json"
    h.cache_path = cache_file  # __slots__ safe

    h._mark_as_researched("pytest-term")
    assert cache_file.exists()

    with cache_file.open("r", encoding="utf-8") as f:
        data: list[str] = json.load(f)
    assert "pytest-term" in data

    h2: KnowledgeHarvester = KnowledgeHarvester(agent, lock)
    h2.cache_path = cache_file
    h2.researched_terms = set()
    h2._load_research_cache()
    assert "pytest-term" in h2.researched_terms


def test_discover_cycle_appends_goal(agent: CognitiveAgent, monkeypatch: Any) -> None:
    """When _find_new_topic returns a topic, it adds an INVESTIGATE goal."""
    lock: threading.Lock = threading.Lock()
    h: KnowledgeHarvester = KnowledgeHarvester(agent, lock)

    monkeypatch.setattr(
        KnowledgeHarvester, "_find_new_topic", lambda self: "Quantum mechanics"
    )
    agent.learning_goals.clear()
    h.discover_cycle()

    assert any("INVESTIGATE" in goal for goal in agent.learning_goals)


def test_get_definition_from_api_variants(
    agent: CognitiveAgent, monkeypatch: Any
) -> None:
    """Test dictionary API with 404, malformed, and valid responses."""
    lock: threading.Lock = threading.Lock()
    h: KnowledgeHarvester = KnowledgeHarvester(agent, lock)

    class FakeResp:
        status_code: int
        _payload: Any

        def __init__(self, status: int = 200, payload: Any = None) -> None:
            self.status_code = status
            self._payload = payload

        def json(self) -> Any:
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    # 404 -> None
    monkeypatch.setattr(
        kh_mod.requests, "get", lambda url, timeout=5: FakeResp(status=404, payload={})
    )
    assert h.get_definition_from_api("nothing") is None

    # malformed -> None
    monkeypatch.setattr(
        kh_mod.requests,
        "get",
        lambda url, timeout=5: FakeResp(status=200, payload={"bad": "data"}),
    )
    assert h.get_definition_from_api("weird") is None

    # valid
    good_payload: list[dict[str, Any]] = [
        {
            "word": "test",
            "meanings": [
                {
                    "partOfSpeech": "noun",
                    "definitions": [{"definition": "An instance used for testing."}],
                }
            ],
        }
    ]
    monkeypatch.setattr(
        kh_mod.requests,
        "get",
        lambda url, timeout=5: FakeResp(status=200, payload=good_payload),
    )
    assert h.get_definition_from_api("test") == (
        "noun",
        "An instance used for testing.",
    )


def test_get_search_result_count_and_exceptions(
    agent: CognitiveAgent, monkeypatch: Any
) -> None:
    """Parses int from results; returns None on exception."""
    lock: threading.Lock = threading.Lock()
    h: KnowledgeHarvester = KnowledgeHarvester(agent, lock)

    class FakeResp:
        text: str

        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    # Successful integer parsing
    monkeypatch.setattr(
        kh_mod.requests,
        "get",
        lambda url, headers, timeout=5: FakeResp("About 12,345 results"),
    )
    assert h._get_search_result_count("query") == 12345

    # Exception -> None
    monkeypatch.setattr(
        kh_mod.requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert h._get_search_result_count("query") is None
