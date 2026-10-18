"""Shared test fixtures. Nothing here needs an API key or a network connection."""

from __future__ import annotations

import pytest

from fakes import KeywordEmbedding, RecordingFakeLLM


@pytest.fixture
def fake_llm():
    """Build a RecordingFakeLLM that gives these answers in order."""

    def make(*answers: str) -> RecordingFakeLLM:
        return RecordingFakeLLM(responses=list(answers))

    return make


@pytest.fixture
def keyword_embedding() -> KeywordEmbedding:
    return KeywordEmbedding()


@pytest.fixture
def portfolio(tmp_path, keyword_embedding):
    from portfolio import Portfolio

    return Portfolio(vectorstore_path=tmp_path / "vectorstore", embedding_function=keyword_embedding)


@pytest.fixture
def no_api_key(monkeypatch):
    """Run as if no Groq key is set, even if the machine has one in its environment."""
    monkeypatch.setenv("GROQ_API_KEY", "")
