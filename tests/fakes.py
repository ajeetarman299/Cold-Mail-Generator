"""Offline stand-ins for the embedding model and the chat model, used by the tests."""

from __future__ import annotations

import hashlib
import re

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class KeywordEmbedding(EmbeddingFunction[Documents]):
    """A tiny, deterministic embedding: the sum of one fixed random vector per word.

    It stands in for ChromaDB's downloaded model, so the tests run offline and fast.
    Texts that share words get a small cosine distance; texts with no words in
    common get a distance close to 1.
    """

    def __init__(self, dimensions: int = 1024):
        self.dimensions = dimensions

    def _word_vector(self, word: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self.dimensions)

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vector = np.zeros(self.dimensions)
            for word in re.findall(r"[a-z0-9+#]+", text.lower()):
                vector += self._word_vector(word)
            norm = np.linalg.norm(vector)
            vectors.append((vector / norm if norm else vector).astype(np.float32))
        return vectors

    @staticmethod
    def name() -> str:
        return "keyword-hash-test"

    def get_config(self) -> dict:
        return {"dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config: dict) -> "KeywordEmbedding":
        return KeywordEmbedding(config["dimensions"])


class RecordingFakeLLM(FakeListChatModel):
    """LangChain's fake chat model, plus a record of every prompt it received."""

    prompts: list[str] = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append("\n\n".join(message.text for message in messages))
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)
