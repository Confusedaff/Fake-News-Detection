"""
Retrieval Module (Section 4, 10).

Responsibility: execute search against the vector (and later BM25) index;
return scored chunks + metadata.

Does NOT judge factuality — only relevance. Whether a chunk is "good enough"
is the orchestrator's retrieval-quality-check policy (THRESHOLDS.RETRIEVAL_
RELEVANCE_FLOOR), not this module's concern.

MVP index: sentence-transformers/all-MiniLM-L6-v2 -> FAISS IndexFlatIP
(Section 6/10) — exact brute-force search, fine at a few-thousand-chunk
corpus. Swapping to a managed vector DB later means implementing a new
Retriever subclass; the orchestrator never changes.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Optional

from packages.schemas import RetrievalChunk, RetrievalResult


class Retriever(ABC):
    """Interface every retrieval implementation must satisfy."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int, attempt_number: int = 1) -> RetrievalResult:
        raise NotImplementedError


class MockRetriever(Retriever):
    """
    Deterministic fixture retriever for local dev/tests and Day 1-3
    orchestrator integration (Section 24: "small fixture corpus of 20-30
    hand-picked chunks ready by Day 2 so integration isn't blocked").

    Synthesizes plausible-looking chunks from a tiny in-memory fixture corpus
    keyed by keyword overlap with the query — good enough to exercise the
    router's retry/relevance-floor logic without a real index.
    """

    _FIXTURE_CORPUS = [
        {
            "document_id": "doc_100",
            "source": "cityrecords.gov",
            "source_quality": 0.95,
            "text": "The city council voted 6-3 against the proposed plastic bag ban "
                    "at the March session, citing cost concerns for small retailers.",
            "publication_date": date(2026, 3, 12),
        },
        {
            "document_id": "doc_101",
            "source": "localnews.example.com",
            "source_quality": 0.7,
            "text": "Council members are expected to revisit the plastic bag "
                    "ordinance next year after further public comment.",
            "publication_date": date(2026, 3, 14),
        },
        {
            "document_id": "doc_102",
            "source": "encyclopedia.example.org",
            "source_quality": 0.9,
            "text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "publication_date": date(2020, 1, 1),
        },
    ]

    def __init__(self, corpus_version: str = "mock-corpus-v0",
                 embedding_model_version: str = "mock-embed-v0",
                 seed: int = 42):
        self.corpus_version = corpus_version
        self.embedding_model_version = embedding_model_version
        self._rng = random.Random(seed)

    def retrieve(self, query: str, top_k: int, attempt_number: int = 1) -> RetrievalResult:
        query_terms = set(query.lower().split())
        scored = []
        for doc in self._FIXTURE_CORPUS:
            doc_terms = set(doc["text"].lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap == 0:
                continue
            # Deterministic pseudo-score from overlap + a stable hash jitter.
            jitter = (int(hashlib.sha256((query + doc["document_id"]).encode()).hexdigest()[:4], 16) % 100) / 1000.0
            score = min(0.99, 0.5 + 0.12 * overlap + jitter)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [
            RetrievalChunk(
                document_id=doc["document_id"],
                chunk_id=f"{doc['document_id']}_c1",
                text=doc["text"],
                source=doc["source"],
                score=round(score, 4),
                source_quality=doc["source_quality"],
                publication_date=doc["publication_date"],
            )
            for score, doc in scored[:top_k]
        ]

        return RetrievalResult(
            query_text=query,
            attempt_number=attempt_number,
            chunks=chunks,
            embedding_model_version=self.embedding_model_version,
            corpus_version=self.corpus_version,
        )


class FaissRetriever(Retriever):
    """
    Real MVP retriever (Section 6/10): sentence-transformer embeddings ->
    FAISS IndexFlatIP (exact inner-product search).

    Lazy-imports sentence_transformers/faiss so importing this module never
    requires those deps unless this class is instantiated.

    `metadata` must be a list, index-aligned with the FAISS index's internal
    ordering, of dicts with keys: document_id, chunk_id, text, source,
    source_quality, publication_date (optional).
    """

    def __init__(
        self,
        index_path: str,
        metadata: list[dict],
        corpus_version: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "FaissRetriever requires `faiss-cpu` and `sentence-transformers`. "
                "Install them or use MockRetriever for development."
            ) from e

        self._faiss = faiss
        self.index = faiss.read_index(index_path)
        self.metadata = metadata
        self.corpus_version = corpus_version
        self.embedding_model_version = embedding_model_name
        self.embedder = SentenceTransformer(embedding_model_name)

    def retrieve(self, query: str, top_k: int, attempt_number: int = 1) -> RetrievalResult:
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_vec, top_k)

        chunks: list[RetrievalChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            chunks.append(
                RetrievalChunk(
                    document_id=meta["document_id"],
                    chunk_id=meta["chunk_id"],
                    text=meta["text"],
                    source=meta["source"],
                    score=float(score),
                    source_quality=float(meta.get("source_quality", 0.5)),
                    publication_date=meta.get("publication_date"),
                )
            )

        return RetrievalResult(
            query_text=query,
            attempt_number=attempt_number,
            chunks=chunks,
            embedding_model_version=self.embedding_model_version,
            corpus_version=self.corpus_version,
        )
