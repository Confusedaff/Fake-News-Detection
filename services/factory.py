"""
Wiring factory.

This is what Sub-team B (Section 23/24) uses on Day 1-3 to run the
orchestrator end-to-end against mocked module implementations, before
Sub-team A's real models exist — and it's exactly the same factory pattern
used in production, just swapping which classes get instantiated.

`build_mock_orchestrator()` has zero ML dependencies (no torch, no faiss,
no transformers) — it only needs `pydantic`, so it's what CI and local dev
should use by default.

`build_production_orchestrator()` shows the wiring for the real MVP
implementations described in Section 6; swap in real credentials/paths when
those are available.
"""

from __future__ import annotations

from services.aggregation import Aggregator
from services.classifier.classifier import DistilBertClassifier, MockClassifier
from services.fusion import Fusion
from services.image.image_analyzer import (
    ClipConsistencyScorer,
    ELAForgeryDetector,
    GoogleVisionReverseSearch,
    MockImageAnalyzer,
    PretrainedImageAnalyzer,
)
from services.orchestrator.orchestrator import Orchestrator
from services.retrieval.query_generator import HeuristicQueryGenerator
from services.retrieval.retriever import FaissRetriever, MockRetriever
from services.verification.verifier import LocalNLIVerifier, MockVerifier


def build_mock_orchestrator(simulate_reverse_search_down: bool = False) -> Orchestrator:
    """Dependency-free orchestrator for local dev, unit tests, and the
    Day 1-3 integration milestone (Section 24)."""
    return Orchestrator(
        classifier=MockClassifier(),
        retriever=MockRetriever(),
        query_generator=HeuristicQueryGenerator(),
        verifier=MockVerifier(),
        image_analyzer=MockImageAnalyzer(simulate_reverse_search_down=simulate_reverse_search_down),
        fusion=Fusion(),
        aggregator=Aggregator(),
    )


def build_production_orchestrator(
    classifier_model_path: str,
    classifier_version: str,
    faiss_index_path: str,
    corpus_metadata: list[dict],
    corpus_version: str,
    reverse_search_api_key: str,
    http_post_fn,
) -> Orchestrator:
    """
    Real MVP wiring (Section 6). Requires torch, transformers, faiss-cpu,
    sentence-transformers, and Pillow to be installed — see requirements.txt.
    """
    return Orchestrator(
        classifier=DistilBertClassifier(
            model_path=classifier_model_path, version=classifier_version,
        ),
        retriever=FaissRetriever(
            index_path=faiss_index_path, metadata=corpus_metadata, corpus_version=corpus_version,
        ),
        query_generator=HeuristicQueryGenerator(),  # swap for LLMQueryGenerator once wired to an LLM call
        verifier=LocalNLIVerifier(),
        image_analyzer=PretrainedImageAnalyzer(
            manipulation_detector=ELAForgeryDetector(),
            reverse_search_client=GoogleVisionReverseSearch(
                api_key=reverse_search_api_key, http_post_fn=http_post_fn,
            ),
            consistency_scorer=ClipConsistencyScorer(),
        ),
        fusion=Fusion(),
        aggregator=Aggregator(),
    )
