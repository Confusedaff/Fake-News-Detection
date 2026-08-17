from datetime import date

from packages.schemas import (
    ChunkVerification,
    ClassificationResult,
    NLILabel,
    RetrievalChunk,
    Verdict,
    VerificationResult,
)
from services.aggregation import Aggregator


def _chunk(chunk_id="c1", source="source-a", source_quality=0.9, score=0.9, pub_date=None):
    return RetrievalChunk(
        document_id=f"doc_{chunk_id}", chunk_id=chunk_id, text="evidence text",
        source=source, score=score, source_quality=source_quality, publication_date=pub_date,
    )


def test_no_evidence_returns_insufficient_evidence():
    agg = Aggregator()
    decision = agg.aggregate(classifier_result=None, retrieval_chunks=[], verification_result=None)
    assert decision.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert decision.heuristic_score == 0.0


def test_strong_uncontested_support_yields_verified_true():
    agg = Aggregator()
    chunks = [_chunk("c1", source="pub-a"), _chunk("c2", source="pub-b")]
    verification = VerificationResult(
        verifications=[
            ChunkVerification(chunk_id="c1", document_id="doc_c1", nli_label=NLILabel.SUPPORTS, nli_confidence=0.95),
            ChunkVerification(chunk_id="c2", document_id="doc_c2", nli_label=NLILabel.SUPPORTS, nli_confidence=0.9),
        ],
        verifier_version="test-v0",
    )
    decision = agg.aggregate(classifier_result=None, retrieval_chunks=chunks, verification_result=verification)
    assert decision.verdict in (Verdict.VERIFIED_TRUE, Verdict.LIKELY_TRUE)
    assert decision.heuristic_score > 0


def test_strong_uncontested_contradiction_yields_false_leaning_verdict():
    agg = Aggregator()
    chunks = [_chunk("c1", source="pub-a"), _chunk("c2", source="pub-b")]
    verification = VerificationResult(
        verifications=[
            ChunkVerification(chunk_id="c1", document_id="doc_c1", nli_label=NLILabel.CONTRADICTS, nli_confidence=0.95),
            ChunkVerification(chunk_id="c2", document_id="doc_c2", nli_label=NLILabel.CONTRADICTS, nli_confidence=0.9),
        ],
        verifier_version="test-v0",
    )
    decision = agg.aggregate(classifier_result=None, retrieval_chunks=chunks, verification_result=verification)
    assert decision.verdict in (Verdict.VERIFIED_FALSE, Verdict.LIKELY_FALSE)
    assert decision.heuristic_score < 0


def test_balanced_high_quality_evidence_yields_conflicting_evidence():
    agg = Aggregator()
    chunks = [
        _chunk("c1", source="pub-a", source_quality=0.95, score=0.95),
        _chunk("c2", source="pub-b", source_quality=0.95, score=0.95),
    ]
    verification = VerificationResult(
        verifications=[
            ChunkVerification(chunk_id="c1", document_id="doc_c1", nli_label=NLILabel.SUPPORTS, nli_confidence=0.95),
            ChunkVerification(chunk_id="c2", document_id="doc_c2", nli_label=NLILabel.CONTRADICTS, nli_confidence=0.95),
        ],
        verifier_version="test-v0",
    )
    decision = agg.aggregate(classifier_result=None, retrieval_chunks=chunks, verification_result=verification)
    assert decision.verdict == Verdict.CONFLICTING_EVIDENCE


def test_repeat_publisher_is_down_weighted_relative_to_independent_sources():
    agg = Aggregator()
    same_pub_chunks = [_chunk("c1", source="pub-a"), _chunk("c2", source="pub-a")]
    diff_pub_chunks = [_chunk("c1", source="pub-a"), _chunk("c2", source="pub-b")]
    verification = VerificationResult(
        verifications=[
            ChunkVerification(chunk_id="c1", document_id="doc_c1", nli_label=NLILabel.SUPPORTS, nli_confidence=0.9),
            ChunkVerification(chunk_id="c2", document_id="doc_c2", nli_label=NLILabel.SUPPORTS, nli_confidence=0.9),
        ],
        verifier_version="test-v0",
    )
    same_pub_decision = agg.aggregate(classifier_result=None, retrieval_chunks=same_pub_chunks, verification_result=verification)
    diff_pub_decision = agg.aggregate(classifier_result=None, retrieval_chunks=diff_pub_chunks, verification_result=verification)

    assert same_pub_decision.support_mass < diff_pub_decision.support_mass


def test_heuristic_score_and_calibrated_probability_are_never_conflated():
    agg = Aggregator()
    chunks = [_chunk("c1")]
    verification = VerificationResult(
        verifications=[
            ChunkVerification(chunk_id="c1", document_id="doc_c1", nli_label=NLILabel.SUPPORTS, nli_confidence=0.9),
        ],
        verifier_version="test-v0",
    )
    classifier = ClassificationResult(
        label="true", raw_confidence=0.8, calibrated_probability=0.7, model_version="test-v0",
    )
    decision = agg.aggregate(classifier_result=classifier, retrieval_chunks=chunks, verification_result=verification)
    # calibrated_probability field must remain None at MVP (Section 11: requires a separate trained model)
    assert decision.calibrated_probability is None
    assert isinstance(decision.heuristic_score, float)
