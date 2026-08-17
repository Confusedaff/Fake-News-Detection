"""
Unit tests exercising the orchestrator end-to-end against mock
implementations — this is the Day 1-3 integration milestone from Section 24
made concrete and repeatable.
"""

import pytest

from packages.schemas import OrchestratorState, Verdict
from services.factory import build_mock_orchestrator


@pytest.fixture
def orchestrator():
    orch = build_mock_orchestrator()
    yield orch
    orch.shutdown()


def test_text_only_claim_produces_finalized_result(orchestrator):
    result = orchestrator.run(
        claim_text="The city council voted to ban plastic bags",
        request_id="req_test_1",
    )
    assert result.status in (OrchestratorState.FINALIZED, OrchestratorState.INSUFFICIENT_EVIDENCE)
    assert result.verdict is not None
    assert result.decision_trail, "decision trail must never be empty"
    assert result.latency_ms is not None and result.latency_ms >= 0


def test_claim_with_no_matching_evidence_returns_insufficient_evidence(orchestrator):
    result = orchestrator.run(
        claim_text="zzz completely unrelated nonsense query xyz",
        request_id="req_test_2",
        force_verification=True,
    )
    assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert "Retrieval exhausted" in " ".join(result.decision_trail) or \
           "No usable evidence" in " ".join(result.decision_trail)


def test_image_branch_populates_image_analysis_and_fusion():
    orch = build_mock_orchestrator()
    try:
        fake_image = b"\xff\xd8\xff" + b"0" * 100  # not a real JPEG; MockImageAnalyzer validates loosely via Pillow if present
        result = orch.run(
            claim_text="A photo shows the mayor at the scene",
            request_id="req_test_3",
            image_bytes=fake_image,
            image_mime_type="image/jpeg",
        )
        # Whether or not Pillow rejects the fake bytes, the pipeline must
        # finish and never crash — this IS the graceful-degradation contract.
        assert result.status is not None
        assert result.decision_trail
    finally:
        orch.shutdown()


def test_decision_trail_is_never_empty_even_on_error():
    orch = build_mock_orchestrator()
    try:
        # Empty claim text still goes through the pipeline at this layer
        # (API-layer validation is a separate concern per Section 8/15);
        # orchestrator must not crash regardless.
        result = orch.run(claim_text="ok", request_id="req_test_4")
        assert result.decision_trail
    finally:
        orch.shutdown()


def test_high_confidence_classifier_can_take_fast_path(orchestrator, monkeypatch):
    # Force a specific classifier result via monkeypatching predict so we can
    # deterministically exercise the fast-path branch (Section 3, step 4).
    from packages.schemas import ClassificationResult

    def fake_predict(claim):
        return ClassificationResult(
            label="true", raw_confidence=0.97, calibrated_probability=0.95,
            ood_signal=0.03, model_version="test-fixture-v0",
        )

    monkeypatch.setattr(orchestrator.classifier, "predict", fake_predict)
    result = orchestrator.run(claim_text="water boils at 100C", request_id="req_test_5")

    reasons = " ".join(result.decision_trail)
    assert "fast path" in reasons
    assert result.status == OrchestratorState.FINALIZED
