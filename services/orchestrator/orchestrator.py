"""
Orchestrator (Section 3, 4, 7, 7A).

"A pipeline with a traffic cop, not an autonomous agent that improvises"
(Section 1). A deterministic, bounded state machine. It decides WHETHER and
HOW MUCH verification to run; it never itself classifies, retrieves, or
verifies — it only calls modules through their frozen contracts.

States (Section 7):
RECEIVED -> CLASSIFIED -> NEEDS_VERIFICATION -> [IMAGE_ANALYZED in parallel]
  -> QUERY_GENERATED -> RETRIEVED -> RETRIEVAL_EVALUATED -> VERIFIED
  -> [FUSED if image present] -> AGGREGATED -> FINALIZED
  (+ terminal failure states: INSUFFICIENT_EVIDENCE, TIMED_OUT, ERROR)

Bounds enforced HERE, not hoped for (Section 7):
  MAX_RETRIEVAL_ATTEMPTS = 3, MAX_EVIDENCE_CHUNKS = 10, MAX_LLM_CALLS = 2,
  soft/hard latency budgets.

Every transition is appended to `self.events` as an OrchestrationEvent —
this list IS the decision trail surfaced to the UI (Section 3: "a projection
of this event log, not a separate narrative generation step, which is why
it can never hallucinate steps that didn't happen").

Image branch (Section 7A): when an image is attached, ImageAnalyzer.analyze
runs "concurrently" with retrieval+verification. This MVP implementation
runs it via a thread pool alongside the synchronous text pipeline so a slow
or failed image call never blocks a text-only verdict from finalizing late
— matching the doc's "it pays the max, not the sum" latency goal without
requiring the whole service to be rewritten async for a hackathon timeline.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from typing import Optional
from uuid import UUID, uuid4

from packages.config import BOUNDS, THRESHOLDS
from packages.schemas import (
    ClaimResult,
    ClassificationResult,
    DecisionResult,
    FusedResult,
    ImageVerificationResult,
    OrchestrationEvent,
    OrchestratorState,
    RetrievalResult,
    Verdict,
    VerificationResult,
)
from services.aggregation import Aggregator
from services.classifier.classifier import Classifier
from services.fusion import Fusion
from services.image.image_analyzer import ImageAnalyzer, ImageValidationError
from services.retrieval.query_generator import QueryGenerator
from services.retrieval.retriever import Retriever
from services.verification.verifier import Verifier


class OrchestratorTimeout(Exception):
    """Raised internally when the hard latency budget is exceeded."""


class Orchestrator:
    def __init__(
        self,
        classifier: Classifier,
        retriever: Retriever,
        query_generator: QueryGenerator,
        verifier: Verifier,
        image_analyzer: Optional[ImageAnalyzer],
        fusion: Fusion,
        aggregator: Aggregator,
        bounds=BOUNDS,
        thresholds=THRESHOLDS,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.query_generator = query_generator
        self.verifier = verifier
        self.image_analyzer = image_analyzer
        self.fusion = fusion
        self.aggregator = aggregator
        self.bounds = bounds
        self.thresholds = thresholds
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="image-branch")

    # ----------------------------------------------------------------
    # Public entrypoint
    # ----------------------------------------------------------------

    def run(
        self,
        claim_text: str,
        request_id: str,
        image_bytes: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        force_verification: bool = False,
        claim_id: Optional[UUID] = None,
    ) -> ClaimResult:
        claim_id = claim_id or uuid4()
        self.events: list[OrchestrationEvent] = []
        self._llm_calls_used = 0
        start = time.monotonic()
        state: Optional[OrchestratorState] = None

        state = self._transition(claim_id, state, OrchestratorState.RECEIVED, "Claim received")

        image_future: Optional[Future] = None
        if image_bytes is not None:
            # Section 7A: kick off the image branch on a worker thread so it
            # runs concurrently with steps 5-7 below, rather than blocking
            # the text pipeline's latency behind it.
            image_future = self._executor.submit(
                self._run_image_branch, image_bytes, image_mime_type, claim_text
            )

        try:
            # --- Step 3: Classify ---
            classifier_result = self._safe_classify(claim_text)
            state = self._transition(
                claim_id, state, OrchestratorState.CLASSIFIED,
                f"Classifier returned label={classifier_result.label if classifier_result else 'N/A'}",
            )
            self._check_latency(start)

            # --- Step 4: Router decision ---
            needs_verification, router_reason = self._route(
                classifier_result, force_verification, image_bytes is not None
            )
            state = self._transition(
                claim_id, state, OrchestratorState.NEEDS_VERIFICATION, router_reason,
            )

            retrieval_result: Optional[RetrievalResult] = None
            verification_result: Optional[VerificationResult] = None

            if needs_verification:
                retrieval_result, state = self._retrieve_with_retries(claim_id, state, claim_text, start)

                if retrieval_result is None:
                    # Retrieval exhausted attempts with nothing usable. Only
                    # a genuine terminal case if the image branch also has
                    # nothing to offer (Section 3, step 6) — checked below
                    # once the image branch has been awaited.
                    pass
                else:
                    verification_result = self.verifier.verify(claim_text, retrieval_result.chunks)
                    state = self._transition(
                        claim_id, state, OrchestratorState.VERIFIED,
                        f"Verified {len(verification_result.verifications)} chunk(s)",
                    )
                    self._check_latency(start)

            # --- Await image branch (Section 7A: "pays the max, not the sum") ---
            image_result: Optional[ImageVerificationResult] = None
            image_note: Optional[str] = None
            if image_future is not None:
                remaining = max(0.1, self.bounds.MAX_LATENCY_HARD_S - (time.monotonic() - start))
                image_result, image_note = self._await_image_branch(image_future, remaining)
                if image_result is not None:
                    state = self._transition(
                        claim_id, state, OrchestratorState.IMAGE_ANALYZED,
                        "Image analysis complete" if image_result.reverse_search_available
                        else "Image analysis complete (reverse search unavailable)",
                    )
                elif image_note:
                    state = self._transition(claim_id, state, state, image_note)

            # --- Terminal check: truly nothing usable from either branch ---
            if needs_verification and retrieval_result is None and image_result is None:
                return self._finalize_insufficient_evidence(
                    claim_id, state, request_id, classifier_result, start,
                )

            # --- Step 7a: Fusion (only when an image result exists) ---
            fused_result: Optional[FusedResult] = None
            if image_result is not None:
                fused_result = self.fusion.reconcile(
                    verification_result, image_result, claim_reference_date=date.today()
                )
                state = self._transition(
                    claim_id, state, OrchestratorState.FUSED,
                    fused_result.note or "Image and text signals reconciled",
                )

            # --- Step 8: Aggregate ---
            decision = self.aggregator.aggregate(
                classifier_result=classifier_result,
                retrieval_chunks=retrieval_result.chunks if retrieval_result else [],
                verification_result=verification_result,
                fused_result=fused_result,
            )
            state = self._transition(
                claim_id, state, OrchestratorState.AGGREGATED,
                f"Verdict computed: {decision.verdict.value} (heuristic_score={decision.heuristic_score})",
            )

            state = self._transition(claim_id, state, OrchestratorState.FINALIZED, "Claim finalized")

            return self._build_result(
                claim_id, request_id, state, decision, classifier_result,
                image_result, fused_result, retrieval_result, start,
            )

        except OrchestratorTimeout:
            state = self._transition(
                claim_id, state, OrchestratorState.TIMED_OUT,
                f"Exceeded hard latency budget of {self.bounds.MAX_LATENCY_HARD_S}s",
            )
            return self._build_error_result(claim_id, request_id, state, start)

        except Exception as e:  # noqa: BLE001 - top-level guard, never crash the API on a pipeline error
            state = self._transition(claim_id, state, OrchestratorState.ERROR, f"Unhandled error: {e}")
            return self._build_error_result(claim_id, request_id, state, start)

    # ----------------------------------------------------------------
    # Step implementations
    # ----------------------------------------------------------------

    def _safe_classify(self, claim_text: str) -> Optional[ClassificationResult]:
        """Section 15: classifier unavailable -> skip classification, proceed
        retrieval-only, never crash the whole claim over it."""
        try:
            return self.classifier.predict(claim_text)
        except Exception:
            return None

    def _route(
        self, classifier_result: Optional[ClassificationResult], force_verification: bool,
        has_image: bool,
    ) -> tuple[bool, str]:
        """
        Step 4 router. High confidence + no image -> skip verification
        (fast path). Otherwise verify. Image attachment always forces
        verification-adjacent processing because image findings must factor
        into the verdict even on an otherwise high-confidence text path
        (Section 3, step 4).
        """
        if force_verification:
            return True, "Verification forced by request options"

        if classifier_result is None:
            return True, "Classifier unavailable — proceeding to evidence-based verification"

        if has_image:
            return True, (
                f"Image attached — verification path engaged regardless of classifier "
                f"confidence {classifier_result.calibrated_probability}"
            )

        if classifier_result.calibrated_probability >= self.thresholds.CLASSIFIER_CONFIDENCE_THRESHOLD:
            return False, (
                f"Classifier confidence {classifier_result.calibrated_probability} met "
                f"threshold {self.thresholds.CLASSIFIER_CONFIDENCE_THRESHOLD} — fast path"
            )

        return True, (
            f"Classifier confidence {classifier_result.calibrated_probability} below "
            f"verification threshold {self.thresholds.CLASSIFIER_CONFIDENCE_THRESHOLD}"
        )

    def _retrieve_with_retries(
        self, claim_id: UUID, state: OrchestratorState, claim_text: str, start: float,
    ) -> tuple[Optional[RetrievalResult], OrchestratorState]:
        """Steps 5-6: retrieval + relevance-floor retry loop, bounded by
        MAX_RETRIEVAL_ATTEMPTS and MAX_LLM_CALLS for reformulation."""
        query = self.query_generator.initial_query(claim_text)

        for attempt in range(1, self.bounds.MAX_RETRIEVAL_ATTEMPTS + 1):
            self._check_latency(start)

            result = self.retriever.retrieve(
                query, top_k=self.bounds.MAX_EVIDENCE_CHUNKS, attempt_number=attempt
            )
            state = self._transition(
                claim_id, state, OrchestratorState.RETRIEVED,
                f"Retrieval attempt {attempt}: top score {result.top_score:.2f}, query='{query}'",
            )

            if result.top_score >= self.thresholds.RETRIEVAL_RELEVANCE_FLOOR:
                state = self._transition(
                    claim_id, state, OrchestratorState.RETRIEVAL_EVALUATED,
                    f"Retrieval attempt {attempt}: top score {result.top_score:.2f} exceeded "
                    f"relevance floor {self.thresholds.RETRIEVAL_RELEVANCE_FLOOR}",
                )
                return result, state

            # Weak retrieval — reformulate and retry if attempts remain.
            if attempt < self.bounds.MAX_RETRIEVAL_ATTEMPTS and self._llm_calls_used < self.bounds.MAX_LLM_CALLS:
                self._llm_calls_used += 1
                query = self.query_generator.reformulate(claim_text, query, attempt + 1)
                state = self._transition(
                    claim_id, state, state,
                    f"Retrieval attempt {attempt} below relevance floor "
                    f"({result.top_score:.2f} < {self.thresholds.RETRIEVAL_RELEVANCE_FLOOR}) — reformulating",
                )

        state = self._transition(
            claim_id, state, state,
            f"Retrieval exhausted {self.bounds.MAX_RETRIEVAL_ATTEMPTS} attempts with no chunk "
            f"above relevance floor {self.thresholds.RETRIEVAL_RELEVANCE_FLOOR}",
        )
        return None, state

    def _run_image_branch(
        self, image_bytes: bytes, image_mime_type: Optional[str], claim_text: str,
    ) -> tuple[Optional[ImageVerificationResult], Optional[str]]:
        """Runs on a worker thread (Section 7A). Returns (result, note).
        note is populated on graceful-degradation paths (invalid image,
        analyzer unavailable) so the caller can still log a decision-trail
        entry even though there's no ImageVerificationResult."""
        if self.image_analyzer is None:
            return None, "Image module unavailable — proceeding as text-only claim"
        try:
            result = self.image_analyzer.analyze(image_bytes, image_mime_type or "image/jpeg", claim_text)
            return result, None
        except ImageValidationError as e:
            # Section 15: reject the image, proceed with text-only pipeline
            # rather than rejecting the whole claim.
            return None, f"Image could not be processed ({e}) — proceeding with text-only verification"
        except Exception as e:  # noqa: BLE001
            return None, f"Image analysis failed unexpectedly ({e}) — proceeding with text-only verification"

    def _await_image_branch(
        self, future: Future, timeout_s: float,
    ) -> tuple[Optional[ImageVerificationResult], Optional[str]]:
        try:
            return future.result(timeout=timeout_s)
        except Exception:
            # Timed out or the thread itself raised unexpectedly — never let
            # a slow/broken image branch block or fail the text verdict.
            return None, "Image analysis timed out or failed — proceeding with text-only verification"

    # ----------------------------------------------------------------
    # Finalization helpers
    # ----------------------------------------------------------------

    def _finalize_insufficient_evidence(
        self, claim_id: UUID, state: OrchestratorState, request_id: str,
        classifier_result: Optional[ClassificationResult], start: float,
    ) -> ClaimResult:
        state = self._transition(
            claim_id, state, OrchestratorState.INSUFFICIENT_EVIDENCE,
            "No usable evidence from text retrieval or image analysis",
        )
        decision = DecisionResult(
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            heuristic_score=0.0,
            calibrated_probability=None,
            support_mass=0.0,
            contradict_mass=0.0,
            supporting_evidence=[],
            contradicting_evidence=[],
        )
        return self._build_result(
            claim_id, request_id, state, decision, classifier_result,
            None, None, None, start,
        )

    def _build_result(
        self, claim_id: UUID, request_id: str, state: OrchestratorState,
        decision: DecisionResult, classifier_result: Optional[ClassificationResult],
        image_result: Optional[ImageVerificationResult], fused_result: Optional[FusedResult],
        retrieval_result: Optional[RetrievalResult], start: float,
    ) -> ClaimResult:
        return ClaimResult(
            claim_id=claim_id,
            request_id=request_id,
            status=state,
            verdict=decision.verdict,
            decision=decision,
            classifier=classifier_result,
            image_analysis=image_result,
            fusion=fused_result,
            decision_trail=[e.reason for e in self.events],
            corpus_version=retrieval_result.corpus_version if retrieval_result else None,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    def _build_error_result(
        self, claim_id: UUID, request_id: str, state: OrchestratorState, start: float,
    ) -> ClaimResult:
        return ClaimResult(
            claim_id=claim_id,
            request_id=request_id,
            status=state,
            verdict=None,
            decision_trail=[e.reason for e in self.events],
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    # ----------------------------------------------------------------
    # Bounds / bookkeeping
    # ----------------------------------------------------------------

    def _check_latency(self, start: float) -> None:
        elapsed = time.monotonic() - start
        if elapsed > self.bounds.MAX_LATENCY_HARD_S:
            raise OrchestratorTimeout(f"Elapsed {elapsed:.2f}s exceeded hard budget")

    def _transition(
        self, claim_id: UUID, from_state: Optional[OrchestratorState],
        to_state: OrchestratorState, reason: str, metadata: Optional[dict] = None,
    ) -> OrchestratorState:
        """Appends a row to the append-only decision trail (Section 3/9).
        This is the ONLY place events are recorded — the UI's step-by-step
        trace is a direct projection of this list, so it can never diverge
        from what actually happened."""
        event = OrchestrationEvent(
            claim_id=claim_id,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            metadata=metadata or {},
        )
        self.events.append(event)
        return to_state

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
