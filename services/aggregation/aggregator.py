"""
Aggregator (Section 4, 7A, 11).

Responsibility: apply the weighted scoring policy across classifier +
evidence (fused, if an image was present); distinguish heuristic score from
calibrated probability; decide verdict taxonomy incl. INSUFFICIENT_EVIDENCE
and CONFLICTING_EVIDENCE.

Does NOT call any models itself — every input here is already-computed
(ClassificationResult, RetrievalResult, VerificationResult, and optionally
FusedResult). This invariant is why Fusion exists as a separate module
(Section 7A) rather than folding image reconciliation in here.

Core rule (Section 11): "Do not average unlike quantities." Classifier
probability, retrieval score, and NLI confidence are not commensurable —
this module NEVER averages them into one blended number. Instead:
  1. per-chunk weight = source_quality * retrieval_relevance * nli_confidence
                         * recency_factor(publication_date)
  2. support_mass / contradict_mass = sum of weights per NLI label, with an
     independent-source bonus (diminishing returns for repeat publishers)
  3. classifier folded in as one more weighted voter (higher weight only
     when evidence is weak/absent)
  4. image_weight folded into the SAME support_mass/contradict_mass
     accumulation as one more weighted vote (Section 7A) — not a separate
     number that then gets averaged in.

heuristic_score is documented and tunable; calibrated_probability is a
SEPARATE, optional field (Section 11: requires a downstream calibration
model, out of scope for MVP) — they are never conflated.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from packages.config import WEIGHTS
from packages.schemas import (
    ChunkVerification,
    ClassificationResult,
    DecisionResult,
    EvidenceItem,
    FusedResult,
    ImageVerificationResult,
    NLILabel,
    RetrievalChunk,
    Verdict,
    VerificationResult,
)


def _recency_factor(pub_date: Optional[date], reference: Optional[date] = None,
                     half_life_days: int = 365) -> float:
    """
    Decays for stale sources on time-sensitive claims; ~1.0 for evergreen
    facts (Section 11: "a config per topic, not a universal constant" — this
    is the MVP default single-topic version; per-topic half-lives are a
    Should-Build item).
    """
    if pub_date is None:
        return 1.0
    reference = reference or date.today()
    age_days = max(0, (reference - pub_date).days)
    # Simple exponential decay with a floor so old-but-relevant sources
    # still count for something rather than vanishing entirely.
    decay = 0.5 ** (age_days / half_life_days)
    return max(0.3, decay)


class Aggregator:
    def aggregate(
        self,
        classifier_result: Optional[ClassificationResult],
        retrieval_chunks: list[RetrievalChunk],
        verification_result: Optional[VerificationResult],
        fused_result: Optional[FusedResult] = None,
        reference_date: Optional[date] = None,
    ) -> DecisionResult:
        chunk_by_id = {c.chunk_id: c for c in retrieval_chunks}
        verifications = verification_result.verifications if verification_result else []

        # -------------------------------------------------------------
        # Step 1 + 2: per-chunk weight, accumulated into support/contradict
        #             mass, with an independent-source bonus.
        # -------------------------------------------------------------
        support_mass = 0.0
        contradict_mass = 0.0
        supporting_evidence: list[EvidenceItem] = []
        contradicting_evidence: list[EvidenceItem] = []
        seen_publishers: dict[str, int] = {}

        for v in verifications:
            chunk = chunk_by_id.get(v.chunk_id)
            if chunk is None:
                continue  # defensive: verifier returned a chunk_id we don't have metadata for

            weight = self._chunk_weight(chunk, v, reference_date, seen_publishers)

            item = EvidenceItem(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                source_quality=chunk.source_quality,
                retrieval_score=chunk.score,
                nli_label=v.nli_label,
                nli_confidence=v.nli_confidence,
            )

            if v.nli_label == NLILabel.SUPPORTS:
                support_mass += weight
                supporting_evidence.append(item)
            elif v.nli_label == NLILabel.CONTRADICTS:
                contradict_mass += weight
                contradicting_evidence.append(item)
            # UNRELATED contributes no mass either way.

        # -------------------------------------------------------------
        # Step 4 (Section 7A): image_weight folded into the SAME
        # support/contradict accumulation as one more weighted vote — not
        # a separate score that gets averaged in later.
        # -------------------------------------------------------------
        if fused_result is not None and fused_result.image_verification is not None:
            image_support, image_contradict = self._image_mass(fused_result.image_verification)
            support_mass += image_support
            contradict_mass += image_contradict

        had_evidence = bool(verifications) or (
            fused_result is not None and fused_result.image_verification is not None
        )

        # -------------------------------------------------------------
        # No evidence at all (retrieval exhausted attempts, no image, or
        # image alone produced nothing usable): INSUFFICIENT_EVIDENCE is a
        # first-class result, never forced onto the True/False spectrum
        # (Section 11).
        # -------------------------------------------------------------
        if not had_evidence:
            return DecisionResult(
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                heuristic_score=0.0,
                calibrated_probability=None,
                support_mass=0.0,
                contradict_mass=0.0,
                supporting_evidence=[],
                contradicting_evidence=[],
            )

        # -------------------------------------------------------------
        # Step 3: classifier folded in as one more weighted voter. Weight is
        # higher when evidence is weak/absent, lower when evidence is
        # strong and specific — never the tie-breaker by default.
        # -------------------------------------------------------------
        evidence_strength = support_mass + contradict_mass
        classifier_weight = (
            WEIGHTS.CLASSIFIER_WEIGHT_WHEN_NO_EVIDENCE
            if evidence_strength < 0.3
            else WEIGHTS.CLASSIFIER_WEIGHT_DEFAULT
        )

        if classifier_result is not None:
            # Map the classifier's label to a directional signal.
            # LIAR-style labels: "true"/"mostly-true"/"half-true" lean
            # support; "mostly-false"/"false"/"pants-fire" lean contradict.
            classifier_leans_true = classifier_result.label in (
                "true", "mostly-true", "half-true"
            )
            classifier_vote = classifier_result.calibrated_probability * classifier_weight
            if classifier_leans_true:
                support_mass += classifier_vote
            else:
                contradict_mass += classifier_vote

        heuristic_score = self._normalize(support_mass, contradict_mass)
        verdict = self._decide_verdict(support_mass, contradict_mass, heuristic_score)

        return DecisionResult(
            verdict=verdict,
            heuristic_score=round(heuristic_score, 4),
            calibrated_probability=None,  # Section 11: requires a trained calibration model; not MVP.
            support_mass=round(support_mass, 4),
            contradict_mass=round(contradict_mass, 4),
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_weight(
        self,
        chunk: RetrievalChunk,
        verification: ChunkVerification,
        reference_date: Optional[date],
        seen_publishers: dict[str, int],
    ) -> float:
        recency = _recency_factor(chunk.publication_date, reference_date)
        base_weight = chunk.source_quality * chunk.score * verification.nli_confidence * recency

        # Independent-source bonus: down-weight repeat chunks from the same
        # publisher (diminishing returns) relative to independent sources.
        occurrence = seen_publishers.get(chunk.source, 0)
        seen_publishers[chunk.source] = occurrence + 1
        decay_multiplier = WEIGHTS.SAME_PUBLISHER_DECAY ** occurrence

        return base_weight * decay_multiplier

    def _image_mass(self, image_result: ImageVerificationResult) -> tuple[float, float]:
        """
        image_weight = manipulation_confidence_inverse * reverse_search_corroboration
                       * consistency_score   (Section 7A)

        A reverse-search hit is treated as one source, same as a single
        document (Section 7A) — hence corroboration is derived from whether
        matches exist, not their count, avoiding double-counting many near-
        duplicate matches from the same event.

        Returns (support_contribution, contradict_contribution). Low
        consistency + old reverse-search matches push toward contradiction
        (image doesn't match the claimed context); clean, consistent,
        unmatched-elsewhere images push toward support.
        """
        manipulation_confidence_inverse = 1.0 - image_result.manipulation_confidence

        if not image_result.reverse_search_available:
            # Missing signal reported as missing (Section 15) — contributes
            # no mass rather than being guessed at.
            reverse_search_corroboration = 0.0
        elif image_result.reverse_search_matches:
            reverse_search_corroboration = max(
                m.context_similarity for m in image_result.reverse_search_matches
            )
        else:
            # No matches found at all is itself weak positive signal (image
            # doesn't appear to be recycled from elsewhere) but far from
            # strong corroboration.
            reverse_search_corroboration = 0.3

        consistency = image_result.caption_image_consistency
        weight = manipulation_confidence_inverse * max(reverse_search_corroboration, 0.15) * consistency

        # Direction: manipulation detected or low consistency or a stale
        # reverse-search match all push toward CONTRADICT; otherwise SUPPORT.
        stale_match = any(
            m.first_seen_date is not None for m in image_result.reverse_search_matches
        ) and image_result.reverse_search_available

        contradicts = image_result.manipulation_detected or consistency < 0.35 or stale_match

        if contradicts:
            return 0.0, weight
        return weight, 0.0

    def _normalize(self, support_mass: float, contradict_mass: float) -> float:
        """Normalize the mass differential into a documented, bounded
        [-1, 1] heuristic_score — never claimed as a calibrated probability."""
        total = support_mass + contradict_mass
        if total == 0:
            return 0.0
        return (support_mass - contradict_mass) / total

    def _decide_verdict(self, support_mass: float, contradict_mass: float, heuristic_score: float) -> Verdict:
        both_strong = (
            support_mass >= WEIGHTS.CONFLICT_MASS_FLOOR
            and contradict_mass >= WEIGHTS.CONFLICT_MASS_FLOOR
        )
        roughly_balanced = abs(heuristic_score) <= WEIGHTS.CONFLICT_BALANCE_EPSILON

        if both_strong and roughly_balanced:
            # Distinct from UNCERTAIN: the issue is disagreement between
            # high-quality sources, not weak/absent evidence (Section 11).
            return Verdict.CONFLICTING_EVIDENCE

        if heuristic_score >= WEIGHTS.STRONG_MARGIN:
            return Verdict.VERIFIED_TRUE
        if heuristic_score >= WEIGHTS.MODERATE_MARGIN:
            return Verdict.LIKELY_TRUE
        if heuristic_score <= -WEIGHTS.STRONG_MARGIN:
            return Verdict.VERIFIED_FALSE
        if heuristic_score <= -WEIGHTS.MODERATE_MARGIN:
            return Verdict.LIKELY_FALSE

        return Verdict.UNCERTAIN
