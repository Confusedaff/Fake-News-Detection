"""
Fusion Module (new, Section 4/7A).

Responsibility: reconcile the text VerificationResult and
ImageVerificationResult into a FusedResult, keeping the two signals
distinguishable rather than blended.

Does NOT call any models itself — it only combines already-computed scores,
same invariant the Aggregator holds (Section 7A explicitly calls out that
Fusion exists as its own module precisely so the Aggregator's
"does not call models" rule is never quietly broken). Does NOT run when no
image was submitted — the orchestrator simply skips this step entirely in
that case (Section 3, step 7a is conditional on step 4a having produced a
result).

The one thing Fusion computes that's "new" rather than passed through is a
short human-readable `note` — e.g. "image predates the claimed event by over
a year" — which is a template over already-computed fields (date deltas,
consistency score), not a model call.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from packages.schemas import FusedResult, ImageVerificationResult, VerificationResult


class Fusion:
    """Fusion.reconcile(text_result, image_result) -> FusedResult."""

    # Below this consistency score, the claimed context is treated as
    # meaningfully mismatched with the image content (Section 7A example
    # uses 0.22 as "low"). Kept as a named constant rather than a magic
    # number so it's easy for the team to tune during eval (Day 5-6).
    LOW_CONSISTENCY_THRESHOLD = 0.35
    STALE_IMAGE_DAYS = 180  # ~6 months; older matches read as "predates the event"

    def reconcile(
        self,
        text_result: Optional[VerificationResult],
        image_result: Optional[ImageVerificationResult],
        claim_reference_date: Optional[date] = None,
    ) -> FusedResult:
        if image_result is None:
            # Nothing to fuse; caller shouldn't invoke this without an image
            # result, but stay defensive rather than raising, since a
            # missing signal should be reported as missing (Section 15),
            # not blow up the pipeline.
            return FusedResult(text_verification=text_result, image_verification=None,
                                caption_image_consistency=None, note=None)

        note = self._build_note(image_result, claim_reference_date)

        return FusedResult(
            text_verification=text_result,
            image_verification=image_result,
            caption_image_consistency=image_result.caption_image_consistency,
            note=note,
        )

    def _build_note(
        self, image_result: ImageVerificationResult, claim_reference_date: Optional[date]
    ) -> Optional[str]:
        notes: list[str] = []

        if image_result.manipulation_detected:
            notes.append(
                f"Image shows signs of manipulation (confidence "
                f"{image_result.manipulation_confidence:.2f})"
            )

        if not image_result.reverse_search_available:
            notes.append("Reverse image search was unavailable for this submission")
        elif image_result.earliest_known_date is not None:
            reference = claim_reference_date or date.today()
            age_days = (reference - image_result.earliest_known_date).days
            if age_days > self.STALE_IMAGE_DAYS:
                years = age_days / 365.0
                notes.append(
                    f"Image predates the claimed context by approximately "
                    f"{years:.1f} year(s) (earliest known: {image_result.earliest_known_date})"
                )

        if image_result.caption_image_consistency < self.LOW_CONSISTENCY_THRESHOLD:
            notes.append(
                f"Caption-image consistency is low "
                f"({image_result.caption_image_consistency:.2f}) — image content may "
                f"not match the claimed context"
            )

        return "; ".join(notes) if notes else None
