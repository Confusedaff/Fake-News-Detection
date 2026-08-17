from datetime import date, timedelta

from packages.schemas import ImageVerificationResult, ReverseSearchMatch
from services.fusion import Fusion


def test_fusion_with_no_image_result_is_defensive():
    fusion = Fusion()
    result = fusion.reconcile(text_result=None, image_result=None)
    assert result.image_verification is None
    assert result.caption_image_consistency is None


def test_fusion_flags_stale_image_and_low_consistency():
    fusion = Fusion()
    old_match = ReverseSearchMatch(
        url="example.com/story", first_seen_date=date.today() - timedelta(days=400),
        context_similarity=0.9,
    )
    image_result = ImageVerificationResult(
        manipulation_detected=False,
        manipulation_confidence=0.05,
        reverse_search_matches=[old_match],
        reverse_search_available=True,
        earliest_known_date=old_match.first_seen_date,
        caption_image_consistency=0.2,
        forensics_model_version="test-v0",
        consistency_model_version="test-v0",
    )
    fused = fusion.reconcile(text_result=None, image_result=image_result)
    assert fused.caption_image_consistency == 0.2
    assert fused.note is not None
    assert "predates" in fused.note
    assert "low" in fused.note.lower()


def test_fusion_reports_unavailable_reverse_search_without_fabricating():
    fusion = Fusion()
    image_result = ImageVerificationResult(
        manipulation_detected=False,
        manipulation_confidence=0.05,
        reverse_search_matches=[],
        reverse_search_available=False,
        earliest_known_date=None,
        caption_image_consistency=0.8,
        forensics_model_version="test-v0",
        consistency_model_version="test-v0",
    )
    fused = fusion.reconcile(text_result=None, image_result=image_result)
    assert "unavailable" in (fused.note or "").lower()
