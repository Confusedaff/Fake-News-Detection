"""
Shared contracts for the Fake-News Verifier system.

This module is the single source of truth for every inter-module data shape
described in the architecture doc (Section 8 / Section 22). Every service
(classifier, retrieval, verification, image, fusion, aggregation,
orchestrator) imports these models instead of defining its own — this is
what prevents drift between "the docs say X has field Y" and what the code
actually returns.

Rules encoded here directly from the architecture doc:
- Every score-bearing result carries a `model_version` (or equivalent) so
  predictions are reproducible (Section 5, versioning strategy).
- Heuristic scores and calibrated probabilities are NEVER the same field
  (Section 11) — they are computed and reported separately.
- INSUFFICIENT_EVIDENCE / CONFLICTING_EVIDENCE are first-class verdicts,
  not error states (Section 11).
- FusedResult keeps text and image signals distinguishable, never blended
  into one number (Section 7A).
"""

from __future__ import annotations

from datetime import datetime, date, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class NLILabel(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    UNRELATED = "UNRELATED"


class Verdict(str, Enum):
    VERIFIED_TRUE = "VERIFIED_TRUE"
    LIKELY_TRUE = "LIKELY_TRUE"
    UNCERTAIN = "UNCERTAIN"
    LIKELY_FALSE = "LIKELY_FALSE"
    VERIFIED_FALSE = "VERIFIED_FALSE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class OrchestratorState(str, Enum):
    RECEIVED = "RECEIVED"
    CLASSIFIED = "CLASSIFIED"
    NEEDS_VERIFICATION = "NEEDS_VERIFICATION"
    IMAGE_ANALYZED = "IMAGE_ANALYZED"
    QUERY_GENERATED = "QUERY_GENERATED"
    RETRIEVED = "RETRIEVED"
    RETRIEVAL_EVALUATED = "RETRIEVAL_EVALUATED"
    VERIFIED = "VERIFIED"
    FUSED = "FUSED"
    AGGREGATED = "AGGREGATED"
    FINALIZED = "FINALIZED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    TIMED_OUT = "TIMED_OUT"
    ERROR = "ERROR"


# --------------------------------------------------------------------------
# Classifier Module contract
# --------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """Output of Classifier.predict(claim). Section 6."""
    model_config = ConfigDict(frozen=True)

    label: str
    raw_confidence: float = Field(..., ge=0.0, le=1.0)
    calibrated_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Temperature/isotonic-calibrated. The orchestrator's "
                    "router threshold operates on THIS field, never raw_confidence.",
    )
    ood_signal: Optional[float] = Field(
        default=None,
        description="Proxy for out-of-distribution-ness (e.g. 1 - max softmax, "
                    "or embedding distance to train distribution).",
    )
    model_version: str


# --------------------------------------------------------------------------
# Retrieval Module contract
# --------------------------------------------------------------------------

class RetrievalChunk(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    source: str
    score: float = Field(..., description="Raw similarity/relevance score from the index.")
    source_quality: float = Field(..., ge=0.0, le=1.0)
    publication_date: Optional[date] = None


class RetrievalResult(BaseModel):
    """Output of Retriever.retrieve(query, top_k). Section 10."""
    model_config = ConfigDict(frozen=True)

    query_text: str
    attempt_number: int = Field(..., ge=1, le=3)
    chunks: list[RetrievalChunk] = Field(default_factory=list)
    embedding_model_version: str
    corpus_version: str

    @property
    def top_score(self) -> float:
        return max((c.score for c in self.chunks), default=0.0)


# --------------------------------------------------------------------------
# Verifier Module contract
# --------------------------------------------------------------------------

class ChunkVerification(BaseModel):
    chunk_id: str
    document_id: str
    nli_label: NLILabel
    nli_confidence: float = Field(..., ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """Output of Verifier.verify(claim, evidence_chunks). Section 6/8."""
    model_config = ConfigDict(frozen=True)

    verifications: list[ChunkVerification] = Field(default_factory=list)
    verifier_version: str


# --------------------------------------------------------------------------
# Image Module contract (new, Section 7A)
# --------------------------------------------------------------------------

class ReverseSearchMatch(BaseModel):
    url: str
    first_seen_date: Optional[date] = None
    context_similarity: float = Field(..., ge=0.0, le=1.0)


class ImageVerificationResult(BaseModel):
    """Output of ImageAnalyzer.analyze(image, claim_text). Section 6/7A."""
    model_config = ConfigDict(frozen=True)

    manipulation_detected: bool
    manipulation_confidence: float = Field(..., ge=0.0, le=1.0)
    reverse_search_matches: list[ReverseSearchMatch] = Field(default_factory=list)
    reverse_search_available: bool = Field(
        default=True,
        description="False if the external reverse-search call failed/timed out — "
                    "downstream must degrade gracefully, never fabricate a match list.",
    )
    earliest_known_date: Optional[date] = None
    caption_image_consistency: float = Field(
        ..., ge=0.0, le=1.0,
        description="CLIP-style similarity between claim text and image content.",
    )
    forensics_model_version: str
    consistency_model_version: str


# --------------------------------------------------------------------------
# Fusion Module contract (new, Section 7A)
# --------------------------------------------------------------------------

class FusedResult(BaseModel):
    """
    Output of Fusion.reconcile(text_result, image_result). Section 7A.

    Deliberately keeps the two signals distinguishable rather than blended
    into one number ("do not average unlike quantities", Section 11).
    """
    model_config = ConfigDict(frozen=True)

    text_verification: Optional[VerificationResult] = None
    image_verification: Optional[ImageVerificationResult] = None
    caption_image_consistency: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    note: Optional[str] = None


# --------------------------------------------------------------------------
# Aggregator contract
# --------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """A single piece of evidence surfaced in the final decision (Section 8 response shape)."""
    document_id: str
    chunk_id: str
    source: str
    source_quality: float
    retrieval_score: float
    nli_label: NLILabel
    nli_confidence: float


class DecisionResult(BaseModel):
    """Output of Aggregator.aggregate(...). Section 8/11."""
    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    heuristic_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Documented, tunable, NOT a calibrated probability.",
    )
    calibrated_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Only populated if a downstream calibration model was trained. "
                    "Never fused with heuristic_score into one field.",
    )
    support_mass: float = 0.0
    contradict_mass: float = 0.0
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Orchestration event (the append-only decision trail, Section 3/9)
# --------------------------------------------------------------------------

class OrchestrationEvent(BaseModel):
    claim_id: UUID
    from_state: Optional[OrchestratorState]
    to_state: OrchestratorState
    reason: str
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# Top-level claim request/response (Section 8 API contract)
# --------------------------------------------------------------------------

class ImageInput(BaseModel):
    data: str = Field(..., description="Base64-encoded image bytes.")
    mime_type: str


class ClaimOptions(BaseModel):
    force_verification: bool = False


class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=4000)
    image: Optional[ImageInput] = None
    options: ClaimOptions = Field(default_factory=ClaimOptions)


class ClaimResult(BaseModel):
    """The full response shape from POST /v1/claims/analyze (Section 8)."""
    claim_id: UUID = Field(default_factory=uuid4)
    request_id: str
    status: OrchestratorState
    verdict: Optional[Verdict] = None
    decision: Optional[DecisionResult] = None
    classifier: Optional[ClassificationResult] = None
    image_analysis: Optional[ImageVerificationResult] = None
    fusion: Optional[FusedResult] = None
    decision_trail: list[str] = Field(default_factory=list)
    corpus_version: Optional[str] = None
    latency_ms: Optional[int] = None
