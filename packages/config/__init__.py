"""
Central, tunable configuration — thresholds, bounds, and weights.

Kept as one module so a config change never requires touching module code.
Values below are the MVP defaults called out explicitly in the architecture
doc (Section 7 bounds, Section 11 verdict thresholds).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OrchestratorBounds:
    MAX_RETRIEVAL_ATTEMPTS: int = 3
    MAX_EVIDENCE_CHUNKS: int = 10
    MAX_LLM_CALLS: int = 2
    MAX_LATENCY_SOFT_S: float = 8.0
    MAX_LATENCY_HARD_S: float = 15.0


@dataclass(frozen=True)
class RouterThresholds:
    # Router acts on CALIBRATED probability only (never raw logits).
    CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.85
    RETRIEVAL_RELEVANCE_FLOOR: float = 0.6


@dataclass(frozen=True)
class AggregationWeights:
    # Classifier is one more weighted voter, not a tie-breaker by default.
    CLASSIFIER_WEIGHT_DEFAULT: float = 0.3
    CLASSIFIER_WEIGHT_WHEN_NO_EVIDENCE: float = 0.7
    SAME_PUBLISHER_DECAY: float = 0.5  # diminishing returns per repeat publisher
    # Verdict-taxonomy cut points on the normalized (-1..1) mass differential.
    STRONG_MARGIN: float = 0.5
    MODERATE_MARGIN: float = 0.15
    CONFLICT_BALANCE_EPSILON: float = 0.1  # |support - contradict| within this AND both high => CONFLICTING
    CONFLICT_MASS_FLOOR: float = 0.4       # both masses must clear this to count as "high-quality"


BOUNDS = OrchestratorBounds()
THRESHOLDS = RouterThresholds()
WEIGHTS = AggregationWeights()
