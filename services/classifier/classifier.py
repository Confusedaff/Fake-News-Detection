"""
Classifier Module (Section 4, 6).

Responsibility: wrap the fake-news classifier; return label + raw + calibrated
probability + model version.

Does NOT decide if confidence is "enough" — that is the orchestrator's policy
(packages/config THRESHOLDS.CLASSIFIER_CONFIDENCE_THRESHOLD), not this
module's. This module only ever answers "what does the model think", never
"is that good enough to stop here".

Two implementations ship in the MVP:
- MockClassifier: deterministic, dependency-free, used by Sub-team B to build
  and test the orchestrator/API against the frozen contract before the real
  model exists (Section 23/24, Day 1-3 parallel workstream).
- DistilBertClassifier: the real MVP model (Section 6) — fine-tuned
  DistilBERT/RoBERTa on LIAR, temperature-scaled calibration applied post-hoc.
  Lazy-imports torch/transformers so the mock path has zero heavy deps.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Optional

from packages.schemas import ClassificationResult


class Classifier(ABC):
    """Interface every classifier implementation must satisfy."""

    @abstractmethod
    def predict(self, claim: str) -> ClassificationResult:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_version(self) -> str:
        raise NotImplementedError


class MockClassifier(Classifier):
    """
    Deterministic stand-in used for local dev, unit tests, and Day 1-3
    orchestrator integration before the real model is trained.

    Deterministic-but-varied: derives a pseudo-confidence from a hash of the
    claim text so repeated calls are stable (good for tests/caching) while
    different claims still produce a spread of confidences, letting you
    exercise both the fast path and the verification path locally.
    """

    LABELS = ("true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire")

    def __init__(self, version: str = "mock-classifier-v0"):
        self._version = version

    @property
    def model_version(self) -> str:
        return self._version

    def predict(self, claim: str) -> ClassificationResult:
        digest = hashlib.sha256(claim.strip().lower().encode("utf-8")).hexdigest()
        # Map the hash to a raw confidence in [0.5, 0.99] and pick a label.
        raw = 0.5 + (int(digest[:8], 16) % 5000) / 10000.0
        label_idx = int(digest[8:10], 16) % len(self.LABELS)
        label = self.LABELS[label_idx]

        calibrated = self._calibrate(raw)
        ood = round(1.0 - raw, 4)  # crude proxy: low max-softmax = more OOD

        return ClassificationResult(
            label=label,
            raw_confidence=round(raw, 4),
            calibrated_probability=round(calibrated, 4),
            ood_signal=ood,
            model_version=self._version,
        )

    @staticmethod
    def _calibrate(raw: float, temperature: float = 1.4) -> float:
        """
        Toy temperature scaling so the mock exhibits the same "calibrated !=
        raw" behavior the real model will — this keeps orchestrator logic that
        depends on calibrated_probability honest even against the mock.
        """
        logit = math.log(raw / (1 - raw + 1e-9) + 1e-9)
        scaled = logit / temperature
        return 1 / (1 + math.exp(-scaled))


class DistilBertClassifier(Classifier):
    """
    Real MVP classifier (Section 6): fine-tuned DistilBERT/RoBERTa-base on
    LIAR, with a temperature-scaling calibration map applied post-training.

    Heavy deps (torch, transformers) are imported lazily in __init__ so
    importing this module (or the package) never requires them unless this
    class is actually instantiated — MockClassifier stays fast/dependency-free
    for CI and local dev.
    """

    def __init__(
        self,
        model_path: str,
        version: str,
        calibration_temperature: float = 1.0,
        device: Optional[str] = None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "DistilBertClassifier requires `torch` and `transformers`. "
                "Install them or use MockClassifier for development."
            ) from e

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self._version = version
        self.temperature = calibration_temperature
        # LIAR's 6-way taxonomy, index-aligned to model output logits.
        self.labels = ("true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire")

    @property
    def model_version(self) -> str:
        return self._version

    def predict(self, claim: str) -> ClassificationResult:
        torch = self._torch
        inputs = self.tokenizer(
            claim, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]

        raw_probs = torch.softmax(logits, dim=-1)
        raw_confidence, label_idx = torch.max(raw_probs, dim=-1)

        # Temperature scaling for calibration — never claim raw softmax is calibrated.
        calibrated_probs = torch.softmax(logits / self.temperature, dim=-1)
        calibrated_probability = calibrated_probs[label_idx].item()

        # OOD proxy: 1 - max raw softmax. Cheap, no extra model needed for MVP.
        ood_signal = 1.0 - raw_confidence.item()

        return ClassificationResult(
            label=self.labels[label_idx.item()],
            raw_confidence=round(raw_confidence.item(), 4),
            calibrated_probability=round(calibrated_probability, 4),
            ood_signal=round(ood_signal, 4),
            model_version=self._version,
        )
