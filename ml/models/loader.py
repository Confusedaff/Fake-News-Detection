"""
ml_classifier/models/loader.py

Team 4 — Model loading and prediction contract for the 6-way LIAR classifier.
Loads fine-tuned RoBERTa sequence classification artifacts with post-hoc temperature scaling.

Output Contract:
{
  "text": "Some claim...",
  "prediction": {
    "label": "mostly-true",
    "label_id": 4,
    "confidence": 0.71
  },
  "probabilities": {
    "pants-fire": 0.01,
    "false": 0.05,
    "barely-true": 0.08,
    "half-true": 0.15,
    "mostly-true": 0.71,
    "true": 0.00
  },
  "raw_logit_confidence": 0.76,
  "calibrated_probability": 0.71,
  "model_version": "classifier-v3.0"
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


@dataclass
class PredictionResult:
    label: str
    label_id: int
    raw_logit_confidence: float
    calibrated_probability: float
    model_version: str
    probabilities: dict[str, float]
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    text: str = ""
    prediction: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.prediction:
            self.prediction = {
                "label": self.label,
                "label_id": self.label_id,
                "confidence": round(self.calibrated_probability, 4),
            }

    @property
    def confidence(self) -> float:
        return self.calibrated_probability

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prediction": self.prediction,
            "probabilities": self.probabilities,
            "raw_logit_confidence": round(self.raw_logit_confidence, 4),
            "calibrated_probability": round(self.calibrated_probability, 4),
            "model_version": self.model_version,
        }

    def to_response(self):
        """Converts result to the shared Pydantic ClassifierResponse schema if installed."""
        try:
            from packages.schemas.classifier import ClassifierResponse, Prediction
        except ImportError:
            try:
                from schemas.classifier import ClassifierResponse, Prediction
            except ImportError:
                return self.as_dict()

        return ClassifierResponse(
            text=self.text,
            prediction=Prediction(
                label=self.prediction["label"],
                label_id=self.prediction["label_id"],
                confidence=self.prediction["confidence"],
            ),
            probabilities=self.probabilities,
            raw_logit_confidence=round(self.raw_logit_confidence, 4),
            calibrated_probability=round(self.calibrated_probability, 4),
            model_version=self.model_version,
        )


def _resolve_models_root(models_root: str | Path | None = None) -> Path:
    if models_root is not None:
        p = Path(models_root)
        if p.exists():
            return p
    # Try package local artifacts directory first
    local_artifacts = Path(__file__).parent / "artifacts"
    if local_artifacts.exists():
        return local_artifacts
    # Fallback search candidates
    for cand in [
        Path("team4_package/ml_classifier/models/artifacts"),
        Path("ml/models/artifacts"),
        Path("models/artifacts"),
    ]:
        if cand.exists():
            return cand
    return local_artifacts


class ClassifierModel:
    """Loads a Hugging Face sequence-classification artifact with calibration.

    Keeps service/API concerns decoupled from the ML layer.
    The Orchestrator and service layer wrap this object directly.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
    ):
        self.artifact_dir = Path(artifact_dir)

        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {self.artifact_dir}")

        metadata_path = self.artifact_dir / "metadata.json"
        manifest_path = self.artifact_dir / "manifest.json"
        mapping_path = self.artifact_dir / "label_mapping.json"
        labels_path = self.artifact_dir / "labels.json"

        if metadata_path.exists():
            self.metadata: dict[str, Any] = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        elif manifest_path.exists():
            self.metadata: dict[str, Any] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        else:
            raise FileNotFoundError(f"Missing metadata.json or manifest.json in {self.artifact_dir}")

        if mapping_path.exists():
            raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        elif labels_path.exists():
            raw_mapping = json.loads(labels_path.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"Missing label_mapping.json or labels.json in {self.artifact_dir}")

        if isinstance(raw_mapping, dict) and "id_to_label" in raw_mapping:
            self.label_mapping = {str(k): v for k, v in raw_mapping["id_to_label"].items()}
        else:
            self.label_mapping = {str(k): v for k, v in raw_mapping.items()}

        self.model_version = self.metadata.get("model_version", self.artifact_dir.name)
        self.temperature = float(self.metadata.get("temperature", 1.0))

        calibration_path = self.artifact_dir / "calibration.json"
        if calibration_path.exists():
            calibration = json.loads(
                calibration_path.read_text(encoding="utf-8")
            )
            self.temperature = float(calibration.get("temperature", self.temperature))

        self.device = device
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Loading a Hugging Face model requires torch and transformers."
            ) from exc

        model_dir = self.artifact_dir / "model"
        tokenizer_dir = self.artifact_dir / "tokenizer"

        if not tokenizer_dir.exists():
            tokenizer_dir = model_dir

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Artifact must contain a model/ directory: {model_dir}"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )

        if self.device:
            self._model.to(self.device)

        self._model.eval()

    def predict(self, text: Union[str, dict, Any]) -> PredictionResult:
        if hasattr(text, "text"):
            raw_text = getattr(text, "text")
        elif isinstance(text, dict) and "text" in text:
            raw_text = text["text"]
        elif isinstance(text, str):
            raw_text = text
        else:
            raw_text = str(text)

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("text must be a non-empty string")

        self._lazy_load()

        import torch

        max_len = 256
        if "max_length" in self.metadata:
            max_len = int(self.metadata["max_length"])
        elif "hyperparameters" in self.metadata and "max_length" in self.metadata["hyperparameters"]:
            max_len = int(self.metadata["hyperparameters"]["max_length"])

        encoded = self._tokenizer(
            raw_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )

        if self.device:
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        elif next(self._model.parameters()).is_cuda:
            encoded = {k: v.cuda() for k, v in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits

        raw_probs_t = torch.softmax(logits, dim=-1)
        calibrated_probs_t = torch.softmax(logits / self.temperature, dim=-1)

        raw_idx = int(torch.argmax(raw_probs_t, dim=-1).item())
        calibrated_idx = int(torch.argmax(calibrated_probs_t, dim=-1).item())

        label = self.label_mapping.get(
            str(calibrated_idx),
            self._model.config.id2label.get(calibrated_idx, str(calibrated_idx)),
        )

        raw_probs_list = [float(x) for x in raw_probs_t[0].tolist()]
        calibrated_probs_list = [float(x) for x in calibrated_probs_t[0].tolist()]

        num_classes = len(calibrated_probs_list)
        calibrated_dict = {}
        raw_dict = {}
        for i in range(num_classes):
            c_label = self.label_mapping.get(
                str(i),
                self._model.config.id2label.get(i, f"class_{i}"),
            )
            calibrated_dict[c_label] = round(calibrated_probs_list[i], 4)
            raw_dict[c_label] = round(raw_probs_list[i], 4)

        return PredictionResult(
            label=label,
            label_id=calibrated_idx,
            raw_logit_confidence=float(raw_probs_list[raw_idx]),
            calibrated_probability=float(calibrated_probs_list[calibrated_idx]),
            model_version=self.model_version,
            probabilities=calibrated_dict,
            raw_probabilities=raw_dict,
            text=raw_text,
        )


def load_classifier(
    version: str = "classifier-v3.0",
    *,
    models_root: str | Path | None = None,
    device: str | None = None,
) -> ClassifierModel:
    """Load classifier-v<version> from the filesystem registry."""
    root = _resolve_models_root(models_root)
    artifact_dir = root / "classifier" / version
    if not artifact_dir.exists():
        if (root / version).exists():
            artifact_dir = root / version
        elif Path(version).exists():
            artifact_dir = Path(version)
    return ClassifierModel(artifact_dir, device=device)
