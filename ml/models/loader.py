from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PredictionResult:
    label: str
    raw_logit_confidence: float
    calibrated_probability: float
    model_version: str
    probabilities: list[float] | None = None


class ClassifierModel:
    """Loads a Hugging Face sequence-classification artifact.

    The loader deliberately keeps service/API concerns out of the ML layer.
    Team 1 can wrap this object inside services/classifier/.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
    ):
        self.artifact_dir = Path(artifact_dir)

        if not self.artifact_dir.exists():
            raise FileNotFoundError(self.artifact_dir)

        metadata_path = self.artifact_dir / "metadata.json"
        mapping_path = self.artifact_dir / "label_mapping.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing {metadata_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing {mapping_path}")

        self.metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        self.label_mapping: dict[str, str] = json.loads(
            mapping_path.read_text(encoding="utf-8")
        )

        self.model_version = self.metadata["model_version"]
        self.temperature = 1.0

        calibration_path = self.artifact_dir / "calibration.json"
        if calibration_path.exists():
            calibration = json.loads(
                calibration_path.read_text(encoding="utf-8")
            )
            self.temperature = float(calibration.get("temperature", 1.0))

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

        if not model_dir.exists() or not tokenizer_dir.exists():
            raise FileNotFoundError(
                "Artifact must contain model/ and tokenizer/ directories."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )

        if self.device:
            self._model.to(self.device)

        self._model.eval()

    def predict(self, text: str) -> PredictionResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        self._lazy_load()

        import torch

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.metadata.get("max_length", 256)),
        )

        if self.device:
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits

        raw_probs = torch.softmax(logits, dim=-1)
        calibrated_probs = torch.softmax(logits / self.temperature, dim=-1)

        raw_idx = int(torch.argmax(raw_probs, dim=-1).item())
        calibrated_idx = int(torch.argmax(calibrated_probs, dim=-1).item())

        label = self.label_mapping.get(
            str(calibrated_idx),
            self._model.config.id2label.get(calibrated_idx, str(calibrated_idx)),
        )

        return PredictionResult(
            label=label,
            raw_logit_confidence=float(raw_probs[0, raw_idx].item()),
            calibrated_probability=float(calibrated_probs[0, calibrated_idx].item()),
            model_version=self.model_version,
            probabilities=[float(x) for x in calibrated_probs[0].tolist()],
        )


def load_classifier(
    version: str,
    *,
    models_root: str | Path = "ml/models/artifacts",
    device: str | None = None,
) -> ClassifierModel:
    """Load classifier-v<version> from the filesystem registry."""
    artifact_dir = Path(models_root) / "classifier" / version
    return ClassifierModel(artifact_dir, device=device)
