from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _default_registry_root() -> Path:
    local_artifacts = Path(__file__).parent / "artifacts"
    if local_artifacts.exists():
        return local_artifacts
    for cand in [
        Path("team4_package/ml_classifier/models/artifacts"),
        Path("ml/models/artifacts"),
        Path("models/artifacts"),
    ]:
        if cand.exists():
            return cand
    return local_artifacts


class ModelRegistry:
    """Small filesystem-backed registry for immutable model versions."""

    def __init__(self, root: str | Path | None = None):
        if root is None:
            self.root = _default_registry_root()
        else:
            self.root = Path(root)

    def list_versions(self, model_name: str = "classifier") -> list[str]:
        base = self.root / model_name
        if not base.exists():
            if self.root.exists() and any((self.root / d).is_dir() for d in ["classifier-v3.0", "classifier-v3"]):
                return sorted(
                    p.name for p in self.root.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                )
            return []
        return sorted(
            p.name for p in base.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def artifact_dir(self, model_name: str, version: str) -> Path:
        path = self.root / model_name / version
        if not path.exists():
            alt_path = self.root / version
            if alt_path.exists():
                return alt_path
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return path

    def metadata(self, model_name: str, version: str) -> dict[str, Any]:
        path = self.artifact_dir(model_name, version) / "metadata.json"
        if not path.exists():
            alt = self.artifact_dir(model_name, version) / "manifest.json"
            if alt.exists():
                return json.loads(alt.read_text(encoding="utf-8"))
            raise FileNotFoundError(f"Missing metadata.json: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def validate(self, model_name: str, version: str) -> list[str]:
        """Return validation errors without modifying the artifact."""
        errors = []
        path = self.artifact_dir(model_name, version)

        required = ["label_mapping.json"]
        if not (path / "metadata.json").exists() and not (path / "manifest.json").exists():
            errors.append("Missing metadata.json or manifest.json")

        for name in required:
            if not (path / name).exists():
                if name == "label_mapping.json" and (path / "labels.json").exists():
                    continue
                errors.append(f"Missing {name}")

        if not (path / "model").exists():
            errors.append("Missing model/ directory")
        if not (path / "tokenizer").exists() and not (path / "model" / "tokenizer.json").exists():
            errors.append("Missing tokenizer/ directory or tokenizer files")

        return errors
