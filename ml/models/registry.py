from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelRegistry:
    """Small filesystem-backed registry for immutable model versions."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def list_versions(self, model_name: str = "classifier") -> list[str]:
        base = self.root / model_name
        if not base.exists():
            return []
        return sorted(
            p.name for p in base.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def artifact_dir(self, model_name: str, version: str) -> Path:
        path = self.root / model_name / version
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return path

    def metadata(self, model_name: str, version: str) -> dict[str, Any]:
        path = self.artifact_dir(model_name, version) / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metadata.json: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def validate(self, model_name: str, version: str) -> list[str]:
        """Return validation errors without modifying the artifact."""
        errors = []
        path = self.artifact_dir(model_name, version)

        required = ["metadata.json", "label_mapping.json"]
        for name in required:
            if not (path / name).exists():
                errors.append(f"Missing {name}")

        if not (path / "model").exists():
            errors.append("Missing model/ directory")
        if not (path / "tokenizer").exists():
            errors.append("Missing tokenizer/ directory")

        return errors
