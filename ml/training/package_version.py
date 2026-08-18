"""
ml_classifier/training/package_version.py

Team 4 — Stage 11-12: PACKAGE + VERSION -> VERSIONED CHECKPOINT / MODEL ARTIFACT

Bundles everything needed to serve the classifier into one immutable,
versioned folder:
    model weights + tokenizer + label mapping + calibration (temperature)
    + full provenance metadata (dataset version, hyperparams, metrics)

This is the artifact the orchestrator's classifier module actually loads.
Nothing here is trainable state — it's a frozen, traceable package.

Consumes:
    team4_package/ml_classifier/training/checkpoints/best/             (model + tokenizer, from train.py)
    team4_package/ml_classifier/training/checkpoints/training_run.json (from train.py)
    team4_package/ml_classifier/training/checkpoints/calibration.json  (from calibrate.py)

Produces:
    team4_package/ml_classifier/training/registry/classifier-v{N}/
        model/                  (HF model + tokenizer files)
        labels.json             (id_to_label / label_to_id)
        calibration.json        (temperature, ECE before/after)
        manifest.json           (model_version, dataset_version, metrics,
                                  hyperparameters, created_at, artifact hashes)
    team4_package/ml_classifier/training/registry/latest.json   (pointer to newest version — convenience
                                          for whoever wires up the classifier
                                          module / orchestrator)

Run:
    python -m ml_classifier.training.package_version
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("package_version")

CONFIG_PATH = Path(__file__).parent / "config.yaml"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # team4_package/ml_classifier/training/package_version.py -> cog_project root


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key in ["data_dir", "label_mapping_path", "output_dir", "registry_dir"]:
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(PROJECT_ROOT / cfg[key])
    return cfg


def dir_hash(path: Path) -> str:
    """Stable content hash over every file in a directory (sorted, so order-independent)."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def next_version_number(registry_dir: Path, model_family: str) -> int:
    if not registry_dir.exists():
        return 1
    existing = [
        p.name for p in registry_dir.iterdir()
        if p.is_dir() and p.name.startswith(f"{model_family}-v")
    ]
    if not existing:
        return 1
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("-v")[-1]))
        except ValueError:
            continue
    return (max(nums) + 1) if nums else 1


def main():
    cfg = load_config()
    checkpoints_dir = Path(cfg["output_dir"])
    best_dir = checkpoints_dir / "best"
    training_run_path = checkpoints_dir / "training_run.json"
    calibration_path = checkpoints_dir / "calibration.json"

    for required in (best_dir, training_run_path, calibration_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Run in order: "
                f"train.py -> calibrate.py -> package_version.py"
            )

    with open(training_run_path) as f:
        training_run = json.load(f)
    with open(calibration_path) as f:
        calibration = json.load(f)

    registry_dir = Path(cfg["registry_dir"])
    registry_dir.mkdir(parents=True, exist_ok=True)

    version_num = next_version_number(registry_dir, cfg["model_family"])
    model_version = f"{cfg['model_family']}-v{version_num}.0"
    version_dir = registry_dir / f"{cfg['model_family']}-v{version_num}"
    model_dir = version_dir / "model"

    log.info(f"Packaging {model_version} -> {version_dir}")

    # Copy model + tokenizer files as-is (immutable snapshot, not a reference).
    shutil.copytree(best_dir, model_dir)

    # labels.json — same mapping used at train time, re-exported here so the
    # served artifact is self-contained and doesn't depend on ml/datasets/ paths.
    with open(version_dir / "labels.json", "w") as f:
        json.dump(training_run["label_mapping"], f, indent=2)

    # calibration.json — carried over as-is.
    with open(version_dir / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    artifact_hash = dir_hash(model_dir)

    manifest = {
        "model_version": model_version,
        "base_model": training_run["base_model"],
        "dataset_version": training_run["dataset_version"],
        "hyperparameters": training_run["hyperparameters"],
        "validation_metrics": training_run["validation_metrics"],
        "calibration": {
            "method": calibration["method"],
            "temperature": calibration["temperature"],
            "ece_before": calibration["ece_before"],
            "ece_after": calibration["ece_after"],
        },
        "artifact_hash": artifact_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        # Output contract this artifact serves, per the architecture spec:
        "output_contract": ["label", "raw_logit_confidence", "calibrated_probability", "model_version"],
    }
    with open(version_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # latest.json — convenience pointer, NOT a substitute for pinning model_version
    # explicitly wherever this artifact is loaded/served.
    with open(registry_dir / "latest.json", "w") as f:
        json.dump({"model_version": model_version, "path": str(version_dir)}, f, indent=2)

    log.info(f"Versioned artifact ready: {model_version}")
    log.info(f"  path:            {version_dir}")
    log.info(f"  dataset_version: {training_run['dataset_version']}")
    log.info(f"  temperature:     {calibration['temperature']:.4f}")
    log.info(f"  artifact_hash:   {artifact_hash}")


if __name__ == "__main__":
    main()
