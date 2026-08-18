"""
ml_classifier/training/calibrate.py

Team 4 — Stage 9-10: CALIBRATION -> CALIBRATED CLASSIFIER

Raw softmax confidence out of a fine-tuned transformer is known to be
overconfident (see: Guo et al. 2017, "On Calibration of Modern Neural
Networks"). This script fits a single scalar temperature T on the held-out
`calibration.csv` split (never seen during training or model selection) via
temperature scaling:

    calibrated_probs = softmax(logits / T)

T > 1 softens (de-confidences) the distribution; T < 1 sharpens it.
T is fit by minimizing NLL of the calibration set — this does not change
the model's argmax predictions or accuracy at all, only how trustworthy the
reported probability is.

Consumes:
    team4_package/ml_classifier/training/checkpoints/best/            (raw fine-tuned model, from train.py)
    team4_package/ml_classifier/training/checkpoints/training_run.json
    team4_package/ml_classifier/datasets/processed/calibration.csv     (held-out split, never trained on)

Produces:
    team4_package/ml_classifier/training/checkpoints/calibration.json
        { temperature, method, ece_before, ece_after, fitted_at_utc }

Fix applied vs previous version:
  - dataset.py now tokenizes with padding=False (dynamic per-batch padding
    for speed — see dataset.py/train.py comments). The manual DataLoader
    here previously relied on every example already being padded to a
    fixed max_length, so it needs an explicit DataCollatorWithPadding as
    its collate_fn now, or batches with mismatched-length sequences would
    fail to stack into a tensor.

Run:
    python -m ml_classifier.training.calibrate
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from ml_classifier.training.dataset import LiarDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("calibrate")

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(__file__).parent.parent.parent.parent
    for key in ["data_dir", "label_mapping_path", "output_dir", "calibration_data_path", "registry_dir"]:
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(project_root / cfg[key])
    return cfg


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen model once over the calibration set and cache logits + labels."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        all_logits.append(out.logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE: weighted gap between confidence and accuracy across confidence bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, lr: float, max_iter: int) -> float:
    """Fit a single scalar T minimizing NLL of (logits / T) against true labels via LBFGS."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    nll_criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=1e-3).item())


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_dir = Path(cfg["output_dir"]) / "best"
    if not best_dir.exists():
        raise FileNotFoundError(
            f"No trained checkpoint at {best_dir} — run `python -m ml_classifier.training.train` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)

    calib_ds = LiarDataset(cfg["calibration_data_path"], tokenizer, max_length=cfg["max_length"])

    # dataset.py tokenizes with padding=False now — pad dynamically per batch here.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    calib_loader = DataLoader(
        calib_ds,
        batch_size=cfg["calibration_batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )

    log.info(f"Collecting logits on {len(calib_ds)} calibration examples...")
    logits, labels = collect_logits(model, calib_loader, device)
    logits_np, labels_np = logits.numpy(), labels.numpy()

    # ECE before calibration (raw softmax).
    raw_probs = F.softmax(logits, dim=1).numpy()
    ece_before = expected_calibration_error(raw_probs, labels_np)
    log.info(f"ECE before calibration: {ece_before:.4f}")

    # Fit temperature on CPU tensors (LBFGS is fine here; calibration sets are small).
    temperature = fit_temperature(
        logits.clone().requires_grad_(False),
        labels,
        lr=cfg["calibration_lr"],
        max_iter=cfg["calibration_max_iter"],
    )
    log.info(f"Fitted temperature T = {temperature:.4f}")

    # ECE after calibration.
    calibrated_probs = F.softmax(logits / temperature, dim=1).numpy()
    ece_after = expected_calibration_error(calibrated_probs, labels_np)
    log.info(f"ECE after calibration:  {ece_after:.4f}")

    calibration_record = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "calibration_set_size": len(calib_ds),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(cfg["output_dir"]) / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(calibration_record, f, indent=2)

    log.info(f"Saved calibration record to {out_path}")
    log.info("Calibrated classifier ready — next: python -m ml_classifier.training.package_version")


if __name__ == "__main__":
    main()