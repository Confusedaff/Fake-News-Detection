"""
team4_package/ml_classifier/evaluation/generate_predictions.py

Generates batch predictions on the held-out test split using the versioned model artifact.
Outputs prediction CSV adhering to the evaluation contract:
sample_id, true_label, predicted_label, raw_confidence, calibrated_probability, probability_0..5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from ml_classifier.models.loader import load_classifier
from ml_classifier.training.dataset import LiarDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("generate_predictions")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def generate_test_predictions(
    model_version: str = "classifier-v3.0",
    test_csv_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> pd.DataFrame:
    if test_csv_path is None:
        test_csv_path = PROJECT_ROOT / "team4_package" / "ml_classifier" / "datasets" / "processed" / "test.csv"
    test_csv_path = Path(test_csv_path)

    if output_csv_path is None:
        output_csv_path = (
            PROJECT_ROOT
            / "team4_package"
            / "ml_classifier"
            / "evaluation"
            / "results"
            / model_version
            / "predictions.csv"
        )
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading {model_version} on {dev}...")
    clf = load_classifier(model_version, device=dev)
    clf._lazy_load()

    log.info(f"Loading test split from {test_csv_path}...")
    raw_df = pd.read_csv(test_csv_path, dtype=str, keep_default_na=False)

    test_ds = LiarDataset(
        str(test_csv_path),
        clf._tokenizer,
        max_length=int(clf.metadata.get("max_length", 256)),
        use_metadata=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=clf._tokenizer)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    all_logits = []
    log.info(f"Running inference over {len(test_ds)} test examples...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch.pop("labels", None)
            batch = {k: v.to(dev) for k, v in batch.items()}
            outputs = clf._model(**batch)
            all_logits.append(outputs.logits.cpu())

    logits = torch.cat(all_logits, dim=0)

    # Raw and calibrated probabilities
    raw_probs = torch.softmax(logits, dim=-1).numpy()
    calibrated_probs = torch.softmax(logits / clf.temperature, dim=-1).numpy()

    preds = np.argmax(calibrated_probs, axis=-1)
    raw_confs = np.max(raw_probs, axis=-1)
    calib_probs = np.max(calibrated_probs, axis=-1)

    id_to_label = clf.label_mapping
    predicted_labels = [id_to_label.get(str(p), str(p)) for p in preds]

    sample_ids = raw_df["id"].tolist() if "id" in raw_df.columns else [str(i) for i in range(len(raw_df))]
    true_labels = raw_df["label"].tolist() if "label" in raw_df.columns else raw_df["label_id"].tolist()
    true_label_ids = raw_df["label_id"].tolist() if "label_id" in raw_df.columns else preds.tolist()

    res_df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": true_labels,
        "true_label_id": true_label_ids,
        "predicted_label": predicted_labels,
        "raw_confidence": raw_confs,
        "calibrated_probability": calib_probs,
    })

    num_classes = calibrated_probs.shape[1]
    for c in range(num_classes):
        res_df[f"probability_{c}"] = calibrated_probs[:, c]

    res_df.to_csv(output_csv_path, index=False)
    log.info(f"Saved {len(res_df)} predictions to {output_csv_path}")

    return res_df


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for model evaluation.")
    parser.add_argument("--model-version", default="classifier-v3.0")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    generate_test_predictions(
        model_version=args.model_version,
        test_csv_path=args.test_csv,
        output_csv_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
