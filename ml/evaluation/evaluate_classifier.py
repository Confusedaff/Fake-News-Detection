from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .error_analysis import load_predictions
from .metrics import classification_metrics, expected_calibration_error


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="ml/evaluation/results")
    parser.add_argument("--positive-label", default=None)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    df = load_predictions(args.predictions)

    required = {
        "sample_id",
        "true_label",
        "predicted_label",
        "raw_confidence",
        "calibrated_probability",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    # If class probabilities are available, they must be named
    # probability_0, probability_1, ... in class-index order.
    prob_cols = sorted(
        [c for c in df.columns if c.startswith("probability_")],
        key=lambda x: int(x.split("_")[-1]),
    )

    labels = sorted(set(df["true_label"]) | set(df["predicted_label"]))

    result = classification_metrics(
        df["true_label"],
        df["predicted_label"],
        labels=labels,
        probabilities=df[prob_cols].to_numpy(float) if prob_cols else None,
        positive_label=args.positive_label,
    )

    if prob_cols:
        # true_label must be integer class IDs for this ECE implementation.
        try:
            y_true_idx = df["true_label"].astype(int).to_numpy()
            probs = df[prob_cols].to_numpy(float)
            ece, bins = expected_calibration_error(
                y_true_idx, probs, n_bins=args.n_bins
            )
            result["ece"] = ece
            result["calibration_bins"] = bins
        except ValueError:
            result["ece_note"] = (
                "ECE skipped because true_label is not integer class indices. "
                "Provide class-index labels/probabilities for multiclass ECE."
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, default=float),
        encoding="utf-8",
    )

    cm = confusion_matrix(
        df["true_label"],
        df["predicted_label"],
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    ).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    df.to_csv(output / "predictions.csv", index=False)

    print(json.dumps(result, indent=2, default=float))
    print(f"\nSaved evaluation artifacts to: {output}")


if __name__ == "__main__":
    main()
