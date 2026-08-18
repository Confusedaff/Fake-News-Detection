from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_predictions(path: str | Path) -> pd.DataFrame:
    # Critical correctness fix:
    # "true"/"false" must remain strings, not pandas booleans.
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )


def build_error_report(
    predictions: pd.DataFrame,
    *,
    confidence_threshold: float = 0.85,
) -> dict:
    required = {
        "sample_id",
        "true_label",
        "predicted_label",
        "raw_confidence",
        "calibrated_probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = predictions.copy()
    df["correct"] = (
        df["true_label"].astype(str) == df["predicted_label"].astype(str)
    )
    df["calibrated_probability"] = pd.to_numeric(
        df["calibrated_probability"], errors="coerce"
    )
    df["raw_confidence"] = pd.to_numeric(
        df["raw_confidence"], errors="coerce"
    )

    high_conf_wrong = df[
        (~df["correct"])
        & (df["calibrated_probability"] >= confidence_threshold)
    ]

    low_conf_correct = df[
        (df["correct"])
        & (df["calibrated_probability"] < 0.60)
    ]

    false_positives = df[
        (~df["correct"])
        & (df["predicted_label"] != df["true_label"])
    ]

    confusion = (
        df.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return {
        "total_examples": int(len(df)),
        "errors": int((~df["correct"]).sum()),
        "error_rate": float((~df["correct"]).mean()),
        "high_confidence_wrong_count": int(len(high_conf_wrong)),
        "low_confidence_correct_count": int(len(low_conf_correct)),
        "confusion_pairs": confusion.to_dict(orient="records"),
        "high_confidence_wrong": high_conf_wrong.to_dict(orient="records"),
    }


def save_error_report(report: dict, path: str | Path):
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
