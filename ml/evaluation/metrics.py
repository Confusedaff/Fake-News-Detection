from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    y_pred,
    *,
    labels=None,
    probabilities=None,
    positive_label=None,
) -> dict:
    """Compute classification metrics without assuming label names."""
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    if probabilities is not None and positive_label is not None:
        unique = sorted(set(y_true))
        if len(unique) == 2:
            positive_index = unique.index(positive_label)
            p = np.asarray(probabilities)
            if p.ndim == 2:
                result["roc_auc"] = float(
                    roc_auc_score(
                        np.asarray(y_true) == positive_label,
                        p[:, positive_index],
                    )
                )
            else:
                result["roc_auc"] = float(
                    roc_auc_score(np.asarray(y_true) == positive_label, p)
                )

    return result


def expected_calibration_error(
    y_true,
    probabilities,
    *,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Compute multiclass ECE using max probability as confidence."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [n_samples, n_classes]")

    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correctness = predictions == y_true

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)

        count = int(mask.sum())
        if count == 0:
            continue

        accuracy = float(correctness[mask].mean())
        avg_confidence = float(confidence[mask].mean())
        gap = abs(accuracy - avg_confidence)
        weight = count / len(y_true)
        ece += weight * gap

        rows.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "count": count,
            "accuracy": accuracy,
            "confidence": avg_confidence,
            "gap": gap,
        })

    return float(ece), rows
