from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc",
    "ece",
]


def read_metrics(path: str | Path, version: str) -> dict:
    data = pd.read_json(path, typ="series").to_dict()
    data["model_version"] = version
    return data


def main():
    parser = argparse.ArgumentParser(description="Compare two evaluated model versions.")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--version-a", required=True)
    parser.add_argument("--version-b", required=True)
    args = parser.parse_args()

    a = read_metrics(args.model_a, args.version_a)
    b = read_metrics(args.model_b, args.version_b)

    rows = []
    for metric in METRICS:
        av = a.get(metric)
        bv = b.get(metric)
        if av is None and bv is None:
            continue
        rows.append({
            "metric": metric,
            args.version_a: av,
            args.version_b: bv,
            "change": None if av is None or bv is None else bv - av,
        })

    table = pd.DataFrame(rows)

    # Explicitly set no narrow fixed-width assumptions. This avoids
    # formatting failures with long version names.
    print(table.to_string(index=False))
    print("\nInterpretation: compare multiple metrics; do not select a model "
          "using accuracy alone.")


if __name__ == "__main__":
    main()
