from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"sample_id", "truth_label", "difficulty", "topic"}


def load_golden_set(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # dtype=str is important:
    # LIAR contains literal labels such as "true"/"false".
    # pandas may otherwise infer those columns as booleans.
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path, dtype=False)
        df = df.fillna("").astype(str)
    else:
        raise ValueError("Golden set must be CSV or JSON")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Golden set missing required columns: {sorted(missing)}")

    if df["sample_id"].duplicated().any():
        duplicates = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate golden-set sample_id values: {duplicates[:10]}")

    return df


def check_no_split_overlap(
    golden: pd.DataFrame,
    *,
    train_ids: set[str] | None = None,
    validation_ids: set[str] | None = None,
    test_ids: set[str] | None = None,
) -> dict:
    """Safety check to prevent golden-set leakage into train/validation/test."""
    golden_ids = set(golden["sample_id"].astype(str))

    overlaps = {
        "train": sorted(golden_ids & (train_ids or set())),
        "validation": sorted(golden_ids & (validation_ids or set())),
        "test": sorted(golden_ids & (test_ids or set())),
    }

    leaked = {k: v for k, v in overlaps.items() if v}
    return {
        "safe": not leaked,
        "overlaps": leaked,
        "golden_count": len(golden_ids),
    }


def save_safety_report(report: dict, path: str | Path):
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
