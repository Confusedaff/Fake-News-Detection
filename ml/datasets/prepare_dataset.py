"""
ml_classifier/datasets/prepare_dataset.py

Team 4 — Dataset pipeline for the fake-news verifier classifier.

Raw LIAR (.tsv) -> validate -> clean -> version -> split
    -> processed/{train,validation,calibration,test}.csv
    -> metadata/{dataset_card.json, dataset_stats.json, label_mapping.json}
    -> processed/version.yaml

Run:
    python -m ml_classifier.datasets.prepare_dataset
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger("prepare_dataset")

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# prepare_dataset.py
#   -> datasets/
#   -> ml_classifier/
#   -> team4_package/
# PROJECT_ROOT is the repository root.
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_config() -> dict:
    """Load dataset configuration and resolve project-relative paths."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ["raw_dir", "processed_dir", "metadata_dir"]:
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(PROJECT_ROOT / cfg[key])

    return cfg


@dataclass
class ValidationReport:
    """Statistics describing validation/cleaning for one dataset split."""

    rows_in: int
    rows_out: int
    removed_malformed: int
    removed_dup: int
    removed_missing_text: int
    removed_unknown_label: int
    removed_bad_length: int
    label_counts: dict

    def as_dict(self) -> dict:
        return self.__dict__


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_raw_split(cfg: dict, split_file: str) -> pd.DataFrame:
    """
    Load one raw LIAR TSV split.

    LIAR is tab-separated and does not use CSV-style quoting.
    Double quotes appearing inside statements are literal text.

    Therefore QUOTE_NONE is required here. Without it, pandas can interpret
    stray quote characters inside statements as CSV quoting delimiters and
    merge multiple physical TSV rows into a single malformed record.
    """
    path = Path(cfg["raw_dir"]) / split_file

    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw LIAR file: {path}\n"
            f"Download LIAR (train.tsv/valid.tsv/test.tsv) into "
            f"{cfg['raw_dir']}/ before running this pipeline."
        )

    log.info(f"Loading raw LIAR split: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=cfg["liar_columns"],
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
    )

    # Structural integrity check.
    expected_columns = len(cfg["liar_columns"])
    if df.shape[1] != expected_columns:
        raise ValueError(
            f"{split_file}: expected {expected_columns} columns, "
            f"but parsed {df.shape[1]} columns."
        )

    # Basic raw-row sanity check.
    if len(df) == 0:
        raise ValueError(f"{split_file}: parsed zero rows.")

    log.info(
        f"{split_file}: parsed {len(df)} rows x {df.shape[1]} columns"
    )

    return df[cfg["keep_columns"]]


# --------------------------------------------------------------------------
# Validate + clean
# --------------------------------------------------------------------------

def validate_and_clean(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate and clean one split in a tracked single pass.

    Steps:
      1. Normalize statement text.
      2. Remove missing/empty statements.
      3. Apply statement length bounds.
      4. Remove unknown/malformed labels.
      5. Remove exact duplicate (statement, label) pairs.
    """
    rows_in = len(df)

    text_col = cfg["text_column"]
    label_col = cfg["label_column"]
    valid_labels = set(cfg["labels"])

    df = df.copy()

    # ------------------------------------------------------------------
    # Normalize text
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    # Quote characters inside statements are preserved.
    # The parser already handled them correctly with QUOTE_NONE.
    #
    df[text_col] = (
        df[text_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.encode("utf-8", errors="ignore")
        .str.decode("utf-8")
    )

    # ------------------------------------------------------------------
    # Missing / empty text
    # ------------------------------------------------------------------

    missing_mask = (
        df[text_col].isna()
        | (df[text_col].str.len() == 0)
        | (df[text_col].str.lower() == "nan")
    )

    removed_missing_text = int(missing_mask.sum())
    df = df[~missing_mask]

    # ------------------------------------------------------------------
    # Statement length sanity bounds
    # ------------------------------------------------------------------

    len_mask = df[text_col].str.len().between(
        cfg["min_statement_chars"],
        cfg["max_statement_chars"],
    )

    removed_bad_length = int((~len_mask).sum())
    df = df[len_mask]

    # ------------------------------------------------------------------
    # Label validation
    # ------------------------------------------------------------------

    label_mask = df[label_col].isin(valid_labels)

    removed_unknown_label = int((~label_mask).sum())
    df = df[label_mask]

    # ------------------------------------------------------------------
    # Exact duplicate claims
    # ------------------------------------------------------------------

    dup_mask = df.duplicated(
        subset=[text_col, label_col],
        keep="first",
    )

    removed_dup = int(dup_mask.sum())
    df = df[~dup_mask]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    removed_malformed = (
        removed_missing_text
        + removed_bad_length
    )

    report = ValidationReport(
        rows_in=rows_in,
        rows_out=len(df),
        removed_malformed=removed_malformed,
        removed_dup=removed_dup,
        removed_missing_text=removed_missing_text,
        removed_unknown_label=removed_unknown_label,
        removed_bad_length=removed_bad_length,
        label_counts=df[label_col].value_counts().to_dict(),
    )

    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def split_calibration_from_valid(
    valid_df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split LIAR's original validation set into:

        validation -> model selection / development
        calibration -> temperature scaling / probability calibration

    Test remains completely untouched until final evaluation.
    """
    validation_df, calibration_df = train_test_split(
        valid_df,
        test_size=cfg["valid_split_for_calibration"],
        random_state=cfg["seed"],
        stratify=valid_df[cfg["label_column"]],
    )

    return (
        validation_df.reset_index(drop=True),
        calibration_df.reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# Versioning + metadata
# --------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    """Return a deterministic SHA-256 content hash prefix."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_outputs(
    cfg: dict,
    splits: dict[str, pd.DataFrame],
    reports: dict[str, ValidationReport],
) -> None:
    """Write processed splits, metadata, hashes, and dataset version."""
    processed_dir = Path(cfg["processed_dir"])
    metadata_dir = Path(cfg["metadata_dir"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Label mapping
    # ------------------------------------------------------------------

    label_to_id = {
        label: i
        for i, label in enumerate(cfg["labels"])
    }

    # ------------------------------------------------------------------
    # Processed CSVs
    # ------------------------------------------------------------------

    for name, df in splits.items():
        df = df.copy()

        df["label_id"] = df[cfg["label_column"]].map(label_to_id)

        # Defensive check: every row should map to a known label.
        if df["label_id"].isna().any():
            bad_labels = (
                df.loc[df["label_id"].isna(), cfg["label_column"]]
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                f"{name}: found labels without a label_id mapping: "
                f"{bad_labels}"
            )

        df.to_csv(
            processed_dir / f"{name}.csv",
            index=False,
        )

        log.info(
            f"wrote {name}.csv  rows={len(df)}"
        )

    # ------------------------------------------------------------------
    # label_mapping.json
    # ------------------------------------------------------------------

    with open(
        metadata_dir / "label_mapping.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": {
                    v: k for k, v in label_to_id.items()
                },
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # dataset_stats.json
    # ------------------------------------------------------------------

    with open(
        metadata_dir / "dataset_stats.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                name: report.as_dict()
                for name, report in reports.items()
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # File hashes
    # ------------------------------------------------------------------

    file_hashes = {
        f"{name}.csv": file_hash(
            processed_dir / f"{name}.csv"
        )
        for name in splits
    }

    # ------------------------------------------------------------------
    # Dataset card
    # ------------------------------------------------------------------

    dataset_card = {
        "dataset": "LIAR",
        "source": (
            "William Yang Wang, "
            "'Liar, Liar Pants on Fire' (2017)"
        ),
        "prepared_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "labels": cfg["labels"],
        "text_column": cfg["text_column"],
        "seed": cfg["seed"],
        "splits": {
            name: len(df)
            for name, df in splits.items()
        },
        "file_hashes": file_hashes,
    }

    with open(
        metadata_dir / "dataset_card.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dataset_card,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Immutable dataset version
    # ------------------------------------------------------------------

    version_id = hashlib.sha256(
        json.dumps(
            file_hashes,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]

    with open(
        processed_dir / "version.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.dump(
            {
                "dataset_version": f"liar-{version_id}",
                "created_at_utc": dataset_card[
                    "prepared_at_utc"
                ],
                "file_hashes": file_hashes,
                "splits": dataset_card["splits"],
            },
            f,
            sort_keys=False,
        )

    log.info(
        f"dataset_version = liar-{version_id}"
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    """Run the complete Team 4 dataset preparation pipeline."""
    cfg = load_config()

    # ------------------------------------------------------------------
    # Load raw LIAR files
    # ------------------------------------------------------------------

    raw = {
        "train": load_raw_split(
            cfg,
            cfg["raw_files"]["train"],
        ),
        "valid": load_raw_split(
            cfg,
            cfg["raw_files"]["valid"],
        ),
        "test": load_raw_split(
            cfg,
            cfg["raw_files"]["test"],
        ),
    }

    # ------------------------------------------------------------------
    # Validate + clean each original split
    # ------------------------------------------------------------------

    cleaned: dict[str, pd.DataFrame] = {}
    reports: dict[str, ValidationReport] = {}

    for name, df in raw.items():
        cleaned[name], reports[name] = validate_and_clean(
            df,
            cfg,
        )

        report = reports[name]

        log.info(
            f"[{name}] "
            f"{report.rows_in} -> {report.rows_out} rows "
            f"(dup={report.removed_dup}, "
            f"missing_text={report.removed_missing_text}, "
            f"bad_length={report.removed_bad_length}, "
            f"unknown_label={report.removed_unknown_label})"
        )

    # ------------------------------------------------------------------
    # Split original LIAR validation set into:
    #   validation + calibration
    # ------------------------------------------------------------------

    validation_df, calibration_df = (
        split_calibration_from_valid(
            cleaned["valid"],
            cfg,
        )
    )

    # ------------------------------------------------------------------
    # Report bookkeeping
    # ------------------------------------------------------------------
    #
    # rows_in for validation/calibration refers to the cleaned original
    # LIAR validation split from which they were created.
    #
    validation_label_counts = (
        validation_df[cfg["label_column"]]
        .value_counts()
        .to_dict()
    )

    calibration_label_counts = (
        calibration_df[cfg["label_column"]]
        .value_counts()
        .to_dict()
    )

    reports["validation"] = ValidationReport(
        rows_in=len(cleaned["valid"]),
        rows_out=len(validation_df),
        removed_malformed=0,
        removed_dup=0,
        removed_missing_text=0,
        removed_unknown_label=0,
        removed_bad_length=0,
        label_counts=validation_label_counts,
    )

    reports["calibration"] = ValidationReport(
        rows_in=len(cleaned["valid"]),
        rows_out=len(calibration_df),
        removed_malformed=0,
        removed_dup=0,
        removed_missing_text=0,
        removed_unknown_label=0,
        removed_bad_length=0,
        label_counts=calibration_label_counts,
    )

    # The original 'valid' split is now represented by
    # validation + calibration.
    del reports["valid"]

    # ------------------------------------------------------------------
    # Final processed splits
    # ------------------------------------------------------------------

    splits = {
        "train": cleaned["train"],
        "validation": validation_df,
        "calibration": calibration_df,
        "test": cleaned["test"],
    }

    # ------------------------------------------------------------------
    # Write everything
    # ------------------------------------------------------------------

    write_outputs(
        cfg,
        splits,
        reports,
    )

    log.info(
        "Dataset preparation complete."
    )


if __name__ == "__main__":
    main()