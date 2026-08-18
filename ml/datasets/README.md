# Datasets Module — LIAR Benchmark & Preprocessing Pipeline

This module is responsible for raw data ingestion, cleaning, validation, deduplication, 4-way splitting, and metadata generation for the **LIAR** political fact-checking benchmark.

---

## 1. Directory Structure

```
datasets/
├── config.yaml              # Dataset configuration (paths, columns, seed, split ratios)
├── prepare_dataset.py       # Main preprocessing pipeline script
├── datatset/                # Raw input data
│   ├── train.tsv            # Raw LIAR training split (10,240 rows)
│   ├── valid.tsv            # Raw LIAR validation split (1,284 rows)
│   ├── test.tsv             # Raw LIAR test split (1,267 rows)
│   └── README               # Original LIAR dataset documentation
├── processed/               # Cleaned, standardized CSV splits
│   ├── train.csv            # Training set (10,258 samples)
│   ├── validation.csv       # Model selection & early stopping split (642 samples)
│   ├── calibration.csv      # Temperature scaling calibration split (642 samples)
│   ├── test.csv             # Held-out final evaluation split (1,283 samples)
│   └── version.yaml         # Immutable dataset version hash & split counts
└── metadata/                # Dataset documentation & statistics
    ├── dataset_card.json    # Provenance, license, citation, and split details
    ├── dataset_stats.json   # Class distributions, token lengths, speaker statistics
    └── label_mapping.json   # 6-class ID-to-label mapping dictionary
```

---

## 2. The 6-Class LIAR Taxonomy

The LIAR benchmark provides fine-grained truthfulness ratings:

| Label ID | Label Name | Description |
| :---: | :--- | :--- |
| `0` | `pants-fire` | Completely fabricated claim with no factual basis. |
| `1` | `false` | Factually incorrect claim. |
| `2` | `barely-true` | Contains a grain of truth but distorts key facts or creates a false impression. |
| `3` | `half-true` | Partially accurate, but leaves out important details or takes facts out of context. |
| `4` | `mostly-true` | Accurate on the main point, with only minor nuances needing clarification. |
| `5` | `true` | Completely accurate statement with no significant errors. |

---

## 3. Data Splitting Strategy

The LIAR validation split is partitioned 50/50 to provide an independent calibration set:

```
RAW LIAR DATA
  ├── train.tsv (10,240) ──────────────► train.csv (10,258 after cleaning & dedup)
  ├── valid.tsv (1,284)  ──┬───────────► validation.csv (642 samples for early stopping)
  │                        └───────────► calibration.csv (642 samples for temperature scaling)
  └── test.tsv  (1,267)  ──────────────► test.csv (1,283 held-out test samples)
```

### Split Integrity & Leakage Prevention
- **0% Leakage**: Zero ID overlap between `train`, `validation`, `calibration`, and `test` splits.
- **Unbiased Test Set**: `test.csv` is never accessed during training, hyperparameter tuning, or calibration.

---

## 4. Column Schema (Processed CSVs)

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` | Unique sample identifier (e.g. `11972.json`) |
| `label` | `str` | 6-class label string (`pants-fire` .. `true`) |
| `label_id` | `int` | Integer class index (`0` .. `5`) |
| `statement` | `str` | Raw claim / statement text |
| `subject` | `str` | Subject / topic categories (e.g. `economy,taxes`) |
| `speaker` | `str` | Name of the speaker / entity |
| `job_title` | `str` | Speaker's job or political title |
| `state_info` | `str` | State / location affiliation |
| `party_affiliation` | `str` | Political party (e.g. `democrat`, `republican`) |
| `barely_true_counts` | `int` | Historical barely-true count for speaker |
| `false_counts` | `int` | Historical false count for speaker |
| `half_true_counts` | `int` | Historical half-true count for speaker |
| `mostly_true_counts` | `int` | Historical mostly-true count for speaker |
| `pants_on_fire_counts`| `int` | Historical pants-on-fire count for speaker |
| `context` | `str` | Setting / venue of statement (e.g. `a speech in Iowa`) |

---

## 5. Usage

To run the dataset preprocessing pipeline:

```bash
python team4_package/ml_classifier/datasets/prepare_dataset.py
```

Options:
- `--config`: Path to `config.yaml` (default: `team4_package/ml_classifier/datasets/config.yaml`).
- `--force`: Force rebuild and re-split even if processed files exist.

### Pandas Correctness Note
Always read processed CSV files with `dtype=str, keep_default_na=False`:
```python
import pandas as pd
df = pd.read_csv("team4_package/ml_classifier/datasets/processed/train.csv", dtype=str, keep_default_na=False)
```
*(Prevents pandas from coercing string labels `"true"` and `"false"` into booleans).*
