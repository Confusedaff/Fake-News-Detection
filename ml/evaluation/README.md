# Evaluation Module — Metrics, Error Analysis & Benchmarking

This module evaluates **versioned classifier releases against the held-out test split (`test.csv`)** to measure unbiased model generalization, error distribution, and confidence calibration.

---

## 1. Directory Structure

```
evaluation/
├── run_pipeline.py          # Complete evaluation orchestrator runner
├── generate_predictions.py  # Batch inference over test split with calibrated probabilities
├── evaluate_classifier.py   # Accuracy, F1, precision, recall, confusion matrix, and ECE
├── error_analysis.py        # High-confidence error analysis and top confusion pairs
├── golden_set.py            # Data leakage and split overlap safety verifier
├── compare_models.py        # Multi-version benchmark comparison table generator
├── metrics.py               # Core metric calculation functions
└── results/                 # Versioned evaluation artifacts
    └── classifier-v3.0/
        ├── metrics.json           # Comprehensive evaluation metrics & classification report
        ├── predictions.csv        # 1,283 sample predictions with calibrated probabilities
        ├── confusion_matrix.png   # 6x6 confusion matrix visualization
        ├── error_report.json      # Error counts and misclassification details
        ├── leakage_report.json    # Verified zero-leakage safety certificate
        └── model_comparison.json  # Multi-version performance benchmark table
```

---

## 2. Held-out Test Split Performance (`classifier-v3.0`)

Evaluated on **1,283 held-out test examples** (`test.csv`):

| Metric | Score | Validation Benchmark |
| :--- | :---: | :---: |
| **Accuracy** | **43.73%** | 47.20% |
| **Macro F1** | **43.89%** | 44.46% |
| **Weighted F1** | **42.54%** | 45.10% |
| **Macro Precision** | **53.23%** | 46.80% |
| **Macro Recall** | **44.31%** | 44.10% |
| **Expected Calibration Error (ECE)** | **3.70%** (`0.0370`) | 6.31% |

### Per-Class Performance Breakdown

| Label | Precision | Recall | F1-Score | Support (# Test Samples) |
| :--- | :---: | :---: | :---: | :---: |
| `pants-fire` | **60.9%** | **57.6%** | **59.2%** | 92 |
| `false` | 42.9% | 51.6% | 46.8% | 250 |
| `barely-true` | 47.0% | 40.2% | 43.3% | 214 |
| `half-true` | 39.1% | 49.8% | 43.8% | 267 |
| `mostly-true` | 37.8% | 51.0% | 43.4% | 249 |
| `true` | 91.7% | 15.6% | 26.7% | 211 |

---

## 3. Evaluation Pipeline Steps (`run_pipeline.py`)

When running the evaluation pipeline, the orchestrator executes 5 automated steps:

```
                  test.csv (1,283 samples)
                            │
                            ▼
              1. generate_predictions.py
                 (Batch inference on GPU)
                            │
                            ▼
                     predictions.csv
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
2. evaluate_classifier  3. error_analysis  4. golden_set
   - metrics.json          - error_report     - leakage_report
   - confusion_matrix.png
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                  5. compare_models.py
                     (model_comparison.json)
```

---

## 4. Execution Commands

### Run Full End-to-End Evaluation Pipeline
```bash
python team4_package/ml_classifier/evaluation/run_pipeline.py
# or from root:
python run_evaluation.py
```

### Generate Predictions Only
```bash
python team4_package/ml_classifier/evaluation/generate_predictions.py \
  --model-version classifier-v3.0 \
  --output team4_package/ml_classifier/evaluation/results/classifier-v3.0/predictions.csv
```

### Compute Metrics from Predictions
```bash
python team4_package/ml_classifier/evaluation/evaluate_classifier.py \
  --predictions team4_package/ml_classifier/evaluation/results/classifier-v3.0/predictions.csv \
  --output-dir team4_package/ml_classifier/evaluation/results/classifier-v3.0
```

### Check Split Leakage & Safety
```bash
python -c "
import pandas as pd
from ml_classifier.evaluation.golden_set import check_no_split_overlap, save_safety_report

train = set(pd.read_csv('team4_package/ml_classifier/datasets/processed/train.csv')['id'])
val = set(pd.read_csv('team4_package/ml_classifier/datasets/processed/validation.csv')['id'])
test = pd.read_csv('team4_package/ml_classifier/datasets/processed/test.csv').rename(columns={'id': 'sample_id', 'label': 'truth_label'})
test['difficulty'] = 'standard'
test['topic'] = 'general'

report = check_no_split_overlap(test, train_ids=train, validation_ids=val)
print('Leakage safety check:', 'PASSED' if report['safe'] else 'FAILED')
"
```
