# Training Module — RoBERTa Fine-Tuning & Calibration

This module implements the **offline model training, early stopping, and confidence calibration** pipeline for the 6-way LIAR sequence classifier.

---

## 1. Directory Structure

```
training/
├── config.yaml              # Hyperparameters (learning rate, batch size, epochs, scheduler)
├── dataset.py               # PyTorch LiarDataset with structured metadata prefixing
├── train.py                 # Training script with Hugging Face Trainer & early stopping
├── calibrate.py             # Temperature scaling calibration module
├── checkpoints/             # Training step checkpoints
│   └── best/                # Top-performing checkpoint saved during training
└── registry/                # Local training registry with run manifests
    ├── latest.json          # Points to the best candidate release
    └── classifier-v3/       # Top candidate training provenance
        └── manifest.json    # Complete hyperparameter, loss, and validation metrics record
```

---

## 2. Model Training Architecture (`train.py`)

The classifier fine-tunes `roberta-base` (125M parameters, 12 attention heads, hidden dimension 768) with a 6-unit sequence classification head.

```
Statement Text + Speaker Metadata
               │
               ▼
       RoBERTa Tokenizer (max_len = 256)
               │
               ▼
         RoBERTa Base
               │
               ▼
     Classification Head (Linear 768 -> 6)
               │
               ▼
          Raw Logits
```

### Hyperparameters (`config.yaml`)

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| **Base Model** | `roberta-base` | Robust masked-language pretrained representation |
| **Max Sequence Length** | `256` | Captures speaker context, credibility history, and claim |
| **Learning Rate** | `1e-5` | Stable fine-tuning for RoBERTa transformer layers |
| **Optimizer** | `AdamW` | Weight decay regularized optimizer (`weight_decay=0.01`) |
| **Batch Size** | `16` per device | Fits comfortably in GPU memory with gradient accumulation |
| **LR Scheduler** | `linear` | Warmup for first 10% steps followed by linear decay |
| **Early Stopping** | `patience = 3` | Monitored on `eval_f1_macro` over `validation.csv` |

---

## 3. Dataset Loader & Metadata Prefixing (`dataset.py`)

`LiarDataset` constructs a structured input sequence prepending the speaker's credibility history counts and background:

```
speaker history pants-fire=18 false=30 barely-true=30 half-true=42 mostly-true=20. barack obama (democrat) on economy: The unemployment rate has dropped for 5 consecutive quarters.
```

---

## 4. Temperature Scaling Calibration (`calibrate.py`)

Neural networks optimized with cross-entropy loss are frequently overconfident. Temperature scaling learns a single scalar parameter $T > 0$ on the **calibration split** (`calibration.csv`) to rescale logits before softmax:

$$\hat{p}_i = \frac{e^{z_i / T}}{\sum_{j=1}^K e^{z_j / T}}$$

- **Optimization**: Minimizes Negative Log-Likelihood (NLL) on `calibration.csv` using L-BFGS.
- **Fitted Temperature**: $T = 1.2267$
- **Effect**: Softens overconfident probabilities without changing the predicted class ($z_i > z_j \iff z_i/T > z_j/T$).

---

## 5. Candidate Checkpoints Comparison

| Version | Base Model | Validation Accuracy | Validation Macro F1 | Loss | Temperature ($T$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `classifier-v1.0` | `roberta-base` | 38.97% | 37.70% | 1.4169 | 1.5723 |
| `classifier-v2.0` | `roberta-base` | 28.19% | 28.52% | 1.8120 | 1.5949 |
| **`classifier-v3.0`** *(Selected)* | `roberta-base` | **47.20%** | **44.46%** | **1.2798** | **1.2267** |
| `classifier-v4.0` | `roberta-base` | 46.73% | 44.05% | 1.2895 | 1.1283 |
| `classifier-v5.0` | `roberta-base` | 27.88% | 20.60% | 1.6862 | 1.5530 |

---

## 6. Execution Commands

### Train Model
```bash
python team4_package/ml_classifier/training/train.py --config team4_package/ml_classifier/training/config.yaml
```

### Calibrate Model
```bash
python team4_package/ml_classifier/training/calibrate.py \
  --checkpoint team4_package/ml_classifier/training/checkpoints/best \
  --calibration-csv team4_package/ml_classifier/datasets/processed/calibration.csv
```
