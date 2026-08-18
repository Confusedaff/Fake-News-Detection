# Models Module — Registry, Artifact Packaging & Loader

This module manages **immutable model releases, artifact validation, and runtime inference loading** for Team 4.

---

## 1. Directory Structure

```
models/
├── artifacts/                           # Immutable versioned release artifacts
│   └── classifier/
│       └── classifier-v3.0/             # Production candidate artifact
│           ├── model/                   # Fine-tuned PyTorch model
│           │   ├── config.json          # Architecture and classification head config
│           │   ├── model.safetensors    # Fine-tuned transformer weights (498.6 MB)
│           │   ├── tokenizer.json       # Fast tokenizer vocabulary & merges
│           │   ├── tokenizer_config.json
│           │   └── special_tokens_map.json
│           ├── tokenizer/               # Dedicated tokenizer directory
│           │   ├── tokenizer.json
│           │   ├── tokenizer_config.json
│           │   └── special_tokens_map.json
│           ├── label_mapping.json       # 6-class LIAR ID <-> Label mappings
│           ├── labels.json              # Class index list
│           ├── calibration.json         # Fitted temperature scaling parameter (T = 1.2267)
│           ├── manifest.json            # Full training run provenance & hyperparameters
│           └── metadata.json            # Model card & architecture metadata
├── loader.py                            # Public ClassifierModel loader & prediction engine
└── registry.py                          # Filesystem ModelRegistry & validation logic
```

---

## 2. Model Immutability Rule

> [!IMPORTANT]
> **Never overwrite an existing released model version.**
> Once a version such as `classifier-v3.0` is published, its weights, configuration, and metadata are frozen. Subsequent improvements must be published as a new version tag (e.g. `classifier-v3.1` or `classifier-v4.0`).

---

## 3. Public Team 4 Access Interface (`loader.py`)

Downstream services (Team 1, the Orchestrator, and API layer) consume the classifier via `load_classifier()` without needing to know Hugging Face internal paths or PyTorch tensor management:

```python
from ml_classifier.models.loader import load_classifier
from packages.schemas import ClassifierRequest, ClassifierResponse

# 1. Load the versioned classifier
classifier = load_classifier("classifier-v3.0")

# 2. Predict on raw text or Pydantic request model
claim = "The government spent $5 billion on the new healthcare program."
result = classifier.predict(claim)

# 3. Retrieve raw dictionary or Pydantic response
dict_output = result.as_dict()
pydantic_output: ClassifierResponse = result.to_response()

print(f"Predicted Label: {result.label} (ID: {result.label_id})")
print(f"Calibrated Confidence: {result.calibrated_probability:.4f}")
print(f"Probabilities: {result.probabilities}")
```

---

## 4. Output Contract Schema

The classifier returns a 6-class probability distribution adhering to the LIAR taxonomy:

```json
{
  "text": "The government spent $5 billion on the new healthcare program.",
  "prediction": {
    "label": "half-true",
    "label_id": 3,
    "confidence": 0.2034
  },
  "probabilities": {
    "pants-fire": 0.1129,
    "false": 0.1476,
    "barely-true": 0.1560,
    "half-true": 0.2034,
    "mostly-true": 0.1825,
    "true": 0.1976
  },
  "raw_logit_confidence": 0.2117,
  "calibrated_probability": 0.2034,
  "model_version": "classifier-v3.0"
}
```

---

## 5. Model Registry Operations (`registry.py`)

`ModelRegistry` provides programmatic discovery and schema validation of stored model releases:

```python
from ml_classifier.models.registry import ModelRegistry

reg = ModelRegistry()

# List available versions
versions = reg.list_versions("classifier")
print(versions)  # ['classifier-v1.0', 'classifier-v3', 'classifier-v3.0']

# Validate artifact completeness
errors = reg.validate("classifier", "classifier-v3.0")
if not errors:
    print("Artifact is valid and ready for production.")
else:
    print(f"Validation errors: {errors}")
```

### Validation Checks
- Presence of `model/` directory with `config.json` and weights.
- Presence of tokenizer assets (`tokenizer.json` or `vocab.json`).
- Presence of `label_mapping.json` (or `labels.json`).
- Presence of `metadata.json` (or `manifest.json`).
