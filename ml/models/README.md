# ML Models

This directory contains **versioned model artifacts and model-loading code** for Team 4.

## Expected artifact

```text
ml/models/artifacts/classifier/classifier-v1.0/
├── model/
├── tokenizer/
├── metadata.json
├── label_mapping.json
└── calibration.json
```

Do not overwrite an existing model version. Create a new version instead.

## Metadata

`metadata.json` should contain at least:

```json
{
  "model_name": "fake-news-classifier",
  "model_version": "classifier-v1.0",
  "base_model": "distilbert-base-uncased",
  "dataset": "LIAR",
  "dataset_version": "liar-v1",
  "task": "fake_news_classification",
  "max_length": 256,
  "random_seed": 42,
  "calibration_method": "temperature_scaling"
}
```

## Loading

Team 1 can use:

```python
from ml.models import load_classifier

classifier = load_classifier("classifier-v1.0")
result = classifier.predict("Example claim")

print(result.label)
print(result.calibrated_probability)
print(result.model_version)
```

The service layer should wrap this object rather than putting API/orchestration logic in `ml/models`.
