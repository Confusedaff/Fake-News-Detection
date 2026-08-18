# ML Evaluation

This folder evaluates the **outputs of Team 4 models**. It does not implement the API, orchestrator, retrieval service, or final verdict aggregation.

## Required prediction CSV

The evaluator expects:

```csv
sample_id,true_label,predicted_label,raw_confidence,calibrated_probability
1,true,false,0.91,0.84
2,false,false,0.88,0.79
```

If class probabilities are available, add:

```text
probability_0
probability_1
...
```

### Important pandas correctness rule

Always load evaluation CSVs with:

```python
pd.read_csv(path, dtype=str, keep_default_na=False)
```

The LIAR dataset contains literal labels such as `"true"` and `"false"`. Without `dtype=str`, pandas can infer them as booleans, which can corrupt comparisons and metrics.

## Evaluation

```bash
python -m ml.evaluation.evaluate_classifier \
  --predictions path/to/predictions.csv \
  --output-dir ml/evaluation/results/classifier-v1.0
```

Outputs include:

```text
metrics.json
predictions.csv
confusion_matrix.png
```

## Golden set

The golden-set loader requires:

```text
sample_id
truth_label
difficulty
topic
```

It also checks duplicate IDs and supports train/validation/test overlap checks.

Never create fake golden labels merely to make an evaluation pass.

## Model comparison

```bash
python -m ml.evaluation.compare_models \
  --model-a ml/evaluation/results/classifier-v1.0/metrics.json \
  --model-b ml/evaluation/results/classifier-v1.1/metrics.json \
  --version-a classifier-v1.0 \
  --version-b classifier-v1.1
```

The comparison is descriptive. It does not automatically declare a winner.

## Calibration

`TemperatureScaler` fits temperature only on a validation/calibration set.

Never fit calibration on the test set.

Save the fitted temperature alongside the model artifact:

```json
{
  "method": "temperature_scaling",
  "temperature": 1.37
}
```

## Expected Team 4 outputs

- Accuracy
- Macro/weighted precision
- Macro/weighted recall
- Macro/weighted F1
- Per-class classification report
- Confusion matrix
- ROC-AUC for an appropriate binary framing
- ECE when class probabilities are available
- Calibration artifacts
- Error analysis
- Golden-set leakage checks
- Version-to-version comparison
