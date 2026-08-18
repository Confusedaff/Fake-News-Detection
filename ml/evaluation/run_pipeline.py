"""
team4_package/ml_classifier/evaluation/run_pipeline.py

Complete Evaluation Orchestrator for Team 4:
1. Generates test split predictions with calibrated probabilities for classifier-v3.0
2. Computes comprehensive evaluation metrics (Accuracy, Macro/Weighted F1, Recall, Precision, ECE)
3. Generates confusion matrix visualization (confusion_matrix.png)
4. Executes detailed error analysis (error_report.json)
5. Executes dataset leakage and split overlap checks (leakage_report.json)
6. Performs version comparison across trained checkpoints

Run:
    python -m ml_classifier.evaluation.run_pipeline
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ml_classifier.evaluation.error_analysis import build_error_report, load_predictions, save_error_report
from ml_classifier.evaluation.evaluate_classifier import main as evaluate_main
from ml_classifier.evaluation.generate_predictions import generate_test_predictions
from ml_classifier.evaluation.golden_set import check_no_split_overlap, save_safety_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("run_pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def run_evaluation_pipeline(model_version: str = "classifier-v3.0"):
    log.info("=" * 60)
    log.info(f"STARTING EVALUATION PIPELINE FOR {model_version}")
    log.info("=" * 60)

    results_dir = (
        PROJECT_ROOT
        / "team4_package"
        / "ml_classifier"
        / "evaluation"
        / "results"
        / model_version
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / "predictions.csv"
    test_csv_path = PROJECT_ROOT / "team4_package" / "ml_classifier" / "datasets" / "processed" / "test.csv"
    train_csv_path = PROJECT_ROOT / "team4_package" / "ml_classifier" / "datasets" / "processed" / "train.csv"
    val_csv_path = PROJECT_ROOT / "team4_package" / "ml_classifier" / "datasets" / "processed" / "validation.csv"

    # Step 1: Generate predictions
    log.info("\n--- STEP 1: Generating Test Split Predictions ---")
    generate_test_predictions(
        model_version=model_version,
        test_csv_path=test_csv_path,
        output_csv_path=predictions_path,
    )

    # Step 2: Calculate evaluation metrics and confusion matrix
    log.info("\n--- STEP 2: Computing Evaluation Metrics & Confusion Matrix ---")
    orig_argv = sys.argv
    sys.argv = [
        "evaluate_classifier.py",
        "--predictions", str(predictions_path),
        "--output-dir", str(results_dir),
        "--n-bins", "10",
    ]
    try:
        evaluate_main()
    finally:
        sys.argv = orig_argv

    # Step 3: Error Analysis
    log.info("\n--- STEP 3: Executing Error Analysis ---")
    df_preds = load_predictions(predictions_path)
    error_report = build_error_report(df_preds, confidence_threshold=0.80)
    error_report_path = results_dir / "error_report.json"
    save_error_report(error_report, error_report_path)
    log.info(f"Saved error report to {error_report_path}")
    log.info(
        f"Total Test Examples: {error_report['total_examples']} | "
        f"Errors: {error_report['errors']} | "
        f"Error Rate: {error_report['error_rate']:.4f} | "
        f"High-Conf Wrong: {error_report['high_confidence_wrong_count']}"
    )

    # Step 4: Dataset Integrity & Leakage Check
    log.info("\n--- STEP 4: Checking Split Integrity & Leakage ---")
    train_df = pd.read_csv(train_csv_path, dtype=str, keep_default_na=False)
    val_df = pd.read_csv(val_csv_path, dtype=str, keep_default_na=False)
    test_df = pd.read_csv(test_csv_path, dtype=str, keep_default_na=False)

    train_ids = set(train_df["id"].tolist())
    val_ids = set(val_df["id"].tolist())
    test_ids = set(test_df["id"].tolist())

    golden_mock = test_df.rename(columns={"id": "sample_id", "label": "truth_label"})
    golden_mock["difficulty"] = "standard"
    golden_mock["topic"] = golden_mock["subject"] if "subject" in golden_mock.columns else "general"

    leakage_report = check_no_split_overlap(
        golden_mock,
        train_ids=train_ids,
        validation_ids=val_ids,
    )
    leakage_path = results_dir / "leakage_report.json"
    save_safety_report(leakage_report, leakage_path)
    log.info(f"Split overlap safety check: {'PASSED (No Leakage)' if leakage_report['safe'] else 'FAILED'}")

    # Step 5: Version Comparison across all iterations
    log.info("\n--- STEP 5: Version-to-Version Comparison ---")
    registry_dir = PROJECT_ROOT / "team4_package" / "ml_classifier" / "training" / "registry"
    all_versions = []
    if registry_dir.exists():
        for p in sorted(registry_dir.iterdir()):
            if p.is_dir() and (p / "manifest.json").exists():
                manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
                val_m = manifest.get("validation_metrics", {})
                calib = manifest.get("calibration", {})
                all_versions.append({
                    "version": manifest.get("model_version", p.name),
                    "base_model": manifest.get("base_model", "unknown"),
                    "val_accuracy": round(val_m.get("eval_accuracy", 0), 4) if val_m.get("eval_accuracy") else None,
                    "val_f1_macro": round(val_m.get("eval_f1_macro", 0), 4) if val_m.get("eval_f1_macro") else None,
                    "val_loss": round(val_m.get("eval_loss", 0), 4) if val_m.get("eval_loss") else None,
                    "calib_temp": round(calib.get("temperature", 1.0), 4) if calib.get("temperature") else None,
                    "ece_after": round(calib.get("ece_after", 0), 4) if calib.get("ece_after") else None,
                })

    comparison_df = pd.DataFrame(all_versions)
    comparison_path = results_dir / "model_comparison.json"
    comparison_df.to_json(comparison_path, orient="records", indent=2)
    print("\n--- MODEL VERSIONS COMPARISON TABLE ---")
    print(comparison_df.to_string(index=False))

    # Read final test metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    log.info("\n" + "=" * 60)
    log.info("FINAL TEST EVALUATION METRICS (classifier-v3.0 on held-out test split):")
    log.info(f"  Test Accuracy:          {metrics_data.get('accuracy', 0):.4f}")
    log.info(f"  Test Macro F1:          {metrics_data.get('f1_macro', 0):.4f}")
    log.info(f"  Test Weighted F1:       {metrics_data.get('f1_weighted', 0):.4f}")
    log.info(f"  Test Macro Precision:   {metrics_data.get('precision_macro', 0):.4f}")
    log.info(f"  Test Macro Recall:      {metrics_data.get('recall_macro', 0):.4f}")
    log.info(f"  Expected Calib Error:   {metrics_data.get('ece', 0):.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_evaluation_pipeline("classifier-v3.0")
