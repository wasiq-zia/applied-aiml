# ============================================================
# File : evaluate.py
# Description :
#   Classification metrics, comparison table helpers, and CLI
#   scoring of saved joblib pipelines on test_data.csv.
# Inputs :
#   Predictions / probabilities, or saved models + test CSV
# Outputs :
#   Metric dicts, DataFrames, markdown table, CLI output
# Notes :
#   - Run: python -m model.evaluate  (from Assignment2 root)
#   - Does not retrain models
# ============================================================

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model.preprocess import FEATURE_COLUMNS, MODEL_DIR, TARGET_COLUMN, TEST_DATA_PATH

METRIC_COLUMNS = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}


# ============================================================
# Function : compute_classification_metrics
# Description :
#   Compute Accuracy, AUC, Precision, Recall, F1, and MCC.
# Outputs :
#   Dict of metric name -> float (AUC may be nan)
# ============================================================
def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_proba is None:
        metrics["AUC"] = float("nan")
    else:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 2:
            positive_scores = y_proba[:, 1]
        else:
            positive_scores = y_proba
        metrics["AUC"] = float(roc_auc_score(y_true, positive_scores))

    # print("WAS_DB:: compute_classification_metrics", list(metrics.keys()))
    return metrics


# ============================================================
# Function : confusion_matrix_report
# Description :
#   Build a confusion matrix and text classification report.
# Outputs :
#   Dict with confusion_matrix and classification_report
# ============================================================
def confusion_matrix_report(y_true, y_pred) -> dict[str, Any]:
    report = {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["no", "yes"],
            zero_division=0,
        ),
    }
    # print("WAS_DB:: confusion_matrix_report", report["confusion_matrix"].shape)
    return report


# ============================================================
# Function : format_metrics
# Description :
#   Format a metrics dict as aligned text lines.
# Outputs :
#   Multi-line string
# ============================================================
def format_metrics(metrics: dict[str, float]) -> str:
    lines = []
    for name in METRIC_COLUMNS:
        value = metrics[name]
        lines.append(f"  {name:<10} {value:.4f}")
    # print("WAS_DB:: format_metrics", len(lines))
    return "\n".join(lines)


# ============================================================
# Function : build_comparison_table
# Description :
#   Build a one-row-per-model comparison table.
# Outputs :
#   DataFrame with model name and metric columns
# ============================================================
def build_comparison_table(all_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_key, display_name in MODEL_DISPLAY_NAMES.items():
        if model_key not in all_metrics:
            raise KeyError(f"Missing metrics for model: {model_key}")
        row = {"ML Model Name": display_name}
        for metric_name in METRIC_COLUMNS:
            row[metric_name] = all_metrics[model_key][metric_name]
        rows.append(row)
    table = pd.DataFrame(rows)
    # print("WAS_DB:: build_comparison_table", len(table))
    return table


# ============================================================
# Function : format_comparison_markdown
# Description :
#   Render the comparison table as a markdown table.
# Outputs :
#   Markdown table string
# ============================================================
def format_comparison_markdown(table: pd.DataFrame, decimals: int = 4) -> str:
    formatted = table.copy()
    for metric_name in METRIC_COLUMNS:
        formatted[metric_name] = formatted[metric_name].map(
            lambda value: f"{value:.{decimals}f}"
        )
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body_lines = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in formatted.to_numpy()
    ]
    text = "\n".join([header, separator, *body_lines])
    # print("WAS_DB:: format_comparison_markdown", len(text))
    return text


# ============================================================
# Function : collect_metrics_from_saved_models
# Description :
#   Load saved pipelines and test CSV; build the comparison table.
# Outputs :
#   Comparison DataFrame
# ============================================================
def collect_metrics_from_saved_models(
    test_data_path=TEST_DATA_PATH,
    model_dir=MODEL_DIR,
) -> pd.DataFrame:
    test = pd.read_csv(test_data_path)
    x_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN].map({"yes": 1, "no": 0}).astype(int)
    # print("WAS_DB:: collect_metrics_from_saved_models rows", len(test))

    all_metrics: dict[str, dict[str, float]] = {}
    for model_key in MODEL_DISPLAY_NAMES:
        model_path = model_dir / f"{model_key}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found: {model_path}")
        pipeline = joblib.load(model_path)
        y_pred = pipeline.predict(x_test)
        y_proba = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None
        all_metrics[model_key] = compute_classification_metrics(y_test, y_pred, y_proba)

    # print("WAS_DB:: collect_metrics_from_saved_models", list(all_metrics.keys()))
    return build_comparison_table(all_metrics)


# ============================================================
# Function : main
# Description :
#   CLI entry: print comparison metrics from saved models.
# Outputs :
#   None
# ============================================================
def main() -> None:
    print("Comparing saved models")
    table = collect_metrics_from_saved_models()
    # print("WAS_DB:: main", table.shape)
    print("\nComparison table:")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMarkdown table:\n")
    print(format_comparison_markdown(table))
    print("\nNo output/ directory created.")


if __name__ == "__main__":
    main()
