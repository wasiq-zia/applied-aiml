# ============================================================
# File : train.py
# Description :
#   Fit classifiers, save pipelines under model/, print metrics.
# Inputs :
#   Train/test helpers from preprocess; MODEL_SPECS
# Outputs :
#   *.joblib under model/, test_data.csv, metrics on stdout
# Notes :
#   - Run: python -m model.train  (from Assignment2 root)
#   - Scoring UI lives in app.py
# ============================================================

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from model.evaluate import (
    build_comparison_table,
    compute_classification_metrics,
    format_comparison_markdown,
    format_metrics,
)
from model.preprocess import (
    MODEL_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    build_preprocessor,
    prepare_train_test,
    save_preprocessor,
    save_test_data,
)

MODEL_SPECS: dict[str, object] = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    # Calibrated linear SVM so predict_proba / AUC are available
    "svm": CalibratedClassifierCV(
        LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000,
            dual="auto",
        ),
        cv=3,
    ),
}

MODEL_PATHS = {name: MODEL_DIR / f"{name}.joblib" for name in MODEL_SPECS}


# ============================================================
# Function : ml_build_pipeline
# Description :
#   Wrap a classifier with the shared preprocessor in a Pipeline.
# Inputs :
#   estimator - unfitted sklearn classifier (or calibrated wrapper)
# Outputs :
#   Unfitted Pipeline with steps: preprocess, model
# Notes :
#   - Fresh preprocessor from build_preprocessor()
# ============================================================
def ml_build_pipeline(estimator) -> Pipeline:
    # print("WAS_DB:: ml_build_pipeline", type(estimator).__name__)
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


# ============================================================
# Function : ml_evaluate_pipeline
# Description :
#   Score a fitted pipeline on held-out features and labels.
# Inputs :
#   pipeline - fitted sklearn Pipeline
#   x_test - held-out feature frame
#   y_test - held-out target labels
# Outputs :
#   Dict of classification metrics
# Notes :
#   - Uses predict_proba when available for AUC
# ============================================================
def ml_evaluate_pipeline(pipeline: Pipeline, x_test, y_test) -> dict[str, float]:
    y_pred = pipeline.predict(x_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(x_test)
    # print("WAS_DB:: ml_evaluate_pipeline has_proba", y_proba is not None)
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    # print("WAS_DB:: ml_evaluate_pipeline", list(metrics.keys()))
    return metrics


# ============================================================
# Function : ml_train_all_models
# Description :
#   Fit all classifiers on one stratified split and save joblibs.
# Inputs :
#   None (uses MODEL_SPECS / prepare_train_test)
# Outputs :
#   Dict mapping model key -> metrics dict; writes model/ artifacts
# Notes :
#   - Writes test_data.csv once from the shared split
#   - Saves preprocessor from the first fitted pipeline
# ============================================================
def ml_train_all_models() -> dict[str, dict[str, float]]:
    x_train, x_test, y_train, y_test = prepare_train_test()
    # print("WAS_DB:: ml_train_all_models", x_train.shape, x_test.shape)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_test_data(x_test, y_test)

    all_metrics: dict[str, dict[str, float]] = {}
    first_preprocessor = None

    for name, estimator in MODEL_SPECS.items():
        print(f"\nTraining {name} ...")
        pipeline = ml_build_pipeline(estimator)
        pipeline.fit(x_train, y_train)

        model_path = MODEL_PATHS[name]
        joblib.dump(pipeline, model_path)
        # print("WAS_DB:: ml_train_all_models saved", name, model_path)

        if first_preprocessor is None:
            first_preprocessor = pipeline.named_steps["preprocess"]
            save_preprocessor(first_preprocessor, PREPROCESSOR_PATH)

        metrics = ml_evaluate_pipeline(pipeline, x_test, y_test)
        all_metrics[name] = metrics
        print(f"Saved model: {model_path}")
        print(format_metrics(metrics))

    return all_metrics


# ============================================================
# Function : ml_main
# Description :
#   Train all models and print the comparison table.
# Inputs :
#   None
# Outputs :
#   None (prints metrics table and markdown to stdout)
# Notes :
#   - Script entrypoint
# ============================================================
def ml_main() -> None:
    print("Training classifiers")
    all_metrics = ml_train_all_models()
    # print("WAS_DB:: ml_main", list(all_metrics.keys()))
    table = build_comparison_table(all_metrics)
    # print("WAS_DB:: ml_main table", table.shape)
    print("\nComparison table:")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMarkdown table:\n")
    print(format_comparison_markdown(table))
    print("\nDone. Models in:", MODEL_DIR)


if __name__ == "__main__":
    ml_main()
