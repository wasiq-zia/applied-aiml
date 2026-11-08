# ============================================================
# File : app.py
# Description :
#   Streamlit UI: load saved pipelines, score an uploaded test CSV,
#   show metrics and a confusion matrix / classification report.
# Inputs :
#   Uploaded test CSV (features + target); joblib files in model/
# Outputs :
#   Interactive Streamlit page
# Notes :
#   - Run: streamlit run app.py
#   - Does not retrain; run python -m model.train first
# ============================================================

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report

from model.evaluate import (
    METRIC_COLUMNS,
    MODEL_DISPLAY_NAMES,
    build_comparison_table,
    compute_classification_metrics,
    confusion_matrix_report,
)
from model.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODEL_DIR,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

st.set_page_config(
    page_title="Term Deposit Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent

APP_CSS = """
<style>
    .stApp { background: #FFFFFF; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }
    h1 { letter-spacing: -0.02em; color: #111111 !important; }
    [data-testid="stSidebar"] { background: #F0F2F6; border-right: 1px solid #E2E5EB; }
    [data-testid="stSidebar"] h2 { color: #0066CC !important; font-size: 1.05rem !important; }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E5EB;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 2px rgba(17, 17, 17, 0.04);
    }
    [data-testid="stMetricLabel"] { color: #444444; }
    [data-testid="stMetricValue"] { color: #111111; }
    div[data-testid="stHorizontalBlock"] { gap: 0.75rem; }
    .app-subtitle { color: #444444; font-size: 1.02rem; margin-top: -0.4rem; margin-bottom: 0.75rem; }
    .app-divider { border: none; border-top: 1px solid #E2E5EB; margin: 0.4rem 0 1.2rem 0; }
    .section-label { color: #0066CC; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.35rem; }
    .footer-note { color: #666666; font-size: 0.85rem; margin-top: 1.5rem; }
</style>
"""


# ============================================================
# Function : apply_theme
# Description :
#   Inject lightweight CSS for a clean analytics look.
# Outputs :
#   None
# ============================================================
def apply_theme() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


# ============================================================
# Function : available_models
# Description :
#   List saved models that exist on disk.
# Outputs :
#   Dict of display name -> joblib Path
# ============================================================
def available_models() -> dict[str, Path]:
    found = {}
    for key, display_name in MODEL_DISPLAY_NAMES.items():
        path = MODEL_DIR / f"{key}.joblib"
        if path.exists():
            found[display_name] = path
    # print("WAS_DB:: available_models", len(found), list(found.keys()))
    return found


# ============================================================
# Function : load_pipeline
# Description :
#   Load a saved Pipeline from disk (cached across reruns).
# Outputs :
#   Fitted Pipeline
# ============================================================
@st.cache_resource
def load_pipeline(model_path_str: str):
    # print("WAS_DB:: load_pipeline", model_path_str)
    return joblib.load(model_path_str)


# ============================================================
# Function : normalize_target
# Description :
#   Map yes/no or 0/1 labels to binary integers.
# Outputs :
#   Series of 0/1 ints
# ============================================================
def normalize_target(series: pd.Series) -> pd.Series:
    # print("WAS_DB:: normalize_target in", series.astype(str).str.lower().unique()[:8].tolist())
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(int)
        if set(values.unique()) <= {0, 1}:
            return values
        raise ValueError("Numeric target must contain only 0 and 1.")

    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    mapped = cleaned.map(mapping)
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:5]
        raise ValueError(f"Unrecognized target values (examples): {list(bad)}")
    # print("WAS_DB:: normalize_target out", mapped.unique().tolist())
    return mapped.astype(int)


# ============================================================
# Function : validate_uploaded_frame
# Description :
#   Check required columns and split features / target.
# Outputs :
#   (feature DataFrame, binary target Series)
# ============================================================
def validate_uploaded_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # print("WAS_DB:: validate_uploaded_frame", frame.shape)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in frame.columns]
    if missing:
        # print("WAS_DB:: validate_uploaded_frame missing", missing)
        raise ValueError(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing)
            + ". Include all input features and the target column 'y'."
        )
    features = frame[FEATURE_COLUMNS].copy()
    target = normalize_target(frame[TARGET_COLUMN])
    return features, target


# ============================================================
# Function : score_pipeline
# Description :
#   Predict and compute metrics plus a confusion report (no fit).
# Outputs :
#   Dict with metrics, y_pred, and report
# ============================================================
def score_pipeline(pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    report = confusion_matrix_report(y_test, y_pred)
    # print("WAS_DB:: score_pipeline", list(metrics.keys()))
    return {"metrics": metrics, "y_pred": y_pred, "report": report}


# ============================================================
# Function : schema_table
# Description :
#   Build a small expected-schema table for the empty state.
# Outputs :
#   DataFrame with Feature and Role columns
# ============================================================
def schema_table() -> pd.DataFrame:
    rows = [{"Feature": col, "Role": "numeric input"} for col in NUMERIC_FEATURES]
    rows.extend({"Feature": col, "Role": "categorical input"} for col in CATEGORICAL_FEATURES)
    rows.append({"Feature": TARGET_COLUMN, "Role": "target (yes/no)"})
    return pd.DataFrame(rows)


# ============================================================
# Function : classification_report_frame
# Description :
#   Convert sklearn classification report dict to a tidy table.
# Outputs :
#   DataFrame of precision / recall / f1 / support by label
# ============================================================
def classification_report_frame(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=["no", "yes"],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in ("no", "yes", "accuracy", "macro avg", "weighted avg"):
        if label not in report:
            continue
        if label == "accuracy":
            rows.append(
                {
                    "label": "accuracy",
                    "precision": report["accuracy"],
                    "recall": report["accuracy"],
                    "f1-score": report["accuracy"],
                    "support": report["weighted avg"]["support"],
                }
            )
            continue
        entry = report[label]
        rows.append(
            {
                "label": label,
                "precision": entry["precision"],
                "recall": entry["recall"],
                "f1-score": entry["f1-score"],
                "support": entry["support"],
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Function : render_confusion_matrix
# Description :
#   Plot a confusion-matrix heatmap in the Streamlit page.
# Outputs :
#   None (st.pyplot)
# ============================================================
def render_confusion_matrix(matrix) -> None:
    # print("WAS_DB:: render_confusion_matrix", getattr(matrix, "shape", None))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    blue_cmap = sns.blend_palette(["#E8F1FB", "#0066CC"], as_cmap=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=blue_cmap,
        cbar=False,
        linewidths=0.6,
        linecolor="#E2E5EB",
        annot_kws={"color": "#111111"},
        xticklabels=["Predicted no", "Predicted yes"],
        yticklabels=["Actual no", "Actual yes"],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors="#111111")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    st.pyplot(fig, width="content")
    plt.close(fig)


# ============================================================
# Function : render_metric_grid
# Description :
#   Show classification metrics in two rows of three.
# Outputs :
#   None
# ============================================================
def render_metric_grid(metrics: dict[str, float]) -> None:
    row1 = METRIC_COLUMNS[:3]
    row2 = METRIC_COLUMNS[3:]
    cols1 = st.columns(3)
    for column, name in zip(cols1, row1):
        value = metrics[name]
        column.metric(name, "—" if pd.isna(value) else f"{value:.4f}")
    cols2 = st.columns(3)
    for column, name in zip(cols2, row2):
        value = metrics[name]
        column.metric(name, "—" if pd.isna(value) else f"{value:.4f}")


# ============================================================
# Function : highlight_selected_model
# Description :
#   Style helper: highlight the selected model row in comparison.
# Outputs :
#   CSS string list for pandas Styler
# ============================================================
def highlight_selected_model(row: pd.Series, selected_name: str) -> list[str]:
    if row.get("ML Model Name") == selected_name:
        return ["background-color: #E8F1FB; color: #111111;"] * len(row)
    return [""] * len(row)


# ============================================================
# Function : main
# Description :
#   Streamlit entry: upload CSV, pick a model, show metrics and
#   optional all-model comparison.
# Outputs :
#   None
# ============================================================
def main() -> None:
    apply_theme()

    st.title("Term Deposit Prediction")
    st.markdown(
        '<p class="app-subtitle">Score pre-trained classifiers on a held-out test CSV. '
        "Models are loaded from disk — nothing is retrained in this app.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="app-divider" />', unsafe_allow_html=True)

    models = available_models()
    if not models:
        st.error(f"No saved models found in `{MODEL_DIR}`. Run `python -m model.train` first.")
        return

    with st.sidebar:
        st.header("Evaluation setup")
        uploaded = st.file_uploader(
            "Upload test CSV",
            type=["csv"],
            help="Must include all 16 input features and target column y.",
        )
        selected_name = st.selectbox("Model", options=list(models.keys()))
        show_all = st.checkbox("Compare all models", value=True)

        st.markdown("---")
        st.caption(
            f"**{len(models)}** models ready  \n"
            f"**{len(FEATURE_COLUMNS)}** input features  \n"
            "Pre-trained pipelines (evaluation only)"
        )
        with st.expander("How to use"):
            st.markdown(
                "1. Upload a test CSV (or use `test_data.csv` from this project).\n"
                "2. Choose a model to inspect metrics and the confusion matrix.\n"
                "3. Optionally compare all five models on the same upload."
            )

    if uploaded is None:
        left, right = st.columns((1.1, 1.0), gap="large")
        with left:
            st.markdown('<div class="section-label">Get started</div>', unsafe_allow_html=True)
            st.markdown(
                "Upload a labeled test CSV from the sidebar to evaluate the classifiers. "
                "A ready-made holdout file is available as **`test_data.csv`** in this project folder."
            )
            st.info("No data is used to retrain models. This app only scores saved pipelines.")
        with right:
            st.markdown('<div class="section-label">Expected schema</div>', unsafe_allow_html=True)
            st.dataframe(schema_table(), use_container_width=True, hide_index=True, height=320)
        st.markdown(
            '<p class="footer-note">Binary classification · five models · evaluation only</p>',
            unsafe_allow_html=True,
        )
        return

    try:
        frame = pd.read_csv(uploaded)
        x_test, y_test = validate_uploaded_frame(frame)
    except Exception as exc:  # noqa: BLE001 — show schema errors in the UI
        st.error(f"Could not use this CSV: {exc}")
        return

    # print("WAS_DB:: main", selected_name, len(frame))
    yes_count = int((y_test == 1).sum())
    no_count = int((y_test == 0).sum())
    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(frame):,}")
    c2.metric("Subscribed (yes)", f"{yes_count:,}")
    c3.metric("Not subscribed (no)", f"{no_count:,}")

    pipeline = load_pipeline(str(models[selected_name]))
    result = score_pipeline(pipeline, x_test, y_test)

    st.markdown("---")
    st.markdown('<div class="section-label">Selected model</div>', unsafe_allow_html=True)
    st.subheader(selected_name)
    render_metric_grid(result["metrics"])

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Confusion matrix")
        render_confusion_matrix(result["report"]["confusion_matrix"])
    with right:
        st.markdown("##### Classification report")
        report_frame = classification_report_frame(y_test, result["y_pred"])
        st.dataframe(
            report_frame.style.format(
                {
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    if show_all:
        st.markdown("---")
        st.markdown('<div class="section-label">Model comparison</div>', unsafe_allow_html=True)
        st.subheader("All models on this upload")
        st.caption("Higher is better for all metrics shown.")
        all_metrics = {}
        key_by_display = {display: key for key, display in MODEL_DISPLAY_NAMES.items()}
        progress = st.progress(0.0, text="Scoring models…")
        for index, (display_name, path) in enumerate(models.items(), start=1):
            pipe = load_pipeline(str(path))
            scored = score_pipeline(pipe, x_test, y_test)
            all_metrics[key_by_display[display_name]] = scored["metrics"]
            progress.progress(index / len(models), text=f"Scored {display_name}")
        progress.empty()
        comparison = build_comparison_table(all_metrics)
        # print("WAS_DB:: main comparison", len(comparison))
        styled = comparison.style.format(
            {name: "{:.4f}" for name in METRIC_COLUMNS}
        ).apply(highlight_selected_model, axis=1, selected_name=selected_name)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("Preview uploaded rows"):
        st.dataframe(frame.head(20), use_container_width=True, hide_index=True)

    st.markdown(
        '<p class="footer-note">Binary classification · five models · evaluation only</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
