# ============================================================
# File : preprocess.py
# Description :
#   Load raw data, build the preprocessor, stratified split, and
#   write test_data.csv plus preprocessor.joblib.
# Inputs :
#   Dataset archive / raw CSV; FEATURE_COLUMNS and TARGET_COLUMN
# Outputs :
#   Splits, ColumnTransformer, saved CSV/joblib
# Notes :
#   - Run: python -m model.preprocess  (from Assignment2 root)
#   - Shared by train.py, evaluate.py, and app.py
# ============================================================

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "bank-full.csv"
ALT_RAW_CSV_PATH = DATA_DIR / "bank.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

BANK_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
)

TARGET_COLUMN = "y"

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2


# download_raw_data
# Download the dataset archive and extract the raw CSV.
def download_raw_data(force: bool = False) -> Path:
    # print("WAS_DB:: download_raw_data", force, RAW_CSV_PATH.exists())
    if RAW_CSV_PATH.exists() and not force:
        return RAW_CSV_PATH
    if ALT_RAW_CSV_PATH.exists() and not force:
        return ALT_RAW_CSV_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from:\n  {BANK_ZIP_URL}")
    with urlopen(BANK_ZIP_URL, timeout=120) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        member_name = None
        for name in archive.namelist():
            if name.endswith("bank-full.csv"):
                member_name = name
                break
        if member_name is None:
            raise FileNotFoundError("bank-full.csv not found inside bank.zip")

        with archive.open(member_name) as source, RAW_CSV_PATH.open("wb") as dest:
            dest.write(source.read())

    print(f"Saved raw dataset to: {RAW_CSV_PATH}")
    # print("WAS_DB:: download_raw_data path", RAW_CSV_PATH)
    return RAW_CSV_PATH


# load_raw_data
# Load the raw CSV (semicolon-separated).
def load_raw_data(csv_path: Path | None = None) -> pd.DataFrame:
    if csv_path is not None:
        path = csv_path
    elif RAW_CSV_PATH.exists():
        path = RAW_CSV_PATH
    elif ALT_RAW_CSV_PATH.exists():
        path = ALT_RAW_CSV_PATH
    else:
        path = download_raw_data()

    if not path.exists():
        path = download_raw_data()

    frame = pd.read_csv(path, sep=";")
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {sorted(missing)}")
    # print("WAS_DB:: load_raw_data", frame.shape)
    return frame


# split_features_target
# Split a frame into features and binary target.
def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = frame[FEATURE_COLUMNS].copy()
    target = frame[TARGET_COLUMN].map({"yes": 1, "no": 0}).astype(int)
    if target.isna().any():
        raise ValueError("Target column contains values other than yes/no.")
    # print("WAS_DB:: split_features_target", int((target == 1).sum()), int((target == 0).sum()))
    return features, target


# build_preprocessor
# Scale numerics and one-hot encode categoricals.
def build_preprocessor() -> ColumnTransformer:
    # print("WAS_DB:: build_preprocessor", len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES))
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# prepare_train_test
# Stratified train/test split on features and target.
def prepare_train_test(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    frame = load_raw_data()
    features, target = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    # print("WAS_DB:: prepare_train_test", x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test


# save_test_data
# Write held-out test CSV with features and yes/no labels.
def save_test_data(x_test: pd.DataFrame, y_test: pd.Series, path: Path = TEST_DATA_PATH) -> Path:
    test_frame = x_test.copy()
    test_frame[TARGET_COLUMN] = y_test.map({1: "yes", 0: "no"})
    test_frame.to_csv(path, index=False)
    # print("WAS_DB:: save_test_data", path, len(test_frame))
    return path


# save_preprocessor
# Persist a fitted preprocessor under model/.
def save_preprocessor(preprocessor: ColumnTransformer, path: Path = PREPROCESSOR_PATH) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    # print("WAS_DB:: save_preprocessor", path)
    return path


# run_prepare
# Fit preprocessor on train split; write test CSV and artifact.
def run_prepare() -> dict:
    x_train, x_test, y_train, y_test = prepare_train_test()

    preprocessor = build_preprocessor()
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    preprocessor_path = save_preprocessor(preprocessor)
    test_data_path = save_test_data(x_test, y_test)

    summary = {
        "n_rows_full": int(len(x_train) + len(x_test)),
        "n_features": int(len(FEATURE_COLUMNS)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_transformed_features": int(x_train_transformed.shape[1]),
        "class_balance_full": {
            "no": int((pd.concat([y_train, y_test]) == 0).sum()),
            "yes": int((pd.concat([y_train, y_test]) == 1).sum()),
        },
        "class_balance_test": {
            "no": int((y_test == 0).sum()),
            "yes": int((y_test == 1).sum()),
        },
        "test_data_path": str(test_data_path),
        "preprocessor_path": str(preprocessor_path),
        "x_train_transformed_shape": tuple(x_train_transformed.shape),
        "x_test_transformed_shape": tuple(x_test_transformed.shape),
    }
    # print("WAS_DB:: run_prepare", summary["x_train_transformed_shape"])
    return summary


# main
# Run prepare and print a short summary.
def main() -> None:
    summary = run_prepare()
    # print("WAS_DB:: main", summary["n_train"], summary["n_test"])
    print("Prepare complete")
    print(f"Full rows:              {summary['n_rows_full']}")
    print(f"Feature columns:        {summary['n_features']}")
    print(f"Train / test sizes:     {summary['n_train']} / {summary['n_test']}")
    print(f"Transformed width:      {summary['n_transformed_features']}")
    print(f"Class balance (full):   {summary['class_balance_full']}")
    print(f"Class balance (test):   {summary['class_balance_test']}")
    print(f"Wrote test data:        {summary['test_data_path']}")
    print(f"Wrote preprocessor:     {summary['preprocessor_path']}")
    print(f"Train matrix shape:     {summary['x_train_transformed_shape']}")
    print(f"Test matrix shape:      {summary['x_test_transformed_shape']}")


if __name__ == "__main__":
    main()
