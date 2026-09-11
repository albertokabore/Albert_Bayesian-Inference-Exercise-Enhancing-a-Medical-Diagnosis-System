"""Loading and preprocessing for the UCI Cleveland heart-disease dataset.

Source: Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J.,
Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International
application of a new probability algorithm for the diagnosis of coronary
artery disease. American Journal of Cardiology, 64, 304-310.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Human-readable category maps, per heart-disease.names.
CP_LABELS = {1: "typical_angina", 2: "atypical_angina", 3: "non_anginal", 4: "asymptomatic"}
RESTECG_LABELS = {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"}
SLOPE_LABELS = {1: "upsloping", 2: "flat", 3: "downsloping"}
THAL_LABELS = {3: "normal", 6: "fixed_defect", 7: "reversible_defect"}

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]


def load_raw(data_path: str | Path) -> pd.DataFrame:
    """Load the raw processed.cleveland.data file (missing values are '?')."""
    df = pd.read_csv(data_path, header=None, names=COLUMN_NAMES, na_values="?")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and binarize the diagnosis target.

    ``ca`` and ``thal`` are the only columns with missing values in this
    dataset (6 rows total out of 303). Given the small number, rows are
    imputed with the column median/mode rather than dropped, to preserve
    the full validation set.
    """
    df = df.copy()

    df["ca"] = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].mode().iloc[0])

    # Original target `num` is 0 (no disease) .. 4 (severity). The
    # standard benchmark task, and the one used by Detrano et al., is
    # binary: presence (1-4) vs. absence (0) of disease.
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    df["cp"] = df["cp"].astype(int).map(CP_LABELS)
    df["restecg"] = df["restecg"].astype(int).map(RESTECG_LABELS)
    df["slope"] = df["slope"].astype(int).map(SLOPE_LABELS)
    df["thal"] = df["thal"].astype(int).map(THAL_LABELS)
    df["sex"] = df["sex"].astype(int)
    df["fbs"] = df["fbs"].astype(int)
    df["exang"] = df["exang"].astype(int)
    df["ca"] = df["ca"].astype(int)

    return df.reset_index(drop=True)


def load_clean(data_path: str | Path) -> pd.DataFrame:
    return clean(load_raw(data_path))


def build_design_matrix(
    df: pd.DataFrame, reference_categories: dict[str, str] | None = None
) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """One-hot encode categoricals (treatment coding) and return (X, y, references).

    The reference (dropped) category for each categorical feature is the
    clinically "lowest risk" level, so that every fitted coefficient reads
    as an increase in log-odds relative to a healthy baseline:
      - cp: typical_angina (classic, but least associated with obstructive
        disease in this cohort; asymptomatic presentation is the high-risk
        category, see Diamond & Forrester, 1979, NEJM 300:1350-8).
      - restecg: normal
      - slope: upsloping (lowest risk per Detrano et al., 1989)
      - thal: normal
    """
    if reference_categories is None:
        reference_categories = {
            "cp": "typical_angina",
            "restecg": "normal",
            "slope": "upsloping",
            "thal": "normal",
        }

    y = df["target"]
    X = df.drop(columns=["target"]).copy()

    for col, ref in reference_categories.items():
        cats = [c for c in X[col].unique() if c != ref]
        dummies = pd.get_dummies(X[col], prefix=col)
        keep_cols = [f"{col}_{c}" for c in cats]
        X = pd.concat([X.drop(columns=[col]), dummies[keep_cols].astype(int)], axis=1)

    return X, y, reference_categories
