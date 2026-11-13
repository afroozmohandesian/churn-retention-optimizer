"""Loading and validating the customer dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: list[str] = [
    "Customer_ID",
    "Actual_Churn",
    "Predicted_Churn_Probability",
    "Estimated_CLV",
    "Retention_Offer_Cost",
]


class DataValidationError(ValueError):
    """Raised when the customer dataset violates a structural rule."""


def validation_report(df: pd.DataFrame) -> dict[str, bool]:
    """Return a ``{rule: passed}`` mapping of every structural check."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        return {f"Required column present: {c}": c not in missing for c in REQUIRED_COLUMNS}

    return {
        "All required columns are present": True,
        "Dataset is not empty": len(df) > 0,
        "Customer IDs are unique": df["Customer_ID"].is_unique,
        "No missing values": not df[REQUIRED_COLUMNS].isna().any().any(),
        "Actual churn contains only 0 and 1": df["Actual_Churn"].isin([0, 1]).all(),
        "Churn probabilities are between 0 and 1": df["Predicted_Churn_Probability"].between(0, 1).all(),
        "CLV values are positive": (df["Estimated_CLV"] > 0).all(),
        "Offer costs are positive": (df["Retention_Offer_Cost"] > 0).all(),
        "Offer costs are whole euros": bool(np.allclose(df["Retention_Offer_Cost"] % 1, 0)),
    }


def validate_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, run every structural check and return a clean copy.

    Raises
    ------
    DataValidationError
        If any check fails. The message lists every failing rule.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    clean = df[REQUIRED_COLUMNS].copy()
    clean["Customer_ID"] = clean["Customer_ID"].astype(str)
    try:
        for column in REQUIRED_COLUMNS[1:]:
            clean[column] = pd.to_numeric(clean[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"Non-numeric value found: {exc}") from exc

    failures = [rule for rule, passed in validation_report(clean).items() if not passed]
    if failures:
        raise DataValidationError("Data validation failed: " + "; ".join(failures))

    clean["Actual_Churn"] = clean["Actual_Churn"].astype(int)
    clean["Retention_Offer_Cost"] = clean["Retention_Offer_Cost"].astype(int)
    return clean


def load_customers(path: str | Path) -> pd.DataFrame:
    """Read a customer CSV and return the validated DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset: {path}")
    return validate_customers(pd.read_csv(path))
