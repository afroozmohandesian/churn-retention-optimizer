from pathlib import Path

import pandas as pd
import pytest

from odin import add_expected_value, load_customers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "project_odin_customers.csv"


@pytest.fixture
def raw() -> pd.DataFrame:
    return load_customers(DATA)


@pytest.fixture
def customers(raw: pd.DataFrame) -> pd.DataFrame:
    return add_expected_value(raw, 0.5)


def make(rows):
    """Tiny helper: rows of (id, churn, prob, clv, cost)."""
    return pd.DataFrame(
        rows,
        columns=[
            "Customer_ID",
            "Actual_Churn",
            "Predicted_Churn_Probability",
            "Estimated_CLV",
            "Retention_Offer_Cost",
        ],
    )
