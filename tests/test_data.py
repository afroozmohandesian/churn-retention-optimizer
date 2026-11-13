import pandas as pd
import pytest

from odin import DataValidationError, validate_customers
from tests.conftest import make


def test_original_dataset_is_valid(raw):
    assert raw.shape == (30, 5)
    assert raw["Customer_ID"].is_unique
    assert raw["Actual_Churn"].sum() == 15


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns="Estimated_CLV"), "Missing required columns"),
        (lambda d: d.assign(Predicted_Churn_Probability=1.5), "between 0 and 1"),
        (lambda d: d.assign(Actual_Churn=2), "only 0 and 1"),
        (lambda d: d.assign(Retention_Offer_Cost=-5), "Offer costs are positive"),
        (lambda d: d.assign(Retention_Offer_Cost=10.5), "whole euros"),
        (lambda d: pd.concat([d, d.iloc[[0]]]), "unique"),
        (lambda d: d.assign(Estimated_CLV="abc"), "Non-numeric"),
    ],
)
def test_invalid_data_is_rejected(mutate, message):
    good = make([("A", 1, 0.5, 1000, 50), ("B", 0, 0.2, 800, 40)])
    with pytest.raises(DataValidationError, match=message):
        validate_customers(mutate(good))


def test_missing_value_is_rejected():
    df = make([("A", 1, None, 1000, 50)])
    with pytest.raises(DataValidationError, match="No missing values"):
        validate_customers(df)
