"""Customer-level financial logic.

ERV_i = P_i(churn) * CLV_i * r        (expected retained value)
ENB_i = ERV_i - C_i                   (expected net benefit)
"""

from __future__ import annotations

import pandas as pd


def add_expected_value(df: pd.DataFrame, success_rate: float) -> pd.DataFrame:
    """Return a copy of ``df`` with ``Expected_Retained_Value`` and
    ``Expected_Net_Benefit`` columns added for the given success rate."""
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError("success_rate must be between 0 and 1")

    out = df.copy()
    out["Expected_Retained_Value"] = out["Predicted_Churn_Probability"] * out["Estimated_CLV"] * success_rate
    out["Expected_Net_Benefit"] = out["Expected_Retained_Value"] - out["Retention_Offer_Cost"]
    return out


def break_even_success_rate(df: pd.DataFrame) -> pd.Series:
    """Minimum offer success rate at which each customer's offer pays for itself.

    ``r* = C_i / (P_i * CLV_i)``. Values above 1 mean the offer can never break
    even, whatever its effectiveness.
    """
    at_risk_value = df["Predicted_Churn_Probability"] * df["Estimated_CLV"]
    return (df["Retention_Offer_Cost"] / at_risk_value).rename("Break_Even_Success_Rate")
