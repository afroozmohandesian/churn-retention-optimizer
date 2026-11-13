"""Uplift modelling: per-subscriber treatment effects instead of one success rate.

The base model assumes every offer prevents churn with the same probability
``r``. In reality some subscribers are *persuadable*, some would stay anyway,
some leave whatever we do (for example after repeated payment failures), and a
few react badly to being contacted. Uplift modelling estimates, for each
subscriber, the reduction in churn probability caused by the offer:

    uplift_i = P(churn | no offer, x_i) - P(churn | offer, x_i)

It needs a randomised experiment: a past campaign in which a random part of the
eligible base received the offer and the rest was held out. This module

1. simulates such an experiment on synthetic data with a *known* true effect,
2. fits a T-learner (one churn model per arm, difference of predictions),
3. evaluates ranking quality with the Qini curve, and
4. turns the estimated uplift into expected net benefit,
   ``ENB_i = uplift_i * CLV_i - C_i``, so every existing solver can use it.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

CATEGORICAL = ["Plan", "Contract", "Region", "Acquisition_Channel"]
NUMERIC = [
    "Predicted_Churn_Probability",
    "Tenure_Months",
    "Viewing_Hours_30d",
    "Support_Tickets_90d",
    "Payment_Failures_90d",
]
FEATURES = CATEGORICAL + NUMERIC


def true_offer_effect(data: pd.DataFrame) -> np.ndarray:
    """Relative churn reduction caused by the offer in the synthetic world.

    Monthly subscribers and recent joiners respond best; Premium subscribers
    respond weakly; payment-failure churn is largely involuntary, so offers do
    little; heavy viewers who raise support tickets react *negatively*
    ("sleeping dogs"). Returned as a fraction of the baseline churn risk,
    between -0.3 and 0.85.
    """
    effect = (
        0.42
        + 0.18 * (data["Contract"] == "Monthly")
        - 0.006 * np.minimum(data["Tenure_Months"], 40)
        - 0.30 * (data["Plan"] == "Premium")
        + 0.10 * (data["Plan"] == "Basic")
        - 0.22 * np.minimum(data["Payment_Failures_90d"], 2)
        - 0.18 * ((data["Viewing_Hours_30d"] > 45) & (data["Support_Tickets_90d"] >= 2))
    )
    return np.clip(effect.to_numpy(dtype=float), -0.3, 0.85)


def true_uplift(data: pd.DataFrame) -> pd.Series:
    """Absolute churn-probability reduction caused by the offer (ground truth)."""
    baseline = data["Predicted_Churn_Probability"].to_numpy()
    return pd.Series(baseline * true_offer_effect(data), index=data.index, name="True_Uplift")


def simulate_campaign_experiment(data: pd.DataFrame, treat_share: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """A randomised past campaign: ``Treatment`` assigned at random, ``Churned`` observed.

    Churn under control uses the subscriber's baseline risk; churn under
    treatment uses ``baseline * (1 - effect)``.
    """
    if not 0 < treat_share < 1:
        raise ValueError("treat_share must be strictly between 0 and 1")
    missing = sorted(set(FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Uplift features missing from data: {missing}")
    rng = np.random.default_rng(seed)
    baseline = data["Predicted_Churn_Probability"].to_numpy()
    treatment = (rng.random(len(data)) < treat_share).astype(int)
    p = np.where(treatment == 1, baseline * (1 - true_offer_effect(data)), baseline)
    out = data.copy()
    out["Treatment"] = treatment
    out["Churned"] = (rng.random(len(data)) < np.clip(p, 0, 1)).astype(int)
    return out


def _churn_model():
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=2)), NUMERIC),
        ]
    )
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=2_000))


class TLearner:
    """Two-model uplift estimator: separate churn models for treated and control."""

    def __init__(self) -> None:
        self.treated_model = _churn_model()
        self.control_model = _churn_model()
        self.fitted = False

    def fit(self, experiment: pd.DataFrame) -> TLearner:
        for arm, model in ((1, self.treated_model), (0, self.control_model)):
            part = experiment.loc[experiment["Treatment"] == arm]
            if part["Churned"].nunique() < 2:
                raise ValueError("Each arm needs both churners and non-churners to fit.")
            model.fit(part[FEATURES], part["Churned"])
        self.fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Estimated uplift: P(churn | control) - P(churn | treated)."""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        p_control = self.control_model.predict_proba(data[FEATURES])[:, 1]
        p_treated = self.treated_model.predict_proba(data[FEATURES])[:, 1]
        return pd.Series(p_control - p_treated, index=data.index, name="Estimated_Uplift")


def add_uplift_value(data: pd.DataFrame, uplift: pd.Series | np.ndarray) -> pd.DataFrame:
    """Expected value columns from a per-subscriber uplift instead of a constant rate.

    ``Expected_Retained_Value = uplift * CLV`` and
    ``Expected_Net_Benefit = Expected_Retained_Value - cost``, the same columns
    :func:`odin.economics.add_expected_value` produces, so every solver works.
    """
    out = data.copy()
    out["Estimated_Uplift"] = np.asarray(uplift, dtype=float)
    out["Expected_Retained_Value"] = out["Estimated_Uplift"] * out["Estimated_CLV"]
    out["Expected_Net_Benefit"] = out["Expected_Retained_Value"] - out["Retention_Offer_Cost"]
    return out


def qini_curve(uplift: Iterable[float], treatment: Iterable[int], churned: Iterable[int]) -> pd.DataFrame:
    """Cumulative churns prevented when targeting by descending predicted uplift.

    At each targeting depth, ``Incremental_Saves = control_churn_rate *
    n_treated - treated_churns`` (scaled to the treated count). A random
    ranking gives the straight line from 0 to the overall effect.
    """
    frame = pd.DataFrame({"u": list(uplift), "t": list(treatment), "y": list(churned)})
    frame = frame.sort_values("u", ascending=False, kind="mergesort").reset_index(drop=True)
    n_t = frame["t"].cumsum()
    n_c = (1 - frame["t"]).cumsum()
    y_t = (frame["y"] * frame["t"]).cumsum()
    y_c = (frame["y"] * (1 - frame["t"])).cumsum()
    saves = (y_c / n_c.replace(0, np.nan)).fillna(0) * n_t - y_t
    curve = pd.DataFrame(
        {"Share_Targeted": (np.arange(len(frame)) + 1) / len(frame), "Incremental_Saves": saves.to_numpy()}
    )
    origin = pd.DataFrame({"Share_Targeted": [0.0], "Incremental_Saves": [0.0]})
    return pd.concat([origin, curve]).reset_index(drop=True)


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the Qini curve and the random-targeting diagonal (higher is better)."""
    x = curve["Share_Targeted"].to_numpy()
    y = curve["Incremental_Saves"].to_numpy()
    random_line = x * y[-1]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz
    return float(trapezoid(y - random_line, x))


def realised_value(
    data: pd.DataFrame, selected_ids: Iterable[str], uplift_truth: pd.Series | None = None
) -> dict[str, float]:
    """Value of a portfolio under the *true* offer effect (synthetic data only)."""
    truth = true_uplift(data) if uplift_truth is None else uplift_truth
    mask = data["Customer_ID"].isin(list(selected_ids)).to_numpy()
    spend = float(data.loc[mask, "Retention_Offer_Cost"].sum())
    retained = float((truth[mask] * data.loc[mask, "Estimated_CLV"]).sum())
    return {
        "Selected Customers": int(mask.sum()),
        "Total Spending (€)": spend,
        "True Retained Value (€)": retained,
        "True Net Benefit (€)": retained - spend,
        "Expected Churns Prevented": float(truth[mask].sum()),
    }
