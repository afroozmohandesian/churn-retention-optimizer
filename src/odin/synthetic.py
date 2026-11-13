"""Reproducible synthetic customer generator.

Produces datasets with the same schema as ``data/raw/project_odin_customers.csv``
so the method can be stress-tested at realistic scale. The generating process
is deliberately simple and fully documented:

* ``Predicted_Churn_Probability`` ~ Beta(2.2, 1.8)  — skewed towards at-risk
  customers, mean ≈ 0.55 (matches the original 30-customer sample);
* ``Estimated_CLV`` ~ LogNormal, median ≈ €950, clipped to [€200, €4,000];
* ``Retention_Offer_Cost`` = 6% of CLV ± noise, clipped to [€30, €250],
  rounded to whole euros (larger accounts receive larger offers);
* ``Actual_Churn`` ~ Bernoulli(probability), so the risk scores are perfectly
  calibrated *by construction* — real model scores would not be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_customers(n: int = 1_000, seed: int = 2024) -> pd.DataFrame:
    """Return ``n`` synthetic customers with the Project Odin schema."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)

    probability = np.round(rng.beta(2.2, 1.8, size=n), 2).clip(0.01, 0.99)
    clv = np.round(rng.lognormal(mean=np.log(950), sigma=0.45, size=n)).clip(200, 4_000)
    cost = np.round(clv * 0.06 + rng.normal(0, 25, size=n)).clip(30, 250)
    churn = (rng.random(n) < probability).astype(int)

    width = max(4, len(str(n)))
    return pd.DataFrame(
        {
            "Customer_ID": [f"S{i:0{width}d}" for i in range(1, n + 1)],
            "Actual_Churn": churn,
            "Predicted_Churn_Probability": probability,
            "Estimated_CLV": clv.astype(int),
            "Retention_Offer_Cost": cost.astype(int),
        }
    )


PLANS = {"Basic": 7.99, "Standard": 12.99, "Premium": 17.99, "Family": 22.99}
REGIONS = ["DACH", "UK & Ireland", "Nordics", "Benelux", "Southern Europe"]
CHANNELS = ["Organic", "Paid Search", "Partner Bundle", "Referral"]
GROSS_MARGIN = 0.60


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_customer_base(n: int = 5_000, seed: int = 7) -> pd.DataFrame:
    """Rich synthetic subscriber base for a streaming-style subscription business.

    Besides the five core columns used by the optimiser it carries the
    behavioural and commercial attributes an analyst would segment on. The
    generating process (all parameters documented in ``data/README.md``):

    1. Draw plan, contract, region, acquisition channel, tenure, viewing hours,
       support tickets and failed payments.
    2. Build a *true* churn log-odds from those drivers (monthly contracts,
       low usage, tickets and payment failures raise risk; tenure, annual
       contracts and referrals lower it).
    3. ``Actual_Churn`` ~ Bernoulli(true probability).
    4. ``Predicted_Churn_Probability`` = the true log-odds plus model error, so
       the score is informative but *not* perfectly calibrated, like a real
       churn model.
    5. ``Estimated_CLV`` = monthly fee x gross margin x expected remaining
       lifetime; ``Retention_Offer_Cost`` = a plan-specific incentive in whole euros.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)

    plan = rng.choice(list(PLANS), size=n, p=[0.34, 0.36, 0.20, 0.10])
    fee = np.array([PLANS[p] for p in plan])
    contract = np.where(rng.random(n) < 0.68, "Monthly", "Annual")
    region = rng.choice(REGIONS, size=n, p=[0.30, 0.20, 0.15, 0.15, 0.20])
    channel = rng.choice(CHANNELS, size=n, p=[0.40, 0.25, 0.20, 0.15])
    tenure = np.clip(np.round(rng.gamma(2.0, 9.0, size=n)), 1, 96).astype(int)
    usage = np.round(np.clip(rng.lognormal(np.log(24), 0.65, size=n), 0.5, 180), 1)
    tickets = rng.poisson(0.6, size=n)
    failures = rng.poisson(0.22, size=n)

    z = (
        -1.75
        + 0.95 * (contract == "Monthly")
        - 0.028 * np.minimum(tenure, 48)
        - 0.55 * (np.log(usage) - np.log(24))
        + 0.38 * tickets
        + 0.85 * failures
        + np.select(
            [channel == "Paid Search", channel == "Partner Bundle", channel == "Referral"],
            [0.30, 0.20, -0.35],
            0.0,
        )
        + np.select([plan == "Basic", plan == "Family"], [0.20, -0.30], 0.0)
        + rng.normal(0, 0.25, size=n)
    )
    true_p = _sigmoid(z)
    churn = (rng.random(n) < true_p).astype(int)
    score = np.round(_sigmoid(0.9 * z + 0.08 + rng.normal(0, 0.35, size=n)), 3).clip(0.001, 0.999)

    lifetime_months = np.clip(
        18 + 0.35 * tenure + 12 * (contract == "Annual") + 8 * (plan == "Family"), 12, 72
    )
    clv = np.round(fee * GROSS_MARGIN * lifetime_months).astype(int)
    offer_months = np.select([plan == "Premium", plan == "Family"], [2.5, 2.5], 2.0)
    cost = np.round(fee * offer_months + 3).astype(int)  # free months + €3 handling

    width = len(str(n))
    return pd.DataFrame(
        {
            "Customer_ID": [f"AUR-{i:0{width}d}" for i in range(1, n + 1)],
            "Actual_Churn": churn,
            "Predicted_Churn_Probability": score,
            "Estimated_CLV": clv,
            "Retention_Offer_Cost": cost,
            "Plan": plan,
            "Monthly_Fee": fee,
            "Contract": contract,
            "Region": region,
            "Acquisition_Channel": channel,
            "Tenure_Months": tenure,
            "Viewing_Hours_30d": usage,
            "Support_Tickets_90d": tickets,
            "Payment_Failures_90d": failures,
        }
    )
