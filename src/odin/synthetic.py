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
