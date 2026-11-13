"""Monte Carlo simulation of campaign outcomes.

Expected values hide risk. Here we simulate the campaign many times:

* each selected customer churns (absent intervention) with probability ``P_i``;
* if they would churn, the offer saves them with probability ``r``;
* a saved customer preserves ``CLV_i``; every offer costs ``C_i`` regardless.

The result is a distribution of *realised* net benefit, from which we read the
probability of losing money and downside percentiles.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def simulate_net_benefit(
    data: pd.DataFrame,
    selected_ids: Iterable[str],
    success_rate: float,
    runs: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Return ``runs`` simulated realised net benefits for one portfolio."""
    selected = data[data["Customer_ID"].isin(list(selected_ids))]
    if selected.empty:
        return np.zeros(runs)

    rng = np.random.default_rng(seed)
    p = selected["Predicted_Churn_Probability"].to_numpy()
    clv = selected["Estimated_CLV"].to_numpy()
    cost = selected["Retention_Offer_Cost"].to_numpy().sum()

    would_churn = rng.random((runs, len(p))) < p
    saved = would_churn & (rng.random((runs, len(p))) < success_rate)
    return saved.astype(float) @ clv - cost


def summarise_simulations(simulations: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Risk summary per strategy: mean, spread, downside and probability of loss."""
    rows = {}
    for name, sims in simulations.items():
        rows[name] = {
            "Mean Net Benefit (€)": sims.mean(),
            "Std Dev (€)": sims.std(ddof=1) if len(sims) > 1 else 0.0,
            "5th Percentile (€)": np.percentile(sims, 5),
            "Median (€)": np.percentile(sims, 50),
            "95th Percentile (€)": np.percentile(sims, 95),
            "Probability of Loss": float((sims < 0).mean()),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "Strategy"
    return table
