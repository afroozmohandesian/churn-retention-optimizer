"""Customer-selection strategies for a budget-constrained retention campaign.

Every strategy takes a DataFrame that already has an ``Expected_Net_Benefit``
column (see :func:`odin.economics.add_expected_value`) and an integer budget,
and returns the list of selected ``Customer_ID`` values.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np
import pandas as pd

_EPS = 1e-9


def _check_budget(budget: int) -> int:
    if budget < 0 or int(budget) != budget:
        raise ValueError("Budget must be a non-negative whole number.")
    return int(budget)


def _greedy_fill(ordered: pd.DataFrame, budget: int) -> list[str]:
    """Walk customers in the given order, taking each one whose cost still fits."""
    selected: list[str] = []
    spent = 0
    for customer_id, cost in zip(ordered["Customer_ID"], ordered["Retention_Offer_Cost"]):
        cost = int(cost)
        if spent + cost <= budget:
            selected.append(customer_id)
            spent += cost
    return selected


def random_selection(data: pd.DataFrame, budget: int, seed: int = 42) -> list[str]:
    """Baseline: shuffle customers reproducibly and fill the budget greedily."""
    budget = _check_budget(budget)
    return _greedy_fill(data.sample(frac=1, random_state=seed), budget)


def churn_ranked_selection(data: pd.DataFrame, budget: int) -> list[str]:
    """Risk-first rule: highest churn probability first, until the budget is used."""
    budget = _check_budget(budget)
    ranked = data.sort_values("Predicted_Churn_Probability", ascending=False, kind="mergesort")
    return _greedy_fill(ranked, budget)


def value_density_selection(data: pd.DataFrame, budget: int) -> list[str]:
    """Economic heuristic: best net benefit *per euro spent* first.

    Only customers with a positive expected net benefit are considered. This is
    the classic greedy approximation to the knapsack problem — fast and usually
    close to optimal, but not guaranteed to be.
    """
    budget = _check_budget(budget)
    eligible = data.loc[data["Expected_Net_Benefit"] > 0].copy()
    eligible["_density"] = eligible["Expected_Net_Benefit"] / eligible["Retention_Offer_Cost"]
    ranked = eligible.sort_values(["_density", "Customer_ID"], ascending=[False, True])
    return _greedy_fill(ranked, budget)


def knapsack_selection(data: pd.DataFrame, budget: int) -> list[str]:
    """Exact 0/1 Knapsack optimum via dynamic programming.

    Maximises ``sum(ENB_i * x_i)`` subject to ``sum(C_i * x_i) <= budget`` and
    ``x_i in {0, 1}``. Runs in ``O(n * budget)`` time; the inner loop over the
    budget is vectorised with NumPy so thousands of customers stay fast.

    Customers with non-positive expected net benefit are excluded up front: they
    can only lower the objective, and selecting nobody is always feasible.
    """
    capacity = _check_budget(budget)
    eligible = data.loc[data["Expected_Net_Benefit"] > 0].sort_values("Customer_ID").reset_index(drop=True)
    if eligible.empty or capacity == 0:
        return []

    costs = eligible["Retention_Offer_Cost"].astype(int).to_numpy()
    values = eligible["Expected_Net_Benefit"].astype(float).to_numpy()
    ids = eligible["Customer_ID"].tolist()

    best = np.zeros(capacity + 1)  # best[b] = optimum using the items seen so far
    take = np.zeros((len(ids), capacity + 1), dtype=bool)

    for i, (cost, value) in enumerate(zip(costs, values)):
        if cost > capacity:
            continue
        candidate = best[: capacity + 1 - cost] + value
        improves = candidate > best[cost:] + _EPS
        take[i, cost:] = improves
        best[cost:] = np.where(improves, candidate, best[cost:])

    selected: list[str] = []
    remaining = capacity
    for i in range(len(ids) - 1, -1, -1):
        if take[i, remaining]:
            selected.append(ids[i])
            remaining -= int(costs[i])
    return selected[::-1]


def brute_force_selection(data: pd.DataFrame, budget: int, max_items: int = 22) -> list[str]:
    """Exhaustive search over every subset — used only to verify the DP solver.

    Refuses to run on more than ``max_items`` positive-benefit customers because
    the search space grows as ``2**n``.
    """
    capacity = _check_budget(budget)
    eligible = data.loc[data["Expected_Net_Benefit"] > 0].sort_values("Customer_ID")
    if len(eligible) > max_items:
        raise ValueError(f"Too many candidates ({len(eligible)}) for brute force.")

    items = list(
        zip(eligible["Customer_ID"], eligible["Retention_Offer_Cost"], eligible["Expected_Net_Benefit"])
    )
    best_value, best_ids = 0.0, []
    for size in range(1, len(items) + 1):
        for combo in combinations(items, size):
            cost = sum(int(c) for _, c, _ in combo)
            if cost > capacity:
                continue
            value = sum(v for _, _, v in combo)
            if value > best_value + _EPS:
                best_value, best_ids = value, [cid for cid, _, _ in combo]
    return best_ids


StrategyFn = Callable[[pd.DataFrame, int], list[str]]

STRATEGIES: dict[str, StrategyFn] = {
    "Random": random_selection,
    "Highest Churn Probability": churn_ranked_selection,
    "Value Density (Greedy)": value_density_selection,
    "Knapsack Optimization": knapsack_selection,
}
