"""Several offer types per subscriber: the multiple-choice knapsack.

In the base model every subscriber has exactly one offer. In practice the
retention team chooses *which* offer to send, from a cheap loyalty email to an
expensive annual-plan discount. Each subscriber can receive at most one offer,
and the best combination of (subscriber, offer) pairs must fit the budget:

    max  sum_ij x_ij * ENB_ij
    s.t. sum_ij x_ij * C_ij <= B
         sum_j  x_ij <= 1          for every subscriber i
         x_ij in {0, 1}

with ``ENB_ij = P_i * CLV_i * r_j - C_ij``. This is the *multiple-choice
knapsack problem* (MCKP). It is solved exactly here by dynamic programming in
``O(n * k * B)`` for ``k`` offers per subscriber.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from odin.strategies import _check_budget

_EPS = 1e-9


@dataclass(frozen=True)
class Offer:
    """One offer type.

    ``cost_months`` is the number of free months (priced at the subscriber's
    monthly fee); ``fixed_cost`` is a flat handling cost in euros;
    ``success_rate`` is the assumed probability that the offer prevents churn.
    ``monthly_only`` restricts the offer to subscribers on monthly contracts.
    """

    name: str
    success_rate: float
    cost_months: float = 0.0
    fixed_cost: float = 0.0
    monthly_only: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be between 0 and 1")
        if self.cost_months < 0 or self.fixed_cost < 0:
            raise ValueError("offer costs must be non-negative")


#: Default catalogue. Success rates are scenario assumptions, increasing with cost.
DEFAULT_CATALOGUE: tuple[Offer, ...] = (
    Offer("Loyalty email", success_rate=0.10, fixed_cost=3),
    Offer("One free month", success_rate=0.32, cost_months=1, fixed_cost=3),
    Offer("Two free months", success_rate=0.50, cost_months=2, fixed_cost=3),
    Offer("Annual-plan discount", success_rate=0.62, cost_months=3, fixed_cost=3, monthly_only=True),
)

MENU_COLUMNS = ["Customer_ID", "Offer", "Offer_Cost", "Offer_Success_Rate", "Offer_Net_Benefit"]


def build_offer_menu(data: pd.DataFrame, catalogue: tuple[Offer, ...] = DEFAULT_CATALOGUE) -> pd.DataFrame:
    """Long table with one row per eligible (subscriber, offer) pair.

    Needs ``Monthly_Fee`` to price free months. Without it (the pilot dataset)
    free months are priced from the subscriber's own offer cost, treating the
    current offer as "two free months". ``Contract`` is used for
    ``monthly_only`` offers when present.
    """
    if not catalogue:
        raise ValueError("catalogue must contain at least one offer")
    if "Monthly_Fee" in data.columns:
        fee = data["Monthly_Fee"].astype(float)
    else:
        fee = (data["Retention_Offer_Cost"].astype(float) - 3).clip(lower=1) / 2
    at_risk = data["Predicted_Churn_Probability"] * data["Estimated_CLV"]
    monthly = data["Contract"].eq("Monthly") if "Contract" in data.columns else pd.Series(True, data.index)

    frames = []
    for offer in catalogue:
        eligible = monthly if offer.monthly_only else pd.Series(True, data.index)
        cost = np.round(fee * offer.cost_months + offer.fixed_cost).clip(lower=1).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "Customer_ID": data["Customer_ID"],
                    "Offer": offer.name,
                    "Offer_Cost": cost,
                    "Offer_Success_Rate": offer.success_rate,
                    "Offer_Net_Benefit": at_risk * offer.success_rate - cost,
                }
            )[eligible.to_numpy()]
        )
    menu = pd.concat(frames, ignore_index=True)
    return menu.sort_values(["Customer_ID", "Offer_Cost"], kind="mergesort").reset_index(drop=True)


def _positive_groups(menu: pd.DataFrame):
    positive = menu.loc[menu["Offer_Net_Benefit"] > 0]
    return [
        (cid, grp["Offer_Cost"].astype(int).to_numpy(), grp["Offer_Net_Benefit"].to_numpy(), grp.index)
        for cid, grp in positive.groupby("Customer_ID", sort=True)
    ]


def multi_offer_selection(menu: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Exact MCKP optimum: at most one offer per subscriber, total cost within budget.

    Returns the chosen rows of ``menu`` (one per targeted subscriber).
    Offers with non-positive net benefit are dropped up front.
    """
    capacity = _check_budget(budget)
    groups = _positive_groups(menu)
    if not groups or capacity == 0:
        return menu.iloc[0:0].copy()

    best = np.zeros(capacity + 1)
    choice = np.full((len(groups), capacity + 1), -1, dtype=np.int16)

    for g, (_, costs, values, _) in enumerate(groups):
        new_best = best.copy()
        for j, (cost, value) in enumerate(zip(costs, values)):
            if cost > capacity:
                continue
            candidate = best[: capacity + 1 - cost] + value
            improves = candidate > new_best[cost:] + _EPS
            new_best[cost:] = np.where(improves, candidate, new_best[cost:])
            choice[g, cost:] = np.where(improves, j, choice[g, cost:])
        best = new_best

    chosen_rows = []
    remaining = capacity
    for g in range(len(groups) - 1, -1, -1):
        j = choice[g, remaining]
        if j >= 0:
            _, costs, _, index = groups[g]
            chosen_rows.append(index[j])
            remaining -= int(costs[j])
    return menu.loc[sorted(chosen_rows)].copy()


def best_single_offer_selection(menu: pd.DataFrame, budget: int, offer: str) -> pd.DataFrame:
    """Benchmark: everyone gets the same offer type, chosen optimally by 0/1 knapsack."""
    return multi_offer_selection(menu.loc[menu["Offer"] == offer], budget)


def brute_force_multi_offer(menu: pd.DataFrame, budget: int, max_customers: int = 10) -> pd.DataFrame:
    """Exhaustive search over every offer assignment — used only to verify the DP."""
    capacity = _check_budget(budget)
    groups = _positive_groups(menu)
    if len(groups) > max_customers:
        raise ValueError(f"Too many candidates ({len(groups)}) for brute force.")

    best_value, best_rows = 0.0, []
    options = [[-1, *range(len(costs))] for _, costs, _, _ in groups]
    for assignment in product(*options):
        cost = value = 0.0
        for (_, costs, values, _), j in zip(groups, assignment):
            if j >= 0:
                cost += costs[j]
                value += values[j]
        if cost <= capacity and value > best_value + _EPS:
            best_value = value
            best_rows = [grp[3][j] for grp, j in zip(groups, assignment) if j >= 0]
    return menu.loc[sorted(best_rows)].copy()


def compare_offer_policies(menu: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Expected net benefit of each single-offer policy versus the per-subscriber choice."""
    rows = {}
    for name in menu["Offer"].unique():
        rows[f"Everyone: {name}"] = best_single_offer_selection(menu, budget, name)
    rows["Best offer per subscriber (MCKP)"] = multi_offer_selection(menu, budget)
    table = pd.DataFrame(
        {
            name: {
                "Subscribers Targeted": len(sel),
                "Total Spending (€)": float(sel["Offer_Cost"].sum()),
                "Expected Net Benefit (€)": float(sel["Offer_Net_Benefit"].sum()),
            }
            for name, sel in rows.items()
        }
    ).T
    table.index.name = "Policy"
    return table


def offer_mix(selection: pd.DataFrame) -> pd.DataFrame:
    """How many subscribers receive each offer type, and what it costs."""
    return (
        selection.groupby("Offer")
        .agg(
            Subscribers=("Customer_ID", "size"),
            Spend=("Offer_Cost", "sum"),
            Expected_Net_Benefit=("Offer_Net_Benefit", "sum"),
        )
        .sort_values("Spend", ascending=False)
    )
