"""Scenario analysis: how much does the recommendation depend on assumptions?"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from odin.economics import add_expected_value
from odin.evaluation import evaluate_selection
from odin.strategies import knapsack_selection


def run_scenario(data: pd.DataFrame, success_rate: float, budget: int) -> tuple[list[str], dict]:
    """Recompute economics for ``success_rate`` and solve the knapsack for ``budget``."""
    scenario = add_expected_value(data, success_rate)
    ids = knapsack_selection(scenario, budget)
    return ids, evaluate_selection(scenario, ids)


def _row(rate: float, budget: int, ids: list[str], metrics: dict) -> dict:
    return {
        "Assumed Success Rate": rate,
        "Budget (€)": budget,
        "Selected Customers": metrics["Selected Customers"],
        "Spending (€)": metrics["Total Spending (€)"],
        "Expected Net Benefit (€)": metrics["Expected Net Benefit (€)"],
        "Selected IDs": ", ".join(ids),
    }


def success_rate_sensitivity(data: pd.DataFrame, rates: Iterable[float], budget: int) -> pd.DataFrame:
    """Vary the success rate, hold the budget fixed."""
    return pd.DataFrame([_row(r, budget, *run_scenario(data, r, budget)) for r in rates])


def budget_sensitivity(data: pd.DataFrame, budgets: Iterable[int], success_rate: float) -> pd.DataFrame:
    """Vary the budget, hold the success rate fixed."""
    rows = [_row(success_rate, b, *run_scenario(data, success_rate, b)) for b in budgets]
    return pd.DataFrame(rows)[
        [
            "Budget (€)",
            "Assumed Success Rate",
            "Selected Customers",
            "Spending (€)",
            "Expected Net Benefit (€)",
            "Selected IDs",
        ]
    ]


def sensitivity_grid(data: pd.DataFrame, rates: Iterable[float], budgets: Iterable[int]) -> pd.DataFrame:
    """Expected net benefit of the optimal portfolio for every (rate, budget) pair.

    Returns a pivot table: rows = success rate, columns = budget.
    """
    rates, budgets = list(rates), list(budgets)
    records = []
    for rate in rates:
        scenario = add_expected_value(data, rate)
        for budget in budgets:
            ids = knapsack_selection(scenario, budget)
            records.append(
                {
                    "Assumed Success Rate": rate,
                    "Budget (€)": budget,
                    "Expected Net Benefit (€)": evaluate_selection(scenario, ids)["Expected Net Benefit (€)"],
                }
            )
    return pd.DataFrame(records).pivot(
        index="Assumed Success Rate", columns="Budget (€)", values="Expected Net Benefit (€)"
    )


def portfolio_stability(data: pd.DataFrame, rates: Iterable[float], budget: int) -> pd.DataFrame:
    """How often each customer is selected across a range of success rates."""
    rates = list(rates)
    counts: dict[str, int] = {}
    for rate in rates:
        ids, _ = run_scenario(data, rate, budget)
        for cid in ids:
            counts[cid] = counts.get(cid, 0) + 1
    table = pd.DataFrame({"Customer_ID": list(counts), "Times Selected": list(counts.values())})
    table["Selection Frequency"] = table["Times Selected"] / len(rates)
    return table.sort_values(["Selection Frequency", "Customer_ID"], ascending=[False, True]).reset_index(
        drop=True
    )
