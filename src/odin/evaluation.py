"""Metrics for comparing customer selections."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping

import pandas as pd

from odin.strategies import STRATEGIES, StrategyFn


def evaluate_selection(data: pd.DataFrame, selected_ids: Iterable[str]) -> dict[str, float]:
    """Financial and retrospective metrics for one set of selected customers."""
    selected = data[data["Customer_ID"].isin(list(selected_ids))]
    total_churners = int(data["Actual_Churn"].sum())
    captured = int(selected["Actual_Churn"].sum())
    spending = float(selected["Retention_Offer_Cost"].sum())
    net_benefit = float(selected["Expected_Net_Benefit"].sum())

    return {
        "Selected Customers": len(selected),
        "Total Spending (€)": spending,
        "Expected Retained Value (€)": float(selected["Expected_Retained_Value"].sum()),
        "Expected Net Benefit (€)": net_benefit,
        "ROI": net_benefit / spending if spending else 0.0,
        "Actual Churners Captured": captured,
        "Actual Churn Capture Rate": captured / total_churners if total_churners else 0.0,
    }


def run_strategies(
    data: pd.DataFrame,
    budget: int,
    strategies: Mapping[str, StrategyFn] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Run every strategy and return ``{name: selected_ids}``."""
    strategies = strategies or STRATEGIES
    results: dict[str, list[str]] = {}
    for name, fn in strategies.items():
        if "seed" in inspect.signature(fn).parameters:
            results[name] = fn(data, budget, seed=seed)  # type: ignore[call-arg]
        else:
            results[name] = fn(data, budget)
    return results


def compare_strategies(data: pd.DataFrame, selections: Mapping[str, list[str]]) -> pd.DataFrame:
    """One row of metrics per strategy."""
    table = pd.DataFrame({name: evaluate_selection(data, ids) for name, ids in selections.items()}).T
    table.index.name = "Strategy"
    return table


def optimality_gap(comparison: pd.DataFrame, reference: str = "Knapsack Optimization") -> pd.Series:
    """Share of the optimal net benefit each strategy achieves (1.0 = optimal)."""
    optimum = comparison.loc[reference, "Expected Net Benefit (€)"]
    return (comparison["Expected Net Benefit (€)"] / optimum).rename("Share of Optimum")
