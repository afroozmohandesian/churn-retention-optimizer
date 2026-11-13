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


def score_diagnostics(data: pd.DataFrame, bins: int = 10) -> tuple[dict[str, float], pd.DataFrame]:
    """Discrimination and calibration of the churn score against observed churn.

    Returns ``(summary, calibration_table)`` where ``summary`` holds the ROC AUC
    (rank-based, Mann-Whitney), the Brier score and the base churn rate, and the
    table compares mean predicted vs observed churn per score decile.
    """
    y = data["Actual_Churn"].to_numpy()
    p = data["Predicted_Churn_Probability"].to_numpy()
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    ranks = pd.Series(p).rank(method="average").to_numpy()
    auc = (
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos and n_neg else float("nan")
    )

    frame = pd.DataFrame({"p": p, "y": y})
    frame["Score Band"] = pd.qcut(frame["p"].rank(method="first"), q=min(bins, len(frame)), labels=False) + 1
    calibration = (
        frame.groupby("Score Band")
        .agg(Customers=("y", "size"), Mean_Predicted=("p", "mean"), Observed_Churn_Rate=("y", "mean"))
        .reset_index()
    )
    summary = {
        "ROC AUC": float(auc),
        "Brier Score": float(((p - y) ** 2).mean()),
        "Base Churn Rate": float(y.mean()),
        "Mean Predicted Probability": float(p.mean()),
    }
    return summary, calibration


def segment_allocation(data: pd.DataFrame, selected_ids: Iterable[str], by: str) -> pd.DataFrame:
    """How a selected portfolio is distributed across the segments of column ``by``."""
    frame = data.assign(_selected=data["Customer_ID"].isin(list(selected_ids)))
    chosen = frame[frame["_selected"]]
    table = pd.DataFrame(
        {
            "Customers": frame.groupby(by).size(),
            "Selected": chosen.groupby(by).size(),
            "Spend (€)": chosen.groupby(by)["Retention_Offer_Cost"].sum(),
            "Expected Net Benefit (€)": chosen.groupby(by)["Expected_Net_Benefit"].sum(),
        }
    ).fillna(0)
    table["Selection Rate"] = table["Selected"] / table["Customers"]
    table["Share of Spend"] = (
        table["Spend (€)"] / table["Spend (€)"].sum() if table["Spend (€)"].sum() else 0.0
    )
    return table.sort_values("Expected Net Benefit (€)", ascending=False)
