"""Benchmark: how far are the heuristics from the optimum, and how fast is the solver?

Writes
    outputs/tables/heuristic_gap_by_budget.csv
    outputs/tables/solver_runtime.csv

Run from the repository root:  python scripts/benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odin import (  # noqa: E402
    add_expected_value,
    churn_ranked_selection,
    knapsack_selection,
    load_customers,
    value_density_selection,
)
from odin.synthetic import generate_customers  # noqa: E402


def _value(df: pd.DataFrame, ids: list[str]) -> float:
    return float(df.loc[df["Customer_ID"].isin(ids), "Expected_Net_Benefit"].sum())


def heuristic_gap(df: pd.DataFrame, budgets: range) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        optimum = _value(df, knapsack_selection(df, budget))
        greedy = _value(df, value_density_selection(df, budget))
        churn = _value(df, churn_ranked_selection(df, budget))
        rows.append(
            {
                "Budget (€)": budget,
                "Knapsack (€)": round(optimum, 2),
                "Value Density (€)": round(greedy, 2),
                "Highest Churn (€)": round(churn, 2),
                "Value Density % of Optimum": round(greedy / optimum, 4) if optimum else 1.0,
                "Highest Churn % of Optimum": round(churn / optimum, 4) if optimum else 1.0,
            }
        )
    return pd.DataFrame(rows)


def runtime(sizes: list[int], budget_per_customer: int = 5) -> pd.DataFrame:
    rows = []
    for n in sizes:
        df = add_expected_value(generate_customers(n, seed=1), 0.5)
        budget = n * budget_per_customer
        start = time.perf_counter()
        knapsack_selection(df, budget)
        rows.append(
            {"Customers": n, "Budget (€)": budget, "Runtime (s)": round(time.perf_counter() - start, 3)}
        )
    return pd.DataFrame(rows)


def main() -> None:
    out = ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    customers = add_expected_value(load_customers(ROOT / "data/raw/project_odin_customers.csv"), 0.5)
    gap = heuristic_gap(customers, range(50, 1001, 50))
    gap.to_csv(out / "heuristic_gap_by_budget.csv", index=False)
    print("Heuristic gap on the 30-customer dataset (success rate 50%):")
    print(gap.to_string(index=False))
    print(f"\nWorst value-density result: {gap['Value Density % of Optimum'].min():.1%} of optimum")
    print(f"Worst highest-churn result: {gap['Highest Churn % of Optimum'].min():.1%} of optimum")

    rt = runtime([100, 1_000, 5_000, 10_000])
    rt.to_csv(out / "solver_runtime.csv", index=False)
    print("\nKnapsack solver runtime:")
    print(rt.to_string(index=False))


if __name__ == "__main__":
    main()
