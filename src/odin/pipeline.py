"""End-to-end analysis: data -> economics -> strategies -> sensitivity -> risk -> exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from odin import plots
from odin.config import ScenarioConfig
from odin.economics import add_expected_value, break_even_success_rate
from odin.evaluation import compare_strategies, optimality_gap, run_strategies
from odin.sensitivity import (
    budget_sensitivity,
    portfolio_stability,
    sensitivity_grid,
    success_rate_sensitivity,
)
from odin.simulation import simulate_net_benefit, summarise_simulations


@dataclass
class AnalysisResult:
    customers: pd.DataFrame
    selections: dict[str, list[str]]
    comparison: pd.DataFrame
    portfolio: pd.DataFrame
    success_sensitivity: pd.DataFrame
    budget_sensitivity: pd.DataFrame
    grid: pd.DataFrame
    stability: pd.DataFrame
    risk: pd.DataFrame
    simulations: dict[str, np.ndarray]


def analyze(df: pd.DataFrame, config: ScenarioConfig | None = None) -> AnalysisResult:
    """Run the full analysis in memory (no files written)."""
    config = config or ScenarioConfig()
    customers = add_expected_value(df, config.success_rate)
    customers["Break_Even_Success_Rate"] = break_even_success_rate(customers)

    selections = run_strategies(customers, config.budget, seed=config.random_seed)
    comparison = compare_strategies(customers, selections)
    comparison["Share of Optimum"] = optimality_gap(comparison)

    portfolio = (
        customers[customers["Customer_ID"].isin(selections["Knapsack Optimization"])]
        .sort_values("Expected_Net_Benefit", ascending=False)
        .reset_index(drop=True)
    )

    rate_grid = np.round(np.arange(0.10, 0.951, 0.05), 2)
    step = max(50, config.budget // 5)
    budget_grid = list(range(step, 2 * config.budget + 1, step))

    simulations = {
        name: simulate_net_benefit(
            customers, ids, config.success_rate, runs=config.monte_carlo_runs, seed=config.random_seed
        )
        for name, ids in selections.items()
    }

    return AnalysisResult(
        customers=customers,
        selections=selections,
        comparison=comparison,
        portfolio=portfolio,
        success_sensitivity=success_rate_sensitivity(df, config.success_rate_scenarios, config.budget),
        budget_sensitivity=budget_sensitivity(df, config.budget_scenarios, config.success_rate),
        grid=sensitivity_grid(df, rate_grid, budget_grid),
        stability=portfolio_stability(df, rate_grid, config.budget),
        risk=summarise_simulations(simulations),
        simulations=simulations,
    )


def export(result: AnalysisResult, out_dir: str | Path, make_figures: bool = True) -> Path:
    """Write every table (CSV) and chart (PNG) under ``out_dir``."""
    out_dir = Path(out_dir)
    tables, figures = out_dir / "tables", out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    result.customers.to_csv(tables / "customer_analysis_full.csv", index=False)
    result.comparison.to_csv(tables / "strategy_comparison.csv")
    result.portfolio.to_csv(tables / "knapsack_selected_customers.csv", index=False)
    result.success_sensitivity.to_csv(tables / "sensitivity_success_rate.csv", index=False)
    result.budget_sensitivity.to_csv(tables / "sensitivity_budget.csv", index=False)
    result.grid.to_csv(tables / "sensitivity_grid.csv")
    result.stability.to_csv(tables / "portfolio_stability.csv", index=False)
    result.risk.to_csv(tables / "monte_carlo_risk.csv")

    if make_figures:
        c = result.customers
        plots.dataset_overview(c, figures / "01_customer_dataset_overview.png")
        plots.strategy_bars(result.comparison, figures)
        plots.sensitivity_line(
            result.success_sensitivity,
            "Assumed Success Rate",
            "Assumed retention success rate (%)",
            "Expected net benefit vs. success-rate assumption",
            figures / "05_sensitivity_to_success_rate.png",
            x_scale=100,
        )
        plots.sensitivity_line(
            result.budget_sensitivity,
            "Budget (€)",
            "Campaign budget (€)",
            "Expected net benefit vs. campaign budget",
            figures / "06_sensitivity_to_campaign_budget.png",
        )
        if len(c) <= 80:
            plots.selection_matrix(c, result.selections, figures / "07_customer_selection_matrix.png")
        plots.sensitivity_heatmap(result.grid, figures / "08_sensitivity_heatmap.png")
        plots.monte_carlo_distribution(result.simulations, figures / "09_monte_carlo_net_benefit.png")
        plots.risk_value_scatter(
            c, result.selections["Knapsack Optimization"], figures / "10_risk_vs_value_selection.png"
        )
    return out_dir


def summary_text(result: AnalysisResult, config: ScenarioConfig) -> str:
    """Human-readable console summary."""
    cmp = result.comparison
    optimum = cmp["Expected Net Benefit (€)"].max()
    best = ", ".join(cmp.index[cmp["Expected Net Benefit (€)"] >= optimum - 1e-6])
    lines = [
        f"Customers analysed : {len(result.customers)}",
        f"Budget             : €{config.budget:,}",
        f"Success rate       : {config.success_rate:.0%}",
        "",
        cmp[
            [
                "Selected Customers",
                "Total Spending (€)",
                "Expected Net Benefit (€)",
                "ROI",
                "Share of Optimum",
            ]
        ]
        .round(2)
        .to_string(),
        "",
        f"Optimal strategies : {best}",
        f"Knapsack portfolio : {', '.join(result.selections['Knapsack Optimization'][:15])}"
        + (" ..." if len(result.selections["Knapsack Optimization"]) > 15 else ""),
        "",
        "Monte Carlo risk (realised net benefit):",
        result.risk[["Mean Net Benefit (€)", "5th Percentile (€)", "Probability of Loss"]]
        .round(3)
        .to_string(),
    ]
    return "\n".join(lines)
