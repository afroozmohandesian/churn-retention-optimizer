"""Charts. Every function saves a PNG and returns the matplotlib Figure."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#9AA5B1", "#E07A5F", "#F2CC8F", "#3D5A80"]  # random, churn, greedy, knapsack

plt.rcParams.update(
    {
        "figure.figsize": (9, 5),
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 160,
        "figure.dpi": 110,
    }
)


def _save(fig: plt.Figure, path: Path) -> plt.Figure:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return fig


def _bar(series: pd.Series, title: str, ylabel: str, fmt: str, path: Path) -> plt.Figure:
    fig, ax = plt.subplots()
    colors = PALETTE[: len(series)] if len(series) <= len(PALETTE) else None
    bars = ax.bar(series.index, series.values, color=colors)
    ax.bar_label(bars, labels=[fmt.format(v) for v in series.values], padding=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, series.max() * 1.15 if series.max() > 0 else 1)
    ax.tick_params(axis="x", rotation=12)
    return _save(fig, path)


def dataset_overview(df: pd.DataFrame, path: Path) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    spec = [
        ("Predicted_Churn_Probability", "Churn risk scores", "Predicted churn probability"),
        ("Estimated_CLV", "Estimated CLV", "Estimated CLV (€)"),
        ("Retention_Offer_Cost", "Retention offer costs", "Offer cost (€)"),
    ]
    for ax, (col, title, xlabel) in zip(axes, spec):
        ax.hist(df[col], bins=min(20, max(8, len(df) // 25)), color="#3D5A80", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Customers")
    fig.suptitle("Understanding the customer dataset", y=1.03, fontweight="bold")
    return _save(fig, path)


def strategy_bars(comparison: pd.DataFrame, figure_dir: Path) -> None:
    _bar(
        comparison["Total Spending (€)"],
        "Campaign spending by strategy",
        "Spending (€)",
        "€{:,.0f}",
        figure_dir / "02_campaign_spending_by_strategy.png",
    )
    _bar(
        comparison["Expected Net Benefit (€)"],
        "Expected net benefit by strategy",
        "Expected net benefit (€)",
        "€{:,.0f}",
        figure_dir / "03_expected_net_benefit_by_strategy.png",
    )
    _bar(
        comparison["Selected Customers"],
        "Selected customers by strategy",
        "Customers",
        "{:.0f}",
        figure_dir / "04_selected_customers_by_strategy.png",
    )


def sensitivity_line(
    table: pd.DataFrame, x: str, xlabel: str, title: str, path: Path, x_scale: float = 1.0
) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(table[x] * x_scale, table["Expected Net Benefit (€)"], marker="o", color="#3D5A80", lw=2)
    for xv, yv in zip(table[x] * x_scale, table["Expected Net Benefit (€)"]):
        ax.annotate(f"€{yv:,.0f}", (xv, yv), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Expected net benefit (€)")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def selection_matrix(df: pd.DataFrame, selections: Mapping[str, list[str]], path: Path) -> plt.Figure:
    matrix = pd.DataFrame(
        {name: df["Customer_ID"].isin(ids).astype(int).values for name, ids in selections.items()},
        index=df["Customer_ID"],
    )
    fig, ax = plt.subplots(figsize=(8, max(6, len(matrix) * 0.26)))
    ax.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1.2)
    ax.set_title("Which customers does each strategy select?")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=15, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=8)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, path)


def sensitivity_heatmap(grid: pd.DataFrame, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(grid.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(grid.shape[1]), [f"€{b}" for b in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{r:.0%}" for r in grid.index])
    ax.set_xlabel("Campaign budget")
    ax.set_ylabel("Assumed retention success rate")
    ax.set_title("Optimal expected net benefit across budget × success-rate scenarios")
    threshold = grid.values.max() * 0.6
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(
                j,
                i,
                f"{v:,.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if v > threshold else "white",
            )
    fig.colorbar(image, ax=ax, label="Expected net benefit (€)")
    return _save(fig, path)


def monte_carlo_distribution(simulations: Mapping[str, np.ndarray], path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    lo = min(s.min() for s in simulations.values())
    hi = max(s.max() for s in simulations.values())
    bins = np.linspace(lo, hi, 60)
    for color, (name, sims) in zip(PALETTE, simulations.items()):
        ax.hist(sims, bins=bins, alpha=0.55, label=f"{name} (mean €{sims.mean():,.0f})", color=color)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_title("Simulated realised net benefit (Monte Carlo)")
    ax.set_xlabel("Realised net benefit (€)")
    ax.set_ylabel("Simulated campaigns")
    ax.legend(frameon=False)
    return _save(fig, path)


def risk_value_scatter(df: pd.DataFrame, selected_ids: list[str], path: Path) -> plt.Figure:
    """Churn probability vs CLV; bubble size = offer cost; highlighted = optimal picks."""
    fig, ax = plt.subplots(figsize=(9, 6))
    picked = df["Customer_ID"].isin(selected_ids)
    size = df["Retention_Offer_Cost"] * 2
    ax.scatter(
        df.loc[~picked, "Predicted_Churn_Probability"],
        df.loc[~picked, "Estimated_CLV"],
        s=size[~picked],
        c="#9AA5B1",
        alpha=0.6,
        label="Not selected",
        edgecolor="white",
    )
    ax.scatter(
        df.loc[picked, "Predicted_Churn_Probability"],
        df.loc[picked, "Estimated_CLV"],
        s=size[picked],
        c="#3D5A80",
        alpha=0.9,
        label="Knapsack selection",
        edgecolor="white",
    )
    if len(df) <= 60:
        for _, row in df.loc[picked].iterrows():
            ax.annotate(
                row["Customer_ID"],
                (row["Predicted_Churn_Probability"], row["Estimated_CLV"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
    ax.set_title("Risk vs value: who gets the offer? (bubble size = offer cost)")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Estimated CLV (€)")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, path)
