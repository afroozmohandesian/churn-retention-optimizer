"""Scenario assumptions, centralised for transparency and reproducibility."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    """Assumptions that drive one run of the analysis.

    All monetary values are in euros. ``success_rate`` is a *hypothetical*
    probability that a retention offer prevents a churn that would otherwise
    happen — it is a scenario input, not something estimated from the data.
    """

    budget: int = 500
    success_rate: float = 0.50
    random_seed: int = 42
    success_rate_scenarios: tuple[float, ...] = (0.30, 0.50, 0.70)
    budget_scenarios: tuple[int, ...] = (350, 500, 650)
    monte_carlo_runs: int = 10_000

    def __post_init__(self) -> None:
        if self.budget < 0 or int(self.budget) != self.budget:
            raise ValueError("budget must be a non-negative whole number of euros")
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be between 0 and 1")
        if self.monte_carlo_runs <= 0:
            raise ValueError("monte_carlo_runs must be positive")
