"""Project Odin — budget-constrained customer retention optimisation.

Turn churn-risk scores into a financially optimal retention campaign:
estimate the expected net benefit of every retention offer, then pick the
combination of customers that maximises total benefit under a fixed budget
(0/1 Knapsack).
"""

from odin.config import ScenarioConfig
from odin.data import REQUIRED_COLUMNS, DataValidationError, load_customers, validate_customers
from odin.economics import add_expected_value
from odin.evaluation import compare_strategies, evaluate_selection
from odin.offers import DEFAULT_CATALOGUE, Offer, build_offer_menu, multi_offer_selection
from odin.strategies import (
    STRATEGIES,
    brute_force_selection,
    churn_ranked_selection,
    knapsack_selection,
    random_selection,
    value_density_selection,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_CATALOGUE",
    "REQUIRED_COLUMNS",
    "STRATEGIES",
    "DataValidationError",
    "Offer",
    "ScenarioConfig",
    "__version__",
    "add_expected_value",
    "brute_force_selection",
    "build_offer_menu",
    "churn_ranked_selection",
    "compare_strategies",
    "evaluate_selection",
    "knapsack_selection",
    "load_customers",
    "multi_offer_selection",
    "random_selection",
    "validate_customers",
    "value_density_selection",
]
