"""Project Odin — budget-constrained customer retention optimisation.

Turn churn-risk scores into a financially optimal retention campaign:
estimate the expected net benefit of every retention offer, then pick the
combination of customers that maximises total benefit under a fixed budget
(0/1 Knapsack).
"""

from odin.config import ScenarioConfig
from odin.constraints import CampaignConstraints, InfeasibleCampaignError, constrained_selection
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
from odin.uplift import TLearner, add_uplift_value, qini_coefficient, qini_curve

__version__ = "1.0.0"

__all__ = [
    "CampaignConstraints",
    "InfeasibleCampaignError",
    "DEFAULT_CATALOGUE",
    "REQUIRED_COLUMNS",
    "STRATEGIES",
    "TLearner",
    "DataValidationError",
    "Offer",
    "ScenarioConfig",
    "__version__",
    "add_expected_value",
    "add_uplift_value",
    "brute_force_selection",
    "build_offer_menu",
    "churn_ranked_selection",
    "compare_strategies",
    "constrained_selection",
    "evaluate_selection",
    "knapsack_selection",
    "load_customers",
    "multi_offer_selection",
    "qini_coefficient",
    "qini_curve",
    "random_selection",
    "validate_customers",
    "value_density_selection",
]
