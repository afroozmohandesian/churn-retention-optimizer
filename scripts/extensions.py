"""Reproduce the extension results on the 5,000-subscriber base.

    python scripts/extensions.py            # writes outputs/extensions/*.csv

1. Several offer types per subscriber (multiple-choice knapsack)
2. Contact-capacity and segment-quota constraints (integer programme)
3. Uplift modelling versus a constant success rate
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from odin import add_expected_value, churn_ranked_selection, knapsack_selection, load_customers
from odin.constraints import CampaignConstraints, shadow_cost
from odin.offers import build_offer_menu, compare_offer_policies, multi_offer_selection, offer_mix
from odin.synthetic import generate_customer_base
from odin.uplift import (
    TLearner,
    add_uplift_value,
    qini_coefficient,
    qini_curve,
    realised_value,
    simulate_campaign_experiment,
    true_uplift,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 25_000


def main() -> None:
    warnings.filterwarnings("ignore")
    out = ROOT / "outputs" / "extensions"
    out.mkdir(parents=True, exist_ok=True)
    base = load_customers(ROOT / "data" / "synthetic" / "customer_base_5000.csv")

    # 1. Several offers per subscriber
    menu = build_offer_menu(base)
    policies = compare_offer_policies(menu, BUDGET)
    mix = offer_mix(multi_offer_selection(menu, BUDGET))
    policies.to_csv(out / "offer_policies.csv")
    mix.to_csv(out / "offer_mix.csv")

    # 2. Operational constraints
    segments = base.set_index("Customer_ID")["Region"]
    scenarios = {
        "Contacts <= 1,500": CampaignConstraints(max_contacts=1_500),
        "DACH <= 25% of spend": CampaignConstraints(segment_column="Region", max_spend_share={"DACH": 0.25}),
        "Nordics >= 300 contacts": CampaignConstraints(
            segment_column="Region", min_contacts={"Nordics": 300}
        ),
        "All three": CampaignConstraints(
            max_contacts=1_500,
            segment_column="Region",
            max_spend_share={"DACH": 0.25},
            min_contacts={"Nordics": 300},
        ),
    }
    constraint_table = pd.DataFrame(
        {name: shadow_cost(menu, BUDGET, rules, segments) for name, rules in scenarios.items()}
    ).T
    constraint_table.index.name = "Constraint"
    constraint_table.to_csv(out / "constraint_cost.csv")

    # 3. Uplift modelling — trained on a separate simulated experiment cohort
    cohort = generate_customer_base(20_000, seed=101)
    model = TLearner().fit(simulate_campaign_experiment(cohort, seed=1))
    estimate = model.predict(base)
    truth = true_uplift(base)
    holdout = simulate_campaign_experiment(base, seed=2)
    quality = {
        "Correlation with true uplift": float(np.corrcoef(estimate, truth)[0, 1]),
        "Qini: churn score": qini_coefficient(
            qini_curve(base["Predicted_Churn_Probability"], holdout["Treatment"], holdout["Churned"])
        ),
        "Qini: uplift model": qini_coefficient(
            qini_curve(model.predict(holdout), holdout["Treatment"], holdout["Churned"])
        ),
        "Qini: true uplift (oracle)": qini_coefficient(
            qini_curve(truth, holdout["Treatment"], holdout["Churned"])
        ),
    }
    pd.Series(quality, name="Value").to_csv(out / "uplift_model_quality.csv")

    constant = add_expected_value(base, 0.5)
    portfolios = {
        "Highest churn probability": churn_ranked_selection(constant, BUDGET),
        "Knapsack, constant 50% rate": knapsack_selection(constant, BUDGET),
        "Knapsack, uplift model": knapsack_selection(add_uplift_value(base, estimate), BUDGET),
        "Knapsack, true uplift (oracle)": knapsack_selection(add_uplift_value(base, truth), BUDGET),
    }
    uplift_table = pd.DataFrame({n: realised_value(base, ids, truth) for n, ids in portfolios.items()}).T
    uplift_table.index.name = "Portfolio"
    uplift_table.to_csv(out / "uplift_portfolios.csv")

    with pd.option_context("display.width", 140, "display.float_format", "{:,.2f}".format):
        for title, table in [
            ("Offer policies", policies),
            ("Offer mix", mix),
            ("Cost of constraints", constraint_table),
            ("Uplift model quality", pd.Series(quality)),
            ("Portfolios under the true offer effect", uplift_table),
        ]:
            print(f"\n== {title} ==\n{table}")
    print(f"\nTables written to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
