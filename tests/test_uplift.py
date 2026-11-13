import numpy as np
import pandas as pd
import pytest

from odin import add_expected_value, knapsack_selection
from odin.synthetic import generate_customer_base
from odin.uplift import (
    TLearner,
    add_uplift_value,
    qini_coefficient,
    qini_curve,
    realised_value,
    simulate_campaign_experiment,
    true_offer_effect,
    true_uplift,
)


@pytest.fixture(scope="module")
def fitted():
    cohort = generate_customer_base(8_000, seed=101)
    return TLearner().fit(simulate_campaign_experiment(cohort, seed=1))


@pytest.fixture(scope="module")
def base():
    return generate_customer_base(2_000, seed=7)


def test_experiment_is_randomised(base):
    exp = simulate_campaign_experiment(base, treat_share=0.3, seed=3)
    assert exp["Treatment"].mean() == pytest.approx(0.3, abs=0.04)
    assert set(exp["Churned"].unique()) <= {0, 1}
    treated = exp.loc[exp["Treatment"] == 1, "Churned"].mean()
    control = exp.loc[exp["Treatment"] == 0, "Churned"].mean()
    assert treated < control


def test_true_effect_bounds_and_heterogeneity(base):
    effect = true_offer_effect(base)
    assert effect.min() >= -0.3 and effect.max() <= 0.85
    assert (effect < 0).any() and effect.std() > 0.1


def test_tlearner_recovers_true_uplift(fitted, base):
    estimate = fitted.predict(base)
    assert np.corrcoef(estimate, true_uplift(base))[0, 1] > 0.8


def test_uplift_targeting_beats_constant_rate(fitted, base):
    budget = 8_000
    constant = knapsack_selection(add_expected_value(base, 0.5), budget)
    uplift = knapsack_selection(add_uplift_value(base, fitted.predict(base)), budget)
    assert (
        realised_value(base, uplift)["True Net Benefit (€)"]
        > realised_value(base, constant)["True Net Benefit (€)"]
    )


def test_qini_ranks_model_above_random(fitted, base):
    hold = simulate_campaign_experiment(base, seed=2)
    model = qini_coefficient(qini_curve(fitted.predict(hold), hold["Treatment"], hold["Churned"]))
    noise = np.random.default_rng(0).random(len(hold))
    random = qini_coefficient(qini_curve(noise, hold["Treatment"], hold["Churned"]))
    assert model > random


def test_qini_curve_shape():
    curve = qini_curve([0.9, 0.5, 0.1, 0.0], [1, 0, 1, 0], [0, 1, 1, 1])
    assert curve["Share_Targeted"].iloc[0] == 0 and curve["Share_Targeted"].iloc[-1] == 1
    assert len(curve) == 5


def test_add_uplift_value_columns(base):
    out = add_uplift_value(base, np.full(len(base), 0.1))
    assert np.allclose(out["Expected_Retained_Value"], 0.1 * base["Estimated_CLV"])
    assert np.allclose(
        out["Expected_Net_Benefit"], out["Expected_Retained_Value"] - base["Retention_Offer_Cost"]
    )


def test_errors(base):
    with pytest.raises(ValueError):
        simulate_campaign_experiment(base, treat_share=1.0)
    with pytest.raises(ValueError):
        simulate_campaign_experiment(base[["Customer_ID", "Predicted_Churn_Probability"]])
    with pytest.raises(RuntimeError):
        TLearner().predict(base)
    one_arm = simulate_campaign_experiment(base, seed=0).assign(Churned=0)
    with pytest.raises(ValueError):
        TLearner().fit(one_arm)
    assert isinstance(realised_value(base, [])["True Net Benefit (€)"], float)
    assert pd.api.types.is_float_dtype(true_uplift(base))
