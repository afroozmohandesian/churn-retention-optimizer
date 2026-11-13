import numpy as np
import pandas as pd
import pytest

from odin import (
    add_expected_value,
    brute_force_selection,
    churn_ranked_selection,
    knapsack_selection,
    random_selection,
    value_density_selection,
)
from odin.evaluation import compare_strategies, evaluate_selection, run_strategies
from odin.synthetic import generate_customers
from tests.conftest import make


def total(df, ids, col="Expected_Net_Benefit"):
    return df.loc[df["Customer_ID"].isin(ids), col].sum()


def test_reference_result_is_reproduced(customers):
    ids = knapsack_selection(customers, 500)
    assert ids == ["C001", "C004", "C007", "C021", "C024", "C030"]
    assert total(customers, ids) == pytest.approx(3001.39)
    assert total(customers, ids, "Retention_Offer_Cost") == 467


@pytest.mark.parametrize("seed", range(25))
def test_knapsack_matches_brute_force(seed):
    df = add_expected_value(generate_customers(14, seed=seed), 0.4)
    budget = int(np.random.default_rng(seed).integers(50, 600))
    dp = knapsack_selection(df, budget)
    bf = brute_force_selection(df, budget)
    assert total(df, dp) == pytest.approx(total(df, bf))
    assert total(df, dp, "Retention_Offer_Cost") <= budget


@pytest.mark.parametrize("budget", [0, 50, 200, 500, 1000, 5000])
def test_every_strategy_respects_budget(customers, budget):
    for ids in run_strategies(customers, budget).values():
        assert total(customers, ids, "Retention_Offer_Cost") <= budget


@pytest.mark.parametrize("budget", [100, 300, 500, 800])
def test_knapsack_dominates_heuristics(customers, budget):
    opt = total(customers, knapsack_selection(customers, budget))
    for fn in (churn_ranked_selection, value_density_selection, random_selection):
        assert total(customers, fn(customers, budget)) <= opt + 1e-9


def test_knapsack_beats_greedy_on_classic_counterexample():
    # Density favours A (ratio 2.0) but B+C fill the budget better.
    df = pd.DataFrame(
        {
            "Customer_ID": ["A", "B", "C"],
            "Actual_Churn": [1, 1, 1],
            "Predicted_Churn_Probability": [1, 1, 1],
            "Estimated_CLV": [0, 0, 0],
            "Retention_Offer_Cost": [6, 5, 5],
            "Expected_Retained_Value": [0, 0, 0],
            "Expected_Net_Benefit": [12.0, 9.0, 9.0],
        }
    )
    assert value_density_selection(df, 10) == ["A"]
    assert sorted(knapsack_selection(df, 10)) == ["B", "C"]


def test_knapsack_ignores_negative_benefit():
    df = add_expected_value(make([("A", 1, 0.1, 100, 90), ("B", 1, 0.9, 1000, 50)]), 0.5)
    assert knapsack_selection(df, 1000) == ["B"]


def test_empty_and_zero_budget(customers):
    assert knapsack_selection(customers, 0) == []
    assert knapsack_selection(customers.iloc[0:0], 500) == []


@pytest.mark.parametrize("bad", [-1, 10.5])
def test_invalid_budget(customers, bad):
    with pytest.raises(ValueError):
        knapsack_selection(customers, bad)


def test_random_is_reproducible(customers):
    assert random_selection(customers, 500, seed=7) == random_selection(customers, 500, seed=7)


def test_evaluation_metrics(customers):
    ids = knapsack_selection(customers, 500)
    m = evaluate_selection(customers, ids)
    assert m["Selected Customers"] == 6
    assert m["Actual Churners Captured"] == 5
    assert m["Actual Churn Capture Rate"] == pytest.approx(5 / 15)
    cmp = compare_strategies(customers, run_strategies(customers, 500))
    assert cmp["Expected Net Benefit (€)"].idxmax() in {"Knapsack Optimization", "Value Density (Greedy)"}


def test_scales_to_thousands():
    df = add_expected_value(generate_customers(3000, seed=1), 0.5)
    ids = knapsack_selection(df, 10_000)
    assert total(df, ids, "Retention_Offer_Cost") <= 10_000
    assert total(df, ids) >= total(df, value_density_selection(df, 10_000)) - 1e-6
