import numpy as np
import pytest

from odin import ScenarioConfig
from odin.cli import main
from odin.pipeline import analyze, export
from odin.sensitivity import budget_sensitivity, sensitivity_grid, success_rate_sensitivity
from odin.simulation import simulate_net_benefit, summarise_simulations
from odin.synthetic import generate_customers

from .conftest import ROOT


def test_sensitivity_matches_reference(raw):
    s = success_rate_sensitivity(raw, [0.3, 0.5, 0.7], 500)
    assert s["Expected Net Benefit (€)"].round(2).tolist() == [1614.03, 3001.39, 4388.75]
    b = budget_sensitivity(raw, [350, 500, 650], 0.5)
    assert b["Expected Net Benefit (€)"].round(2).tolist() == [2153.24, 3001.39, 3535.71]


def test_grid_is_monotone(raw):
    grid = sensitivity_grid(raw, [0.2, 0.5, 0.8], [200, 400, 600])
    assert (np.diff(grid.values, axis=0) >= -1e-9).all()  # more effective offers never hurt
    assert (np.diff(grid.values, axis=1) >= -1e-9).all()  # more budget never hurts


def test_monte_carlo_mean_converges_to_expectation(customers):
    ids = ["C001", "C004", "C007", "C021", "C024", "C030"]
    sims = simulate_net_benefit(customers, ids, 0.5, runs=200_000, seed=0)
    assert sims.mean() == pytest.approx(3001.39, rel=0.01)
    summary = summarise_simulations({"k": sims})
    assert 0 <= summary.loc["k", "Probability of Loss"] <= 1


def test_synthetic_generator_is_reproducible_and_valid():
    a, b = generate_customers(200, seed=3), generate_customers(200, seed=3)
    assert a.equals(b)
    assert a["Predicted_Churn_Probability"].between(0, 1).all()
    assert (a["Retention_Offer_Cost"] > 0).all()


def test_config_validation():
    with pytest.raises(ValueError):
        ScenarioConfig(success_rate=2)
    with pytest.raises(ValueError):
        ScenarioConfig(budget=-10)


def test_pipeline_and_export(raw, tmp_path):
    result = analyze(raw, ScenarioConfig(monte_carlo_runs=500))
    export(result, tmp_path, make_figures=True)
    assert (tmp_path / "tables" / "strategy_comparison.csv").exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 10


def test_cli(tmp_path, capsys):
    data = tmp_path / "c.csv"
    assert main(["generate", "--n", "40", "--out", str(data)]) == 0
    assert (
        main(
            [
                "analyze",
                "--data",
                str(data),
                "--budget",
                "300",
                "--runs",
                "200",
                "--out",
                str(tmp_path / "o"),
                "--no-figures",
            ]
        )
        == 0
    )
    assert "Knapsack Optimization" in capsys.readouterr().out


def test_customer_base_generator():
    from odin import validate_customers
    from odin.synthetic import generate_customer_base

    base = generate_customer_base(2000, seed=7)
    clean = validate_customers(base)
    assert {"Plan", "Region", "Contract", "Tenure_Months"} <= set(clean.columns)
    assert 0.15 < base["Actual_Churn"].mean() < 0.45
    assert base.equals(generate_customer_base(2000, seed=7))


def test_score_diagnostics():
    import pandas as pd

    from odin.evaluation import score_diagnostics

    perfect = pd.DataFrame(
        {"Actual_Churn": [0, 0, 1, 1], "Predicted_Churn_Probability": [0.1, 0.2, 0.8, 0.9]}
    )
    summary, table = score_diagnostics(perfect, bins=2)
    assert summary["ROC AUC"] == 1.0
    assert table["Customers"].sum() == 4


def test_segment_allocation(customers):
    from odin import knapsack_selection
    from odin.evaluation import segment_allocation

    seg = customers.assign(Group=["A" if i % 2 else "B" for i in range(len(customers))])
    ids = knapsack_selection(seg, 500)
    table = segment_allocation(seg, ids, "Group")
    assert table["Selected"].sum() == len(ids)
    assert abs(table["Share of Spend"].sum() - 1) < 1e-9


def test_cli_target_list(tmp_path):
    import pandas as pd

    out = tmp_path / "targets.csv"
    data = ROOT / "data" / "synthetic" / "customer_base_5000.csv"
    args = ["target", "--data", str(data), "--budget", "5000", "--out", str(out)]
    assert main(args) == 0
    single = pd.read_csv(out)
    assert single["Offer_Cost"].sum() <= 5000 and single["Customer_ID"].is_unique

    assert (
        main(
            [
                *args,
                "--multi-offer",
                "--max-contacts",
                "300",
                "--segment",
                "Region",
                "--max-share",
                "DACH=0.2",
            ]
        )
        == 0
    )
    multi = pd.read_csv(out)
    assert len(multi) <= 300
    assert multi.loc[multi["Region"] == "DACH", "Offer_Cost"].sum() <= 0.2 * 5000 + 1e-6

    assert main([*args, "--segment", "Region", "--min-contacts", "Nordics=100000"]) == 2
