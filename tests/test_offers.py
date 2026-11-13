import numpy as np
import pandas as pd
import pytest

from odin import knapsack_selection
from odin.offers import (
    DEFAULT_CATALOGUE,
    Offer,
    brute_force_multi_offer,
    build_offer_menu,
    compare_offer_policies,
    multi_offer_selection,
    offer_mix,
)
from odin.synthetic import generate_customer_base


@pytest.fixture(scope="module")
def base():
    return generate_customer_base(400, seed=11)


def test_menu_has_one_row_per_eligible_pair(base):
    menu = build_offer_menu(base)
    monthly = int((base["Contract"] == "Monthly").sum())
    assert len(menu) == 3 * len(base) + monthly
    assert set(menu["Offer"]) == {o.name for o in DEFAULT_CATALOGUE}
    assert (menu["Offer_Cost"] > 0).all()
    annual_ids = set(base.loc[base["Contract"] == "Annual", "Customer_ID"])
    assert not menu.loc[menu["Offer"] == "Annual-plan discount", "Customer_ID"].isin(annual_ids).any()


def test_menu_works_without_segment_columns(raw):
    menu = build_offer_menu(raw)
    assert len(menu) == len(raw) * len(DEFAULT_CATALOGUE)


@pytest.mark.parametrize("seed", range(6))
def test_mckp_matches_brute_force(seed):
    small = generate_customer_base(9, seed=seed)
    menu = build_offer_menu(small)
    budget = int(menu.groupby("Customer_ID")["Offer_Cost"].min().sum() * 1.3)
    dp = multi_offer_selection(menu, budget)
    exact = brute_force_multi_offer(menu, budget)
    assert dp["Offer_Net_Benefit"].sum() == pytest.approx(exact["Offer_Net_Benefit"].sum())


def test_mckp_respects_constraints(base):
    menu = build_offer_menu(base)
    sel = multi_offer_selection(menu, 2_000)
    assert sel["Offer_Cost"].sum() <= 2_000
    assert sel["Customer_ID"].is_unique
    assert (sel["Offer_Net_Benefit"] > 0).all()


def test_single_offer_reduces_to_knapsack():
    df = generate_customer_base(300, seed=3)
    menu = build_offer_menu(df, (Offer("Only", success_rate=0.5, cost_months=2, fixed_cost=3),))
    sel = multi_offer_selection(menu, 1_500)
    economics = df.assign(
        Retention_Offer_Cost=menu.set_index("Customer_ID").loc[df["Customer_ID"], "Offer_Cost"].to_numpy(),
    )
    economics["Expected_Net_Benefit"] = (
        economics["Predicted_Churn_Probability"] * economics["Estimated_CLV"] * 0.5
        - economics["Retention_Offer_Cost"]
    )
    ks = knapsack_selection(economics, 1_500)
    expected = economics.set_index("Customer_ID").loc[ks, "Expected_Net_Benefit"].sum()
    assert sel["Offer_Net_Benefit"].sum() == pytest.approx(expected)


def test_choice_dominates_every_single_offer_policy(base):
    table = compare_offer_policies(build_offer_menu(base), 3_000)
    best = table.loc["Best offer per subscriber (MCKP)", "Expected Net Benefit (€)"]
    assert np.all(best >= table["Expected Net Benefit (€)"] - 1e-6)


def test_zero_budget_and_empty_menu(base):
    menu = build_offer_menu(base)
    assert multi_offer_selection(menu, 0).empty
    assert multi_offer_selection(menu.iloc[0:0], 100).empty


def test_offer_mix_totals(base):
    sel = multi_offer_selection(build_offer_menu(base), 2_500)
    mix = offer_mix(sel)
    assert mix["Subscribers"].sum() == len(sel)
    assert mix["Spend"].sum() == sel["Offer_Cost"].sum()


def test_invalid_offer():
    with pytest.raises(ValueError):
        Offer("Bad", success_rate=1.5)
    with pytest.raises(ValueError):
        build_offer_menu(pd.DataFrame(), ())


def test_rescale_success_rates(base):
    from odin.offers import rescale_success_rates

    menu = build_offer_menu(base)
    same = rescale_success_rates(menu, 0.5)
    assert np.allclose(same["Offer_Net_Benefit"], menu["Offer_Net_Benefit"])
    higher = rescale_success_rates(menu, 0.7)
    assert np.allclose(higher["Offer_Success_Rate"], (menu["Offer_Success_Rate"] * 1.4).clip(upper=1))
    assert (higher["Offer_Net_Benefit"] >= menu["Offer_Net_Benefit"] - 1e-9).all()
    with pytest.raises(ValueError):
        rescale_success_rates(menu, 1.5)
