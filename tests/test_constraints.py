import pytest

from odin import knapsack_selection
from odin.constraints import (
    CampaignConstraints,
    InfeasibleCampaignError,
    constrained_selection,
    menu_from_single_offer,
    shadow_cost,
)
from odin.offers import build_offer_menu, multi_offer_selection
from odin.synthetic import generate_customer_base


@pytest.fixture(scope="module")
def base():
    return generate_customer_base(500, seed=5)


@pytest.fixture(scope="module")
def menu(base):
    return build_offer_menu(base)


def test_unconstrained_ilp_matches_knapsack_dp(customers):
    sel = constrained_selection(menu_from_single_offer(customers), 500)
    dp = knapsack_selection(customers, 500)
    expected = customers.set_index("Customer_ID").loc[dp, "Expected_Net_Benefit"].sum()
    assert sel["Offer_Net_Benefit"].sum() == pytest.approx(expected)


def test_unconstrained_ilp_matches_mckp_dp(menu):
    ilp = constrained_selection(menu, 3_000)
    dp = multi_offer_selection(menu, 3_000)
    assert ilp["Offer_Net_Benefit"].sum() == pytest.approx(dp["Offer_Net_Benefit"].sum())
    assert ilp["Customer_ID"].is_unique


def test_contact_capacity(menu):
    free = constrained_selection(menu, 3_000)
    capped = constrained_selection(menu, 3_000, CampaignConstraints(max_contacts=50))
    assert len(capped) <= 50 < len(free)
    assert capped["Offer_Cost"].sum() <= 3_000


def test_segment_spend_cap_and_minimum(base, menu):
    segments = base.set_index("Customer_ID")["Region"]
    rules = CampaignConstraints(
        segment_column="Region", max_spend_share={"DACH": 0.2}, min_contacts={"Nordics": 40}
    )
    sel = constrained_selection(menu, 3_000, rules, segments)
    region = sel["Customer_ID"].map(segments)
    assert sel.loc[region == "DACH", "Offer_Cost"].sum() <= 0.2 * 3_000 + 1e-6
    assert (region == "Nordics").sum() >= 40
    assert sel["Customer_ID"].is_unique


def test_shadow_cost_is_non_negative(base, menu):
    segments = base.set_index("Customer_ID")["Plan"]
    rules = CampaignConstraints(max_contacts=80, segment_column="Plan", max_spend_share={"Standard": 0.25})
    report = shadow_cost(menu, 3_000, rules, segments)
    assert report["Cost of Constraints (€)"] >= -1e-6
    assert 0 < report["Share Retained"] <= 1 + 1e-9


def test_infeasible_minimum_raises(base, menu):
    segments = base.set_index("Customer_ID")["Region"]
    rules = CampaignConstraints(segment_column="Region", min_contacts={"Benelux": 10_000})
    with pytest.raises(InfeasibleCampaignError):
        constrained_selection(menu, 3_000, rules, segments)


def test_invalid_constraints():
    with pytest.raises(ValueError):
        CampaignConstraints(max_contacts=-1)
    with pytest.raises(ValueError):
        CampaignConstraints(max_spend_share={"DACH": 0.3})
    with pytest.raises(ValueError):
        CampaignConstraints(segment_column="Region", max_spend_share={"DACH": 1.4})
