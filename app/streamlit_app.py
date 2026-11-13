"""Project Odin scenario explorer.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Pick a dataset, a budget and an offer-effect assumption, optionally switch on
several offer types and operational constraints, and see the optimal
retention portfolio update.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from odin import add_expected_value, load_customers, validate_customers
from odin.constraints import (
    CampaignConstraints,
    InfeasibleCampaignError,
    constrained_selection,
    menu_from_single_offer,
)
from odin.evaluation import compare_strategies, run_strategies
from odin.offers import build_offer_menu, multi_offer_selection, offer_mix, rescale_success_rates
from odin.synthetic import generate_customer_base
from odin.uplift import FEATURES, TLearner, add_uplift_value, simulate_campaign_experiment

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "Full subscriber base (5,000)": ROOT / "data" / "synthetic" / "customer_base_5000.csv",
    "Pilot cohort (30)": ROOT / "data" / "raw" / "project_odin_customers.csv",
}
SEGMENT_COLUMNS = ["Plan", "Contract", "Region", "Acquisition_Channel"]

st.set_page_config(page_title="Project Odin · Scenario explorer", layout="wide")


@st.cache_data
def load(path: str) -> pd.DataFrame:
    return load_customers(path)


@st.cache_resource
def uplift_model() -> TLearner:
    """T-learner trained on a simulated randomised campaign (separate synthetic cohort)."""
    cohort = generate_customer_base(20_000, seed=101)
    return TLearner().fit(simulate_campaign_experiment(cohort, seed=1))


def euros(x: float) -> str:
    return f"€{x:,.0f}"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Scenario")
source = st.sidebar.radio("Dataset", [*DATASETS, "Upload CSV"])
if source == "Upload CSV":
    upload = st.sidebar.file_uploader("Customer CSV", type="csv")
    if upload is None:
        st.info("Upload a CSV with the Project Odin schema to begin.")
        st.stop()
    try:
        data = validate_customers(pd.read_csv(upload))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
else:
    data = load(str(DATASETS[source]))

has_segments = set(SEGMENT_COLUMNS) <= set(data.columns)
has_features = set(FEATURES) <= set(data.columns)
default_budget = int(min(data["Retention_Offer_Cost"].sum(), 25_000 if len(data) > 100 else 500))
max_budget = int(data["Retention_Offer_Cost"].sum() * 4)
budget = st.sidebar.number_input("Budget (€)", 0, max_budget, default_budget, step=100)

effect_options = ["Constant success rate"] + (["Uplift model (per subscriber)"] if has_features else [])
effect = st.sidebar.radio("Offer effect", effect_options)
success_rate = 0.5
if effect == "Constant success rate":
    success_rate = st.sidebar.slider("Offer success rate", 0.05, 0.95, 0.50, 0.05)

multi_offer = (
    st.sidebar.toggle(
        "Choose the best offer per subscriber",
        help="Loyalty email, one or two free months, or an annual-plan discount (multiple-choice knapsack).",
        disabled=effect != "Constant success rate",
    )
    and effect == "Constant success rate"
)

st.sidebar.subheader("Operational constraints")
use_capacity = st.sidebar.checkbox("Limit the number of contacts")
max_contacts = None
if use_capacity:
    max_contacts = st.sidebar.number_input("Maximum contacts", min_value=0, value=max(1, len(data) // 10))
segment_column = None
caps: dict[str, float] = {}
minimums: dict[str, int] = {}
if has_segments and st.sidebar.checkbox("Segment quotas"):
    segment_column = st.sidebar.selectbox("Segment", SEGMENT_COLUMNS, index=2)
    for seg in sorted(data[segment_column].unique()):
        c1, c2 = st.sidebar.columns(2)
        cap = c1.number_input(f"{seg}: max % spend", 0, 100, 100, 5, key=f"cap-{seg}")
        low = c2.number_input(f"{seg}: min contacts", 0, value=0, key=f"min-{seg}")
        if cap < 100:
            caps[seg] = cap / 100
        if low > 0:
            minimums[seg] = int(low)

# ---------------------------------------------------------------- economics
if effect == "Constant success rate":
    economics = add_expected_value(data, success_rate)
else:
    economics = add_uplift_value(data, uplift_model().predict(data))

if multi_offer:
    # Scale the catalogue so the "Two free months" offer matches the slider value.
    menu = rescale_success_rates(build_offer_menu(data), success_rate)
else:
    menu = menu_from_single_offer(economics)

constraints = CampaignConstraints(
    max_contacts=max_contacts, segment_column=segment_column, max_spend_share=caps, min_contacts=minimums
)
constrained = bool(max_contacts is not None or caps or minimums)
segments = data.set_index("Customer_ID")[segment_column] if segment_column else None

try:
    if constrained:
        portfolio = constrained_selection(menu, int(budget), constraints, segments)
    else:
        portfolio = multi_offer_selection(menu, int(budget))
except InfeasibleCampaignError as exc:
    st.error(f"{exc} Relax a minimum or raise the budget.")
    st.stop()

# ---------------------------------------------------------------- headline
st.title("Project Odin · Retention scenario explorer")
st.caption(
    "Synthetic data for a fictional subscription business. Expected values depend on the offer-effect "
    "assumption chosen in the sidebar."
)
spend = float(portfolio["Offer_Cost"].sum())
benefit = float(portfolio["Offer_Net_Benefit"].sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Subscribers targeted", f"{len(portfolio):,}")
k2.metric("Spend", euros(spend), help=f"{spend / budget:.0%} of the budget" if budget else None)
k3.metric("Expected net benefit", euros(benefit))
k4.metric("Return per euro", f"{benefit / spend:.2f}" if spend else "–")

tab_compare, tab_segments, tab_frontier, tab_list = st.tabs(
    ["Strategy comparison", "Where the budget goes", "Budget frontier", "Target list"]
)

with tab_compare:
    if multi_offer:
        st.subheader("Offer mix")
        st.dataframe(offer_mix(portfolio).style.format("{:,.0f}"), use_container_width=True)
    elif constrained:
        st.info("The strategy comparison covers the unconstrained problem. Switch off constraints to see it.")
    else:
        table = compare_strategies(economics, run_strategies(economics, int(budget)))
        st.bar_chart(
            table["Expected Net Benefit (€)"].sort_values(),
            horizontal=True,
            x_label="",
            y_label="Expected net benefit (€)",
        )
        st.dataframe(
            table.style.format(
                {
                    "Total Spending (€)": "€{:,.0f}",
                    "Expected Retained Value (€)": "€{:,.0f}",
                    "Expected Net Benefit (€)": "€{:,.0f}",
                    "ROI": "{:.2f}",
                    "Actual Churn Capture Rate": "{:.0%}",
                    "Selected Customers": "{:,.0f}",
                    "Actual Churners Captured": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )

with tab_segments:
    if not has_segments:
        st.info("This dataset has no segment columns.")
    else:
        by = st.selectbox("Segment by", SEGMENT_COLUMNS, key="by")
        chosen = portfolio.assign(Segment=portfolio["Customer_ID"].map(data.set_index("Customer_ID")[by]))
        alloc = chosen.groupby("Segment").agg(Spend=("Offer_Cost", "sum"), Targeted=("Customer_ID", "size"))
        alloc["Base"] = data.groupby(by).size()
        alloc["Selection rate"] = alloc["Targeted"] / alloc["Base"]
        st.bar_chart(alloc["Spend"], y_label="Spend (€)")
        st.dataframe(
            alloc.style.format({"Spend": "€{:,.0f}", "Selection rate": "{:.1%}"}), use_container_width=True
        )

with tab_frontier:
    st.caption("Optimal expected net benefit at each budget level, with today's other settings.")
    points = sorted({int(budget * f) for f in (0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0)} - {0})
    values = []
    for b in points:
        try:
            sel = (
                constrained_selection(menu, b, constraints, segments)
                if constrained
                else multi_offer_selection(menu, b)
            )
            values.append(float(sel["Offer_Net_Benefit"].sum()))
        except InfeasibleCampaignError:
            values.append(float("nan"))
    frontier = pd.DataFrame({"Budget (€)": points, "Expected net benefit (€)": values})
    frontier = frontier.set_index("Budget (€)")
    frontier["Marginal return per €"] = (
        frontier["Expected net benefit (€)"].diff() / pd.Series(points, index=frontier.index).diff()
    )
    st.line_chart(frontier["Expected net benefit (€)"])
    st.dataframe(
        frontier.style.format({"Expected net benefit (€)": "€{:,.0f}", "Marginal return per €": "{:.2f}"}),
        use_container_width=True,
    )

with tab_list:
    shown = ["Plan", "Contract", "Region", "Predicted_Churn_Probability", "Estimated_CLV"]
    extra = [c for c in shown if c in data]
    target_list = portfolio.merge(data[["Customer_ID", *extra]], on="Customer_ID").sort_values(
        "Offer_Net_Benefit", ascending=False
    )
    st.dataframe(target_list, use_container_width=True, hide_index=True)
    st.download_button(
        "Download target list (CSV)", target_list.to_csv(index=False), "odin_target_list.csv", "text/csv"
    )
