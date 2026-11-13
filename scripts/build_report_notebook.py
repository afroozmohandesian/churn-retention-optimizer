"""Generate notebooks/Project_Odin_Report.ipynb from source.

The report is kept as code so its narrative and analysis are reviewable in diffs.
Run from the repository root, then execute it:

    python scripts/build_report_notebook.py
    make notebook
"""

import sys

import nbformat as nbf

OUT = sys.argv[1] if len(sys.argv) > 1 else "notebooks/Project_Odin_Report.ipynb"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ---------------------------------------------------------------- Title
md(r"""
<div align="center">

# PROJECT ODIN
### Retention Budget Optimisation: Decision Analytics Report

</div>

| | |
|---|---|
| **Client** | Aurvandil Media, a pan-European subscription streaming service *(fictional client used for this case study)* |
| **Engagement** | Allocation of the retention-campaign budget across at-risk subscribers |
| **Business question** | *With a fixed budget, which subscribers should receive a retention offer to maximise expected net financial benefit?* |
| **Deliverables** | This report · the reusable `odin` optimisation package · exported tables and charts |
| **Version** | 2.0, November 2026 |
| **Data classification** | Synthetic replica. Contains **no real customer data**. |

---

## Executive summary

**Recommendation:** replace the current *"highest churn score first"* targeting rule with **value-based portfolio optimisation** (0/1 Knapsack). It chooses the combination of subscribers that maximises expected net benefit within the budget.

| | Current rule (risk-first) | **Optimised portfolio** | Change |
|---|---:|---:|---:|
| **Pilot**: 30 high-value accounts, €500 | €1,048 | **€3,001** | **2.9×** |
| **Full base**: 5,000 subscribers, €25,000 | €20,438 | **€23,405** | **+€2,967 (+15%)** |
| Probability the pilot campaign loses money | 16.1% | **7.3%** | −8.8 pp |

*Values are expected net benefit (retained value minus offer cost) at an assumed 50% offer success rate.*

**Key findings**

1. **Risk alone is the wrong target.** The best pilot accounts combine above-average risk with high lifetime value and cheap offers. None of them is the single riskiest account.
2. **The answer is provably optimal and robust.** It is verified against exhaustive search, and the recommended pilot portfolio is identical across every tested offer success rate from 10% to 95%.
3. **Targeting quality matters more as margins thin.** On the full base, random targeting barely breaks even (a 36% chance of a loss), while the optimised campaign is profitable in all 20,000 simulated scenarios.
4. **Budget has diminishing returns.** At €25k, the next euro of budget returns about **€0.59** of net benefit, down from €1.05 at €10k. This gives a clear basis for sizing future budgets.
5. **Where the budget goes:** monthly-contract subscribers, Basic and Standard plans, and paid-search and partner-bundle acquisitions. These are high-churn segments with affordable offers.

---

### Report structure

| Section | Question answered |
|---|---|
| 1. Business context | Why is "contact the riskiest customers" not good enough? |
| 2. Analytical approach | How do we turn risk scores into a financially optimal decision? |
| 3. Setup | Configuration and assumptions in one place |
| **Part A: Pilot cohort** (30 high-value accounts) | Does optimisation beat the current rule, and is the answer provably optimal and robust? |
| **Part B: Full subscriber base** (5,000 accounts) | What drives churn, how good is the score, and where should a €25k budget go? |
| 4. Recommendations | What should the client do next? |
| 5. Limitations | What the analysis can and cannot claim |
| Appendix | Reproducibility and exported artefacts |
""")

# ---------------------------------------------------------------- 1
md(r"""
# 1. Business context

Aurvandil Media loses a meaningful share of its subscribers every year. The retention team can win some of them back with incentives such as free months or discounted upgrades. Every incentive costs money, and the **quarterly retention budget is fixed**.

The team's current rule is **"contact the subscribers with the highest churn score first"**. It is intuitive, but it ignores two things:

| Blind spot | Consequence |
|---|---|
| **Value**: a high-risk subscriber may be worth very little | Budget is spent protecting low-value accounts |
| **Cost**: offers differ in price by plan | Cheap, high-value saves are crowded out by expensive ones |
| **Combination**: taking customers one at a time ignores how they fit together | Budget is left unused or allocated to a sub-optimal mix |

The decision is therefore not *"who is most likely to leave?"* but:

> **Which combination of subscribers maximises total expected net benefit without exceeding the budget?**

This is a budget-constrained combinatorial optimisation problem, the **0/1 Knapsack problem**.
""")

# ---------------------------------------------------------------- 2
md(r"""
# 2. Analytical approach

### 2.1 Customer economics

For every subscriber $i$, with churn score $P_i$, lifetime value $CLV_i$, offer cost $C_i$ and assumed offer success rate $r$:

$$
ERV_i = P_i \times CLV_i \times r \qquad\qquad ENB_i = ERV_i - C_i \qquad\qquad r^{*}_i = \frac{C_i}{P_i \times CLV_i}
$$

* **ERV**: *expected retained value*, the value at risk times the chance the offer prevents the loss.
* **ENB**: *expected net benefit*, ERV minus the offer cost. This is the quantity we maximise.
* **r\***: *break-even success rate*, the minimum offer effectiveness at which the offer pays for itself.

### 2.2 Portfolio optimisation

$$
\max_{x}\; \sum_i x_i \, ENB_i \quad \text{subject to} \quad \sum_i x_i \, C_i \le B, \qquad x_i \in \{0,1\}
$$

The model is solved **exactly** by dynamic programming in $O(n \cdot B)$ time. Subscribers with $ENB_i \le 0$ are excluded, because they can only lower the objective.

### 2.3 Strategies compared

| Strategy | Rule | Role |
|---|---|---|
| Random | Seeded shuffle, take each subscriber whose cost still fits | Floor / sanity check |
| Highest churn probability | Rank by score, take each subscriber whose cost still fits | **Client's current practice** |
| Value density (greedy) | Rank by ENB ÷ cost | Strong heuristic benchmark |
| **Knapsack optimisation** | Exact optimum | **Proposed method** |

### 2.4 Evaluation

* **Primary metric:** total expected net benefit (€).
* **Supporting metrics:** spend, ROI (ENB ÷ spend), share of the optimum, and the probability that the realised result is a loss (Monte Carlo).
* **Retrospective only:** actual churners among the selected subscribers. This is descriptive and is not evidence that the offers *caused* retention.
""")

# ---------------------------------------------------------------- 3
md(r"""
# 3. Setup and assumptions

All logic lives in the tested `odin` package (`src/odin/`), so this report and the command-line tool always produce identical numbers. Every scenario assumption is declared once, below.
""")
code(r"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)

# Resolve the repository root so the report runs from any working directory.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "odin").exists())
sys.path.insert(0, str(ROOT / "src"))

import odin
from odin import add_expected_value, brute_force_selection, knapsack_selection, load_customers
from odin.economics import break_even_success_rate
from odin.evaluation import (compare_strategies, evaluate_selection, optimality_gap, run_strategies,
                             score_diagnostics, segment_allocation)
from odin.sensitivity import (budget_sensitivity, portfolio_stability, sensitivity_grid,
                              success_rate_sensitivity)
from odin.simulation import simulate_net_benefit, summarise_simulations
from odin.data import validation_report

# ---- Scenario assumptions -------------------------------------------------
SUCCESS_RATE = 0.50          # assumed probability an offer prevents a churn (scenario input)
SEED = 42                    # reproducibility of random baseline and simulations
MC_RUNS = 20_000             # Monte Carlo simulations per portfolio

PILOT_FILE = Path("data/raw/project_odin_customers.csv")
PILOT_BUDGET = 500           # € for the 30-account pilot

BASE_FILE = Path("data/synthetic/customer_base_5000.csv")
BASE_BUDGET = 25_000         # € quarterly retention budget for the full base

# ---- Output locations (relative to the repository root) -------------------
OUT = Path("outputs/report")
FIG, TAB = OUT / "figures", OUT / "tables"
(ROOT / FIG).mkdir(parents=True, exist_ok=True)
(ROOT / TAB).mkdir(parents=True, exist_ok=True)

COLORS = {"Random": "#9AA5B1", "Highest Churn Probability": "#E07A5F",
          "Value Density (Greedy)": "#F2CC8F", "Knapsack Optimization": "#3D5A80"}
ACCENT, MUTED = "#3D5A80", "#9AA5B1"
plt.rcParams.update({"figure.figsize": (10, 5), "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

def save(fig, name):
    fig.savefig(ROOT / FIG / name)
    plt.show()

def euro(v):
    return f"€{v:,.0f}"

print(f"odin package v{odin.__version__} | pandas {pd.__version__} | numpy {np.__version__}")
print(f"Assumed offer success rate: {SUCCESS_RATE:.0%} | pilot budget: {euro(PILOT_BUDGET)} | "
      f"full-base budget: {euro(BASE_BUDGET)}")
""")

# ================================================================ PART A
md(r"""
---
# Part A: Pilot cohort (30 high-value accounts)

Account management flagged 30 high-value subscribers for a pilot campaign with a **€500 budget**. The pilot is small enough to inspect every decision by hand and to **prove** the optimiser's answer by exhaustive search.
""")

md("## A1. Data validation")
code(r"""
pilot_raw = load_customers(ROOT / PILOT_FILE)
checks = validation_report(pilot_raw)
display(pd.DataFrame({"Check": list(checks), "Result": ["PASS" if v else "FAIL" for v in checks.values()]})
        .set_index("Check"))
print(f"{len(pilot_raw)} accounts | {pilot_raw['Actual_Churn'].sum()} observed churners")
""")
md(r"""
All structural checks pass: unique IDs, no missing values, binary outcomes, valid probabilities and positive whole-euro costs. These are *technical* validity checks. They confirm that the file is usable, not that it represents reality.
""")

md("## A2. Pilot profile")
code(r"""
display(pilot_raw[["Predicted_Churn_Probability", "Estimated_CLV", "Retention_Offer_Cost"]]
        .describe().T[["mean", "std", "min", "50%", "max"]])

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, (col, title) in zip(axes, [("Predicted_Churn_Probability", "Churn score"),
                                   ("Estimated_CLV", "Customer lifetime value (€)"),
                                   ("Retention_Offer_Cost", "Offer cost (€)")]):
    ax.hist(pilot_raw[col], bins=8, color=ACCENT, edgecolor="white")
    ax.set_title(title); ax.set_ylabel("Accounts")
fig.suptitle("Pilot cohort: input distributions", fontweight="bold", y=1.04)
save(fig, "A2_pilot_profile.png")
""")
md(r"""
Risk, value and cost all vary widely and are only weakly related to each other. That is exactly the situation in which ranking on risk alone goes wrong.
""")

md("## A3. Customer economics")
code(r"""
pilot = add_expected_value(pilot_raw, SUCCESS_RATE)
pilot["Break_Even_Success_Rate"] = break_even_success_rate(pilot)

print(f"Offers with positive expected net benefit: {(pilot['Expected_Net_Benefit'] > 0).sum()} of {len(pilot)}")
display(pilot.sort_values("Expected_Net_Benefit", ascending=False)
        [["Customer_ID", "Predicted_Churn_Probability", "Estimated_CLV", "Retention_Offer_Cost",
          "Expected_Retained_Value", "Expected_Net_Benefit", "Break_Even_Success_Rate"]]
        .head(10).style.format({"Predicted_Churn_Probability": "{:.2f}", "Estimated_CLV": "€{:,.0f}",
                                "Retention_Offer_Cost": "€{:,.0f}", "Expected_Retained_Value": "€{:,.0f}",
                                "Expected_Net_Benefit": "€{:,.0f}", "Break_Even_Success_Rate": "{:.0%}"})
        .hide(axis="index"))
""")
md(r"""
The ten most valuable offers have break-even success rates of 17% or less, so they pay off even if the offer works far less often than the assumed 50%. Only **two of the five riskiest accounts** make this list. The other three are worth too little relative to the cost of their offer.
""")

md("## A4. Strategy comparison")
code(r"""
pilot_sel = run_strategies(pilot, PILOT_BUDGET, seed=SEED)
pilot_cmp = compare_strategies(pilot, pilot_sel)
pilot_cmp["Share of Optimum"] = optimality_gap(pilot_cmp)
display(pilot_cmp.style.format({"Selected Customers": "{:.0f}", "Total Spending (€)": "€{:,.0f}",
        "Expected Retained Value (€)": "€{:,.0f}", "Expected Net Benefit (€)": "€{:,.0f}", "ROI": "{:.2f}x",
        "Actual Churners Captured": "{:.0f}", "Actual Churn Capture Rate": "{:.0%}",
        "Share of Optimum": "{:.0%}"}))

fig, ax = plt.subplots()
vals = pilot_cmp["Expected Net Benefit (€)"]
bars = ax.bar(vals.index, vals.values, color=[COLORS[i] for i in vals.index])
ax.bar_label(bars, [euro(v) for v in vals.values], padding=3)
ax.set_title(f"Pilot: expected net benefit by strategy ({euro(PILOT_BUDGET)} budget)")
ax.set_ylabel("Expected net benefit (€)"); ax.tick_params(axis="x", rotation=10)
save(fig, "A4_pilot_strategy_comparison.png")
""")
md(r"""
For almost the same spend, the optimised portfolio is worth about **2.9 times the client's current risk-first rule** and **5.3 times random selection**. The greedy value-density heuristic happens to reach the optimum on this cohort. Section A6 shows that it does not at every budget.
""")

md("## A5. Recommended pilot portfolio")
code(r"""
pilot_ids = pilot_sel["Knapsack Optimization"]
portfolio = pilot[pilot["Customer_ID"].isin(pilot_ids)].sort_values("Expected_Net_Benefit", ascending=False)
display(portfolio[["Customer_ID", "Predicted_Churn_Probability", "Estimated_CLV", "Retention_Offer_Cost",
                   "Expected_Net_Benefit", "Break_Even_Success_Rate"]]
        .style.format({"Predicted_Churn_Probability": "{:.2f}", "Estimated_CLV": "€{:,.0f}",
                       "Retention_Offer_Cost": "€{:,.0f}", "Expected_Net_Benefit": "€{:,.0f}",
                       "Break_Even_Success_Rate": "{:.0%}"}).hide(axis="index"))
spend = portfolio["Retention_Offer_Cost"].sum()
print(f"{len(portfolio)} accounts | spend {euro(spend)} of {euro(PILOT_BUDGET)} | "
      f"expected net benefit {euro(portfolio['Expected_Net_Benefit'].sum())}")

fig, ax = plt.subplots(figsize=(9, 6))
picked = pilot["Customer_ID"].isin(pilot_ids)
for mask, color, label in [(~picked, MUTED, "Not selected"), (picked, ACCENT, "Recommended")]:
    ax.scatter(pilot.loc[mask, "Predicted_Churn_Probability"], pilot.loc[mask, "Estimated_CLV"],
               s=pilot.loc[mask, "Retention_Offer_Cost"] * 2, c=color, alpha=0.85, edgecolor="white", label=label)
for _, r in pilot[picked].iterrows():
    ax.annotate(r["Customer_ID"], (r["Predicted_Churn_Probability"], r["Estimated_CLV"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8)
ax.set_xlabel("Churn score"); ax.set_ylabel("Customer lifetime value (€)")
ax.set_title("Risk vs value: the recommended accounts (bubble size = offer cost)")
ax.legend(frameon=False, loc="lower left")
save(fig, "A5_pilot_risk_vs_value.png")
""")
md(r"""
The recommended accounts sit in the **upper-right**: they combine above-average risk with the highest lifetime values and moderate offer costs. None of them is the single riskiest account.
""")

md("## A6. Verification and heuristic gap")
code(r"""
# 1) Prove optimality: exhaustive search over the 20 highest-benefit candidates (2^20 ≈ 1M subsets).
top20 = pilot[pilot["Expected_Net_Benefit"] > 0].nlargest(20, "Expected_Net_Benefit")
assert sorted(knapsack_selection(top20, PILOT_BUDGET)) == sorted(brute_force_selection(top20, PILOT_BUDGET, max_items=20))
print("Verified: dynamic programming == exhaustive search.")

# 2) How close do the heuristics get across budgets?
rows = []
for b in range(50, 1001, 50):
    opt = evaluate_selection(pilot, knapsack_selection(pilot, b))["Expected Net Benefit (€)"]
    sel = run_strategies(pilot, b, seed=SEED)
    rows.append({"Budget (€)": b, **{k: evaluate_selection(pilot, v)["Expected Net Benefit (€)"] / opt
                                     for k, v in sel.items() if k != "Knapsack Optimization"}})
gap = pd.DataFrame(rows).set_index("Budget (€)")
display(gap.agg(["min", "mean"]).style.format("{:.1%}"))

fig, ax = plt.subplots()
for col in gap.columns:
    ax.plot(gap.index, gap[col] * 100, marker="o", ms=4, color=COLORS[col], label=col)
ax.axhline(100, color="black", lw=1, ls="--", label="Knapsack optimum")
ax.set_ylabel("% of optimal net benefit"); ax.set_xlabel("Budget (€)")
ax.set_title("Heuristics vs. the exact optimum across budgets"); ax.legend(frameon=False)
save(fig, "A6_heuristic_gap.png")
""")
md(r"""
The optimiser's answer is **verified** against exhaustive search. The risk-first rule captures only a fraction of the achievable value at every budget. The value-density heuristic is strong but can still leave close to a tenth of the value on the table at some budgets, which is why the exact solver is recommended.
""")

md("## A7. Sensitivity to assumptions")
code(r"""
display(success_rate_sensitivity(pilot_raw, [0.30, 0.50, 0.70], PILOT_BUDGET).style
        .format({"Assumed Success Rate": "{:.0%}", "Spending (€)": "€{:,.0f}",
                 "Expected Net Benefit (€)": "€{:,.0f}"}).hide(axis="index"))
display(budget_sensitivity(pilot_raw, [350, 500, 650], SUCCESS_RATE).style
        .format({"Assumed Success Rate": "{:.0%}", "Spending (€)": "€{:,.0f}",
                 "Expected Net Benefit (€)": "€{:,.0f}"}).hide(axis="index"))

rates = np.round(np.arange(0.10, 0.951, 0.05), 2)
grid = sensitivity_grid(pilot_raw, rates, range(100, 1001, 100))
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(grid.values, aspect="auto", origin="lower", cmap="viridis")
ax.set_xticks(range(grid.shape[1]), [f"€{b}" for b in grid.columns])
ax.set_yticks(range(grid.shape[0]), [f"{r:.0%}" for r in grid.index])
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid.values[i, j]
        ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=7,
                color="black" if v > grid.values.max() * 0.6 else "white")
ax.set_xlabel("Budget"); ax.set_ylabel("Assumed offer success rate")
ax.set_title("Optimal expected net benefit (€) across scenarios")
fig.colorbar(im, ax=ax, label="€")
save(fig, "A7_sensitivity_heatmap.png")

stability = portfolio_stability(pilot_raw, rates, PILOT_BUDGET)
display(stability.head(8).style.format({"Selection Frequency": "{:.0%}"}).hide(axis="index"))
""")
md(r"""
* Across **every** tested success rate from 10% to 95%, the **recommended accounts do not change**. Only the size of the benefit changes, so the recommendation is robust to the largest uncertainty in the model.
* More budget always helps, but with **diminishing returns**: the first €500 buys far more value than the next €500.
* The stability table confirms that all six recommended accounts are selected in 100% of scenarios. They are the "always fund" core.
""")

md("## A8. Risk of the recommended portfolio")
code(r"""
pilot_sims = {k: simulate_net_benefit(pilot, v, SUCCESS_RATE, runs=MC_RUNS, seed=SEED) for k, v in pilot_sel.items()}
pilot_risk = summarise_simulations(pilot_sims)
display(pilot_risk.style.format({c: "€{:,.0f}" for c in pilot_risk.columns if "€" in c} | {"Probability of Loss": "{:.1%}"}))

fig, ax = plt.subplots()
bins = np.linspace(min(s.min() for s in pilot_sims.values()), max(s.max() for s in pilot_sims.values()), 60)
for k in ["Random", "Highest Churn Probability", "Knapsack Optimization"]:
    ax.hist(pilot_sims[k], bins=bins, alpha=0.55, color=COLORS[k], label=f"{k} (mean {euro(pilot_sims[k].mean())})")
ax.axvline(0, color="black", ls="--", lw=1)
ax.set_xlabel("Realised net benefit (€)"); ax.set_ylabel("Simulated campaigns")
ax.set_title("Pilot: distribution of realised outcomes (Monte Carlo)"); ax.legend(frameon=False)
save(fig, "A8_pilot_monte_carlo.png")
""")
md(r"""
The simulated mean matches the analytical expectation, which validates the model. The optimised portfolio has the **highest expected value and the lowest probability of a loss**. The worst case is bounded by the campaign spend.
""")

# ================================================================ PART B
md(r"""
---
# Part B: Full subscriber base (5,000 accounts)

Following the pilot, the method is applied to the whole subscriber base with a **€25,000 quarterly budget**. The base dataset carries the commercial and behavioural attributes the retention team works with (plan, contract, region, acquisition channel, tenure, viewing hours, support tickets and payment failures), so the recommendation can be explained segment by segment.
""")

md("## B1. Data overview")
code(r"""
base_raw = load_customers(ROOT / BASE_FILE)
failed = [k for k, v in validation_report(base_raw).items() if not v]
print(f"{len(base_raw):,} subscribers | {base_raw.shape[1]} attributes | validation: "
      f"{'all checks passed' if not failed else failed}")
display(base_raw.head(5).style.hide(axis="index").format(precision=2))

mix = pd.concat({c: base_raw[c].value_counts(normalize=True) for c in ["Plan", "Contract", "Region",
                 "Acquisition_Channel"]}).rename("Share").to_frame()
display(mix.style.format("{:.0%}"))
""")

md("## B2. What drives churn?")
code(r"""
base_raw["Tenure_Band"] = pd.cut(base_raw["Tenure_Months"], [0, 6, 12, 24, 48, 120],
                                 labels=["0-6", "7-12", "13-24", "25-48", "49+"])
base_raw["Usage_Band"] = pd.qcut(base_raw["Viewing_Hours_30d"], 4, labels=["Low", "Medium", "High", "Very high"])
base_raw["Payment_Issue"] = np.where(base_raw["Payment_Failures_90d"] > 0, "Yes", "No")

drivers = ["Contract", "Plan", "Acquisition_Channel", "Tenure_Band", "Usage_Band", "Payment_Issue"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
overall = base_raw["Actual_Churn"].mean()
for ax, col in zip(axes.ravel(), drivers):
    rate = base_raw.groupby(col, observed=True)["Actual_Churn"].mean()
    bars = ax.bar(rate.index.astype(str), rate.values * 100,
                  color=[ACCENT if v > overall else MUTED for v in rate.values])
    ax.bar_label(bars, [f"{v:.0%}" for v in rate.values], padding=2, fontsize=9)
    ax.axhline(overall * 100, color="black", ls="--", lw=1)
    ax.set_title(col.replace("_", " ")); ax.set_ylabel("Churn rate (%)"); ax.tick_params(axis="x", rotation=15)
fig.suptitle(f"Observed churn rate by segment (dashed line = base average {overall:.0%})",
             fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "B2_churn_drivers.png")
""")
md(r"""
Churn concentrates in subscribers on **monthly contracts**, those with **low viewing hours**, **short tenure**, **failed payments**, and those acquired through **paid search and partner bundles**. These are the segments where a retention offer has the most value at stake. Value and cost still decide who is worth an offer, which is the job of the optimiser.
""")

md("## B3. How reliable is the churn score?")
code(r"""
diag, calib = score_diagnostics(base_raw, bins=10)
display(pd.Series(diag).to_frame("Value").style.format("{:.3f}"))

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1, label="Perfect calibration")
ax.plot(calib["Mean_Predicted"], calib["Observed_Churn_Rate"], marker="o", color=ACCENT, lw=2, label="Churn score (deciles)")
ax.set_xlabel("Mean predicted churn probability"); ax.set_ylabel("Observed churn rate")
ax.set_title(f"Score calibration (ROC AUC = {diag['ROC AUC']:.2f})"); ax.legend(frameon=False)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "B3_score_calibration.png")
""")
md(r"""
The score separates churners from non-churners usefully (ROC AUC of about 0.70), and its deciles follow the diagonal reasonably closely. It is good enough to drive an expected-value decision. The remaining calibration error is also why Section B7 stress-tests the result instead of trusting point estimates.
""")

md("## B4. Customer economics at scale")
code(r"""
base = add_expected_value(base_raw, SUCCESS_RATE)
base["Break_Even_Success_Rate"] = break_even_success_rate(base)
positive = base["Expected_Net_Benefit"] > 0
print(f"Offers with positive expected net benefit: {positive.sum():,} of {len(base):,} ({positive.mean():.0%})")
print(f"Cost of funding every positive offer: {euro(base.loc[positive, 'Retention_Offer_Cost'].sum())} "
      f"vs. budget {euro(BASE_BUDGET)}, so the budget is binding")

econ = base.groupby("Plan").agg(Subscribers=("Customer_ID", "size"), Avg_CLV=("Estimated_CLV", "mean"),
                                Avg_Offer_Cost=("Retention_Offer_Cost", "mean"),
                                Avg_Churn_Score=("Predicted_Churn_Probability", "mean"),
                                Avg_ENB=("Expected_Net_Benefit", "mean"),
                                Share_Positive=("Expected_Net_Benefit", lambda s: (s > 0).mean()))
display(econ.style.format({"Avg_CLV": "€{:,.0f}", "Avg_Offer_Cost": "€{:,.0f}", "Avg_Churn_Score": "{:.2f}",
                           "Avg_ENB": "€{:,.1f}", "Share_Positive": "{:.0%}"}))
""")

md("## B5. Optimisation at €25,000")
code(r"""
base_sel = run_strategies(base, BASE_BUDGET, seed=SEED)
base_cmp = compare_strategies(base, base_sel)
base_cmp["Share of Optimum"] = optimality_gap(base_cmp)
display(base_cmp.style.format({"Selected Customers": "{:,.0f}", "Total Spending (€)": "€{:,.0f}",
        "Expected Retained Value (€)": "€{:,.0f}", "Expected Net Benefit (€)": "€{:,.0f}", "ROI": "{:.2f}x",
        "Actual Churners Captured": "{:,.0f}", "Actual Churn Capture Rate": "{:.1%}",
        "Share of Optimum": "{:.1%}"}))

uplift = (base_cmp.loc["Knapsack Optimization", "Expected Net Benefit (€)"]
          - base_cmp.loc["Highest Churn Probability", "Expected Net Benefit (€)"])
display(Markdown(f"**Value added vs. the current risk-first rule: {euro(uplift)} per campaign "
                 f"(+{uplift / base_cmp.loc['Highest Churn Probability', 'Expected Net Benefit (€)']:.0%}) "
                 f"with the same budget.**"))

fig, ax = plt.subplots()
vals = base_cmp["Expected Net Benefit (€)"]
bars = ax.bar(vals.index, vals.values, color=[COLORS[i] for i in vals.index])
ax.bar_label(bars, [euro(v) for v in vals.values], padding=3)
ax.set_title(f"Full base: expected net benefit by strategy ({euro(BASE_BUDGET)} budget)")
ax.set_ylabel("Expected net benefit (€)"); ax.tick_params(axis="x", rotation=10)
save(fig, "B5_base_strategy_comparison.png")
""")
md(r"""
Margins are much thinner on the mass-market base than in the pilot: random targeting turns a €25k budget into under €1k of net value. The risk-first rule does far better, but it still leaves about **€3,000 (13% of the achievable value)** on the table in every campaign. The greedy heuristic matches the optimum here, because thousands of small, similar offers make the budget easy to pack. The exact solver costs nothing extra (it runs in well under a second) and removes any doubt.
""")

md("## B6. Where does the budget go?")
code(r"""
opt_ids = base_sel["Knapsack Optimization"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col in zip(axes, ["Plan", "Contract", "Acquisition_Channel"]):
    alloc = segment_allocation(base, opt_ids, col).sort_values("Selection Rate")
    bars = ax.barh(alloc.index, alloc["Selection Rate"] * 100, color=ACCENT)
    ax.bar_label(bars, [f"{r:.0%} · {euro(s)}" for r, s in zip(alloc["Selection Rate"], alloc["Spend (€)"])],
                 padding=3, fontsize=9)
    ax.set_title(col.replace("_", " ")); ax.set_xlabel("% of segment receiving an offer")
    ax.set_xlim(0, alloc["Selection Rate"].max() * 140)
fig.suptitle("Optimal allocation by segment (label = selection rate · spend)", fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "B6_allocation_by_segment.png")

region = segment_allocation(base, opt_ids, "Region")
display(region.style.format({"Customers": "{:,.0f}", "Selected": "{:,.0f}", "Spend (€)": "€{:,.0f}",
                             "Expected Net Benefit (€)": "€{:,.0f}", "Selection Rate": "{:.0%}",
                             "Share of Spend": "{:.0%}"}))
""")
md(r"""
* **Contract:** monthly subscribers receive offers at a higher rate (18% vs 14%) and take almost three-quarters of the spend.
* **Plan:** Basic and Standard subscribers are targeted most. **Premium is targeted least (8%)**, because its two-and-a-half-month offer is expensive relative to the value at risk. Premium retention therefore needs a cheaper incentive, such as a temporary downgrade path, rather than more budget.
* **Channel:** paid-search and partner-bundle acquisitions are targeted at four times the rate of referrals, which churn rarely.
* **Region** allocation broadly follows market size, so region is not a primary lever.
""")

md("## B7. How large should the budget be?")
code(r"""
budgets = list(range(5_000, 60_001, 5_000))
frontier = budget_sensitivity(base_raw, budgets, SUCCESS_RATE)
frontier["Marginal Return per €"] = frontier["Expected Net Benefit (€)"].diff() / frontier["Budget (€)"].diff()
display(frontier.drop(columns=["Selected IDs", "Assumed Success Rate"]).style.format(
    {"Budget (€)": "€{:,.0f}", "Selected Customers": "{:,.0f}", "Spending (€)": "€{:,.0f}",
     "Expected Net Benefit (€)": "€{:,.0f}", "Marginal Return per €": "{:.2f}"}, na_rep="").hide(axis="index"))

fig, ax1 = plt.subplots()
ax1.plot(frontier["Budget (€)"] / 1000, frontier["Expected Net Benefit (€)"] / 1000, marker="o", color=ACCENT, lw=2)
ax1.axvline(BASE_BUDGET / 1000, color="black", ls="--", lw=1)
ax1.set_xlabel("Budget (€ thousand)"); ax1.set_ylabel("Optimal expected net benefit (€ thousand)", color=ACCENT)
ax2 = ax1.twinx()
ax2.bar(frontier["Budget (€)"] / 1000, frontier["Marginal Return per €"], width=3, alpha=0.3, color="#E07A5F")
ax2.set_ylabel("Marginal net benefit per extra €", color="#E07A5F"); ax2.spines["right"].set_visible(True)
ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)
ax1.set_title("Budget frontier: total benefit and the return on the next euro")
save(fig, "B7_budget_frontier.png")
""")
md(r"""
Each additional euro of budget earns less than the one before. As long as the marginal return per euro stays above zero, a larger budget still adds value, but at a declining rate. The client can set the budget where the marginal return falls to its hurdle rate. Beyond the point where every positive-value offer is funded, extra budget adds nothing.
""")

md("## B8. Risk and retrospective check")
code(r"""
base_sims = {k: simulate_net_benefit(base, v, SUCCESS_RATE, runs=MC_RUNS, seed=SEED) for k, v in base_sel.items()}
base_risk = summarise_simulations(base_sims)
display(base_risk.style.format({c: "€{:,.0f}" for c in base_risk.columns if "€" in c} | {"Probability of Loss": "{:.1%}"}))

fig, ax = plt.subplots()
for k in ["Random", "Highest Churn Probability", "Knapsack Optimization"]:
    ax.hist(base_sims[k] / 1000, bins=60, alpha=0.6, color=COLORS[k], label=f"{k} (mean {euro(base_sims[k].mean())})")
ax.set_xlabel("Realised net benefit (€ thousand)"); ax.set_ylabel("Simulated campaigns")
ax.set_title("Full base: distribution of realised outcomes (Monte Carlo)"); ax.legend(frameon=False)
save(fig, "B8_base_monte_carlo.png")
""")
md(r"""
With hundreds of offers, outcomes average out: the optimised campaign is **profitable in every simulated scenario**, and in 95% of scenarios it returns at least about €18,000. Random targeting loses money in more than a third of scenarios. The retrospective churn-capture figures in B5 are descriptive only. Proving causal impact requires a randomised hold-out group (see Section 4).
""")

# ---------------------------------------------------------------- 4/5
md(r"""
---
# 4. Recommendations

1. **Replace the risk-first targeting rule with value-based optimisation.** Under the stated assumptions this adds about €3,000 (+15%) of expected net benefit per €25k campaign, with no increase in spend.
2. **Focus offers where risk, value and offer cost line up:** monthly contracts, Basic and Standard plans, and paid-search and partner-bundle acquisitions. **Redesign the Premium offer**: at its current cost it rarely pays back.
3. **Run the next campaign as a randomised test.** Hold out a random 10–20% of the selected subscribers. Comparing churn between treated and held-out subscribers measures the true offer success rate, the single most important assumption. Feed that estimate back into the model.
4. **Set the budget using the frontier (B7).** Fund the campaign up to the point where the marginal return per euro falls to the client's hurdle rate.
5. **Operationalise.** The `odin` package exposes the full pipeline (`python -m odin analyze`), so the list can be refreshed every cycle from the latest churn scores.

# 5. Limitations

* **Synthetic data.** Both datasets are synthetic replicas generated for this case study. Results illustrate the method and are not measured business outcomes.
* **Assumed offer effectiveness.** The 50% success rate is a scenario input, which is why the report stress-tests it from 10% to 95%.
* **Simplified economics.** CLV is margin-based but undiscounted. Partial retention, offer-specific effects and cannibalisation are not modelled.
* **No causal claim.** Observed churn among selected subscribers is descriptive. Causal impact requires the experiment in recommendation 3.
* **Single offer per subscriber.** Multiple offer types per subscriber, contact-capacity limits and segment quotas would turn the problem into a general integer program. The roadmap in the README covers this.
""")

# ---------------------------------------------------------------- appendix
md(r"""
---
# Appendix: Reproducibility and exported artefacts

Every table below is written to `outputs/report/tables/` and every chart to `outputs/report/figures/`. Paths are shown relative to the repository root.
""")
code(r"""
exports = {
    "pilot_customer_economics.csv": pilot,
    "pilot_strategy_comparison.csv": pilot_cmp,
    "pilot_recommended_portfolio.csv": portfolio,
    "pilot_sensitivity_grid.csv": grid,
    "pilot_portfolio_stability.csv": stability,
    "pilot_monte_carlo_risk.csv": pilot_risk,
    "pilot_heuristic_gap.csv": gap,
    "base_strategy_comparison.csv": base_cmp,
    "base_recommended_portfolio.csv": base[base["Customer_ID"].isin(opt_ids)],
    "base_allocation_by_region.csv": region,
    "base_budget_frontier.csv": frontier,
    "base_monte_carlo_risk.csv": base_risk,
    "base_score_calibration.csv": calib,
}
for name, table in exports.items():
    table.to_csv(ROOT / TAB / name, index=not isinstance(table.index, pd.RangeIndex))

print("Tables:")
for p in sorted((ROOT / TAB).glob("*.csv")):
    print("  ", p.relative_to(ROOT))
print("Figures:")
for p in sorted((ROOT / FIG).glob("*.png")):
    print("  ", p.relative_to(ROOT))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, OUT)
print("wrote", OUT)
