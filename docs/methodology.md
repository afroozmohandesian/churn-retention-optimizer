# Methodology

## 1. Decision framing

The business controls a binary decision per customer: make a retention offer (`x_i = 1`) or not (`x_i = 0`). Offers consume a shared, fixed budget `B`. The objective is the total expected net financial benefit of the campaign.

## 2. Customer-level value

| Symbol | Meaning |
|---|---|
| `P_i` | probability customer *i* churns without intervention |
| `CLV_i` | estimated customer lifetime value (€) |
| `C_i` | offer cost (€) |
| `r` | probability the offer prevents a churn that would otherwise happen (scenario assumption) |

```
ERV_i = P_i · CLV_i · r          expected retained value
ENB_i = ERV_i − C_i              expected net benefit
r*_i  = C_i / (P_i · CLV_i)      break-even success rate
```

An offer is worth making in isolation only when `r > r*_i`.

## 3. Optimisation

```
maximise    Σ x_i · ENB_i
subject to  Σ x_i · C_i ≤ B
            x_i ∈ {0, 1}
```

This is the 0/1 Knapsack problem. It is NP-hard in general but pseudo-polynomial: with integer costs, dynamic programming solves it exactly in `O(n · B)` time and `O(n · B)` bits of memory for back-tracking.

Implementation notes (`odin.strategies.knapsack_selection`):

- Customers with `ENB ≤ 0` are removed first. They can only lower the objective, and the empty selection is always feasible.
- A single value vector `best[b]` is updated per customer with NumPy slicing. A boolean `take[i, b]` matrix records decisions for back-tracking.
- Ties are broken deterministically: an item is taken only if it improves the value by more than 1e-9. Customers are processed in ID order.
- Optimality is checked against exhaustive enumeration in the test suite.

## 4. Baselines

- **Random**: seeded shuffle, then first-fit.
- **Highest churn probability**: sort by `P_i` (stable), then first-fit. This is the common "risk-first" rule.
- **Value density**: sort by `ENB_i / C_i`, then first-fit. This is the textbook greedy knapsack heuristic. It is usually near-optimal, but it can fail when a high-density item blocks a better-fitting combination.

## 5. Evaluation metrics

| Metric | Role |
|---|---|
| Expected net benefit | **primary objective** |
| Spend, customers selected, ROI (= ENB / spend) | efficiency |
| Share of optimum | heuristic quality |
| Actual churners captured / capture rate | retrospective description only. Not causal, not model recall. |

## 6. Uncertainty

**Scenario analysis.** The optimal portfolio is re-solved for success rates from 10% to 95% and for a range of budgets. The two-way grid is non-decreasing in both dimensions, which the tests check.

**Portfolio stability.** This counts how often each customer is selected across success-rate scenarios. It separates "always worth it" customers from marginal ones.

**Monte Carlo.** For a fixed portfolio, each simulation draws:

```
churn_i  ~ Bernoulli(P_i)
saved_i  ~ Bernoulli(r)          (only relevant if churn_i = 1)
realised = Σ churn_i · saved_i · CLV_i − Σ C_i
```

The simulated mean converges to `Σ ENB_i`. The distribution adds the spread, the percentiles and `P(realised < 0)`. Offer costs are assumed to be incurred whether or not the customer would have churned.

## 7. Churn-score diagnostics (full base)

`odin.evaluation.score_diagnostics` reports:

- **ROC AUC**, computed with the rank-based Mann–Whitney formula.
- **Brier score.**
- **A decile calibration table** comparing the mean predicted probability with the observed churn rate.

An expected-value decision is only as good as the score's calibration, so these numbers are reported alongside the optimisation.

## 8. Budget frontier

The optimal ENB is solved for a ladder of budgets. The marginal return per extra euro, ΔENB / ΔB, is non-increasing. The budget should be set where this marginal return reaches the business's hurdle rate.

## 9. What a production version needs

1. A calibrated churn model, checked with a reliability curve and the Brier score.
2. A margin-based, discounted CLV.
3. Per-customer **uplift** (heterogeneous treatment effect) from randomised experiments, replacing the single `r`.
4. Operational constraints such as contact capacity, eligibility and fairness. These turn the problem into a general integer linear program.
5. Post-campaign monitoring that compares predicted and realised outcomes.
