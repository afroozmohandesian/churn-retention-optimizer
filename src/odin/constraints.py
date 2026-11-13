"""Operational constraints: contact capacity and segment quotas (integer programming).

The dynamic programmes in :mod:`odin.strategies` and :mod:`odin.offers` handle a
single budget constraint. Real campaigns add more: the call centre can only make
so many calls, no region may absorb more than a set share of the budget, and
strategic segments must receive a minimum number of contacts. With several
constraints the problem becomes a general 0/1 integer linear programme:

    max  sum_ij x_ij * ENB_ij
    s.t. sum_ij x_ij * C_ij               <= B              (budget)
         sum_j  x_ij                      <= 1   for all i   (one offer each)
         sum_ij x_ij                      <= K              (contact capacity)
         sum_{i in s, j} x_ij * C_ij      <= share_s * B    (spend cap per segment)
         sum_{i in s, j} x_ij             >= m_s            (minimum contacts per segment)
         x_ij in {0, 1}

solved exactly with the HiGHS branch-and-cut solver via ``scipy.optimize.milp``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from odin.strategies import _check_budget


class InfeasibleCampaignError(ValueError):
    """Raised when the constraints cannot all be satisfied at once."""


@dataclass(frozen=True)
class CampaignConstraints:
    """Operational limits on a retention campaign.

    ``max_spend_share`` and ``min_contacts`` are keyed by segment value of the
    column named in ``segment_column``.
    """

    max_contacts: int | None = None
    segment_column: str | None = None
    max_spend_share: Mapping[str, float] = field(default_factory=dict)
    min_contacts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_contacts is not None and self.max_contacts < 0:
            raise ValueError("max_contacts must be non-negative")
        if (self.max_spend_share or self.min_contacts) and not self.segment_column:
            raise ValueError("segment quotas need a segment_column")
        if any(not 0 <= s <= 1 for s in self.max_spend_share.values()):
            raise ValueError("max_spend_share values must be between 0 and 1")
        if any(m < 0 for m in self.min_contacts.values()):
            raise ValueError("min_contacts values must be non-negative")


def menu_from_single_offer(data: pd.DataFrame) -> pd.DataFrame:
    """Express the one-offer-per-subscriber problem in offer-menu form."""
    return pd.DataFrame(
        {
            "Customer_ID": data["Customer_ID"].to_numpy(),
            "Offer": "Standard offer",
            "Offer_Cost": data["Retention_Offer_Cost"].astype(int).to_numpy(),
            "Offer_Net_Benefit": data["Expected_Net_Benefit"].to_numpy(),
        }
    )


def constrained_selection(
    menu: pd.DataFrame,
    budget: int,
    constraints: CampaignConstraints | None = None,
    segments: pd.Series | None = None,
    time_limit: float = 60.0,
) -> pd.DataFrame:
    """Optimal offer assignment under budget and operational constraints.

    Parameters
    ----------
    menu
        Offer menu (see :func:`odin.offers.build_offer_menu` or
        :func:`menu_from_single_offer`).
    segments
        Segment label per subscriber, indexed by ``Customer_ID``; required when
        the constraints include segment quotas.

    Returns the chosen menu rows. Raises :class:`InfeasibleCampaignError` when
    no assignment satisfies every constraint (for example, minimum contacts
    that the budget cannot pay for).
    """
    capacity = _check_budget(budget)
    constraints = constraints or CampaignConstraints()

    # Offers with non-positive benefit can still be needed to meet a minimum
    # contact quota, so they are only dropped when there are no minimums.
    candidates = menu if constraints.min_contacts else menu.loc[menu["Offer_Net_Benefit"] > 0]
    candidates = candidates.reset_index()
    if candidates.empty:
        if any(constraints.min_contacts.values()):
            raise InfeasibleCampaignError("No candidate offers to meet the minimum contacts.")
        return menu.iloc[0:0].copy()

    n = len(candidates)
    cost = candidates["Offer_Cost"].astype(float).to_numpy()
    value = candidates["Offer_Net_Benefit"].astype(float).to_numpy()
    rows: list[sparse.spmatrix] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(row, lo, hi):
        rows.append(sparse.csr_matrix(row))
        lower.append(lo)
        upper.append(hi)

    add(cost.reshape(1, -1), -np.inf, capacity)

    customer_codes, _ = pd.factorize(candidates["Customer_ID"])
    one_each = sparse.csr_matrix((np.ones(n), (customer_codes, np.arange(n))))
    rows.append(one_each)
    lower.extend([-np.inf] * one_each.shape[0])
    upper.extend([1.0] * one_each.shape[0])

    if constraints.max_contacts is not None:
        add(np.ones((1, n)), -np.inf, constraints.max_contacts)

    if constraints.segment_column:
        if segments is None:
            raise ValueError("segments must be given for segment quotas")
        label = candidates["Customer_ID"].map(segments)
        for seg, share in constraints.max_spend_share.items():
            add((cost * (label == seg).to_numpy()).reshape(1, -1), -np.inf, share * capacity)
        for seg, minimum in constraints.min_contacts.items():
            add((label == seg).to_numpy().astype(float).reshape(1, -1), minimum, np.inf)

    result = milp(
        c=-value,
        constraints=LinearConstraint(sparse.vstack(rows).tocsr(), lower, upper),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 1e-6},
    )
    if result.status == 2:
        raise InfeasibleCampaignError("The campaign constraints cannot all be satisfied.")
    if result.x is None:
        raise RuntimeError(f"Solver did not return a solution: {result.message}")

    chosen = candidates.loc[result.x > 0.5, "index"]
    return menu.loc[sorted(chosen)].copy()


def shadow_cost(
    menu: pd.DataFrame,
    budget: int,
    constraints: CampaignConstraints,
    segments: pd.Series | None = None,
) -> dict[str, float]:
    """Expected net benefit with and without the operational constraints.

    The difference is the price of the constraints: what the business gives up
    to respect capacity and quotas.
    """
    free = constrained_selection(menu, budget)["Offer_Net_Benefit"].sum()
    bound = constrained_selection(menu, budget, constraints, segments)["Offer_Net_Benefit"].sum()
    return {
        "Unconstrained (€)": float(free),
        "Constrained (€)": float(bound),
        "Cost of Constraints (€)": float(free - bound),
        "Share Retained": float(bound / free) if free else 1.0,
    }
