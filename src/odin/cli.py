"""Command-line interface.

python -m odin analyze --data data/raw/project_odin_customers.csv --budget 500
python -m odin generate --n 5000 --out data/synthetic/customer_base_5000.csv
python -m odin target --data data/synthetic/customer_base_5000.csv --budget 25000 --multi-offer \
    --max-contacts 1500 --segment Region --max-share DACH=0.25 --out target_list.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from odin.config import ScenarioConfig
from odin.data import load_customers
from odin.pipeline import analyze, export, summary_text
from odin.synthetic import generate_customer_base, generate_customers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odin", description="Budget-constrained retention optimisation.")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Run the full analysis and export tables/figures.")
    a.add_argument("--data", default="data/raw/project_odin_customers.csv", type=Path)
    a.add_argument("--budget", default=500, type=int, help="Campaign budget in whole euros.")
    a.add_argument("--success-rate", default=0.50, type=float, help="Assumed offer success rate (0-1).")
    a.add_argument(
        "--budget-scenarios", default="350,500,650", help="Comma-separated budgets for one-way sensitivity."
    )
    a.add_argument("--seed", default=42, type=int)
    a.add_argument("--runs", default=10_000, type=int, help="Monte Carlo simulations.")
    a.add_argument("--out", default="outputs", type=Path)
    a.add_argument("--no-figures", action="store_true")

    t = sub.add_parser("target", help="Build the campaign target list under the chosen assumptions.")
    t.add_argument("--data", default="data/synthetic/customer_base_5000.csv", type=Path)
    t.add_argument("--budget", default=25_000, type=int, help="Campaign budget in whole euros.")
    effect = t.add_mutually_exclusive_group()
    effect.add_argument("--success-rate", default=0.50, type=float, help="Assumed offer success rate (0-1).")
    effect.add_argument(
        "--uplift", action="store_true", help="Per-subscriber uplift from a simulated randomised campaign."
    )
    t.add_argument("--multi-offer", action="store_true", help="Choose the best offer type per subscriber.")
    t.add_argument("--max-contacts", type=int, help="Maximum number of subscribers contacted.")
    t.add_argument("--segment", help="Column used for --max-share and --min-contacts (e.g. Region).")
    t.add_argument("--max-share", action="append", default=[], metavar="SEG=SHARE", help="Spend cap.")
    t.add_argument("--min-contacts", action="append", default=[], metavar="SEG=N", help="Contact minimum.")
    t.add_argument("--out", default=Path("outputs/target_list.csv"), type=Path)

    g = sub.add_parser("generate", help="Generate a synthetic customer dataset.")
    g.add_argument(
        "--profile",
        choices=["base", "simple"],
        default="base",
        help="'base': rich subscriber base with segments; 'simple': core columns only.",
    )
    g.add_argument("--n", default=5_000, type=int)
    g.add_argument("--seed", default=None, type=int)
    g.add_argument("--out", default=Path("data/synthetic/customer_base_5000.csv"), type=Path)
    return parser


def _pairs(items: list[str], cast, flag: str) -> dict:
    out = {}
    for item in items:
        key, sep, value = item.rpartition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} expects SEGMENT=VALUE, got {item!r}")
        out[key] = cast(value)
    return out


def _target(args: argparse.Namespace) -> int:
    from odin.constraints import (
        CampaignConstraints,
        InfeasibleCampaignError,
        constrained_selection,
        menu_from_single_offer,
    )
    from odin.economics import add_expected_value
    from odin.offers import build_offer_menu, multi_offer_selection, offer_mix, rescale_success_rates

    data = load_customers(args.data)
    if args.uplift and args.multi_offer:
        raise SystemExit("--uplift models the standard offer only; it cannot be combined with --multi-offer")
    if args.uplift:
        from odin.uplift import TLearner, add_uplift_value, simulate_campaign_experiment

        cohort = generate_customer_base(20_000, seed=101)
        model = TLearner().fit(simulate_campaign_experiment(cohort, seed=1))
        economics = add_uplift_value(data, model.predict(data))
    else:
        economics = add_expected_value(data, args.success_rate)

    if args.multi_offer:
        menu = rescale_success_rates(build_offer_menu(data), args.success_rate)
    else:
        menu = menu_from_single_offer(economics)

    rules = CampaignConstraints(
        max_contacts=args.max_contacts,
        segment_column=args.segment,
        max_spend_share=_pairs(args.max_share, float, "--max-share"),
        min_contacts=_pairs(args.min_contacts, int, "--min-contacts"),
    )
    if args.segment and args.segment not in data.columns:
        raise SystemExit(f"Unknown segment column: {args.segment}")
    segments = data.set_index("Customer_ID")[args.segment] if args.segment else None
    constrained = args.max_contacts is not None or rules.max_spend_share or rules.min_contacts
    try:
        portfolio = (
            constrained_selection(menu, args.budget, rules, segments)
            if constrained
            else multi_offer_selection(menu, args.budget)
        )
    except InfeasibleCampaignError as exc:
        print(f"Infeasible: {exc}", file=sys.stderr)
        return 2

    target = portfolio.merge(data, on="Customer_ID").sort_values("Offer_Net_Benefit", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(args.out, index=False)
    spend, benefit = target["Offer_Cost"].sum(), target["Offer_Net_Benefit"].sum()
    print(f"Subscribers targeted : {len(target):,}")
    print(f"Spend                : €{spend:,.0f} of €{args.budget:,}")
    print(f"Expected net benefit : €{benefit:,.0f}")
    if args.multi_offer:
        print("\nOffer mix:\n" + offer_mix(portfolio).to_string())
    print(f"\nTarget list written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")  # headless rendering for CLI runs
    args = _parser().parse_args(argv)

    if args.command == "target":
        return _target(args)

    if args.command == "generate":
        if args.profile == "base":
            df = generate_customer_base(args.n, seed=7 if args.seed is None else args.seed)
        else:
            df = generate_customers(args.n, seed=2024 if args.seed is None else args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df):,} synthetic customers to {args.out}")
        return 0

    config = ScenarioConfig(
        budget=args.budget,
        success_rate=args.success_rate,
        random_seed=args.seed,
        budget_scenarios=tuple(int(b) for b in args.budget_scenarios.split(",")),
        monte_carlo_runs=args.runs,
    )
    result = analyze(load_customers(args.data), config)
    export(result, args.out, make_figures=not args.no_figures)
    print(summary_text(result, config))
    print(f"\nTables and figures written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
