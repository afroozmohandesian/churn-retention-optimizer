"""Command-line interface.

python -m odin analyze --data data/raw/project_odin_customers.csv --budget 500
python -m odin generate --n 1000 --out data/synthetic/odin_customers_1000.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from odin.config import ScenarioConfig
from odin.data import load_customers
from odin.pipeline import analyze, export, summary_text
from odin.synthetic import generate_customers


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

    g = sub.add_parser("generate", help="Generate a synthetic customer dataset.")
    g.add_argument("--n", default=1_000, type=int)
    g.add_argument("--seed", default=2024, type=int)
    g.add_argument("--out", default=Path("data/synthetic/odin_customers_1000.csv"), type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "generate":
        df = generate_customers(args.n, seed=args.seed)
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
