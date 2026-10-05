"""CLI for generating publication tables and figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipedream.analysis.visualize import generate_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication comparison tables and figures."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/raw/overnight_pilot"),
        help="Directory containing test.jsonl, baselines_report.json, etc.",
    )
    parser.add_argument(
        "--output-tables-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory to save summary tables (CSV and Markdown).",
    )
    parser.add_argument(
        "--output-figures-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to save publication figures.",
    )
    args = parser.parse_args(argv)

    artifacts = generate_all(
        results_dir=args.results_dir,
        output_tables_dir=args.output_tables_dir,
        output_figures_dir=args.output_figures_dir,
    )

    print("Generated publication artifacts:")
    for key, path in artifacts.items():
        print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
