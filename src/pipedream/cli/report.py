"""Create a comparison table from common-schema JSONL results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipedream.evaluation.report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = build_report(args.input, args.output)
    print(f"wrote {len(report)} method rows to {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    sys.exit(main())
