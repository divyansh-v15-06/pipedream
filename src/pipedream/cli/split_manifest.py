"""Create a family-aware train/validation/test manifest from a source manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipedream.benchmarks.manifest import load_manifest, split_manifest


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    source = load_manifest(args.input)
    result = split_manifest(
        source,
        args.output,
        tier=args.tier,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        source_root=args.input.parent,
    )
    counts = {split: sum(record.split == split for record in result.records) for split in ("train", "validation", "test")}
    print(f"generated {len(result.records)} records at {args.output}: {counts}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", default="pilot")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser


if __name__ == "__main__":
    sys.exit(main())
