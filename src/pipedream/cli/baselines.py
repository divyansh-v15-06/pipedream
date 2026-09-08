"""Run LLVM pipeline, random, and greedy pass-ordering baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipedream.benchmarks import load_manifest
from pipedream.compiler import PassCatalog, PassEngine
from pipedream.evaluation import evaluate_record

DEFAULT_METHODS = ("-O2", "-O3", "-Oz", "random", "greedy")
ALL_METHODS = (*DEFAULT_METHODS, "beam")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    catalog = PassCatalog.from_yaml(args.catalog)
    records = [record for record in manifest.records if args.split in {record.split, "all"}]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error(f"manifest has no records for split {args.split!r}")

    engine = PassEngine(catalog, cache_dir=args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            for seed in args.seeds:
                results = evaluate_record(
                    record,
                    args.manifest.parent,
                    catalog,
                    engine,
                    methods=tuple(args.methods),
                    seed=seed,
                    max_steps=args.max_steps,
                    clang=args.clang,
                    beam_width=args.beam_width,
                )
                for result in results:
                    handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    print(f"wrote baseline results for {len(records)} programs to {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/pipedream/passes"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/baselines.jsonl"))
    return parser


if __name__ == "__main__":
    sys.exit(main())
