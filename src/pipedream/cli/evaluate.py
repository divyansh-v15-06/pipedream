"""Evaluate a saved PPO policy on a selected manifest split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipedream.benchmarks import BenchmarkManifest, load_manifest
from pipedream.compiler import PassCatalog, PassEngine
from pipedream.env import PipedreamEnv
from pipedream.evaluation.ppo import evaluate_model
from pipedream.representations import AutophaseExtractor, FeatureSchema, NormalizationStats


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    full_manifest = load_manifest(args.manifest)
    records = tuple(record for record in full_manifest.records if args.split in {record.split, "all"})
    if not records:
        parser.error(f"manifest has no records for split {args.split!r}")
    manifest = BenchmarkManifest(full_manifest.schema_version, full_manifest.tier, records)
    catalog = PassCatalog.from_yaml(args.catalog)
    schema = FeatureSchema.from_yaml(args.schema)
    normalizer = NormalizationStats.load(args.normalization_stats) if args.normalization_stats else None
    env = PipedreamEnv(
        manifest,
        args.manifest.parent,
        PassEngine(catalog, cache_dir=args.cache_dir),
        catalog,
        max_steps=args.max_steps,
        clang=args.clang,
        extractor=AutophaseExtractor(schema),
        normalizer=normalizer,
    )
    evaluate_model(args.model, env, manifest, args.output, seed=args.seed)
    print(f"wrote evaluation results to {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("results/raw/ppo_evaluation.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/pipedream/passes"))
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    sys.exit(main())
