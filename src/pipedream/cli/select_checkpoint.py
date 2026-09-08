"""Select the best PPO checkpoint using validation split performance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipedream.benchmarks import BenchmarkManifest, load_manifest
from pipedream.compiler import PassCatalog, PassEngine
from pipedream.env import PipedreamEnv
from pipedream.evaluation.select import select_checkpoint
from pipedream.representations import AutophaseExtractor, FeatureSchema, NormalizationStats


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    full_manifest = load_manifest(args.manifest)
    records = tuple(record for record in full_manifest.records if record.split == "validation")
    if not records:
        parser.error("manifest has no validation records")
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
    selected = select_checkpoint(tuple(args.models), manifest, env, args.output, seed=args.seed)
    print(f"selected validation checkpoint: {selected}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--normalization-stats", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/pipedream/passes"))
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    sys.exit(main())
