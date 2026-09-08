"""Train PPO on the project-owned smoke environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipedream.agents import PPOConfig, train_ppo
from pipedream.benchmarks import BenchmarkManifest, load_manifest
from pipedream.compiler import PassCatalog, PassEngine
from pipedream.env import PipedreamEnv
from pipedream.representations import AutophaseExtractor, FeatureSchema, NormalizationStats


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    full_manifest = load_manifest(args.manifest)
    records = tuple(
        record for record in full_manifest.records if args.split == "all" or record.split == args.split
    )
    if not records:
        parser.error(f"manifest has no records for split {args.split!r}")
    manifest = BenchmarkManifest(full_manifest.schema_version, full_manifest.tier, records)
    catalog = PassCatalog.from_yaml(args.catalog)
    schema = FeatureSchema.from_yaml(args.schema)
    extractor = AutophaseExtractor(schema)
    normalizer = NormalizationStats.load(args.normalization_stats) if args.normalization_stats else None
    env = PipedreamEnv(
        manifest,
        args.manifest.parent,
        PassEngine(catalog, cache_dir=args.cache_dir),
        catalog,
        max_steps=args.max_steps,
        clang=args.clang,
        extractor=extractor,
        normalizer=normalizer,
    )
    seeds = args.seeds if args.seeds is not None else [args.seed]
    for seed in seeds:
        output_dir = args.output_dir if len(seeds) == 1 else args.output_dir / f"seed_{seed}"
        model_path = train_ppo(
            env,
            output_dir,
            PPOConfig(
                total_timesteps=args.total_timesteps,
                rollout_steps=args.rollout_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                seed=seed,
                device=args.device,
            ),
        )
        (output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "tier": manifest.tier,
                    "split": args.split,
                    "records": len(records),
                    "schema": str(args.schema),
                    "normalization_stats": str(args.normalization_stats)
                    if args.normalization_stats
                    else None,
                    "seed": seed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"saved PPO model to {model_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path)
    parser.add_argument("--split", default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw/ppo_smoke"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/pipedream/passes"))
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--total-timesteps", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    sys.exit(main())
