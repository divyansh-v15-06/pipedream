"""Fit Autophase normalization statistics on one manifest split only."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from pipedream.benchmarks import compile_source, load_manifest
from pipedream.representations import AutophaseExtractor, FeatureSchema, NormalizationStats


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    records = [record for record in manifest.records if record.split == args.split]
    if not records:
        parser.error(f"manifest has no records for split {args.split!r}")
    schema = FeatureSchema.from_yaml(args.schema)
    extractor = AutophaseExtractor(schema)
    vectors = []
    for record in records:
        source = args.manifest.parent / record.source
        ir = compile_source(source, clang=args.clang, target_triple=record.target_triple)
        if hashlib.sha256(ir).hexdigest() != record.initial_ir_sha256:
            raise RuntimeError(f"initial IR checksum mismatch for {record.benchmark_id}")
        vectors.append(extractor.extract_bitcode(ir))
    stats = NormalizationStats.fit(np.stack(vectors), schema.checksum)
    stats.save(args.output)
    print(f"fit normalization on {len(vectors)} {args.split} records: {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    sys.exit(main())
