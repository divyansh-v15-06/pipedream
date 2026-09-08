"""Create and validate an immutable benchmark manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipedream.benchmarks.manifest import build_manifest, load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.validate:
        manifest = load_manifest(args.manifest, verify_hashes=True)
        print(f"validated {len(manifest.records)} records from {args.manifest}")
        return 0
    if args.source_dir is None:
        parser.error("--source-dir is required when generating a manifest")
    manifest = build_manifest(
        args.source_dir,
        args.manifest,
        tier=args.tier,
        split=args.split,
        clang=args.clang,
        target_triple=args.target_triple,
        llvm_version=args.llvm_version,
    )
    print(f"generated {len(manifest.records)} records at {args.manifest}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--tier", default="smoke")
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--target-triple", default="x86_64-unknown-linux-gnu")
    parser.add_argument("--llvm-version", default="llvm20")
    parser.add_argument("--validate", action="store_true")
    return parser


if __name__ == "__main__":
    sys.exit(main())
