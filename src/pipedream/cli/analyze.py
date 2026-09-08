"""Analyze paired JSONL baseline results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pipedream.analysis import paired_bootstrap, summarize_differences


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    baseline = _load(args.baseline, args.baseline_method)
    candidate = _load(args.candidate, args.candidate_method)
    baseline_by_key = {_key(row): row for row in baseline}
    differences = []
    for row in candidate:
        key = _key(row)
        if key not in baseline_by_key:
            raise ValueError(f"candidate row has no paired baseline: {key}")
        differences.append(
            float(baseline_by_key[key][args.metric]) - float(row[args.metric])
        )
    values = np.asarray(differences, dtype=np.float64)
    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "metric": args.metric,
        "pairs": len(values),
        "summary": summarize_differences(values),
        "bootstrap": paired_bootstrap(
            values,
            samples=args.samples,
            confidence=args.confidence,
            seed=args.seed,
        ).__dict__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote analysis to {args.output}")
    return 0


def _load(path: Path, method: str | None) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if method is not None:
        rows = [row for row in rows if row.get("method") == method]
    if not rows:
        raise ValueError(f"no rows found in {path} for method {method!r}")
    return rows


def _key(row: dict[str, object]) -> tuple[object, object]:
    return row["benchmark_id"], row.get("seed", 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metric", default="final_instruction_count")
    parser.add_argument("--baseline-method")
    parser.add_argument("--candidate-method")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    sys.exit(main())
