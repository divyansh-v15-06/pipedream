"""Replay a fixed LLVM pass sequence and write reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipedream.compiler import CompilerConfig, PassCatalog, PassEngine, append_trace, trace_record


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    program = args.program.resolve()
    if not program.is_file():
        parser.error(f"program does not exist: {program}")
    if args.output.resolve() == program:
        parser.error("--output must be different from --program")

    catalog = PassCatalog.from_yaml(args.catalog.resolve())
    try:
        specs = [catalog.by_name(name) for name in args.passes]
    except KeyError as exc:
        parser.error(str(exc))

    if args.trace.exists() and not args.append_trace:
        args.trace.unlink()
    cache_dir = None if args.no_cache else args.cache_dir
    engine = PassEngine(
        catalog,
        config=CompilerConfig(timeout_seconds=args.timeout_seconds),
        cache_dir=cache_dir,
    )

    current_ir = program.read_bytes()
    current_count = engine.instruction_count(current_ir)
    initial_count = current_count
    failed_steps = 0
    cache_hits = 0
    total_duration_ms = 0.0

    for step, spec in enumerate(specs):
        result = engine.apply(current_ir, spec.action_id)
        append_trace(args.trace, trace_record(step, current_count, spec.action_id, result))
        current_ir = result.ir
        total_duration_ms += result.duration_ms
        if result.committed:
            current_count = result.instruction_count
            cache_hits += result.status == "cached"
        else:
            failed_steps += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(current_ir)
    summary = {
        "program": str(program),
        "output": str(args.output.resolve()),
        "trace": str(args.trace.resolve()),
        "catalog_version": catalog.version,
        "llvm_major": catalog.llvm_major,
        "sequence": [spec.name for spec in specs],
        "initial_instruction_count": initial_count,
        "final_instruction_count": current_count,
        "total_reward": (initial_count - current_count) / max(initial_count, 1),
        "failed_steps": failed_steps,
        "cache_hits": cache_hits,
        "total_duration_ms": total_duration_ms,
        "success": failed_steps == 0,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 1 if failed_steps else 0


def _build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True, help="input LLVM bitcode (.bc)")
    parser.add_argument(
        "--passes",
        nargs="+",
        required=True,
        metavar="PASS",
        help="catalog pass names to apply in order",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=repo_root / "configs" / "pass_catalog.yaml",
    )
    parser.add_argument("--output", type=Path, default=Path("results/raw/replay.bc"))
    parser.add_argument("--trace", type=Path, default=Path("results/raw/replay.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/raw/replay.summary.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/pipedream/passes"))
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--append-trace", action="store_true")
    return parser


if __name__ == "__main__":
    sys.exit(main())
