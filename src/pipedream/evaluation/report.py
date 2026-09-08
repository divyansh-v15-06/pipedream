"""Aggregate replayable JSONL results into a comparison table."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


def build_report(input_path: Path, output_path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    report = []
    for method, method_rows in sorted(grouped.items()):
        reductions = [float(row["instruction_reduction"]) for row in method_rows]
        final_counts = [float(row["final_instruction_count"]) for row in method_rows]
        costs = [float(row["total_optimization_time_ms"]) for row in method_rows]
        report.append(
            {
                "method": method,
                "programs": len(method_rows),
                "mean_final_instruction_count": mean(final_counts),
                "median_final_instruction_count": median(final_counts),
                "mean_instruction_reduction": mean(reductions),
                "improved_fraction": sum(value > 0 for value in reductions) / len(reductions),
                "mean_optimization_cost_ms": mean(costs),
                "std_optimization_cost_ms": stdev(costs) if len(costs) > 1 else 0.0,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
