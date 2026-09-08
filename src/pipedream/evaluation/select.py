"""Select a PPO checkpoint using validation programs only."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pipedream.benchmarks import BenchmarkManifest
from pipedream.env import PipedreamEnv
from pipedream.evaluation.ppo import evaluate_model


def select_checkpoint(
    models: tuple[Path, ...],
    manifest: BenchmarkManifest,
    env: PipedreamEnv,
    output: Path,
    seed: int = 0,
) -> Path:
    candidates = []
    with TemporaryDirectory(prefix="pipedream-validation-") as temporary_dir:
        for model in models:
            result_path = Path(temporary_dir) / f"{len(candidates)}.jsonl"
            evaluate_model(model, env, manifest, result_path, seed=seed)
            rows = [json.loads(line) for line in result_path.read_text().splitlines() if line]
            score = sum(float(row["instruction_reduction"]) for row in rows) / len(rows)
            candidates.append({"model": str(model), "mean_instruction_reduction": score})
    selected = max(candidates, key=lambda item: (item["mean_instruction_reduction"], item["model"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "metric": "mean_instruction_reduction",
                "selected_model": selected["model"],
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return Path(selected["model"])
