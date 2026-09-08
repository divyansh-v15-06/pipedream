"""Deterministic evaluation of a saved PPO policy."""

from __future__ import annotations

import json
from pathlib import Path

from pipedream.benchmarks import BenchmarkManifest
from pipedream.env import PipedreamEnv


def evaluate_model(
    model_path: Path,
    env: PipedreamEnv,
    manifest: BenchmarkManifest,
    output: Path,
    seed: int = 0,
) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("install PPO dependencies with `uv sync --extra rl`") from exc

    model = PPO.load(model_path, env=env)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in manifest.records:
            observation, info = env.reset(seed=seed, options={"benchmark_id": record.benchmark_id})
            initial_count = int(info["initial_instruction_count"])
            sequence: list[str] = []
            failed_steps = 0
            total_reward = 0.0
            step_info: dict[str, object] = {}
            for _ in range(env.max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, step_info = env.step(int(action))
                sequence.append(str(step_info["pass_name"]))
                total_reward += reward
                failed_steps += int(step_info["failed"])
                if terminated or truncated:
                    break
            final_count = int(step_info["after_instruction_count"])
            handle.write(
                json.dumps(
                    {
                        "benchmark_id": record.benchmark_id,
                        "seed": seed,
                        "initial_instruction_count": initial_count,
                        "final_instruction_count": final_count,
                        "instruction_reduction": (initial_count - final_count)
                        / max(initial_count, 1),
                        "episode_reward": total_reward,
                        "sequence": sequence,
                        "failed_steps": failed_steps,
                        "model": str(model_path),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
