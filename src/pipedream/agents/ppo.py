"""Stable-Baselines3 PPO training with reproducible metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 128
    rollout_steps: int = 16
    batch_size: int = 16
    learning_rate: float = 3e-4
    gamma: float = 0.99
    seed: int = 0
    device: str = "auto"

    def checksum(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def train_ppo(env, output_dir: Path, config: PPOConfig) -> Path:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "PPO dependencies are optional; install with `uv sync --extra rl`"
        ) from exc

    if config.total_timesteps <= 0 or config.rollout_steps <= 0:
        raise ValueError("PPO timesteps and rollout_steps must be positive")
    if config.batch_size <= 0 or config.batch_size > config.rollout_steps:
        raise ValueError("batch_size must be between 1 and rollout_steps")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.rollout_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        seed=config.seed,
        device=config.device,
        tensorboard_log=str(output_dir / "tensorboard"),
        verbose=0,
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    model_path = output_dir / "ppo_model"
    model.save(model_path)
    (output_dir / "training_config.json").write_text(
        json.dumps(
            {"config": asdict(config), "config_checksum": config.checksum()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return model_path.with_suffix(".zip")
