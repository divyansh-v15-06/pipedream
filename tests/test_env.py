from pathlib import Path

import numpy as np

from pipedream.benchmarks import load_manifest
from pipedream.compiler import PassCatalog, PassEngine
from pipedream.env import PipedreamEnv


def test_reset_is_deterministic_for_a_seed() -> None:
    root = Path(__file__).parents[1]
    manifest_path = root / "benchmarks" / "manifest.json"
    manifest = load_manifest(manifest_path)
    catalog = PassCatalog.from_yaml(root / "configs" / "pass_catalog.yaml")
    engine = PassEngine(catalog, cache_dir=root / ".cache" / "test-env")
    env = PipedreamEnv(manifest, manifest_path.parent, engine, catalog, max_steps=2)

    first, first_info = env.reset(seed=42)
    second, second_info = env.reset(seed=42)

    np.testing.assert_array_equal(first, second)
    assert first_info["benchmark_id"] == second_info["benchmark_id"]


def test_step_uses_fixed_budget_and_returns_valid_observations() -> None:
    root = Path(__file__).parents[1]
    manifest_path = root / "benchmarks" / "manifest.json"
    manifest = load_manifest(manifest_path)
    catalog = PassCatalog.from_yaml(root / "configs" / "pass_catalog.yaml")
    engine = PassEngine(catalog, cache_dir=root / ".cache" / "test-env")
    env = PipedreamEnv(manifest, manifest_path.parent, engine, catalog, max_steps=2)
    observation, _ = env.reset(seed=0)
    assert env.observation_space.contains(observation)

    observation, _, terminated, truncated, info = env.step(2)
    assert env.observation_space.contains(observation)
    assert terminated is False
    assert truncated is False
    assert info["step"] == 1

    _, _, _, truncated, info = env.step(6)
    assert truncated is True
    assert info["step"] == 2
