"""Gymnasium environment for program-adaptive LLVM pass ordering."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from pipedream.benchmarks import BenchmarkManifest, BenchmarkRecord, compile_source
from pipedream.compiler import PassCatalog, PassEngine
from pipedream.representations import AutophaseExtractor, NormalizationStats


class PipedreamEnv(gym.Env[NDArray[np.float32], int]):
    """Apply catalog actions to one randomly selected benchmark program."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        manifest: BenchmarkManifest,
        manifest_root: Path,
        engine: PassEngine,
        catalog: PassCatalog,
        max_steps: int = 12,
        clang: str = "clang",
        failure_penalty: float = -0.01,
        extractor: AutophaseExtractor | None = None,
        normalizer: NormalizationStats | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not manifest.records:
            raise ValueError("manifest must contain at least one record")
        self.manifest = manifest
        self.manifest_root = manifest_root
        self.engine = engine
        self.catalog = catalog
        self.max_steps = max_steps
        self.clang = clang
        self.failure_penalty = failure_penalty
        self.extractor = extractor
        self.normalizer = normalizer
        if normalizer is not None and extractor is None:
            raise ValueError("normalizer requires an extractor")
        if normalizer is not None and normalizer.feature_checksum != extractor.schema.checksum:
            raise ValueError("normalization statistics do not match the feature schema")
        self.action_space = gym.spaces.Discrete(len(catalog.passes))
        feature_low = (
            np.zeros(56, dtype=np.float32) if extractor is not None else np.empty(0, dtype=np.float32)
        )
        feature_high = (
            np.full(
                56,
                np.finfo(np.float32).max,
                dtype=np.float32,
            )
            if extractor is not None
            else np.empty(0, dtype=np.float32)
        )
        if normalizer is not None:
            feature_low = np.full(56, -np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((feature_low, np.array([0.0, 0.0, -1.0], dtype=np.float32))),
            high=np.concatenate(
                (feature_high, np.array([np.finfo(np.float32).max, 1.0, 1.0], dtype=np.float32))
            ),
            dtype=np.float32,
        )
        self._record: BenchmarkRecord | None = None
        self._current_ir = b""
        self._initial_count = 0
        self._current_count = 0
        self._step_count = 0
        self._last_reward = 0.0
        self._features = np.empty(0, dtype=np.float32)

    @property
    def current_record(self) -> BenchmarkRecord:
        if self._record is None:
            raise RuntimeError("environment has not been reset")
        return self._record

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        super().reset(seed=seed)
        requested_id = options.get("benchmark_id") if options else None
        if requested_id is None:
            index = int(self.np_random.integers(0, len(self.manifest.records)))
        else:
            matching = [
                index
                for index, record in enumerate(self.manifest.records)
                if record.benchmark_id == requested_id
            ]
            if not matching:
                raise ValueError(f"unknown benchmark ID: {requested_id}")
            index = matching[0]
        self._record = self.manifest.records[index]
        source_path = self.manifest_root / self._record.source
        self._current_ir = compile_source(
            source_path,
            clang=self.clang,
            target_triple=self._record.target_triple,
        )
        actual_hash = _sha256_bytes(self._current_ir)
        if actual_hash != self._record.initial_ir_sha256:
            raise RuntimeError(
                f"initial IR checksum mismatch for {self._record.benchmark_id}: "
                f"expected {self._record.initial_ir_sha256}, got {actual_hash}"
            )
        self._initial_count = self.engine.instruction_count(self._current_ir)
        self._current_count = self._initial_count
        self._step_count = 0
        self._last_reward = 0.0
        self._refresh_features()
        return self._observation(), self._reset_info()

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        if self._record is None:
            raise RuntimeError("reset must be called before step")
        action_id = int(action)
        if not self.action_space.contains(action_id):
            raise ValueError(f"action outside catalog: {action_id}")
        before_count = self._current_count
        result = self.engine.apply(self._current_ir, action_id)
        self._step_count += 1
        if result.committed:
            self._current_ir = result.ir
            self._current_count = result.instruction_count
            self._refresh_features()
            reward = (before_count - self._current_count) / max(self._initial_count, 1)
        else:
            reward = self.failure_penalty
        self._last_reward = float(reward)
        truncated = self._step_count >= self.max_steps
        info = {
            "benchmark_id": self._record.benchmark_id,
            "action_id": action_id,
            "pass_name": result.pass_name,
            "status": result.status,
            "before_instruction_count": before_count,
            "after_instruction_count": self._current_count,
            "initial_instruction_count": self._initial_count,
            "step": self._step_count,
            "cache_key": result.cache_key,
            "stderr": result.stderr,
            "failed": not result.committed,
        }
        return self._observation(), float(reward), False, truncated, info

    def _observation(self) -> NDArray[np.float32]:
        scalar_features = np.asarray(
            [
                self._current_count / max(self._initial_count, 1),
                (self.max_steps - self._step_count) / self.max_steps,
                self._last_reward,
            ],
            dtype=np.float32,
        )
        return np.concatenate((self._features, scalar_features)).astype(np.float32)

    def _refresh_features(self) -> None:
        if self.extractor is not None:
            self._features = self.extractor.extract_bitcode(self._current_ir)
            if self.normalizer is not None:
                self._features = self.normalizer.transform(self._features)

    def _reset_info(self) -> dict[str, object]:
        return {
            "benchmark_id": self.current_record.benchmark_id,
            "source": self.current_record.source,
            "initial_instruction_count": self._initial_count,
            "max_steps": self.max_steps,
            "catalog_version": self.catalog.version,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
