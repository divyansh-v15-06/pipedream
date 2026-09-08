"""Configurable adapter for an externally supplied LLVM IR2Vec executable."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray


class IR2VecUnavailable(RuntimeError):
    """Raised when the configured IR2Vec executable is not installed."""


@dataclass(frozen=True)
class IR2VecConfig:
    version: str
    mode: str
    dimension: int
    executable: str = "llvm-ir2vec"
    args: tuple[str, ...] = ("{input}",)

    @property
    def checksum(self) -> str:
        value = f"{self.version}|{self.mode}|{self.dimension}|{self.executable}|{self.args}"
        return hashlib.sha256(value.encode()).hexdigest()


class IR2VecExtractor:
    def __init__(self, config: IR2VecConfig) -> None:
        if config.dimension <= 0:
            raise ValueError("IR2Vec dimension must be positive")
        self.config = config

    def available(self) -> bool:
        return shutil.which(self.config.executable) is not None

    def extract_text(self, ir_text: str) -> NDArray[np.float32]:
        with TemporaryDirectory(prefix="pipedream-ir2vec-") as temporary_dir:
            input_path = Path(temporary_dir) / "module.ll"
            input_path.write_text(ir_text, encoding="utf-8")
            return self._run(input_path)

    def extract_bitcode(self, ir: bytes) -> NDArray[np.float32]:
        with TemporaryDirectory(prefix="pipedream-ir2vec-") as temporary_dir:
            input_path = Path(temporary_dir) / "module.bc"
            input_path.write_bytes(ir)
            return self._run(input_path)

    def _run(self, input_path: Path) -> NDArray[np.float32]:
        if not self.available():
            raise IR2VecUnavailable(
                f"{self.config.executable!r} is unavailable; install LLVM IR2Vec "
                "and record its exact version before enabling this representation"
            )
        args = [argument.replace("{input}", str(input_path)) for argument in self.config.args]
        completed = subprocess.run(
            [self.config.executable, *args],
            capture_output=True,
            check=False,
            timeout=30,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "IR2Vec command failed")
        try:
            values = np.asarray([float(token) for token in completed.stdout.split()], dtype=np.float32)
        except ValueError as exc:
            raise RuntimeError("IR2Vec output must be whitespace-separated floats") from exc
        if values.shape != (self.config.dimension,):
            raise ValueError(
                f"IR2Vec output dimension {values.size} does not match {self.config.dimension}"
            )
        return values


def mean_pool(function_vectors: NDArray[np.float32], dimension: int) -> NDArray[np.float32]:
    """Mean-pool function-level vectors, using zeros for an empty module."""
    if function_vectors.size == 0:
        return np.zeros(dimension, dtype=np.float32)
    if function_vectors.ndim != 2 or function_vectors.shape[1] != dimension:
        raise ValueError("function vectors do not match the configured IR2Vec dimension")
    return function_vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
