"""Small deterministic statistical utilities for held-out comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int


def paired_bootstrap(
    differences: NDArray[np.float64],
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("bootstrap requires at least one paired difference")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence must be between zero and one")
    rng = np.random.default_rng(seed)
    resamples = rng.choice(values, size=(samples, values.size), replace=True)
    estimates = resamples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(values.mean()),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        confidence=confidence,
        samples=samples,
    )


def paired_effect_size(differences: NDArray[np.float64]) -> float:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("effect size requires at least one paired difference")
    scale = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean() / scale) if scale > 0 else 0.0


def summarize_differences(differences: NDArray[np.float64]) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("summary requires at least one paired difference")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "improved_fraction": float(np.mean(values > 0)),
        "worse_fraction": float(np.mean(values < 0)),
        "effect_size": paired_effect_size(values),
    }
