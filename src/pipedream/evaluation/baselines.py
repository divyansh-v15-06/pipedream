"""Common evaluation records and deterministic baseline implementations."""

from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pipedream.benchmarks import BenchmarkRecord, compile_source
from pipedream.compiler import CompilerConfig, PassCatalog, PassEngine

PIPELINES = {"-O2": "default<O2>", "-O3": "default<O3>", "-Oz": "default<Oz>"}


@dataclass(frozen=True)
class BaselineResult:
    method: str
    benchmark_id: str
    seed: int
    initial_instruction_count: int
    final_instruction_count: int
    instruction_reduction: float
    sequence: tuple[str, ...]
    llvm_time_ms: float
    representation_time_ms: float
    total_optimization_time_ms: float
    catalog_version: str
    toolchain_version: str
    config_checksum: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_record(
    record: BenchmarkRecord,
    manifest_root: Path,
    catalog: PassCatalog,
    engine: PassEngine,
    methods: tuple[str, ...] = ("-O2", "-O3", "-Oz", "random", "greedy"),
    seed: int = 0,
    max_steps: int = 12,
    clang: str = "clang",
    beam_width: int = 2,
) -> list[BaselineResult]:
    source_path = manifest_root / record.source
    initial_ir = compile_source(source_path, clang=clang, target_triple=record.target_triple)
    actual_hash = hashlib.sha256(initial_ir).hexdigest()
    if actual_hash != record.initial_ir_sha256:
        raise RuntimeError(
            f"initial IR checksum mismatch for {record.benchmark_id}: "
            f"expected {record.initial_ir_sha256}, got {actual_hash}"
        )
    initial_count = engine.instruction_count(initial_ir)
    results: list[BaselineResult] = []
    for method in methods:
        if method in PIPELINES:
            final_ir, duration_ms = _run_pipeline(
                initial_ir, PIPELINES[method], engine.config
            )
            final_count = engine.instruction_count(final_ir)
            sequence: tuple[str, ...] = ()
            llvm_time_ms = duration_ms
        elif method == "random":
            final_count, sequence, llvm_time_ms = _run_random(
                initial_ir, catalog, engine, seed=seed, max_steps=max_steps
            )
        elif method == "greedy":
            final_count, sequence, llvm_time_ms = _run_greedy(
                initial_ir, catalog, engine, max_steps=max_steps
            )
        elif method == "beam":
            final_count, sequence, llvm_time_ms = _run_beam(
                initial_ir, catalog, engine, max_steps=max_steps, beam_width=beam_width
            )
        else:
            raise ValueError(f"unsupported baseline method: {method}")
        results.append(
            BaselineResult(
                method=method,
                benchmark_id=record.benchmark_id,
                seed=seed,
                initial_instruction_count=initial_count,
                final_instruction_count=final_count,
                instruction_reduction=(initial_count - final_count) / max(initial_count, 1),
                sequence=sequence,
                llvm_time_ms=llvm_time_ms,
                representation_time_ms=0.0,
                total_optimization_time_ms=llvm_time_ms,
                catalog_version=catalog.version,
                toolchain_version=engine.config.toolchain_version,
                config_checksum=_config_checksum(method, seed, max_steps),
            )
        )
    return results


def _run_pipeline(ir: bytes, pipeline: str, config: CompilerConfig) -> tuple[bytes, float]:
    with TemporaryDirectory(prefix="pipedream-baseline-") as temporary_dir:
        work_dir = Path(temporary_dir)
        input_path = work_dir / "input.bc"
        output_path = work_dir / "output.bc"
        input_path.write_bytes(ir)
        started = time.perf_counter()
        completed = subprocess.run(
            [
                config.opt,
                f"-passes={pipeline},verify",
                str(input_path),
                "-o",
                str(output_path),
            ],
            capture_output=True,
            check=False,
            timeout=config.timeout_seconds,
            text=True,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        if completed.returncode != 0 or not output_path.is_file():
            raise RuntimeError(completed.stderr.strip() or f"opt exited with {completed.returncode}")
        return output_path.read_bytes(), duration_ms


def _run_random(
    initial_ir: bytes,
    catalog: PassCatalog,
    engine: PassEngine,
    seed: int,
    max_steps: int,
) -> tuple[int, tuple[str, ...], float]:
    rng = np.random.default_rng(seed)
    current_ir = initial_ir
    current_count = engine.instruction_count(current_ir)
    sequence: list[str] = []
    duration_ms = 0.0
    for action_id in rng.integers(0, len(catalog.passes), size=max_steps):
        spec = catalog.by_action_id(int(action_id))
        result = engine.apply(current_ir, spec.action_id)
        sequence.append(spec.name)
        duration_ms += result.duration_ms
        if result.committed:
            current_ir = result.ir
            current_count = result.instruction_count
    return current_count, tuple(sequence), duration_ms


def _run_greedy(
    initial_ir: bytes,
    catalog: PassCatalog,
    engine: PassEngine,
    max_steps: int,
) -> tuple[int, tuple[str, ...], float]:
    current_ir = initial_ir
    current_count = engine.instruction_count(current_ir)
    sequence: list[str] = []
    duration_ms = 0.0
    for _ in range(max_steps):
        candidates = []
        for spec in catalog.passes:
            result = engine.apply(current_ir, spec.action_id)
            duration_ms += result.duration_ms
            candidate_count = result.instruction_count if result.committed else current_count
            candidates.append((candidate_count, spec.action_id, result))
        candidate_count, action_id, result = min(candidates, key=lambda item: (item[0], item[1]))
        sequence.append(catalog.by_action_id(action_id).name)
        if not result.committed:
            break
        current_ir = result.ir
        current_count = candidate_count
    return current_count, tuple(sequence), duration_ms


def _run_beam(
    initial_ir: bytes,
    catalog: PassCatalog,
    engine: PassEngine,
    max_steps: int,
    beam_width: int,
) -> tuple[int, tuple[str, ...], float]:
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    initial_count = engine.instruction_count(initial_ir)
    beam: list[tuple[bytes, int, tuple[str, ...]]] = [(initial_ir, initial_count, ())]
    duration_ms = 0.0
    for _ in range(max_steps):
        candidates: list[tuple[bytes, int, tuple[str, ...]]] = []
        for current_ir, current_count, sequence in beam:
            for spec in catalog.passes:
                result = engine.apply(current_ir, spec.action_id)
                duration_ms += result.duration_ms
                if result.committed:
                    candidates.append(
                        (result.ir, result.instruction_count, sequence + (spec.name,))
                    )
                else:
                    candidates.append((current_ir, current_count, sequence + (spec.name,)))
        beam = sorted(candidates, key=lambda item: (item[1], item[2]))[:beam_width]
    _, final_count, sequence = min(beam, key=lambda item: (item[1], item[2]))
    return final_count, sequence, duration_ms


def _config_checksum(method: str, seed: int, max_steps: int) -> str:
    value = f"{method}|{seed}|{max_steps}".encode()
    return hashlib.sha256(value).hexdigest()
