"""Transactional LLVM pass execution with replay metadata and caching."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .catalog import PassCatalog, PassSpec


@dataclass(frozen=True)
class CompilerConfig:
    opt: str = "opt"
    llvm_dis: str = "llvm-dis"
    timeout_seconds: float = 30.0
    toolchain_version: str = "llvm20"


@dataclass(frozen=True)
class PassResult:
    status: str
    ir: bytes
    instruction_count: int
    duration_ms: float
    action_id: int
    pass_name: str
    cache_key: str
    stderr: str = ""

    @property
    def committed(self) -> bool:
        return self.status in {"ok", "cached"}


class PassEngine:
    """Apply catalog passes without mutating the caller's live state on failure."""

    def __init__(
        self,
        catalog: PassCatalog,
        config: CompilerConfig | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config or CompilerConfig()
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def apply(self, ir: bytes, action_id: int) -> PassResult:
        spec = self.catalog.by_action_id(action_id)
        cache_key = self._cache_key(ir, spec)
        started = time.perf_counter()
        if self.cache_dir is not None:
            cached = self._read_cache(cache_key, spec)
            if cached is not None:
                return PassResult(
                    status="cached",
                    ir=cached,
                    instruction_count=count_instructions(self._disassemble(cached)),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    action_id=spec.action_id,
                    pass_name=spec.name,
                    cache_key=cache_key,
                )

        with TemporaryDirectory(prefix="pipedream-pass-") as temporary_dir:
            work_dir = Path(temporary_dir)
            input_path = work_dir / "input.bc"
            output_path = work_dir / "output.bc"
            input_path.write_bytes(ir)
            command = [
                self.config.opt,
                f"-passes={spec.pipeline},verify",
                str(input_path),
                "-o",
                str(output_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    timeout=self.config.timeout_seconds,
                    text=True,
                )
            except subprocess.TimeoutExpired as exc:
                return self._failed_result(spec, cache_key, started, ir, f"timeout: {exc}")
            except OSError as exc:
                return self._failed_result(spec, cache_key, started, ir, str(exc))

            if completed.returncode != 0 or not output_path.is_file():
                stderr = completed.stderr.strip() or f"opt exited with {completed.returncode}"
                return self._failed_result(spec, cache_key, started, ir, stderr)

            output = output_path.read_bytes()
            try:
                disassembled = self._disassemble(output)
                instruction_count = count_instructions(disassembled)
            except (OSError, subprocess.SubprocessError) as exc:
                return self._failed_result(spec, cache_key, started, ir, str(exc))

        if self.cache_dir is not None:
            self._write_cache(cache_key, output, spec)
        return PassResult(
            status="ok",
            ir=output,
            instruction_count=instruction_count,
            duration_ms=(time.perf_counter() - started) * 1000,
            action_id=spec.action_id,
            pass_name=spec.name,
            cache_key=cache_key,
        )

    def instruction_count(self, ir: bytes) -> int:
        """Measure a verified bitcode module without applying a pass."""
        return count_instructions(self._disassemble(ir))

    def _cache_key(self, ir: bytes, spec: PassSpec) -> str:
        digest = hashlib.sha256()
        digest.update(ir)
        digest.update(self.catalog.version.encode())
        digest.update(self.config.toolchain_version.encode())
        digest.update(spec.pipeline.encode())
        return digest.hexdigest()

    def _cache_paths(self, cache_key: str) -> tuple[Path, Path]:
        if self.cache_dir is None:
            raise RuntimeError("cache is disabled")
        return self.cache_dir / f"{cache_key}.bc", self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str, spec: PassSpec) -> bytes | None:
        bitcode_path, metadata_path = self._cache_paths(cache_key)
        if not bitcode_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if metadata.get("pass_name") != spec.name:
            return None
        return bitcode_path.read_bytes()

    def _write_cache(self, cache_key: str, ir: bytes, spec: PassSpec) -> None:
        bitcode_path, metadata_path = self._cache_paths(cache_key)
        bitcode_path.write_bytes(ir)
        metadata_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "catalog_version": self.catalog.version,
                    "toolchain_version": self.config.toolchain_version,
                    "pass_name": spec.name,
                    "pipeline": spec.pipeline,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _disassemble(self, ir: bytes) -> str:
        with TemporaryDirectory(prefix="pipedream-disassemble-") as temporary_dir:
            work_dir = Path(temporary_dir)
            bitcode_path = work_dir / "module.bc"
            text_path = work_dir / "module.ll"
            bitcode_path.write_bytes(ir)
            completed = subprocess.run(
                [self.config.llvm_dis, str(bitcode_path), "-o", str(text_path)],
                capture_output=True,
                check=False,
                timeout=self.config.timeout_seconds,
                text=True,
            )
            if completed.returncode != 0 or not text_path.is_file():
                raise subprocess.SubprocessError(completed.stderr.strip())
            return text_path.read_text(encoding="utf-8")

    def _failed_result(
        self,
        spec: PassSpec,
        cache_key: str,
        started: float,
        original_ir: bytes,
        stderr: str,
    ) -> PassResult:
        return PassResult(
            status="failed",
            ir=original_ir,
            instruction_count=0,
            duration_ms=(time.perf_counter() - started) * 1000,
            action_id=spec.action_id,
            pass_name=spec.name,
            cache_key=cache_key,
            stderr=stderr,
        )


def count_instructions(ir_text: str) -> int:
    """Count non-debug instructions in llvm-dis text, excluding labels and metadata."""
    in_function = False
    count = 0
    for raw_line in ir_text.splitlines():
        line = raw_line.strip()
        if line.startswith("define ") and line.endswith("{"):
            in_function = True
            continue
        if not in_function or not line or line.startswith((";", "!")):
            continue
        if line == "}":
            in_function = False
            continue
        if line.endswith(":") or line.startswith(("attributes ", "...")):
            continue
        if "@llvm.dbg." in line:
            continue
        count += 1
    return count


def trace_record(
    step: int,
    before_count: int,
    action_id: int,
    result: PassResult,
) -> dict[str, object]:
    """Create a JSON-serializable step record for a JSONL trace."""
    record = asdict(result)
    record.pop("ir", None)
    after_count = result.instruction_count if result.committed else before_count
    record.update(
        {
            "step": step,
            "before_instruction_count": before_count,
            "instruction_count": after_count,
            "after_instruction_count": after_count,
            "reward": (before_count - result.instruction_count)
            / max(before_count, 1)
            if result.committed
            else 0.0,
            "action_id": action_id,
            "rolled_back": not result.committed,
        }
    )
    return record


def append_trace(path: Path, record: dict[str, object]) -> None:
    """Append one stable JSON object to a JSONL trace file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
