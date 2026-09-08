"""Immutable benchmark manifest creation, loading, and validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ALLOWED_SPLITS = frozenset({"train", "validation", "test", "smoke"})
DEFAULT_FLAGS = (
    "-O0",
    "-Xclang",
    "-disable-O0-optnone",
    "-fno-discard-value-names",
    "-emit-llvm",
    "-c",
)


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_id: str
    source: str
    source_sha256: str
    source_family: str
    split: str
    language: str
    standard: str
    compiler_flags: tuple[str, ...]
    target_triple: str
    initial_ir_sha256: str
    llvm_version: str


@dataclass(frozen=True)
class BenchmarkManifest:
    schema_version: str
    tier: str
    records: tuple[BenchmarkRecord, ...]

    def validate(self, root: Path, verify_hashes: bool = True) -> None:
        if self.schema_version != "pipedream-manifest-v1":
            raise ValueError(f"unsupported manifest schema: {self.schema_version}")
        if not self.records:
            raise ValueError("manifest must contain at least one record")
        ids = [record.benchmark_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark IDs must be unique")
        for record in self.records:
            if record.split not in ALLOWED_SPLITS:
                raise ValueError(f"unsupported split: {record.split}")
            if Path(record.source).is_absolute() or ".." in Path(record.source).parts:
                raise ValueError(f"source path must stay below manifest root: {record.source}")
            source_path = root / record.source
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if not record.source_family.strip():
                raise ValueError(f"empty source family for {record.benchmark_id}")
            if verify_hashes and sha256_file(source_path) != record.source_sha256:
                raise ValueError(f"source checksum mismatch for {record.benchmark_id}")


def load_manifest(path: Path, verify_hashes: bool = True) -> BenchmarkManifest:
    with path.open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("manifest must contain a JSON object")
    records = tuple(_parse_record(item) for item in _required_list(raw, "records"))
    manifest = BenchmarkManifest(
        schema_version=_required_string(raw, "schema_version"),
        tier=_required_string(raw, "tier"),
        records=records,
    )
    manifest.validate(path.parent, verify_hashes=verify_hashes)
    return manifest


def build_manifest(
    source_dir: Path,
    output: Path,
    tier: str = "smoke",
    split: str = "smoke",
    clang: str = "clang",
    target_triple: str = "x86_64-unknown-linux-gnu",
    llvm_version: str = "llvm20",
) -> BenchmarkManifest:
    sources = sorted(source_dir.glob("*.c"))
    if not sources:
        raise ValueError(f"no C sources found in {source_dir}")
    records: list[BenchmarkRecord] = []
    for index, source_path in enumerate(sources, start=1):
        source_hash = sha256_file(source_path)
        initial_ir_hash = hashlib.sha256(
            compile_source(source_path, clang=clang, target_triple=target_triple)
        ).hexdigest()
        records.append(
            BenchmarkRecord(
                benchmark_id=f"{tier}_{index:03d}_{source_path.stem}",
                source=str(source_path.relative_to(output.parent)).replace("\\", "/"),
                source_sha256=source_hash,
                source_family=f"{source_path.stem}_family",
                split=split,
                language="c",
                standard="c11",
                compiler_flags=DEFAULT_FLAGS,
                target_triple=target_triple,
                initial_ir_sha256=initial_ir_hash,
                llvm_version=llvm_version,
            )
        )
    manifest = BenchmarkManifest("pipedream-manifest-v1", tier, tuple(records))
    manifest.validate(output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_to_json(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def split_manifest(
    manifest: BenchmarkManifest,
    output: Path,
    tier: str = "pilot",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    source_root: Path | None = None,
) -> BenchmarkManifest:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    groups: dict[str, list[BenchmarkRecord]] = {}
    for record in manifest.records:
        groups.setdefault(record.source_family, []).append(record)
    ordered = [record for family in sorted(groups) for record in sorted(groups[family], key=lambda x: x.source)]
    train_count = max(1, round(len(ordered) * train_fraction))
    validation_count = max(1, round(len(ordered) * validation_fraction))
    if train_count + validation_count >= len(ordered):
        validation_count = max(1, len(ordered) - train_count - 1)
    split_records: list[BenchmarkRecord] = []
    for index, record in enumerate(ordered):
        split = "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
        split_records.append(
            BenchmarkRecord(
                benchmark_id=f"{tier}_{index + 1:03d}_{Path(record.source).stem}",
                source=record.source,
                source_sha256=record.source_sha256,
                source_family=record.source_family,
                split=split,
                language=record.language,
                standard=record.standard,
                compiler_flags=record.compiler_flags,
                target_triple=record.target_triple,
                initial_ir_sha256=record.initial_ir_sha256,
                llvm_version=record.llvm_version,
            )
        )
    result = BenchmarkManifest("pipedream-manifest-v1", tier, tuple(split_records))
    result.validate(source_root or output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_to_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_source(
    source_path: Path,
    clang: str = "clang",
    target_triple: str = "x86_64-unknown-linux-gnu",
) -> bytes:
    with TemporaryDirectory(prefix="pipedream-manifest-") as temporary_dir:
        output = Path(temporary_dir) / "module.bc"
        command = [
            clang,
            *DEFAULT_FLAGS,
            "-target",
            target_triple,
            "-std=c11",
            "-fdebug-compilation-dir=.",
            source_path.name,
            "-o",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=source_path.parent,
            capture_output=True,
            check=False,
            timeout=30,
            text=True,
        )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(completed.stderr.strip() or f"clang exited with {completed.returncode}")
        return output.read_bytes()


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest field {key!r} must be a non-empty string")
    return value


def _required_list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise TypeError(f"manifest field {key!r} must be a list")
    return value


def _parse_record(raw: Any) -> BenchmarkRecord:
    if not isinstance(raw, dict):
        raise TypeError("manifest records must be objects")
    flags = raw.get("compiler_flags")
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise TypeError("compiler_flags must be a list of strings")
    return BenchmarkRecord(
        benchmark_id=_required_string(raw, "benchmark_id"),
        source=_required_string(raw, "source"),
        source_sha256=_required_string(raw, "source_sha256"),
        source_family=_required_string(raw, "source_family"),
        split=_required_string(raw, "split"),
        language=_required_string(raw, "language"),
        standard=_required_string(raw, "standard"),
        compiler_flags=tuple(flags),
        target_triple=_required_string(raw, "target_triple"),
        initial_ir_sha256=_required_string(raw, "initial_ir_sha256"),
        llvm_version=_required_string(raw, "llvm_version"),
    )


def _to_json(manifest: BenchmarkManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "tier": manifest.tier,
        "records": [
            {
                "benchmark_id": record.benchmark_id,
                "source": record.source,
                "source_sha256": record.source_sha256,
                "source_family": record.source_family,
                "split": record.split,
                "language": record.language,
                "standard": record.standard,
                "compiler_flags": list(record.compiler_flags),
                "target_triple": record.target_triple,
                "initial_ir_sha256": record.initial_ir_sha256,
                "llvm_version": record.llvm_version,
            }
            for record in manifest.records
        ],
    }
