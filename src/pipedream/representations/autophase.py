"""Project-owned, versioned 56-feature LLVM IR representation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from pipedream.compiler import count_instructions


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    names: tuple[str, ...]
    kinds: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> FeatureSchema:
        with path.open(encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle)
        if not isinstance(raw, dict) or not isinstance(raw.get("features"), list):
            raise TypeError("feature schema must contain a features list")
        version = raw.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("feature schema needs a version")
        names: list[str] = []
        kinds: list[str] = []
        for item in raw["features"]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise TypeError("each feature must have a name")
            kind = item.get("kind")
            if kind not in {"scalar", "opcode"}:
                raise ValueError(f"unsupported feature kind: {kind}")
            names.append(item["name"])
            kinds.append(kind)
        if len(names) != 56:
            raise ValueError(f"Autophase schema must contain 56 features, got {len(names)}")
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        return cls(version, tuple(names), tuple(kinds))

    @property
    def checksum(self) -> str:
        content = "\n".join(f"{name}:{kind}" for name, kind in zip(self.names, self.kinds))
        return hashlib.sha256(content.encode()).hexdigest()


class AutophaseExtractor:
    """Extract deterministic counts from textual llvm-dis output."""

    def __init__(self, schema: FeatureSchema) -> None:
        self.schema = schema

    def extract(self, ir_text: str) -> NDArray[np.float32]:
        counts = _collect_counts(ir_text)
        values = np.asarray([counts.get(name, 0) for name in self.schema.names], dtype=np.float32)
        return values

    def extract_bitcode(self, ir: bytes, llvm_dis: str = "llvm-dis") -> NDArray[np.float32]:
        with TemporaryDirectory(prefix="pipedream-features-") as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "module.bc"
            output_path = root / "module.ll"
            input_path.write_bytes(ir)
            completed = subprocess.run(
                [llvm_dis, str(input_path), "-o", str(output_path)],
                capture_output=True,
                check=False,
                timeout=30,
                text=True,
            )
            if completed.returncode != 0 or not output_path.is_file():
                raise RuntimeError(completed.stderr.strip() or "llvm-dis failed")
            return self.extract(output_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class NormalizationStats:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    feature_checksum: str

    @classmethod
    def fit(cls, values: NDArray[np.float32], feature_checksum: str) -> NormalizationStats:
        if values.ndim != 2 or values.shape[1] != 56:
            raise ValueError("normalization input must have shape (rows, 56)")
        mean = values.mean(axis=0, dtype=np.float64)
        scale = values.std(axis=0, dtype=np.float64)
        scale[scale < 1e-8] = 1.0
        return cls(tuple(mean.tolist()), tuple(scale.tolist()), feature_checksum)

    def transform(self, values: NDArray[np.float32]) -> NDArray[np.float32]:
        if values.shape[-1] != len(self.mean):
            raise ValueError("feature dimension does not match normalization statistics")
        return ((values - np.asarray(self.mean)) / np.asarray(self.scale)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "feature_checksum": self.feature_checksum,
                    "mean": list(self.mean),
                    "scale": list(self.scale),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> NormalizationStats:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            mean=tuple(float(value) for value in raw["mean"]),
            scale=tuple(float(value) for value in raw["scale"]),
            feature_checksum=str(raw["feature_checksum"]),
        )


def _collect_counts(ir_text: str) -> dict[str, int]:
    lines = ir_text.splitlines()
    function_lines = [line.strip() for line in lines if line.strip().startswith("define ")]
    body_lines = _function_body_lines(lines)
    counts: dict[str, int] = {
        "instruction_count": count_instructions(ir_text),
        "function_count": len(function_lines),
        "defined_function_count": len(function_lines),
        "basic_block_count": sum(line.endswith(":") for line in body_lines),
        "argument_count": sum(len(re.findall(r"%[-.A-Za-z0-9_$]+", line)) for line in function_lines),
        "global_count": sum(line.startswith("@") and " = " in line for line in lines),
        "metadata_count": sum(line.startswith("!") and " = " in line for line in lines),
        "debug_intrinsic_count": ir_text.count("@llvm.dbg."),
        "function_attribute_count": sum(line.startswith("attributes ") for line in lines),
        "loop_hint_count": ir_text.count("!llvm.loop"),
    }
    for name in FEATURE_OPCODES:
        opcode = name.removesuffix("_count")
        counts[name] = sum(_has_opcode(line, opcode) for line in body_lines)
    return counts


def _function_body_lines(lines: list[str]) -> list[str]:
    body: list[str] = []
    in_function = False
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("define ") and line.endswith("{"):
            in_function = True
            continue
        if in_function and line == "}":
            in_function = False
            continue
        if in_function and line and not line.startswith((";", "!")):
            body.append(line)
    return body


def _has_opcode(line: str, opcode: str) -> bool:
    if line.endswith(":") or "@llvm.dbg." in line:
        return False
    without_assignment = line.split(" = ", 1)[-1]
    return re.match(rf"(?:tail|musttail|notail)?\s*{re.escape(opcode)}\b", without_assignment) is not None


FEATURE_OPCODES = frozenset(
    {
        "phi_count",
        "alloca_count",
        "load_count",
        "store_count",
        "getelementptr_count",
        "call_count",
        "invoke_count",
        "ret_count",
        "br_count",
        "switch_count",
        "indirectbr_count",
        "unreachable_count",
        "icmp_count",
        "fcmp_count",
        "select_count",
        "add_count",
        "sub_count",
        "mul_count",
        "udiv_count",
        "sdiv_count",
        "fadd_count",
        "fsub_count",
        "fmul_count",
        "fdiv_count",
        "and_count",
        "or_count",
        "xor_count",
        "shl_count",
        "lshr_count",
        "ashr_count",
        "trunc_count",
        "zext_count",
        "sext_count",
        "bitcast_count",
        "ptrtoint_count",
        "inttoptr_count",
        "sitofp_count",
        "fptosi_count",
        "freeze_count",
        "extractvalue_count",
        "insertvalue_count",
        "extractelement_count",
        "insertelement_count",
        "shufflevector_count",
        "landingpad_count",
        "resume_count",
    }
)
