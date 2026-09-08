"""Versioned LLVM pass catalog loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PassSpec:
    action_id: int
    name: str
    pipeline: str
    category: str


@dataclass(frozen=True)
class PassCatalog:
    version: str
    llvm_major: str
    passes: tuple[PassSpec, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> PassCatalog:
        with path.open(encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle)

        if not isinstance(raw, dict):
            raise TypeError("pass catalog must contain a YAML mapping")
        version = _required_string(raw, "version")
        llvm_major = _required_string(raw, "llvm_major")
        raw_passes = raw.get("passes")
        if not isinstance(raw_passes, list) or not raw_passes:
            raise ValueError("pass catalog must contain a non-empty passes list")

        passes = tuple(_parse_pass(item) for item in raw_passes)
        action_ids = [item.action_id for item in passes]
        names = [item.name for item in passes]
        if action_ids != list(range(len(passes))):
            raise ValueError("pass action IDs must be contiguous and start at zero")
        if len(names) != len(set(names)):
            raise ValueError("pass names must be unique")
        if any(not item.pipeline for item in passes):
            raise ValueError("pass pipelines cannot be empty")
        return cls(version=version, llvm_major=llvm_major, passes=passes)

    def by_action_id(self, action_id: int) -> PassSpec:
        if action_id < 0:
            raise KeyError(f"unknown action ID: {action_id}")
        try:
            return self.passes[action_id]
        except IndexError as exc:
            raise KeyError(f"unknown action ID: {action_id}") from exc

    def by_name(self, name: str) -> PassSpec:
        for spec in self.passes:
            if spec.name == name:
                return spec
        raise KeyError(f"unknown pass name: {name}")


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"catalog field {key!r} must be a non-empty string")
    return value


def _parse_pass(raw: Any) -> PassSpec:
    if not isinstance(raw, dict):
        raise TypeError("each catalog pass must be a YAML mapping")
    action_id = raw.get("action_id")
    if not isinstance(action_id, int) or isinstance(action_id, bool):
        raise TypeError("pass action_id must be an integer")
    return PassSpec(
        action_id=action_id,
        name=_required_string(raw, "name"),
        pipeline=_required_string(raw, "pipeline"),
        category=_required_string(raw, "category"),
    )
