import json

import pytest

from pipedream.benchmarks.manifest import load_manifest


def test_smoke_manifest_has_unique_ids_and_source_families(tmp_path) -> None:
    source = tmp_path / "sample.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "pipedream-manifest-v1",
                "tier": "smoke",
                "records": [
                    {
                        "benchmark_id": "smoke_001",
                        "source": "sample.c",
                        "source_sha256": "not-computed",
                        "source_family": "sample",
                        "split": "smoke",
                        "language": "c",
                        "standard": "c11",
                        "compiler_flags": ["-O0"],
                        "target_triple": "x86_64-unknown-linux-gnu",
                        "initial_ir_sha256": "not-computed",
                        "llvm_version": "llvm20",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source checksum mismatch"):
        load_manifest(manifest_path)
