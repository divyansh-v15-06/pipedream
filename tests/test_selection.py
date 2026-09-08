from pathlib import Path

from pipedream.benchmarks import load_manifest


def test_pilot_manifest_has_a_validation_split() -> None:
    manifest = load_manifest(Path(__file__).parents[1] / "benchmarks" / "pilot_manifest.json")

    assert sum(record.split == "validation" for record in manifest.records) == 2
