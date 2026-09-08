from pathlib import Path

from pipedream.benchmarks import load_manifest


def test_smoke_manifest_can_be_filtered_for_held_out_evaluation() -> None:
    manifest = load_manifest(Path(__file__).parents[1] / "benchmarks" / "manifest.json")
    smoke_records = tuple(record for record in manifest.records if record.split == "smoke")

    assert len(smoke_records) == 12
