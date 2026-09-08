from pathlib import Path

from pipedream.benchmarks import load_manifest, split_manifest


def test_split_manifest_keeps_families_in_one_split(tmp_path) -> None:
    root = Path(__file__).parents[1]
    source = load_manifest(root / "benchmarks" / "manifest.json")
    output = tmp_path / "pilot.json"

    pilot = split_manifest(source, output, source_root=root / "benchmarks")

    assert {record.split for record in pilot.records} == {"train", "validation", "test"}
    family_splits = {}
    for record in pilot.records:
        family_splits.setdefault(record.source_family, set()).add(record.split)
    assert all(len(splits) == 1 for splits in family_splits.values())
