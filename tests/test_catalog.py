from pathlib import Path

from pipedream.compiler import PassCatalog


def test_smoke_catalog_is_versioned_and_contiguous() -> None:
    catalog_path = Path(__file__).parents[1] / "configs" / "pass_catalog.yaml"
    catalog = PassCatalog.from_yaml(catalog_path)

    assert catalog.version == "llvm20-v1"
    assert catalog.llvm_major == "20"
    assert len(catalog.passes) == 12
    assert [item.action_id for item in catalog.passes] == list(range(12))
