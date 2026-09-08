from pathlib import Path

import numpy as np

from pipedream.representations import AutophaseExtractor, FeatureSchema, NormalizationStats


def test_project_autophase_schema_is_56_dimensional() -> None:
    root = Path(__file__).parents[1]
    schema = FeatureSchema.from_yaml(root / "configs" / "autophase_schema.yaml")
    extractor = AutophaseExtractor(schema)
    values = extractor.extract((root / "tests" / "fixtures" / "simple.ll").read_text())
    assert values.shape == (56,)
    assert values[schema.names.index("instruction_count")] == 2
    assert values[schema.names.index("add_count")] == 1


def test_normalization_is_reusable() -> None:
    values = np.asarray([[1] * 56, [3] * 56], dtype=np.float32)
    stats = NormalizationStats.fit(values, "schema")

    transformed = stats.transform(values)

    np.testing.assert_allclose(transformed.mean(axis=0), 0.0)
    assert stats.feature_checksum == "schema"
