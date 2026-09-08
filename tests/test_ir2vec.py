import numpy as np
import pytest

from pipedream.representations import IR2VecConfig, IR2VecExtractor, IR2VecUnavailable, mean_pool


def test_ir2vec_config_and_empty_pool_are_deterministic() -> None:
    config = IR2VecConfig("v1", "symbolic", 4)
    extractor = IR2VecExtractor(config)

    assert len(config.checksum) == 64
    np.testing.assert_array_equal(mean_pool(np.empty((0, 4), dtype=np.float32), 4), np.zeros(4))
    if not extractor.available():
        with pytest.raises(IR2VecUnavailable):
            extractor.extract_text("define void @f() { ret void }\n")
