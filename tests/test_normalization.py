import numpy as np

from pipedream.representations import NormalizationStats


def test_normalization_stats_round_trip(tmp_path) -> None:
    stats = NormalizationStats.fit(np.ones((2, 56), dtype=np.float32), "checksum")
    path = tmp_path / "stats.json"

    stats.save(path)

    assert NormalizationStats.load(path) == stats
