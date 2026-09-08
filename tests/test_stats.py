import numpy as np

from pipedream.analysis import paired_bootstrap, summarize_differences


def test_paired_statistics_are_seeded_and_reproducible() -> None:
    values = np.asarray([1.0, 2.0, 3.0, -1.0])

    first = paired_bootstrap(values, samples=200, seed=7)
    second = paired_bootstrap(values, samples=200, seed=7)
    summary = summarize_differences(values)

    assert first == second
    assert summary["improved_fraction"] == 0.75
    assert first.lower <= first.estimate <= first.upper
