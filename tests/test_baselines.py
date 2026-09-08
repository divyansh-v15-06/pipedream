from pipedream.evaluation.baselines import PIPELINES


def test_required_pipeline_baselines_are_explicit() -> None:
    assert PIPELINES == {
        "-O2": "default<O2>",
        "-O3": "default<O3>",
        "-Oz": "default<Oz>",
    }
