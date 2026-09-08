import json

from pipedream.evaluation.report import build_report


def test_report_groups_common_schema_rows(tmp_path) -> None:
    source = tmp_path / "results.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "method": method,
                    "instruction_reduction": reduction,
                    "final_instruction_count": final_count,
                    "total_optimization_time_ms": 1.0,
                }
            )
            for method, reduction, final_count in (("random", 0.1, 9), ("random", 0.2, 8))
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_report(source, tmp_path / "report.json")

    assert report[0]["method"] == "random"
    assert report[0]["programs"] == 2
