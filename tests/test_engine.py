import json

from pipedream.compiler import append_trace, count_instructions, trace_record
from pipedream.compiler.engine import PassResult


def test_instruction_count_excludes_labels_metadata_and_debug_calls() -> None:
    ir = """
    define i32 @main() {
    entry:
      %value = add i32 20, 22
      call void @llvm.dbg.value(metadata i32 %value, metadata !1, metadata !2)
      ret i32 %value
    }
    !1 = !{}
    """

    assert count_instructions(ir) == 2


def test_trace_record_is_replay_metadata_without_bitcode() -> None:
    result = PassResult(
        status="ok",
        ir=b"bitcode",
        instruction_count=8,
        duration_ms=1.5,
        action_id=2,
        pass_name="instcombine",
        cache_key="abc",
    )

    record = trace_record(step=1, before_count=10, action_id=2, result=result)

    assert record["reward"] == 0.2
    assert "ir" not in record


def test_failed_trace_record_reports_rollback_at_the_previous_count() -> None:
    result = PassResult(
        status="failed",
        ir=b"original",
        instruction_count=0,
        duration_ms=1.5,
        action_id=2,
        pass_name="instcombine",
        cache_key="abc",
        stderr="compiler failure",
    )

    record = trace_record(step=1, before_count=10, action_id=2, result=result)

    assert record["instruction_count"] == 10
    assert record["after_instruction_count"] == 10
    assert record["rolled_back"] is True
    assert record["reward"] == 0.0


def test_append_trace_writes_one_json_object_per_line(tmp_path) -> None:
    trace_path = tmp_path / "steps.jsonl"

    append_trace(trace_path, {"step": 0, "action_id": 2})
    append_trace(trace_path, {"step": 1, "action_id": 6})

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [0, 1]
