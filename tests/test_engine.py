import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from pipedream.compiler import append_trace, count_instructions, trace_record
from pipedream.compiler.catalog import PassCatalog, PassSpec
from pipedream.compiler.engine import CompilerConfig, PassEngine, PassResult


def _create_test_catalog() -> PassCatalog:
    specs = (
        PassSpec(
            action_id=0,
            name="mem2reg",
            pipeline="mem2reg",
            category="scalar",
        ),
        PassSpec(
            action_id=1,
            name="instcombine",
            pipeline="instcombine",
            category="scalar",
        ),
    )
    return PassCatalog(version="test-v1", llvm_major="20", passes=specs)


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


def test_append_trace_writes_one_json_object_per_line(tmp_path: Path) -> None:
    trace_path = tmp_path / "steps.jsonl"

    append_trace(trace_path, {"step": 0, "action_id": 2})
    append_trace(trace_path, {"step": 1, "action_id": 6})

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [0, 1]


def test_pass_engine_apply_success(tmp_path: Path) -> None:
    catalog = _create_test_catalog()
    engine = PassEngine(catalog, cache_dir=tmp_path / "cache")
    initial_ir = b"input_bitcode"

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd[0] == "opt":
            out_file = Path(cmd[4])
            out_file.write_bytes(b"optimized_bitcode")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "llvm-dis":
            out_file = Path(cmd[3])
            out_file.write_text(
                "define i32 @f() {\nentry:\n  %1 = add i32 1, 2\n  ret i32 %1\n}\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise ValueError(f"Unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        result = engine.apply(initial_ir, action_id=0)

    assert result.status == "ok"
    assert result.ir == b"optimized_bitcode"
    assert result.instruction_count == 2
    assert result.action_id == 0
    assert result.pass_name == "mem2reg"
    assert result.committed is True


def test_pass_engine_apply_noop() -> None:
    catalog = _create_test_catalog()
    engine = PassEngine(catalog)
    initial_ir = b"input_bitcode"

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd[0] == "opt":
            out_file = Path(cmd[4])
            out_file.write_bytes(b"same_bitcode")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "llvm-dis":
            out_file = Path(cmd[3])
            out_file.write_text(
                "define i32 @f() {\nentry:\n  ret i32 0\n}\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise ValueError(f"Unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        result = engine.apply(initial_ir, action_id=0)

    assert result.status == "ok"
    assert result.instruction_count == 1
    rec = trace_record(step=0, before_count=1, action_id=0, result=result)
    assert rec["reward"] == 0.0
    assert rec["rolled_back"] is False
    assert rec["instruction_count"] == 1


def test_pass_engine_apply_failure_preserves_ir_and_rolls_back() -> None:
    catalog = _create_test_catalog()
    engine = PassEngine(catalog)
    initial_ir = b"original_bitcode_pre_failure"

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd[0] == "opt":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="opt syntax error")
        raise ValueError(f"Unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        result = engine.apply(initial_ir, action_id=1)

    assert result.status == "failed"
    assert result.ir == initial_ir
    assert result.instruction_count == 0
    assert "opt syntax error" in result.stderr
    assert result.committed is False

    rec = trace_record(step=0, before_count=10, action_id=1, result=result)
    assert rec["rolled_back"] is True
    assert rec["instruction_count"] == 10
    assert rec["after_instruction_count"] == 10
    assert rec["reward"] == 0.0


def test_pass_engine_apply_timeout_preserves_ir() -> None:
    catalog = _create_test_catalog()
    config = CompilerConfig(timeout_seconds=0.01)
    engine = PassEngine(catalog, config=config)
    initial_ir = b"original_bitcode"

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd[0] == "opt":
            raise subprocess.TimeoutExpired(cmd, 0.01)
        raise ValueError(f"Unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        result = engine.apply(initial_ir, action_id=0)

    assert result.status == "failed"
    assert result.ir == initial_ir
    assert "timeout" in result.stderr
    assert result.committed is False

    rec = trace_record(step=0, before_count=8, action_id=0, result=result)
    assert rec["rolled_back"] is True
    assert rec["after_instruction_count"] == 8
    assert rec["reward"] == 0.0


def test_pass_engine_caching_avoids_reexecution(tmp_path: Path) -> None:
    catalog = _create_test_catalog()
    engine = PassEngine(catalog, cache_dir=tmp_path / "cache")
    initial_ir = b"input_bitcode"
    opt_call_count = 0

    def fake_subprocess_run(cmd, *args, **kwargs):
        nonlocal opt_call_count
        if cmd[0] == "opt":
            opt_call_count += 1
            out_file = Path(cmd[4])
            out_file.write_bytes(b"cached_output_bitcode")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "llvm-dis":
            out_file = Path(cmd[3])
            out_file.write_text(
                "define i32 @f() {\nentry:\n  ret i32 0\n}\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise ValueError(f"Unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        res1 = engine.apply(initial_ir, action_id=0)
        assert res1.status == "ok"
        assert opt_call_count == 1

        res2 = engine.apply(initial_ir, action_id=0)
        assert res2.status == "cached"
        assert res2.ir == b"cached_output_bitcode"
        assert res2.committed is True
        assert opt_call_count == 1
