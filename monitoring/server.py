#!/usr/bin/env python3
"""Local monitoring server for Pipedream training runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent
CONTAINER = os.environ.get("PIPEDREAM_CONTAINER", "pipedream-overnight")
OUTPUT_DIR = ROOT / os.environ.get(
    "PIPEDREAM_OUTPUT_DIR", "results/raw/overnight_pilot/ppo"
)
TOTAL_TIMESTEPS = int(os.environ.get("PIPEDREAM_TOTAL_TIMESTEPS", "100000"))


def command(args: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def inspect_container() -> dict[str, Any] | None:
    try:
        result = command(["docker", "inspect", CONTAINER])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)[0]
    except (IndexError, json.JSONDecodeError):
        return None


def parse_timestamp(value: str | None) -> datetime | None:
    if not value or value.startswith("0001-"):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def duration_seconds(started: datetime | None) -> float | None:
    if started is None:
        return None
    return max(0.0, (datetime.now(UTC) - started).total_seconds())


def parse_size(value: str) -> float:
    match = re.match(r"\s*([\d.]+)\s*([KMGT]?i?B)?", value or "", re.IGNORECASE)
    if not match:
        return 0.0
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    multipliers = {"B": 1, "KB": 1000, "KIB": 1024, "MB": 1000**2, "MIB": 1024**2,
                   "GB": 1000**3, "GIB": 1024**3, "TB": 1000**4, "TIB": 1024**4}
    return number * multipliers.get(unit, 1)


def docker_stats() -> dict[str, Any] | None:
    try:
        result = command(["docker", "stats", "--no-stream", "--format", "{{json .}}", CONTAINER])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        raw = json.loads(result.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return None
    memory = raw.get("MemUsage", "")
    used, _, limit = memory.partition("/")
    return {
        "cpuPercent": _percent(raw.get("CPUPerc")),
        "memoryBytes": parse_size(used),
        "memoryLimitBytes": parse_size(limit),
        "memoryPercent": _percent(raw.get("MemPerc")),
        "network": raw.get("NetIO", "—"),
        "blockIo": raw.get("BlockIO", "—"),
        "pids": int(raw.get("PIDs", "0") or 0),
    }


def _percent(value: str | None) -> float:
    try:
        return float((value or "0").strip().rstrip("%"))
    except ValueError:
        return 0.0


def gpu_stats() -> dict[str, Any] | None:
    try:
        result = command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    if len(parts) != 4:
        return None
    try:
        return {
            "name": parts[0],
            "memoryTotalMiB": float(parts[1]),
            "memoryUsedMiB": float(parts[2]),
            "utilizationPercent": float(parts[3]),
        }
    except ValueError:
        return None


def tensorboard_progress() -> list[dict[str, Any]]:
    # The repository root is mounted at /workspace in the project container.
    # Keep this derived from the monitor's configured output directory so a
    # dashboard can follow a named experiment other than the overnight pilot.
    container_output_dir = Path("/workspace") / OUTPUT_DIR.relative_to(ROOT)
    script = r'''
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(__OUTPUT_DIR__)
rows = []
for seed_dir in sorted(root.glob("seed_*")):
    match = seed_dir.name.removeprefix("seed_")
    if not match.isdigit():
        continue
    files = sorted(seed_dir.glob("tensorboard/*/events.out.tfevents.*"))
    row = {"seed": int(match), "step": 0, "fps": 0.0, "reward": None, "episodeLength": None}
    model = seed_dir / "ppo_model.zip"
    row["modelExists"] = model.is_file()
    if files:
        try:
            events = EventAccumulator(str(files[-1]))
            events.Reload()
            def last(tag):
                values = events.Scalars(tag)
                return values[-1] if values else None
            step = last("time/fps")
            reward = last("rollout/ep_rew_mean")
            length = last("rollout/ep_len_mean")
            if step:
                row["step"] = int(step.step)
                row["fps"] = float(step.value)
            if reward:
                row["reward"] = float(reward.value)
            if length:
                row["episodeLength"] = float(length.value)
        except Exception as exc:
            row["error"] = str(exc)
    rows.append(row)
print(json.dumps(rows))
'''.replace("__OUTPUT_DIR__", repr(str(container_output_dir)))
    try:
        result = command(
            ["docker", "exec", CONTAINER, "/workspace/.venv/bin/python", "-c", script],
            timeout=12.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return []


def container_logs() -> list[str]:
    try:
        result = command(["docker", "logs", "--tail", "80", CONTAINER], timeout=8.0)
    except (OSError, subprocess.TimeoutExpired):
        return ["Unable to read container logs."]
    output = result.stdout + result.stderr
    return output.splitlines()[-80:]


def system_stats() -> dict[str, Any]:
    memory = {}
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, raw = line.partition(":")
            values[key] = int(raw.strip().split()[0]) * 1024
        memory = {
            "totalBytes": values.get("MemTotal", 0),
            "availableBytes": values.get("MemAvailable", 0),
            "usedBytes": values.get("MemTotal", 0) - values.get("MemAvailable", 0),
        }
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(ROOT)
    return {
        "memory": memory,
        "loadAverage": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "cpuCount": os.cpu_count() or 1,
        "disk": {"totalBytes": disk.total, "freeBytes": disk.free, "usedBytes": disk.used},
    }


def get_status() -> dict[str, Any]:
    inspected = inspect_container()
    state = inspected.get("State", {}) if inspected else {}
    started = parse_timestamp(state.get("StartedAt"))
    progress = tensorboard_progress() if inspected else []
    seed_dirs = sorted(OUTPUT_DIR.glob("seed_*"))
    known_seeds = {int(path.name.removeprefix("seed_")) for path in seed_dirs if path.name.removeprefix("seed_").isdigit()}
    known_seeds.update(int(row["seed"]) for row in progress)
    if not known_seeds:
        known_seeds = set(range(5))
    by_seed = {int(row["seed"]): row for row in progress}
    seeds = []
    for seed in sorted(known_seeds):
        row = by_seed.get(seed, {"seed": seed, "step": 0, "fps": 0.0, "reward": None, "episodeLength": None})
        model_exists = bool(row.get("modelExists")) or (OUTPUT_DIR / f"seed_{seed}" / "ppo_model.zip").is_file()
        step = min(TOTAL_TIMESTEPS, int(row.get("step", 0)))
        seeds.append({
            **row,
            "seed": seed,
            "step": step,
            "totalTimesteps": TOTAL_TIMESTEPS,
            "progressPercent": round(step * 100 / TOTAL_TIMESTEPS, 2),
            "status": "completed" if model_exists else "training" if step else "queued",
            "modelExists": model_exists,
        })
    completed = sum(seed["status"] == "completed" for seed in seeds)
    current = next((seed for seed in seeds if seed["status"] == "training"), None)
    fps = float(current.get("fps", 0)) if current else 0.0
    remaining_steps = sum(seed["totalTimesteps"] - seed["step"] for seed in seeds)
    eta = remaining_steps / fps if fps > 0 else None
    total_steps = sum(seed["step"] for seed in seeds)
    command_text = " ".join((inspected or {}).get("Config", {}).get("Cmd", []))
    status = state.get("Status", "missing")
    return {
        "now": datetime.now(UTC).isoformat(),
        "container": {
            "name": CONTAINER,
            "status": status,
            "running": bool(state.get("Running")),
            "exitCode": state.get("ExitCode"),
            "startedAt": state.get("StartedAt"),
            "elapsedSeconds": duration_seconds(started),
            "command": command_text,
        },
        "run": {
            "totalSeeds": len(seeds),
            "completedSeeds": completed,
            "totalTimesteps": TOTAL_TIMESTEPS,
            "totalSteps": total_steps,
            "progressPercent": round(total_steps * 100 / max(1, len(seeds) * TOTAL_TIMESTEPS), 2),
            "etaSeconds": eta,
            "activeSeed": current["seed"] if current else None,
            "outputDir": str(OUTPUT_DIR.relative_to(ROOT)),
        },
        "seeds": seeds,
        "resources": {"docker": docker_stats(), "gpu": gpu_stats(), "system": system_stats()},
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._json(get_status())
            return
        if path == "/api/logs":
            self._json({"lines": container_logs()})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def _json(self, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Pipedream monitor: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
