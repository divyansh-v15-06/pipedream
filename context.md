# Pipedream Peer Context

This is the short onboarding document for contributors joining the Pipedream
project. The README is the research specification; this file describes the
state of the implementation and the safest way to extend it.

## What This Project Does

Pipedream studies whether one PPO policy can learn program-adaptive LLVM pass
sequences. The policy observes the current LLVM IR, selects one pass from a
fixed catalog, receives an instruction-count reward, and repeats for a fixed
episode budget.

The central experiment is:

```text
training programs -> PPO -> validation checkpoint selection -> unseen test programs
```

The comparison set is LLVM `-O2`, `-O3`, `-Oz`, random search, greedy search,
and shallow beam search. The first representation is a project-owned,
Autophase-compatible 56-feature vector. IR2Vec is a planned controlled
ablation, not yet a completed result.

## Current State

Completed and verified:

- Python 3.11, LLVM/Clang/opt 20.1.8, and `uv.lock` are pinned.
- A Docker image builds and passes the LLVM compile/optimize/verify smoke test.
- The pass engine supports verification, timeout handling, rollback, caching,
  instruction counting, and JSONL traces.
- The repository contains 12 owned C smoke fixtures and a checksum manifest.
- A development pilot manifest has train/validation/test splits: 8/2/2.
- `PipedreamEnv` passes Gymnasium checks and supports deterministic resets.
- Baselines, the 56-feature extractor, training-only normalization, PPO,
  checkpoint selection, deterministic evaluation, and paired bootstrap analysis
  are implemented.
- Five development-pilot seeds were trained for 100,000 PPO timesteps each
  (500,000 total) in `results/raw/overnight_pilot/ppo`. The five checkpoints,
  per-seed metadata/configuration, and training-only normalization statistics are
  versioned with the repository. The run completed, but its validation
  selection and held-out test evaluation are still the next actions.
- A local read-only dashboard in `monitoring/` can follow a Docker training
  container, TensorBoard event files, checkpoints, host resources, and NVIDIA
  telemetry. It is support tooling only; it cannot control a training run.

Open research boundaries:

- The external AnghaBench subset has not been acquired or frozen locally.
- The development pilot is smoke-derived and is not the final benchmark.
- The main benchmark, final training budget, figures, and final held-out result
  are still open. The current overnight run is a development-pilot result, not
  the final research claim.
- The exact IR2Vec executable/vocabulary and matched ablation are still open.

## Repository Map

| Path | Responsibility |
|---|---|
| `configs/pass_catalog.yaml` | Versioned action space |
| `configs/autophase_schema.yaml` | Ordered 56-feature schema |
| `configs/ir2vec.yaml` | Deferred IR2Vec adapter configuration |
| `benchmarks/manifest.json` | Immutable smoke manifest |
| `benchmarks/pilot_manifest.json` | Development train/validation/test split |
| `src/pipedream/compiler/` | LLVM command execution and replay state |
| `src/pipedream/benchmarks/` | Manifest, compilation, and split logic |
| `src/pipedream/env/` | Gymnasium environment |
| `src/pipedream/representations/` | Autophase, normalization, and IR2Vec adapter |
| `src/pipedream/agents/` | PPO configuration and training |
| `src/pipedream/evaluation/` | Baselines, PPO evaluation, reports, statistics |
| `src/pipedream/cli/` | User-facing command-line entry points |
| `tests/` | Unit and integration tests |
| `TODO.md` | Chunk-level implementation board |
| `train.md` | Reproducible training/evaluation procedure |
| `remaining_work.md` | Prioritized work after the current pilot |

## Development Contract

Use the pinned environment and keep experiment changes explicit. A change to
the LLVM version, pass catalog, reward, observation schema, split, episode
budget, or PPO budget creates a new experiment version and must be recorded in
metadata and documentation.

The test split must not be used for fitting normalization, selecting
checkpoints, tuning hyperparameters, or deciding when to stop training. Every
result should retain its manifest, configuration, seed, model path, toolchain,
and failure records.

### Required First Check On A Different Machine

Before any training-related action, inspect the active host's CPU, RAM, free
disk, GPU/VRAM, and whether CUDA is visible inside Docker. `train.md` contains
the exact commands and the decision table. If those specifications differ from
the documented workstation profile, update `README.md` and `train.md` with the
new dated profile before selecting device, parallelism, or budget. Then record
the resolved device and hardware details in the new experiment directory.

For analysis-only work, read the existing profile and run metadata; do not
rewrite the profile just because the analysis agent is on a different host.

Useful checks:

```bash
uv run pytest -q
uv run ruff check .
uv lock --check
docker build -t pipedream:llvm20 .
docker run --rm -v "$PWD":/workspace pipedream:llvm20
```

The container entrypoint runs the M0 smoke test. PPO commands require the RL
extra (`uv run --extra rl ...`); analysis commands that need tabular/statistical
packages require the analysis extra.

## First Places To Read Before Editing

1. `README.md` for the research contract and locked decisions.
2. `TODO.md` for chunk gates and current evidence.
3. `train.md` for the experiment protocol.
4. The relevant module tests before changing a behavior.
5. `monitoring/README.md` when observing a long local training run.

Keep the public command-line interfaces stable when possible. Add a focused
test for behavior changes, and prefer extending the existing result metadata
over creating a second incompatible schema.
