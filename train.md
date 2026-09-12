# Training Guide

## What Training Means Here

Yes, the model must be trained for the final research result. The existing
smoke and development-pilot checkpoints only verify that PPO, LLVM, rewards,
observations, checkpointing, and evaluation connect correctly. They are not
evidence that PPO improves pass ordering.

A defensible run has three separate stages:

1. Fit normalization on training IR only.
2. Train one or more PPO seeds on training programs and select a checkpoint on
   validation programs only.
3. Evaluate the selected checkpoint once on the locked test split.

Do not use test results to change the model, budget, split, normalization, or
hyperparameters.

## Current Training Status (2026-09-09)

The latest completed run is the overnight development-pilot run. It is a real
trained PPO policy, but it is not yet the final research experiment.

| Item | Recorded value |
|---|---|
| Run | `results/raw/overnight_pilot/ppo` |
| Algorithm/policy | Stable-Baselines3 PPO with `MlpPolicy` |
| Representation | 56-feature Autophase-compatible vector |
| Manifest | `benchmarks/pilot_manifest.json` |
| Split used for training | `train` (8 programs) |
| Validation/test programs | 2 / 2, kept separate from training |
| Seeds | 0, 1, 2, 3, 4 |
| Timesteps | 100,000 per seed; 500,000 total |
| Rollout / batch | 128 / 64 |
| Learning rate / gamma | `3e-4` / `0.99` |
| Episode budget | 12 pass selections |
| Device | CPU |
| Status | All five checkpoints completed |

The run started at `2026-09-08T19:49:09+00:00` and completed at
`2026-09-09T01:39:34+00:00`. The five model files, per-seed configs, metadata,
and training-only normalization statistics are retained with the repository.

Do not report this pilot as the final research result: the benchmark is
smoke-derived and the overnight run still needs validation checkpoint
selection, one locked test evaluation, and baseline comparison.

## Hardware-Aware Plan For This Machine

The project currently runs on this workstation:

| Resource | Detected specification | Training implication |
|---|---|---|
| CPU | AMD Ryzen 7 7435HS, 8 cores / 16 threads | LLVM compilation and pass execution are CPU-bound. Run one PPO seed at a time. |
| RAM | 24 GiB installed, 8 GiB swap | Keep at least 6 GiB free; do not parallelize the five seeds. |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB VRAM | Optional for PPO only; it does not accelerate `clang` or `opt`. |
| Storage | Local workspace volume | Keep checkpoints, TensorBoard events, JSONL, and manifests together under `results/raw/<run-name>/`. |

For the present small `MlpPolicy`, use `--device cpu` as the default. The
overhead of moving small observations to an 8 GiB laptop GPU can outweigh any
PPO benefit, while LLVM remains on the CPU. Try `--device cuda` only as a
separate, documented pilot after `nvidia-smi` works inside the container; never
mix CPU and CUDA seeds in one comparison without recording that decision.

Use power connected and a performance thermal profile for long runs. Start
with one seed and confirm its checkpoint, TensorBoard events, and metadata
exist before scheduling the remaining four sequentially. A main run of five
500,000-step seeds may take multiple overnight sessions on this machine, so
use a new named run directory rather than overwriting a completed pilot.

### Mandatory Host Check For A New Agent Or Machine

This table is a snapshot, not an assumption that every contributor has the
same computer. Before an agent starts, resumes, tunes, or evaluates a training
run, it must inspect the machine it is actually using:

```bash
lscpu
free -h
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
df -h .
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi || true
```

If CPU model/thread count, available RAM, GPU name/VRAM, driver/CUDA access, or
free storage differs materially from the table above, first update the
**Training Workstation Profile** in `README.md` and the table in this file.
Record the date, host/container distinction, and the device selected for the
run in that run's metadata. Do this before choosing a timestep budget or
starting a seed; do not copy a CPU/GPU recommendation from another machine.

Use the observations to decide the next action:

| Observed condition | Next action |
|---|---|
| No CUDA device inside the container | Use `--device cpu`; do not attempt GPU training. |
| 8 GiB laptop GPU and the current small MLP | Prefer CPU; benchmark one short CUDA seed only if desired. |
| GPU with enough VRAM and a larger future network/vectorized environment | Run a recorded CPU-vs-CUDA pilot, then lock one device for the full comparison. |
| Less than 6 GiB free RAM or low disk space | Stop before training; free capacity or reduce only a newly declared pilot budget. |
| More CPU cores/RAM than this workstation | Still begin with one seed and measure LLVM throughput before adding controlled parallelism. |

An agent doing analysis only should inspect existing artifacts and the recorded
profile, but must not change hardware recommendations merely because it is
running in a different environment. Update the profile only when preparing or
reviewing an experiment that will use that environment.

### Optional Live Monitor

The monitor is read-only and runs on the host. In another terminal, after the
training container has started:

```bash
python3 monitoring/server.py
```

Open <http://127.0.0.1:8765>. It reports the configured Docker container,
per-seed checkpoints/TensorBoard progress, container usage, and GPU telemetry.
For a different run, configure all three values together:

```bash
PIPEDREAM_CONTAINER=pipedream-main \
PIPEDREAM_OUTPUT_DIR=results/raw/main/ppo \
PIPEDREAM_TOTAL_TIMESTEPS=500000 \
python3 monitoring/server.py
```

The monitor never controls training. If its TensorBoard rows are empty, first
confirm the container name and that the host project is mounted at
`/workspace` inside the container.

## Immediate Next Step: Select And Evaluate The Completed Run

Do this before starting another PPO run. These commands use only validation
records for checkpoint selection and use the test split exactly once.

```bash
MANIFEST=benchmarks/pilot_manifest.json
CATALOG=configs/pass_catalog.yaml
SCHEMA=configs/autophase_schema.yaml
NORM=results/raw/overnight_pilot/normalization_train.json
RUN=results/raw/overnight_pilot/ppo

uv run --extra rl pipedream-select-checkpoint \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --schema "$SCHEMA" \
  --normalization-stats "$NORM" \
  --models "$RUN/seed_0/ppo_model.zip" \
           "$RUN/seed_1/ppo_model.zip" \
           "$RUN/seed_2/ppo_model.zip" \
           "$RUN/seed_3/ppo_model.zip" \
           "$RUN/seed_4/ppo_model.zip" \
  --output results/raw/overnight_pilot/selection.json

MODEL=$(python -c \
  'import json; print(json.load(open("results/raw/overnight_pilot/selection.json"))["selected_model"])')

uv run --extra rl pipedream-evaluate \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --schema "$SCHEMA" \
  --normalization-stats "$NORM" \
  --model "$MODEL" \
  --split test \
  --output results/raw/overnight_pilot/test.jsonl \
  --max-steps 12
```

After the test evaluation is written, run the matching baselines and paired
analysis. Do not use the test output to change the selected checkpoint or PPO
configuration.

## Next Training Specification: Main Benchmark

The next actual training run should use a frozen, family-aware main manifest
with legally usable benchmark sources. Acquire and validate that manifest
before training; do not substitute the 12-program pilot for the main result.

Predeclare this configuration for the first main run:

| Parameter | Value |
|---|---|
| Algorithm | PPO, Stable-Baselines3 `MlpPolicy` |
| Observation | 56-feature Autophase-compatible vector |
| Action space | 12-pass catalog in `configs/pass_catalog.yaml` |
| Reward | Relative instruction-count reduction |
| Training split | Main manifest `train` only |
| Seeds | `0 1 2 3 4` |
| Timesteps | 500,000 per seed; 2,500,000 total |
| Rollout steps | 128 |
| Batch size | 64 |
| Learning rate | `0.0003` |
| Discount factor | `0.99` |
| Episode max steps | 12 |
| Device | `auto` (record the resolved device) |
| Normalization | Fit on training IR only; freeze before validation/test |
| Selection | Best mean instruction reduction on validation only |
| Final evaluation | One pass on the locked test split |

Use the following command after replacing `MANIFEST` with the frozen main
manifest and fitting its training-only normalization statistics:

```bash
uv run --extra rl pipedream-train \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --schema "$SCHEMA" \
  --normalization-stats results/raw/main/normalization_train.json \
  --split train \
  --output-dir results/raw/main/ppo \
  --seeds 0 1 2 3 4 \
  --total-timesteps 500000 \
  --rollout-steps 128 \
  --batch-size 64 \
  --max-steps 12 \
  --device auto
```

The 500,000-step main budget is a predeclared starting point, not a claim that
more steps always improve results. If convergence or compute cost requires a
change, record the new budget and create a new experiment directory before
using any test results.

## Environment

The preferred environment is the repository's Docker image. A local Python
3.11 environment with LLVM 20 and `uv` is also supported.

```bash
docker build -t pipedream:llvm20 .
docker run --rm -v "$PWD":/workspace pipedream:llvm20
```

For local development, install the development and RL extras:

```bash
uv sync --all-groups --extra rl
```

The RL extra includes PyTorch, Stable-Baselines3, and TensorBoard. On a CPU
machine, pass `--device cpu` to training. `--device auto` is the default.

## Prepare A Manifest

The checked-in smoke and pilot manifests are already reproducible. For a new
benchmark directory, generate and validate a manifest first:

```bash
uv run pipedream-manifest \
  --source-dir benchmarks/anghaben_subset \
  --manifest benchmarks/anghaben_manifest.json \
  --tier main \
  --split all \
  --clang clang \
  --llvm-version llvm20

uv run pipedream-split-manifest \
  --input benchmarks/anghaben_manifest.json \
  --output benchmarks/anghaben_pilot_manifest.json \
  --tier main

uv run pipedream-manifest \
  --manifest benchmarks/anghaben_pilot_manifest.json \
  --validate
```

Replace the placeholder directory and filenames with the frozen benchmark
paths. Keep source-family groups intact across splits.

For the current development pilot, use:

```bash
MANIFEST=benchmarks/pilot_manifest.json
CATALOG=configs/pass_catalog.yaml
SCHEMA=configs/autophase_schema.yaml
```

## Fit Training-Only Normalization

```bash
uv run pipedream-fit-normalizer \
  --manifest "$MANIFEST" \
  --schema "$SCHEMA" \
  --split train \
  --output results/raw/normalization_train.json
```

The resulting checksum is part of the experiment identity. Refit it whenever
the training split, compiler version, feature schema, or IR canonicalization
changes.

## Train PPO

Before the main run, predeclare the timestep budget, rollout length, batch
size, episode budget, seeds, and device in the experiment record. The command
below is the locked starting specification described above. The 16-step pilot
budget is only a smoke setting and must not be used as the final claim.

```bash
uv run --extra rl pipedream-train \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --schema "$SCHEMA" \
  --normalization-stats results/raw/normalization_train.json \
  --split train \
  --output-dir results/raw/main/ppo \
  --seeds 0 1 2 3 4 \
  --total-timesteps 500000 \
  --rollout-steps 128 \
  --batch-size 64 \
  --max-steps 12 \
  --device auto
```

The final timestep budget is an experiment decision, not a magic constant.
Increase it after measuring convergence and compute cost, then record the
decision before looking at test results. Each multi-seed run writes a model,
`training_config.json`, `run_metadata.json`, and TensorBoard logs under its
seed directory.

On the current workstation, run the seed sweep sequentially. The CLI accepts
multiple seeds, but invoking the command separately makes progress, restart,
and thermal/power failures easier to isolate:

```bash
for SEED in 0 1 2 3 4; do
  uv run --extra rl pipedream-train \
    --manifest "$MANIFEST" \
    --catalog "$CATALOG" \
    --schema "$SCHEMA" \
    --normalization-stats results/raw/main/normalization_train.json \
    --split train \
    --output-dir results/raw/main/ppo \
    --seeds "$SEED" \
    --total-timesteps 500000 \
    --rollout-steps 128 \
    --batch-size 64 \
    --max-steps 12 \
    --device cpu || exit 1
done
```

This loop is intentionally resumable by seed, but do not silently replace an
existing seed checkpoint. If a seed must be rerun, use a fresh experiment
directory (for example `results/raw/main_rerun_2026-09-13/`) and record why.

## Select A Validation Checkpoint

Selection reads only validation records. Give it all candidate seed models:

```bash
uv run --extra rl pipedream-select-checkpoint \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --schema "$SCHEMA" \
  --normalization-stats results/raw/normalization_train.json \
  --models results/raw/main/ppo/seed_0/ppo_model.zip \
           results/raw/main/ppo/seed_1/ppo_model.zip \
           results/raw/main/ppo/seed_2/ppo_model.zip \
           results/raw/main/ppo/seed_3/ppo_model.zip \
           results/raw/main/ppo/seed_4/ppo_model.zip \
  --output results/raw/main/selection.json
```

The selection artifact records the validation split and chosen model. Keep it
with the run; do not hand-edit it.

## Evaluate Once On Test

Read the selected model from the selection artifact and evaluate it on the
locked test split:

```bash
MODEL=$(python -c \
  'import json; print(json.load(open("results/raw/main/selection.json"))["selected_model"])')

uv run --extra rl pipedream-evaluate \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --schema "$SCHEMA" \
  --normalization-stats results/raw/normalization_train.json \
  --model "$MODEL" \
  --split test \
  --output results/raw/main/test.jsonl \
  --max-steps 12
```

The evaluator records per-program actions, final instruction counts, reward,
compiler failures, and model metadata. A failed program is retained as a
failure record rather than silently removed.

## Run Baselines And Analysis

Run baselines against the same manifest and episode budget:

```bash
uv run pipedream-baselines \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --split test \
  --methods -O2 -O3 -Oz random greedy beam \
  --seeds 0 1 2 3 4 \
  --max-steps 12 \
  --output results/raw/baselines_test.jsonl
```

Use the paired analysis command only after both result files are complete:

```bash
uv run --extra analysis pipedream-analyze \
  --baseline results/raw/baselines_test.jsonl \
  --candidate results/raw/ppo_main/test.jsonl \
  --metric final_instruction_count \
  --output results/raw/ppo_main/paired_analysis.json
```

For a compact method table:

```bash
uv run --extra analysis pipedream-report \
  --input results/raw/baselines_test.jsonl \
  --output results/raw/baselines_test.csv
```

## Reproducibility Checklist

- Record the git commit, manifest checksum, compiler versions, Python version,
  pass catalog, feature schema, normalization checksum, PPO configuration,
  seed list, and device.
- Keep raw JSONL results, checkpoints, TensorBoard logs, and failure records.
- Run `uv run pytest -q`, `uv run ruff check .`, and `uv lock --check` before a
  result is treated as publishable.
- Never tune on the test split.
- Report absolute instruction counts, relative reduction, failures, compiler
  time, representation time, and total optimization cost.

## Current Limitation

The repository has an IR2Vec adapter and configuration, but the exact IR2Vec
executable and vocabulary are not part of the current environment. Do not
claim an IR2Vec comparison until those inputs are frozen and the matched
Autophase/IR2Vec experiments are complete.
