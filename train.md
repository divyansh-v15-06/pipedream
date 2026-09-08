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
below shows a practical starting point; the 16-step pilot budget is only a
smoke setting and must not be used as the final claim.

```bash
uv run --extra rl pipedream-train \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --schema "$SCHEMA" \
  --normalization-stats results/raw/normalization_train.json \
  --split train \
  --output-dir results/raw/ppo_main \
  --seeds 0 1 2 3 4 \
  --total-timesteps 100000 \
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

## Select A Validation Checkpoint

Selection reads only validation records. Give it all candidate seed models:

```bash
uv run --extra rl pipedream-select-checkpoint \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --schema "$SCHEMA" \
  --normalization-stats results/raw/normalization_train.json \
  --models results/raw/ppo_main/seed_0/ppo_model.zip \
           results/raw/ppo_main/seed_1/ppo_model.zip \
           results/raw/ppo_main/seed_2/ppo_model.zip \
           results/raw/ppo_main/seed_3/ppo_model.zip \
           results/raw/ppo_main/seed_4/ppo_model.zip \
  --output results/raw/ppo_main/selection.json
```

The selection artifact records the validation split and chosen model. Keep it
with the run; do not hand-edit it.

## Evaluate Once On Test

Read the selected model from the selection artifact and evaluate it on the
locked test split:

```bash
MODEL=$(python -c \
  'import json; print(json.load(open("results/raw/ppo_main/selection.json"))["selected_model"])')

uv run --extra rl pipedream-evaluate \
  --manifest "$MANIFEST" \
  --catalog "$CATALOG" \
  --schema "$SCHEMA" \
  --normalization-stats results/raw/normalization_train.json \
  --model "$MODEL" \
  --split test \
  --output results/raw/ppo_main/test.jsonl \
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
