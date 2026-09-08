# Pipedream TODO

This file is the implementation board. A chunk is complete only when its gate passes and its artifacts are replayable.

Status: [ ] planned, [~] in progress, [x] complete, [-] deferred

## Chunk 0: Project contract and toolchain

- [x] Define the research question and contribution boundary.
- [x] Choose the initial toolchain and dependency strategy.
- [x] Define the environment, reward, action, and observation contracts.
- [x] Create the repository implementation layout.
- [x] Add the pinned development container.
- [x] Add the LLVM smoke test.
- [x] Generate and include uv.lock.
- [x] Build the container successfully.
- [x] Run the smoke test inside the container.
- [x] Record exact tool versions, package versions, and image digest.

Recorded environment: LLVM/Clang 20.1.8, Python 3.11.16, uv 0.12.10,
Gymnasium 1.3.0, NumPy 2.4.6, and image manifest
sha256:26d04488a869b071ebe7b950037b78c9fbcf5317c4a247d6bb7a5c6392e8a2a6.

Gate: a clean environment can compile a smoke program, apply an explicit pass sequence, verify the output, and round-trip the bitcode.

## Chunk 1: Deterministic pass engine

- [x] Define the versioned pass catalog in YAML.
- [x] Implement compiler command execution with timeouts.
- [x] Implement verified state snapshots and rollback.
- [x] Implement non-debug instruction counting.
- [x] Implement JSONL step traces.
- [x] Implement content-addressed pass-result caching.
- [~] Add unit tests for success, no-op, failure, timeout, and rollback.
- [x] Add a sequence replay CLI.

Current slice: `src/pipedream/compiler/catalog.py`,
`src/pipedream/compiler/engine.py`, `src/pipedream/cli/sequence.py`,
and `configs/pass_catalog.yaml`.

Gate: the same program, sequence, toolchain, and seed produce a replayable trace and equivalent metrics.

## Chunk 2: Benchmark manifest

- [ ] Acquire a small legal AnghaBench subset.
- [x] Define source-family and near-duplicate grouping.
- [x] Create the smoke manifest with checksums.
- [x] Create the development pilot manifest with train/validation/test splits.
- [x] Record compiler flags, target triple, and initial IR checksums.
- [x] Add manifest validation tests.

Smoke tier complete: 12 repository-owned C fixtures are recorded in
`benchmarks/manifest.json`. AnghaBench acquisition and the family-aware pilot
expansion remain open; `benchmarks/pilot_manifest.json` is a smoke-derived
protocol fixture only.

Gate: the manifest is immutable and no known family or duplicate crosses a split boundary.

## Chunk 3: Gymnasium environment

- [x] Implement PipedreamEnv.
- [x] Implement deterministic reset(seed=...).
- [x] Implement the fixed 12-action episode budget.
- [x] Implement the discrete pass action space.
- [x] Implement the initial normalized instruction-count reward.
- [x] Implement scalar budget and metric observation features.
- [x] Implement failure handling and episode termination.
- [x] Run Gymnasium check_env.
- [x] Add the one-episode integration test.

Gate: a random episode runs from source/IR input through twelve actions or a documented terminal failure.

## Chunk 4: Baselines

- [x] Implement -O2, -O3, and -Oz evaluation.
- [x] Implement multi-seed random sequences.
- [x] Implement one-step greedy search.
- [x] Implement a budgeted shallow beam-search baseline.
- [x] Use one common result schema for every method.
- [x] Report LLVM time, representation time, and total optimization cost.

Gate: all baselines start from the same canonical IR and produce replayable results.

## Chunk 5: Project-owned Autophase

- [x] Define and document the 56-feature schema.
- [x] Implement the LLVM analysis helper.
- [x] Create feature fixture IR files.
- [x] Validate feature ordering and dimensions.
- [x] Fit normalization statistics on a selected split only.
- [x] Add representation checksum metadata.

Gate: every fixture produces a deterministic 56-dimensional vector with documented semantics.

## Chunk 6: PPO with Autophase

- [x] Add PPO configuration and seed handling.
- [x] Train on the smoke tier.
- [x] Train on the development pilot tier.
- [~] Log rollout, policy, value, entropy, action, and compiler metrics.
- [x] Select checkpoints using validation only.
- [x] Add model loading and deterministic evaluation.

Gate: PPO trains end to end and produces a result different from random under the declared protocol.

## Chunk 7: Held-out evaluation

- [x] Lock the development pilot manifest and training configuration.
- [x] Run five development pilot training seeds.
- [x] Evaluate once on the development pilot test split.
- [x] Verify the evaluation CLI on the smoke split.
- [x] Produce per-program raw artifacts.
- [ ] Produce the first comparison table and figures.
- [x] Produce a smoke comparison table.
- [x] Record failures instead of dropping them silently.

Gate: no test-derived tuning remains and every result carries complete metadata.

## Chunk 8: IR2Vec representation study

- [~] Choose symbolic and/or flow-aware IR2Vec mode.
- [ ] Freeze or train the vocabulary using training IR only.
- [x] Implement function-level mean pooling.
- [ ] Fit training-only normalization statistics.
- [ ] Run Autophase, IR2Vec, and hybrid PPO experiments.
- [ ] Match policy capacity and training budget.
- [ ] Include representation computation cost.

Adapter and configuration are present, but the empirical study stays blocked
until an exact IR2Vec executable and vocabulary are supplied.

Gate: the ablation answers whether IR2Vec changes quality, generalization, or cost.

## Chunk 9: Research analysis

- [x] Add paired bootstrap confidence intervals.
- [~] Add effect sizes and predeclared paired tests.
- [ ] Analyze pass transitions and repeated motifs.
- [ ] Analyze program-adaptive behavior.
- [ ] Compare seed disagreement.
- [ ] Write limitations and threats to validity.
- [x] Add the smoke/pilot reproduction script.

Gate: every final claim is supported by held-out data and a replayable command.

## Chunk 10: Optional extensions

- [-] Runtime as a primary reward.
- [-] Multi-objective and Pareto optimization.
- [-] Adaptive stop action.
- [-] Action masking.
- [-] Public replay/demo UI.
- [-] Larger benchmark collection.

These remain out of the critical path until Chunk 9 is complete.

## Immediate next commands

~~~bash
docker build -t pipedream:llvm20 .
docker run --rm -v "$PWD":/workspace pipedream:llvm20
~~~
