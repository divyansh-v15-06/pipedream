# Remaining Work

This is the prioritized work after the current smoke and development-pilot
vertical slice. The pilot proves the protocol wiring; it does not close the
research question.

## P0: Main Benchmark And Final Experiment

- Acquire a legally usable AnghaBench subset and record its provenance,
  license, source checksum, family group, compiler flags, target triple, and
  initial IR checksum.
- Freeze a family-aware main manifest. No near-duplicate family may cross
  train, validation, and test.
- Predeclare the main PPO budget, episode length, action catalog, reward,
  representation, seed count, checkpoint rule, and stopping rule.
- Fit normalization on training IR only.
- Train the planned seed set, select on validation only, and evaluate once on
  unseen test programs.
- Produce the first complete comparison table and figures against `-O2`,
  `-O3`, `-Oz`, random, greedy, and beam search.

## P1: IR2Vec Ablation

- Freeze the exact IR2Vec executable, mode, vocabulary, and dimensionality.
- Decide whether the vocabulary is supplied or trained from training IR only.
- Implement and verify the selected representation path, including
  training-only normalization and representation timing.
- Match policy capacity, training budget, action space, and evaluation splits
  across Autophase, IR2Vec, and any hybrid condition.
- Report quality, generalization, and representation cost, including negative
  results.

## P1: Research Analysis

- Add effect sizes and predeclared paired significance tests to the bootstrap
  analysis.
- Analyze pass transitions, repeated motifs, and program-adaptive behavior.
- Quantify seed disagreement and failure rates.
- Write limitations and threats to validity, especially benchmark scope,
  LLVM-version dependence, search-budget dependence, and reward choice.
- Make every plot regenerate from checked-in raw artifacts and metadata.

## P2: Engineering Hardening

- Complete focused compiler-engine tests for no-op, failure, timeout, and
  rollback behavior.
- Standardize the aggregate result schema and add a Parquet writer for larger
  experiment tables.
- Add more feature fixtures and cross-version diagnostics if the LLVM helper
  is extended.
- Add a clean-room reproduction job that starts from the pinned container and
  verifies the complete smoke/pilot pipeline.

## Deferred Until The Core Result

Runtime reward as a primary objective, multi-objective optimization, adaptive
stopping, action masking, large action spaces, a public UI, and per-program
policies are explicitly outside the critical path.

## Definition Of Done

The core project is complete when a clean environment can replay the frozen
manifest, train from training programs, select without test access, evaluate
on unseen test programs, compare all declared baselines, and regenerate the
reported tables and figures from recorded raw artifacts.
