# Pipedream

## Program-Adaptive Reinforcement Learning for LLVM Pass Ordering

Pipedream is a reproducible research system that learns which LLVM optimization pass to apply next for a program.

The core loop is:

~~~text
C/C++ program -> LLVM IR -> state representation -> PPO policy
             -> one LLVM pass -> new LLVM IR -> reward and repeat
~~~

The project is about reinforcement learning for compiler phase/pass ordering. It is not an LLM wrapper, a chatbot, or a hand-written program classifier.

This README is the current project source of truth. There is intentionally no separate MASTERPLAN.md until the implementation becomes large enough to justify splitting the documentation.

## Research Question

The primary question is:

> Can one PPO policy trained across general-purpose LLVM programs learn program-adaptive pass sequences that improve held-out optimization results over LLVM pipelines and simple search baselines?

The representation question is:

> Does IR2Vec improve pass-ordering quality, generalization, or cost compared with the 56-dimensional Autophase representation?

The policy must be trained on the training programs and evaluated on completely unseen test programs. Test programs are never used for training, vocabulary fitting, hyperparameter tuning, or early stopping.

The project does not assume that IR2Vec will beat Autophase. A clear negative result is valuable if the comparison is controlled and reproducible.

## Current Implementation Status

The repository has completed the toolchain baseline and deterministic pass engine. The current implementation includes:

- a Python 3.11 and LLVM 20 development container;
- a generated and pinned `uv.lock` plus a versioned 12-action pass catalog;
- transactional LLVM pass execution with verification, timeout handling, rollback, and content-addressed caching;
- non-debug instruction counting and JSONL step-trace records;
- a sequence replay CLI that writes final bitcode, a JSONL trace, and a summary artifact; and
- a container smoke test that compiles, optimizes, verifies, disassembles, and reassembles LLVM bitcode.

The current vertical slice also includes a 12-program checksum manifest, a deterministic Gymnasium environment, LLVM pipeline/random/greedy/beam baselines, a project-owned 56-feature representation, training-only normalization, PPO smoke and development-pilot training, deterministic held-out evaluation, and paired analysis utilities. Research-scale AnghaBench acquisition, main-scale training, final figures, and the empirical IR2Vec comparison remain open. IR2Vec is intentionally validated in the later representation-study milestone rather than treated as a M0 dependency.

A local, read-only training dashboard is also available in `monitoring/`. It follows a named Docker training container, checkpoint files, TensorBoard scalars, host resource use, and NVIDIA telemetry without being able to start, stop, or alter a run.

### Do We Need To Train PPO?

Yes. The short smoke model only verifies that PPO, LLVM, observations, rewards, checkpointing, and evaluation connect correctly. It is not a research result. A defensible result requires training on the training split with multiple seeds, selecting a checkpoint using validation programs only, and evaluating that locked checkpoint once on unseen test programs. The current pilot commands exercise this protocol with a tiny timestep budget; final training must use a substantially larger predeclared budget.

## Contribution Boundary

Prior work has already applied deep reinforcement learning to LLVM pass ordering in the HLS domain. CompilerGym has also provided LLVM optimization environments, Autophase observations, AnghaBench benchmarks, and instruction-count/code-size rewards.

Pipedream therefore does not claim that PPO, LLVM, Autophase, or pass ordering are novel by themselves. The intended contribution is a controlled, project-owned study of:

1. One shared policy generalizing across previously unseen programs.
2. Autophase versus IR2Vec versus their hybrid under the same action space and training budget.
3. Optimization quality versus search and representation cost.
4. Interpretable, replayable pass sequences rather than only aggregate scores.

Relevant references:

- [AutoPhase paper](https://experts.illinois.edu/en/publications/autophase-compiler-phase-ordering-for-hls-with-deep-reinforcement/)
- [CompilerGym LLVM environment reference](https://compilergym.com/llvm/index.html)
- [LLVM MLGO and IR2Vec documentation](https://www.llvm.org/docs/MLGO.html)
- [LLVM 20.1.0 release notes](https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html)

## Locked Initial Decisions

These choices are deliberately conservative. Changing one after experiments begin creates a new experiment version and must be recorded.

| Area | Initial decision |
|---|---|
| Host environment | Ubuntu 24.04 container or an equivalent Linux host |
| LLVM toolchain | LLVM/Clang/opt 20.x from apt.llvm.org, with exact package versions recorded |
| Python | Python 3.11 |
| Python environment | uv, pyproject.toml, and committed uv.lock |
| RL environment API | Gymnasium Env with reset/step/terminated/truncated |
| RL algorithm | Stable-Baselines3 PPO |
| Neural framework | PyTorch |
| Initial representation | Project-owned Autophase-compatible 56-feature extractor |
| Learned representation | Upstream LLVM IR2Vec, frozen vocabulary during PPO training |
| Initial benchmark | Small AnghaBench subset with immutable manifest |
| Primary objective | Reduction in non-debug LLVM IR instruction count |
| Initial episode budget | 12 pass applications |
| Initial action catalog | 12 stable, versioned LLVM optimization passes |
| Core evaluation | -O2, -O3, -Oz, random, greedy, and PPO |
| Trace format | JSONL for step traces and Parquet for aggregated results |
| Test framework | pytest plus Gymnasium environment checks |
| Formatting and linting | ruff; mypy where it helps |
| Build tools | CMake and Ninja for LLVM helper/plugin code |
| Experiment tracking | Local TensorBoard logs plus immutable JSON metadata |

CompilerGym may be used to compare reference feature values during development, but it is not a runtime dependency of the final system. Its historical LLVM environment and API must not define the project contract.

## Scope and Success Criteria

### Core result

The first complete research result is:

~~~text
one shared PPO policy
    trained on training programs only
    with a fixed pass catalog and fixed episode budget
    evaluated on unseen test programs
    compared against LLVM pipelines and simple search
~~~

Minimum success means:

- LLVM pass execution is deterministic enough to replay.
- The environment passes Gymnasium checks.
- Random episodes run end to end.
- PPO learns behavior different from random selection.
- Evaluation can be repeated from a recorded configuration and seed.

Strong success means PPO improves over random and is useful on a meaningful held-out subset.

Excellent success means the controlled representation study shows a statistically supported quality, generalization, or cost difference between Autophase and IR2Vec. A result showing that Autophase is cheaper and equally effective is also a successful outcome.

### Explicitly deferred work

The following are extensions, not prerequisites for the core result:

- Runtime optimization as a primary objective.
- Multi-objective or Pareto optimization.
- Large action spaces.
- GNNs, transformers, or LLMs.
- A web UI.
- Large-scale benchmark collection.
- Training separate policies per program.

## System Architecture

~~~text
source program
    |
    v
clang -O0 -Xclang -disable-O0-optnone
    |
    v
canonical starting LLVM IR
    |
    v
project-owned environment
    |-- Autophase / IR2Vec observation
    |-- PPO chooses one pass
    |-- opt applies the pass
    |-- verifier and metric collector
    |-- trace and cache
    |
    v
next LLVM IR and reward
~~~

The initial IR must not contain Clang's O0 optnone attributes, because those attributes can prevent later optimization passes from acting. The exact compile command, target triple, data layout, language standard, and flags are recorded in the benchmark manifest.

## Environment Contract

The environment is a project-owned Gymnasium environment, implemented as a normal Python package around a small LLVM helper layer.

### Episode state

Each episode owns:

- one benchmark program;
- the current verified LLVM module;
- the immutable initial instruction count;
- the current instruction count;
- the current step index;
- the versioned pass catalog;
- the trace and timing data;
- a cache key based on IR bytes, catalog version, and toolchain version.

### Action space

The initial catalog contains these pass names, subject to validation against the pinned LLVM 20.x package set:

~~~text
mem2reg
sroa
instcombine
simplifycfg
early-cse
gvn
dce
adce
reassociate
loop-simplify
licm
indvars
~~~

Each pass has a stable integer action ID, exact new-pass-manager syntax, category, expected prerequisites, and LLVM version. The catalog is part of every result's metadata.

Repeated passes are allowed initially. No-op actions are logged and receive zero improvement reward. A failed pass never mutates the live state: the environment restores the previous verified IR, records the error, and applies a small deterministic failure penalty.

### Observation space

The policy receives a flat float32 vector:

~~~text
representation features
normalized current instruction count
fraction of the episode budget remaining
~~~

The policy does not receive a benchmark ID. This prevents memorizing a lookup table and makes program adaptation depend on the observed IR.

The three representation variants are:

~~~text
A: normalized Autophase(56) + scalar state features
B: normalized pooled IR2Vec + scalar state features
C: normalized Autophase(56) + normalized pooled IR2Vec + scalar state features
~~~

The PPO network family and training budget remain fixed across A, B, and C. Hidden-layer width is chosen to keep parameter counts approximately comparable, and all representation dimensions are recorded.

### Reward

The primary reward is normalized local instruction-count improvement:

~~~text
reward_t = (instructions_before - instructions_after)
           / max(initial_instructions, 1)
~~~

The environment also logs cumulative reduction, final reduction, no-op count, failure count, and every metric before and after the action. Runtime and binary code size are measured in later phases and are not silently mixed into the initial reward.

### Termination

- The episode is truncated after 12 actions.
- An unrecoverable compiler or verification error terminates the episode.
- Early stopping is not used in the first PPO comparison.
- An optional stop action may be studied only after fixed-budget results are stable.

### Safety and correctness

Every transformed module is checked with LLVM verification before becoming the new state. Every subprocess has a timeout and isolated temporary directory. Benchmark binaries are not executed in the core experiment. Runtime evaluation, when added, must run inside a resource-limited sandbox with fixed inputs.

## Representation Design

### Autophase

Autophase is implemented by a project-owned LLVM analysis helper that emits exactly 56 documented features. The helper is tested against reference values on a small set of IR fixtures. CompilerGym is allowed as a development oracle only.

The feature schema, ordering, units, LLVM version, and normalization statistics are versioned. Normalization statistics are fitted on the training split only and then frozen.

### IR2Vec

IR2Vec is generated with the LLVM toolchain rather than a separately maintained research fork. The initial variant is:

- frozen vocabulary;
- flow-aware and symbolic variants tested separately if available;
- function-level embeddings;
- mean pooling over defined functions;
- zero vector for a module with no defined functions;
- normalization statistics fitted on training IR only.

The vocabulary file, embedding kind, dimension, command-line options, and checksum are recorded. If the vocabulary is trained rather than taken from a fixed upstream artifact, only training-split IR may be used.

## Benchmark and Data Protocol

The benchmark manifest is immutable and contains:

~~~text
benchmark_id
source checksum
source family/group ID
split
language and language standard
compiler flags
target triple
initial IR checksum
LLVM/toolchain version
~~~

Dataset tiers:

- Smoke: 12 programs for toolchain and integration tests.
- Pilot: approximately 60 programs for environment and PPO debugging.
- Main: at least 240 programs when the pipeline is stable.

The main split targets 70% train, 15% validation, and 15% test, but family-aware grouping takes priority over the ratio. Near-duplicate sources and generated variants must not cross the split boundary. The final manifest is locked before PPO hyperparameter tuning.

The policy is trained once across the training programs. Validation programs are used for model selection and hyperparameters. Test programs are used exactly once for final reporting.

## Baselines

Every method starts from the same canonical initial IR and is evaluated on the same programs.

Required baselines:

1. LLVM -O2.
2. LLVM -O3.
3. LLVM -Oz.
4. Uniform random pass sequences with multiple seeds.
5. One-step greedy search over the same pass catalog and action budget.

Recommended search baseline:

- Shallow beam search with a fixed expansion budget, included only when its LLVM execution cost is reported.

The comparison record includes final and initial instruction count, reduction, code size when enabled, runtime when enabled, sequence length, pass sequence, LLVM time, representation time, environment step time, total optimization time, seed, and configuration checksum.

## Training Protocol

PPO is the first and only RL algorithm in the core experiment.

Training stages:

1. Verify PPO on the smoke tier.
2. Tune a small number of hyperparameters on the pilot tier using validation only.
3. Lock the configuration.
4. Train on the main training split with at least five random seeds.
5. Select the final checkpoint using validation results.
6. Evaluate once on the held-out test split.

Record:

~~~text
episode reward
final instruction reduction
episode length
policy entropy
policy loss
value loss
action distribution
LLVM execution time
representation time
seed and configuration
~~~

Because LLVM subprocess execution is expensive, the environment uses content-addressed caching for verified IR states and observations. Cache hits and misses are reported rather than hidden.

## Evaluation and Statistics

The primary test-set report includes mean, median, standard deviation, median improvement, percentage of programs improved, percentage worse than -O3, and optimization cost.

Methods are compared per program using paired differences. The report includes bootstrap confidence intervals, effect sizes, and a suitable paired statistical test selected before looking at final results. Multiple-comparison handling is documented when more than one primary claim is tested.

The final result table is:

| Method | Test IR count | Change vs -O3 | Code size | Runtime | Total optimization cost |
|---|---:|---:|---:|---:|---:|
| -O2 | | | | | |
| -O3 | | | | | |
| -Oz | | | | | |
| Random | | | | | |
| Greedy | | | | | |
| PPO + Autophase | | | | | |
| PPO + IR2Vec | | | | | |
| PPO + Hybrid | | | | | |

No result is reported from a cherry-picked benchmark. Per-program distributions and failures remain available in raw artifacts.

## Reproducibility Contract

Each experiment records:

~~~text
git commit
LLVM and Clang versions
Python and dependency lock versions
OS/container image digest
CPU/GPU and thread settings
benchmark manifest checksum
pass catalog version
representation and vocabulary checksum
normalization statistics checksum
PPO hyperparameters
reward version
random seeds
training steps
evaluation command
~~~

The repository must provide:

- a smoke-test command;
- a sequence replay command;
- a baseline evaluation command;
- a PPO training command;
- a final evaluation command;
- a reproduction script for the reported experiments.

## Training Workstation Profile

The current development machine is an AMD Ryzen 7 7435HS (8 cores / 16
threads), 24 GiB RAM, and an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GiB
VRAM. This is enough for the current PPO experiments, but LLVM subprocesses
and feature extraction are the practical bottleneck—not the small MLP policy.

The recommended default is one seed at a time with `--device cpu`; it leaves
memory headroom for LLVM and avoids GPU transfer overhead for this small model.
Use `--device cuda` only as a recorded comparison after confirming CUDA is
available in the container. Do not launch all five seeds concurrently on this
machine: run them sequentially and preserve the complete artifact directory
for every seed. See [train.md](train.md) for hardware-aware commands and the
monitoring workflow.

## Repository Structure

The implementation will use a src layout:

~~~text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .python-version
├── Dockerfile
├── .dockerignore
├── .gitignore
│
├── configs/
│   ├── smoke.yaml
│   ├── pass_catalog.yaml
│   ├── autophase_schema.yaml
│   └── ir2vec.yaml
│
├── src/pipedream/
│   ├── cli/
│   ├── compiler/
│   ├── env/
│   ├── representations/
│   ├── agents/
│   ├── benchmarks/
│   ├── evaluation/
│   └── analysis/
│
├── benchmarks/
│   ├── manifest.json
│   ├── pilot_manifest.json
│   └── smoke/
│
├── tests/
│   ├── fixtures/
│   ├── test_compiler.py
│   ├── test_env.py
│   ├── test_pass_catalog.py
│   ├── test_representations.py
│   ├── test_rewards.py
│   └── test_reproducibility.py
│
├── scripts/
│   ├── smoke_test.sh
│   └── reproduce_smoke.sh
│
├── monitoring/
│   ├── server.py
│   └── README.md
│
├── results/
│   ├── raw/
│   ├── processed/
│   ├── figures/
│   └── tables/
│
├── context.md
├── train.md
├── remaining_work.md
└── TODO.md
~~~

Do not create every directory before it is needed. Each phase must leave a working artifact and tests.

## Milestones and Gates

### M0: Toolchain and container

Install and verify the pinned LLVM 20.x package set, Clang, opt, llvm-dis, and llvm-as. Compile a smoke program, generate canonical IR, run explicit new-pass-manager pipelines, and verify the output. IR2Vec is validated separately in M7 when its representation settings are frozen.

Gate: the smoke script succeeds on a clean environment and records tool versions.

### M1: Deterministic pass engine

Implement single-pass execution, rollback, verification, metrics, timeouts, cache keys, and JSONL traces.

Gate: the same program, sequence, toolchain, and seed reproduce the same trace and metrics within documented tolerances.

### M2: Benchmark manifest

Prepare the smoke and pilot tiers, compute checksums, assign family-aware splits, and lock the manifest format.

Gate: no source family or near-duplicate crosses the split boundary.

### M3: Gymnasium environment

Implement reset, step, observation space, action space, reward, truncation, failure handling, and deterministic seeding.

Gate: Gymnasium environment checks pass and one random episode completes end to end.

### M4: Baselines

Implement LLVM pipelines, random sequences, greedy search, and their common evaluation record.

Gate: all baselines run on the pilot tier from the same initial IR and produce replayable artifacts.

### M5: PPO + Autophase

Implement the 56-feature helper, normalization, PPO configuration, and local TensorBoard logging.

Gate: PPO trains on the pilot tier and beats or meaningfully differs from random under a predeclared metric.

### M6: Held-out evaluation

Lock the training protocol, run multiple seeds, evaluate on the test split, and produce the first research table.

Gate: no test-derived tuning remains and all artifacts contain metadata.

### M7: IR2Vec ablation

Add frozen IR2Vec embeddings, then compare Autophase, IR2Vec, and hybrid representations under matched protocols.

Gate: vocabulary, embedding settings, model parameter counts, and representation costs are recorded.

### M8: Analysis and report

Add sequence motifs, program-adaptive behavior, paired statistics, limitations, figures, and reproduction scripts.

Gate: every final claim is supported by held-out data and a replayable command.

### M9: Optional extensions

Only after M8, consider multi-objective rewards, runtime measurement, adaptive stopping, beam search, action masking, or a public demo.

## Initial Commands

The first implementation should expose commands with stable names:

~~~bash
uv run pipedream-smoke
uv run pipedream-sequence --program path/to/program.bc --passes mem2reg instcombine gvn dce
uv run pipedream-baselines --config configs/baseline.yaml
uv run pipedream-train --config configs/ppo_autophase.yaml
uv run pipedream-evaluate --config configs/ppo_autophase.yaml --split test
~~~

The exact entry-point implementation may change, but command behavior and output schemas must be documented and tested.

## Out-of-the-Box Ideas That Preserve the Aim

These ideas improve the research system without changing its central question:

1. Content-addressed pass-result caching can make greedy search and PPO data collection share work.
2. A replay viewer can show IR metrics and pass decisions step by step, making failures debuggable.
3. A budget-aware observation lets the same policy learn whether it is early or late in an episode.
4. A shallow beam-search baseline can expose whether PPO finds genuinely non-myopic sequences.
5. Pass-transition and state-transition graphs can reveal whether different representations lead to different optimization strategies.
6. A later stop action can test whether the policy learns when further optimization is harmful.

These must remain secondary to the fixed-budget, single-objective comparison.

## Non-Negotiable Rules

1. Do not tune on the final test set.
2. Do not silently change the benchmark split, reward, pass catalog, toolchain, or vocabulary.
3. Do not claim that PPO or IR2Vec is novel without a literature review.
4. Do not report only the best benchmark.
5. Do not hide representation, LLVM, or search cost.
6. Do not treat valid IR as proof of application-level semantic correctness.
7. Do not expand the pass catalog until the current catalog is stable.
8. Do not add a UI before the research pipeline is reliable.
9. Preserve failed experiments when they explain limitations.
10. Prefer a clear negative result over a fragile or cherry-picked win.

## Expected Final Narrative

LLVM optimization quality depends on pass ordering, while fixed pipelines do not explicitly adapt their decisions to each program. Pipedream models pass ordering as a sequential decision problem and learns a shared policy over LLVM IR states.

The final report should answer:

1. Does PPO learn useful sequences beyond random selection?
2. Does it generalize to unseen programs?
3. How does it compare with LLVM pipelines and greedy search?
4. Does IR2Vec improve over Autophase?
5. What quality is gained per unit of optimization cost?
6. Which pass sequences and state transitions explain the result?

The final claim should be narrow and evidence-based:

> Pipedream evaluates whether program representations improve reinforcement-learning-based LLVM pass ordering under a reproducible held-out benchmark protocol.

The project succeeds by producing a defensible result, not by claiming to beat LLVM everywhere.
