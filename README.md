# Master Plan — Reinforcement Learning for LLVM Compiler Phase-Sequence Optimization

## 0. Project North Star

### Project title

**Reinforcement Learning for LLVM Compiler Phase-Sequence Optimization**

### CV-facing subtitle

**Program-Adaptive Reinforcement Learning for LLVM Optimization Pass Scheduling**

### Core objective

Build a reproducible research system that learns, on a per-program basis, which LLVM optimization pass should be applied next.

The central loop is:

`Program → LLVM IR → IR representation → RL policy → LLVM pass → new LLVM IR → repeat`

The project should not become an LLM wrapper or a hand-written program-type classifier. The main research problem remains **reinforcement learning for LLVM compiler phase/pass ordering**.

The AI/representation extension is used to make the RL agent understand the current LLVM IR better:

`Autophase → PPO`
`IR2Vec → PPO`
`Autophase + IR2Vec → PPO`

The goal is to determine whether learned LLVM-IR representations improve phase ordering compared with handcrafted compiler features.

---

# 1. Research Question

## Primary research question

> Can a reinforcement-learning policy learn program-specific LLVM optimization pass sequences that improve compiler optimization outcomes over fixed LLVM pipelines and simple search baselines, while generalizing to previously unseen programs?

## Secondary representation question

> Does adding a learned LLVM-IR representation such as IR2Vec improve RL-based pass ordering over the 56-dimensional Autophase representation?

## Optional extension question

> Can a multi-objective reward improve code-size/performance trade-offs without sacrificing generalization or making optimization prohibitively expensive?

Do not assume the learned representation will outperform Autophase. A negative result is still a useful experimental finding if the comparison is controlled and reproducible.

---

# 2. Non-Negotiable Project Principles

1. **The main project is RL-based LLVM phase ordering.**
2. **Do not add an LLM merely to label programs.**
3. Use learned representations as model inputs rather than hard-coded pass-selection rules.
4. Separate training, validation, and test programs.
5. Never tune the final model using the held-out test set.
6. Always compare against strong non-RL baselines.
7. Record every experiment with exact configuration and random seed.
8. Optimize for reproducibility before optimizing for maximum benchmark score.
9. Never claim novelty merely because PPO, IR2Vec, Autophase, or LLVM are used.
10. Prefer honest empirical conclusions over cherry-picked wins.
11. Keep optimization quality and optimization cost separate in reporting.
12. Preserve a minimal baseline implementation so every advanced version has an interpretable reference.

---

# 3. Current Technical Direction

## 3.1 LLVM IR

LLVM IR is the target representation.

Use LLVM's modern/new pass manager and `opt` for applying optimization pipelines/pass sequences.

Do not depend on undocumented or legacy pass-manager behavior.

## 3.2 Environment

### Preferred final architecture

Build a lightweight project-owned Gymnasium-compatible environment around LLVM.

The environment owns:

- current benchmark/program
- current LLVM IR
- available action/pass set
- pass execution
- state extraction
- reward calculation
- episode termination
- measurements
- trace logging

### CompilerGym

CompilerGym may be used during early prototyping/reference comparison because the original plan already uses its LLVM + Autophase environment.

However, do not make the entire final project dependent on CompilerGym.

CompilerGym's repository is archived/read-only, so the final system should be portable and project-owned.

## 3.3 RL framework

Preferred first algorithm:

**PPO — Proximal Policy Optimization**

Use Stable-Baselines3 initially unless a custom policy/network is required.

For advanced learned representations, write a custom PyTorch policy/network only when necessary.

## 3.4 RL environment API

Use the modern Gymnasium interface:

- `reset()`
- `step(action)`
- `observation_space`
- `action_space`
- `terminated`
- `truncated`
- `info`

Use a fixed maximum episode length initially.

---

# 4. High-Level System Architecture

```text
                    ┌──────────────────────┐
                    │   C / C++ Program    │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │       LLVM IR        │
                    └──────────┬───────────┘
                               ↓
              ┌─────────────────────────────────┐
              │      State Representation       │
              │                                 │
              │  Autophase                      │
              │  IR2Vec                         │
              │  CFG / IR statistics (optional) │
              └───────────────┬─────────────────┘
                              ↓
                     ┌─────────────────┐
                     │    PPO Policy   │
                     └────────┬────────┘
                              ↓
                    choose optimization pass
                              ↓
                     ┌─────────────────┐
                     │   LLVM `opt`    │
                     │  New Pass Mgmt  │
                     └────────┬────────┘
                              ↓
                         New LLVM IR
                              ↓
                     reward + next state
                              │
                              └────────→ repeat
```

---

# 5. Repository Structure

The repository should eventually look approximately like:

```text
llvm-rl-phase-ordering/
│
├── README.md
├── MASTERPLAN.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── baseline.yaml
│   ├── ppo_autophase.yaml
│   ├── ppo_ir2vec.yaml
│   ├── ppo_hybrid.yaml
│   └── multi_objective.yaml
│
├── env/
│   ├── __init__.py
│   ├── llvm_env.py
│   ├── pass_catalog.py
│   ├── observations.py
│   ├── rewards.py
│   ├── termination.py
│   └── wrappers.py
│
├── representation/
│   ├── __init__.py
│   ├── autophase.py
│   ├── ir2vec.py
│   ├── hybrid.py
│   └── normalization.py
│
├── agents/
│   ├── __init__.py
│   ├── random_agent.py
│   ├── greedy_agent.py
│   ├── ppo_baseline.py
│   ├── ppo_autophase.py
│   ├── ppo_ir2vec.py
│   └── ppo_hybrid.py
│
├── benchmarks/
│   ├── download_or_prepare.py
│   ├── split.py
│   ├── manifest.json
│   └── README.md
│
├── evaluation/
│   ├── evaluate.py
│   ├── compare_baselines.py
│   ├── aggregate.py
│   ├── significance.py
│   └── metrics.py
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_ppo.py
│   ├── run_ablation.py
│   └── run_generalization.py
│
├── instrumentation/
│   ├── trace.py
│   ├── llvm_metrics.py
│   ├── timing.py
│   └── episode_logger.py
│
├── visualization/
│   ├── training_curves.py
│   ├── benchmark_comparison.py
│   ├── pass_sequences.py
│   ├── representation_projection.py
│   └── reward_curves.py
│
├── tests/
│   ├── test_env.py
│   ├── test_pass_catalog.py
│   ├── test_rewards.py
│   ├── test_representations.py
│   └── test_reproducibility.py
│
├── scripts/
│   ├── smoke_test.sh
│   ├── benchmark_all.sh
│   └── reproduce_results.sh
│
├── results/
│   ├── raw/
│   ├── processed/
│   ├── figures/
│   └── tables/
│
└── docs/
    ├── methodology.md
    ├── experiments.md
    └── limitations.md
```

Do not create all files at the beginning. Build the project phase-by-phase.

---

# 6. Required Tooling

## Core languages

- Python for environment, RL, experiments, metrics, visualization.
- C/C++ only when direct LLVM integration/custom LLVM code genuinely requires it.

## Compiler

- LLVM/Clang
- `opt`
- LLVM new pass manager
- LLVM analysis/optimization infrastructure
- LLVM IR (`.ll` / `.bc`)

## RL / ML

- Python 3.x
- PyTorch
- Stable-Baselines3 PPO for the first implementation
- Gymnasium
- NumPy
- SciPy where statistical tests are needed

## Representation

- Autophase
- IR2Vec
- Optional CFG/IR structural features

## Data / experiments

- AnghaBench subset as the initial benchmark family
- Deterministic train/validation/test manifest
- JSON/CSV/Parquet as appropriate for traces
- YAML configs

## Tracking / visualization

Preferred:

- TensorBoard
- Matplotlib
- pandas

Optional:

- Weights & Biases, only if it improves experiment tracking and does not make the project dependent on a cloud service.

## Testing / engineering

- pytest
- ruff
- mypy where practical
- pre-commit (optional)
- GitHub Actions for CI if the repository is public

---

# 7. Phase 0 — Environment and Toolchain Verification

## Goal

Make sure the machine can compile, optimize, represent, measure, and reproduce LLVM programs before any RL work begins.

## Tasks

1. Install a supported LLVM version.
2. Verify:
   - `clang`
   - `opt`
   - LLVM IR generation
   - new pass manager
3. Compile a tiny C benchmark to LLVM IR.
4. Run:
   - `-O0`
   - `-O2`
   - `-O3`
   - `-Oz`
5. Run an explicit pass sequence with `opt`.
6. Confirm optimized IR is valid.
7. Record LLVM version/build information.

## Acceptance criteria

```text
clang --version
opt --version
```

both work.

A C source can be converted to valid LLVM IR and transformed using explicit new-PM pass pipelines.

Create:

```text
scripts/smoke_test.sh
```

which verifies the complete compiler toolchain.

---

# 8. Phase 1 — Build the Pass-Ordering Engine

## Goal

Create the deterministic engine that the RL agent will later control.

The engine must accept:

```text
program + sequence of passes
```

and return:

```text
final IR
instruction count
code size
runtime if available
compile/optimization time
full pass trace
```

## Example

```text
input.bc

sequence:
  mem2reg
  instcombine
  gvn
  dce

output:
  final.bc
  final.ll
  metrics.json
  trace.json
```

## Requirements

- Each pass execution is independently logged.
- Invalid/failed transformations do not silently corrupt an episode.
- Every step has a timestamp/duration.
- State before and after every pass can be reconstructed.
- Pass names map to stable integer action IDs.

## Deliverable

A command such as:

```bash
python -m experiments.run_sequence \
    --program benchmark.bc \
    --passes mem2reg instcombine gvn dce
```

---

# 9. Phase 2 — Benchmark Infrastructure

## Initial dataset

Use an **AnghaBench subset** as the initial benchmark family, consistent with the original plan.

Do not start with hundreds/thousands of programs.

First use a small smoke-test subset.

Then scale.

## Dataset manifest

Create one immutable manifest containing:

```text
benchmark_id
source/program path
split
checksum
compiler settings
target triple
```

## Split

Recommended initial structure:

```text
70% train
15% validation
15% test
```

The exact ratio may change if benchmark size makes another split more statistically sensible.

The critical requirement is:

**the final test set remains unseen during training and hyperparameter tuning.**

Where possible, use function/program-family aware splitting to reduce near-duplicate leakage.

---

# 10. Phase 3 — Baseline Environment

## Goal

Implement the Gymnasium environment before training PPO.

Example:

```python
obs, info = env.reset(seed=seed)

for _ in range(max_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

## Observation baseline

Start with:

**Autophase 56-dimensional vector**

Normalize/scale inputs appropriately.

## Action baseline

Start with a **small curated pass set**, not the full LLVM pass universe.

The first pass catalog should contain optimization passes that:

- are stable on the benchmark set
- are individually executable
- make sense for IR-level optimization
- produce measurable effects

Example category groups:

```text
canonicalization
dead-code elimination
scalar simplification
CFG simplification
memory-to-register
global value optimization
loop transformations
```

Do not arbitrarily include every LLVM pass.

Create a versioned pass catalog.

## Reward baseline

Start with instruction-count improvement.

For example:

```text
reward_t = instruction_count_before - instruction_count_after
```

Normalize it when necessary for stable RL training.

Do not immediately use a complicated composite reward.

## Termination

Start with:

```text
max episode length = fixed T
```

Then investigate early stopping later.

Possible later termination conditions:

- max steps
- invalid state
- no useful progress for N steps
- repeated-state/pipeline guard
- explicit budget exhausted

---

# 11. Phase 4 — Non-RL Baselines

This phase is mandatory before making claims about PPO.

Implement:

## Baseline A — Random

Uniformly sample available passes.

Use many random seeds.

## Baseline B — Greedy one-step search

At each state:

```text
for each candidate pass:
    apply pass
    measure immediate improvement

choose the best immediate pass
```

The greedy agent provides a strong simple search baseline.

## Baseline C — LLVM pipelines

Evaluate:

- `-O0`
- `-O2`
- `-O3`
- `-Oz`

Do not confuse `-Oz` with "best possible program"; it is simply one official LLVM optimization target.

## Evaluation record

For each program and baseline store:

```text
final_instruction_count
instruction_reduction
code_size
runtime (if enabled)
optimization_time
sequence length
pass sequence
seed
```

---

# 12. Phase 5 — PPO / RL Baseline

## Goal

Train a PPO agent using Autophase.

Architecture:

```text
Autophase(56)
      ↓
MLP policy/value network
      ↓
discrete pass action
```

## Training stages

### 5.1 Tiny benchmark

Use a very small training set.

Verify learning exists.

### 5.2 Small benchmark

Tune:

- learning rate
- batch size
- rollout length
- entropy coefficient
- discount factor
- GAE parameter
- network size

Do not perform broad hyperparameter searches yet.

### 5.3 Main training set

Train on the locked training split.

Use multiple random seeds.

## Mandatory logging

- episode reward
- mean reward
- best reward
- episode length
- policy entropy
- value loss
- policy loss
- pass-action distribution
- benchmark-level final metrics

---

# 13. Phase 6 — Generalization Evaluation

This is one of the most important CV/research phases.

Run the trained models on the completely held-out test set.

Report:

```text
-O2
-O3
-Oz
Random
Greedy
PPO + Autophase
```

For each metric report at least:

- mean
- median
- standard deviation
- median improvement
- percentage of programs improved
- percentage of programs worse than `-O3`
- optimization cost

Avoid reporting only the best benchmark.

## Recommended result table

| Method | Test Instr. Count | Δ vs O3 | Code Size | Runtime | Opt. Cost |
|---|---:|---:|---:|---:|---:|
| -O2 | | | | | |
| -O3 | | | | | |
| -Oz | | | | | |
| Random | | | | | |
| Greedy | | | | | |
| PPO + Autophase | | | | | |

---

# 14. Phase 7 — Learned Program Representation

Now add the AI representation layer.

## Representation A

Autophase only:

```text
S_A = Autophase(IR)
```

## Representation B

IR2Vec only:

```text
S_B = IR2Vec(IR)
```

## Representation C

Hybrid:

```text
S_C = concat(
    normalized Autophase(IR),
    normalized IR2Vec(IR)
)
```

The policy architecture should remain as similar as possible across A/B/C so the comparison isolates representation quality.

## Critical rule

Do not create a manual program-type classifier such as:

```text
loop-heavy → LICM
memory-heavy → SROA
branch-heavy → CFG simplification
```

That turns the experiment into hand-engineered heuristics.

Instead, let the learned representation feed the policy directly.

---

# 15. Phase 8 — Representation Ablation

Run:

```text
Experiment A: Autophase + PPO
Experiment B: IR2Vec + PPO
Experiment C: Autophase + IR2Vec + PPO
```

Keep as much as possible fixed:

- benchmark splits
- pass action space
- training budget
- evaluation metrics
- PPO hyperparameters
- seeds

Question:

> Does learned representation improve optimization quality, generalization, or optimization cost?

Possible outcome:

```text
Hybrid improves quality
but costs more computation
```

or:

```text
Autophase is competitive
and much cheaper
```

Both are useful results.

---

# 16. Phase 9 — Sequence Analysis

Do not treat the pass sequence as a black box.

For successful episodes record:

```text
program
pass_1
pass_2
...
pass_T
reward_t
instruction_count_t
```

Then analyze:

- most frequent passes
- common pass transitions
- repeated pass motifs
- pass-pair effectiveness
- sequence length
- per-program sequence diversity
- disagreement across seeds

Build transition matrices.

Example:

```text
pass A → pass B
pass A → pass C
pass B → pass D
```

This gives the project a compiler-analysis component in addition to RL.

---

# 17. Phase 10 — Program-Adaptive Behavior

Demonstrate that different programs receive different learned sequences.

For example:

```text
Program A:
mem2reg → instcombine → gvn → dce

Program B:
simplifycfg → sroa → dce → instcombine

Program C:
loop-simplify → licm → gvn → dce
```

Do not claim the sequence is universally optimal.

Instead report:

> The learned policy produces program-dependent pass sequences.

Then correlate sequence behavior with IR characteristics.

---

# 18. Phase 11 — Multi-Objective Extension

Only begin after the single-objective system is stable.

Potential objectives:

1. instruction count
2. code size
3. execution time

Example weighted reward:

```text
R = α * size_improvement + β * runtime_improvement
```

Keep the single-objective benchmark as the control.

Compare:

```text
Instruction-only PPO
vs
Multi-objective PPO
```

Report Pareto-style trade-offs where practical.

Do not let this phase delay the core project.

---

# 19. Phase 12 — Efficiency / Cost Analysis

RL compiler optimization has two costs:

## Optimization quality

How much does the generated sequence improve the program?

## Search/optimization cost

How long does the agent take to find/apply the sequence?

Track:

```text
LLVM pass execution time
environment step time
episode time
number of passes
model inference time
total optimization time
```

A model that gains 0.5% instruction reduction while taking 100x longer should not be presented as an unconditional improvement.

---

# 20. Phase 13 — Statistical Evaluation

Do not rely only on averages.

For held-out benchmarks:

- report distributions
- use paired comparisons because methods run on the same programs
- include confidence intervals where feasible
- use a suitable paired statistical test for the final comparison
- report effect size where appropriate

The exact statistical test should match the metric distribution and assumptions.

The purpose is to establish whether an observed difference is robust rather than a lucky seed/benchmark artifact.

---

# 21. Phase 14 — Reproducibility

Every experiment must be reconstructable.

Record:

```text
git commit
LLVM version
Python version
OS
hardware
benchmark manifest checksum
pass catalog version
model architecture
PPO hyperparameters
reward version
random seed
training steps
evaluation script version
```

Every result artifact should carry enough metadata to identify how it was generated.

Create:

```bash
scripts/reproduce_results.sh
```

that reproduces the final reported experiments as closely as practical.

---

# 22. Phase 15 — Testing

## Unit tests

Test:

- reward calculation
- pass-name mapping
- environment reset
- environment step
- observation dimensions
- representation concatenation
- normalization
- termination
- metric collection

## Integration tests

At minimum:

```text
program → LLVM IR → environment reset → one action → next state → metrics
```

## Reproducibility test

Same:

```text
benchmark
seed
configuration
```

should produce deterministic behavior where the underlying LLVM/toolchain path is deterministic.

Document any unavoidable nondeterminism.

---

# 23. Phase 16 — Visualization

The final project should have publication-quality figures.

Required:

### Figure 1
Training reward curve.

### Figure 2
Test-set comparison against:

```text
-O2 / -O3 / -Oz / Random / Greedy / PPO
```

### Figure 3
Autophase vs IR2Vec vs Hybrid.

### Figure 4
Optimization-quality vs optimization-cost trade-off.

### Figure 5
Example learned pass sequence over time for selected programs.

### Figure 6
Optional dimensionality reduction of learned representations for analysis.

Avoid visualizations that merely look impressive without answering a research question.

---

# 24. Final Demonstration

Build a small reproducible demo:

```text
Input program
     ↓
Generate LLVM IR
     ↓
Show baseline metrics
     ↓
Run PPO policy
     ↓
Display selected passes
     ↓
Show metric after every pass
     ↓
Show final comparison
```

Example:

```text
Program: benchmark_x

              Instructions

-O3             812
Greedy          794
RL              761

RL sequence:
1. mem2reg       1248 → 1031
2. instcombine   1031 → 974
3. gvn            974 → 941
4. dce            941 → 761
```

The exact numbers above are illustrative only; never fabricate benchmark results.

---

# 25. Final CV-Level Deliverables

By completion, the repository should contain:

1. Working LLVM optimization environment.
2. Versioned pass catalog.
3. Benchmark train/validation/test split.
4. Random baseline.
5. Greedy baseline.
6. LLVM `-O2/-O3/-Oz` baselines.
7. PPO + Autophase.
8. PPO + IR2Vec.
9. PPO + Autophase/IR2Vec hybrid.
10. Held-out generalization results.
11. Ablation studies.
12. Per-step pass traces.
13. Reproducible experiment configs.
14. Figures/tables.
15. README explaining the scientific contribution.
16. Reproduction script.
17. Tests/CI where practical.

---

# 26. What Counts as Success

## Minimum success

A stable RL environment and PPO agent that can learn non-random behavior.

## Good success

PPO beats random and provides useful program-specific sequences on held-out programs.

## Strong success

PPO beats or complements simple search/LLVM baselines on a meaningful subset while generalizing to unseen programs.

## Excellent success

The project demonstrates a statistically supported advantage from learned LLVM-IR representations, or produces a clear and useful negative result showing where learned representations fail versus simpler Autophase features.

## Exceptional outcome

A reproducible learned compiler optimization system with:

- strong held-out results
- representation ablations
- cost/quality analysis
- sequence interpretation
- multi-objective extension
- polished public repository
- technical report/paper-style write-up

Do not optimize the narrative for "beating LLVM everywhere." Optimize it for a rigorous, defensible result.

---

# 27. Anti-Goals

Do not:

- build an LLM chatbot around compiler optimization
- hard-code program-type → pass rules
- claim a method is novel without literature verification
- train and test on the same programs
- tune on the test set
- report only cherry-picked programs
- compare only against random
- hide optimization time
- use a huge pass action space before the environment is stable
- add GNNs/Transformers solely because they sound advanced
- add multi-objective RL before the single-objective baseline is working
- build a large UI before the research pipeline is reliable

---

# 28. Suggested Milestone Order

```text
M0  LLVM toolchain verification
 ↓
M1  Pass execution engine
 ↓
M2  Benchmark manifest + split
 ↓
M3  Gymnasium environment
 ↓
M4  Random + LLVM baselines
 ↓
M5  Greedy baseline
 ↓
M6  PPO + Autophase
 ↓
M7  Held-out evaluation
 ↓
M8  IR2Vec integration
 ↓
M9  Autophase vs IR2Vec vs Hybrid
 ↓
M10 Sequence/program-adaptation analysis
 ↓
M11 Efficiency analysis
 ↓
M12 Optional multi-objective extension
 ↓
M13 Statistical analysis
 ↓
M14 Reproducibility + CI
 ↓
M15 Final figures + report + README
```

---

# 29. Codex Operating Instructions

Codex should treat this document as the project's source of truth.

## Before implementing a phase

1. Read the relevant section of `MASTERPLAN.md`.
2. Inspect the current repository state.
3. Determine which phase is actually complete.
4. Do not skip acceptance criteria.
5. Prefer a minimal working implementation over speculative abstractions.

## While implementing

- Keep changes modular.
- Write tests with the feature.
- Keep configuration outside code.
- Do not silently change benchmark splits.
- Do not silently change reward definitions.
- Do not silently expand the pass set.
- Log tool/compiler versions.
- Keep failed experiments available when useful for analysis.

## Before advancing to the next phase

Codex should verify the previous phase's acceptance criteria.

If a phase fails, fix the phase instead of building additional complexity on top of it.

## When external libraries/APIs have changed

Prefer current official documentation.

Do not assume an old CompilerGym/Gym API is still correct.

---

# 30. Decision Gates

## Gate A — Before RL

Must have:

```text
LLVM pass execution works
environment step/reset works
metrics are correct
random episodes run
baseline pipelines run
```

## Gate B — Before learned representation

Must have:

```text
PPO + Autophase trains
held-out evaluation works
baseline comparison works
results are reproducible
```

## Gate C — Before multi-objective RL

Must have:

```text
single-objective PPO stable
representation comparison complete
optimization cost measured
```

## Gate D — Before final claims

Must have:

```text
locked test set
multiple seeds
baseline comparisons
statistical analysis
limitations documented
```

---

# 31. Expected Final Research Narrative

The final report should follow this structure:

## Problem

LLVM optimization is highly dependent on phase/pass ordering.

## Gap

Fixed pipelines do not adapt explicitly to each individual program.

## Approach

Model pass ordering as a sequential decision problem and use RL to choose passes.

## Representation

Compare handcrafted IR features (Autophase) against learned LLVM-IR representation (IR2Vec) and their hybrid.

## Evaluation

Compare learned policies against LLVM pipelines, random selection, and greedy search on unseen benchmarks.

## Findings

Answer:

1. Does RL learn useful phase sequences?
2. Does it generalize?
3. Which representation works best?
4. What is the optimization-quality/cost trade-off?
5. Which pass sequences emerge?

## Limitations

Discuss:

- benchmark coverage
- LLVM-version dependence
- reward proxy limitations
- search/optimization overhead
- pass-space restrictions
- representation cost

---

# 32. Final Positioning

The project is NOT:

> "Use AI to optimize code."

It is:

> **A reinforcement-learning compiler optimization system that learns program-specific LLVM phase/pass sequences and studies whether learned representations of LLVM IR improve optimization decisions.**

The main technical chain is:

```text
LLVM Compiler
      +
Program Representation
      +
Reinforcement Learning
      +
Experimental Evaluation
      =
Research-grade learned compiler optimizer
```

The project should remain centered on **Reinforcement Learning for LLVM Compiler Phase-Sequence Optimization** throughout all phases.
