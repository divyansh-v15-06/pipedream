# Reinforcement Learning for LLVM Compiler Phase-Sequence Optimization
### Landscape & Gap Analysis (Aug 2026)

> **Status:** Planning document for a 2-3 month solo research project.
> **Verification note:** Two of the report's most load-bearing claims — the CompilerGym
> archival date and the existence of the June 2026 empirical LLVM study — were manually
> checked against GitHub and arXiv directly and confirmed accurate as of Aug 10, 2026.
> Venue deadlines (CGO 2027, PLDI 2027) were **not** independently re-verified here and
> should be checked on the official CFP pages before you commit to a target date.

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Recency Check: State of the Art (Feb–Aug 2026)](#2-recency-check-state-of-the-art-febaug-2026)
3. [Gap Analysis](#3-gap-analysis)
4. [Tooling & Dataset Landscape](#4-tooling--dataset-landscape)
5. [Venue Strategy](#5-venue-strategy)
6. [Recommendation: Ranked Shortlist](#6-recommendation-ranked-shortlist)
7. [Verified Sources](#7-verified-sources)

---

## 1. Problem Statement

LLVM's optimization passes are non-commutative, order-dependent, and non-additive —
running them in a different order on the same program produces different IR, so no single
handcrafted pipeline (`-O0`…`-O3`, `-Os`, `-Oz`) is optimal for every program. The search
space is combinatorially intractable (the `-Oz` pipeline alone spans roughly 54⁹⁰ ≈ 10¹⁵⁶
possible sequences), which motivates learning-based search — most recently, reinforcement
learning framed as a Markov Decision Process over pass choices.

The tooling and research landscape supporting this is currently in flux, which materially
changes what's worth building right now.

---

## 2. Recency Check: State of the Art (Feb–Aug 2026)

The field has shifted from single-objective RL agents using cheap static proxies toward
LLM agents, multi-objective graph-based methods, and — critically — empirical work that
undermines assumptions the earlier work relied on.

| Work | Venue / Date | Core Contribution |
|---|---|---|
| **Per-Pass Empirical Study of LLVM -O3** | arXiv:2606.31238, Jun 2026 | Decomposes `-O3` into cumulative pass prefixes; **empirically shows the pipeline is non-monotone** (6.6–9.7% of pass-additions regress performance) and back-loaded (median kernel needs ~85% of the pipeline for 80% of its speedup) — directly undercuts IR-instruction-count as a reliable reward proxy |
| **MileStone** | PLDI 2026, arXiv:2605.23435 | Multi-objective (time / size / energy) phase ordering using a GNN over the IR's control/data-flow graph as a static predictor, paired with RL for exploration; reports up to 45% execution-time reduction under energy budgets vs. standard `-O` levels |
| **Compiler-R1** | arXiv:2506.15701, 2025 | Two-stage LLM agent: SFT on ~19.6k CoT reasoning traces, then RL (PPO/GRPO) fine-tuning against a CompilerGym-derived environment; ~8.46% avg. IR-count reduction vs. `-Oz` across 7 datasets |
| **AutoPass** | arXiv:2606.20373, Jun 2026 | Multi-agent LLM framework that edits from an `-O3` starting point using compiler diagnostics + runtime feedback, aiming at the sample-inefficiency of classic autotuners like OpenTuner |
| **WASM code-size study** | ResearchGate preprint, 2025/2026 | Differential testing + contrastive learning + LLM-generated passes targeting missed WebAssembly code-size optimizations — signals the field diversifying beyond saturated x86 execution-time work |

**Takeaway:** the "IR instruction count as reward proxy" assumption that underpinned most
2020–2024 work is now empirically contested. Any new project leaning on that proxy without
addressing this needs to justify it explicitly, not assume it.

---

## 3. Gap Analysis

| Candidate Angle | Status | Reasoning | Confidence |
|---|---|---|---|
| **A. Cheap/proxy rewards (IR count)** | ❌ Taken & invalidated | Directly contradicted by arXiv:2606.31238's empirical results | High |
| **B. Rigorous cross-domain generalization eval** | ⚠️ Partially open | Some train/test splitting exists (CompilerDream, Compiler-R1) but mostly reports aggregated point estimates, not variance-reported, deliberately-disjoint-domain transfer studies | Medium |
| **C. RL vs. simple baselines under equal compute budget** | ✅ Genuinely open | Papers benchmark against static `-O3`/`-Oz` or OpenTuner, not a strict equal-evaluation-count comparison against random search / simulated annealing | High |
| **D. Narrow targets (WASM code size)** | ✅ Open, high-value | x86 execution time is saturated; WASM is an emerging, deployment-relevant target with structurally different (stack-based) optimization behavior | High |
| **E. Offline / sample-efficient RL** | ❌ Taken & advancing fast | Field has moved to MLIR with Implicit Q-Learning, DOGE, etc. — too much algorithmic depth for a solo 2-3 month project to meaningfully contribute to | High |

---

## 4. Tooling & Dataset Landscape

### CompilerGym is dead — confirmed
**Verified directly:** the `facebookresearch/CompilerGym` GitHub repo shows *"This
repository was archived by the owner on May 27, 2026. It is now read-only."* Last actual
release was v0.2.5 in Nov 2022, and open issues document real dependency rot (e.g.,
`setuptools` conflicts requiring a downgrade to install at all). Building new work on it
is a real engineering risk — no upstream maintainer to fix breakage if the LLVM FFI layer
breaks mid-project.

### MLIR-RL — the actively maintained alternative
`Modern-Compilers-Lab/MLIR-RL` (arXiv:2409.11068) targets MLIR's `Linalg` dialect with a
multi-discrete action space (tiling, tiled parallelization/fusion, loop interchange) rather
than 1D LLVM pass sequencing. Actively maintained, modern toolchain (Python 3.11, LLVM
21.1.5, Conda). Reasonable pivot if you don't need to stay strictly in "LLVM pass ordering."

### Standard benchmark suites (don't invent your own)
- **cBench** — general-purpose C, the field's baseline
- **CHStone / MiBench** — embedded/low-resource, required if targeting code size
- **NAS Parallel Benchmarks (NPB)** — parallel/execution-time evaluation
- **FormAI** — ~109k AI-generated C programs, emerging standard for generalization testing

---

## 5. Venue Strategy

| Venue | Type | Deadline (unverified — confirm on official CFP) | Fit |
|---|---|---|---|
| **CGO 2027 — Student Research Competition** | Extended abstract (2-3 pp) + poster | Trails main-track deadline (main R2 reportedly Sept 10, 2026) | **Best fit** — designed for undergrad/grad students, realistic timeline |
| **CGO 2027 — Main Track** | Full 11-page paper | R2 reportedly Sept 10, 2026 | Not realistic for a Aug-start 2-3 month project |
| **PLDI 2027** | Full paper | CFP not yet published as of this report; likely Nov/Dec 2026 | Plausible if timeline stretches slightly |
| **LCTES 2027** (co-located with PLDI) | Full paper | Follows PLDI 2027 CFP | Good fit specifically if targeting embedded/code-size work |

**Action item:** verify these dates directly on `conf.researchr.org` and the PLDI SIGPLAN
site before planning around them — venue deadlines are exactly the kind of detail that
degrades fastest and is easiest for a research agent to get subtly wrong.

---

## 6. Recommendation: Ranked Shortlist

### 🥇 Rank 1 — "Equal Compute Budget" Reality Check (Angle C)
Lowest engineering risk, highest analytical payoff. Run PPO against random search and
simulated annealing, all capped at the same number of compile evaluations (e.g., 1,000
steps/program), on a frozen/containerized legacy CompilerGym instance. Plot Pareto
frontiers of optimization quality vs. compute budget. Answers a real open question: does
RL's complexity actually earn its keep over dumb parallel search? Well-suited to a CGO SRC
extended abstract.

### 🥈 Rank 2 — Pivot to MLIR Loop-Nest Optimization (MLIR-RL)
Sidesteps the dead-CompilerGym problem entirely by building on actively maintained tooling.
Targets tiling/fusion/interchange on the `Linalg` dialect — directly relevant to current
ML-compiler trends (PyTorch→MLIR lowering).

### 🥉 Rank 3 — RL for WASM Binary Size (Angle D)
Stays within the LLVM ecosystem while moving to a genuinely under-served target. WASM's
stack-based execution model means optimal pass orderings likely differ meaningfully from
register-based x86, giving a clean, contained novelty story.

---

## 7. Verified Sources

| Type | Title | Identifier |
|---|---|---|
| Paper | A Multi-Dimensional, Per-Pass Empirical Study of the LLVM Optimization Pipeline | arXiv:2606.31238 (verified) |
| Paper | MileStone: A Multi-Objective Compiler Phase Ordering Framework | arXiv:2605.23435 |
| Paper | AutoPass: Evidence-Guided LLM Agents for Compiler Performance Tuning | arXiv:2606.20373 |
| Paper | Compiler-R1: Towards Agentic Compiler Auto-tuning with Reinforcement Learning | arXiv:2506.15701 |
| Paper | A Reinforcement Learning Environment for Automatic Code Optimization in the MLIR Compiler | arXiv:2409.11068 |
| Paper | Finding Missed Code Size Optimizations in Compilers using LLMs (WASM) | ResearchGate preprint |
| Repo | CompilerGym (archived) | github.com/facebookresearch/CompilerGym (verified archived May 27, 2026) |
| Repo | MLIR-RL artifact | github.com/mohph197/MLIR-RL-artifact |
| Venue | CGO 2027 | conf.researchr.org/home/cgo-2027 *(deadline unverified)* |
