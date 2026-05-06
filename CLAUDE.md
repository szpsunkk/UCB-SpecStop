# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This is a **research paper repository in the experimental-planning stage**, not a software project. It currently contains only:

- `main.pdf` — the compiled paper "Optimal Stopping Theory for Speculative Decoding Under Communication Constraints" (INFOCOM-style, 9+1 pages)
- `experiment.md` — the full experiment plan (simulation S1–S5 and hardware H0–H5), with Python snippets intended to be extracted into runnable scripts

There is no LaTeX source, no build/lint/test tooling, and no implementation code yet. Any scripts, Makefiles, or project scaffolding must be created from scratch — check what already exists before assuming a structure.

## Paper: Core Framework

The work formulates distributed speculative decoding as an **optimal stopping problem** and contributes an online-learning algorithm (UCB-SpecStop) for choosing the draft length `k` under uncertain communication delay.

Notation used throughout `experiment.md` — keep consistent in any code:

- `alpha` — per-token acceptance probability (geometric model, Assumption 1)
- `cd`, `cv` — per-token draft (edge) and verify (cloud) latency in ms
- `d` — mean one-way network delay in ms
- `k` — draft length (action / arm); `k*` is the optimum; `K_max` is the arm-count cap
- `N_t`, `A_t` — round-`t` elapsed time and accepted-token count (incl. bonus)
- `B(k, alpha) = (1 - alpha^(k+1)) / (1 - alpha)` — expected accepted tokens
- `C(k, d, alpha, cd, cv) = (k*(cd+cv) + 2*d + cv) / B(k, alpha)` — per-token cost
- `d_c` — critical delay for the phase transition (Theorem 5); closed form in `experiment.md` `dc_theory`

Five theoretical predictions drive the experiments: Phase Transition (Thm 5), Monotonicity (Thm 2), Latency Improvement (Cor 1), UCB-SpecStop regret `O(sqrt(T log T))` (Thm 7), and Value of Information (Thm 6).

## Baselines to Implement (B1–B6)

When writing experiment code, all six baselines from `experiment.md` §II must be supported side-by-side: Fixed-k (B1), Greedy/`d=0`-optimal (B2), SLED-style timeout (B3), SpecDec++ threshold (B4), Oracle `k*(d)` (B5), and Naive UCB1 on `mean(N/A)` (B6). B6 is critical — its purpose is to show why the ratio-of-sums `S_N/S_A` estimator in UCB-SpecStop is necessary rather than the naive `E[N/A]`.

## Two-Stage Experiment Plan

`experiment.md` is organized as simulation first, hardware second — match this order when implementing:

1. **Simulation (S1–S5)**: pure Python/NumPy Monte Carlo. Outputs Fig. 2 (phase transition), Fig. 3 (regret curves, two gap regimes), Fig. 4 (VOI), plus Table I. Default params: `alpha=0.7, cd=1.0, cv=0.5`.
2. **Hardware (H0–H5)**: Jetson Orin Nano Super (edge, runs draft model, `cd ≈ 15 ms/token`) ↔ RTX 3090 server (cloud, runs target model, `cv ≈ 3 ms/token`). Two model pairs: Qwen2.5-0.5B/7B and LLaMA-3.2-1B/8B. Network conditions are shaped with `tc netem` (deterministic, normal, Pareto, and Markov two-state good/bad) plus a real-WiFi condition (H5).

Any end-to-end implementation splits into `edge_client.py` (Jetson) and `cloud_server.py` (3090) — see `experiment.md` §H2 for the reference skeleton and protocol (the client sends draft tokens, the server returns `n_accepted`, verified tokens, and `verify_time`).

## UCB-SpecStop: Implementation Invariants

When writing or reviewing the algorithm (`experiment.md` §S3), verify:

- Bookkeeping is `S_N[k]`, `S_A[k]`, `T_k[k]` — **sums**, not running means. The index is `S_N/S_A - beta * sqrt(T_k * log(t)) / S_A`. Swapping in `mean(N/A)` silently turns this into B6 and breaks the regret guarantee.
- Arms are 1-indexed in the math (`k = 1..K_max`) but stored 0-indexed in arrays; keep the `+1` conversion explicit.
- The accepted-token count includes the bonus token: `A_t = n_accepted + 1`.
- Acceptance is simulated as a truncated geometric: stop at the first rejection.

## Figure / Table Targets

The paper expects specifically: Fig. 2 (`fig_phase_transition.pdf`), Fig. 3 (`fig_regret.pdf`, two subplots), Fig. 4 (`fig_voi.pdf`), Fig. 5 (per-position alpha, hardware), Table I (simulation), Table II (hardware). File names in `experiment.md` should be preserved so the eventual LaTeX can reference them unchanged.
