# UCB-SpecStop

**Paper**: "How Long to Speculate? Optimal Stopping for LLM Inference Under Communication Constraints"

This repository contains simulation and real-hardware experiments for UCB-SpecStop, an online-learning algorithm that adaptively selects the speculative draft length `k` under uncertain edge-cloud communication delay.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Simulation (S1–S5)](#3-simulation-s1s5)
  - [Environment Setup](#31-environment-setup)
  - [One-click Run](#32-one-click-run)
  - [Run a Single Experiment](#33-run-a-single-experiment)
  - [Simulation Outputs](#34-simulation-outputs)
4. [Hardware Experiments (R1–R6)](#4-hardware-experiments-r1r6)
  - [Hardware Architecture](#41-hardware-architecture)
  - [Model Suites](#42-model-suites)
  - [Hardware Environment Setup](#43-hardware-environment-setup)
  - [Experiment Design Notes (review-driven)](#44-experiment-design-notes-review-driven)
  - [Step 1 — Start Cloud Server](#45-step-1--start-cloud-server)
  - [Step 2 — Download Models](#46-step-2--download-models)
  - [Step 3 — Run Experiments (R1–R6)](#47-step-3--run-experiments-r1r6)
  - [Hardware Outputs](#48-hardware-outputs)
  - [Background Execution](#49-background-execution)

---

## 1. Overview

UCB-SpecStop formulates distributed speculative decoding as an **optimal stopping problem** and contributes a UCB-based bandit algorithm for choosing draft length `k` when the round-trip communication delay is unknown and variable.

Key components:

- `B(k, α)` — expected accepted tokens (truncated geometric)
- `C(k, d, α, cd, cv)` — per-token cost (cost function to minimize)
- `dc` — critical delay threshold for the phase transition (Theorem 5)
- UCB-SpecStop bandit using the **ratio-of-sums** estimator `S_N / S_A` (not per-round mean) to achieve `O(√(T log T))` regret (Theorem 7)

---

## 2. Repository Structure

```
UCB-SpecStop/
├── src/                          # Simulation source
│   ├── core.py                   # B(k,α), C(k,d), dc_theory, acceptance simulation
│   ├── baselines.py              # All bandit algorithms and baseline policies
│   ├── plot_style.py             # IEEE-style matplotlib settings
│   ├── sim_s1_phase_transition.py
│   ├── sim_s2_latency_table.py
│   ├── sim_s3_regret.py
│   ├── sim_s4_voi.py
│   └── sim_s5_robustness.py
├── hardware/                     # Real hardware experiment code
│   ├── cloud_server.py           # FastAPI verify/ping server (runs on 3090)
│   ├── measure_params.py         # Legacy H0 (cd/cv/α) — superseded by R1, kept for compat
│   ├── run_revised_experiments.py # R1–R6: main experiment runner (runs on Jetson)
│   ├── tune_beta_offline.py      # Offline β sweep using a recorded R5 round log
│   ├── run_all_suites.sh         # Batch runner for all model suites
│   ├── edge_client.py            # Low-level edge–cloud protocol helper
│   ├── prompts.txt               # Prompt pool for hardware runs (550 lines)
│   ├── requirements_hw.txt       # Hardware-side Python dependencies
│   └── DEPLOY.md                 # Detailed deployment guide
├── outputs/
│   ├── figures/                  # PDF + PNG figures for paper
│   ├── tables/                   # CSV + Markdown tables
│   ├── raw/                      # Monte Carlo arrays from S3
│   ├── hardware/                 # Legacy H0 parameter files
│   └── hardware_revised/         # R1–R6 per-suite results (calibrated_state.json + figures)
├── run_all.sh                    # One-click simulation runner
├── requirements.txt              # Simulation Python dependencies
└── experiment.md                 # Full experiment specification
```

---

## 3. Simulation (S1–S5)

Pure Python/NumPy Monte Carlo — no GPU required.

### 3.1 Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 One-click Run

```bash
bash run_all.sh
```

S3 (regret) and S5 (robustness) are computationally heavy at paper-scale settings. For a quick smoke test:

```bash
REGRET_T_ROUNDS=2000 REGRET_N_TRIALS=100 ROBUSTNESS_SAMPLES=20000 bash run_all.sh
```


| Variable             | Default | Description                       |
| -------------------- | ------- | --------------------------------- |
| `REGRET_T_ROUNDS`    | 10000   | Rounds per bandit trial (T)       |
| `REGRET_N_TRIALS`    | 1000    | Monte Carlo trials                |
| `REGRET_BETA`        | 1.0     | UCB exploration parameter β       |
| `REGRET_D_MEAN`      | 50.0    | Mean delay for regret figure (ms) |
| `ROBUSTNESS_SAMPLES` | 100000  | MC samples per cell in S5         |


### 3.3 Run a Single Experiment

```bash
python -m src.sim_s1_phase_transition   # Fig. 2  — phase transition k*(d)
python -m src.sim_s2_latency_table      # Table I — per-token latency
python -m src.sim_s3_regret             # Fig. 3  — regret + Fig. 4 ablation
python -m src.sim_s4_voi                # Fig. 5  — value of information
python -m src.sim_s5_robustness         # Table II — distribution robustness
```

### 3.4 Simulation Outputs

**Figures** (`outputs/figures/`)


| File                           | Paper figure                                       |
| ------------------------------ | -------------------------------------------------- |
| `fig_phase_transition.pdf/png` | Fig. 2 — k*(d) phase transition                    |
| `fig_regret.pdf/png`           | Fig. 3 — regret comparison                         |
| `fig_ablation.pdf/png`         | Fig. 4 — ratio-of-sums vs per-round-ratio ablation |
| `fig_voi.pdf/png`              | Fig. 5 — value of information                      |
| `fig_robustness.pdf/png`       | S5 — distribution robustness                       |


**Tables** (`outputs/tables/`)


| File                                | Paper table                                 |
| ----------------------------------- | ------------------------------------------- |
| `table_simulation_baselines.csv/md` | Table I (α=0.7)                             |
| `table_multi_alpha_d100.csv/md`     | Improvement across α ∈ {0.5…0.9} at d=100ms |
| `table_phase_transition_dc.csv/md`  | dc: theory vs empirical                     |
| `table_voi_scan.csv/md`             | VOI scan                                    |
| `table_robustness.csv/md`           | Table II                                    |


**Raw arrays** (`outputs/raw/`): regret Monte Carlo arrays `regret_{alg}_d{d}.npy` from S3.

---

## 4. Hardware Experiments (R1–R6)

Real edge-cloud deployment validating simulation predictions with actual LLM inference. The hardware suite is **calibration-driven**: R1 measures real per-k draft / verify / RTT, R2 measures the empirical prefix acceptance, and R3–R6 reuse those measurements through a chained `calibrated_state.json` so every cost figure and oracle is derived from the same data. The earlier `H0` step is folded into R1 — `measure_params.py` is no longer required.

### 4.1 Hardware Architecture

```
┌─────────────────────────────────┐       LAN / WiFi        ┌─────────────────────────────────┐
│  Edge (Jetson Orin Nano Super)  │ ◄────────────────────► │  Cloud (RTX 3090 server)        │
│                                 │   POST /verify           │                                 │
│  • Draft model (0.5–1B params)  │   GET  /ping             │  • Target model (7–8B params)   │
│  • run_revised_experiments.py   │                          │  • cloud_server.py              │
│  • calibrated_state.json (R1+R2)│   IP: 192.168.3.72:8000  │  • FastAPI + uvicorn            │
│  IP: 192.168.3.108              │                          │                                 │
└─────────────────────────────────┘                          └─────────────────────────────────┘
```

**Protocol**: the edge sends `context_ids + draft_ids (+ optional draft_log_probs + optional seed)` to `/verify`; the cloud returns `n_accepted`, `bonus_token_id`, `verify_time_ms`, plus a server-side timing breakdown (`server_recv_to_verify_start_ms`, `verify_split_ms`, `pack_split_ms`). Proper rejection sampling is used when log-probs are provided; greedy argmax fallback otherwise. When the optional `seed` is set, both the rejection draw and bonus sampling become deterministic for that `(prompt, draft_ids)` pair, enabling **paired-prompt comparison** across strategies.

**Delay control**: `sch_netem` for kernel-level shaping when available; software-injected `time.sleep` as fallback (auto-detected). Every round records `d_eff_one_way_ms = (measured_comm_round_ms − server_total_ms) / 2`, which is the actual one-way network delay used as the input to all oracle formulas (review §D#4).

### 4.2 Model Suites


| Suite   | Edge draft model                   | Cloud target model                 |
| ------- | ---------------------------------- | ---------------------------------- |
| `qwen`  | `Qwen/Qwen2.5-0.5B`                | `Qwen/Qwen2.5-7B-Instruct`         |
| `llama` | `meta-llama/Llama-3.2-1B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` |


> **Note**: Llama models are gated. You must run `huggingface-cli login` with an authorized account before downloading.

### 4.3 Hardware Environment Setup

On both machines:

```bash
pip install -r hardware/requirements_hw.txt
```

### 4.4 Experiment Design Notes (review-driven)

The hardware experiments deliberately avoid mixing closed-form theory parameters with measured ones. Reviewers consistently flag this; the design here is structured to keep them separable:

- **Cost is `Ĉ = ΣT_r / ΣA_r` (ratio-of-sums), not `mean(T_r/A_r)`** — matches the objective UCB-SpecStop optimises (Theorem 7). The `mean_cost_per_token_sanity` column is kept only as a sanity check.
- `**d_eff` replaces configured delay in oracle formulas** — `d_eff = (measured_comm_round − server_total) / 2`. RPC and serialization overhead would otherwise be invisible to any closed-form oracle, shifting the predicted phase boundary the wrong way.
- **Three oracles reported side-by-side** (R3, R4):
  - `theory_oracle` — paper defaults (α=0.7, cd=1, cv=0.5)
  - `calibrated_geometric_oracle` — closed form with R1-measured cd/cv (still geometric `B(k)`)
  - `empirical_oracle` — uses R3 sweep argmin or R2 prefix `B̂(k) = 1 + Σ P(L≥i)`
- **R4 delays are a subset of R3 grid** — `{20, 55, 111, 150}` ms, all keys present in `empirical_kstar_per_delay`. Earlier `int(d_c · {0.5,1,1.5,2})` produced delays not in the R3 sweep, so `empirical_oracle` silently fell back to a prefix approximation that wasn't actually cost-min — making the §D#5 sanity check fire. The script now `assert`s this subset relation at startup.
- **R4 round budget is `max(--n-rounds, 1000)`** — review §D#6 requires UCB to leave the exploration phase. At T=200, UCB still has visible exploration cost vs. an oracle and looks weaker than it actually is. Long-horizon convergence is the job of R5 (T=5000 by default).
- **R4 figure shows 8 bars per delay, not 14** — `fixed1..fixed10` are collapsed into a single `fixed_best (k=…)` bar (per-delay argmin over fixed-k strategies). The full 14-strategy table still lives in `r4_strategy_compare.csv` for the appendix; UCB-SpecStop ("Ours") is highlighted in red. The per-delay choice of best fixed-k is exported to `r4_fixed_best.csv`.
- **Paired-prompt replay** — every strategy walks the same prompt order and forwards `verify_seed = base + prompt_id` so cloud-side rejection sampling is deterministic per `(prompt, draft_ids)` pair. Cross-strategy differences are attributable to `k` selection only.
- **Calibrated state chaining** — each experiment writes `outputs/hardware_revised/<suite>/calibrated_state.json` (cd/cv per k from R1, prefix `P(L≥k)` from R2, empirical k* per delay from R3). Downstream experiments warn loudly if the file is missing rather than silently using stale defaults.
- **R1 measures every k** in `{1, 2, 3, 5, 7, 10}` across delays `{0, 5, 10, 20, 40, 55, 83, 111, 150}` ms (review §B1). Per-k cd, cv vary with batch size and verify length; assuming a single value silently breaks the cost model.
- **R5 runs T=5000 by default**, with a per-arm probe to derive the empirical oracle on the same trace before the bandits start (review §D#6, §D#5).
- **Offline β sweep** — `hardware/tune_beta_offline.py` reads `r5_round_log.csv`, builds per-arm `(T_r, A_r)` bootstrap pools from the recorded UCB run, and replays UCBSpecStop for a β grid without touching the hardware. Mean cumulative regret with 95% CI is written to `tune_beta_results.csv`. β values whose CIs overlap are statistically indistinguishable on that trace — typically a sign that k* is so dominant the bandit collapses to one arm regardless of β; in that case re-tune on the R6 round log where k* changes between regimes.

### 4.5 Step 1 — Start Cloud Server

On the 3090, start the verifier for the suite you want to run. Only one model at a time per port:

```bash
# Qwen suite
nohup python hardware/cloud_server.py \
  --model /home/skk/local/models/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 &

# Llama suite
nohup python hardware/cloud_server.py \
  --model /home/skk/local/models/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 &
```

Verify the server is reachable from the Jetson:

```bash
curl http://192.168.3.72:8000/ping
# → {"status":"ok","ts":...}
```

### 4.6 Step 2 — Download Models

Models are loaded from local paths by default (`local_files_only=True`). Use the HF mirror for initial downloads:

**On the Jetson** (edge draft models):

```bash
# Qwen
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir /home/jetson/local/models/Qwen2.5-0.5B

# Llama (requires authorized HF account — login first)
huggingface-cli login
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /home/jetson/local/models/Llama-3.2-1B-Instruct
```

**On the 3090** (cloud target models):

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /home/skk/local/models/Qwen2.5-7B-Instruct

huggingface-cli login
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir /home/skk/local/models/Llama-3.1-8B-Instruct
```

### 4.7 Step 3 — Run Experiments (R1–R6)

`run_revised_experiments.py` runs the full chain. R1→R2→R3→R4→R5→R6 share state through `calibrated_state.json` in the suite output directory, so `--exp all` is the recommended invocation. The earlier `H0` step (`measure_params.py`) is no longer needed — R1 produces the same `cd / cv / α` quantities and stores them in the chained state file.

> **First-time setup** — make sure no stale state from earlier runs is mixed in: `rm -rf outputs/hardware_revised/<suite>` before the first full run after pulling these changes (the state file schema changed: `cd_per_k_calibrated`, `prefix_P_Lge_k`, `empirical_kstar_per_delay` are new keys).

**Single suite (recommended):**

```bash
nohup python hardware/run_revised_experiments.py \
  --suite qwen --exp all \
  --draft-model /home/jetson/local/models/Qwen2.5-0.5B \
  > qwen_v2.log 2>&1 &

nohup python3 hardware/run_revised_experiments.py \
    --suite llama --exp all \
    --draft-model /home/jetson/local/models/Llama-3.2-1B-Instruct \
  > llama_v2.log 2>&1 &
```

`--suite qwen` is enough. The script auto-resolves `--params` to `outputs/hardware/params_<suite>.json` if not given, and uses the defaults `--server http://192.168.3.72:8000`, `--prompts hardware/prompts.txt`, `--n-prompts 500`, `--n-rounds 200`. Pass any of those explicitly to override. Once R1 finishes, R3/R4/R5/R6 prefer the calibrated values from `calibrated_state.json` — `--params` is only the bootstrap input.

Pass `--exp r2,r3,r4,r5,r6` (comma-separated subset) to skip R1 and reuse the existing state file.

**Per-experiment defaults**


| Exp | Delays (ms)                          | k set                  | Rounds                       | Notes                                                                    |
| --- | ------------------------------------ | ---------------------- | ---------------------------- | ------------------------------------------------------------------------ |
| R1  | {0, 5, 10, 20, 40, 55, 83, 111, 150} | {1, 2, 3, 5, 7, 10}    | 300 / cell                   | per-k cd, cv, RTT, server timing breakdown                               |
| R2  | 0 (no injection)                     | {1..k_max}             | n-prompts (≥500)             | empirical prefix P(L≥k) and conditional q_k                              |
| R3  | {0, 5, 20, 40, 55, 83, 111, 150}     | {1, 2, 3, 4, 5, 7, 10} | 300 / cell                   | three k* curves + U-shaped cost curves; uses d_eff                       |
| R4  | {20, 55, 111, 150}                   | {1..10} (csv)          | `max(--n-rounds, 1000)`      | 14 strategies in csv, 8-bar figure with `fixed_best`; `assert` R3 subset |
| R5  | d_c                                  | {1..k_max}             | T = max(5000, n-rounds × 25) | UCB / NaiveUCB / EXP3 + per-arm oracle probe                             |
| R6  | (d_good, d_bad) derived from d_c     | {1..k_max}             | n-rounds × 2                 | Markov VOI: blind vs contextual UCB                                      |


**CLI reference**


| Argument           | Default                                      | Description                                                                                                            |
| ------------------ | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `--suite`          | —                                            | `qwen` / `llama` — selects model pair                                                                                  |
| `--server`         | `http://192.168.3.72:8000`                   | Cloud server URL                                                                                                       |
| `--params`         | auto: `outputs/hardware/params_<suite>.json` | Initial parameter file (overridden by R1 once it runs). Falls back to `params_measured.json` if no `--suite` is given. |
| `--prompts`        | `hardware/prompts.txt`                       | Prompt pool (≥500 lines for review §D#2)                                                                               |
| `--n-prompts`      | 500                                          | Prompts to sample from                                                                                                 |
| `--n-rounds`       | 200                                          | R4 uses `max(--n-rounds, 1000)`; R5 uses `max(5000, --n-rounds × 25)`; R6 uses `--n-rounds × 2`                        |
| `--exp`            | `all`                                        | `r1` / `r2` / `r3` / `r4` / `r5` / `r6` / `all`                                                                        |
| `--k-max`          | 10                                           | Maximum draft length                                                                                                   |
| `--beta`           | 1.0                                          | UCB exploration parameter β (tune offline with `tune_beta_offline.py`)                                                 |
| `--allow-download` | false                                        | Allow on-the-fly HF model download                                                                                     |


**Batch run (both suites sequentially):**

```bash
chmod +x hardware/run_all_suites.sh
SERVER=http://192.168.3.72:8000 N_PROMPTS=500 N_ROUNDS=200 EXP=all \
  ./hardware/run_all_suites.sh
```

The script pauses before each suite and prompts you to confirm the correct cloud model is running.

**Wall-clock budget (Qwen suite, single Jetson + 3090)**

Estimates derive from a previous full run (`logs/r_qwen.log` mtimes), adjusted for the two settings that changed:


| Stage     | Previous run         | This config                | Driver of change                         |
| --------- | -------------------- | -------------------------- | ---------------------------------------- |
| R1        | ~50 min              | **~50 min**                | unchanged (9 delays × 6 ks × 300 rounds) |
| R2        | ~2 min (80 prompts)  | **~10 min**                | prompts.txt 80 → 550 (review §D#2)       |
| R3        | ~2 h 32 min          | **~2 h 32 min**            | unchanged                                |
| R4        | ~45 min (200 rounds) | **~3 h 45 min**            | `n_rounds` 200 → 1000 (review §D#6)      |
| R5        | ~2 h 18 min          | **~2 h 18 min**            | unchanged (T=5000 already)               |
| R6        | ~6 min               | **~6 min**                 | unchanged                                |
| **Total** | **~6.5 h**           | **≈ 9.5 h (range 9–11 h)** |                                          |


Llama suite is comparable (slightly faster: cd ≈ 49 ms/tok vs 73). Plan for a roughly 10-hour overnight run per suite.

**Risk factors that can blow up the estimate**

- `requests.ReadTimeout=60s` on cold-start verify: previous run crashed mid-R3 with this. Warm up the cloud server (`curl ... /ping`) and run a 5-prompt R2 dry-run first.
- The cloud server (3090) running another job → `/verify` queues up and the timeout fires.
- Network drops: software-injected delay is `time.sleep()`, but real RTT spikes still hit `requests.post`.

**Crash-tolerant alternative**: split the run so a mid-stage failure doesn't waste the earlier hours.

```bash
python3 hardware/run_revised_experiments.py --suite qwen --exp r1            # ~50 min
python3 hardware/run_revised_experiments.py --suite qwen --exp r2,r3         # ~2 h 42 min
python3 hardware/run_revised_experiments.py --suite qwen --exp r4            # ~3 h 45 min
python3 hardware/run_revised_experiments.py --suite qwen --exp r5,r6         # ~2 h 24 min
```

`calibrated_state.json` is written incrementally, so each segment picks up cleanly from the last.

### 4.8 Hardware Outputs

Results are written to `outputs/hardware_revised/<suite>/`.

**Experiment files**


| File                          | Experiment | Description                                                                                                                                     |
| ----------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `r1_calibration.csv`          | R1         | Per-(delay, k) medians: cd/token, cv/token, comm round, server total, d_eff, cost_ratio_of_sums                                                 |
| `r2_acceptance.csv`           | R2         | Prefix `P(L≥k)`, conditional `q_k`, sample counts `n_k`                                                                                         |
| `r3_phase_transition.csv`     | R3         | Per-(delay, k) sweep with ratio-of-sums cost, median d_eff, median server_total                                                                 |
| `r3_phase_summary.csv`        | R3         | Per-delay: d_eff, `k_theory_geometric / k_theory_calibrated / k_theory_empirical / k_empirical_oracle`                                          |
| `r4_strategy_compare.csv`     | R4         | 14 strategies × 4 delays: `cost_ratio_of_sums`, `median_d_eff_one_way_ms`, `ci95`. Full table for the appendix.                                 |
| `r4_fixed_best.csv`           | R4         | Per-delay best fixed-k (the one drawn as `fixed_best` in `fig_r4_strategy_compare.pdf`)                                                         |
| `table_ii_revised.csv`        | R4         | Pivot (strategy × delay) of ratio-of-sums cost                                                                                                  |
| `r5_regret_data.npz`          | R5         | Arrays: `regret_<strat>`, `pulls_<strat>`, `c_oracle`, `oracle_per_arm_costs`                                                                   |
| `r5_round_log.csv`            | R5         | Per round: `t / k / T_r / A_r / S_N[k] / S_A[k] / estimated_C[k] / oracle_C / instant_regret / cumulative_regret / running_Ĉ` (review §B5 spec) |
| `r5_arm_pull_diagnostics.txt` | R5         | Per-100-round arm histograms + UCB index values (audit UCB vs NaiveUCB overlap)                                                                 |
| `r6_markov_voi.json`          | R6         | Blind / contextual ratio-of-sums cost, VOI %, both geometric and empirical k_good / k_bad                                                       |
| `tune_beta_results.csv`       | offline    | Output of `hardware/tune_beta_offline.py`: mean cumulative regret ± CI per β, sorted ascending                                                  |
| `calibrated_state.json`       | chained    | R1 cd/cv per k, R2 prefix, R3 empirical k* per delay                                                                                            |
| `run_config.json`             | —          | Full run configuration (seed, params, suite, timestamp)                                                                                         |


**Figures**


| File                          | Experiment                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `fig_r1_calibration.pdf`      | R1 — comm round vs configured delay, one line per k                                                                            |
| `fig_r2_acceptance.pdf`       | R2 — prefix bar + conditional `q_k` with `q_1` outlier annotated                                                               |
| `fig_r3_phase_transition.pdf` | R3 — four k* curves: geom/d_cfg, geom/d_eff, empirical B(k), sweep argmin                                                      |
| `fig_r3_cost_curves.pdf`      | R3 — U-shaped `Ĉ(k,d)` curves, one per delay                                                                                   |
| `fig_r4_strategy_compare.pdf` | R4 — per-delay 8-bar chart: `fixed_best (k=…)`, greedy, Spec++, three oracles, NaiveUCB, Ours (red). Sorted ascending by cost. |
| `fig_r5_regret.pdf`           | R5 — cumulative regret vs empirical oracle                                                                                     |
| `fig_r5_regret_loglog.pdf`    | R5 — log-log slope with `√(T log T)` reference                                                                                 |
| `fig_r5_arm_pulls.pdf`        | R5 — cumulative arm pull share, one panel per algorithm                                                                        |
| `fig_r6_markov_regret.pdf`    | R6 — blind vs contextual cumulative regret                                                                                     |


**Unified per-round log fields** (every round in R1–R6):


| Field                            | Description                                                     |
| -------------------------------- | --------------------------------------------------------------- |
| `configured_one_way_delay_ms`    | Software-injected one-way delay                                 |
| `bare_rtt_ms`                    | Measured round-trip time (no delay injection)                   |
| `measured_comm_round_ms`         | Actual observed communication time                              |
| `server_recv_to_verify_start_ms` | Server-side queue / deserialization time                        |
| `server_verify_split_ms`         | Server-side forward pass                                        |
| `server_pack_split_ms`           | Server-side response packing                                    |
| `server_total_ms`                | Sum of the three server splits                                  |
| `d_eff_one_way_ms`               | `(measured_comm_round − server_total) / 2` — fed to all oracles |
| `accepted_draft_len`             | Number of draft tokens accepted (L)                             |
| `accepted_total`                 | Total accepted tokens including bonus (L + 1 = A_t)             |
| `total_round_time_ms`            | Full round wall-clock time (T_r)                                |
| `cost_per_token`                 | `T_r / A_r` — sanity only; aggregated cost is ratio-of-sums     |


**Sanity checks built in**

- R1 prints per-cell median cd/cv, d_eff — expect d_eff ≈ configured delay.
- R3 writes `k_empirical_oracle` from the actual sweep argmin; a gap between this and `k_theory_geometric` is the review §B3 finding, not a bug.
- R4 asserts at startup that its delays are a subset of the R3 grid (so `empirical_kstar_per_delay[d]` always exists).
- R4 prints a sanity summary: **empirical_oracle must be the lowest-cost strategy at every delay**. If any fixed-k or greedy is cheaper, the sanity banner lists the violation (review §D#5). With the corrected R3-aligned delays, this banner should print "OK"; if it doesn't, look at `r4_fixed_best.csv` to see which fixed-k beat the oracle and at what delay.

### 4.9 Background Execution

Hardware runs are long-lived (R1+R5 dominate; expect ~6–8 hours per suite at default settings). Use `nohup` + `&` so the job survives terminal disconnects, and redirect stdout/stderr to a log file you can `tail -f` later.

**Cloud server (3090)** — keep one model resident per port. Restart the server after pulling these changes; the `/verify` schema gained `seed`, `server_recv_to_verify_start_ms`, `verify_split_ms`, `pack_split_ms` fields.

```bash
mkdir -p logs
nohup python hardware/cloud_server.py \
  --model /home/skk/local/models/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  > logs/cloud_qwen.log 2>&1 &
echo $! > logs/cloud_qwen.pid

# Tail progress
tail -f logs/cloud_qwen.log

# Confirm it stayed up
curl http://192.168.3.72:8000/ping

# Stop when done
kill "$(cat logs/cloud_qwen.pid)"
```

**Edge experiments (Jetson, R1–R6)** — single suite, full chain. With the auto `--params` resolution, the command shrinks to:

```bash
mkdir -p logs

# Wipe any stale calibrated_state.json (schema changed)
rm -rf outputs/hardware_revised/qwen

nohup python hardware/run_revised_experiments.py \
  --suite qwen --exp all \
  > logs/r_qwen.log 2>&1 &
echo $! > logs/r_qwen.pid
sleep 5 && head -20 logs/r_qwen.log     # confirm "[HW] auto-selected params: ..."
tail -f logs/r_qwen.log
```

If the log immediately ends with a Python traceback, the script crashed at startup — common causes: `outputs/hardware/params_<suite>.json` missing (run `measure_params.py` first, or pass `--params <file>` explicitly), or the cloud server unreachable (`curl http://192.168.3.72:8000/ping`).

**Resume after R1 / skip calibration** — once `calibrated_state.json` exists, you can rerun downstream experiments without redoing R1:

```bash
nohup python hardware/run_revised_experiments.py \
  --suite qwen --exp r5 \
  > logs/r_qwen_r5.log 2>&1 &
```

**Offline β sweep** — once R5 has produced `r5_round_log.csv`, you can search for a better UCB exploration coefficient without re-running the cluster:

```bash
python hardware/tune_beta_offline.py \
  --suite qwen \
  --betas 0.3,0.5,0.7,1.0,1.5,2.0 \
  --n-seeds 50
```

The tool bootstraps fresh trajectories from the per-arm `(T_r, A_r)` samples already in the round log. The result is written to `outputs/hardware_revised/<suite>/tune_beta_results.csv`. **Caveat**: when k* is so dominant that the original UCB collapsed to one arm (≥ 95 % pulls on a single k), the per-arm pools for other arms are too thin to discriminate β values — confidence intervals will overlap. In that case, re-tune on the R6 round log where k* alternates between regimes.

**Edge experiments — batch (both suites)**: the batch script reads from stdin to confirm each suite, so feed it `</dev/null` only after you've already restarted the cloud server for *every* suite. In practice, run the suites one at a time as shown above.

**Operational notes**

- `nohup` writes its own banner to `nohup.out` if you don't redirect — always pass `> logs/<name>.log 2>&1`.
- Saving the PID (`echo $! > logs/<name>.pid`) lets you stop the run cleanly with `kill "$(cat logs/<name>.pid)"`.
- For sessions that must survive SSH drops, prefer `tmux new -s ucb` then run the foreground command inside; detach with `Ctrl-b d` and reattach with `tmux attach -t ucb`.
- Re-check `nvidia-smi` and `curl .../ping` before starting a new suite — the previous cloud model must be stopped to free GPU memory.

---


