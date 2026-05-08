# UCB-SpecStop

**Paper**: "Optimal Stopping Theory for Speculative Decoding Under Communication Constraints"

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
4. [Hardware Experiments (H0, R1–R6)](#4-hardware-experiments-h0-r1r6)
  - [Hardware Architecture](#41-hardware-architecture)
  - [Model Suites](#42-model-suites)
  - [Hardware Environment Setup](#43-hardware-environment-setup)
  - [Step 1 — Start Cloud Server](#44-step-1--start-cloud-server)
  - [Step 2 — Download Models](#45-step-2--download-models)
  - [Step 3 — Measure Parameters (H0)](#46-step-3--measure-parameters-h0)
  - [Step 4 — Run Experiments (R1–R6)](#47-step-4--run-experiments-r1r6)
  - [Hardware Outputs](#48-hardware-outputs)
5. [Baselines Implemented](#5-baselines-implemented)
6. [Reproducibility](#6-reproducibility)
7. [Notes on Paper Values](#7-notes-on-paper-values)

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
│   ├── measure_params.py         # H0: measure cd, cv, alpha from real hardware
│   ├── run_revised_experiments.py # R1–R6: main experiment runner (runs on Jetson)
│   ├── run_all_suites.sh         # Batch runner for all three model suites
│   ├── edge_client.py            # Low-level edge–cloud protocol helper
│   ├── prompts.txt               # Prompt pool for hardware runs
│   ├── requirements_hw.txt       # Hardware-side Python dependencies
│   └── DEPLOY.md                 # Detailed deployment guide
├── outputs/
│   ├── figures/                  # PDF + PNG figures for paper
│   ├── tables/                   # CSV + Markdown tables
│   ├── raw/                      # Monte Carlo arrays from S3
│   ├── hardware/                 # H0 measured parameter files
│   └── hardware_revised/         # R1–R6 per-suite results
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

## 4. Hardware Experiments (H0, R1–R6)

Real edge-cloud deployment validating simulation predictions with actual LLM inference.

### 4.1 Hardware Architecture

```
┌─────────────────────────────────┐       LAN / WiFi        ┌─────────────────────────────────┐
│  Edge (Jetson Orin Nano Super)  │ ◄────────────────────► │  Cloud (RTX 3090 server)        │
│                                 │   POST /verify           │                                 │
│  • Draft model (0.5–1B params)  │   GET  /ping             │  • Target model (7–8B params)   │
│  • run_revised_experiments.py   │                          │  • cloud_server.py              │
│  • measure_params.py            │   IP: 192.168.3.72:8000  │  • FastAPI + uvicorn            │
│  IP: 192.168.3.108              │                          │                                 │
└─────────────────────────────────┘                          └─────────────────────────────────┘
```

**Protocol**: the edge sends `context_ids + draft_ids (+ optional draft_log_probs)` to `/verify`; the cloud returns `n_accepted`, `bonus_token_id`, and `verify_time_ms`. Proper rejection sampling is used when log-probs are provided; greedy argmax fallback otherwise.

**Delay control**: `sch_netem` for kernel-level shaping when available; software-injected `time.sleep` as fallback (auto-detected).

### 4.2 Model Suites


| Suite   | Edge draft model                   | Cloud target model                    |
| ------- | ---------------------------------- | ------------------------------------- |
| `qwen`  | `Qwen/Qwen2.5-0.5B`                | `Qwen/Qwen2.5-7B-Instruct`            |
| `llama` | `meta-llama/Llama-3.2-1B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct`    |
| `phi`   | `microsoft/Phi-3-mini-4k-instruct` | `microsoft/Phi-3-small-128k-instruct` |


> **Note**: Llama models are gated. You must run `huggingface-cli login` with an authorized account before downloading.

### 4.3 Hardware Environment Setup

On both machines:

```bash
pip install -r hardware/requirements_hw.txt
```

### 4.4 Step 1 — Start Cloud Server

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

# Phi suite
nohup python hardware/cloud_server.py \
  --model /home/skk/local/models/Phi-3-small-128k-instruct \
  --host 0.0.0.0 --port 8000 &
```

Verify the server is reachable from the Jetson:

```bash
curl http://192.168.3.72:8000/ping
# → {"status":"ok","ts":...}
```

### 4.5 Step 2 — Download Models

Models are loaded from local paths by default (`local_files_only=True`). Use the HF mirror for initial downloads:

**On the Jetson** (edge draft models):

```bash
# Qwen
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir /home/jetson/local/models/Qwen2.5-0.5B

# Phi
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download microsoft/Phi-3-mini-4k-instruct \
  --local-dir /home/jetson/local/models/Phi-3-mini-4k-instruct

# Llama (requires authorized HF account — login first)
huggingface-cli login
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /home/jetson/local/models/Llama-3.2-1B-Instruct
```

**On the 3090** (cloud target models):

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /home/skk/local/models/Qwen2.5-7B-Instruct

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download microsoft/Phi-3-small-128k-instruct \
  --local-dir /home/skk/local/models/Phi-3-small-128k-instruct

huggingface-cli login
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir /home/skk/local/models/Llama-3.1-8B-Instruct
```

### 4.6 Step 3 — Measure Parameters (H0)

Run `measure_params.py` on the Jetson once per model suite. This measures the real `cd` (draft latency/token), `cv` (verify latency/token), and `alpha` (acceptance rate) and saves a JSON file used by all subsequent experiments.

```bash
# Qwen
python hardware/measure_params.py \
  --draft-model /home/jetson/local/models/Qwen2.5-0.5B \
  --server http://192.168.3.72:8000 \
  --prompts hardware/prompts.txt
mv outputs/hardware/params_measured.json outputs/hardware/params_qwen.json

# Llama
python hardware/measure_params.py \
  --draft-model /home/jetson/local/models/Llama-3.2-1B-Instruct \
  --server http://192.168.3.72:8000 \
  --prompts hardware/prompts.txt
mv outputs/hardware/params_measured.json outputs/hardware/params_llama.json

# Phi
python hardware/measure_params.py \
  --draft-model /home/jetson/local/models/Phi-3-mini-4k-instruct \
  --server http://192.168.3.72:8000 \
  --prompts hardware/prompts.txt
mv outputs/hardware/params_measured.json outputs/hardware/params_phi.json
```

### 4.7 Step 4 — Run Experiments (R1–R6)

**Single suite:**

```bash
python hardware/run_revised_experiments.py \
  --suite qwen \
  --server http://192.168.3.72:8000 \
  --params outputs/hardware/params_qwen.json \
  --prompts hardware/prompts.txt \
  --n-prompts 500 \
  --n-rounds 200 \
  --exp all

  python hardware/run_revised_experiments.py \
    --suite llama \
    --draft-model /home/jetson/local/models/Llama-3.2-1B-Instruct \
    --server http://192.168.3.72:8000 \
    --params outputs/hardware/params_llama.json \
    --prompts hardware/prompts.txt \
    --n-prompts 500 \
    --n-rounds 200 \
    --exp all
```

Replace `--suite qwen` / `--params outputs/hardware/params_qwen.json` with `llama` or `phi` for the other suites.

**CLI reference:**


| Argument           | Default                                 | Description                                   |
| ------------------ | --------------------------------------- | --------------------------------------------- |
| `--suite`          | —                                       | `qwen` / `llama` / `phi` — selects model pair |
| `--server`         | `http://192.168.3.72:8000`              | Cloud server URL                              |
| `--params`         | `outputs/hardware/params_measured.json` | Measured parameter file from H0               |
| `--prompts`        | `hardware/prompts.txt`                  | Prompt pool                                   |
| `--n-prompts`      | 500                                     | Prompts to sample from                        |
| `--n-rounds`       | 200                                     | Rounds per strategy (R4/R5)                   |
| `--exp`            | `all`                                   | `r1`/`r2`/`r3`/`r4`/`r5`/`r6`/`all`           |
| `--k-max`          | 10                                      | Maximum draft length                          |
| `--beta`           | 1.0                                     | UCB exploration parameter β                   |
| `--allow-download` | false                                   | Allow on-the-fly HF model download            |


**Batch run (all three suites sequentially):**

```bash
chmod +x hardware/run_all_suites.sh
SERVER=http://192.168.3.72:8000 N_PROMPTS=500 N_ROUNDS=200 EXP=all \
  ./hardware/run_all_suites.sh
```

The script pauses before each suite and prompts you to confirm the correct cloud model is running.

### 4.8 Hardware Outputs

Results are written to `outputs/hardware_revised/<suite>/`:


| File                      | Experiment | Description                                              |
| ------------------------- | ---------- | -------------------------------------------------------- |
| `r1_calibration.csv`      | R1         | Per-delay cd, cv, RTT calibration                        |
| `r2_acceptance.csv`       | R2         | Prefix acceptance P(L≥k), conditional q_k, sample counts |
| `r3_phase_transition.csv` | R3         | k-sweep around d_c, theory vs measured                   |
| `r4_strategy_compare.csv` | R4         | All baselines comparison per delay setting               |
| `r5_regret_data.npz`      | R5         | UCB vs Naive-UCB vs EXP3 vs Oracle regret curves         |
| `r6_markov_voi.json`      | R6         | Markov two-state channel VOI                             |
| `run_config.json`         | —          | Full run configuration (seed, params, suite, timestamp)  |


**Unified log fields** (every round in R1–R6):


| Field                         | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| `configured_one_way_delay_ms` | Software-injected one-way delay                     |
| `bare_rtt_ms`                 | Measured round-trip time (no delay injection)       |
| `measured_comm_round_ms`      | Actual observed communication time                  |
| `accepted_draft_len`          | Number of draft tokens accepted (L)                 |
| `accepted_total`              | Total accepted tokens including bonus (L + 1 = A_t) |
| `total_round_time_ms`         | Full round wall-clock time                          |


---

## 5. Baselines Implemented


| Label               | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| Fixed-k             | Static draft length k ∈ {1, 3, 5, 7, 10}                    |
| Confidence-Stop     | Halt when α^k < p_min (EAGLE/SpecDec++ analogue, p_min=0.3) |
| Oracle-Mean         | Optimal fixed k with known delay mean (Theorem 3 baseline)  |
| ε-Greedy-Ratio      | ε=0.1 greedy with ratio-of-sums estimator                   |
| EXP3-Ratio          | Adversarial EXP3 adapted to ratio objective                 |
| Per-Round-Ratio UCB | UCB1 on mean(N_t/A_t) — biased estimator (ablation B6)      |
| **UCB-SpecStop**    | **Algorithm 1 (ours): ratio-of-sums UCB, `S_N/S_A`**        |


> **B6 is a critical ablation target.** Its purpose is to demonstrate why the ratio-of-sums estimator `S_N/S_A` is necessary: using the per-round mean `mean(N_t/A_t)` breaks the regret guarantee.

---

## 6. Reproducibility

- **Simulation**: no fabricated data; all numbers computed from closed-form formulas or Monte Carlo. Seeds are fixed via `BaseConfig.seed`.
- **Hardware**: every round records `run_id`, `prompt_id`, `seed`, `strategy`, and `k_selected` for full traceability. The `run_config.json` in each output directory captures the complete run configuration.
- **Model loading**: scripts default to `local_files_only=True`. Pass `--allow-download` only when intentionally downloading from HF.

---

## 7. Notes on Paper Values

**dc (critical delay):** The paper text states dc ≈ 3.8ms for α=0.7, cd=1ms, cv=0.5ms. The analytic formula (9) and simulation both give **dc ≈ 1.6ms** for these parameters. The derivation (from stopping condition C(1,d) ≤ C(2,d)) has been independently verified.

**"Up to 38%" improvement vs fixed k=5:** At α=0.7, d=100ms, simulation gives **6.7% improvement**. The ~38–42% figure is achieved at α=0.9, d=100ms (see `table_multi_alpha_d100.md`). Since Table I in the paper draft is not yet filled, these estimates may need revision.