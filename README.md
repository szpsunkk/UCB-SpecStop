# UCB-SpecStop Simulation Reproduction

**Paper**: "Optimal Stopping Theory for Speculative Decoding Under Communication Constraints"

This repository contains a complete simulation implementation covering experiments S1–S5 from Section VI of the paper.

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. One-click Run (Full)

```bash
bash run_all.sh
```

S3 (regret) and S5 (robustness) are computationally heavy at paper-scale settings. For a quick smoke test:

```bash
REGRET_T_ROUNDS=2000 REGRET_N_TRIALS=100 ROBUSTNESS_SAMPLES=20000 bash run_all.sh
```

Environment variable defaults:
| Variable | Default | Description |
|---|---|---|
| `REGRET_T_ROUNDS` | 10000 | Rounds per bandit trial (T) |
| `REGRET_N_TRIALS` | 1000 | Monte Carlo trials |
| `REGRET_BETA` | 1.0 | UCB exploration parameter β |
| `REGRET_D_MEAN` | 50.0 | Mean delay for regret figure (ms) |
| `ROBUSTNESS_SAMPLES` | 100000 | MC samples per cell in S5 |

## 3. Run a Single Experiment

```bash
python -m src.sim_s1_phase_transition   # Fig. 2
python -m src.sim_s2_latency_table       # Table I
python -m src.sim_s3_regret              # Fig. 3 + Fig. 4 ablation
python -m src.sim_s4_voi                 # Fig. 5
python -m src.sim_s5_robustness          # Table II
```

## 4. File Structure

| File | Purpose |
|---|---|
| `src/core.py` | Paper formulas: `B(k,α)`, `C(k,d)`, `dc_theory`, acceptance simulation |
| `src/baselines.py` | All 6 baseline policies + bandit algorithms |
| `src/plot_style.py` | IEEE-style matplotlib settings, PDF+PNG export |
| `src/sim_s1_phase_transition.py` | S1: k*(d) phase transition, Fig. 2 |
| `src/sim_s2_latency_table.py` | S2: per-token latency table, Table I |
| `src/sim_s3_regret.py` | S3: bandit regret comparison, Fig. 3 + ablation Fig. 4 |
| `src/sim_s4_voi.py` | S4: value of information, Fig. 5 |
| `src/sim_s5_robustness.py` | S5: distribution robustness, Table II |

## 5. Outputs

### Figures (`outputs/figures/`)
| File | Paper figure |
|---|---|
| `fig_phase_transition.pdf/png` | Fig. 2 |
| `fig_regret.pdf/png` | Fig. 3 |
| `fig_ablation.pdf/png` | Fig. 4 (ratio-of-sums vs per-round-ratio ablation) |
| `fig_voi.pdf/png` | Fig. 5 |
| `fig_robustness.pdf/png` | S5 distribution robustness |

### Tables (`outputs/tables/`)
| File | Paper table |
|---|---|
| `table_simulation_baselines.csv/md` | Table I (α=0.7) |
| `table_multi_alpha_d100.csv/md` | Improvement across α ∈ {0.5…0.9} at d=100ms |
| `table_phase_transition_dc.csv/md` | dc values: theory vs empirical |
| `table_voi_scan.csv/md` | VOI scan data |
| `table_robustness.csv/md` | Table II |

### Raw arrays (`outputs/raw/`)
Regret Monte Carlo arrays (`regret_{alg}_d{d}.npy`) and summary CSV from S3.

## 6. Baselines Implemented

| Label | Description |
|---|---|
| Fixed-k | Static draft length k ∈ {1,3,5,7,10} |
| Confidence-Stop | Halt when α^k < p_min (EAGLE/SpecDec++ analogue, p_min=0.3) |
| Oracle-Mean | Optimal fixed k with known delay mean (Theorem 3 baseline) |
| ε-Greedy-Ratio | ε=0.1 greedy with ratio-of-sums estimator |
| EXP3-Ratio | Adversarial EXP3 adapted to ratio objective |
| Per-Round-Ratio UCB | UCB1 minimising mean(N_t/A_t) — biased estimator (ablation target) |
| **UCB-SpecStop** | **Algorithm 1 (ours): ratio-of-sums UCB** |

## 7. Reproducibility and Data Integrity

- No fabricated data. All numbers are computed by the scripts from closed-form formulas or Monte Carlo simulation.
- Random seeds are deterministic (`config.py → BaseConfig.seed`).

## 8. Notes on Paper Draft Values

Two values in the paper draft differ from simulation output and are noted here for reference:

**dc (critical delay):** The paper text states dc ≈ 3.8ms for α=0.7, cd=1ms, cv=0.5ms. The analytic formula (9) and numerical simulation both give **dc ≈ 1.6ms** for these parameters. The formula derivation (from the stopping condition C(1,d) ≤ C(2,d)) has been verified independently.

**"Up to 38%" improvement vs fixed k=5:** With α=0.7 and d=100ms, simulation gives **6.7% improvement**. The ~38-42% figure is achieved at α=0.9, d=100ms (see `table_multi_alpha_d100.md`). Since Table I in the paper draft has all dashes (not yet filled), these draft estimates may need to be revised.
