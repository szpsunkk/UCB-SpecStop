"""Offline β sweep for UCB-SpecStop using a previously recorded r5_round_log.csv.

Why offline:
    The cloud is busy / measurement runs are expensive. We already paid to
    collect a full round log at one β; the (T_r, A_r) samples observed for
    each arm k are noisy draws from the *same* distributions UCB would face
    at any β. Bootstrapping fresh trajectories from those per-arm pools is a
    standard, honest way to compare β values without re-running the cluster.

What it does:
    1. Read r5_round_log.csv. Extract the rows for strategy='ucb' (the run we
       care about — naive_ucb is on a different estimator).
    2. For each arm k, build a sample pool of observed (T_r, A_r) pairs.
    3. For each candidate β, simulate UCBSpecStop for T rounds, drawing the
       observation at the chosen arm with replacement from that arm's pool.
       Use oracle_C from the original log (a constant for the same delay) to
       compute regret.
    4. Average over `n_seeds` bootstrap resamples → mean cumulative regret
       and 95% CI per β.
    5. Print a ranking table and write tune_beta_results.csv into the suite
       output directory.

Usage:
    python hardware/tune_beta_offline.py \
        --suite qwen \
        --betas 0.3,0.5,0.7,1.0,1.5,2.0 \
        --n-seeds 50

This does NOT mutate r5_round_log.csv or any other artefact. Pick the winning
β, then re-run R5 with --beta <winner> if you want fresh hardware numbers; or
just cite the offline sweep in the appendix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.baselines import UCBSpecStop


def parse_betas(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip()]


def simulate_one(pools: dict, oracle_C: float, k_max: int, beta: float,
                 T: int, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap trajectory. Returns cumulative regret array of length T."""
    alg = UCBSpecStop(k_max=k_max, beta=beta)
    cum = np.zeros(T, dtype=float)
    running = 0.0
    for t in range(1, T + 1):
        k = alg.select_arm(t)
        pool = pools.get(k)
        if pool is None or len(pool) == 0:
            # Arm not visited in original log → can't bootstrap. Skip update,
            # but treat regret as 0 for this step (rare; only happens if the
            # original UCB never tried arm k).
            cum[t - 1] = running
            continue
        idx = rng.integers(0, len(pool))
        T_r, A_r = pool[idx]
        alg.update(k, T_r, A_r)
        inst = (T_r / A_r) - oracle_C
        running += inst
        cum[t - 1] = running
    return cum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="qwen", help="Output suite folder under outputs/hardware_revised/")
    ap.add_argument("--log", default=None,
                    help="Override r5_round_log.csv path (default: outputs/hardware_revised/<suite>/r5_round_log.csv)")
    ap.add_argument("--betas", type=parse_betas, default=parse_betas("0.3,0.5,0.7,1.0,1.5,2.0"))
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--T", type=int, default=None,
                    help="Number of rounds to simulate (default: same as the log)")
    ap.add_argument("--n-seeds", type=int, default=50,
                    help="Bootstrap replicates per β")
    ap.add_argument("--out", default=None,
                    help="Override output csv path")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else \
        REPO / "outputs" / "hardware_revised" / args.suite / "r5_round_log.csv"
    out_path = Path(args.out) if args.out else \
        REPO / "outputs" / "hardware_revised" / args.suite / "tune_beta_results.csv"

    if not log_path.exists():
        sys.exit(f"r5_round_log.csv not found at {log_path}. Run R5 first.")

    df = pd.read_csv(log_path)
    df_ucb = df[df.strategy == "ucb"].copy()
    if df_ucb.empty:
        sys.exit("No 'ucb' rows in round log. Did R5 run with the ucb strategy?")

    # Oracle reference cost is constant within a single R5 run (same delay).
    oracle_C = float(df_ucb["oracle_C"].iloc[0])
    if not np.allclose(df_ucb["oracle_C"].values, oracle_C, atol=1e-6):
        print(f"[warn] oracle_C varies across rows; using mean={df_ucb['oracle_C'].mean():.3f}")
        oracle_C = float(df_ucb["oracle_C"].mean())

    # Build per-arm pools of (T_r, A_r)
    pools = {}
    for k, sub in df_ucb.groupby("k"):
        pools[int(k)] = list(zip(sub["T_r"].astype(float).tolist(),
                                 sub["A_r"].astype(float).tolist()))
    print(f"[tune-beta] log: {log_path}")
    print(f"[tune-beta] oracle_C = {oracle_C:.4f} ms/token")
    print(f"[tune-beta] arm pool sizes: " +
          ", ".join(f"k={k}:{len(v)}" for k, v in sorted(pools.items())))

    T = args.T if args.T is not None else len(df_ucb)
    print(f"[tune-beta] T={T} per replicate, n_seeds={args.n_seeds}")
    print(f"[tune-beta] β grid: {args.betas}\n")

    rows = []
    for beta in args.betas:
        regrets_T = np.zeros(args.n_seeds, dtype=float)
        for s in range(args.n_seeds):
            rng = np.random.default_rng(2026_05_09 + s)
            traj = simulate_one(pools, oracle_C, args.k_max, beta, T, rng)
            regrets_T[s] = traj[-1]
        mean = float(regrets_T.mean())
        std = float(regrets_T.std(ddof=1))
        ci95 = 1.96 * std / np.sqrt(args.n_seeds)
        rows.append({
            "beta": beta,
            "mean_cumulative_regret_ms": mean,
            "std_cumulative_regret_ms": std,
            "ci95_ms": ci95,
            "n_seeds": args.n_seeds,
            "T": T,
        })
        print(f"  β={beta:>4}:  cum.regret = {mean:8.1f} ± {ci95:5.1f} ms")

    out = pd.DataFrame(rows).sort_values("mean_cumulative_regret_ms").reset_index(drop=True)
    out.to_csv(out_path, index=False)
    best = out.iloc[0]
    print(f"\n[tune-beta] Best β = {best['beta']} "
          f"(cum.regret {best['mean_cumulative_regret_ms']:.1f} ± {best['ci95_ms']:.1f} ms)")
    print(f"[tune-beta] Saved {out_path}")


if __name__ == "__main__":
    main()
