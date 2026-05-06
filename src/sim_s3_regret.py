from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .baselines import (
    EXP3Ratio,
    EpsilonGreedyRatio,
    PerRoundRatioUCB,
    UCBSpecStop,
)
from .config import BaseConfig, FIGURE_DIR, RAW_DIR, RegretConfig
from .core import C, compute_kstar, set_seed, simulate_acceptance_count
from .plot_style import apply_ieee_style, save_figure

ALGORITHMS = {
    "UCB-SpecStop": "ours",
    r"$\varepsilon$-Greedy-Ratio": "eg",
    "EXP3-Ratio": "exp3",
    "Per-Round-Ratio UCB": "perround",
}

LINESTYLES = {
    "UCB-SpecStop": "-",
    r"$\varepsilon$-Greedy-Ratio": "--",
    "EXP3-Ratio": "-.",
    "Per-Round-Ratio UCB": ":",
}

COLORS = {
    "UCB-SpecStop": "#1f77b4",
    r"$\varepsilon$-Greedy-Ratio": "#ff7f0e",
    "EXP3-Ratio": "#2ca02c",
    "Per-Round-Ratio UCB": "#d62728",
}


def _make_alg(tag: str, k_max: int, beta: float):
    if tag == "ours":
        return UCBSpecStop(k_max=k_max, beta=beta)
    if tag == "eg":
        return EpsilonGreedyRatio(k_max=k_max, epsilon=0.1)
    if tag == "exp3":
        return EXP3Ratio(k_max=k_max)
    if tag == "perround":
        return PerRoundRatioUCB(k_max=k_max, beta=beta)
    raise ValueError(f"Unknown algorithm tag: {tag}")


def _simulate_one(
    tag: str,
    d_mean: float,
    base_cfg: BaseConfig,
    reg_cfg: RegretConfig,
    stochastic_delay: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (regret array [n_trials, t_rounds], arm_choice array [n_trials, t_rounds])."""
    all_regrets = np.zeros((reg_cfg.n_trials, reg_cfg.t_rounds), dtype=float)
    all_arms = np.zeros((reg_cfg.n_trials, reg_cfg.t_rounds), dtype=int)

    k_star = compute_kstar(base_cfg.alpha, base_cfg.cd, base_cfg.cv, d_mean, base_cfg.k_max)
    c_star = float(C(k_star, d_mean, base_cfg.alpha, base_cfg.cd, base_cfg.cv))

    # cost scale for EXP3 loss normalisation
    cost_scale = float(C(1, d_mean, base_cfg.alpha, base_cfg.cd, base_cfg.cv))

    for trial in range(reg_cfg.n_trials):
        rng = set_seed(base_cfg.seed + trial)
        alg = _make_alg(tag, base_cfg.k_max, reg_cfg.beta)
        cumulative_regret = 0.0

        for t in range(1, reg_cfg.t_rounds + 1):
            if tag in ("ours", "perround"):
                k = alg.select_arm(t)
            else:
                k = alg.select_arm(t, rng)

            # Deterministic delay: N_t depends only on k (Thm 3 / mean-sufficiency).
            # Stochastic delay adds variance that slows convergence without changing
            # which arm is optimal — use deterministic as the primary scenario.
            if stochastic_delay:
                d_t = rng.exponential(d_mean)
            else:
                d_t = d_mean  # deterministic

            n_t = k * (base_cfg.cd + base_cfg.cv) + 2.0 * d_t + base_cfg.cv
            accepted = simulate_acceptance_count(k, base_cfg.alpha, rng)
            a_t = accepted + 1

            if tag == "exp3":
                alg.update(k, n_t, a_t, cost_scale=cost_scale)
            else:
                alg.update(k, n_t, a_t)

            # Pseudo-regret: use theoretical cost at d_mean (not realised d_t).
            c_k = float(C(k, d_mean, base_cfg.alpha, base_cfg.cd, base_cfg.cv))
            cumulative_regret += c_k - c_star
            all_regrets[trial, t - 1] = cumulative_regret
            all_arms[trial, t - 1] = k

    return all_regrets, all_arms


def run() -> None:
    base_cfg = BaseConfig()
    reg_cfg = RegretConfig(
        t_rounds=int(os.getenv("REGRET_T_ROUNDS", "10000")),
        n_trials=int(os.getenv("REGRET_N_TRIALS", "1000")),
        beta=float(os.getenv("REGRET_BETA", "1.0")),
    )
    d_mean = float(os.getenv("REGRET_D_MEAN", "50.0"))

    k_star = compute_kstar(base_cfg.alpha, base_cfg.cd, base_cfg.cv, d_mean, base_cfg.k_max)
    t_range = np.arange(1, reg_cfg.t_rounds + 1)

    apply_ieee_style()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---- main regret figure: cumulative (linear) + log-log growth rate ----
    fig, (ax_reg, ax_loglog) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # reference curve — auto-scaled after collecting all algorithms' final regret
    sqrt_ref = np.sqrt(t_range * np.log(t_range + 1.0))

    summary_rows = []
    final_regrets: dict[str, float] = {}

    for label, tag in ALGORITHMS.items():
        regrets, arms = _simulate_one(tag, d_mean, base_cfg, reg_cfg, stochastic_delay=False)
        mean_r = regrets.mean(axis=0)
        std_r = regrets.std(axis=0)

        ax_reg.plot(t_range, mean_r, label=label,
                    linestyle=LINESTYLES[label], color=COLORS[label], linewidth=1.2)
        ax_reg.fill_between(t_range, mean_r - std_r, mean_r + std_r,
                            alpha=0.10, color=COLORS[label])

        # log-log panel: slope ≈ 0.5 → O(√(T log T)), slope ≈ 1 → linear
        ax_loglog.plot(t_range, mean_r, label=label,
                       linestyle=LINESTYLES[label], color=COLORS[label], linewidth=1.2)

        np.save(RAW_DIR / f"regret_{tag}_d{int(d_mean)}.npy", regrets)
        np.save(RAW_DIR / f"arms_{tag}_d{int(d_mean)}.npy", arms)
        final_regrets[label] = float(mean_r[-1])
        summary_rows.append({
            "algorithm": label,
            "d_mean_ms": d_mean,
            "delay_mode": "deterministic",
            "k_star": k_star,
            "final_regret_mean": float(mean_r[-1]),
            "final_regret_std": float(std_r[-1]),
        })

    # scale the O(sqrt(t log t)) reference to the median of all final regrets so it
    # sits in the middle of the figure and clearly shows sublinear growth
    ref_scale = np.median(list(final_regrets.values())) / sqrt_ref[-1]
    ax_reg.plot(t_range, ref_scale * sqrt_ref, linestyle=":", color="gray",
                linewidth=1.0, label=r"$O(\sqrt{t\log t})$")

    ax_reg.set_xlabel("Round $t$")
    ax_reg.set_ylabel("Cumulative regret $R(t)$")
    ax_reg.set_title(fr"$\alpha={base_cfg.alpha}$, $d={int(d_mean)}$ ms (det.)")
    ax_reg.legend(frameon=True, fontsize=6)

    # log-log panel: growth rate visible as slope.
    # UCB-SpecStop empirical slope < 1 ↔ sublinear; Per-Round slope ≈ 1 ↔ linear.
    ax_loglog.set_xscale("log")
    ax_loglog.set_yscale("log")
    t_ends = np.array([t_range[0], t_range[-1]], dtype=float)
    ucb_final = final_regrets["UCB-SpecStop"]
    for exp, lbl, col in [(0.5, r"slope $\frac{1}{2}$", "#aaaaaa"),
                          (1.0, "slope $1$", "#cccccc")]:
        y_start = ucb_final * (t_ends[0] / t_ends[-1]) ** exp
        ax_loglog.plot(t_ends, [y_start, ucb_final], ":", color=col,
                       linewidth=0.8, label=lbl)
    ax_loglog.set_xlabel("Round $t$")
    ax_loglog.set_ylabel("Cumulative regret $R(t)$")
    ax_loglog.set_title("Log-log scale (slope = growth rate)")
    ax_loglog.legend(frameon=True, fontsize=6)

    plt.tight_layout()
    save_figure(fig, FIGURE_DIR / "fig_regret")
    plt.close(fig)

    # ---- ablation: ratio-of-sums vs per-round-ratio ----
    fig2, (ax_ab, ax_ab_ll) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    abl_pairs = [
        ("UCB-SpecStop\n(ratio-of-sums)", "ours", "-", "#1f77b4"),
        ("Per-Round-Ratio UCB\n(biased)", "perround", "--", "#d62728"),
    ]
    abl_finals: dict[str, float] = {}
    for label, tag, ls, col in abl_pairs:
        regrets, arms = _simulate_one(tag, d_mean, base_cfg, reg_cfg, stochastic_delay=False)
        mean_r = regrets.mean(axis=0)
        std_r = regrets.std(axis=0)
        ax_ab.plot(t_range, mean_r, label=label, linestyle=ls, color=col, linewidth=1.2)
        ax_ab.fill_between(t_range, mean_r - std_r, mean_r + std_r, alpha=0.12, color=col)
        ax_ab_ll.plot(t_range, mean_r, label=label, linestyle=ls, color=col, linewidth=1.2)
        abl_finals[label] = float(mean_r[-1])

    ref_scale2 = max(abl_finals.values()) / sqrt_ref[-1]
    ax_ab.plot(t_range, ref_scale2 * sqrt_ref, ":", color="gray", linewidth=1.0,
               label=r"$O(\sqrt{t\log t})$")
    ax_ab.set_xlabel("Round $t$")
    ax_ab.set_ylabel("Cumulative regret $R(t)$")
    ax_ab.set_title("Ablation: ratio-of-sums vs per-round")
    ax_ab.legend(frameon=True, fontsize=6)

    # log-log: UCB-SpecStop slope < 1 (sublinear), Per-Round slope ≈ 1 (linear)
    ax_ab_ll.set_xscale("log")
    ax_ab_ll.set_yscale("log")
    ucb_abl_final = abl_finals["UCB-SpecStop\n(ratio-of-sums)"]
    for exp, lbl, col in [(0.5, r"slope $\frac{1}{2}$", "#aaaaaa"),
                          (1.0, "slope $1$", "#cccccc")]:
        y_s = ucb_abl_final * (t_ends[0] / t_ends[-1]) ** exp
        ax_ab_ll.plot(t_ends, [y_s, ucb_abl_final], ":", color=col,
                      linewidth=0.8, label=lbl)
    ax_ab_ll.set_xlabel("Round $t$")
    ax_ab_ll.set_ylabel("Cumulative regret $R(t)$")
    ax_ab_ll.set_title("Log-log: UCB slope$<$1, Per-Round slope$≈$1")
    ax_ab_ll.legend(frameon=True, fontsize=6)

    plt.tight_layout()
    save_figure(fig2, FIGURE_DIR / "fig_ablation")
    plt.close(fig2)

    # save summary
    df = pd.DataFrame(summary_rows)
    df.to_csv(RAW_DIR / "regret_summary.csv", index=False)
    df.to_markdown(RAW_DIR / "regret_summary.md", index=False)
    print(f"  k*={k_star} at d={d_mean}ms, alpha={base_cfg.alpha}")
    for row in summary_rows:
        print(f"  {row['algorithm']}: final_regret={row['final_regret_mean']:.2f}")


if __name__ == "__main__":
    run()
