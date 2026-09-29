from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import BaseConfig, FIGURE_DIR, TABLE_DIR
from .core import C, set_seed, simulate_acceptance_count
from .plot_style import apply_ieee_style, save_figure


def deterministic_sampler(d_mean: float, _rng: np.random.Generator) -> float:
    return d_mean


def exponential_sampler(d_mean: float, rng: np.random.Generator) -> float:
    return float(rng.exponential(d_mean))


def lognormal_sampler(d_mean: float, rng: np.random.Generator) -> float:
    variance = d_mean
    sigma2 = np.log(1.0 + variance / (d_mean * d_mean))
    sigma = np.sqrt(sigma2)
    mu = np.log(d_mean) - 0.5 * sigma2
    return float(rng.lognormal(mean=mu, sigma=sigma))


def mc_cost(k: int, d_mean: float, sampler, alpha: float, cd: float, cv: float, n_samples: int, rng: np.random.Generator) -> float:
    total_n = 0.0
    total_a = 0.0
    for _ in range(n_samples):
        d = sampler(d_mean, rng)
        n_t = k * (cd + cv) + 2.0 * d + cv
        accepted = simulate_acceptance_count(k, alpha, rng)
        a_t = accepted + 1
        total_n += n_t
        total_a += a_t
    return total_n / total_a


def run() -> None:
    cfg = BaseConfig(alpha=0.7, cd=1.0, cv=0.5, k_max=20, seed=42)
    rng = set_seed(cfg.seed)

    d_means = [10.0, 50.0, 100.0, 200.0]
    ks = [1, 3, 5, 7, 10]
    n_samples = int(os.getenv("ROBUSTNESS_SAMPLES", "100000"))

    samplers = {
        "Deterministic": deterministic_sampler,
        "Exponential": exponential_sampler,
        "LogNormal": lognormal_sampler,
    }

    rows = []
    for dist_name, sampler in samplers.items():
        for d_mean in d_means:
            for k in ks:
                mc = mc_cost(k, d_mean, sampler, cfg.alpha, cfg.cd, cfg.cv, n_samples, rng)
                theory = float(C(k, d_mean, cfg.alpha, cfg.cd, cfg.cv))
                rows.append(
                    {
                        "distribution": dist_name,
                        "d_mean_ms": d_mean,
                        "k": k,
                        "mc_cost": mc,
                        "theory_cost": theory,
                        "abs_error": abs(mc - theory),
                    }
                )

    df = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "table_robustness.csv", index=False)
    df.to_markdown(TABLE_DIR / "table_robustness.md", index=False)

    apply_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)

    for ax, (dist_name, _) in zip(axes, samplers.items()):
        sub = df[df["distribution"] == dist_name]
        for d_mean in d_means:
            s = sub[sub["d_mean_ms"] == d_mean]
            ax.plot(s["k"].to_numpy(), s["mc_cost"].to_numpy(), marker="o", label=f"d={int(d_mean)}")
        ax.set_title(dist_name)
        ax.set_xlabel("k")

    axes[0].set_ylabel("MC cost (ms/token)")
    axes[-1].legend(frameon=True, loc="best")

    save_figure(fig, FIGURE_DIR / "fig_robustness")
    plt.close(fig)


if __name__ == "__main__":
    run()
