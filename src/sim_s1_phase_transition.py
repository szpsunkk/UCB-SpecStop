from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from .config import BaseConfig, FIGURE_DIR, TABLE_DIR
from .core import C, compute_kstar, dc_theory
from .plot_style import apply_ieee_style, save_figure


def _find_dc_exact(alpha: float, cd: float, cv: float) -> float:
    """Exact d_c: root of C(1,d) - C(2,d) = 0 via Brent's method.

    Equivalent to dc_theory(alpha,cd,cv) analytically, but computed numerically
    as an independent empirical cross-check.
    """
    def f(d: float) -> float:
        return float(C(1, d, alpha, cd, cv) - C(2, d, alpha, cd, cv))

    return brentq(f, 1e-9, 1000.0, xtol=1e-10)


def run() -> None:
    cfg = BaseConfig()
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    d_range = np.arange(0.0, 500.0 + 0.5, 0.5)

    # Use a larger K_max for α=0.9 so the optimal arm is not capped at 20.
    # For α=0.5..0.8 K_max=20 is sufficient; for α=0.9 the true k* can exceed 20
    # at large delays, so we evaluate with K_max=40 to show the uncapped behaviour.
    k_max_by_alpha = {0.5: 20, 0.6: 20, 0.7: 20, 0.8: 20, 0.9: 40}

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    summary_rows = []

    for alpha in alphas:
        k_max = k_max_by_alpha[alpha]
        kstars = np.array([compute_kstar(alpha, cfg.cd, cfg.cv, d, k_max) for d in d_range])
        ax.step(d_range, kstars, where="post", label=fr"$\alpha={alpha}$")

        dc = dc_theory(alpha, cfg.cd, cfg.cv)
        # Exact empirical d_c via Brent's method on C(1,d) = C(2,d).
        # dc_theory is analytically exact for the k=1→2 transition, so the
        # two values agree to <0.01%; this cross-check confirms no coding error.
        dc_empirical = _find_dc_exact(alpha, cfg.cd, cfg.cv)

        mask = d_range > dc
        if np.any(mask):
            raw_env = (
                np.log(2.0 * d_range[mask] * (1.0 - alpha) / (cfg.cd + cfg.cv))
                / np.log(1.0 / alpha)
                - 1.0
            )
            # Clip to k ≥ 1: the log envelope is only valid for d >> d_c;
            # below that regime k* = 1 and negative values are meaningless.
            envelope = np.maximum(raw_env, 1.0)
            ax.plot(d_range[mask], envelope, linestyle="--", color="gray", alpha=0.5)

        ax.axvline(dc, linestyle=":", color="black", alpha=0.25)

        rel_err = abs(dc_empirical - dc) / max(abs(dc), 1e-9) * 100.0

        summary_rows.append(
            {
                "alpha": alpha,
                "dc_theory_ms": round(dc, 6),
                "dc_empirical_ms": round(dc_empirical, 6),
                "relative_error_percent": round(rel_err, 4),
            }
        )

    ax.set_xlabel("Mean delay $d$ (ms)")
    ax.set_ylabel("Optimal draft length $k^*$")
    ax.set_title("Phase transition and logarithmic scaling")
    # Note for α=0.9: K_max=40 so the curve is not artificially capped
    ax.legend(ncol=2, frameon=True, fontsize=7)
    ax.set_ylim(bottom=0)  # k* ≥ 1, no negative y-axis

    save_figure(fig, FIGURE_DIR / "fig_phase_transition")
    plt.close(fig)

    df = pd.DataFrame(summary_rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "table_phase_transition_dc.csv", index=False)
    df.to_markdown(TABLE_DIR / "table_phase_transition_dc.md", index=False)


if __name__ == "__main__":
    run()
