from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import BaseConfig, FIGURE_DIR, TABLE_DIR
from .core import C, compute_kstar, dc_theory
from .plot_style import apply_ieee_style, save_figure


def compute_voi(alpha: float, cd: float, cv: float, d_good: float, d_bad: float, p_leave_good: float, p_leave_bad: float | None = None, k_max: int = 20):
    if p_leave_bad is None:
        p_leave_bad = 1.0 - p_leave_good

    pi_good = p_leave_bad / (p_leave_good + p_leave_bad)
    pi_bad = p_leave_good / (p_leave_good + p_leave_bad)

    mu_d = pi_good * d_good + pi_bad * d_bad

    k_blind = compute_kstar(alpha, cd, cv, mu_d, k_max)
    c_blind = float(C(k_blind, mu_d, alpha, cd, cv))

    k_good = compute_kstar(alpha, cd, cv, d_good, k_max)
    k_bad = compute_kstar(alpha, cd, cv, d_bad, k_max)

    c_adaptive = pi_good * float(C(k_good, d_good, alpha, cd, cv)) + pi_bad * float(C(k_bad, d_bad, alpha, cd, cv))

    voi = c_blind - c_adaptive
    return voi, k_blind, k_good, k_bad


def run() -> None:
    cfg = BaseConfig(alpha=0.7, cd=1.0, cv=0.5, k_max=20, seed=42)
    d_good = 5.0
    # Start from 1 ms so that the single-state threshold d_c ≈ 1.6 ms is visible.
    d_bad_range = np.arange(1.0, 500.0 + 1.0, 1.0)
    p_transition = 0.1

    rows = []
    vois = []

    for d_bad in d_bad_range:
        voi, k_blind, k_good, k_bad = compute_voi(
            cfg.alpha,
            cfg.cd,
            cfg.cv,
            d_good,
            float(d_bad),
            p_transition,
            None,
            cfg.k_max,
        )
        vois.append(voi)
        rows.append(
            {
                "d_bad_ms": d_bad,
                "voi_ms_per_token": voi,
                "k_blind": k_blind,
                "k_good": k_good,
                "k_bad": k_bad,
            }
        )

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(d_bad_range, vois, color="#1f77b4")

    # Single-state phase-transition threshold (Thm 5): k*(d) jumps 1→2 at d_c.
    dc = dc_theory(cfg.alpha, cfg.cd, cfg.cv)
    ax.axvline(
        dc,
        linestyle="--",
        color="gray",
        linewidth=0.9,
        label=fr"$d_c = {dc:.2f}$ ms (single-state threshold)",
    )

    # First d_bad > d_good where k_bad strictly exceeds k_good: the onset of
    # meaningful VOI gain on the high-delay side of the plot.
    k_good_val = rows[0]["k_good"] if rows else None
    first_exceed = next(
        (r["d_bad_ms"] for r in rows if r["d_bad_ms"] > d_good and r["k_bad"] > (k_good_val or 0)),
        None,
    )
    if first_exceed is not None:
        ax.axvline(
            first_exceed,
            linestyle=":",
            color="red",
            linewidth=0.9,
            label=fr"$k_{{bad}} > k_{{good}}$ at $d_{{bad}} = {first_exceed:.0f}$ ms",
        )

    ax.set_xlabel("Bad-state mean delay (ms)")
    ax.set_ylabel("VOI (ms/token)")
    ax.set_title("Value of Information")
    ax.legend(frameon=True)

    save_figure(fig, FIGURE_DIR / "fig_voi")
    plt.close(fig)

    df = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "table_voi_scan.csv", index=False)
    df.to_markdown(TABLE_DIR / "table_voi_scan.md", index=False)


if __name__ == "__main__":
    run()
