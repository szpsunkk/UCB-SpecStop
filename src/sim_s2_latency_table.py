from __future__ import annotations

import pandas as pd

from .baselines import FIXED_K_CHOICES, confidence_stop_policy, oracle_mean_policy
from .config import BaseConfig, TABLE_DIR
from .core import C, compute_kstar


def _build_table(alpha: float, cd: float, cv: float, k_max: int, delays: list[int]) -> pd.DataFrame:
    k_conf = confidence_stop_policy(alpha, p_min=0.3, k_max=k_max)
    rows = []
    for d in delays:
        k_star = compute_kstar(alpha, cd, cv, float(d), k_max)
        c_opt = float(C(k_star, d, alpha, cd, cv))
        fixed_costs = {f"k={k}": float(C(k, d, alpha, cd, cv)) for k in FIXED_K_CHOICES}
        k_oracle_mean = oracle_mean_policy(alpha, cd, cv, float(d), k_max)
        c_oracle_mean = float(C(k_oracle_mean, d, alpha, cd, cv))
        c_conf = float(C(k_conf, d, alpha, cd, cv))
        best_fixed = min(fixed_costs.values())
        delta = (best_fixed - c_opt) / best_fixed * 100.0
        improve_vs_k5 = (fixed_costs["k=5"] - c_opt) / fixed_costs["k=5"] * 100.0
        rows.append({
            "d_ms": d, "k_star": k_star,
            **fixed_costs,
            "Confidence-Stop": c_conf,
            "Oracle-Mean": c_oracle_mean,
            "Ours (k*)": c_opt,
            "Delta_vs_best_fixed_pct": delta,
            "improve_vs_k5_pct": improve_vs_k5,
        })
    return pd.DataFrame(rows)


def run() -> None:
    cfg = BaseConfig(alpha=0.7, cd=1.0, cv=0.5, k_max=20, seed=42)
    delays = [5, 10, 30, 50, 100, 200, 500]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Paper Table I: α=0.7 deterministic delay
    df_main = _build_table(cfg.alpha, cfg.cd, cfg.cv, cfg.k_max, delays)
    df_main.to_csv(TABLE_DIR / "table_simulation_baselines.csv", index=False)
    df_main.to_markdown(TABLE_DIR / "table_simulation_baselines.md", index=False)

    # Multi-alpha sweep at d=100ms.
    # For α=0.9 the true k* can exceed 20, so use K_max=40 to avoid truncation artefact.
    k_max_by_alpha = {0.5: 20, 0.6: 20, 0.7: 20, 0.8: 20, 0.9: 40}
    rows_multi = []
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        k_max = k_max_by_alpha[alpha]
        k_star = compute_kstar(alpha, cfg.cd, cfg.cv, 100.0, k_max)
        truncated = (k_star >= k_max)
        c_opt = float(C(k_star, 100.0, alpha, cfg.cd, cfg.cv))
        c_k5 = float(C(5, 100.0, alpha, cfg.cd, cfg.cv))
        best_fixed = min(float(C(k, 100.0, alpha, cfg.cd, cfg.cv)) for k in FIXED_K_CHOICES)
        rows_multi.append({
            "alpha": alpha,
            "k_max_used": k_max,
            "d_ms": 100,
            "k_star": k_star,
            "k_star_truncated": truncated,
            "C(k*,100)": c_opt,
            "C(k=5,100)": c_k5,
            "improve_vs_k5_pct": (c_k5 - c_opt) / c_k5 * 100.0,
            "Delta_vs_best_fixed_pct": (best_fixed - c_opt) / best_fixed * 100.0,
        })
    df_multi = pd.DataFrame(rows_multi)
    df_multi.to_csv(TABLE_DIR / "table_multi_alpha_d100.csv", index=False)
    df_multi.to_markdown(TABLE_DIR / "table_multi_alpha_d100.md", index=False)


if __name__ == "__main__":
    run()
