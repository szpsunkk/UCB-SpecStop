"""Continuous-request rerun of the Table 4 policy families.

Measure every fixed depth once in each prompt block. Static selectors that
choose the same depth share that measurement; they are not independent runs.
The SpecDec++ label denotes the historical acceptance-threshold adaptation,
not the original paper's trained acceptance predictor.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware.closed_loop import ServiceController
from hardware.run_cgp_closed_loop import main
from hardware.run_cgp_warm_confirmation import make_warm_confirmation
from src.baselines import confidence_stop_policy, greedy_policy
from src.core import compute_kstar


ONLINE_METHODS = (
    "cgp_warm", "sn_warm", "per_round_ucb_warm", "per_round_frontier_warm",
    "ucbspec_release_warm", "ucbspec_arm_bonus_warm",
)
METHODS = ONLINE_METHODS + tuple("fixed_depth_%02d" % k for k in range(1, 11)) + ("direct_target",)


def static_selectors(calibration, raw_rows, configured_one_way_ms):
    """Fit only on the supplied complete calibration cells.

    Use the censored geometric likelihood for alpha (one longest-prefix
    observation per cell). Match the historical cd/cv summary: average the
    per-depth medians of drafting/k and server verification/(k+1).
    """
    costs = np.asarray(calibration["costs"], float)
    yields = np.asarray(calibration["yields"], float)
    if costs.shape != yields.shape or costs.ndim != 2 or costs.shape[1] != 10:
        raise ValueError("requires the paired ten-depth calibration grid")
    if not np.array_equal(yields, np.minimum(yields[:, -1, None], np.arange(2, 12))):
        raise ValueError("non-nested calibration cannot fit prefix survival")
    accepted = yields[:, -1] - 1
    successes = float(accepted.sum())
    failures = int((accepted < 10).sum())
    alpha_raw = successes / (successes + failures) if successes + failures else 0.
    alpha = float(np.clip(alpha_raw, 1e-9, 1 - 1e-9))
    selected = set(calibration["complete_cells"])
    rows = [r for r in raw_rows if r["cell_id"] in selected]
    if len(rows) != costs.size or len({(r["cell_id"], r["k_selected"]) for r in rows}) != costs.size:
        raise ValueError("calibration component rows are missing or duplicated")
    cd = float(np.mean([np.median([r["draft_time_ms"] / k for r in rows if r["k_selected"] == k])
                        for k in range(1, 11)]))
    cv = float(np.mean([np.median([r["verification_response"]["verify_time_ms"] / (k + 1)
                                  for r in rows if r["k_selected"] == k]) for k in range(1, 11)]))
    geometric_yield = 1 + np.cumsum(alpha ** np.arange(1, 11))
    arms = {
        "fixed_k5": 5,
        "greedy": greedy_policy(alpha, cd, cv, 10),
        "specdec_pp_threshold": confidence_stop_policy(alpha, p_min=.3, k_max=10),
        "b4_idealized": compute_kstar(.7, 1., .5, configured_one_way_ms, 10),
        "b5_calibrated_geometric": compute_kstar(alpha, cd, cv, configured_one_way_ms, 10),
        "b6_calibrated_fixed": int(np.argmin(costs.sum(axis=0) / yields.sum(axis=0))) + 1,
        "geometric_measured_cost": int(np.argmin(costs.mean(axis=0) / geometric_yield)) + 1,
    }
    if arms["b6_calibrated_fixed"] != calibration["fixed_arm"]:
        raise ValueError("fixed reference differs from frozen calibration")
    return dict(arms=arms, alpha=alpha, alpha_unclipped=alpha_raw,
                geometric_successes=successes, geometric_failures=failures,
                cd_ms=cd, cv_ms=cv, configured_one_way_ms=configured_one_way_ms,
                calibration_cells=len(costs), acquisition_scope="existing independent calibration only",
                alpha_scope="censored geometric MLE; historical proxy alpha is not reused",
                b5_scope="historical linear-cost rule recalibrated to this runtime",
                strengthened_geometric_scope="measured complete cost curve / geometric yield",
                specdec_scope="historical threshold adaptation, p_min=0.3; no trained predictor",
                alias_scope="identical constant-depth policies reuse their fixed-depth request logs")


def make_table4_controller(method, calibration):
    if method.startswith("fixed_depth_"):
        arm = int(method.rsplit("_", 1)[1])
        if arm not in range(1, 11):
            raise ValueError("unsupported fixed depth")
        return ServiceController("fixed", list(range(1, 11)), fixed_arm=arm)
    return make_warm_confirmation(method, calibration)


if __name__ == "__main__":
    main(controller_factory=make_table4_controller, cold_methods=METHODS, warm_methods=METHODS)
