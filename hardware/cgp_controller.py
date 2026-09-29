"""Continuous-request adapters for the frozen CGP experiment."""

from __future__ import annotations

import numpy as np

from src.cgp_specstop import CGPSpecStop
from hardware.closed_loop import ServiceController


COLD_METHODS = ("cgp", "cgp_unpooled", "cgp_full", "per_round_frontier",
                "sn", "per_round_ucb", "fixed", "direct_target")
WARM_METHODS = ("cgp_warm", "per_round_frontier_warm", "sn_warm",
                "per_round_ucb_warm", "fixed", "direct_target")


class CGPController:
    def __init__(self, method):
        self.method = method
        base = method.removesuffix("_warm") if hasattr(method, "removesuffix") else method.replace("_warm", "")
        self.algorithm = CGPSpecStop(
            prefix_pool=base not in ("cgp_unpooled", "per_round_frontier"),
            frontier=base != "cgp_full",
            estimator="per_round" if base == "per_round_frontier" else "ratio",
        )
        self.rounds_selected = self.calibration_observations = 0
        self.decisions = []

    def begin_request(self):
        self.decisions.clear()

    def select_arm(self):
        self.rounds_selected += 1
        t = self.rounds_selected + self.calibration_observations
        before = self.algorithm.hi
        arm = self.algorithm.select_arm(t)
        self.decisions.append({"frontier_before_selection": before,
                               "frontier_after_selection": self.algorithm.hi,
                               "cgp_state_before_feedback": self.algorithm.snapshot(t)})
        return arm

    def update(self, selected_arm, elapsed_ms, delivered_tokens,
               censored=False, censor_reason=None):
        if censored:
            return False, censor_reason or "request_boundary"
        if delivered_tokens < 1:
            return False, "zero_delivered_tokens"
        self.algorithm.update(selected_arm, elapsed_ms, delivered_tokens)
        return True, None

    def warm_start(self, costs, yields):
        costs, yields = np.asarray(costs, float), np.asarray(yields, float)
        if (costs.ndim != 2 or costs.shape[1] != 10 or costs.shape != yields.shape
                or len(costs) == 0 or not np.isfinite(costs).all() or (costs < 0).any()
                or not np.isfinite(yields).all() or (yields != np.floor(yields)).any()
                or (yields < 1).any() or (yields > np.arange(2, 12)).any()):
            raise ValueError("warm start requires complete measured ten-arm cells")
        for ns, aa in zip(costs, yields):
            for arm in range(1, 11):
                self.algorithm.update(arm, ns[arm - 1], aa[arm - 1])
        # One potential prefix per paired cell, using the longest observed arm.
        # Do not count the same cell ten times for q1.
        self.algorithm.y_cnt[:] = len(costs)
        self.algorithm.y_hit[:] = (yields[:, -1, None] > np.arange(1, 11)).sum(axis=0)
        self.algorithm.hi = 10  # All ten costs have genuinely been measured.
        self.calibration_observations = int(costs.size)

    def state_dict(self):
        return {"method": self.method, "rounds_selected": self.rounds_selected,
                "calibration_observations": self.calibration_observations,
                "algorithm_state": self.algorithm.snapshot(max(2, self.rounds_selected + self.calibration_observations)),
                "feedback_scope": "complete selected-arm cost; observed prefixes only; censored rounds skipped"}


def make_controller(method, calibration):
    base = method.replace("_warm", "")
    if base in ("cgp", "cgp_unpooled", "cgp_full", "per_round_frontier"):
        obj = CGPController(method)
    elif base in ("sn", "per_round_ucb", "fixed"):
        obj = ServiceController(base, list(range(1, 11)), beta=1.0,
                                fixed_arm=calibration["fixed_arm"] if base == "fixed" else None)
    else:
        raise ValueError("unknown CGP comparison method")
    if method.endswith("_warm"):
        obj.warm_start(np.asarray(calibration["costs"]), np.asarray(calibration["yields"]))
    return obj
