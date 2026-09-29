"""Version-three low-overhead ratio/prefix controllers and frozen runner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHODS_V3 = (
    "fixed", "sn", "per_round_ucb", "ratio_regularized", "finite_horizon_fixed",
    "prefix_ratio", "prefix_global_dp", "prefix_local_dp", "position_ratio", "position_dp",
    "position_online_dp", "direct_target",
)
SOURCES = [
    "hardware/run_algorithm_closed_loop.py", "hardware/closed_loop.py", "hardware/cloud_server.py",
    "hardware/content_features.py", "src/content_ratio.py", "src/structured_bandits.py",
    "src/regularized_ratio.py", "src/baselines.py", "hardware/kv_cache.py",
    "hardware/run_tmc_revision.py", "hardware/run_revised_experiments.py",
    "hardware/run_closed_loop_experiment.py", "hardware/delay_proxy.py", "hardware/target_precision.py",
    "src/prefix_hazard.py", "hardware/graph_draft.py",
    "hardware/run_algorithm_v3.py", "src/horizon_ratio.py",
    "src/core.py", "src/config.py", "src/plot_style.py", "src/__init__.py",
]


class HorizonControllerAdapter:
    """Use only past complete physical-arm feedback, including clipped tails."""

    def __init__(self, method, calibration):
        from src.horizon_ratio import PositionRatioController, RenewalRatioController
        self.method = method
        costs, yields = np.asarray(calibration["costs"], float), np.asarray(calibration["yields"], float)
        zero = np.asarray(calibration["zero_costs"], float)
        settings = calibration.get("finite_cap_parameters", {})
        if method.startswith("position_"):
            objective = "ratio" if method == "position_ratio" else "horizon"
            parameter = "position_ratio_shrink" if objective == "ratio" else "position_dp_shrink"
            self.algorithm = PositionRatioController(
                costs, yields, zero, calibration["offsets"],
                position_shrink=float(settings.get(parameter, 16)), objective=objective,
                max_horizon=32, online=method == "position_online_dp",
            )
        else:
            mode = {"finite_horizon_fixed": "frozen", "prefix_ratio": "shared",
                    "prefix_global_dp": "shared", "prefix_local_dp": "request_local"}[method]
            self.algorithm = RenewalRatioController(
                costs, yields, zero, mode=mode,
                objective="ratio" if method == "prefix_ratio" else "horizon",
                local_prior_strength=float(settings.get("request_dp_prior_strength", 8)),
                max_horizon=32,
            )
        self.decisions = []
        self.rounds_selected = 0

    def begin_request(self):
        self._request_reset_pending = True
        self.decisions.clear()

    def select_arm_with_context(self, draft_session, remaining_tokens):
        if getattr(self, "_request_reset_pending", True):
            self.algorithm.begin_request()
            self._request_reset_pending = False
        self.remaining = int(remaining_tokens)
        arm = self.algorithm.select_arm(self.remaining)
        self.rounds_selected += 1
        self.decisions.append({"remaining_tokens_before_selection": self.remaining,
                               "selection_uses_logit_features": False})
        return int(arm)

    def update(self, selected_arm, elapsed_ms, delivered_tokens, censored=False, censor_reason=None):
        if censored and censor_reason != "generation_cap":
            return False, censor_reason
        actual = min(int(selected_arm), max(self.remaining - 1, 0))
        if delivered_tokens < 1:
            return False, "zero_delivered_tokens"
        updated = self.algorithm.update(actual, float(elapsed_ms), int(delivered_tokens))
        return bool(updated), None if updated else "frozen_calibrated_model"

    def state_dict(self):
        return {"method": self.method, "rounds_selected": self.rounds_selected,
                "algorithm_state": self.algorithm.snapshot(),
                "feedback_scope": "actual executed depth; EOS skipped; no unchosen costs"}


def make_v3_controller(method, calibration):
    if method in ("finite_horizon_fixed", "prefix_ratio", "prefix_global_dp", "prefix_local_dp",
                  "position_ratio", "position_dp", "position_online_dp"):
        return HorizonControllerAdapter(method, calibration)
    from hardware.run_algorithm_closed_loop import make_controller
    return make_controller(method, calibration)


if __name__ == "__main__":
    from hardware.run_algorithm_closed_loop import main
    main()
