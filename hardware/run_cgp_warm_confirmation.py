"""Independent equal-calibration comparison, including throughput UCB controls."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware.cgp_controller import make_controller
from hardware.closed_loop import ServiceController
from hardware.run_cgp_closed_loop import main
from src.baselines import BanditSpecThroughputUCB


METHODS = (
    "cgp_warm", "sn_warm", "per_round_ucb_warm", "per_round_frontier_warm",
    "ucbspec_release_warm", "ucbspec_arm_bonus_warm", "fixed", "direct_target",
)
PRIMARY_BASELINES = (
    "sn_warm", "per_round_ucb_warm", "ucbspec_release_warm",
    "ucbspec_arm_bonus_warm", "fixed",
)


class WarmThroughputController(ServiceController):
    def __init__(self, label, calibration):
        super().__init__("ucbspec_throughput", list(range(1, 11)))
        self.declared_label = label
        self.algorithm = BanditSpecThroughputUCB(
            10, exploration_weight=6.0, common_bonus=label == "ucbspec_release_warm")
        self.warm_start(np.asarray(calibration["costs"]), np.asarray(calibration["yields"]))

    def state_dict(self):
        result = super().state_dict()
        result.update(method=self.declared_label,
                      internal_update_method=self.method,
                      common_bonus=self.algorithm.common_bonus,
                      exploration_weight=self.algorithm.exploration_weight)
        return result


def make_warm_confirmation(method, calibration):
    if method in ("ucbspec_release_warm", "ucbspec_arm_bonus_warm"):
        return WarmThroughputController(method, calibration)
    return make_controller(method, calibration)


if __name__ == "__main__":
    main(controller_factory=make_warm_confirmation,
         cold_methods=METHODS, warm_methods=METHODS)
