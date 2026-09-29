"""Add coarse search and its per-round control to all prior CGP comparisons."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware.cgp_controller import CGPController, WARM_METHODS
from hardware.run_cgp_refresh_closed_loop import METHODS, make_refresh_controller
from hardware.run_cgp_closed_loop import main
from src.cgp_coarse import CGPCoarseSpecStop


COARSE_METHODS = tuple(METHODS) + ("cgp_coarse", "per_round_coarse")


def make_coarse_controller(method, calibration):
    if method not in ("cgp_coarse", "per_round_coarse"):
        return make_refresh_controller(method, calibration)
    pooled = method == "cgp_coarse"
    obj = CGPController("cgp" if pooled else "per_round_frontier")
    obj.method = method
    obj.algorithm = CGPCoarseSpecStop(prefix_pool=pooled,
                                    estimator="ratio" if pooled else "per_round")
    return obj


if __name__ == "__main__":
    main(controller_factory=make_coarse_controller, cold_methods=COARSE_METHODS,
         warm_methods=WARM_METHODS)
