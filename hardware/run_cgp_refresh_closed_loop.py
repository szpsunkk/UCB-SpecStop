"""CGP boundary-refresh development comparison with the same request runtime."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware.cgp_controller import CGPController, COLD_METHODS, WARM_METHODS, make_controller
from src.cgp_refresh import CGPRefreshSpecStop
from hardware.run_cgp_closed_loop import main


METHODS = tuple(COLD_METHODS) + ("cgp_refresh", "per_round_refresh")


def make_refresh_controller(method, calibration):
    if method not in ("cgp_refresh", "per_round_refresh"):
        return make_controller(method, calibration)
    obj = CGPController("cgp" if method == "cgp_refresh" else "per_round_frontier")
    obj.method = method
    obj.algorithm = CGPRefreshSpecStop(prefix_pool=method == "cgp_refresh",
                                      estimator="ratio" if method == "cgp_refresh" else "per_round")
    return obj


if __name__ == "__main__":
    main(controller_factory=make_refresh_controller, cold_methods=METHODS, warm_methods=WARM_METHODS)
