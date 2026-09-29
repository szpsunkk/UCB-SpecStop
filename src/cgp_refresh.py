"""Diagnostic CGP variant that rechecks an under-observed frontier.

The boundary receives at least about sqrt(t)/2 real cost observations while
it remains active. This is a heuristic rescue for an unlucky early sample,
not confidence-based elimination and not an inherited LCB regret result.
"""

from __future__ import annotations

import numpy as np

from src.cgp_specstop import CGPSpecStop


class CGPRefreshSpecStop(CGPSpecStop):
    def select_arm(self, t):
        if self.t_k[0] > 0 and self.frontier and self.t_k[self.hi - 1] < np.sqrt(t) / 2:
            return self.hi
        return super().select_arm(t)
