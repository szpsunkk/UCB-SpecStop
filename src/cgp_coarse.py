"""Exploratory coarse-to-fine search with the existing CGP estimators.

Probe a fixed geometric grid, then unmeasured neighbors of the best observed
length. This pays fewer initialization rounds than a full sweep, while probing
long lengths even when a short local comparison is unfavorable. It remains
an empirical search heuristic, not a certificate for arbitrary cost curves.
"""

from __future__ import annotations

import numpy as np

from src.cgp_specstop import CGPSpecStop


class CGPCoarseSpecStop(CGPSpecStop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        grid, k = [], 1
        while k < self.k_max:
            grid.append(k)
            k *= 2
        self.grid = tuple(grid + [self.k_max])

    def select_arm(self, t):
        for k in self.grid:
            if self.t_k[k - 1] == 0:
                self.hi = max(self.hi, k)
                return k
        played = np.flatnonzero(self.t_k > 0)
        best = int(played[np.argmin(self.indices(t)[played])]) + 1
        for k in (best - 1, best + 1):
            if 1 <= k <= self.k_max and self.t_k[k - 1] == 0:
                self.hi = max(self.hi, k)
                return k
        return best
