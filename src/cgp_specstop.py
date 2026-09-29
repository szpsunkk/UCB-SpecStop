"""Empirical prefix pooling and expanding-frontier draft selection.

CGP matches the supplied 2026-09-22 design with beta=.10 and lambda=1.
The relative bonus is a heuristic, not a confidence sequence. Population
single crossing alone does not certify elimination with noisy estimates.
Sharing requires a common potential prefix law and complete-round feedback;
it never supplies unobserved costs or unseen deeper prefix outcomes.
"""

from __future__ import annotations

import numpy as np


class CGPSpecStop:
    def __init__(self, k_max=10, beta=0.10, lam=1.0,
                 prefix_pool=True, frontier=True, estimator="ratio"):
        if type(k_max) is not int or k_max < 1:
            raise ValueError("k_max must be a positive integer")
        if not np.isfinite(beta) or beta < 0 or not np.isfinite(lam) or lam <= 0:
            raise ValueError("invalid exploration parameters")
        if estimator not in ("ratio", "per_round"):
            raise ValueError("unknown estimator")
        self.k_max, self.beta, self.lam = k_max, float(beta), float(lam)
        self.prefix_pool, self.frontier, self.estimator = prefix_pool, frontier, estimator
        self.hi = 1 if frontier else k_max
        self.sum_n = np.zeros(k_max)
        self.sum_a = np.zeros(k_max)
        self.sum_ratio = np.zeros(k_max)
        self.t_k = np.zeros(k_max)
        self.y_hit = np.zeros(k_max)
        self.y_cnt = np.zeros(k_max)

    def estimates(self):
        played = self.t_k > 0
        raw_n = self.sum_n / np.maximum(self.t_k, 1)
        optimistic_n = raw_n[played].min() if played.any() else 0.0
        mean_n = np.where(played, raw_n, optimistic_n)
        q = np.where(self.y_cnt > 0, self.y_hit / np.maximum(self.y_cnt, 1), 0.5)
        mean_a = (1 + np.cumsum(q) if self.prefix_pool else
                  np.where(played, self.sum_a / np.maximum(self.t_k, 1), 1.0))
        if self.estimator == "per_round":
            raw_ratio = self.sum_ratio / np.maximum(self.t_k, 1)
            c = np.where(played, raw_ratio, mean_n / mean_a)
        else:
            c = mean_n / mean_a
        return c

    def indices(self, t):
        c = self.estimates()
        return c - self.beta * c * np.sqrt(np.log(max(t, 2)) / (self.t_k + self.lam))

    def select_arm(self, t):
        if self.frontier:
            if self.t_k[0] == 0:
                return 1
            idx = self.indices(t)
            best = int(np.argmin(idx[:self.hi])) + 1
            if best == self.hi and self.hi < self.k_max:
                self.hi += 1
                return self.hi
            return best
        unseen = np.flatnonzero(self.t_k == 0)
        return int(unseen[0]) + 1 if len(unseen) else int(np.argmin(self.indices(t))) + 1

    def update(self, k, n_t, a_t):
        if int(k) != k or not 1 <= k <= self.k_max:
            raise ValueError("invalid physical draft length")
        if not np.isfinite(n_t) or n_t < 0 or not np.isfinite(a_t):
            raise ValueError("feedback must be finite with nonnegative cost")
        if int(a_t) != a_t or not 1 <= a_t <= k + 1:
            raise ValueError("feedback must describe a complete prefix-plus-bonus round")
        k = int(k)
        self.sum_n[k - 1] += n_t
        self.sum_a[k - 1] += a_t
        self.sum_ratio[k - 1] += n_t / a_t
        self.t_k[k - 1] += 1
        self.y_cnt[:k] += 1
        self.y_hit[:k] += np.arange(1, k + 1) < a_t

    def snapshot(self, t=2):
        return {"hi": self.hi, "beta": self.beta, "lambda": self.lam,
                "prefix_pool": self.prefix_pool, "frontier": self.frontier,
                "estimator": self.estimator, "cost_count": self.t_k.tolist(),
                "prefix_count": self.y_cnt.tolist(), "prefix_hit": self.y_hit.tolist(),
                "cost_estimates": self.estimates().tolist(),
                "heuristic_indices": self.indices(t).tolist(),
                "guarantee": "none inherited from the original LCB theorem"}
