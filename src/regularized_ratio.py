"""Finite-sample interpolation between two empirical ratio objectives.

This experimental policy interpolates toward the mean per-round ratio at
small arm-specific sample counts, then approaches the ratio-of-sums target.
It uses the original SN exploration term. Neither the interpolated center
nor the exploration index is a certified confidence bound, and this module
provides no regret guarantee or unbiasedness claim.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Tuple

import numpy as np


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_finite(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        or not math.isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


class RegularizedRatioBandit:
    """Use a count-dependent interpolation with the existing SN exploration.

    For an observed arm with n selected-arm pairs, let R=sum(N)/sum(A),
    M=mean(N/A), and w=s/(n+s), where s is prior_strength. The selection
    index is R+w*(M-R)-beta*sqrt(n*log(t))/sum(A). At s=0 the arithmetic
    agrees with UCBSpecStop. At fixed finite s, w tends to zero as n grows.

    Despite its name, prior_strength supplies no prior data or oracle mean.
    No unselected-arm outcome or future count enters these statistics. The
    interpolation can move the center in either direction; it is not a
    universal penalty for long arms. Unknown arms are sampled once each,
    shortest first. Equal finite indices also prefer the shorter arm.
    """

    def __init__(
        self, k_max: int, prior_strength: float = 0.0, beta: float = 1.0
    ) -> None:
        self.k_max = _positive_integer(k_max, "k_max")
        self.prior_strength = _nonnegative_finite(prior_strength, "prior_strength")
        self.beta = _nonnegative_finite(beta, "beta")
        self.t_k = np.zeros(self.k_max, dtype=np.int64)
        self.s_n = np.zeros(self.k_max, dtype=float)
        self.s_a = np.zeros(self.k_max, dtype=float)
        self.sum_ratio = np.zeros(self.k_max, dtype=float)
        self.round_observations = 0

    def _validate_observation(self, k: int, n_t: float, a_t: float) -> Tuple[int, float, float]:
        arm = _positive_integer(k, "arm")
        if arm > self.k_max:
            raise ValueError("arm must lie in 1..k_max")
        cost = _nonnegative_finite(n_t, "round cost")
        token_yield = _nonnegative_finite(a_t, "committed yield")
        if not 1.0 <= token_yield <= arm + 1:
            raise ValueError("committed yield must lie in [1, k+1]")
        return arm, cost, token_yield

    def _check_cumulative_finite(self, costs: np.ndarray, yields: np.ndarray) -> None:
        with np.errstate(over="ignore"):
            cumulative_costs = self.s_n + costs.sum(axis=0)
            cumulative_yields = self.s_a + yields.sum(axis=0)
        if not np.isfinite(cumulative_costs).all() or not np.isfinite(cumulative_yields).all():
            raise ValueError("observations would overflow cumulative statistics")

    def _observe(self, arm: int, cost: float, token_yield: float) -> None:
        index = arm - 1
        self.t_k[index] += 1
        self.s_n[index] += cost
        self.s_a[index] += token_yield
        self.sum_ratio[index] += cost / token_yield

    def update(self, k: int, n_t: float, a_t: float) -> None:
        """Add one observed selected-arm pair, without inferring other arms."""

        arm, cost, token_yield = self._validate_observation(k, n_t, a_t)
        with np.errstate(over="ignore"):
            new_cost = self.s_n[arm - 1] + cost
            new_yield = self.s_a[arm - 1] + token_yield
        if not math.isfinite(new_cost) or not math.isfinite(new_yield):
            raise ValueError("observation would overflow cumulative statistics")
        self._observe(arm, cost, token_yield)
        self.round_observations += 1

    def warm_start(self, costs: np.ndarray, yields: np.ndarray) -> None:
        """Add complete calibration tables with shape (paired cells, K).

        Every arm receives one own observation per cell; its count increases
        by the number of cells, not by K times that number. Calibration cells
        are supplied explicitly by the caller. No future evaluation table is
        consulted. Validate the complete batch before changing any statistics.
        Prefix nesting is unnecessary because this policy shares no feedback.
        """

        costs = np.asarray(costs, dtype=float)
        yields = np.asarray(yields, dtype=float)
        if costs.ndim != 2 or costs.shape[1] != self.k_max or yields.shape != costs.shape:
            raise ValueError("warm-start arrays must both have shape (n, k_max)")
        if not np.isfinite(costs).all() or np.any(costs < 0):
            raise ValueError("round costs must be finite and nonnegative")
        if (
            not np.isfinite(yields).all() or np.any(yields < 1)
            or np.any(yields > np.arange(2, self.k_max + 2))
        ):
            raise ValueError("committed yields must lie in [1, k+1]")
        self._check_cumulative_finite(costs, yields)
        for row_n, row_a in zip(costs, yields):
            for index in range(self.k_max):
                self._observe(index + 1, float(row_n[index]), float(row_a[index]))
        self.round_observations += len(costs)

    def _estimates(self, t: int) -> Tuple[np.ndarray, ...]:
        known = self.t_k > 0
        ratio = np.full(self.k_max, np.inf)
        mean_ratio = np.full(self.k_max, np.inf)
        center = np.full(self.k_max, np.inf)
        widths = np.full(self.k_max, np.inf)
        weights = np.zeros(self.k_max)
        if self.prior_strength > 0:
            weights = self.prior_strength / (self.t_k + self.prior_strength)
        ratio[known] = self.s_n[known] / self.s_a[known]
        mean_ratio[known] = self.sum_ratio[known] / self.t_k[known]
        center[known] = ratio[known] + weights[known] * (mean_ratio[known] - ratio[known])
        # Use the original SN arithmetic, rather than an algebraically equal
        # rearrangement, so prior_strength=0 preserves historical decisions.
        widths[known] = self.beta * np.sqrt(self.t_k[known] * np.log(t)) / self.s_a[known]
        indices = np.full(self.k_max, -np.inf)
        indices[known] = center[known] - widths[known]
        return ratio, mean_ratio, weights, center, widths, indices

    def select_arm(self, t: int) -> int:
        """Choose from past observations at a positive caller-supplied round."""

        t = _positive_integer(t, "t")
        unknown = np.flatnonzero(self.t_k == 0)
        if len(unknown):
            return int(unknown[0] + 1)
        return int(np.argmin(self._estimates(t)[-1]) + 1)

    def snapshot(self, t: int) -> dict:
        """Return detached diagnostics; inspection does not advance any count."""

        t = _positive_integer(t, "t")
        ratio, mean_ratio, weights, center, widths, indices = self._estimates(t)
        return {
            "counts": self.t_k.tolist(),
            "cost_sums": self.s_n.tolist(),
            "yield_sums": self.s_a.tolist(),
            "ratio_estimates": ratio.tolist(),
            "mean_ratio_estimates": mean_ratio.tolist(),
            "regularization_weights": weights.tolist(),
            "estimates": center.tolist(),
            "exploration_widths": widths.tolist(),
            "indices": indices.tolist(),
            "round_observations": self.round_observations,
            "prior_strength": self.prior_strength,
            "beta": self.beta,
        }
