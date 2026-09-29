"""Experimental ratio learners and reusable, non-permanent arm screening.

``residual`` is an empirical standard-error heuristic, not a confidence
sequence. ``bounded`` uses empirical Bernstein intervals with a summable
allocation over sample counts. Its statistical interpretation additionally
requires stationary, history-independent observations and an externally
justified complete-round cost bound. This module does not certify those laws.

Prefix sharing additionally requires coupled, nested draft outcomes whose
position-wise survival laws do not depend on the selected arm or history.
It shares yield observations only; shorter-arm costs are never reconstructed.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def confidence_candidates(
    lower: Sequence[float],
    upper: Sequence[float],
    known_cost: Optional[Sequence[bool]] = None,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Keep arms not strictly dominated by a current upper confidence bound.

    Bounds must describe the same objective. Unknown-cost arms are retained;
    the returned mask is a fresh value, never a permanent elimination record.
    """
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.ndim != 1 or not lo.size or hi.shape != lo.shape:
        raise ValueError("lower and upper must be nonempty equal-length vectors")
    if np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any():
        raise ValueError("confidence intervals must be ordered and contain no NaN")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    keep = lo <= float(np.min(hi)) + tolerance
    if known_cost is not None:
        known = np.asarray(known_cost, dtype=bool)
        if known.shape != lo.shape:
            raise ValueError("known_cost must match the bound vectors")
        keep |= ~known
    return keep


def prefix_shape_intervals(
    lower: Sequence[float], upper: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Intersect marginal intervals with q_1 >= ... >= q_K.

    Inconsistent input intervals are returned unchanged with conflict=True.
    This fail-open behavior matters for empirical, non-guaranteed intervals.
    """
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.ndim != 1 or not lo.size or hi.shape != lo.shape:
        raise ValueError("prefix bounds must be equal-length nonempty vectors")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("prefix bounds must be finite")
    if (lo < 0).any() or (hi > 1).any() or (lo > hi).any():
        raise ValueError("prefix intervals must lie in [0, 1]")
    shaped_lo = np.maximum.accumulate(lo[::-1])[::-1]
    shaped_hi = np.minimum.accumulate(hi)
    conflict = bool(np.any(shaped_lo > shaped_hi))
    if conflict:
        return lo.copy(), hi.copy(), True
    return shaped_lo, shaped_hi, False


def bounded_mean_interval(
    mean: float,
    sample_variance: float,
    count: int,
    lower: float,
    upper: float,
    error_probability: float,
) -> Tuple[float, float]:
    """Two-sided empirical Bernstein interval for bounded i.i.d. samples.

    sample_variance uses denominator n-1. The log(4/error_probability)
    accommodates both tails of the bounded empirical Bernstein inequality.
    With fewer than two samples, use the entire supplied support.
    """
    if not all(math.isfinite(v) for v in (lower, upper, error_probability)):
        raise ValueError("support and error probability must be finite")
    if lower > upper or not 0 < error_probability < 1:
        raise ValueError("invalid support or error probability")
    if isinstance(count, bool) or int(count) != count or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if count < 2:
        return float(lower), float(upper)
    if not math.isfinite(mean) or not math.isfinite(sample_variance):
        raise ValueError("sample moments must be finite")
    if not lower <= mean <= upper or sample_variance < 0:
        raise ValueError("sample moments violate the supplied support")
    log_term = math.log(4.0 / error_probability)
    radius = math.sqrt(2.0 * sample_variance * log_term / count)
    radius += 7.0 * (upper - lower) * log_term / (3.0 * (count - 1))
    return max(lower, mean - radius), min(upper, mean + radius)


class StructuredRatioBandit:
    """Select the smallest current lower-bound index among candidate arms.

    Equal indices prefer fewer own cost observations, then the shorter arm.
    exploration_probability samples allowed arms, including dynamically
    screened arms. An enabled cost_tolerance rule takes priority: select the
    least upper-bound arm if its upper bound minus the smallest allowed lower
    bound is at most the tolerance. Only bounded-mode joint-event coverage
    makes this a conditional certificate; residual mode remains heuristic.
    For mean_ratio, intervals estimate E[N/A]; prefix sharing is disallowed.

    allowed_arms is an externally supplied calibration restriction, not a
    claimed elimination certificate. All-arm warm-start data remain usable.
    Any near-optimal certificate is relative to this allowed set only.

    Without prefix sharing, residual mode uses paired Var(N-C_hat*A) for
    ratio estimates and sample Var(N/A) for mean_ratio. With prefix sharing,
    it uses separate empirical mean intervals for costs and prefix survival,
    since an own-pair residual variance cannot cover additional yield data.
    No residual-mode interval has a finite-sample coverage guarantee.
    """

    def __init__(
        self,
        k_max: int,
        estimator: str = "ratio",
        confidence: str = "residual",
        prefix_share: bool = False,
        prefix_compatible: bool = False,
        shape_constraints: bool = False,
        screening: bool = True,
        beta: float = 1.0,
        delta: float = 0.05,
        n_max: Optional[float] = None,
        exploration_probability: float = 0.0,
        seed: int = 0,
        cost_tolerance: Optional[float] = None,
        allowed_arms: Optional[Sequence[int]] = None,
    ) -> None:
        if isinstance(k_max, bool) or int(k_max) != k_max or k_max < 1:
            raise ValueError("k_max must be a positive integer")
        if estimator not in ("ratio", "mean_ratio"):
            raise ValueError("estimator must be ratio or mean_ratio")
        if confidence not in ("residual", "bounded"):
            raise ValueError("confidence must be residual or bounded")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not 0 < delta < 1 or not 0 <= exploration_probability <= 1:
            raise ValueError("delta must be in (0,1) and exploration_probability in [0,1]")
        if cost_tolerance is not None and (
            not math.isfinite(cost_tolerance) or cost_tolerance < 0
        ):
            raise ValueError("cost_tolerance must be None or finite and nonnegative")
        if n_max is not None and (not math.isfinite(n_max) or n_max <= 0):
            raise ValueError("n_max must be finite and positive")
        if confidence == "bounded" and (n_max is None or beta != 1.0):
            raise ValueError("bounded mode requires explicit n_max and beta=1")
        if prefix_share and not prefix_compatible:
            raise ValueError("prefix sharing requires explicit prefix_compatible capability")
        if prefix_share and estimator != "ratio":
            raise ValueError("prefix sharing cannot estimate the mean_ratio objective")
        if shape_constraints and not prefix_share:
            raise ValueError("prefix shape constraints require prefix sharing")
        self.k_max, self.estimator, self.confidence = int(k_max), estimator, confidence
        self.prefix_share, self.prefix_compatible = prefix_share, prefix_compatible
        self.shape_constraints, self.screening = shape_constraints, screening
        self.beta, self.delta, self.n_max = float(beta), float(delta), n_max
        self.exploration_probability = float(exploration_probability)
        self.cost_tolerance = cost_tolerance
        self._allowed_mask = np.ones(self.k_max, dtype=bool)
        if allowed_arms is not None:
            allowed = np.asarray(allowed_arms)
            if allowed.ndim != 1 or not allowed.size:
                raise ValueError("allowed_arms must be a nonempty arm list or boolean mask")
            if allowed.dtype == np.bool_:
                if allowed.shape != (self.k_max,):
                    raise ValueError("allowed_arms boolean mask must have length k_max")
                self._allowed_mask = allowed.copy()
            else:
                self._allowed_mask[:] = False
                for arm in allowed:
                    if (
                        not isinstance(arm, (int, np.integer))
                        or isinstance(arm, (bool, np.bool_))
                        or not 1 <= arm <= self.k_max
                    ):
                        raise ValueError("allowed_arms labels must be integers in 1..k_max")
                    self._allowed_mask[int(arm) - 1] = True
            if not self._allowed_mask.any():
                raise ValueError("allowed_arms must permit at least one arm")
        self.rng = np.random.default_rng(seed)
        self.t_k = np.zeros(self.k_max, dtype=np.int64)
        # Welford moments: complete cost, own-pair yield, and per-round ratio.
        self._means = np.zeros((self.k_max, 3), dtype=float)
        self._m2 = np.zeros((self.k_max, 3), dtype=float)
        self._cross_na = np.zeros(self.k_max, dtype=float)
        self.q_counts = np.zeros(self.k_max, dtype=np.int64)
        self.q_successes = np.zeros(self.k_max, dtype=np.int64)
        self.round_observations = 0

    def _validate_observation(self, k: int, n_t: float, a_t: float) -> None:
        if isinstance(k, bool) or int(k) != k or not 1 <= k <= self.k_max:
            raise ValueError("arm must be an integer in 1..k_max")
        if not math.isfinite(n_t) or n_t < 0:
            raise ValueError("round cost must be finite and nonnegative")
        if not math.isfinite(a_t) or not 1 <= a_t <= k + 1:
            raise ValueError("committed yield must lie in [1, k+1]")
        if self.n_max is not None and n_t > self.n_max:
            raise ValueError("observed complete-round cost exceeds explicit n_max")
        if self.prefix_share and float(a_t) != int(a_t):
            raise ValueError("prefix observations require integer committed yields")

    def _observe_arm(self, k: int, n_t: float, a_t: float) -> None:
        idx = k - 1
        values = np.array([n_t, a_t, n_t / a_t], dtype=float)
        delta = values - self._means[idx]
        self.t_k[idx] += 1
        self._means[idx] += delta / self.t_k[idx]
        remainder = values - self._means[idx]
        self._m2[idx] += delta * remainder
        self._cross_na[idx] += delta[0] * remainder[1]

    def _observe_prefix(self, k: int, a_t: float) -> None:
        if self.prefix_share:
            self.q_counts[:k] += 1
            self.q_successes[:k] += a_t >= np.arange(2, k + 2)

    def update(self, k: int, n_t: float, a_t: float) -> None:
        """Observe only one selected arm from one new round."""
        self._validate_observation(k, n_t, a_t)
        k = int(k)
        self._observe_arm(k, float(n_t), float(a_t))
        self._observe_prefix(k, a_t)
        self.round_observations += 1

    def warm_start(self, costs: np.ndarray, yields: np.ndarray) -> None:
        """Use explicit calibration matrices with shape (paired cells, K).

        Each arm receives its measured pair. Each paired cell contributes
        prefix information exactly once, from its largest arm. Validate the
        entire input before mutation, including cross-arm nesting when sharing.
        """
        costs, yields = np.asarray(costs, dtype=float), np.asarray(yields, dtype=float)
        if costs.ndim != 2 or costs.shape[1] != self.k_max or yields.shape != costs.shape:
            raise ValueError("warm-start arrays must both have shape (n, k_max)")
        for row in range(len(costs)):
            for idx in range(self.k_max):
                self._validate_observation(idx + 1, costs[row, idx], yields[row, idx])
            if self.prefix_share:
                expected = np.minimum(yields[row, -1], np.arange(2, self.k_max + 2))
                if not np.array_equal(yields[row], expected):
                    raise ValueError("warm-start paired cell violates prefix nesting")
        for row in range(len(costs)):
            for idx in range(self.k_max):
                self._observe_arm(idx + 1, costs[row, idx], yields[row, idx])
            self._observe_prefix(self.k_max, yields[row, -1])
        self.round_observations += len(costs)

    def _mean_interval(
        self, mean: float, variance: float, count: int,
        lower: float, upper: float, t: int,
    ) -> Tuple[float, float]:
        if count < 2:
            return lower, upper
        if self.confidence == "bounded":
            # At most 4K scalar sequences; sum_n 6/(pi^2*n^2) = 1.
            error = self.delta * 6.0 / (math.pi**2 * count**2 * 4 * self.k_max)
            return bounded_mean_interval(mean, variance, count, lower, upper, error)
        radius = self.beta * math.sqrt(2.0 * math.log(max(t, 2)) * variance / count)
        return max(lower, mean - radius), min(upper, mean + radius)

    def snapshot(self, t: int) -> dict:
        """Return detached diagnostics without modifying statistics or RNG."""
        if isinstance(t, bool) or int(t) != t or t < 1:
            raise ValueError("t must be a positive integer")
        t = int(t)
        estimates = np.full(self.k_max, np.inf)
        lower = np.zeros(self.k_max)
        upper = np.full(self.k_max, np.inf)
        yield_mean = self._means[:, 1].copy()
        yield_lower = np.ones(self.k_max)
        yield_upper = np.arange(2, self.k_max + 2, dtype=float)
        q_mean = np.full(self.k_max, 0.5)
        q_lo, q_hi = np.zeros(self.k_max), np.ones(self.k_max)
        cost_upper = float(self.n_max) if self.n_max is not None else math.inf
        shape_conflict = False
        if self.prefix_share:
            for idx, count in enumerate(self.q_counts):
                if count:
                    q_mean[idx] = self.q_successes[idx] / count
                    variance = q_mean[idx] * (1 - q_mean[idx]) * count / (count - 1) if count > 1 else 0.0
                    q_lo[idx], q_hi[idx] = self._mean_interval(
                        q_mean[idx], variance, int(count), 0.0, 1.0, t
                    )
            raw_q_lo, raw_q_hi = q_lo.copy(), q_hi.copy()
            if self.shape_constraints:
                q_lo, q_hi, shape_conflict = prefix_shape_intervals(q_lo, q_hi)
            yield_mean = 1.0 + np.cumsum(q_mean)
            yield_lower, yield_upper = 1.0 + np.cumsum(q_lo), 1.0 + np.cumsum(q_hi)
        else:
            raw_q_lo, raw_q_hi = q_lo.copy(), q_hi.copy()
        for idx, count in enumerate(self.t_k):
            if not count:
                continue
            mean_n, mean_a, mean_r = self._means[idx]
            var_n, var_a, var_r = self._m2[idx] / (count - 1) if count > 1 else (0.0, 0.0, 0.0)
            if self.estimator == "mean_ratio":
                estimates[idx] = mean_r
                lower[idx], upper[idx] = self._mean_interval(
                    mean_r, max(0.0, var_r), int(count), 0.0, cost_upper, t
                )
            elif self.confidence == "residual" and not self.prefix_share:
                estimates[idx] = mean_n / mean_a
                if count < 2:
                    continue
                ratio = estimates[idx]
                residual_var = max(
                    0.0,
                    var_n + ratio**2 * var_a
                    - 2 * ratio * self._cross_na[idx] / (count - 1),
                )
                se_variance = residual_var / mean_a**2
                lower[idx], upper[idx] = self._mean_interval(
                    ratio, se_variance, int(count), 0.0, cost_upper, t
                )
            else:
                n_lo, n_hi = self._mean_interval(
                    mean_n, max(0.0, var_n), int(count), 0.0, cost_upper, t
                )
                if not self.prefix_share:
                    yield_lower[idx], yield_upper[idx] = self._mean_interval(
                        mean_a, max(0.0, var_a), int(count), 1.0, float(idx + 2), t
                    )
                estimates[idx] = mean_n / yield_mean[idx]
                lower[idx] = n_lo / yield_upper[idx]
                upper[idx] = n_hi / yield_lower[idx]
        allowed_indices = np.flatnonzero(self._allowed_mask)
        mask = self._allowed_mask.copy()
        if self.screening:
            mask[allowed_indices] = confidence_candidates(
                lower[allowed_indices], upper[allowed_indices],
                self.t_k[allowed_indices] > 0,
            )
        best_upper_idx = min(
            allowed_indices,
            key=lambda idx: (upper[idx], int(self.t_k[idx]), int(idx)),
        )
        upper_gap = float(upper[best_upper_idx] - np.min(lower[allowed_indices]))
        near_optimal = (
            self.cost_tolerance is not None
            and math.isfinite(upper_gap)
            and upper_gap <= self.cost_tolerance
        )
        if self.confidence == "bounded":
            interval_kind = "bounded_empirical_bernstein_separate_means"
        elif self.prefix_share:
            interval_kind = "heuristic_separate_cost_and_prefix_standard_errors"
        else:
            interval_kind = (
                "heuristic_paired_residual_standard_error"
                if self.estimator == "ratio"
                else "heuristic_mean_ratio_standard_error"
            )
        return {
            "t": t, "estimator": self.estimator, "confidence": self.confidence,
            "interval_kind": interval_kind, "beta": self.beta,
            "bounds_require_external_cost_cap_and_stationary_laws": self.confidence == "bounded",
            "n_max": self.n_max, "counts": self.t_k.tolist(),
            "exploration_probability": self.exploration_probability,
            "allowed_mask": self._allowed_mask.tolist(),
            "allowed_arms": (allowed_indices + 1).tolist(),
            "near_optimal": {
                "enabled": self.cost_tolerance is not None,
                "cost_tolerance": self.cost_tolerance,
                "best_upper_arm": int(best_upper_idx) + 1,
                "upper_minus_min_lower": upper_gap,
                "criterion_met": bool(near_optimal),
                "certified_on_bounded_joint_event": bool(
                    near_optimal and self.confidence == "bounded"
                ),
                "scope": "allowed_arm_set",
                "interpretation": (
                    "conditional_on_bounded_joint_event_and_model_assumptions"
                    if self.confidence == "bounded"
                    else "heuristic_only_no_coverage_certificate"
                ),
            },
            "round_observations": self.round_observations,
            "cost_means": self._means[:, 0].tolist(), "yield_means": yield_mean.tolist(),
            "estimates": estimates.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
            "indices": lower.tolist(), "candidate_mask": mask.tolist(),
            "candidates": (np.flatnonzero(mask) + 1).tolist(),
            "prefix_counts": self.q_counts.tolist(), "prefix_successes": self.q_successes.tolist(),
            "prefix_estimates": q_mean.tolist(), "prefix_lower_raw": raw_q_lo.tolist(),
            "prefix_upper_raw": raw_q_hi.tolist(), "prefix_lower": q_lo.tolist(),
            "prefix_upper": q_hi.tolist(), "shape_conflict": shape_conflict,
        }

    def select_arm(self, t: int) -> int:
        state = self.snapshot(t)
        if state["near_optimal"]["criterion_met"]:
            return state["near_optimal"]["best_upper_arm"]
        if (
            self.exploration_probability
            and self.rng.random() < self.exploration_probability
        ):
            return int(self.rng.choice(state["allowed_arms"]))
        return min(
            state["candidates"],
            key=lambda arm: (state["indices"][arm - 1], int(self.t_k[arm - 1]), arm),
        )
