"""Calibrated prefix-yield controllers for finite token budgets.

The Bellman rule is optimal only for the supplied stationary renewal model.
Learned or request-local versions are empirical controllers: the original
UCB-SpecStop regret bound does not apply. Arm zero is one actual empty-draft
verification RPC, never a substitute for streaming target-only generation.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Optional, Sequence

import numpy as np


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def prefix_probabilities(survival: Sequence[float], arm: int) -> np.ndarray:
    """Return P(A=a), a=1..arm+1, from a monotone survival vector."""
    q = np.asarray(survival, dtype=float)
    arm = _integer(arm, "arm")
    if q.ndim != 1 or arm > len(q) or not np.isfinite(q).all():
        raise ValueError("invalid survival vector or arm")
    if (q < 0).any() or (q > 1).any() or (np.diff(q) > 0).any():
        raise ValueError("survival must be nonincreasing within [0,1]")
    if arm == 0:
        return np.ones(1)
    return np.concatenate(([1.0 - q[0]], q[:arm - 1] - q[1:arm], [q[arm - 1]]))


def renewal_table(
    costs: Sequence[float],
    survival: Sequence[float],
    horizon: int,
    allowed_arms: Optional[Sequence[int]] = None,
    zero_terminal_only: bool = True,
) -> tuple:
    """Exact finite-budget DP for the provided cost/yield model, O(H K^2).

    costs[k] is the measured mean for actual physical depth k. The selected
    arm never exceeds remaining_tokens-1. Zero must be available to finish
    the last token; by default it is unavailable in other states to preserve
    the original positive-depth action set away from the boundary.
    """
    n = np.asarray(costs, dtype=float)
    q = np.asarray(survival, dtype=float)
    horizon = _integer(horizon, "horizon")
    if n.ndim != 1 or not len(n) or not np.isfinite(n).all() or (n < 0).any():
        raise ValueError("costs must be a finite nonnegative vector")
    if q.shape != (len(n) - 1,):
        raise ValueError("costs must contain zero plus every survival depth")
    probabilities = [prefix_probabilities(q, k) for k in range(len(n))]
    arms = tuple(range(len(n))) if allowed_arms is None else tuple(allowed_arms)
    if not arms or len(arms) != len(set(arms)):
        raise ValueError("allowed arms must be nonempty and unique")
    arms = tuple(sorted(_integer(k, "allowed arm") for k in arms))
    if arms[0] != 0 or arms[-1] >= len(n):
        raise ValueError("allowed arms must include zero and lie in 0..K")
    if zero_terminal_only and horizon >= 2 and 1 not in arms:
        raise ValueError("terminal-only zero requires depth one for all horizons")
    values = np.zeros(horizon + 1)
    choices = np.zeros(horizon + 1, dtype=np.int64)
    for remaining in range(1, horizon + 1):
        available = [k for k in arms if k < remaining and (
            not zero_terminal_only or remaining == 1 or k > 0
        )]
        best_value, best_arm = math.inf, None
        for arm in available:
            future = values[remaining - arm - 1:remaining][::-1]
            value = float(n[arm] + probabilities[arm].dot(future))
            if value < best_value:
                best_value, best_arm = value, arm
        if best_arm is None:
            raise ValueError("no feasible arm for remaining token budget")
        values[remaining], choices[remaining] = best_value, best_arm
    return values, choices


def solve_position_horizon(
    costs_by_position: np.ndarray,
    survival_by_position: np.ndarray,
    cap: int,
    allowed_arms: Optional[Sequence[int]] = None,
    zero_terminal_only: bool = True,
) -> tuple:
    """Finite-cap Bellman table indexed by already delivered token count.

    Row x contains N_x(k), k=0..K, and q_x(j), j=1..K. Values[cap]=0;
    choices[x] gives the physical draft depth at delivered position x. This
    explicitly models policy-dependent visits to output positions instead of
    averaging statewise ratios. The result is exact only for the supplied
    position-dependent Markov renewal model, not arbitrary prompt histories.
    """
    cap = _integer(cap, "cap", 1)
    n, q = np.asarray(costs_by_position, dtype=float), np.asarray(survival_by_position, dtype=float)
    if n.ndim != 2 or n.shape[0] != cap or n.shape[1] < 1:
        raise ValueError("position costs need shape (cap,K+1)")
    if q.shape != (cap, n.shape[1] - 1):
        raise ValueError("position survival needs shape (cap,K)")
    if not np.isfinite(n).all() or (n < 0).any():
        raise ValueError("position costs must be finite and nonnegative")
    # Reuse all feasibility and physical probability checks on each row.
    # The row-zero call also validates arbitrary action subsets.
    renewal_table(n[0], q[0], cap, allowed_arms, zero_terminal_only)
    arms = tuple(range(n.shape[1])) if allowed_arms is None else tuple(sorted(allowed_arms))
    probabilities = [
        [prefix_probabilities(row, arm) for arm in range(n.shape[1])]
        for row in q
    ]
    values = np.zeros(cap + 1)
    choices = np.zeros(cap + 1, dtype=np.int64)
    for position in range(cap - 1, -1, -1):
        remaining = cap - position
        available = [k for k in arms if k < remaining and (
            not zero_terminal_only or remaining == 1 or k > 0
        )]
        best_value, best_arm = math.inf, None
        for arm in available:
            future = values[position + 1:position + arm + 2]
            value = float(n[position, arm] + probabilities[position][arm].dot(future))
            if value < best_value:
                best_value, best_arm = value, arm
        if best_arm is None:
            raise ValueError("no feasible arm for position")
        values[position], choices[position] = best_value, best_arm
    return values, choices


def interpolate_position_curves(
    offsets: Sequence[int],
    mean_costs: np.ndarray,
    survival: np.ndarray,
    cap: int,
) -> tuple:
    """Interpolate calibrated anchor curves, holding endpoint values outside.

    mean_costs includes measured k=0 in column zero. Convex interpolation of
    valid survival vectors preserves nonnegative PMFs. Shrinkage and prompt
    splits belong to the fitter: this function never fits or reads outcomes.
    """
    cap = _integer(cap, "cap", 1)
    anchors = np.asarray([_integer(x, "offset") for x in offsets], dtype=int)
    n, q = np.asarray(mean_costs, dtype=float), np.asarray(survival, dtype=float)
    if not len(anchors) or (np.diff(anchors) <= 0).any():
        raise ValueError("offsets must be nonempty and strictly increasing")
    if n.ndim != 2 or n.shape[0] != len(anchors) or not n.shape[1]:
        raise ValueError("anchor costs need shape (anchors,K+1)")
    if q.shape != (len(anchors), n.shape[1] - 1):
        raise ValueError("anchor survival needs shape (anchors,K)")
    if not np.isfinite(n).all() or (n < 0).any():
        raise ValueError("anchor costs must be finite and nonnegative")
    for row in q:
        prefix_probabilities(row, len(row))
    positions = np.arange(cap)
    interpolated_costs = np.column_stack([
        np.interp(positions, anchors, n[:, col]) for col in range(n.shape[1])
    ])
    interpolated_q = np.empty((cap, q.shape[1]))
    for col in range(q.shape[1]):
        interpolated_q[:, col] = np.interp(positions, anchors, q[:, col])
    return interpolated_costs, interpolated_q


class PositionRatioController:
    """Frozen position curves with finite-cap or shared-lambda selection.

    Each calibration offset bin is shrunk toward global means with
    position_shrink pseudo-cells, then linearly interpolated. Offset denotes
    delivered output position, not prompt length. Costs for zero and positive
    depths must be measured at the same paired cells. No current target
    result, logits, or future request outcome enters selection.

    objective='ratio' is a shared-Dinkelbach surrogate fitted to the empirical
    calibration-offset distribution. Actual position visits depend on earlier
    actions, so this surrogate has no aggregate-optimality guarantee for the
    continuous process. objective='horizon' handles x -> x+A transitions in
    the supplied position Markov renewal model.
    """

    def __init__(
        self,
        costs: np.ndarray,
        yields: np.ndarray,
        zero_costs: Sequence[float],
        offsets: Sequence[int],
        position_shrink: float = 16.0,
        objective: str = "horizon",
        max_horizon: int = 128,
        allowed_arms: Optional[Sequence[int]] = None,
        zero_terminal_only: bool = True,
        online: bool = False,
    ) -> None:
        # Validation reuses the same paired-cell and prefix nesting contract.
        validated = RenewalRatioController(
            costs, yields, zero_costs, mode="frozen", max_horizon=1,
            allowed_arms=allowed_arms, zero_terminal_only=zero_terminal_only,
        )
        costs, yields = np.asarray(costs, dtype=float), np.asarray(yields, dtype=float)
        zero = np.asarray(zero_costs, dtype=float)
        positions = np.asarray([_integer(x, "offset") for x in offsets], dtype=int)
        if positions.shape != (len(costs),) or zero.shape != (len(costs),):
            raise ValueError("offsets and zero costs must be paired with every calibration cell")
        if isinstance(position_shrink, (bool, np.bool_)) or not math.isfinite(position_shrink) or position_shrink < 0:
            raise ValueError("position_shrink must be finite and nonnegative")
        if objective not in ("horizon", "ratio"):
            raise ValueError("position objective must be horizon or ratio")
        self.k_max = validated.k_max
        self.allowed_arms = validated.allowed_arms
        self.zero_terminal_only = validated.zero_terminal_only
        self.position_shrink = float(position_shrink)
        self.objective = objective
        self.online = bool(online)
        self.max_horizon = _integer(max_horizon, "max_horizon", 1)
        self.calibration_cells = len(costs)
        self.offsets = np.unique(positions)
        complete_costs = np.column_stack((zero, costs))
        # Largest-depth observation gives every q_j once per cell.
        prefix = yields[:, -1, None] >= np.arange(2, self.k_max + 2)
        global_costs, global_q = complete_costs.mean(axis=0), prefix.mean(axis=0)
        self.anchor_costs = np.zeros((len(self.offsets), self.k_max + 1))
        self.anchor_survival = np.zeros((len(self.offsets), self.k_max))
        self.offset_counts = np.zeros(len(self.offsets), dtype=np.int64)
        for index, offset in enumerate(self.offsets):
            mask = positions == offset
            count = int(mask.sum())
            self.offset_counts[index] = count
            weight = count / (count + self.position_shrink)
            self.anchor_costs[index] = weight * complete_costs[mask].mean(axis=0) + (1 - weight) * global_costs
            self.anchor_survival[index] = weight * prefix[mask].mean(axis=0) + (1 - weight) * global_q
        prior_count = self.offset_counts.astype(float) + self.position_shrink
        self.cost_effective_counts = np.repeat(prior_count[:, None], self.k_max + 1, axis=1)
        self.online_cost_counts = np.zeros_like(self.cost_effective_counts, dtype=np.int64)
        risk = np.column_stack((np.ones(len(self.offsets)), self.anchor_survival[:, :-1]))
        self.prefix_trials = prior_count[:, None] * risk
        self.prefix_successes = prior_count[:, None] * self.anchor_survival
        self.online_prefix_counts = np.zeros(len(self.offsets), dtype=np.int64)
        self.online_outcome_updates = 0
        self.skipped_eos_updates = 0
        self._pending_selection = None
        self.shared_lambda = self._fit_shared_lambda()
        self._tables = {}
        self._active_cap = None
        self.requests_started = 0
        self.decisions = 0
        self.table_recomputations = 0
        self._ensure_table(self.max_horizon)

    def _fit_shared_lambda(self) -> float:
        arms = np.array([k for k in self.allowed_arms if k > 0 or not self.zero_terminal_only], dtype=int)
        if not len(arms):
            arms = np.array([0], dtype=int)
        n = self.anchor_costs[:, arms]
        b = np.column_stack((np.ones(len(self.offsets)), 1 + np.cumsum(self.anchor_survival, axis=1)))[:, arms]
        weights = self.offset_counts / self.offset_counts.sum()
        value = float(np.min(weights.dot(n) / weights.dot(b)))
        rows = np.arange(len(self.offsets))
        for _ in range(100):
            chosen = np.argmin(n - value * b, axis=1)
            updated = float(weights.dot(n[rows, chosen]) / weights.dot(b[rows, chosen]))
            if abs(updated - value) <= 1e-12 * max(1., abs(value)):
                return updated
            value = updated
        raise ArithmeticError("shared Dinkelbach fit did not converge")

    def _ensure_table(self, cap: int) -> None:
        if cap in self._tables:
            return
        n, q = interpolate_position_curves(self.offsets, self.anchor_costs, self.anchor_survival, cap)
        if self.objective == "horizon":
            values, choices = solve_position_horizon(n, q, cap, self.allowed_arms, self.zero_terminal_only)
        else:
            # Validate the same feasible action set before computing surrogate.
            renewal_table(n[0], q[0], cap, self.allowed_arms, self.zero_terminal_only)
            b = np.column_stack((np.ones(cap), 1 + np.cumsum(q, axis=1)))
            choices = np.zeros(cap + 1, dtype=np.int64)
            values = None
            for x in range(cap):
                available = [k for k in self.allowed_arms if k < cap - x and (
                    not self.zero_terminal_only or cap - x == 1 or k > 0
                )]
                choices[x] = min(available, key=lambda k: (n[x, k] - self.shared_lambda * b[x, k], k))
        self._tables[cap] = (values, choices)
        self.table_recomputations += 1

    def begin_request(self, token_cap: Optional[int] = None) -> None:
        self.requests_started += 1
        self._pending_selection = None
        self._active_cap = None if token_cap is None else _integer(token_cap, "token_cap", 1)
        if self._active_cap is not None:
            self._ensure_table(self._active_cap)

    def select_arm(self, remaining_tokens: int) -> int:
        remaining = _integer(remaining_tokens, "remaining_tokens", 1)
        if self._active_cap is None:
            self._active_cap = remaining
            self._ensure_table(remaining)
        if remaining > self._active_cap:
            raise ValueError("remaining tokens exceed active cap; call begin_request")
        self._ensure_table(self._active_cap)
        self.decisions += 1
        position = self._active_cap - remaining
        arm = int(self._tables[self._active_cap][1][position])
        self._pending_selection = (position, arm)
        return arm

    def update(self, executed_k: int, elapsed_ms: float, committed_yield: int, eos: bool = False) -> bool:
        """Optional selected-position update; ties use the smaller offset bin.

        No other bin's costs are changed. Prefix conditional hazards use the
        same right-censoring convention as RenewalRatioController. This
        nearest-bin adaptation is a heuristic for nonstationary content.
        """
        if not self.online:
            return False
        arm = _integer(executed_k, "executed_k")
        if arm > self.k_max:
            raise ValueError("executed_k exceeds supported depth")
        count = _integer(committed_yield, "committed_yield")
        if isinstance(elapsed_ms, (bool, np.bool_)) or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("elapsed cost must be finite and nonnegative")
        if self._pending_selection is None:
            raise ValueError("online feedback requires an unconsumed selection")
        position, selected_arm = self._pending_selection
        if arm > selected_arm:
            raise ValueError("executed depth cannot exceed selected depth")
        if eos:
            self.skipped_eos_updates += 1
            self._pending_selection = None
            return False
        if not 1 <= count <= arm + 1:
            raise ValueError("uncensored yield must lie in 1..executed_k+1")
        index = int(np.argmin(np.abs(self.offsets - position)))
        self.cost_effective_counts[index, arm] += 1
        self.online_cost_counts[index, arm] += 1
        self.anchor_costs[index, arm] += (
            elapsed_ms - self.anchor_costs[index, arm]
        ) / self.cost_effective_counts[index, arm]
        if arm:
            accepted = count - 1
            self.prefix_trials[index, :min(accepted + 1, arm)] += 1
            self.prefix_successes[index, :accepted] += 1
            hazards = np.divide(
                self.prefix_successes[index], self.prefix_trials[index],
                out=np.zeros(self.k_max), where=self.prefix_trials[index] > 0,
            )
            self.anchor_survival[index] = np.cumprod(hazards)
            self.online_prefix_counts[index] += 1
        self.online_outcome_updates += 1
        self._pending_selection = None
        self._tables.clear()
        if self.objective == "ratio":
            self.shared_lambda = self._fit_shared_lambda()
        return True

    def snapshot(self) -> dict:
        return {
            "objective": self.objective,
            "online": self.online,
            "position_shrink": self.position_shrink,
            "calibration_cells": self.calibration_cells,
            "offsets": self.offsets.tolist(),
            "offset_counts": self.offset_counts.tolist(),
            "anchor_costs": self.anchor_costs.tolist(),
            "anchor_survival": self.anchor_survival.tolist(),
            "shared_lambda": self.shared_lambda,
            "lambda_scope": "fixed_calibration_offset_distribution_surrogate",
            "active_cap": self._active_cap,
            "allowed_arms": list(self.allowed_arms),
            "zero_terminal_only": self.zero_terminal_only,
            "requests_started": self.requests_started,
            "decisions": self.decisions,
            "table_recomputations": self.table_recomputations,
            "online_outcome_updates": self.online_outcome_updates,
            "online_cost_counts": self.online_cost_counts.tolist(),
            "cost_effective_counts": self.cost_effective_counts.tolist(),
            "prefix_trials": self.prefix_trials.tolist(),
            "prefix_successes": self.prefix_successes.tolist(),
            "online_prefix_counts": self.online_prefix_counts.tolist(),
            "skipped_eos_updates": self.skipped_eos_updates,
            "guarantee": "supplied_position_Markov_renewal_model_only_no_inherited_regret_bound",
        }


class RenewalRatioController:
    """Frozen, globally learned, or request-local calibrated prefix policy.

    costs and yields contain one complete paired cell per row and depths
    1..K per column. zero_costs are genuine measurements of physical k=0.
    Each cell supplies K own-arm costs but only one observed latent prefix.

    mode='shared' retains conditional prefix-success counts across requests.
    mode='request_local' resets those counts to a fractional calibration prior
    at begin_request(), but retains online cost means across requests. The
    prior's total equivalent full-prefix count is local_prior_strength;
    deeper positions receive only their calibrated at-risk fraction.
    mode='frozen' makes no outcome updates.

    objective='ratio' ablates Bellman planning and uses N/B subject to the
    same remaining-token action constraint. Confidence/regret guarantees are
    not asserted. In particular, stationary prefix assumptions need not hold
    on policy-dependent continuous-request boundaries.
    """

    def __init__(
        self,
        costs: np.ndarray,
        yields: np.ndarray,
        zero_costs: Sequence[float],
        mode: str = "frozen",
        objective: str = "horizon",
        local_prior_strength: float = 8.0,
        max_horizon: int = 128,
        allowed_arms: Optional[Sequence[int]] = None,
        zero_terminal_only: bool = True,
    ) -> None:
        costs, yields = np.asarray(costs, dtype=float), np.asarray(yields, dtype=float)
        zero = np.asarray(zero_costs, dtype=float)
        if costs.ndim != 2 or not all(costs.shape) or yields.shape != costs.shape:
            raise ValueError("costs and yields need nonempty equal (cells,K) shape")
        if not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("calibration costs must be finite and nonnegative")
        if zero.ndim != 1 or not len(zero) or not np.isfinite(zero).all() or (zero < 0).any():
            raise ValueError("zero_costs require genuine finite nonnegative measurements")
        self.k_max = costs.shape[1]
        if not np.isfinite(yields).all() or (yields != np.floor(yields)).any():
            raise ValueError("calibration yields must be finite integers")
        if (yields < 1).any() or (yields > np.arange(2, self.k_max + 2)).any():
            raise ValueError("calibration yields violate physical depth bounds")
        expected = np.minimum(yields[:, -1, None], np.arange(2, self.k_max + 2))
        if not np.array_equal(yields, expected):
            raise ValueError("paired calibration yields violate prefix nesting")
        if mode not in ("frozen", "shared", "request_local"):
            raise ValueError("unknown update mode")
        if objective not in ("horizon", "ratio"):
            raise ValueError("unknown objective")
        if isinstance(local_prior_strength, (bool, np.bool_)) or not math.isfinite(local_prior_strength) or local_prior_strength <= 0:
            raise ValueError("local_prior_strength must be finite and positive")
        self.mode, self.objective = mode, objective
        self.local_prior_strength = float(local_prior_strength)
        self.max_horizon = _integer(max_horizon, "max_horizon", 1)
        self.zero_terminal_only = bool(zero_terminal_only)
        self.allowed_arms = tuple(range(self.k_max + 1)) if allowed_arms is None else tuple(allowed_arms)
        self.calibration_cells = len(costs)
        self.cost_counts = np.concatenate(([len(zero)], np.full(self.k_max, len(costs)))).astype(np.int64)
        self.cost_means = np.concatenate(([zero.mean()], costs.mean(axis=0)))
        self._calibration_trials = np.zeros(self.k_max)
        self._calibration_successes = np.zeros(self.k_max)
        for yield_value in yields[:, -1]:
            accepted = int(yield_value) - 1
            self._calibration_trials[:min(accepted + 1, self.k_max)] += 1
            self._calibration_successes[:accepted] += 1
        self.prefix_trials = self._calibration_trials.copy()
        self.prefix_successes = self._calibration_successes.copy()
        self.requests_started = 0
        self.online_cost_updates = 0
        self.online_prefix_updates = 0
        self.skipped_eos_updates = 0
        self.table_recomputations = 0
        self.decisions = 0
        self._table_dirty = True
        self._values = np.zeros(0)
        self._choices = np.zeros(0, dtype=np.int64)
        if mode == "request_local":
            self._reset_local_prefix()
        # Validate feasible actions immediately, including ratio-only mode.
        self._rebuild_table(self.max_horizon)

    def _reset_local_prefix(self) -> None:
        factor = self.local_prior_strength / self.calibration_cells
        self.prefix_trials = self._calibration_trials * factor
        self.prefix_successes = self._calibration_successes * factor
        self._table_dirty = True

    def begin_request(self) -> None:
        self.requests_started += 1
        if self.mode == "request_local":
            self._reset_local_prefix()

    def survival(self) -> np.ndarray:
        hazards = np.divide(
            self.prefix_successes, self.prefix_trials,
            out=np.zeros(self.k_max), where=self.prefix_trials > 0,
        )
        return np.cumprod(hazards)

    def _rebuild_table(self, horizon: int) -> None:
        self._values, self._choices = renewal_table(
            self.cost_means, self.survival(), horizon,
            self.allowed_arms, self.zero_terminal_only,
        )
        self._table_dirty = False
        self.table_recomputations += 1

    def select_arm(self, remaining_tokens: int) -> int:
        remaining = _integer(remaining_tokens, "remaining_tokens", 1)
        self.decisions += 1
        if self.objective == "ratio":
            means = np.concatenate(([1.0], 1.0 + np.cumsum(self.survival())))
            feasible = [k for k in self.allowed_arms if k < remaining and (
                not self.zero_terminal_only or remaining == 1 or k > 0
            )]
            return min(feasible, key=lambda k: (self.cost_means[k] / means[k], k))
        if self._table_dirty or remaining >= len(self._choices):
            self._rebuild_table(max(remaining, self.max_horizon))
        return int(self._choices[remaining])

    def update(
        self,
        executed_k: int,
        elapsed_ms: float,
        committed_yield: int,
        eos: bool = False,
    ) -> bool:
        """Update the physical arm only, including an explicitly clipped arm.

        Without EOS, right censoring after all k draft tokens succeed exposes
        successes at positions 1..k and no failure at k+1. A rejection exposes
        one failure, and later conditional hazards receive no observation.
        EOS rounds are skipped in their entirety because this model assumes
        the ordinary bonus-token yield. Adapters must pass executed depth and
        complete physical yield, never a nominal depth or partial delivery.
        """
        arm = _integer(executed_k, "executed_k")
        if arm > self.k_max:
            raise ValueError("executed_k exceeds supported depth")
        count = _integer(committed_yield, "committed_yield")
        if isinstance(elapsed_ms, (bool, np.bool_)) or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("elapsed cost must be finite and nonnegative")
        if eos:
            self.skipped_eos_updates += 1
            return False
        if not 1 <= count <= arm + 1:
            raise ValueError("uncensored yield must lie in 1..executed_k+1")
        if self.mode == "frozen":
            return False
        self.cost_counts[arm] += 1
        self.cost_means[arm] += (elapsed_ms - self.cost_means[arm]) / self.cost_counts[arm]
        self.online_cost_updates += 1
        if arm:
            accepted = count - 1
            self.prefix_trials[:min(accepted + 1, arm)] += 1
            self.prefix_successes[:accepted] += 1
            self.online_prefix_updates += 1
        self._table_dirty = True
        return True

    def snapshot(self) -> dict:
        q = self.survival()
        return {
            "mode": self.mode, "objective": self.objective,
            "cost_counts": self.cost_counts.tolist(),
            "cost_means": self.cost_means.tolist(),
            "prefix_trials": self.prefix_trials.tolist(),
            "prefix_successes": self.prefix_successes.tolist(),
            "survival": q.tolist(),
            "yield_means": np.concatenate(([1.0], 1.0 + np.cumsum(q))).tolist(),
            "calibration_cells": self.calibration_cells,
            "local_prior_strength": self.local_prior_strength,
            "allowed_arms": list(self.allowed_arms),
            "zero_terminal_only": self.zero_terminal_only,
            "requests_started": self.requests_started,
            "online_cost_updates": self.online_cost_updates,
            "online_prefix_updates": self.online_prefix_updates,
            "skipped_eos_updates": self.skipped_eos_updates,
            "table_recomputations": self.table_recomputations,
            "table_dirty": self._table_dirty,
            "decisions": self.decisions,
            "guarantee": "exact_supplied_renewal_model_only_no_inherited_regret_bound",
            "zero_arm": "one_measured_empty_draft_verification_rpc_not_streaming_AR",
        }
