from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import C, compute_kstar, simulate_acceptance_count


FIXED_K_CHOICES = (1, 3, 5, 7, 10)


def fixed_k_policy(k_fixed: int) -> int:
    return k_fixed


def greedy_policy(alpha: float, cd: float, cv: float, k_max: int = 20) -> int:
    return compute_kstar(alpha, cd, cv, d=0.0, k_max=k_max)


def sled_policy(k_current: int, last_rtt: float, timeout_threshold: float, k_max: int = 20) -> int:
    if last_rtt > timeout_threshold:
        return max(1, k_current - 1)
    return min(k_max, k_current + 1)


def confidence_stop_policy(alpha: float, p_min: float = 0.3, k_max: int = 20) -> int:
    """Halt when α^k < p_min — analogue of SpecDec++ / EAGLE threshold."""
    k = 1
    while k < k_max and alpha**k >= p_min:
        k += 1
    return k


def oracle_mean_policy(alpha: float, cd: float, cv: float, mu_d: float, k_max: int = 20) -> int:
    """Oracle-Mean: optimal fixed k with known delay mean (Theorem 3)."""
    return compute_kstar(alpha, cd, cv, d=mu_d, k_max=k_max)


def oracle_policy(alpha: float, cd: float, cv: float, d: float, k_max: int = 20) -> int:
    return compute_kstar(alpha, cd, cv, d=d, k_max=k_max)


# ---------------------------------------------------------------------------
# Bandit algorithms  (all use arm indices 1..k_max; stored 0-indexed)
# ---------------------------------------------------------------------------

@dataclass
class UCBSpecStop:
    """Self-normalized practical ratio-of-sums UCB used by historical runs."""
    k_max: int
    beta: float

    def __post_init__(self) -> None:
        self.s_n = np.zeros(self.k_max, dtype=float)
        self.s_a = np.zeros(self.k_max, dtype=float)
        self.t_k = np.zeros(self.k_max, dtype=float)

    def select_arm(self, t: int) -> int:
        unvisited = np.where(self.t_k == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0]) + 1
        indices = self.s_n / self.s_a - self.beta * np.sqrt(self.t_k * np.log(t)) / self.s_a
        return int(np.argmin(indices)) + 1

    def update(self, k: int, n_t: float, a_t: float) -> None:
        idx = k - 1
        self.s_n[idx] += n_t
        self.s_a[idx] += a_t
        self.t_k[idx] += 1.0


@dataclass
class DisplayedRatioLCB:
    """Ratio LCB with the fixed confidence width displayed in Algorithm 1."""

    k_max: int
    horizon: int
    l_max: float
    beta: float = 1.0
    context_count: int = 1

    def __post_init__(self) -> None:
        if self.k_max <= 0 or self.horizon <= 0 or self.context_count <= 0:
            raise ValueError("k_max, horizon, and context_count must be positive")
        if self.l_max <= 0 or self.beta < 0:
            raise ValueError("l_max must be positive and beta nonnegative")
        self.s_n = np.zeros(self.k_max, dtype=float)
        self.s_a = np.zeros(self.k_max, dtype=float)
        self.t_k = np.zeros(self.k_max, dtype=float)
        self.log_term = np.log(
            4.0 * self.context_count * self.k_max * self.horizon**2
        )

    def select_arm(self, t: int) -> int:
        del t  # The displayed finite-horizon index uses a fixed log term.
        unvisited = np.where(self.t_k == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0]) + 1
        estimates = self.s_n / self.s_a
        widths = self.l_max * np.sqrt(self.log_term / self.t_k)
        return int(np.argmin(estimates - self.beta * widths)) + 1

    def update(self, k: int, n_t: float, a_t: float) -> None:
        idx = k - 1
        self.s_n[idx] += n_t
        self.s_a[idx] += a_t
        self.t_k[idx] += 1.0


@dataclass
class PerRoundRatioUCB:
    """Per-Round-Ratio UCB (B6 in paper): minimises mean(N_t/A_t) — biased estimator."""
    k_max: int
    beta: float

    def __post_init__(self) -> None:
        self.sum_ratio = np.zeros(self.k_max, dtype=float)
        self.t_k = np.zeros(self.k_max, dtype=float)

    def select_arm(self, t: int) -> int:
        unvisited = np.where(self.t_k == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0]) + 1
        mean_ratio = self.sum_ratio / self.t_k
        indices = mean_ratio - self.beta * np.sqrt(np.log(t) / self.t_k)
        return int(np.argmin(indices)) + 1

    def update(self, k: int, n_t: float, a_t: float) -> None:
        self.sum_ratio[k - 1] += n_t / a_t
        self.t_k[k - 1] += 1.0


# Keep old name as alias for backwards compatibility
NaiveUCB1 = PerRoundRatioUCB


@dataclass
class EpsilonGreedyRatio:
    """ε-Greedy-Ratio: with prob ε explore uniformly; else exploit ratio-of-sums estimate."""
    k_max: int
    epsilon: float = 0.1

    def __post_init__(self) -> None:
        self.s_n = np.zeros(self.k_max, dtype=float)
        self.s_a = np.zeros(self.k_max, dtype=float)
        self.t_k = np.zeros(self.k_max, dtype=float)

    def select_arm(self, t: int, rng: np.random.Generator) -> int:
        unvisited = np.where(self.t_k == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0]) + 1
        if rng.random() < self.epsilon:
            return int(rng.integers(1, self.k_max + 1))
        ratios = self.s_n / self.s_a
        return int(np.argmin(ratios)) + 1

    def update(self, k: int, n_t: float, a_t: float) -> None:
        idx = k - 1
        self.s_n[idx] += n_t
        self.s_a[idx] += a_t
        self.t_k[idx] += 1.0


@dataclass
class EXP3Ratio:
    """EXP3 adapted to the ratio objective: minimises per-token latency."""
    k_max: int
    eta: float = 0.0          # 0 means auto-set per standard EXP3 schedule

    def __post_init__(self) -> None:
        self.weights = np.ones(self.k_max, dtype=float)
        self._t = 0

    def _current_eta(self) -> float:
        if self.eta > 0:
            return self.eta
        t = max(self._t, 1)
        return np.sqrt(np.log(self.k_max) / (self.k_max * t))

    def select_arm(self, t: int, rng: np.random.Generator) -> int:
        self._t = t
        w_sum = self.weights.sum()
        if w_sum == 0 or not np.isfinite(w_sum):
            self.weights[:] = 1.0
            w_sum = float(self.k_max)
        probs = self.weights / w_sum
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        return int(rng.choice(self.k_max, p=probs)) + 1

    def update(self, k: int, n_t: float, a_t: float, cost_scale: float = 1.0) -> None:
        if cost_scale <= 0:
            raise ValueError("cost_scale must be positive")
        w_sum = self.weights.sum()
        if w_sum == 0 or not np.isfinite(w_sum):
            self.weights[:] = 1.0
            w_sum = float(self.k_max)
        probs = self.weights / w_sum
        idx = k - 1
        # EXP3 requires losses in [0, 1].  Deployment supplies a calibration-
        # fixed upper scale; clipping makes an out-of-range evaluation sample
        # explicit in the algorithm rather than silently using an invalid loss.
        loss = float(np.clip((n_t / a_t) / cost_scale, 0.0, 1.0))
        eta = self._current_eta()
        self.weights[idx] *= np.exp(-eta * loss / max(probs[idx], 1e-10))
        # Rescale to prevent underflow: keep max weight = 1
        w_max = self.weights.max()
        if w_max > 0:
            self.weights /= w_max


@dataclass
class BanditSpecUCB:
    """BanditSpec Algorithm 4 adapted to the shared draft-length arm set.

    BanditSpec maximizes the expected number of tokens returned by a
    speculative round. This differs deliberately from UCB-SpecStop's
    ratio-of-sums latency objective. ``l_max`` bounds the per-pull token reward
    and equals ``k_max + 1`` when every round returns at most ``k`` accepted
    draft tokens plus one target-model bonus token.
    """

    k_max: int
    delta: float = 0.05
    l_max: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.l_max is None:
            self.l_max = float(self.k_max + 1)
        self.sum_reward = np.zeros(self.k_max, dtype=float)
        self.t_k = np.zeros(self.k_max, dtype=float)

    def select_arm(self, t: int) -> int:
        unvisited = np.where(self.t_k == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0]) + 1

        n = self.t_k
        mean = self.sum_reward / n
        log_arg = self.k_max * max(t, 1) ** 2 * np.sqrt(1.0 + n) / self.delta
        radius = 0.5 * float(self.l_max) * np.sqrt(
            ((1.0 + n) / n**2) * (1.0 + 2.0 * np.log(log_arg))
        )
        return int(np.argmax(mean + radius)) + 1

    def update(self, k: int, accepted_total: float) -> None:
        idx = k - 1
        self.sum_reward[idx] += accepted_total
        self.t_k[idx] += 1.0


@dataclass
class BanditSpecThroughputUCB:
    """Throughput-reward UCBSpec adapted from the public length experiment.

    BanditSpec's public ``ucb_length_spec_generate`` implementation (commit
    4e6dd869a3a1677ca62ee8055713cec439f26ccc) uses the committed tokens in one
    speculative round divided by that round's wall-clock duration as reward.
    It cycles through the arms while ``t < 2 * k_max`` and then adds one
    common scalar bonus to every empirical mean. ``common_bonus=True`` keeps
    that released behavior; ``False`` is an explicitly separate arm-wise UCB
    diagnostic.

    Reward is converted to tokens/s. ``exploration_weight`` consequently has
    squared tokens/s units and remains meaningful when logs use milliseconds.
    """

    k_max: int
    exploration_weight: float = 6.0
    common_bonus: bool = True

    def __post_init__(self) -> None:
        if self.k_max <= 0:
            raise ValueError("k_max must be positive")
        if self.exploration_weight < 0:
            raise ValueError("exploration_weight must be nonnegative")
        self.sum_reward = np.zeros(self.k_max, dtype=float)
        self.t_k = np.zeros(self.k_max, dtype=float)

    def select_arm(self, t: int) -> int:
        if t < 1:
            raise ValueError("t must be at least 1")
        if t < 2 * self.k_max:
            return int((t - 1) % self.k_max) + 1

        unvisited = np.where(self.t_k == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0]) + 1

        means = self.sum_reward / self.t_k
        log_term = np.log(max(t, 2))
        if self.common_bonus:
            radius = np.sqrt(
                self.exploration_weight * log_term / self.t_k.sum()
            )
        else:
            radius = np.sqrt(
                self.exploration_weight * log_term / self.t_k
            )
        return int(np.argmax(means + radius)) + 1

    def update(
        self,
        k: int,
        committed_tokens: float,
        elapsed_time: float,
        time_scale_to_seconds: float = 1.0,
    ) -> None:
        if not 1 <= k <= self.k_max:
            raise ValueError("k is outside the configured arm set")
        if committed_tokens < 0:
            raise ValueError("committed_tokens must be nonnegative")
        elapsed_seconds = elapsed_time * time_scale_to_seconds
        if elapsed_seconds <= 0:
            raise ValueError("elapsed time must be positive")
        idx = k - 1
        self.sum_reward[idx] += committed_tokens / elapsed_seconds
        self.t_k[idx] += 1.0


@dataclass
class EXP3Spec:
    """Accepted-token EXP3 comparator matching BanditSpec's reward choice."""

    k_max: int
    gamma: float = 0.0
    l_max: float | None = None

    def __post_init__(self) -> None:
        if self.l_max is None:
            self.l_max = float(self.k_max + 1)
        self.log_weights = np.zeros(self.k_max, dtype=float)
        self._t = 1
        self._last_probs = np.full(self.k_max, 1.0 / self.k_max)

    def _current_gamma(self, t: int) -> float:
        if self.gamma > 0.0:
            return min(self.gamma, 1.0)
        if self.k_max <= 1:
            return 0.0
        return min(
            1.0,
            np.sqrt(self.k_max * np.log(self.k_max) / ((np.e - 1.0) * max(t, 1))),
        )

    def select_arm(self, t: int, rng: np.random.Generator) -> int:
        self._t = max(t, 1)
        shifted = self.log_weights - np.max(self.log_weights)
        base = np.exp(shifted)
        base /= base.sum()
        gamma = self._current_gamma(self._t)
        self._last_probs = (1.0 - gamma) * base + gamma / self.k_max
        return int(rng.choice(self.k_max, p=self._last_probs)) + 1

    def update(self, k: int, accepted_total: float) -> None:
        idx = k - 1
        reward = np.clip(accepted_total / float(self.l_max), 0.0, 1.0)
        estimated_reward = reward / max(self._last_probs[idx], 1e-12)
        gamma = self._current_gamma(self._t)
        self.log_weights[idx] += gamma * estimated_reward / self.k_max
        self.log_weights -= np.max(self.log_weights)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def rollout_cost_for_k(
    k: int,
    alpha: float,
    cd: float,
    cv: float,
    d_mean: float,
    n_rounds: int,
    rng: np.random.Generator,
) -> float:
    total_n = 0.0
    total_a = 0.0
    for _ in range(n_rounds):
        d_t = rng.exponential(d_mean)
        n_t = k * (cd + cv) + 2.0 * d_t + cv
        accepted = simulate_acceptance_count(k, alpha, rng)
        a_t = accepted + 1
        total_n += n_t
        total_a += a_t
    return total_n / total_a


def deterministic_cost_for_k(k: int, alpha: float, cd: float, cv: float, d: float) -> float:
    return float(C(k, d, alpha, cd, cv))
