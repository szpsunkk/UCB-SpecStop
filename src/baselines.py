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
    """Algorithm 1 from the paper: ratio-of-sums UCB."""
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
        w_sum = self.weights.sum()
        if w_sum == 0 or not np.isfinite(w_sum):
            self.weights[:] = 1.0
            w_sum = float(self.k_max)
        probs = self.weights / w_sum
        idx = k - 1
        loss = (n_t / a_t) / cost_scale
        eta = self._current_eta()
        self.weights[idx] *= np.exp(-eta * loss / max(probs[idx], 1e-10))
        # Rescale to prevent underflow: keep max weight = 1
        w_max = self.weights.max()
        if w_max > 0:
            self.weights /= w_max


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
