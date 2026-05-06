from __future__ import annotations

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def B(k: int | np.ndarray, alpha: float) -> float | np.ndarray:
    k_arr = np.asarray(k)
    return (1.0 - np.power(alpha, k_arr + 1)) / (1.0 - alpha)


def C(k: int | np.ndarray, d: float | np.ndarray, alpha: float, cd: float, cv: float) -> float | np.ndarray:
    k_arr = np.asarray(k)
    d_arr = np.asarray(d)
    numerator = k_arr * (cd + cv) + 2.0 * d_arr + cv
    return numerator / B(k_arr, alpha)


def compute_kstar(alpha: float, cd: float, cv: float, d: float, k_max: int = 20) -> int:
    ks = np.arange(1, k_max + 1)
    costs = C(ks, d, alpha, cd, cv)
    return int(ks[np.argmin(costs)])


def dc_theory(alpha: float, cd: float, cv: float) -> float:
    return (cd + cv) * (1.0 + alpha) / (2.0 * alpha * alpha) - (cd + 2.0 * cv) / 2.0


def simulate_acceptance_count(k: int, alpha: float, rng: np.random.Generator) -> int:
    accepted = 0
    for _ in range(k):
        if rng.random() < alpha:
            accepted += 1
        else:
            break
    return accepted


def empirical_jump_point(d_values: np.ndarray, kstars: np.ndarray) -> float:
    diffs = np.diff(kstars)
    jump_indices = np.where(diffs > 0)[0]
    if len(jump_indices) == 0:
        return float("nan")
    return float(d_values[jump_indices[0] + 1])
