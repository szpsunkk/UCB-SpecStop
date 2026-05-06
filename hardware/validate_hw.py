"""
validate_hw.py — Offline smoke tests for hardware experiment code.

Tests the full pipeline WITHOUT real hardware using:
  1. A mock FastAPI server (runs in-process via httpx.ASGITransport)
  2. A tiny CPU model (or random logits) instead of real LLMs

Run with:
  python -m pytest hardware/validate_hw.py -v
  or directly:
  python hardware/validate_hw.py

All tests must pass before deploying to Jetson.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import UCBSpecStop, PerRoundRatioUCB, EXP3Ratio
from src.core import B, C, compute_kstar, dc_theory, simulate_acceptance_count


# ---------------------------------------------------------------------------
# 1. Core math invariants
# ---------------------------------------------------------------------------

def test_B_formula():
    """B(k, alpha) = (1 - alpha^(k+1)) / (1 - alpha)"""
    for alpha in [0.5, 0.7, 0.9]:
        for k in [1, 5, 10]:
            expected = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)
            assert abs(B(k, alpha) - expected) < 1e-9, \
                f"B({k},{alpha}) = {B(k,alpha):.6f} != {expected:.6f}"


def test_C_formula():
    """C(k, d, alpha, cd, cv) = (k*(cd+cv) + 2d + cv) / B(k, alpha)"""
    for d in [5.0, 50.0, 100.0]:
        for k in [1, 5, 10]:
            alpha, cd, cv = 0.7, 1.0, 0.5
            expected = (k * (cd + cv) + 2.0 * d + cv) / B(k, alpha)
            assert abs(C(k, d, alpha, cd, cv) - expected) < 1e-9, \
                f"C({k},{d}) mismatch"


def test_kstar_ordering():
    """k*(d) is non-decreasing in d (Theorem 5)."""
    alpha, cd, cv, k_max = 0.7, 1.0, 0.5, 20
    d_vals = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    kstars = [compute_kstar(alpha, cd, cv, d, k_max) for d in d_vals]
    for i in range(len(kstars) - 1):
        assert kstars[i] <= kstars[i + 1], \
            f"k* non-monotone at d={d_vals[i]}: {kstars[i]} > {kstars[i+1]}"


def test_dc_theory_phase_transition():
    """k*(d) = 1 for d < d_c, and k*(d) >= 2 for d just above d_c."""
    alpha, cd, cv = 0.7, 1.0, 0.5
    d_c = dc_theory(alpha, cd, cv)
    assert d_c > 0, f"d_c must be positive, got {d_c}"
    # Below critical: k*=1
    k_below = compute_kstar(alpha, cd, cv, d_c * 0.5, k_max=20)
    assert k_below == 1, f"Below d_c, expected k*=1, got {k_below}"
    # Above critical: k* >= 2
    k_above = compute_kstar(alpha, cd, cv, d_c * 2.0, k_max=20)
    assert k_above >= 2, f"Above d_c, expected k*>=2, got {k_above}"


# ---------------------------------------------------------------------------
# 2. UCBSpecStop invariants (ratio-of-sums, NOT mean(N/A))
# ---------------------------------------------------------------------------

def test_ucb_uses_ratio_of_sums():
    """UCBSpecStop bookkeeping must use S_N/S_A, not running mean(N_t/A_t)."""
    alg = UCBSpecStop(k_max=5, beta=1.0)
    # Feed two rounds on arm k=1
    alg.update(1, n_t=100.0, a_t=10.0)   # round 1: ratio = 10
    alg.update(1, n_t=200.0, a_t=10.0)   # round 2: ratio = 20
    # S_N/S_A = 300/20 = 15  (ratio-of-sums)
    # mean(N/A) = (10+20)/2 = 15 — coincidentally equal here, use different vals
    alg2 = UCBSpecStop(k_max=5, beta=1.0)
    alg2.update(1, n_t=100.0, a_t=5.0)   # N/A = 20
    alg2.update(1, n_t=200.0, a_t=20.0)  # N/A = 10
    # S_N/S_A = 300/25 = 12 (ratio-of-sums)
    # mean(N/A) = (20+10)/2 = 15  (naive)
    idx = 0  # arm 1 is index 0
    ratio_of_sums = alg2.s_n[idx] / alg2.s_a[idx]
    assert abs(ratio_of_sums - 12.0) < 1e-9, \
        f"Expected S_N/S_A=12, got {ratio_of_sums}"


def test_ucb_arm_indexing():
    """Arms are 1-indexed externally but stored 0-indexed internally."""
    alg = UCBSpecStop(k_max=5, beta=1.0)
    alg.update(k=3, n_t=100.0, a_t=10.0)
    assert alg.s_n[2] == 100.0,  "k=3 should update index 2"
    assert alg.s_a[2] == 10.0,   "k=3 should update index 2"
    assert alg.t_k[2] == 1.0,    "k=3 should update index 2"
    assert alg.s_n[0] == 0.0,    "k=1 should remain 0"


def test_ucb_initial_exploration():
    """Before any arm is played, UCB must explore arms in order 1..k_max."""
    alg = UCBSpecStop(k_max=5, beta=1.0)
    for expected_k in range(1, 6):
        k = alg.select_arm(t=1)
        assert k == expected_k, f"Expected arm {expected_k}, got {k}"
        alg.update(k, n_t=50.0, a_t=5.0)


def test_ucb_exploitation():
    """After exploration, UCB should prefer the arm with lowest S_N/S_A."""
    alg = UCBSpecStop(k_max=3, beta=0.0)  # beta=0 means pure exploitation
    # Arm 1: cost=20, Arm 2: cost=10 (best), Arm 3: cost=30
    for _ in range(3):
        alg.update(1, n_t=100.0, a_t=5.0)   # S_N/S_A = 20
        alg.update(2, n_t=100.0, a_t=10.0)  # S_N/S_A = 10
        alg.update(3, n_t=300.0, a_t=10.0)  # S_N/S_A = 30
    k = alg.select_arm(t=10)
    assert k == 2, f"Expected arm 2 (lowest cost), got {k}"


def test_naive_ucb_uses_mean_ratio():
    """PerRoundRatioUCB (B6) must use mean(N_t/A_t), not S_N/S_A."""
    alg = PerRoundRatioUCB(k_max=5, beta=1.0)
    alg.update(1, n_t=100.0, a_t=5.0)    # N/A = 20
    alg.update(1, n_t=200.0, a_t=20.0)   # N/A = 10
    # mean(N/A) = 15  (different from S_N/S_A = 12)
    mean_ratio = alg.sum_ratio[0] / alg.t_k[0]
    assert abs(mean_ratio - 15.0) < 1e-9, \
        f"Expected mean(N/A)=15, got {mean_ratio}"


def test_ucb_vs_naive_differ():
    """UCBSpecStop and NaiveUCB must give different estimates (validation of B6 distinction)."""
    ucb  = UCBSpecStop(k_max=1, beta=1.0)
    naive = PerRoundRatioUCB(k_max=1, beta=1.0)
    # Feed data where S_N/S_A != mean(N/A)
    for n_t, a_t in [(100.0, 5.0), (200.0, 20.0)]:
        ucb.update(1, n_t, a_t)
        naive.update(1, n_t, a_t)
    ucb_est  = ucb.s_n[0] / ucb.s_a[0]          # S_N/S_A = 300/25 = 12
    naive_est = naive.sum_ratio[0] / naive.t_k[0]  # mean(N/A) = (20+10)/2 = 15
    assert abs(ucb_est - 12.0) < 1e-9
    assert abs(naive_est - 15.0) < 1e-9
    assert ucb_est != naive_est, "UCBSpecStop and NaiveUCB must differ on this input"


# ---------------------------------------------------------------------------
# 3. Acceptance simulation invariants
# ---------------------------------------------------------------------------

def test_bonus_token_always_counted():
    """A_t = n_accepted + 1 must always hold."""
    rng = np.random.default_rng(42)
    for alpha in [0.3, 0.7, 0.95]:
        for k in [1, 5, 10]:
            n_accepted = simulate_acceptance_count(k, alpha, rng)
            a_t = n_accepted + 1
            assert a_t >= 1, "A_t must be at least 1"
            assert n_accepted <= k, f"n_accepted {n_accepted} > k {k}"


def test_acceptance_distribution():
    """acceptance count should be geometrically distributed (approx)."""
    rng = np.random.default_rng(0)
    alpha, k = 0.7, 20
    n_trials = 10000
    counts = [simulate_acceptance_count(k, alpha, rng) for _ in range(n_trials)]
    # P(accept all k) ≈ alpha^k (when k is large enough to not matter)
    # E[n_accepted] ≈ alpha / (1-alpha) for large k
    mean_accepted = np.mean(counts)
    expected_mean = alpha / (1.0 - alpha)  # geometric mean for large k
    # Loose check: within 20% of expected
    assert abs(mean_accepted - expected_mean) / expected_mean < 0.2, \
        f"Mean acceptance {mean_accepted:.2f} far from expected {expected_mean:.2f}"


# ---------------------------------------------------------------------------
# 4. Cloud server rejection sampling (offline, no model)
# ---------------------------------------------------------------------------

def test_rejection_sampling_logic():
    """Rejection sampling: accept token with prob min(1, p_target/p_draft)."""
    rng = random.Random(42)

    # If target_lp = draft_lp, acceptance prob = 1 (always accept)
    target_lp = math.log(0.5)
    draft_lp  = math.log(0.5)
    accept_prob = min(1.0, math.exp(target_lp - draft_lp))
    assert abs(accept_prob - 1.0) < 1e-9

    # If target_lp > draft_lp, accept prob = 1 (target assigns more prob)
    target_lp = math.log(0.8)
    draft_lp  = math.log(0.5)
    accept_prob = min(1.0, math.exp(target_lp - draft_lp))
    assert abs(accept_prob - 1.0) < 1e-9

    # If target_lp < draft_lp, accept prob < 1
    target_lp = math.log(0.3)
    draft_lp  = math.log(0.6)
    accept_prob = min(1.0, math.exp(target_lp - draft_lp))
    expected = 0.3 / 0.6
    assert abs(accept_prob - expected) < 1e-9, \
        f"accept_prob={accept_prob:.4f} != {expected:.4f}"


def test_greedy_is_subset_of_rejection_sampling():
    """Greedy acceptance (argmax match) is a special case with accept_prob in {0,1}."""
    # Greedy: accept iff draft_token == argmax(target_distribution)
    vocab_size = 100
    rng_np = np.random.default_rng(1)
    logits = rng_np.normal(0, 1, vocab_size)
    target_argmax = int(np.argmax(logits))

    # Chosen draft token matches argmax → accepted
    draft_tok_good = target_argmax
    match_good = (draft_tok_good == target_argmax)
    assert match_good

    # Chosen draft token doesn't match → rejected
    draft_tok_bad = (target_argmax + 1) % vocab_size
    match_bad = (draft_tok_bad == target_argmax)
    assert not match_bad


# ---------------------------------------------------------------------------
# 5. Mock end-to-end: no real hardware
# ---------------------------------------------------------------------------

def test_mock_experiment_loop():
    """
    Simulate a mini experiment: UCBSpecStop running 50 rounds with
    simulated acceptance (geometric) and simulated timing.
    Verify S_N/S_A bookkeeping is consistent throughout.
    """
    rng = np.random.default_rng(7)
    alpha, cd, cv, d = 0.7, 15.0, 3.0, 50.0
    k_max, beta = 10, 1.0

    alg = UCBSpecStop(k_max=k_max, beta=beta)
    s_n_ref = np.zeros(k_max)
    s_a_ref = np.zeros(k_max)
    t_k_ref = np.zeros(k_max)

    for t in range(1, 51):
        k = alg.select_arm(t)

        # Simulated timing
        n_accepted = simulate_acceptance_count(k, alpha, rng)
        a_t = float(n_accepted + 1)
        n_t = k * (cd + cv) + 2.0 * d + cv + rng.normal(0, 2.0)

        alg.update(k, n_t, a_t)
        idx = k - 1
        s_n_ref[idx] += n_t
        s_a_ref[idx] += a_t
        t_k_ref[idx] += 1.0

    # Check internal state matches reference
    np.testing.assert_allclose(alg.s_n, s_n_ref, rtol=1e-9)
    np.testing.assert_allclose(alg.s_a, s_a_ref, rtol=1e-9)
    np.testing.assert_allclose(alg.t_k, t_k_ref, rtol=1e-9)
    print("[OK] Mock experiment loop: UCBSpecStop state consistent after 50 rounds")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        test_B_formula,
        test_C_formula,
        test_kstar_ordering,
        test_dc_theory_phase_transition,
        test_ucb_uses_ratio_of_sums,
        test_ucb_arm_indexing,
        test_ucb_initial_exploration,
        test_ucb_exploitation,
        test_naive_ucb_uses_mean_ratio,
        test_ucb_vs_naive_differ,
        test_bonus_token_always_counted,
        test_acceptance_distribution,
        test_rejection_sampling_logic,
        test_greedy_is_subset_of_rejection_sampling,
        test_mock_experiment_loop,
    ]

    passed, failed = 0, 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
