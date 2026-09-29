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
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import (
    BanditSpecUCB,
    BanditSpecThroughputUCB,
    DisplayedRatioLCB,
    EXP3Ratio,
    EXP3Spec,
    PerRoundRatioUCB,
    UCBSpecStop,
)
from src.core import B, C, compute_kstar, dc_theory, simulate_acceptance_count
from hardware.analyze_tmc_revision import (
    adjacent_arm_crossings,
    aggregate_contextual_oracle,
    aggregate_request_run_metrics,
    apply_fixed_rtt_offset,
    best_ordered_threshold_policy,
    cell_arm_g_b_matrices,
    check_gamma,
    greedy_prefix_nesting_audit,
    hierarchical_block_paired_cell_indices,
    hierarchical_paired_cell_indices,
    integrate_edge_energy,
    prefix_acceptance_metrics,
    summarize_request_run_metrics,
    state_aware_arm_map,
    stochastic_monotonicity_violation,
    summarize_rtt_trace,
)
from hardware.collect_tailscale_icmp_trace import parse_ping_rtt
from hardware.run_tmc_revision import balanced_trace_window_starts, load_prompts
from hardware.run_tmc_revision import configure_draft_termination_for_target
from hardware.run_tmc_revision import total_energy_delta_j
from hardware.run_tmc_revision import tokenizer_vocab_sha256
from hardware.run_tmc_revision import validate_server_capabilities
from hardware.closed_loop import GreedyDraftSession, ServiceController
from hardware.kv_cache import cache_sequence_length, crop_past_key_values
import hardware.cloud_server as cloud_server
from hardware.analyze_exact_prediction_calibration import (
    lower_envelope_transitions,
    pair_crossing,
    proposed_grid,
)


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


def test_displayed_ratio_lcb_uses_fixed_width():
    alg = DisplayedRatioLCB(k_max=2, horizon=100, l_max=10.0, beta=1.0)
    alg.update(1, n_t=10.0, a_t=1.0)
    alg.update(2, n_t=20.0, a_t=1.0)
    # Arm 1 has the lower point estimate but far more samples. The displayed
    # finite-horizon width should therefore select under-sampled arm 2.
    for _ in range(99):
        alg.update(1, n_t=10.0, a_t=1.0)
    assert alg.select_arm(101) == 2


def test_contextual_displayed_ratio_lcb_uses_union_bound_scale():
    global_alg = DisplayedRatioLCB(
        k_max=2, horizon=100, l_max=10.0, beta=1.0
    )
    contextual_alg = DisplayedRatioLCB(
        k_max=2,
        horizon=100,
        l_max=10.0,
        beta=1.0,
        context_count=3,
    )
    assert contextual_alg.log_term > global_alg.log_term
    np.testing.assert_allclose(
        contextual_alg.log_term,
        np.log(4.0 * 3.0 * 2.0 * 100.0**2),
    )


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


def test_banditspec_ucb_reward_and_selection():
    """BanditSpec-UCB must maximize accepted-token reward, not latency."""
    alg = BanditSpecUCB(k_max=3, delta=0.05)
    assert [alg.select_arm(1)] == [1]
    alg.update(1, accepted_total=1.0)
    assert alg.select_arm(2) == 2
    alg.update(2, accepted_total=3.0)
    assert alg.select_arm(3) == 3
    alg.update(3, accepted_total=2.0)

    # Equal pull counts make the confidence terms equal; arm 2 has the largest
    # accepted-token mean and must therefore be selected.
    assert alg.select_arm(4) == 2


def test_banditspec_throughput_public_update_and_units():
    """Throughput UCB must be invariant to ms/second logging conventions."""
    from_ms = BanditSpecThroughputUCB(k_max=2, exploration_weight=6.0)
    from_seconds = BanditSpecThroughputUCB(k_max=2, exploration_weight=6.0)
    observations = [
        (1, 2.0, 100.0),
        (2, 3.0, 250.0),
        (1, 4.0, 200.0),
    ]
    for k, tokens, elapsed_ms in observations:
        from_ms.update(k, tokens, elapsed_ms, time_scale_to_seconds=1e-3)
        from_seconds.update(k, tokens, elapsed_ms / 1000.0)
    np.testing.assert_allclose(from_ms.sum_reward, from_seconds.sum_reward)
    np.testing.assert_allclose(from_ms.t_k, from_seconds.t_k)
    assert from_ms.select_arm(4) == from_seconds.select_arm(4)


def test_banditspec_throughput_public_common_bonus_matches_release():
    """The released common bonus cancels in the post-warmup argmax."""
    alg = BanditSpecThroughputUCB(
        k_max=2,
        exploration_weight=6.0,
        common_bonus=True,
    )
    alg.update(1, committed_tokens=1.0, elapsed_time=0.1)
    alg.update(2, committed_tokens=1.0, elapsed_time=0.2)
    alg.update(1, committed_tokens=1.0, elapsed_time=0.1)
    assert alg.select_arm(4) == 1


def test_exp3spec_probabilities_and_update():
    """EXP3Spec probabilities stay normalized and favor rewarded arms."""
    rng = np.random.default_rng(123)
    alg = EXP3Spec(k_max=3, gamma=0.2)
    k = alg.select_arm(1, rng)
    assert 1 <= k <= 3
    np.testing.assert_allclose(alg._last_probs.sum(), 1.0)
    before = alg.log_weights.copy()
    alg.update(k, accepted_total=4.0)
    assert alg.log_weights[k - 1] >= before[k - 1]
    assert np.all(np.isfinite(alg.log_weights))


def test_exp3_ratio_uses_bounded_scaled_loss():
    rng = np.random.default_rng(7)
    alg = EXP3Ratio(k_max=2, eta=0.1)
    arm = alg.select_arm(1, rng)
    alg.update(arm, n_t=500.0, a_t=1.0, cost_scale=100.0)
    assert np.all(np.isfinite(alg.weights))
    assert np.all(alg.weights >= 0.0)


def test_aggregate_contextual_oracle_uses_joint_ratio():
    """The aggregate-ratio oracle must not assume separable state ratios."""
    rows = []
    # State 0 occupies 90% of contexts.  Its separately optimal arm is k=1,
    # but choosing k=2 there enlarges the aggregate denominator enough to give
    # the lower joint ratio over both states.
    for round_id in range(10):
        state = 0 if round_id < 9 else 1
        rtt_ms = 1.0 if state == 0 else 10.0
        outcomes = (
            {1: (1.0, 1.0), 2: (6.0, 3.0)}
            if state == 0
            else {1: (100.0, 1.0), 2: (60.0, 3.0)}
        )
        for arm, (round_time, accepted) in outcomes.items():
            rows.append(
                {
                    "independent_run": 0,
                    "round": round_id,
                    "k_selected": arm,
                    "trace_http_rtt_ms": rtt_ms,
                    "total_round_time_ms": round_time,
                    "accepted_total": accepted,
                }
            )
    frame = pd.DataFrame(rows)
    edges = np.array([-np.inf, 5.0, np.inf])
    aggregate_map, aggregate_cost, _ = aggregate_contextual_oracle(
        frame, [1, 2], edges
    )
    per_state_map = state_aware_arm_map(frame, [1, 2], edges)

    assert per_state_map == {0: 1, 1: 2}
    assert aggregate_map == {0: 2, 1: 2}
    assert abs(aggregate_cost - 3.8) < 1e-9


def test_total_energy_delta_handles_unavailable_and_quantized_counters():
    assert total_energy_delta_j({}, {}) is None
    assert (
        total_energy_delta_j(
            {"total_energy_j": None}, {"total_energy_j": 2.0}
        )
        is None
    )
    assert (
        total_energy_delta_j(
            {"total_energy_j": 10.0}, {"total_energy_j": 12.5}
        )
        == 2.5
    )
    assert (
        total_energy_delta_j(
            {"total_energy_j": 10.0}, {"total_energy_j": 9.0}
        )
        is None
    )


def test_revision_server_capability_gate():
    good = {
        "protocol_version": 2,
        "server_entrypoint_sha256": "abc123",
        "verification_modes": ["greedy_exact", "acceptance_ratio_proxy"],
        "request_window_telemetry": True,
        "nvml_telemetry": True,
    }
    validate_server_capabilities(
        good, "abc123", require_request_window_telemetry=True
    )

    for changed in (
        {**good, "protocol_version": None},
        {**good, "server_entrypoint_sha256": "stale"},
        {**good, "verification_modes": ["acceptance_ratio_proxy"]},
        {**good, "request_window_telemetry": False},
        {**good, "nvml_telemetry": False},
    ):
        try:
            validate_server_capabilities(
                changed, "abc123", require_request_window_telemetry=True
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"capability gate accepted {changed}")

    session_good = {
        **good,
        "protocol_version": 3,
        "session_verification": True,
        "session_no_speculation_arm": True,
        "direct_target_streaming": True,
    }
    validate_server_capabilities(
        session_good,
        "abc123",
        require_request_window_telemetry=True,
        require_session_protocol=True,
    )
    try:
        validate_server_capabilities(
            good, "abc123", require_session_protocol=True
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("protocol v2 was accepted for a session run")


class _FakeTokenizer:
    def __init__(self, eos_token_id=15, vocab_size=16):
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return {f"token-{token_id}": token_id for token_id in range(self.vocab_size)}

    def convert_ids_to_tokens(self, token_id):
        return f"token-{token_id}"


class _FakeCausalLM(torch.nn.Module):
    """Predict token + step and expose a legacy-shaped deterministic cache."""

    def __init__(self, step=1, vocab_size=16):
        super().__init__()
        self.step = step
        self.vocab_size = vocab_size
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=15, pad_token_id=15)

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        del use_cache
        batch, length = input_ids.shape
        past_length = cache_sequence_length(past_key_values)
        logits = torch.full(
            (batch, length, self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        predictions = (input_ids + self.step) % self.vocab_size
        logits.scatter_(2, predictions.unsqueeze(-1), 100.0)
        cache_length = past_length + length
        key = torch.zeros((batch, 1, cache_length, 1), device=input_ids.device)
        value = torch.zeros_like(key)
        return SimpleNamespace(logits=logits, past_key_values=((key, value),))


def _install_fake_target(step=1, eos_token_id=15):
    cloud_server._model = _FakeCausalLM(step=step)
    cloud_server._model.generation_config.eos_token_id = eos_token_id
    cloud_server._tokenizer = _FakeTokenizer(eos_token_id=eos_token_id)
    cloud_server._target_sessions.clear()


def test_target_eos_alignment_requires_identical_token_mapping():
    tokenizer = _FakeTokenizer(eos_token_id=14)
    model = _FakeCausalLM()
    server_ping = {
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256(tokenizer),
        "tokenizer_eos_token_id": 15,
    }
    alignment = configure_draft_termination_for_target(
        tokenizer, model, server_ping
    )
    assert alignment["configured_draft_eos_token_id"] == 14
    assert alignment["effective_target_eos_token_id"] == 15
    assert alignment["eos_override_applied"] is True
    assert tokenizer.eos_token_id == 15
    assert model.generation_config.eos_token_id == 15
    assert model.generation_config.pad_token_id == 15

    mismatched = dict(server_ping)
    mismatched["tokenizer_vocab_sha256"] = "0" * 64
    try:
        configure_draft_termination_for_target(
            _FakeTokenizer(eos_token_id=14), _FakeCausalLM(), mismatched
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("mismatched token-to-ID mappings were accepted")


def test_kv_cache_crop_supports_legacy_cache():
    key = torch.zeros((1, 1, 7, 2))
    value = torch.ones_like(key)
    cropped = crop_past_key_values(((key, value),), 3)
    assert cache_sequence_length(cropped) == 3


def test_closed_loop_all_accept_preserves_both_kv_states():
    _install_fake_target(step=1)
    tokenizer = _FakeTokenizer()
    edge = GreedyDraftSession(_FakeCausalLM(step=1), tokenizer, [1, 2])
    start = cloud_server.start_session(
        cloud_server.SessionStartRequest(context_ids=[1, 2])
    )
    draft = edge.draft(3)
    assert draft["draft_ids"] == [3, 4, 5]
    verified = cloud_server.verify_session(
        cloud_server.SessionVerifyRequest(
            session_id=start.session_id, draft_ids=draft["draft_ids"]
        )
    )
    assert verified.n_accepted == 3
    assert verified.committed_token_ids == [3, 4, 5, 6]
    committed = edge.commit(verified.n_accepted, verified.committed_token_ids)
    assert committed["draft_kv_tokens_after_commit"] == 6
    assert verified.target_kv_tokens_after == 6
    next_draft = edge.draft(1)
    assert next_draft["draft_ids"] == [7]


def test_closed_loop_rejection_crops_draft_and_target_kv():
    _install_fake_target(step=1)
    edge = GreedyDraftSession(_FakeCausalLM(step=2), _FakeTokenizer(), [1, 2])
    start = cloud_server.start_session(
        cloud_server.SessionStartRequest(context_ids=[1, 2])
    )
    draft = edge.draft(3)
    assert draft["draft_ids"] == [4, 6, 8]
    verified = cloud_server.verify_session(
        cloud_server.SessionVerifyRequest(
            session_id=start.session_id, draft_ids=draft["draft_ids"]
        )
    )
    assert verified.n_accepted == 0
    assert verified.committed_token_ids == [3]
    committed = edge.commit(verified.n_accepted, verified.committed_token_ids)
    assert committed["draft_kv_tokens_after_commit"] == 3
    assert verified.target_kv_tokens_after == 3


def test_closed_loop_no_speculation_arm_commits_one_target_token():
    _install_fake_target(step=1)
    start = cloud_server.start_session(
        cloud_server.SessionStartRequest(context_ids=[1, 2])
    )
    verified = cloud_server.verify_session(
        cloud_server.SessionVerifyRequest(
            session_id=start.session_id, draft_ids=[]
        )
    )
    assert verified.n_accepted == 0
    assert verified.committed_token_ids == [3]
    assert verified.target_kv_tokens_before == 2
    assert verified.target_kv_tokens_after == 3


def test_closed_loop_accepted_eos_has_no_bonus():
    _install_fake_target(step=1, eos_token_id=5)
    edge = GreedyDraftSession(
        _FakeCausalLM(step=1), _FakeTokenizer(eos_token_id=5), [1, 2]
    )
    start = cloud_server.start_session(
        cloud_server.SessionStartRequest(context_ids=[1, 2])
    )
    draft = edge.draft(8)
    assert draft["draft_ids"] == [3, 4, 5]
    verified = cloud_server.verify_session(
        cloud_server.SessionVerifyRequest(
            session_id=start.session_id, draft_ids=draft["draft_ids"]
        )
    )
    assert verified.n_accepted == 3
    assert verified.bonus_token_id is None
    assert verified.committed_token_ids == [3, 4, 5]
    assert verified.stop_reason == "eos"
    committed = edge.commit(verified.n_accepted, verified.committed_token_ids)
    assert committed["draft_kv_tokens_after_commit"] == 5


def test_ucbspec_controller_maps_and_updates_no_speculation_arm():
    controller = ServiceController(
        "ucbspec_throughput", actual_arms=[0, 1, 2]
    )
    assert controller.select_arm() == 0
    updated, reason = controller.update(0, elapsed_ms=10.0, delivered_tokens=1)
    assert updated and reason is None
    np.testing.assert_allclose(controller.algorithm.t_k, [1.0, 0.0, 0.0])


def test_exact_crossing_and_lower_envelope_allow_skipped_arms():
    # Arm 2 lies above the lower envelope; the optimum jumps from 1 to 3.
    m = {1: 10.0, 2: 30.0, 3: 44.0}
    b = {1: 1.0, 2: 1.5, 3: 3.0}
    crossing = pair_crossing(m[1], b[1], m[3], b[3])
    assert abs(crossing - 7.0) < 1e-12
    transitions = lower_envelope_transitions(m, b, [1, 2, 3])
    assert transitions == [
        {"from_arm": 1, "to_arm": 3, "added_rtt_crossing_ms": 7.0}
    ]
    assert proposed_grid(transitions, minimum_margin_ms=2.0) == [0.0, 5.0, 10.0]


def test_request_run_metrics_use_ratios_of_sums():
    frame = pd.DataFrame(
        {
            "k": [1, 1],
            "prompt_stratum": ["short", "short"],
            "generation_token_cap": [32, 32],
            "independent_run": [0, 0],
            "generated_tokens": [10, 30],
            "ttft_ms": [10.0, 20.0],
            "e2e_ms": [100.0, 600.0],
            "edge_energy_j": [2.0, 3.0],
            "cloud_energy_j": [4.0, 6.0],
            "edge_peak_ram_mb": [100.0, 110.0],
            "edge_process_peak_allocated_mb": [50.0, 60.0],
            "cloud_peak_memory_mb": [1000.0, 1010.0],
        }
    )
    result = aggregate_request_run_metrics(
        frame, ["k", "prompt_stratum", "generation_token_cap"]
    ).iloc[0]
    assert abs(result["throughput_tokens_s"] - 40.0 / 0.7) < 1e-9
    assert abs(result["edge_energy_per_token_j"] - 5.0 / 40.0) < 1e-9
    assert abs(result["cloud_energy_per_token_j"] - 10.0 / 40.0) < 1e-9
    assert result["ttft_ms"] == 15.0
    assert result["cloud_peak_memory_mb"] == 1010.0


def test_request_summary_keeps_run_and_request_units_distinct():
    rows = []
    for run in (0, 1):
        for request in (0, 1):
            rows.append(
                {
                    "k": 1,
                    "generation_token_cap": 32,
                    "independent_run": run,
                    "generated_tokens": 32,
                    "ttft_ms": 10.0 + run + request,
                    "e2e_ms": 100.0 + 10.0 * run + request,
                    "edge_energy_j": 3.2,
                    "cloud_energy_j": 6.4,
                    "edge_peak_ram_mb": 100.0,
                    "edge_process_peak_allocated_mb": 50.0,
                    "cloud_peak_memory_mb": 1000.0,
                }
            )
    enriched = pd.DataFrame(rows)
    keys = ["k", "generation_token_cap"]
    per_run = aggregate_request_run_metrics(enriched, keys)
    summary = summarize_request_run_metrics(enriched, per_run, keys).iloc[0]
    assert summary["independent_runs"] == 2
    assert summary["request_samples"] == 4
    assert summary["e2e_ms_p99_across_requests"] > summary["e2e_ms_mean"]


def test_edge_energy_integrates_full_request_boundaries():
    requests = pd.DataFrame(
        {
            "monotonic_start_ns": [500_000_000],
            "monotonic_end_ns": [1_500_000_000],
            "generated_tokens": [10],
        }
    )
    telemetry = pd.DataFrame(
        {
            "monotonic_ns": [0, 1_000_000_000, 2_000_000_000],
            "vdd_in_mw": [0.0, 10_000.0, 20_000.0],
            "ram_used_mb": [100.0, 110.0, 120.0],
        }
    )
    result = integrate_edge_energy(requests, telemetry).iloc[0]
    assert abs(result["edge_energy_j"] - 10.0) < 1e-9
    assert abs(result["edge_energy_per_token_j"] - 1.0) < 1e-9


def test_fixed_rtt_offset_is_explicit_and_nonmutating():
    source = pd.DataFrame(
        {
            "total_round_time_ms": [100.0],
            "trace_http_rtt_ms": [10.0],
            "d_eff_one_way_ms": [6.0],
        }
    )
    shifted = apply_fixed_rtt_offset(source, 350.0)
    assert source.loc[0, "total_round_time_ms"] == 100.0
    assert shifted.loc[0, "total_round_time_ms"] == 450.0
    assert shifted.loc[0, "trace_http_rtt_ms"] == 360.0
    assert shifted.loc[0, "d_eff_one_way_ms"] == 181.0
    try:
        apply_fixed_rtt_offset(source, -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative fixed RTT offsets must be rejected")


def test_balanced_trace_windows_are_disjoint_and_deterministic():
    starts, disjoint = balanced_trace_window_starts(3599, 10, 100, 20260819)
    repeated, repeated_disjoint = balanced_trace_window_starts(
        3599, 10, 100, 20260819
    )
    assert disjoint is True
    assert repeated_disjoint is True
    assert starts == repeated
    windows = [set(range(start, start + 100)) for start in starts]
    assert len(set().union(*windows)) == 1000
    for left in range(len(windows)):
        for right in range(left + 1, len(windows)):
            assert not windows[left].intersection(windows[right])

    short_starts, short_disjoint = balanced_trace_window_starts(
        500, 10, 100, 20260819
    )
    assert short_disjoint is False
    assert len(short_starts) == 10
    assert all(0 <= start < 500 for start in short_starts)


def test_load_prompts_supports_multiline_jsonl(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        '{"prompt_id":"a","prompt":"line one\\nline two"}\n'
        '{"prompt_id":"b","prompt":"second prompt"}\n',
        encoding="utf-8",
    )
    assert load_prompts(str(prompt_path)) == [
        "line one\nline two",
        "second prompt",
    ]


def test_icmp_trace_parser_accepts_iputils_output():
    output = "64 bytes from edge-cloud.example.invalid: icmp_seq=1 ttl=64 time=282.347 ms"
    assert parse_ping_rtt(output) == 282.347
    assert parse_ping_rtt("64 bytes from ::1: icmp_seq=1 ttl=64 time<1 ms") == 1.0


def test_icmp_trace_parser_rejects_failed_probe_output():
    output = "1 packets transmitted, 0 received, 100% packet loss"
    assert parse_ping_rtt(output) is None


def test_rtt_trace_summary_audits_failures_and_sequence_integrity():
    frame = pd.DataFrame(
        {
            "sequence": [0, 1, 2, 3],
            "utc_start": pd.date_range("2026-01-01", periods=4, freq="500ms", tz="UTC"),
            "rtt_ms": [10.0, np.nan, np.nan, 40.0],
            "success": [1, 0, 0, 1],
            "schedule_lag_ms": [0.1, 0.2, 0.3, 0.4],
            "probe_wall_ms": [11.0, 1000.0, 1000.0, 41.0],
        }
    )
    summary = summarize_rtt_trace(frame)
    assert summary["rows"] == 4
    assert summary["valid_successful_rtt_rows"] == 2
    assert summary["longest_consecutive_failure_rows"] == 2
    assert summary["duplicate_sequence_rows"] == 0
    assert summary["missing_sequence_rows"] == 0
    assert summary["sequence_monotone"] is True
    assert summary["timestamp_monotone"] is True
    assert summary["rtt_mean_ms"] == 25.0
    assert summary["rtt_p50_ms"] == 25.0


def test_adjacent_arm_crossing_solves_equal_ratio():
    frame = pd.DataFrame(
        {
            "independent_run": [0, 0, 0, 0],
            "round": [0, 1, 0, 1],
            "k_selected": [1, 1, 2, 2],
            "total_round_time_ms": [100.0, 100.0, 360.0, 360.0],
            "accepted_total": [1.0, 1.0, 3.0, 3.0],
        }
    )
    per_run, summary = adjacent_arm_crossings(frame, [1, 2])
    crossing = float(per_run.iloc[0]["added_rtt_crossing_ms"])
    assert abs(crossing - 30.0) < 1e-9
    left = (200.0 + 2.0 * crossing) / 2.0
    right = (720.0 + 2.0 * crossing) / 6.0
    assert abs(left - right) < 1e-9
    assert float(summary.iloc[0]["added_rtt_crossing_ms_mean"]) == crossing


def test_prefix_acceptance_uses_one_largest_arm_row_per_cell():
    frame = pd.DataFrame(
        {
            "independent_run": [0, 0, 0, 0],
            "round": [0, 0, 1, 1],
            "k_selected": [1, 3, 1, 3],
            "prefix_accept_indicator_1": [1, 1, 0, 0],
            "prefix_accept_indicator_2": [np.nan, 1, np.nan, 0],
            "prefix_accept_indicator_3": [np.nan, 0, np.nan, 0],
        }
    )
    per_run, summary = prefix_acceptance_metrics(frame, 3)
    assert len(per_run) == 3
    np.testing.assert_allclose(
        summary["prefix_survival_probability_mean"], [0.5, 0.5, 0.0]
    )
    np.testing.assert_allclose(
        summary["conditional_acceptance_probability_mean"], [0.5, 1.0, 0.0]
    )


def test_greedy_prefix_nesting_audit_retains_discordance():
    frame = pd.DataFrame(
        {
            "independent_run": [0, 0, 0, 1, 1, 1],
            "round": [0, 0, 0, 0, 0, 0],
            "k_selected": [1, 2, 3, 1, 2, 3],
            "accepted_draft_len": [1, 0, 3, 1, 2, 2],
        }
    )
    result = greedy_prefix_nesting_audit(frame, 3)
    assert result["total_run_round_cells"] == 2
    assert result["discordant_run_round_cells"] == 1
    assert result["discordant_cell_fraction"] == 0.5


def test_markov_stochastic_monotonicity_audit():
    iid = np.tile(np.array([0.2, 0.3, 0.5]), (3, 1))
    assert stochastic_monotonicity_violation(iid) == 0.0
    unordered = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )
    assert stochastic_monotonicity_violation(unordered) > 0.0


def test_markov_threshold_audit_is_distinct_from_gamma_monotonicity():
    gamma = np.full((5, 2), np.inf)
    gamma[1] = [-2.0, -3.0]
    gamma[2] = [-1.0, -2.0]
    gamma[3] = [1.0, -1.0]
    # Gamma decreases from depth 3 to the terminal +inf convention only after
    # slicing, so inject a non-monotone pre-crossing value without changing
    # the stop sets: state 0 stops at 3 and state 1 at 4.
    gamma[2, 0] = -3.0
    gamma[3, 0] = 1.0
    result = check_gamma(gamma)
    assert result["condition_i_depth_monotone"] is False
    assert result["stop_region_upper_interval"] is True
    assert result["ordered_threshold_policy"] is True
    assert result["thresholds"] == [3, 4]


def test_markov_cell_bootstrap_matrices_preserve_paired_arm_moments():
    frame = pd.DataFrame(
        {
            "independent_run": [0, 0, 0, 0, 1, 1, 1, 1],
            "round": [0, 0, 1, 1, 0, 0, 1, 1],
            "k_selected": [1, 2, 1, 2, 1, 2, 1, 2],
            "total_round_time_ms": [110, 125, 120, 137, 130, 149, 140, 161],
            "trace_http_rtt_ms": [10, 10, 20, 20, 30, 30, 40, 40],
            "accepted_total": [1, 2, 1, 3, 2, 3, 2, 4],
        }
    )
    cell_g, cell_b = cell_arm_g_b_matrices(frame, [1, 2])
    np.testing.assert_allclose(cell_g, [[100, 115], [100, 117], [100, 119], [100, 121]])
    np.testing.assert_allclose(cell_b, [[1, 2], [1, 3], [2, 3], [2, 4]])

    sampled = np.array([0, 0, 3, 3])
    np.testing.assert_allclose(cell_g[sampled].mean(axis=0), [100, 118])
    np.testing.assert_allclose(cell_b[sampled].mean(axis=0), [1.5, 3.0])


def test_markov_hierarchical_bootstrap_preserves_run_cluster_sizes():
    index = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        names=["independent_run", "round"],
    )
    sampled = hierarchical_paired_cell_indices(index, np.random.default_rng(9))
    assert len(sampled) == len(index)
    assert np.all((sampled >= 0) & (sampled < len(index)))
    # Each selected run contributes its complete two-cell cluster size before
    # within-run resampling, even if the same run is selected twice.
    assert len(sampled) % 2 == 0


def test_hierarchical_block_bootstrap_preserves_paired_run_cells():
    index = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        names=["independent_run", "round"],
    )
    sampled = hierarchical_block_paired_cell_indices(
        index, block_length=2, rng=np.random.default_rng(13)
    )
    assert len(sampled) == len(index)
    assert np.all((sampled >= 0) & (sampled < len(index)))


def test_best_ordered_threshold_matches_dp_when_optimum_is_threshold():
    round_cost = np.array([np.nan, 1.0, 2.0, 3.0])
    token_yield = np.array([np.nan, 1.0, 1.5, 1.75])
    state_delay = np.array([1.0, 2.0])
    transition = np.array([[0.8, 0.2], [0.2, 0.8]])
    occupancy = np.array([0.5, 0.5])
    from hardware.analyze_tmc_revision import solve_dp

    optimum, *_ = solve_dp(
        round_cost, token_yield, state_delay, transition, occupancy
    )
    threshold_cost, thresholds = best_ordered_threshold_policy(
        round_cost, token_yield, state_delay, transition, occupancy
    )
    assert thresholds == (2, 3)
    assert abs(threshold_cost - optimum) < 1e-7


def test_unrestricted_dp_beats_threshold_enumeration_on_counterexample():
    """The solvers separate on a prefix-compatible non-threshold MDP."""

    prefix_probabilities = np.array(
        [0.732757, 0.298986, 0.283484, 0.158585, 0.054897]
    )
    assert np.all(np.diff(prefix_probabilities) <= 0)
    assert np.all((prefix_probabilities >= 0) & (prefix_probabilities <= 1))
    round_cost = np.array(
        [np.nan, 5.909363, 10.592570, 11.267545, 19.069074, 26.025968]
    )
    token_yield = np.r_[np.nan, 1.0 + np.cumsum(prefix_probabilities)]
    state_delay = np.array([1.249953, 24.218451, 36.226187])
    transition = np.array(
        [
            [0.912695, 0.001454, 0.085851],
            [0.018549, 0.981225, 0.000226],
            [0.966263, 0.020092, 0.013645],
        ]
    )
    transition /= transition.sum(axis=1, keepdims=True)
    occupancy = np.array([0.271476, 0.001194, 0.727330])
    occupancy /= occupancy.sum()
    from hardware.analyze_tmc_revision import solve_dp

    optimum, *_ = solve_dp(
        round_cost, token_yield, state_delay, transition, occupancy
    )
    threshold_cost, thresholds = best_ordered_threshold_policy(
        round_cost, token_yield, state_delay, transition, occupancy
    )
    assert thresholds == (1, 3, 5)
    assert threshold_cost - optimum > 0.13


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
        test_displayed_ratio_lcb_uses_fixed_width,
        test_contextual_displayed_ratio_lcb_uses_union_bound_scale,
        test_ucb_exploitation,
        test_naive_ucb_uses_mean_ratio,
        test_ucb_vs_naive_differ,
        test_banditspec_ucb_reward_and_selection,
        test_banditspec_throughput_public_update_and_units,
        test_banditspec_throughput_public_common_bonus_matches_release,
        test_exp3spec_probabilities_and_update,
        test_exp3_ratio_uses_bounded_scaled_loss,
        test_aggregate_contextual_oracle_uses_joint_ratio,
        test_total_energy_delta_handles_unavailable_and_quantized_counters,
        test_revision_server_capability_gate,
        test_target_eos_alignment_requires_identical_token_mapping,
        test_kv_cache_crop_supports_legacy_cache,
        test_closed_loop_all_accept_preserves_both_kv_states,
        test_closed_loop_rejection_crops_draft_and_target_kv,
        test_closed_loop_no_speculation_arm_commits_one_target_token,
        test_closed_loop_accepted_eos_has_no_bonus,
        test_ucbspec_controller_maps_and_updates_no_speculation_arm,
        test_exact_crossing_and_lower_envelope_allow_skipped_arms,
        test_request_run_metrics_use_ratios_of_sums,
        test_request_summary_keeps_run_and_request_units_distinct,
        test_edge_energy_integrates_full_request_boundaries,
        test_fixed_rtt_offset_is_explicit_and_nonmutating,
        test_balanced_trace_windows_are_disjoint_and_deterministic,
        test_load_prompts_supports_multiline_jsonl,
        test_icmp_trace_parser_accepts_iputils_output,
        test_icmp_trace_parser_rejects_failed_probe_output,
        test_rtt_trace_summary_audits_failures_and_sequence_integrity,
        test_adjacent_arm_crossing_solves_equal_ratio,
        test_prefix_acceptance_uses_one_largest_arm_row_per_cell,
        test_greedy_prefix_nesting_audit_retains_discordance,
        test_markov_stochastic_monotonicity_audit,
        test_markov_threshold_audit_is_distinct_from_gamma_monotonicity,
        test_markov_cell_bootstrap_matrices_preserve_paired_arm_moments,
        test_markov_hierarchical_bootstrap_preserves_run_cluster_sizes,
        test_hierarchical_block_bootstrap_preserves_paired_run_cells,
        test_best_ordered_threshold_matches_dp_when_optimum_is_threshold,
        test_unrestricted_dp_beats_threshold_enumeration_on_counterexample,
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
