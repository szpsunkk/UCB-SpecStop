"""Offline controller, boundary and completion-timing tests; no hardware calls."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hardware import closed_loop as loop


@pytest.mark.parametrize("method", ["ratio_regularized", "ratio_residual"])
def test_new_controllers_use_physical_sparse_arm_indices(method):
    controller = loop.ServiceController(
        method, actual_arms=[1, 4, 8], beta=0, prior_strength=2,
    )
    controller.warm_start(
        np.tile([20, 25, 9], (2, 1)), np.tile([2, 5, 9], (2, 1)),
    )
    assert controller.select_arm() == 8
    assert controller.update(8, 18, 9) == (True, None)
    np.testing.assert_array_equal(controller.algorithm.t_k, [2, 0, 0, 2, 0, 0, 0, 3])
    assert controller.calibration_cells == 2
    assert controller.calibration_observations == 6
    assert controller.rounds_selected == 1


@pytest.mark.parametrize(
    "method", ["sn", "per_round_ucb", "lcb", "ucbspec_throughput",
               "ratio_regularized", "ratio_residual"]
)
def test_shared_warm_data_and_mask_do_not_fabricate_feedback(method):
    controller = loop.ServiceController(
        method, actual_arms=[1, 4, 8], allowed_arms=[4, 8], beta=0,
    )
    controller.warm_start(
        np.tile([1, 50, 90], (3, 1)), np.tile([2, 5, 9], (3, 1)),
    )
    assert controller.calibration_observations == 9
    for _ in range(5):
        arm = controller.select_arm()
        assert arm in [4, 8]
        controller.update(arm, 10, 1)
    count = controller.algorithm.t_k
    assert count.sum() == 9 + 5
    assert count[0] == 3  # Excluded arm retains its real calibration observations.
    json.dumps(controller.state_dict(), allow_nan=False)


def test_prefix_sharing_requires_capability_and_deduplicates_warm_cells():
    with pytest.raises(ValueError, match="capability"):
        loop.ServiceController("prefix_shared", actual_arms=[1, 2, 3])
    controller = loop.ServiceController(
        "prefix_shared", actual_arms=[1, 2, 3], prefix_compatible=True,
        allowed_arms=[2, 3],
    )
    controller.warm_start(
        np.array([[10, 20, 30], [11, 21, 31]]), np.array([[2, 3, 4], [1, 1, 1]]),
    )
    state = controller.state_dict()["algorithm_state"]
    assert state["prefix_counts"] == [2, 2, 2]
    assert state["counts"] == [2, 2, 2]
    assert controller.select_arm() in [2, 3]
    before = controller.algorithm.t_k.copy()
    assert controller.update(3, 10, 2, censored=True, censor_reason="terminal_eos") == (
        False, "terminal_eos",
    )
    np.testing.assert_array_equal(controller.algorithm.t_k, before)


def test_invalid_warm_batch_does_not_partially_update_controller():
    controller = loop.ServiceController("ratio_regularized", actual_arms=[1, 4, 8])
    with pytest.raises(ValueError, match="physical arm"):
        controller.warm_start(np.ones((2, 3)), np.array([[2, 5, 9], [3, 5, 9]]))
    assert controller.algorithm.t_k.sum() == 0
    assert controller.calibration_observations == 0
    with pytest.raises(ValueError, match="all actual_arms"):
        controller.warm_start(np.ones((2, 2)), np.ones((2, 2)), arms=[1, 8])


def test_zero_and_boundary_feedback_are_skipped_for_all_online_methods():
    for method in ["sn", "per_round_ucb", "ucbspec_throughput", "ratio_regularized", "ratio_residual"]:
        controller = loop.ServiceController(method, actual_arms=[1, 2])
        assert controller.update(1, 10, 0) == (False, "zero_delivered_tokens")
        assert controller.update(1, 10, 1, censored=True)[0] is False
        assert controller.algorithm.t_k.sum() == 0


class Clock:
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            return self.value

    def advance(self, seconds):
        with self.lock:
            self.value += seconds


def install_fake_speculative_runtime(monkeypatch, responses):
    clock = Clock()
    monkeypatch.setattr(loop.time, "perf_counter", clock.now)

    class FakeDraftSession:
        def __init__(self, model, tokenizer, context_ids):
            self.kv_tokens = len(context_ids)
            self.next_logits = torch.tensor([0.1, 0.9])
            clock.advance(0.3)

        def draft(self, k_value):
            clock.advance(0.4)
            return {"draft_ids": [7] * k_value, "draft_time_ms": 400.0,
                    "draft_kv_tokens_before": self.kv_tokens,
                    "draft_kv_tokens_after_draft": self.kv_tokens + k_value}

        def commit(self, n_accepted, committed):
            clock.advance(0.25)
            self.kv_tokens += len(committed)
            return {"draft_kv_tokens_after_commit": self.kv_tokens,
                    "draft_commit_time_ms": 250.0}

    def start(server, context_ids, delay):
        clock.advance(0.2)
        return {"session_id": "fake", "target_kv_tokens": len(context_ids),
                "prefill_time_ms": 100.0, "_network_and_server_ms": 200.0}

    remaining = iter(responses)

    def verify(server, session_id, draft_ids, delay):
        clock.advance(0.5)
        response = next(remaining)
        return {"n_accepted": 0, "verify_time_ms": 500.0,
                "target_kv_tokens_before": 1, "target_kv_tokens_after": 2, **response}

    def close(server, session_id):
        clock.advance(1.0)
        return {"closed": True}

    monkeypatch.setattr(loop, "GreedyDraftSession", FakeDraftSession)
    monkeypatch.setattr(loop, "start_target_session", start)
    monkeypatch.setattr(loop, "verify_target_session", verify)
    monkeypatch.setattr(loop, "close_target_session", close)

    class TimedController(loop.ServiceController):
        def select_arm(self):
            clock.advance(0.2)
            return super().select_arm()

        def update(self, *args, **kwargs):
            clock.advance(0.1)
            return super().update(*args, **kwargs)

    return clock, TimedController


def test_completion_includes_terminal_eos_commit_and_update_but_not_cleanup(monkeypatch):
    clock, controller_type = install_fake_speculative_runtime(monkeypatch, [
        {"committed_token_ids": [7], "stop_reason": None},
        {"committed_token_ids": [99], "stop_reason": "eos"},
    ])
    controller = controller_type("sn", actual_arms=[1])
    request, rounds = loop.run_speculative_request(
        "http://fake", None, SimpleNamespace(eos_token_id=99), [1], controller, 10,
    )
    assert request["generated_tokens"] == 1
    assert request["last_token_latency_ms"] == pytest.approx(1600)
    assert request["e2e_ms"] == request["completion_latency_ms"] == pytest.approx(3400)
    assert request["cleanup_time_ms"] == pytest.approx(1000)
    assert request["request_wall_ms"] == pytest.approx(4400)
    assert request["controller_time_ms"] == pytest.approx(600)
    for row in rounds:
        assert row["controller_time_ms"] == pytest.approx(300)
        assert row["controller_select_time_ms"] == pytest.approx(200)
        assert row["controller_update_time_ms"] == pytest.approx(100)
        assert row["round_cycle_ms"] - row["feedback_elapsed_ms"] == pytest.approx(100)
    assert rounds[1]["feedback_skipped_reason"] == "terminal_eos"
    assert controller.algorithm.t_k.tolist() == [1]


def test_output_cap_censoring_preserves_timing_and_does_not_update_wrong_arm(monkeypatch):
    _, controller_type = install_fake_speculative_runtime(monkeypatch, [
        {"committed_token_ids": [7], "stop_reason": None},
    ])
    controller = controller_type("ratio_regularized", actual_arms=[4])
    request, rounds = loop.run_speculative_request(
        "http://fake", None, SimpleNamespace(eos_token_id=99), [1], controller, 1,
    )
    assert request["generated_tokens"] == 1
    assert request["stop_reason"] == "cap"
    assert request["e2e_ms"] > request["last_token_latency_ms"]
    assert rounds[0]["selected_arm"] == 4 and rounds[0]["effective_k"] == 0
    assert rounds[0]["feedback_skipped_reason"] == "generation_cap"
    assert controller.algorithm.t_k.sum() == 0


def test_context_hook_runs_before_draft_inside_controller_timing(monkeypatch):
    clock, controller_type = install_fake_speculative_runtime(monkeypatch, [
        {"committed_token_ids": [99], "stop_reason": "eos"},
    ])

    class ContextController(controller_type):
        def select_arm_with_context(self, draft_session, remaining_tokens):
            assert draft_session.kv_tokens == 1
            assert remaining_tokens == 10
            assert draft_session.next_logits.argmax().item() == 1
            clock.advance(0.05)
            return self.select_arm()

    controller = ContextController("sn", actual_arms=[1])
    request, rounds = loop.run_speculative_request(
        "http://fake", None, SimpleNamespace(eos_token_id=99), [1], controller, 10,
    )
    assert request["ttft_ms"] is None and request["last_token_latency_ms"] is None
    assert request["e2e_ms"] > 0
    assert rounds[0]["controller_select_time_ms"] == pytest.approx(250)
    assert controller.rounds_selected == 1


def test_cuda_sync_targets_only_the_model_cuda_device(monkeypatch):
    devices = []
    monkeypatch.setattr(loop.torch.cuda, "synchronize", lambda device: devices.append(device))
    loop._synchronize_device(torch.device("cpu"))
    loop._synchronize_device(torch.device("cuda:2"))
    assert devices == [torch.device("cuda:2")]


def test_real_draft_session_synchronizes_prefill_draft_and_commit(monkeypatch):
    from hardware.validate_hw import _FakeCausalLM, _FakeTokenizer

    devices = []
    monkeypatch.setattr(loop, "_synchronize_device", devices.append)
    session = loop.GreedyDraftSession(_FakeCausalLM(), _FakeTokenizer(), [1, 2])
    assert len(devices) == 1
    draft = session.draft(1)
    assert len(devices) == 2
    session.commit(0, [int(draft["draft_ids"][0])])
    assert len(devices) == 3


def test_direct_stream_records_token_and_completion_boundaries_without_chunk_buffering(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(loop.time, "perf_counter", clock.now)

    class Response:
        closed = False

        def raise_for_status(self):
            pass

        def iter_lines(self, chunk_size):
            assert chunk_size == 1
            clock.advance(0.2)
            yield json.dumps({"event": "token", "token_id": 7}).encode()
            clock.advance(0.3)
            yield json.dumps({"event": "done", "generated_tokens": 1,
                              "stop_reason": "eos", "prefill_time_ms": 10,
                              "server_total_ms": 200}).encode()

        def close(self):
            self.closed = True
            clock.advance(1)

    response = Response()
    monkeypatch.setattr(loop.requests, "post", lambda *args, **kwargs: response)
    result = loop.run_direct_target_request("http://fake", [1], 10)
    assert response.closed
    assert result["ttft_ms"] == result["last_token_latency_ms"] == pytest.approx(200)
    assert result["e2e_ms"] == result["completion_latency_ms"] == pytest.approx(500)
    assert result["cleanup_time_ms"] == pytest.approx(1000)
    assert result["request_wall_ms"] == pytest.approx(1500)


def test_direct_stream_closes_connection_on_corrupt_done_count(monkeypatch):
    class Response:
        closed = False

        def raise_for_status(self):
            pass

        def iter_lines(self, chunk_size):
            yield b'{"event":"done","generated_tokens":1}'

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr(loop.requests, "post", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="count disagrees"):
        loop.run_direct_target_request("http://fake", [1], 10)
    assert response.closed
