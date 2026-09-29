"""CPU state-machine checks using the real graph engine/session methods."""

import threading
from types import SimpleNamespace

import pytest
import torch

from hardware.graph_draft import DraftGraphEngine, GraphDraftSession
from hardware.validate_lazy_draft import HistoryModel


class StaticHistoryModel(HistoryModel):
    """History-dependent fake with reusable output and fixed cache slots."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=32)

    def forward(self, input_ids, past_key_values, cache_position,
                use_cache=True, return_dict=True, logits_to_keep=1):
        del return_dict, logits_to_keep
        start = int(cache_position[0])
        previous = past_key_values.slots[:start].reshape(1, 1, -1, 1)
        output = super().forward(input_ids, ((previous, previous.clone()),), use_cache)
        past_key_values.slots[start:start + input_ids.shape[1]] = input_ids[0].float()
        return SimpleNamespace(logits=output.logits, past_key_values=past_key_values)


def engine(maximum=32):
    result = DraftGraphEngine.__new__(DraftGraphEngine)
    result.model = StaticHistoryModel()
    result.device = torch.device("cpu")
    result.max_cache_len = maximum
    result._lock = threading.RLock()
    result._active_lease = None
    result._next_lease = 0
    result.position = 0
    result.replay_count = 0
    slots = torch.zeros(maximum)
    result.cache = SimpleNamespace(slots=slots, reset=lambda: slots.zero_())
    result.input_ids = torch.zeros((1, 1), dtype=torch.long)
    result.cache_position = torch.zeros(1, dtype=torch.long)
    result._graph_logits = torch.zeros(32)

    def replay():
        result._graph_logits.copy_(result._eager_forward())

    result.graph = SimpleNamespace(replay=replay)
    return result


def session(backend=None, lazy=True, eos=None, context=(2, 4), cap=20):
    return GraphDraftSession(backend or engine(), SimpleNamespace(eos_token_id=eos),
                             context, lazy_final_forward=lazy, max_new_tokens=cap)


def assert_history(current, context):
    expected = HistoryModel()(torch.tensor([context]))
    assert current.kv_tokens == len(context)
    torch.testing.assert_close(current.next_logits, expected.logits[0, -1])
    torch.testing.assert_close(current.engine.cache.slots[:len(context)], torch.tensor(context).float())


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("accepted", (0, 1, 2, 4))
@pytest.mark.parametrize("bonus", (False, True))
def test_rollback_restore_logits_and_committed_prefix(lazy, accepted, bonus):
    current = session(lazy=lazy)
    result = current.draft(4)
    assert result["draft_ids"] == [7, 14, 28, 24]
    assert result["draft_forward_calls"] == (3 if lazy else 4)
    committed = result["draft_ids"][:accepted] + ([30] if bonus else [])
    output = current.commit(accepted, committed)
    missing = int(bonus) + int(lazy and accepted == 4)
    assert output["draft_commit_forward_tokens"] == missing
    assert output["draft_commit_forward_calls"] == missing
    assert_history(current, [2, 4] + committed)
    # The next round reveals a stale/base/history logits alias immediately.
    expected_next = (sum([2, 4] + committed) + 1) % 32
    assert current.draft(1)["draft_ids"] == [expected_next]
    current.close()


@pytest.mark.parametrize("lazy", (False, True))
def test_policy_stop_and_eos_have_no_later_callbacks(lazy):
    current = session(lazy=lazy)
    callbacks = []

    def stop(position, logits, context_tokens):
        callbacks.append((position, context_tokens, int(logits.argmax())))
        return position < 3

    result = current.draft(8, stop)
    assert result["draft_ids"] == [7, 14]
    assert result["draft_stopped_by_policy"] is True
    assert callbacks == [(1, 2, 7), (2, 3, 14), (3, 4, 28)]
    current.commit(2, [7, 14, 30])
    assert_history(current, [2, 4, 7, 14, 30])
    current.close()
    eos_session = session(lazy=lazy, eos=28)
    eos_draft = eos_session.draft(8)
    assert eos_draft["draft_ids"] == [7, 14, 28]
    assert eos_draft["draft_forward_calls"] == (2 if lazy else 3)
    eos_session.commit(3, [7, 14, 28])
    assert_history(eos_session, [2, 4, 7, 14, 28])
    eos_session.close()


def test_zero_draft_and_one_token_deferred_commit():
    current = session()
    assert current.draft(0)["draft_ids"] == []
    assert current.commit(0, [19])["draft_commit_forward_calls"] == 1
    assert_history(current, [2, 4, 19])
    next_token = int(current.next_logits.argmax())
    assert current.draft(1)["draft_forward_calls"] == 0
    assert current.commit(1, [next_token, 30])["draft_commit_forward_calls"] == 2
    assert_history(current, [2, 4, 19, next_token, 30])
    current.close()


def test_engine_reuse_requires_close_and_resets_other_context():
    backend = engine()
    first = session(backend)
    first.draft(4)
    live = first.execution_state()
    assert live["graph_replays"] == 3
    assert live["final_logical_kv_tokens"] == live["engine_position"] == 5
    assert live["session_closed"] is False
    with pytest.raises(RuntimeError, match="active session"):
        session(backend, context=(9, 3, 2))
    first.close()
    closed = first.execution_state()
    assert closed["session_closed"] is True
    first.close()
    with pytest.raises(RuntimeError, match="closed"):
        first.draft(1)
    second = session(backend, context=(9, 3, 2))
    assert second.engine is backend
    assert_history(second, [9, 3, 2])
    assert torch.count_nonzero(backend.cache.slots[3:]) == 0
    assert second.draft(1)["draft_ids"] == [15]
    assert first.execution_state() == closed
    second.close()


def test_capacity_failures_do_not_silently_truncate_or_lose_lease():
    backend = engine(maximum=5)
    with pytest.raises(ValueError, match="exceeds"):
        session(backend, cap=4)
    current = session(backend, cap=3)
    with pytest.raises(ValueError, match="capacity"):
        current.draft(4)
    draft = current.draft(3)
    with pytest.raises(ValueError, match="capacity"):
        current.commit(3, draft["draft_ids"] + [30])
    current.commit(3, draft["draft_ids"])
    assert_history(current, [2, 4] + draft["draft_ids"])
    with pytest.raises(ValueError, match="capacity"):
        current.draft(1)
    current.close()


@pytest.mark.parametrize("failure", ("target_prefill", "prefill_length", "verification"))
def test_request_releases_graph_lease_even_when_target_cleanup_raises(monkeypatch, failure):
    from hardware import closed_loop as loop
    backend, sessions, target_close_calls = engine(), [], []

    def factory(model, tokenizer, context_ids, **kwargs):
        value = GraphDraftSession(backend, tokenizer, context_ids, max_new_tokens=4, **kwargs)
        sessions.append(value)
        return value

    def start(*args):
        if failure == "target_prefill":
            raise LookupError("target prefill failed")
        return {"session_id": "cpu-graph", "target_kv_tokens": 3 if failure == "prefill_length" else 2,
                "prefill_time_ms": 0, "_network_and_server_ms": 0}

    def verify(*args):
        raise ValueError("verification failed")

    def close(*args):
        target_close_calls.append(args)
        raise RuntimeError("target cleanup failed")

    monkeypatch.setattr(loop, "start_target_session", start)
    monkeypatch.setattr(loop, "verify_target_session", verify)
    monkeypatch.setattr(loop, "close_target_session", close)
    controller = loop.ServiceController("fixed", [1], fixed_arm=1)
    expected = LookupError if failure == "target_prefill" else RuntimeError
    with pytest.raises(expected):
        loop.run_speculative_request("cpu", backend.model, SimpleNamespace(eos_token_id=None),
            [2, 4], controller, 4, lazy_final_forward=True, draft_session_factory=factory)
    assert len(sessions) == 1
    assert sessions[0].execution_state()["session_closed"] is True
    assert backend._active_lease is None
    assert len(target_close_calls) == (0 if failure == "target_prefill" else 1)
    replacement = session(backend)
    replacement.close()
