"""CPU checks for opt-in lazy drafting and explicit commit-time work."""

from types import SimpleNamespace

import pytest
import torch

from hardware import closed_loop as loop


class HistoryModel(torch.nn.Module):
    """Exact fake whose predictions depend on all cached token identities."""

    def __init__(self, on_forward=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.on_forward = on_forward

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        del use_cache
        previous = (torch.empty(0, dtype=torch.long) if past_key_values is None
                    else past_key_values[0][0][0, 0, :, 0].long())
        added = input_ids[0].cpu()
        self.calls.append(added.tolist())
        if self.on_forward is not None:
            self.on_forward(len(added))
        all_tokens = torch.cat((previous, added))
        predictions = (previous.sum() + torch.cumsum(added, dim=0) + 1) % 32
        logits = torch.full((1, len(added), 32), -100.0)
        logits.scatter_(2, predictions.reshape(1, -1, 1), 100.0)
        key = all_tokens.float().reshape(1, 1, -1, 1)
        return SimpleNamespace(logits=logits, past_key_values=((key, key.clone()),))


def make_session(lazy=True, eos=None, model=None):
    model = model or HistoryModel()
    return loop.GreedyDraftSession(
        model, SimpleNamespace(eos_token_id=eos), [2, 4],
        lazy_final_forward=lazy,
    )


def assert_matches_fresh_prefix(session, committed):
    expected = HistoryModel()(torch.tensor([[2, 4] + list(committed)]))
    assert session.kv_tokens == 2 + len(committed)
    torch.testing.assert_close(session.past_key_values[0][0], expected.past_key_values[0][0])
    torch.testing.assert_close(session.next_logits, expected.logits[0, -1])


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("accepted", (0, 1, 2, 4))
def test_rejection_and_full_accept_restore_correct_history(lazy, accepted):
    session = make_session(lazy)
    draft = session.draft(4)
    assert draft["draft_ids"] == [7, 14, 28, 24]
    assert draft["draft_forward_calls"] == (3 if lazy else 4)
    assert draft["draft_kv_tokens_after_draft"] == (5 if lazy else 6)
    assert draft["draft_tokens_pending_forward"] == int(lazy)
    before_commit_calls = len(session.model.calls)
    committed = draft["draft_ids"][:accepted] + [30]
    result = session.commit(accepted, committed)
    assert result["draft_commit_forward_calls"] == 1
    assert len(session.model.calls) == before_commit_calls + 1
    assert session.model.calls[-1] == ([24, 30] if lazy and accepted == 4 else [30])
    assert result["draft_commit_forward_tokens"] == (2 if lazy and accepted == 4 else 1)
    assert_matches_fresh_prefix(session, committed)
    # Following-round output must reflect the committed history, not rejected
    # speculative tokens or stale logits from before cache truncation.
    fresh = loop.GreedyDraftSession(HistoryModel(), SimpleNamespace(eos_token_id=None), [2, 4] + committed)
    assert session.draft(3)["draft_ids"] == fresh.draft(3)["draft_ids"]


def test_lazy_one_token_draft_defers_all_forward_work_to_commit():
    session = make_session()
    draft = session.draft(1)
    assert session.model.calls == [[2, 4]]
    assert draft["draft_ids"] == [7]
    assert session.kv_tokens == 2
    result = session.commit(1, [7, 19])
    assert session.model.calls == [[2, 4], [7, 19]]
    assert result["draft_commit_forward_tokens"] == 2
    assert_matches_fresh_prefix(session, [7, 19])


@pytest.mark.parametrize("accept_eos", (False, True))
def test_shortened_eos_draft_materializes_only_committed_suffix(accept_eos):
    session = make_session(eos=28)
    draft = session.draft(10)
    assert draft["draft_ids"] == [7, 14, 28]
    assert draft["draft_forward_calls"] == 2
    assert session.kv_tokens == 4
    accepted, committed = (3, [7, 14, 28]) if accept_eos else (2, [7, 14, 9])
    result = session.commit(accepted, committed)
    assert result["draft_commit_forward_tokens"] == 1
    assert session.model.calls[-1] == [committed[-1]]
    assert_matches_fresh_prefix(session, committed)


def test_zero_draft_arm_for_one_remaining_token_commits_target_bonus():
    session = make_session()
    draft = session.draft(0)
    assert draft["draft_ids"] == []
    assert draft["draft_forward_calls"] == draft["draft_tokens_pending_forward"] == 0
    assert session.kv_tokens == 2
    result = session.commit(0, [19])
    assert result["draft_commit_forward_tokens"] == 1
    assert_matches_fresh_prefix(session, [19])


@pytest.mark.parametrize("lazy", (False, True))
def test_empty_commit_restores_original_cache_and_logits(lazy):
    session = make_session(lazy)
    session.draft(4)
    before = len(session.model.calls)
    result = session.commit(0, [])
    assert len(session.model.calls) == before
    assert result["draft_commit_forward_calls"] == 0
    assert_matches_fresh_prefix(session, [])
    assert session.draft(1)["draft_ids"] == [7]


def test_full_accept_without_bonus_materializes_pending_last_token():
    session = make_session()
    draft = session.draft(4)
    result = session.commit(4, draft["draft_ids"])
    assert session.model.calls[-1] == [24]
    assert result["draft_commit_forward_tokens"] == 1
    assert_matches_fresh_prefix(session, draft["draft_ids"])


def test_invalid_commit_leaves_pending_draft_available():
    session = make_session()
    draft = session.draft(4)
    before_calls = list(session.model.calls)
    before_cache = session.past_key_values[0][0].clone()
    with pytest.raises(RuntimeError, match="not the draft prefix"):
        session.commit(2, [7, 5, 30])
    assert session.model.calls == before_calls
    torch.testing.assert_close(session.past_key_values[0][0], before_cache)
    session.commit(2, draft["draft_ids"][:2] + [30])
    assert_matches_fresh_prefix(session, [7, 14, 30])


def test_deferred_forward_is_charged_to_commit_timing(monkeypatch):
    timing = {"clock": 0.0, "pending": 0.0}

    def enqueue(token_count):
        timing["pending"] += 0.1 + 0.01 * token_count

    def finish(device):
        assert device.type == "cpu"
        timing["clock"] += timing["pending"]
        timing["pending"] = 0.0

    monkeypatch.setattr(loop.time, "perf_counter", lambda: timing["clock"])
    monkeypatch.setattr(loop, "_synchronize_device", finish)
    session = make_session(model=HistoryModel(enqueue))
    start = timing["clock"]
    draft = session.draft(4)
    result = session.commit(4, draft["draft_ids"] + [30])
    assert draft["draft_time_ms"] == pytest.approx(330.0)
    assert result["draft_commit_time_ms"] == pytest.approx(120.0)
    assert draft["draft_time_ms"] + result["draft_commit_time_ms"] == pytest.approx((timing["clock"] - start) * 1000)
    assert timing["pending"] == 0.0
    assert session.model.calls == [[2, 4], [7], [14], [28], [24, 30]]


def test_lazy_mode_is_opt_in_and_flag_is_explicit():
    model = HistoryModel()
    session = loop.GreedyDraftSession(model, SimpleNamespace(eos_token_id=None), [2, 4])
    assert session.draft(1)["draft_forward_calls"] == 1
    assert session.lazy_final_forward is False
    with pytest.raises(ValueError, match="boolean"):
        make_session(lazy="false")


@pytest.mark.parametrize("cap,expected_rounds,expected_updates", ((1, 1, 0), (3, 1, 0), (8, 2, 1)))
def test_lazy_continuous_request_keeps_cap_censoring_and_cache_alignment(
    monkeypatch, cap, expected_rounds, expected_updates,
):
    target_model = HistoryModel()
    target = {}

    def start(server, context_ids, delay):
        target["state"] = target_model(torch.tensor([context_ids]))
        target["tokens"] = len(context_ids)
        return {"session_id": "cpu-fixture", "target_kv_tokens": len(context_ids),
                "prefill_time_ms": 0.0, "_network_and_server_ms": 0.0}

    def forward_token(token):
        target["state"] = target_model(
            torch.tensor([[token]]), past_key_values=target["state"].past_key_values,
        )
        target["tokens"] += 1

    def verify(server, session_id, draft_ids, delay):
        assert session_id == "cpu-fixture"
        before = target["tokens"]
        committed = []
        for token in draft_ids:
            if token != int(target["state"].logits[0, -1].argmax()):
                break
            committed.append(token)
            forward_token(token)
        accepted = len(committed)
        bonus = int(target["state"].logits[0, -1].argmax())
        committed.append(bonus)
        forward_token(bonus)
        return {"n_accepted": accepted, "committed_token_ids": committed,
                "verify_time_ms": 0.0, "target_kv_tokens_before": before,
                "target_kv_tokens_after": target["tokens"], "stop_reason": None}

    monkeypatch.setattr(loop, "start_target_session", start)
    monkeypatch.setattr(loop, "verify_target_session", verify)
    monkeypatch.setattr(loop, "close_target_session", lambda *args: {"closed": True})
    controller = loop.ServiceController("sn", [4], beta=0.0)
    edge = HistoryModel()
    request, rounds = loop.run_speculative_request(
        "http://cpu-fixture", edge, SimpleNamespace(eos_token_id=None), [2, 4],
        controller, cap, lazy_final_forward=True,
    )
    assert request["generated_tokens"] == cap
    assert request["stop_reason"] == "cap"
    assert request["rounds"] == expected_rounds
    assert request["lazy_final_forward"] is True
    assert request["target_session_closed"] is True
    assert sum(row["feedback_updated"] for row in rounds) == expected_updates
    assert controller.algorithm.t_k.tolist() == [expected_updates]
    for row in rounds:
        assert row["draft_forward_calls"] == max(row["effective_k"] - 1, 0)
        assert row["draft_kv_tokens_after_draft"] == row["draft_kv_tokens_before"] + row["draft_forward_calls"]
        assert row["draft_kv_tokens_after_commit"] == row["target_kv_tokens_after"]
    assert rounds[-1]["arm_censored"] is True
    assert rounds[-1]["feedback_skipped_reason"] == "generation_cap"
    assert target["tokens"] == 2 + cap
