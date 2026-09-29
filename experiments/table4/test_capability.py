"""Check answer extraction and that edge-only has an independent generation loop."""
import importlib.util
import json
from pathlib import Path
import sys

from experiments.table4.score import extract_answer, score
from hardware import edge_only
from hardware.answer_stopping import answer_complete
from hardware.runtime_eos import model_eos_ids, tokenizer_eos_ids


def test_scoring_exact_numbers_and_fractions():
    assert score("#### 1,200", "1200")["flexible_correct"]
    assert score(r"Hence $\boxed{1/2}$.", "0.5")["strict_correct"]
    assert score("Final answer: $-12.5", "-12.5")["strict_correct"]
    assert not score("Answer: 13", "12")["flexible_correct"]
    assert not score("No answer.", "12")["flexible_correct"]
    assert not score("I calculated 12.", "12")["strict_correct"]
    assert score("I calculated 12.", "12")["flexible_correct"]
    assert score("There are 30 apples; final answer: 12.", "12")["strict_correct"]


class FakeLogits:
    def __init__(self, token):
        self.token = token

    def argmax(self):
        return self.token


class FakeEngine:
    device = "fake"

    def __init__(self, sequence):
        self.sequence = sequence
        self.calls, self.steps = [], []

    def acquire(self, context_ids, max_new_tokens):
        self.calls.append((list(context_ids), max_new_tokens))
        return 1, FakeLogits(self.sequence[0])

    def step(self, lease, token):
        self.steps.append(token)
        return FakeLogits(self.sequence[len(self.steps)])

    def release(self, lease):
        self.released = lease


def test_edge_only_uses_own_tokens_and_stops_before_eos(monkeypatch):
    monkeypatch.setattr(edge_only, "synchronize", lambda _: None)
    engine = FakeEngine([12, 13, 99])
    result = edge_only.run_edge_only_request(engine, [1, 2], 10, 99)
    assert json.loads(result["delivered_token_ids_json"]) == [12, 13]
    assert engine.steps == [12, 13] and engine.calls == [([1, 2], 10)]
    assert engine.released == 1 and result["cloud_calls"] == 0
    assert result["stop_reason"] == "eos"
    assert result["request_payload_bytes"] == result["response_payload_bytes"] == 0


def test_edge_only_cap_has_no_unused_final_forward(monkeypatch):
    monkeypatch.setattr(edge_only, "synchronize", lambda _: None)
    engine = FakeEngine([12, 13, 14])
    result = edge_only.run_edge_only_request(engine, [1], 2, 99)
    assert json.loads(result["delivered_token_ids_json"]) == [12, 13]
    assert engine.steps == [12] and result["stop_reason"] == "cap"


def test_edge_only_empty_eos_is_recorded(monkeypatch):
    monkeypatch.setattr(edge_only, "synchronize", lambda _: None)
    engine = FakeEngine([99])
    result = edge_only.run_edge_only_request(engine, [1], 10, 99)
    assert result["generated_tokens"] == 0 and result["ttft_ms"] is None
    assert result["stop_reason"] == "eos" and engine.released == 1


def test_native_multiple_eos_and_independent_edge_termination(monkeypatch):
    from types import SimpleNamespace
    tokenizer = SimpleNamespace(eos_token_id=99)
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[99, 98]))
    assert model_eos_ids(model, tokenizer) == (99, 98)
    tokenizer.specstop_eos_ids = (99, 98)
    assert tokenizer_eos_ids(tokenizer) == (99, 98)
    monkeypatch.setattr(edge_only, "synchronize", lambda _: None)
    engine = FakeEngine([12, 98, 15])
    result = edge_only.run_edge_only_request(engine, [1], 10, (99, 98))
    assert json.loads(result["delivered_token_ids_json"]) == [12]
    assert result["stop_reason"] == "eos" and engine.steps == [12]


def test_answer_stopping_waits_for_number_completion():
    for text in ("Final answer: 1", "Final answer: 1.", "Final answer: 1.2", "Final answer: 1,2",
                 "Final answer: 1 /", "Final answer: 1 /2", "Final answer: 1 /2.5"):
        assert not answer_complete(text)
    for text in ("Final answer: 1.25.\n", "Final answer: 1,200\n", r"The result is $\boxed{12}$.",
                 "#### -12\n", "Final Answer: 70.Human", "Final answer: **70**", "Final answer: 1 /2.5\n"):
        assert answer_complete(text), text
    assert not answer_complete("The problem mentions 70 helmets.\n")


def test_first_explicit_answer_wins_over_later_invented_questions():
    assert score("Final answer: 70.Human: another question. Final answer: 150.", "70")["strict_correct"]
    assert score("Final answer: 70.Human: another question. #### 150.", "70")["strict_correct"]


def test_edge_only_answer_stop_before_unused_forward(monkeypatch):
    monkeypatch.setattr(edge_only, "synchronize", lambda _: None)
    class Tokenizer:
        def decode(self, tokens, skip_special_tokens):
            return "Final answer: 12\n" if len(tokens) == 2 else "Final answer: 1"
    engine = FakeEngine([12, 13, 14])
    result = edge_only.run_edge_only_request(engine, [1], 10, 99, tokenizer=Tokenizer())
    assert result["stop_reason"] == "answer" and engine.steps == [12]
