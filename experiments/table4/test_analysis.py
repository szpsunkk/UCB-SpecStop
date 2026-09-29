"""Check inferential units and paired statistical calculations."""
from pathlib import Path
import sys

import pytest

from experiments.table4.analyze import aggregate, holm, paired_test, quality_summary


def test_holm_uses_full_family_and_original_order():
    assert holm([0.04, 0.001, 0.03, 0.5]) == pytest.approx([0.09, 0.004, 0.09, 0.5])


def test_mcnemar_counts_discordant_questions_not_request_replicates():
    pairs = [dict(edge={"ok": False}, target={"ok": True}) for _ in range(6)]
    pairs += [dict(edge={"ok": True}, target={"ok": True}) for _ in range(2)]
    result = quality_summary(pairs, "ok")
    assert result["delta_pp"] == 75
    assert result["exact_mcnemar_p"] == pytest.approx(0.03125)
    assert result["n"] == 8 and result["improved_questions"] == 6


def test_pair_sign_convention():
    result = paired_test([-1, -2, -3, -4, -5, -6, -7, -8])
    assert result["mean_difference_ms_token"] == -4.5
    assert result["ci_high"] < 0
    assert result["sign_flip_p"] == pytest.approx(2/256)


def test_cost_is_session_ratio_then_equal_session_mean():
    rows = []
    fields = ("application_kib", "generation_interactions", "verification_rounds", "verification_rpc_ms",
              "verification_server_ms", "verification_residual_ms", "edge_generation_wall_ms",
              "target_stream_residual_ms", "controller_time_ms", "cleanup_time_ms")
    for session in range(8):
        for token_count, elapsed in ((1, 10), (9, 18), (1, 10), (9, 18)):
            rows.append(dict(measurement_session=str(session), generated_tokens=token_count,
                completion_latency_ms=elapsed*(session+1), ttft_ms=1, stop_reason="answer",
                **{field: 0 for field in fields}))
    summary, ratios = aggregate(rows)
    assert ratios["0"] == pytest.approx(2.8)
    assert summary["cost_ms_token"] == pytest.approx(2.8*4.5)
    assert summary["requests"] == 32 and summary["sessions"] == 8
