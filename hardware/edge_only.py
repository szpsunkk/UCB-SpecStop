"""Pure edge autoregression: no verifier, calibration or target feedback."""
import json
import time

from hardware.closed_loop import token_ids_sha256
from hardware.graph_draft import synchronize
from hardware.runtime_eos import normalize_eos_ids


def run_edge_only_request(engine, context_ids, max_new_tokens, eos_token_id, tokenizer=None):
    start_ns, wall_start_ns = time.monotonic_ns(), time.time_ns()
    start = time.perf_counter()
    lease, logits = engine.acquire(context_ids, max_new_tokens=max_new_tokens)
    prefill_ms = (time.perf_counter() - start) * 1000
    tokens, ttft, stop_reason = [], None, "cap"
    try:
        for i in range(max_new_tokens):
            token = int(logits.argmax())
            if token in normalize_eos_ids(eos_token_id):
                stop_reason = "eos"
                break
            tokens.append(token)
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            if tokenizer is not None:
                from hardware.answer_stopping import token_answer_complete
                if token_answer_complete(tokenizer, tokens):
                    stop_reason = "answer"
                    break
            if i + 1 < max_new_tokens:
                logits = engine.step(lease, token)
        synchronize(engine.device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        end_ns = time.monotonic_ns()
    finally:
        cleanup_start = time.perf_counter()
        engine.release(lease)
        cleanup_ms = (time.perf_counter() - cleanup_start) * 1000
    return dict(method="edge_only", completion_latency_ms=elapsed_ms, e2e_ms=elapsed_ms,
        ttft_ms=ttft, generated_tokens=len(tokens), generation_token_cap=max_new_tokens,
        initial_context_tokens=len(context_ids), stop_reason=stop_reason, rounds=0,
        edge_prefill_time_ms=prefill_ms, edge_model_work_ms=elapsed_ms,
        controller_time_ms=0.0, request_payload_bytes=0, response_payload_bytes=0,
        delivered_token_ids_json=json.dumps(tokens, separators=(",", ":")),
        delivered_token_ids_sha256=token_ids_sha256(tokens),
        monotonic_start_ns=start_ns, monotonic_end_ns=end_ns, wall_start_ns=wall_start_ns,
        wall_end_ns=time.time_ns(), cleanup_time_ms=cleanup_ms,
        edge_eos_token_id=eos_token_id, cloud_calls=0)
