"""
cloud_server.py — Run on the RTX 3090 server (cloud side).

Loads the target model, exposes HTTP endpoints:
  POST /verify   — verify k draft tokens, return n_accepted + bonus
  POST /session/start — initialize a persistent target-model KV session
  POST /session/verify — verify one selected arm and update the session KV
  POST /session/close — release a target-model KV session
  POST /generate/stream — stream direct target-model greedy generation
  GET  /ping     — latency probe for RTT measurement
  GET  /telemetry — NVML/process snapshot for request-window accounting
  POST /telemetry/reset-peak — reset the sequential-request peak counter

Protocol:
  - context_ids + draft_ids required
  - draft_log_probs optional: when provided, uses the historical selected-token
    p/q acceptance proxy; exact residual sampling would require the complete
    draft distribution;
    when absent, uses exact target/draft argmax matching and returns the target
    argmax at the first mismatch, preserving target-model greedy decoding

Usage:
  python cloud_server.py --model Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from hardware.runtime_eos import model_eos_ids
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from hardware.kv_cache import cache_sequence_length, crop_past_key_values
except ModuleNotFoundError:  # Direct execution from the hardware directory.
    from kv_cache import cache_sequence_length, crop_past_key_values

try:
    import pynvml
except ImportError:  # Optional: inference remains available without NVML telemetry.
    pynvml = None

app = FastAPI()

_model = None
_tokenizer = None
_model_ref = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_target_dtype = None
_session_verifier = "parallel"
_numerical_guard_margin = 0.10
_attention_implementation = None
_output_head_precision = "native"
_nvml_handle = None
_model_lock = threading.RLock()


@dataclass
class _TargetSession:
    past_key_values: Any
    next_logits: torch.Tensor
    target_kv_tokens: int
    created_monotonic: float
    last_access_monotonic: float
    original_context_ids: List[int] = field(default_factory=list)
    committed_token_ids: List[int] = field(default_factory=list)


_target_sessions: Dict[str, _TargetSession] = {}


def _tokenizer_vocab_sha256() -> Optional[str]:
    if _tokenizer is None:
        return None
    vocabulary = sorted(
        (str(token), int(token_id))
        for token, token_id in _tokenizer.get_vocab().items()
    )
    payload = json.dumps(
        vocabulary, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _entrypoint_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _init_nvml() -> None:
    global _nvml_handle
    if pynvml is None or not torch.cuda.is_available():
        return
    try:
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        _nvml_handle = None


def _nvml_snapshot() -> Dict[str, Optional[float]]:
    snapshot = {"power_w": None, "memory_mb": None, "energy_mj": None}
    if _nvml_handle is None:
        return snapshot
    try:
        snapshot["power_w"] = pynvml.nvmlDeviceGetPowerUsage(_nvml_handle) / 1000.0
    except pynvml.NVMLError:
        pass
    try:
        memory = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        snapshot["memory_mb"] = memory.used / (1024.0**2)
    except pynvml.NVMLError:
        pass
    try:
        snapshot["energy_mj"] = float(
            pynvml.nvmlDeviceGetTotalEnergyConsumption(_nvml_handle)
        )
    except pynvml.NVMLError:
        pass
    return snapshot


_init_nvml()


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    context_ids: List[int]
    draft_ids: List[int]
    # Log-probs of each draft token under the draft model.
    # When provided, enables the historical selected-token p/q acceptance
    # proxy. Exact residual sampling would require the full draft distribution.
    # When absent, target/draft argmax matching preserves greedy decoding.
    draft_log_probs: Optional[List[float]] = None
    # Optional per-request seed: when provided, rejection sampling and bonus
    # sampling become deterministic for this (prompt, draft_ids) pair, enabling
    # paired across-strategy comparisons. Pass the same seed across strategies
    # for the same prompt to remove sampling noise from the comparison.
    seed: Optional[int] = None
    # Diagnostics are opt-in and are excluded from performance runs.
    return_diagnostics: bool = False


class VerifyResponse(BaseModel):
    verification_mode: str
    n_accepted: int
    bonus_token_id: int
    verify_time_ms: float
    # Timing breakdown (review B1) — backwards compatible: clients that ignore
    # these fields still work; clients that read them get a clock-independent
    # decomposition of server-side wall time.
    server_recv_to_verify_start_ms: float = 0.0
    verify_split_ms: float = 0.0
    pack_split_ms: float = 0.0
    cloud_power_w_start: Optional[float] = None
    cloud_power_w_end: Optional[float] = None
    cloud_memory_mb_start: Optional[float] = None
    cloud_memory_mb_end: Optional[float] = None
    cloud_energy_j: Optional[float] = None
    cloud_process_peak_allocated_mb: float = 0.0
    diagnostics: Optional[Dict[str, Any]] = None


class TelemetryResponse(BaseModel):
    power_w: Optional[float] = None
    memory_mb: Optional[float] = None
    total_energy_j: Optional[float] = None
    process_peak_allocated_mb: float = 0.0


class SessionStartRequest(BaseModel):
    context_ids: List[int]


class SessionStartResponse(BaseModel):
    session_id: str
    target_kv_tokens: int
    prefill_time_ms: float
    cloud_power_w_start: Optional[float] = None
    cloud_power_w_end: Optional[float] = None
    cloud_memory_mb_start: Optional[float] = None
    cloud_memory_mb_end: Optional[float] = None
    cloud_energy_j: Optional[float] = None
    cloud_process_peak_allocated_mb: float = 0.0


class SessionVerifyRequest(BaseModel):
    session_id: str
    draft_ids: List[int]


class SessionVerifyResponse(BaseModel):
    verification_mode: str = "greedy_exact_session"
    numerical_guard_triggered: bool = False
    canonical_rebuild_tokens: int = 0
    n_accepted: int
    bonus_token_id: Optional[int] = None
    committed_token_ids: List[int]
    stop_reason: Optional[str] = None
    target_kv_tokens_before: int
    target_kv_tokens_after: int
    verify_time_ms: float
    cloud_power_w_start: Optional[float] = None
    cloud_power_w_end: Optional[float] = None
    cloud_memory_mb_start: Optional[float] = None
    cloud_memory_mb_end: Optional[float] = None
    cloud_energy_j: Optional[float] = None
    cloud_process_peak_allocated_mb: float = 0.0


class SessionCloseRequest(BaseModel):
    session_id: str


class DirectGenerateRequest(BaseModel):
    context_ids: List[int]
    max_new_tokens: int
    answer_stop: bool = False


def _model_device() -> torch.device:
    if _model is None:
        raise RuntimeError("target model is not loaded")
    return next(_model.parameters()).device


def _synchronize_model_device() -> None:
    """Complete queued target work before returning a measured RPC boundary."""
    device = _model_device()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _energy_delta_j(
    start: Dict[str, Optional[float]], end: Dict[str, Optional[float]]
) -> Optional[float]:
    if start["energy_mj"] is None or end["energy_mj"] is None:
        return None
    if end["energy_mj"] < start["energy_mj"]:
        return None
    return (end["energy_mj"] - start["energy_mj"]) / 1000.0


def _cloud_peak_allocated_mb() -> float:
    return (
        torch.cuda.max_memory_allocated() / (1024.0**2)
        if torch.cuda.is_available()
        else 0.0
    )


def _forward_token(past_key_values: Any, token_id: int) -> tuple[Any, torch.Tensor]:
    token = torch.tensor([[token_id]], device=_model_device(), dtype=torch.long)
    output = _model(token, past_key_values=past_key_values, use_cache=True)
    return output.past_key_values, output.logits[0, -1]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "ts": time.time(),
        "protocol_version": 3,
        "server_entrypoint_sha256": _entrypoint_sha256(),
        "model": _model_ref,
        "device": _device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "target_dtype": str(_target_dtype),
        "session_verifier": _session_verifier,
        "numerical_guard_margin": _numerical_guard_margin,
        "canonical_rebuild_policy": "original_prefill_then_single_token_history",
        "attention_implementation": _attention_implementation,
        "output_head_precision": _output_head_precision,
        "tokenizer_vocab_size": (
            len(_tokenizer) if _tokenizer is not None else None
        ),
        "tokenizer_vocab_sha256": _tokenizer_vocab_sha256(),
        "tokenizer_eos_token_id": (
            _tokenizer.eos_token_id if _tokenizer is not None else None
        ),
        "generation_eos_token_ids": list(model_eos_ids(_model, _tokenizer)),
        "nvml_telemetry": _nvml_handle is not None,
        "request_window_telemetry": True,
        "verification_modes": ["greedy_exact", "acceptance_ratio_proxy"],
        "session_verification": True,
        "session_no_speculation_arm": True,
        "direct_target_streaming": True,
        "active_target_sessions": len(_target_sessions),
        "session_response_cuda_synchronized": True,
    }


@app.get("/telemetry", response_model=TelemetryResponse)
def telemetry():
    """Return a process-local GPU snapshot for request-window accounting."""

    snapshot = _nvml_snapshot()
    peak_allocated_mb = (
        torch.cuda.max_memory_allocated() / (1024.0**2)
        if torch.cuda.is_available()
        else 0.0
    )
    total_energy_j = (
        snapshot["energy_mj"] / 1000.0
        if snapshot["energy_mj"] is not None
        else None
    )
    return TelemetryResponse(
        power_w=snapshot["power_w"],
        memory_mb=snapshot["memory_mb"],
        total_energy_j=total_energy_j,
        process_peak_allocated_mb=peak_allocated_mb,
    )


@app.post("/telemetry/reset-peak", response_model=TelemetryResponse)
def reset_telemetry_peak():
    """Reset PyTorch's peak counter before a sequential request experiment."""

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return telemetry()


@app.post("/session/start", response_model=SessionStartResponse)
def start_session(req: SessionStartRequest):
    """Prefill a target KV cache for one continuous speculative request."""

    if not req.context_ids:
        raise HTTPException(status_code=400, detail="context_ids must not be empty")
    t_start = time.perf_counter()
    gpu_start = _nvml_snapshot()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with _model_lock, torch.inference_mode():
        context = torch.tensor(
            [req.context_ids], device=_model_device(), dtype=torch.long
        )
        output = _model(context, use_cache=True)
        kv_tokens = cache_sequence_length(output.past_key_values)
        if kv_tokens != len(req.context_ids):
            raise RuntimeError(
                f"target KV length {kv_tokens} != context length {len(req.context_ids)}"
            )
        now = time.monotonic()
        session_id = uuid.uuid4().hex
        _target_sessions[session_id] = _TargetSession(
            past_key_values=output.past_key_values,
            next_logits=output.logits[0, -1],
            target_kv_tokens=kv_tokens,
            created_monotonic=now,
            last_access_monotonic=now,
            original_context_ids=[int(token) for token in req.context_ids],
        )
        _synchronize_model_device()

    gpu_end = _nvml_snapshot()
    return SessionStartResponse(
        session_id=session_id,
        target_kv_tokens=kv_tokens,
        prefill_time_ms=(time.perf_counter() - t_start) * 1000.0,
        cloud_power_w_start=gpu_start["power_w"],
        cloud_power_w_end=gpu_end["power_w"],
        cloud_memory_mb_start=gpu_start["memory_mb"],
        cloud_memory_mb_end=gpu_end["memory_mb"],
        cloud_energy_j=_energy_delta_j(gpu_start, gpu_end),
        cloud_process_peak_allocated_mb=_cloud_peak_allocated_mb(),
    )


class _NumericalTie(Exception):
    pass


def _check_numerical_margin(logits):
    """Request canonical computation at a close greedy decision.

    The margin is a practical numerical guard, not a certified forward-error
    bound. Complete request output equality remains an experimental gate.
    """
    values = torch.topk(logits, 2).values
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("nonfinite target logits")
    if float(values[0] - values[1]) < _numerical_guard_margin:
        raise _NumericalTie()


def _canonical_fallback(req, state, t_start, gpu_start):
    """Rebuild the same original-prefill/one-token history used by direct AR."""
    history = list(state.committed_token_ids)
    try:
        context = torch.tensor([state.original_context_ids], device=_model_device(), dtype=torch.long)
        output = _model(context, use_cache=True)
        past, logits = output.past_key_values, output.logits[0, -1]
        for token in history:
            past, logits = _forward_token(past, token)
        if cache_sequence_length(past) != state.target_kv_tokens:
            raise RuntimeError("canonical history reconstruction length differs")
        state.past_key_values, state.next_logits = past, logits
        response = verify_session_sequential(req)
        end = _nvml_snapshot()
        response.verification_mode = "greedy_guarded_session"
        response.numerical_guard_triggered = True
        response.canonical_rebuild_tokens = len(history)
        response.cloud_power_w_start = gpu_start["power_w"]
        response.cloud_power_w_end = end["power_w"]
        response.cloud_energy_j = _energy_delta_j(gpu_start, end)
        # Includes the original attempted parallel pass and reconstruction.
        response.verify_time_ms = (time.perf_counter() - t_start) * 1000.0
        return response
    except Exception:
        _target_sessions.pop(req.session_id, None)
        raise


@app.post("/session/verify", response_model=SessionVerifyResponse)
def verify_session(req: SessionVerifyRequest):
    """Verify only the selected arm, crop rejected KV entries, and commit output."""

    if _session_verifier == "sequential":
        return verify_session_sequential(req)

    t_start = time.perf_counter()
    gpu_start = _nvml_snapshot()
    with _model_lock, torch.inference_mode():
        state = _target_sessions.get(req.session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown target session")

        kv_before = cache_sequence_length(state.past_key_values)
        if kv_before != state.target_kv_tokens:
            _target_sessions.pop(req.session_id, None)
            raise RuntimeError("target KV bookkeeping changed outside the session")

        draft_output = None
        try:
            if req.draft_ids:
                draft = torch.tensor(
                    [req.draft_ids], device=_model_device(), dtype=torch.long
                )
                draft_output = _model(
                    draft,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )
                trial_past = draft_output.past_key_values
            else:
                trial_past = state.past_key_values

            accepted = 0
            eos_token_ids = model_eos_ids(_model, _tokenizer)
            for index, draft_token in enumerate(req.draft_ids):
                target_logits = (
                    state.next_logits
                    if index == 0
                    else draft_output.logits[0, index - 1]
                )
                _check_numerical_margin(target_logits)
                if int(target_logits.argmax()) != int(draft_token):
                    break
                accepted += 1
                if int(draft_token) in eos_token_ids:
                    break

            accepted_ids = [int(value) for value in req.draft_ids[:accepted]]
            accepted_eos = bool(
                accepted_ids
                and accepted_ids[-1] in eos_token_ids
            )
            trial_past = crop_past_key_values(trial_past, kv_before + accepted)

            if accepted_eos:
                bonus_token_id = None
                committed_ids = accepted_ids
                stop_reason = "eos"
                next_logits = draft_output.logits[0, accepted - 1]
                committed_past = trial_past
            else:
                bonus_logits = (
                    state.next_logits
                    if accepted == 0
                    else draft_output.logits[0, accepted - 1]
                )
                _check_numerical_margin(bonus_logits)
                bonus_token_id = int(bonus_logits.argmax())
                committed_ids = accepted_ids + [bonus_token_id]
                stop_reason = (
                    "eos"
                    if bonus_token_id in eos_token_ids
                    else None
                )
                committed_past, next_logits = _forward_token(
                    trial_past, bonus_token_id
                )

            kv_after = cache_sequence_length(committed_past)
            expected_kv_after = kv_before + len(committed_ids)
            if kv_after != expected_kv_after:
                raise RuntimeError(
                    f"target KV length {kv_after} != expected {expected_kv_after}"
                )
            state.past_key_values = committed_past
            state.next_logits = next_logits
            state.target_kv_tokens = kv_after
            state.last_access_monotonic = time.monotonic()
            state.committed_token_ids.extend(committed_ids)
            _synchronize_model_device()
        except _NumericalTie:
            return _canonical_fallback(req, state, t_start, gpu_start)
        except Exception:
            # A partially mutated cache cannot be resumed safely.
            _target_sessions.pop(req.session_id, None)
            raise

    gpu_end = _nvml_snapshot()
    return SessionVerifyResponse(
        n_accepted=accepted,
        bonus_token_id=bonus_token_id,
        committed_token_ids=committed_ids,
        stop_reason=stop_reason,
        target_kv_tokens_before=kv_before,
        target_kv_tokens_after=kv_after,
        verify_time_ms=(time.perf_counter() - t_start) * 1000.0,
        cloud_power_w_start=gpu_start["power_w"],
        cloud_power_w_end=gpu_end["power_w"],
        cloud_memory_mb_start=gpu_start["memory_mb"],
        cloud_memory_mb_end=gpu_end["memory_mb"],
        cloud_energy_j=_energy_delta_j(gpu_start, gpu_end),
        cloud_process_peak_allocated_mb=_cloud_peak_allocated_mb(),
    )


def verify_session_sequential(req: SessionVerifyRequest):
    """Canonical AR reference: one target forward per committed token.

    This deliberately forgoes parallel verification. It is a correctness
    reference and an explicitly timed alternative, not a free numeric repair.
    """
    t_start = time.perf_counter()
    gpu_start = _nvml_snapshot()
    with _model_lock, torch.inference_mode():
        state = _target_sessions.get(req.session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown target session")
        kv_before = cache_sequence_length(state.past_key_values)
        if kv_before != state.target_kv_tokens:
            _target_sessions.pop(req.session_id, None)
            raise RuntimeError("target KV bookkeeping changed outside the session")
        accepted, committed_ids = 0, []
        bonus_token_id, stop_reason = None, None
        past, logits = state.past_key_values, state.next_logits
        eos = model_eos_ids(_model, _tokenizer)
        try:
            for draft_token in req.draft_ids:
                token = int(logits.argmax())
                if token != int(draft_token):
                    break
                accepted += 1
                committed_ids.append(token)
                past, logits = _forward_token(past, token)
                if token in eos:
                    stop_reason = "eos"
                    break
            if stop_reason is None:
                bonus_token_id = int(logits.argmax())
                committed_ids.append(bonus_token_id)
                past, logits = _forward_token(past, bonus_token_id)
                if bonus_token_id in eos:
                    stop_reason = "eos"
            kv_after = cache_sequence_length(past)
            if kv_after != kv_before + len(committed_ids):
                raise RuntimeError("sequential target KV length disagrees with output")
            state.past_key_values, state.next_logits = past, logits
            state.target_kv_tokens = kv_after
            state.last_access_monotonic = time.monotonic()
            state.committed_token_ids.extend(committed_ids)
            _synchronize_model_device()
        except Exception:
            _target_sessions.pop(req.session_id, None)
            raise
    gpu_end = _nvml_snapshot()
    return SessionVerifyResponse(
        verification_mode="greedy_sequential_session",
        n_accepted=accepted, bonus_token_id=bonus_token_id,
        committed_token_ids=committed_ids, stop_reason=stop_reason,
        target_kv_tokens_before=kv_before, target_kv_tokens_after=kv_after,
        verify_time_ms=(time.perf_counter() - t_start) * 1000.0,
        cloud_power_w_start=gpu_start["power_w"], cloud_power_w_end=gpu_end["power_w"],
        cloud_memory_mb_start=gpu_start["memory_mb"], cloud_memory_mb_end=gpu_end["memory_mb"],
        cloud_energy_j=_energy_delta_j(gpu_start, gpu_end),
        cloud_process_peak_allocated_mb=_cloud_peak_allocated_mb(),
    )


@app.post("/session/close")
def close_session(req: SessionCloseRequest):
    with _model_lock:
        state = _target_sessions.pop(req.session_id, None)
    return {
        "closed": state is not None,
        "target_kv_tokens": state.target_kv_tokens if state is not None else None,
    }


@app.post("/generate/stream")
def stream_direct_generation(req: DirectGenerateRequest):
    """Stream exact greedy target tokens over one persistent HTTP response."""

    if not req.context_ids:
        raise HTTPException(status_code=400, detail="context_ids must not be empty")
    if req.max_new_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_new_tokens must be positive")
    received = time.perf_counter()

    def events():
        with _model_lock, torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            gpu_start = _nvml_snapshot()
            context = torch.tensor(
                [req.context_ids], device=_model_device(), dtype=torch.long
            )
            prefill_start = time.perf_counter()
            output = _model(context, use_cache=True)
            past = output.past_key_values
            next_logits = output.logits[0, -1]
            _synchronize_model_device()
            prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
            eos_token_ids = model_eos_ids(_model, _tokenizer)
            delivered = 0
            answer_tokens = []
            first_token_ms = None
            stop_reason = "cap"

            for _ in range(req.max_new_tokens):
                token_id = int(next_logits.argmax())
                if token_id in eos_token_ids:
                    stop_reason = "eos"
                    break
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - received) * 1000.0
                delivered += 1
                answer_tokens.append(token_id)
                yield json.dumps(
                    {
                        "event": "token",
                        "token_id": token_id,
                        "index": delivered - 1,
                        "server_elapsed_ms": (
                            time.perf_counter() - received
                        ) * 1000.0,
                    },
                    separators=(",", ":"),
                ) + "\n"
                if req.answer_stop:
                    from hardware.answer_stopping import token_answer_complete
                    if token_answer_complete(_tokenizer, answer_tokens):
                        stop_reason = "answer"
                        break
                if delivered == req.max_new_tokens:
                    break
                past, next_logits = _forward_token(past, token_id)

            gpu_end = _nvml_snapshot()
            done = {
                "event": "done",
                "stop_reason": stop_reason,
                "generated_tokens": delivered,
                "prefill_time_ms": prefill_ms,
                "first_token_time_ms": first_token_ms,
                "server_total_ms": (time.perf_counter() - received) * 1000.0,
                "target_kv_tokens": cache_sequence_length(past),
                "cloud_power_w_start": gpu_start["power_w"],
                "cloud_power_w_end": gpu_end["power_w"],
                "cloud_memory_mb_start": gpu_start["memory_mb"],
                "cloud_memory_mb_end": gpu_end["memory_mb"],
                "cloud_energy_j": _energy_delta_j(gpu_start, gpu_end),
                "cloud_process_peak_allocated_mb": _cloud_peak_allocated_mb(),
            }
            yield json.dumps(done, separators=(",", ":")) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    t_recv = time.perf_counter()
    gpu_start = _nvml_snapshot()

    if req.seed is not None:
        rng = random.Random(req.seed)
        bonus_gen = torch.Generator(device=_device).manual_seed(int(req.seed))
    else:
        rng = random
        bonus_gen = None

    ctx_len = len(req.context_ids)
    k = len(req.draft_ids)

    context = torch.tensor([req.context_ids], device=_device)
    draft = torch.tensor([req.draft_ids], device=_device)
    full_seq = torch.cat([context, draft], dim=1)

    t_verify_start = time.perf_counter()

    with torch.no_grad():
        logits = _model(full_seq).logits          # (1, ctx_len+k, vocab)
        log_probs = (
            torch.log_softmax(logits[0], dim=-1)
            if req.draft_log_probs is not None
            else None
        )

    n_accepted = 0

    if req.draft_log_probs is not None:
        # Selected-token acceptance proxy: min(1, p_target/p_draft).
        for i, (draft_tok, draft_lp) in enumerate(
            zip(req.draft_ids, req.draft_log_probs)
        ):
            # logits[0, ctx_len+i-1] is the distribution PREDICTING position ctx_len+i
            target_lp = log_probs[ctx_len + i - 1, draft_tok].item()
            accept_prob = min(1.0, math.exp(target_lp - draft_lp))
            if rng.random() < accept_prob:
                n_accepted += 1
            else:
                break
    else:
        # Exact greedy protocol: accept while draft and target argmax agree.
        for i, draft_tok in enumerate(req.draft_ids):
            target_tok = int(logits[0, ctx_len + i - 1].argmax())
            if draft_tok == target_tok:
                n_accepted += 1
            else:
                break

    t_verify_done = time.perf_counter()

    # Position ctx_len+n_accepted-1 predicts token ctx_len+n_accepted. The
    # log-prob protocol does not transmit the full draft distribution needed
    # for exact residual sampling, so it is explicitly labelled a proxy. The
    # no-log-prob protocol exactly preserves target-model greedy decoding.
    bonus_logits = logits[0, ctx_len + n_accepted - 1]
    if req.draft_log_probs is not None:
        verification_mode = "acceptance_ratio_proxy"
        bonus_probs = torch.softmax(bonus_logits, dim=-1)
        if bonus_gen is not None:
            bonus_token_id = int(
                torch.multinomial(bonus_probs, 1, generator=bonus_gen).item()
            )
        else:
            bonus_token_id = int(torch.multinomial(bonus_probs, 1).item())
    else:
        verification_mode = "greedy_exact"
        bonus_token_id = int(bonus_logits.argmax().item())

    diagnostics = None
    if req.return_diagnostics:
        predicted = logits[0, ctx_len - 1:ctx_len + k]
        top_values, top_ids = torch.topk(predicted.float(), k=2, dim=-1)
        diagnostics = {
            "target_argmax_ids": predicted.argmax(dim=-1).tolist(),
            "target_top2_margins": (top_values[:, 0] - top_values[:, 1]).tolist(),
            "first_rejection_index": n_accepted if n_accepted < k else None,
            "input_sequence_length": ctx_len + k,
            "dtype": str(logits.dtype),
            "timing_scope": "diagnostic_overhead_included_not_for_performance_comparison",
        }
    t_pack_done = time.perf_counter()
    gpu_end = _nvml_snapshot()

    server_recv_to_verify_start_ms = (t_verify_start - t_recv) * 1000.0
    verify_split_ms = (t_verify_done - t_verify_start) * 1000.0
    pack_split_ms = (t_pack_done - t_verify_done) * 1000.0
    verify_time_ms = (t_pack_done - t_recv) * 1000.0
    cloud_energy_j = None
    if gpu_start["energy_mj"] is not None and gpu_end["energy_mj"] is not None:
        cloud_energy_j = max(
            0.0, (gpu_end["energy_mj"] - gpu_start["energy_mj"]) / 1000.0
        )
    cloud_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024.0**2)
        if torch.cuda.is_available()
        else 0.0
    )
    return VerifyResponse(
        verification_mode=verification_mode,
        diagnostics=diagnostics,
        n_accepted=n_accepted,
        bonus_token_id=bonus_token_id,
        verify_time_ms=verify_time_ms,
        server_recv_to_verify_start_ms=server_recv_to_verify_start_ms,
        verify_split_ms=verify_split_ms,
        pack_split_ms=pack_split_ms,
        cloud_power_w_start=gpu_start["power_w"],
        cloud_power_w_end=gpu_end["power_w"],
        cloud_memory_mb_start=gpu_start["memory_mb"],
        cloud_memory_mb_end=gpu_end["memory_mb"],
        cloud_energy_j=cloud_energy_j,
        cloud_process_peak_allocated_mb=cloud_peak_mb,
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    allow_download: bool = False,
    device: str = "auto",
    dtype: Optional[str] = None,
    attention_implementation: Optional[str] = None,
    output_head_precision: str = "native",
    session_verifier: str = "parallel",
):
    global _device, _model, _tokenizer, _model_ref, _target_dtype
    global _attention_implementation, _output_head_precision, _session_verifier
    if session_verifier not in ("parallel", "sequential"):
        raise ValueError("session_verifier must be parallel or sequential")
    model_ref = str(model_name)
    local_only = not allow_download or Path(model_ref).expanduser().exists()
    if device == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    else:
        _device = device
    if dtype is None:
        dtype = "float16" if _device == "cuda" else "float32"
    dtype_value = getattr(torch, dtype)
    print(
        f"Loading target model: {model_ref}  device={_device} "
        f"dtype={dtype} local_only={local_only}"
    )
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_only)
        attention_kwargs = ({"attn_implementation": attention_implementation}
                            if attention_implementation is not None else {})
        _model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=dtype_value,
            device_map={"": _device},
            local_files_only=local_only,
            **attention_kwargs,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load cloud model '{model_name}'. Use local path or pass --allow-download after fixing network."
        ) from e
    _model.eval()
    if output_head_precision != "native":
        try:
            from hardware.target_precision import configure_output_head
        except ModuleNotFoundError:
            from target_precision import configure_output_head
        # TF32 must not undo the requested FP32 output projection.
        torch.backends.cuda.matmul.allow_tf32 = False
        configure_output_head(_model, output_head_precision)
    _session_verifier = session_verifier
    _attention_implementation = getattr(_model.config, "_attn_implementation", None)
    _output_head_precision = output_head_precision
    _model_ref = model_ref
    _target_dtype = dtype_value
    _target_sessions.clear()
    print("Target model loaded.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-download", action="store_true", default=False)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"))
    parser.add_argument("--output-head-precision", choices=("native", "float32"), default="native")
    parser.add_argument("--session-verifier", choices=("parallel", "sequential"), default="parallel")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default=None,
    )
    args = parser.parse_args()

    load_model(
        args.model,
        allow_download=args.allow_download,
        device=args.device,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        output_head_precision=args.output_head_precision,
        session_verifier=args.session_verifier,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
