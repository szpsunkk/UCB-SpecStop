"""
cloud_server.py — Run on the RTX 3090 server (cloud side).

Loads the target model, exposes HTTP endpoints:
  POST /verify   — verify k draft tokens, return n_accepted + bonus
  GET  /ping     — latency probe for RTT measurement

Protocol:
  - context_ids + draft_ids required
  - draft_log_probs optional: when provided, uses full rejection sampling
    (accept token i with prob min(1, p_target(i)/p_draft(i)));
    when absent, falls back to greedy argmax comparison (faster approximation)

Usage:
  python cloud_server.py --model Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    context_ids: List[int]
    draft_ids: List[int]
    # Log-probs of each draft token under the draft model.
    # When provided, enables proper rejection sampling (ratio test).
    # When absent, greedy argmax comparison is used as an approximation.
    draft_log_probs: Optional[List[float]] = None
    # Optional per-request seed: when provided, rejection sampling and bonus
    # sampling become deterministic for this (prompt, draft_ids) pair, enabling
    # paired across-strategy comparisons. Pass the same seed across strategies
    # for the same prompt to remove sampling noise from the comparison.
    seed: Optional[int] = None


class VerifyResponse(BaseModel):
    n_accepted: int
    bonus_token_id: int
    verify_time_ms: float
    # Timing breakdown (review B1) — backwards compatible: clients that ignore
    # these fields still work; clients that read them get a clock-independent
    # decomposition of server-side wall time.
    server_recv_to_verify_start_ms: float = 0.0
    verify_split_ms: float = 0.0
    pack_split_ms: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/ping")
def ping():
    return {"status": "ok", "ts": time.time()}


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    t_recv = time.perf_counter()

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
        # log_softmax over all positions at once (reused for bonus)
        log_probs = torch.log_softmax(logits[0], dim=-1)  # (ctx_len+k, vocab)

    n_accepted = 0

    if req.draft_log_probs is not None:
        # Full rejection sampling: accept token i with prob min(1, p_target/p_draft)
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
        # Greedy approximation: accept if draft token == target argmax
        for i, draft_tok in enumerate(req.draft_ids):
            target_tok = int(logits[0, ctx_len + i - 1].argmax())
            if draft_tok == target_tok:
                n_accepted += 1
            else:
                break

    t_verify_done = time.perf_counter()

    # Bonus token: sample from target distribution at rejection position.
    # Position ctx_len+n_accepted-1 in logits predicts token at ctx_len+n_accepted.
    bonus_logits = logits[0, ctx_len + n_accepted - 1]
    bonus_probs = torch.softmax(bonus_logits, dim=-1)
    if bonus_gen is not None:
        bonus_token_id = int(torch.multinomial(bonus_probs, 1, generator=bonus_gen).item())
    else:
        bonus_token_id = int(torch.multinomial(bonus_probs, 1).item())

    t_pack_done = time.perf_counter()

    server_recv_to_verify_start_ms = (t_verify_start - t_recv) * 1000.0
    verify_split_ms = (t_verify_done - t_verify_start) * 1000.0
    pack_split_ms = (t_pack_done - t_verify_done) * 1000.0
    verify_time_ms = (t_pack_done - t_recv) * 1000.0
    return VerifyResponse(
        n_accepted=n_accepted,
        bonus_token_id=bonus_token_id,
        verify_time_ms=verify_time_ms,
        server_recv_to_verify_start_ms=server_recv_to_verify_start_ms,
        verify_split_ms=verify_split_ms,
        pack_split_ms=pack_split_ms,
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_model(model_name: str, allow_download: bool = False):
    global _model, _tokenizer
    model_ref = str(model_name)
    local_only = not allow_download or Path(model_ref).expanduser().exists()
    print(f"Loading target model: {model_ref}  device={_device} local_only={local_only}")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_only)
        _model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=local_only,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load cloud model '{model_name}'. Use local path or pass --allow-download after fixing network."
        ) from e
    _model.eval()
    print("Target model loaded.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-download", action="store_true", default=False)
    args = parser.parse_args()

    load_model(args.model, allow_download=args.allow_download)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
