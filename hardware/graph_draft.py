"""Persistent, single-session CUDA Graph execution for greedy draft decoding.

Capture is a service setup cost. Each session still pays for cache reset and
eager prefill. All decode and commit forwards use the same one-token graph;
there is no compilation, target RPC, controller, or hidden feedback here.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Sequence

import torch
from hardware.runtime_eos import tokenizer_eos_ids


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


class DraftGraphEngine:
    """Own one fixed-capacity cache and a captured batch-one decode forward.

Only one session may use this engine at a time. Call session.close() in the
request's finally block. Closing does not destroy the graph; the next request
resets the cache and performs a separately charged prefill.
    """

    def __init__(self, model: Any, max_cache_len: int = 512):
        from transformers import StaticCache

        if isinstance(max_cache_len, bool) or int(max_cache_len) != max_cache_len or max_cache_len < 2:
            raise ValueError("max_cache_len must be an integer of at least two")
        self.model = model
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("DraftGraphEngine requires an actual CUDA model")
        if model.training:
            raise ValueError("draft model must be in evaluation mode")
        if next(model.parameters()).dtype != torch.float16:
            raise ValueError("this graph backend has been validated only for float16")
        if getattr(model.config, "_attn_implementation", None) != "sdpa":
            raise ValueError("this graph backend requires SDPA attention")
        self.max_cache_len = int(max_cache_len)
        self._lock = threading.RLock()
        self._active_lease = None
        self._next_lease = 0
        self.position = 0
        self.replay_count = 0
        synchronize(self.device)
        setup_start = time.perf_counter()
        with torch.inference_mode():
            self.cache = StaticCache(model.config, max_batch_size=1,
                                     max_cache_len=self.max_cache_len,
                                     device=self.device, dtype=torch.float16)
            self.input_ids = torch.ones((1, 1), dtype=torch.long, device=self.device)
            self.cache_position = torch.ones(1, dtype=torch.long, device=self.device)
            synchronize(self.device)
            allocation_end = time.perf_counter()
            # Valid vocabulary token used solely for graph setup, never as a
            # request input or a measured calibration observation.
            self.model(input_ids=self.input_ids, past_key_values=self.cache,
                       cache_position=torch.zeros(1, dtype=torch.long, device=self.device),
                       use_cache=True, return_dict=True, logits_to_keep=1)
            synchronize(self.device)
            prefill_end = time.perf_counter()
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._eager_forward()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            synchronize(self.device)
            warmup_end = time.perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._graph_logits = self._eager_forward()
            synchronize(self.device)
            capture_end = time.perf_counter()
        self.setup_metrics = {
            "backend": "cuda_graph_static_cache", "batch_size": 1,
            "actual_enabled": True, "capture_success": True, "fallback_used": False,
            "max_cache_len": self.max_cache_len, "dtype": "float16",
            "attention": "sdpa", "capture_count": 1,
            "static_cache_allocation_ms": (allocation_end - setup_start) * 1000,
            "setup_prefill_ms": (prefill_end - allocation_end) * 1000,
            "warmup_ms": (warmup_end - prefill_end) * 1000,
            "capture_ms": (capture_end - warmup_end) * 1000,
            "total_setup_ms": (capture_end - setup_start) * 1000,
            "setup_scope": "one engine per loaded draft model; excludes model loading; charge separately to service/calibration acquisition",
            "request_prefill_scope": "each request pays cache reset plus eager prefill",
        }

    def _eager_forward(self):
        return self.model(input_ids=self.input_ids, past_key_values=self.cache,
                          cache_position=self.cache_position, use_cache=True,
                          return_dict=True, logits_to_keep=1).logits[0, -1]

    def ensure_active(self, lease):
        if lease is None or lease != self._active_lease:
            raise RuntimeError("graph draft session is closed or does not own this cache")

    def acquire(self, context_ids: Sequence[int], max_new_tokens: Optional[int] = None):
        context = [int(token) for token in context_ids]
        if not context:
            raise ValueError("context_ids must not be empty")
        if max_new_tokens is not None and (
            isinstance(max_new_tokens, bool) or int(max_new_tokens) != max_new_tokens or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a nonnegative integer")
        required = len(context) + (0 if max_new_tokens is None else int(max_new_tokens))
        if required > self.max_cache_len:
            raise ValueError(f"context plus output capacity {required} exceeds graph cache {self.max_cache_len}")
        if any(token < 0 or token >= self.model.config.vocab_size for token in context):
            raise ValueError("context token lies outside the model vocabulary")
        with self._lock:
            if self._active_lease is not None:
                raise RuntimeError("graph engine already has an active session; close it before reuse")
            self._next_lease += 1
            lease = self._next_lease
            self._active_lease = lease
            try:
                with torch.inference_mode():
                    self.cache.reset()
                    ids = torch.tensor([context], dtype=torch.long, device=self.device)
                    output = self.model(
                        input_ids=ids, past_key_values=self.cache,
                        cache_position=torch.arange(len(context), device=self.device),
                        use_cache=True, return_dict=True, logits_to_keep=1,
                    )
                    self.position = len(context)
                    logits = output.logits[0, -1].clone()
                synchronize(self.device)
                return lease, logits
            except BaseException:
                self._active_lease = None
                raise

    def step(self, lease, token_id: int):
        with self._lock:
            self.ensure_active(lease)
            if self.position >= self.max_cache_len:
                raise ValueError("graph cache capacity exceeded; no silent truncation is allowed")
            if not 0 <= int(token_id) < self.model.config.vocab_size:
                raise ValueError("forward token lies outside the model vocabulary")
            with torch.inference_mode():
                self.input_ids.fill_(int(token_id))
                self.cache_position.fill_(self.position)
                self.graph.replay()
                self.replay_count += 1
                self.position += 1
                # Graph output storage is reused. Preserve each token's logits
                # for rollback/no-bonus commits and callback consumers.
                return self._graph_logits.clone()

    def rewind(self, lease, position: int):
        with self._lock:
            self.ensure_active(lease)
            if not 0 <= position <= self.position:
                raise ValueError("rewind must retain an existing prefix")
            # Future static slots remain populated but are masked by the
            # explicit cache_position. Each new token overwrites its own slot.
            self.position = int(position)

    def release(self, lease):
        with self._lock:
            self.ensure_active(lease)
            synchronize(self.device)
            self._active_lease = None


class GraphDraftSession:
    """GreedyDraftSession-compatible session backed by a persistent engine."""

    def __init__(self, engine: DraftGraphEngine, tokenizer: Any,
                 context_ids: Sequence[int], lazy_final_forward: bool = True,
                 max_new_tokens: Optional[int] = None):
        if not isinstance(lazy_final_forward, bool):
            raise ValueError("lazy_final_forward must be a boolean")
        self.engine, self.model = engine, engine.model
        self.device, self.tokenizer = engine.device, tokenizer
        self.lazy_final_forward = lazy_final_forward
        self._lease, self.next_logits = engine.acquire(context_ids, max_new_tokens)
        self._initial_replay_count = engine.replay_count
        self.kv_tokens = len(context_ids)
        self._pending: Optional[Dict[str, Any]] = None
        self._closed_state = None

    def close(self):
        if self._lease is not None:
            final_engine_position = self.engine.position
            final_replays = self.engine.replay_count - self._initial_replay_count
            self.engine.release(self._lease)
            self._lease = None
            self._closed_state = {"engine_position": final_engine_position,
                                  "graph_replays": final_replays}

    def execution_state(self):
        state = {"backend": "cuda_graph_static_cache",
                 "max_cache_len": self.engine.max_cache_len,
                 "final_logical_kv_tokens": self.kv_tokens,
                 "session_closed": self._lease is None}
        if self._closed_state is not None:
            state.update(self._closed_state)
        else:
            self.engine.ensure_active(self._lease)
            state.update({"engine_position": self.engine.position,
                          "graph_replays": self.engine.replay_count - self._initial_replay_count})
        return state

    def draft(self, k_value: int, token_decider=None):
        self.engine.ensure_active(self._lease)
        if isinstance(k_value, bool) or int(k_value) != k_value or k_value < 0:
            raise ValueError("k_value must be a nonnegative integer")
        if self._pending is not None:
            raise RuntimeError("previous draft has not been committed")
        if self.kv_tokens + int(k_value) > self.engine.max_cache_len:
            raise ValueError("requested draft exceeds graph cache capacity")
        base_kv_tokens, base_next_logits = self.kv_tokens, self.next_logits
        draft_ids, logits_after_token = [], []
        stopped_by_policy, policy_time_ms = False, 0.0
        started = time.perf_counter()
        with torch.inference_mode():
            for index in range(int(k_value)):
                if token_decider is not None:
                    synchronize(self.device)
                    decision_start = time.perf_counter()
                    proceed = token_decider(index + 1, self.next_logits,
                                            base_kv_tokens + len(logits_after_token))
                    policy_time_ms += (time.perf_counter() - decision_start) * 1000
                    if not proceed:
                        stopped_by_policy = True
                        break
                token_id = int(self.next_logits.argmax())
                draft_ids.append(token_id)
                eos = token_id in tokenizer_eos_ids(self.tokenizer)
                if self.lazy_final_forward and (index + 1 == k_value or eos):
                    break
                self.next_logits = self.engine.step(self._lease, token_id)
                logits_after_token.append(self.next_logits)
                if eos:
                    break
        self.kv_tokens = self.engine.position
        if self.kv_tokens != base_kv_tokens + len(logits_after_token):
            raise RuntimeError("graph draft logical cache length mismatch")
        self._pending = {"base_kv_tokens": base_kv_tokens,
                         "base_next_logits": base_next_logits, "draft_ids": draft_ids,
                         "logits_after_token": logits_after_token}
        synchronize(self.device)
        return {"draft_ids": draft_ids,
                "draft_time_ms": (time.perf_counter() - started) * 1000,
                "draft_kv_tokens_before": base_kv_tokens,
                "draft_kv_tokens_after_draft": self.kv_tokens,
                "draft_forward_calls": len(logits_after_token),
                "draft_tokens_pending_forward": len(draft_ids) - len(logits_after_token),
                "lazy_final_forward": self.lazy_final_forward,
                "draft_stopped_by_policy": stopped_by_policy,
                "draft_policy_time_ms": policy_time_ms,
                "draft_graph_replays": len(logits_after_token),
                "draft_execution_backend": "cuda_graph_static_cache"}

    def commit(self, n_accepted: int, committed_token_ids: Sequence[int]):
        self.engine.ensure_active(self._lease)
        if self._pending is None:
            raise RuntimeError("no pending draft to commit")
        pending = self._pending
        draft_ids = pending["draft_ids"]
        if isinstance(n_accepted, bool) or int(n_accepted) != n_accepted or not 0 <= n_accepted <= len(draft_ids):
            raise ValueError("n_accepted is outside the pending draft")
        n_accepted = int(n_accepted)
        committed = [int(value) for value in committed_token_ids]
        if committed[:n_accepted] != draft_ids[:n_accepted]:
            raise RuntimeError("cloud accepted tokens are not the draft prefix")
        if len(committed) not in (n_accepted, n_accepted + 1):
            raise RuntimeError("cloud must return accepted prefix and at most one bonus")
        base = pending["base_kv_tokens"]
        if base + len(committed) > self.engine.max_cache_len:
            raise ValueError("committed suffix exceeds graph cache capacity")
        started = time.perf_counter()
        materialized = min(n_accepted, len(pending["logits_after_token"]))
        self.engine.rewind(self._lease, base + materialized)
        missing = committed[materialized:]
        if missing:
            for token_id in missing:
                self.next_logits = self.engine.step(self._lease, token_id)
        elif materialized:
            self.next_logits = pending["logits_after_token"][materialized - 1]
        else:
            self.next_logits = pending["base_next_logits"]
        self.kv_tokens = self.engine.position
        if self.kv_tokens != base + len(committed):
            raise RuntimeError("graph committed logical cache length mismatch")
        self._pending = None
        synchronize(self.device)
        return {"draft_kv_tokens_after_commit": self.kv_tokens,
                "draft_commit_time_ms": (time.perf_counter() - started) * 1000,
                "draft_commit_forward_tokens": len(missing),
                "draft_commit_forward_calls": len(missing),
                "draft_commit_graph_replays": len(missing),
                "lazy_final_forward": self.lazy_final_forward,
                "draft_execution_backend": "cuda_graph_static_cache"}
