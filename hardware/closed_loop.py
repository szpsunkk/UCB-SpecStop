"""Stateful request primitives for the revision-2 closed-loop experiment."""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import requests
import torch

from hardware.kv_cache import cache_sequence_length, crop_past_key_values
from hardware.runtime_eos import tokenizer_eos_ids
from src.baselines import (
    BanditSpecThroughputUCB,
    DisplayedRatioLCB,
    PerRoundRatioUCB,
    UCBSpecStop,
)
from src.regularized_ratio import RegularizedRatioBandit
from src.structured_bandits import StructuredRatioBandit


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = np.asarray(token_ids, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _synchronize_device(device: torch.device) -> None:
    """Charge queued CUDA work to the operation that issued it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


class GreedyDraftSession:
    """Maintain a draft cache, optionally deferring the last draft forward.

    In lazy mode, selecting the final draft token does not compute logits for
    another token. Its KV entry is materialized during commit if accepted,
    together with any target bonus. The default retains eager drafting.
    """

    def __init__(
        self, model: Any, tokenizer: Any, context_ids: Sequence[int],
        lazy_final_forward: bool = False,
    ):
        if not context_ids:
            raise ValueError("context_ids must not be empty")
        if not isinstance(lazy_final_forward, bool):
            raise ValueError("lazy_final_forward must be a boolean")
        self.model = model
        self.tokenizer = tokenizer
        self.lazy_final_forward = lazy_final_forward
        self.device = _model_device(model)
        context = torch.tensor([list(context_ids)], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            output = self.model(context, use_cache=True)
        self.past_key_values = output.past_key_values
        self.next_logits = output.logits[0, -1]
        self.kv_tokens = cache_sequence_length(self.past_key_values)
        if self.kv_tokens != len(context_ids):
            raise RuntimeError(
                f"draft KV length {self.kv_tokens} != context length {len(context_ids)}"
            )
        _synchronize_device(self.device)
        self._pending: Optional[Dict[str, Any]] = None

    def draft(self, k_value: int, token_decider=None) -> Dict[str, object]:
        if k_value < 0:
            raise ValueError("k_value must be nonnegative")
        if self._pending is not None:
            raise RuntimeError("previous draft has not been committed")

        base_kv_tokens = self.kv_tokens
        base_next_logits = self.next_logits
        draft_ids: List[int] = []
        logits_after_token: List[torch.Tensor] = []
        stopped_by_policy = False
        policy_time_ms = 0.0
        draft_start = time.perf_counter()
        with torch.inference_mode():
            for index in range(k_value):
                if token_decider is not None:
                    # Attribute pending draft work to drafting, not to the
                    # feature callback whose first operation may synchronize.
                    _synchronize_device(self.device)
                    decision_start = time.perf_counter()
                    proceed = token_decider(index + 1, self.next_logits,
                                            base_kv_tokens + len(logits_after_token))
                    policy_time_ms += (time.perf_counter() - decision_start) * 1000.0
                    if not proceed:
                        stopped_by_policy = True
                        break
                token_id = int(self.next_logits.argmax())
                draft_ids.append(token_id)
                eos = token_id in tokenizer_eos_ids(self.tokenizer)
                if self.lazy_final_forward and (index + 1 == k_value or eos):
                    break
                token = torch.tensor(
                    [[token_id]], device=self.device, dtype=torch.long
                )
                output = self.model(
                    token,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                )
                self.past_key_values = output.past_key_values
                self.next_logits = output.logits[0, -1]
                logits_after_token.append(self.next_logits)
                if eos:
                    break

        self.kv_tokens = cache_sequence_length(self.past_key_values)
        expected = base_kv_tokens + len(logits_after_token)
        if self.kv_tokens != expected:
            raise RuntimeError(f"draft KV length {self.kv_tokens} != expected {expected}")
        self._pending = {
            "base_kv_tokens": base_kv_tokens,
            "base_next_logits": base_next_logits,
            "draft_ids": draft_ids,
            "logits_after_token": logits_after_token,
        }
        _synchronize_device(self.device)
        return {
            "draft_ids": draft_ids,
            "draft_time_ms": (time.perf_counter() - draft_start) * 1000.0,
            "draft_kv_tokens_before": base_kv_tokens,
            "draft_kv_tokens_after_draft": self.kv_tokens,
            "draft_forward_calls": len(logits_after_token),
            "draft_tokens_pending_forward": len(draft_ids) - len(logits_after_token),
            "lazy_final_forward": self.lazy_final_forward,
            "draft_stopped_by_policy": stopped_by_policy,
            "draft_policy_time_ms": policy_time_ms,
        }

    def commit(
        self, n_accepted: int, committed_token_ids: Sequence[int]
    ) -> Dict[str, object]:
        if self._pending is None:
            raise RuntimeError("no pending draft to commit")
        pending = self._pending
        draft_ids = pending["draft_ids"]
        if not 0 <= n_accepted <= len(draft_ids):
            raise ValueError("n_accepted is outside the pending draft")
        committed = [int(value) for value in committed_token_ids]
        if committed[:n_accepted] != draft_ids[:n_accepted]:
            raise RuntimeError("cloud accepted tokens are not the draft prefix")
        if len(committed) not in (n_accepted, n_accepted + 1):
            raise RuntimeError("cloud must return accepted prefix and at most one bonus")

        commit_start = time.perf_counter()
        base_kv_tokens = int(pending["base_kv_tokens"])
        # A lazy draft may have selected one more token than it materialized.
        # Never extend a cache by cropping; replay the committed suffix instead.
        materialized_accepted = min(n_accepted, len(pending["logits_after_token"]))
        self.past_key_values = crop_past_key_values(
            self.past_key_values, base_kv_tokens + materialized_accepted
        )

        missing_committed = committed[materialized_accepted:]
        if missing_committed:
            token = torch.tensor(
                [missing_committed], device=self.device, dtype=torch.long
            )
            with torch.inference_mode():
                output = self.model(
                    token,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                )
            self.past_key_values = output.past_key_values
            self.next_logits = output.logits[0, -1]
        elif materialized_accepted > 0:
            self.next_logits = pending["logits_after_token"][materialized_accepted - 1]
        else:
            self.next_logits = pending["base_next_logits"]

        self.kv_tokens = cache_sequence_length(self.past_key_values)
        expected = base_kv_tokens + len(committed)
        if self.kv_tokens != expected:
            raise RuntimeError(
                f"committed draft KV length {self.kv_tokens} != expected {expected}"
            )
        self._pending = None
        _synchronize_device(self.device)
        return {
            "draft_kv_tokens_after_commit": self.kv_tokens,
            "draft_commit_time_ms": (time.perf_counter() - commit_start) * 1000.0,
            "draft_commit_forward_tokens": len(missing_committed),
            "draft_commit_forward_calls": int(bool(missing_committed)),
            "lazy_final_forward": self.lazy_final_forward,
        }


@dataclass
class ServiceController:
    method: str
    actual_arms: Sequence[int]
    beta: float = 1.0
    horizon: int = 10000
    lcb_l_max_ms_per_token: float = 1000.0
    ucbspec_exploration_weight: float = 6.0
    fixed_arm: Optional[int] = None
    prior_strength: float = 0.0
    allowed_arms: Optional[Sequence[int]] = None
    prefix_compatible: bool = False
    confidence: str = "residual"
    confidence_delta: float = 0.05
    complete_round_cost_bound_ms: Optional[float] = None
    cost_tolerance: Optional[float] = None
    exploration_probability: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.actual_arms = tuple(int(value) for value in self.actual_arms)
        if not self.actual_arms or len(set(self.actual_arms)) != len(self.actual_arms):
            raise ValueError("actual_arms must be non-empty and unique")
        if any(value < 0 for value in self.actual_arms):
            raise ValueError("arms must be nonnegative")
        requested = self.actual_arms if self.allowed_arms is None else tuple(self.allowed_arms)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("allowed_arms must be nonempty and unique")
        if any(value not in self.actual_arms for value in requested):
            raise ValueError("allowed_arms must be a subset of actual_arms")
        self.allowed_arms = tuple(arm for arm in self.actual_arms if arm in requested)
        self.rounds_selected = 0
        self.calibration_observations = 0
        self.calibration_cells = 0
        self._physical_indices = self.method in {
            "ratio_regularized", "ratio_residual", "prefix_shared"
        }
        if self._physical_indices and any(arm <= 0 for arm in self.actual_arms):
            raise ValueError("new ratio controllers require positive physical draft lengths")
        arm_count = len(self.actual_arms)
        if self.method == "sn":
            self.algorithm = UCBSpecStop(arm_count, self.beta)
        elif self.method == "lcb":
            self.algorithm = DisplayedRatioLCB(
                arm_count,
                horizon=self.horizon,
                l_max=self.lcb_l_max_ms_per_token,
                beta=self.beta,
            )
        elif self.method == "per_round_ucb":
            self.algorithm = PerRoundRatioUCB(arm_count, self.beta)
        elif self.method == "ucbspec_throughput":
            self.algorithm = BanditSpecThroughputUCB(
                arm_count,
                exploration_weight=self.ucbspec_exploration_weight,
                common_bonus=True,
            )
        elif self.method == "fixed":
            if self.fixed_arm not in self.allowed_arms:
                raise ValueError("fixed_arm must belong to allowed_arms")
            self.algorithm = None
        elif self.method == "ratio_regularized":
            self.algorithm = RegularizedRatioBandit(
                max(self.actual_arms), prior_strength=self.prior_strength, beta=self.beta
            )
        elif self.method in {"ratio_residual", "prefix_shared"}:
            self.algorithm = StructuredRatioBandit(
                max(self.actual_arms), confidence=self.confidence,
                prefix_share=self.method == "prefix_shared",
                prefix_compatible=self.prefix_compatible,
                shape_constraints=self.method == "prefix_shared",
                beta=self.beta, delta=self.confidence_delta,
                n_max=self.complete_round_cost_bound_ms,
                allowed_arms=self.allowed_arms, cost_tolerance=self.cost_tolerance,
                exploration_probability=self.exploration_probability, seed=self.seed,
            )
        else:
            raise ValueError(f"unsupported controller method: {self.method}")

    def _masked_baseline_arm(self, t: int) -> int:
        """Apply the same external calibration mask without fabricating samples."""
        positions = np.array([self.actual_arms.index(arm) for arm in self.allowed_arms])
        if self.method == "ucbspec_throughput" and t < 2 * len(positions):
            return int(self.allowed_arms[(t - 1) % len(positions)])
        count = self.algorithm.t_k[positions]
        unknown = np.flatnonzero(count == 0)
        if len(unknown):
            return int(self.allowed_arms[int(unknown[0])])
        if self.method == "sn":
            indices = (
                self.algorithm.s_n[positions] / self.algorithm.s_a[positions]
                - self.beta * np.sqrt(count * np.log(t)) / self.algorithm.s_a[positions]
            )
        elif self.method == "per_round_ucb":
            indices = self.algorithm.sum_ratio[positions] / count
            indices -= self.beta * np.sqrt(np.log(t) / count)
        elif self.method == "lcb":
            indices = self.algorithm.s_n[positions] / self.algorithm.s_a[positions]
            indices -= self.beta * self.algorithm.l_max * np.sqrt(
                self.algorithm.log_term / count
            )
        else:
            # The source-faithful common bonus is identical for every arm.
            indices = -self.algorithm.sum_reward[positions] / count
        return int(self.allowed_arms[int(np.argmin(indices))])

    def select_arm(self) -> int:
        self.rounds_selected += 1
        if self.method == "fixed":
            return int(self.fixed_arm)
        t = self.calibration_observations + self.rounds_selected
        if self.method == "ratio_regularized":
            indices = self.algorithm.snapshot(t)["indices"]
            return min(self.allowed_arms, key=lambda arm: (indices[arm - 1], arm))
        if self._physical_indices:
            return int(self.algorithm.select_arm(t))
        if self.allowed_arms != self.actual_arms:
            return self._masked_baseline_arm(t)
        internal_arm = int(self.algorithm.select_arm(t))
        return self.actual_arms[internal_arm - 1]

    def update(
        self,
        selected_arm: int,
        elapsed_ms: float,
        delivered_tokens: int,
        censored: bool = False,
        censor_reason: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        if self.method == "fixed":
            return False, "fixed_controller"
        if censored:
            return False, censor_reason or "arm_truncated_by_request_boundary"
        if selected_arm not in self.actual_arms:
            raise ValueError("selected arm is not in this controller")
        internal_arm = (
            selected_arm if self._physical_indices
            else self.actual_arms.index(selected_arm) + 1
        )
        if delivered_tokens <= 0:
            return False, "zero_delivered_tokens"
        if self.method == "ucbspec_throughput":
            self.algorithm.update(
                internal_arm,
                committed_tokens=float(delivered_tokens),
                elapsed_time=float(elapsed_ms),
                time_scale_to_seconds=1e-3,
            )
            return True, None
        self.algorithm.update(
            internal_arm, n_t=float(elapsed_ms), a_t=float(delivered_tokens)
        )
        return True, None

    def warm_start(
        self, costs: np.ndarray, yields: np.ndarray,
        arms: Optional[Sequence[int]] = None,
    ) -> None:
        """Supply the same explicit calibration pairs to every controller.

        Columns default to actual_arms order; all actual arms must be present,
        including arms excluded by a shared calibration mask. Only complete,
        positive-yield, uncensored observations belong in this input. Costs
        must use the same pre-update feedback timing as continuous rounds.
        Prefix cells are counted once, not once per measured arm.
        """
        columns = self.actual_arms if arms is None else tuple(int(arm) for arm in arms)
        costs, yields = np.asarray(costs, dtype=float), np.asarray(yields, dtype=float)
        if len(columns) != len(set(columns)) or set(columns) != set(self.actual_arms):
            raise ValueError("calibration columns must cover all actual_arms exactly once")
        if costs.ndim != 2 or costs.shape[1] != len(columns) or yields.shape != costs.shape:
            raise ValueError("calibration arrays must have equal (cells, actual arms) shape")
        if not np.isfinite(costs).all() or np.any(costs < 0):
            raise ValueError("calibration costs must be finite and nonnegative")
        if not np.isfinite(yields).all() or np.any(yields < 1):
            raise ValueError("calibration yields must be finite and positive")
        if np.any(yields > np.asarray(columns)[None, :] + 1):
            raise ValueError("calibration yields exceed physical arm bounds")
        if (
            isinstance(self.algorithm, StructuredRatioBandit)
            and self.complete_round_cost_bound_ms is not None
            and np.any(costs > self.complete_round_cost_bound_ms)
        ):
            raise ValueError("calibration costs exceed the explicit cost bound")
        if self.method == "prefix_shared":
            if set(columns) != set(range(1, max(columns) + 1)):
                raise ValueError("prefix warm start requires the complete physical arm grid")
            order = np.argsort(columns)
            self.algorithm.warm_start(costs[:, order], yields[:, order])
        else:
            for row_n, row_a in zip(costs, yields):
                for index, arm in enumerate(columns):
                    self.update(arm, float(row_n[index]), float(row_a[index]))
        self.calibration_cells += len(costs)
        self.calibration_observations += costs.size

    def state_dict(self) -> Dict[str, object]:
        state: Dict[str, object] = {
            "method": self.method,
            "actual_arms": list(self.actual_arms),
            "rounds_selected": self.rounds_selected,
            "fixed_arm": self.fixed_arm,
            "allowed_arms": list(self.allowed_arms),
            "calibration_cells": self.calibration_cells,
            "calibration_observations": self.calibration_observations,
            "prefix_compatible": self.prefix_compatible,
            "feedback_cost_scope": "selection_through_commit_before_controller_update",
        }
        if self.algorithm is not None:
            for name in ("s_n", "s_a", "sum_ratio", "sum_reward", "t_k"):
                value = getattr(self.algorithm, name, None)
                if value is not None:
                    state[name] = np.asarray(value, dtype=float).tolist()
            if hasattr(self.algorithm, "snapshot"):
                def serializable(value):
                    if isinstance(value, dict):
                        return {key: serializable(item) for key, item in value.items()}
                    if isinstance(value, (tuple, list)):
                        return [serializable(item) for item in value]
                    if isinstance(value, float) and not math.isfinite(value):
                        return None
                    return value
                state["algorithm_state"] = serializable(self.algorithm.snapshot(
                    max(1, self.calibration_observations + self.rounds_selected)
                ))
        return state


def _post_json(
    server: str,
    path: str,
    payload: Dict[str, object],
    one_way_delay_ms: float,
    timeout_s: float = 120.0,
) -> Dict[str, object]:
    serialize_start = time.perf_counter()
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
    if one_way_delay_ms > 0:
        time.sleep(one_way_delay_ms / 1000.0)
    request_start = time.perf_counter()
    response = requests.post(
        f"{server.rstrip('/')}/{path.lstrip('/')}",
        data=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout_s,
    )
    network_and_server_ms = (time.perf_counter() - request_start) * 1000.0
    if one_way_delay_ms > 0:
        time.sleep(one_way_delay_ms / 1000.0)
    response.raise_for_status()
    deserialize_start = time.perf_counter()
    data = response.json()
    deserialize_ms = (time.perf_counter() - deserialize_start) * 1000.0
    data.update(
        {
            "_serialize_ms": serialize_ms,
            "_deserialize_ms": deserialize_ms,
            "_network_and_server_ms": network_and_server_ms,
            "_request_payload_bytes": len(body),
            "_response_payload_bytes": len(response.content),
        }
    )
    return data


def start_target_session(
    server: str, context_ids: Sequence[int], one_way_delay_ms: float = 0.0
) -> Dict[str, object]:
    return _post_json(
        server,
        "/session/start",
        {"context_ids": [int(value) for value in context_ids]},
        one_way_delay_ms,
    )


def verify_target_session(
    server: str,
    session_id: str,
    draft_ids: Sequence[int],
    one_way_delay_ms: float = 0.0,
) -> Dict[str, object]:
    return _post_json(
        server,
        "/session/verify",
        {
            "session_id": session_id,
            "draft_ids": [int(value) for value in draft_ids],
        },
        one_way_delay_ms,
    )


def close_target_session(server: str, session_id: str) -> Dict[str, object]:
    return _post_json(
        server,
        "/session/close",
        {"session_id": session_id},
        one_way_delay_ms=0.0,
        timeout_s=30.0,
    )


def run_speculative_request(
    server: str,
    model: Any,
    tokenizer: Any,
    context_ids: Sequence[int],
    controller: ServiceController,
    generation_token_cap: int,
    one_way_delay_ms: float = 0.0,
    lazy_final_forward: bool = False,
    draft_session_factory=None,
    answer_stop: bool = False,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """Run one true continuous request with selected-arm-only feedback."""

    if generation_token_cap <= 0:
        raise ValueError("generation_token_cap must be positive")
    wall_start_ns = time.time_ns()
    monotonic_start_ns = time.monotonic_ns()
    request_start = time.perf_counter()
    prefill_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        target_future = executor.submit(
            start_target_session, server, context_ids, one_way_delay_ms
        )
        edge_prefill_start = time.perf_counter()
        draft_options = {"lazy_final_forward": True} if lazy_final_forward else {}
        draft_session = None
        try:
            factory = draft_session_factory or GreedyDraftSession
            draft_session = factory(model, tokenizer, context_ids, **draft_options)
            edge_prefill_ms = (time.perf_counter() - edge_prefill_start) * 1000.0
            target_start = target_future.result()
        except Exception:
            try:
                if callable(getattr(draft_session, "close", None)):
                    draft_session.close()
            finally:
                try:
                    abandoned = target_future.result()
                    close_target_session(server, str(abandoned["session_id"]))
                except Exception:
                    pass
            raise
    parallel_prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
    session_id = str(target_start["session_id"])
    if int(target_start["target_kv_tokens"]) != draft_session.kv_tokens:
        try:
            close_target_session(server, session_id)
        finally:
            if callable(getattr(draft_session, "close", None)):
                draft_session.close()
        raise RuntimeError("draft and target prefill KV lengths differ")

    delivered: List[int] = []
    round_rows: List[Dict[str, object]] = []
    ttft_ms: Optional[float] = None
    last_token_latency_ms: Optional[float] = None
    completion_ms: Optional[float] = None
    stop_reason = "cap"
    total_controller_ms = 0.0
    total_request_bytes = int(target_start.get("_request_payload_bytes", 0))
    total_response_bytes = int(target_start.get("_response_payload_bytes", 0))
    cloud_energy_values = [target_start.get("cloud_energy_j")]
    cloud_peak_memory_mb = max(
        float(target_start.get("cloud_memory_mb_start") or 0.0),
        float(target_start.get("cloud_memory_mb_end") or 0.0),
        float(target_start.get("cloud_process_peak_allocated_mb") or 0.0),
    )
    round_index = 0

    try:
        while len(delivered) < generation_token_cap:
            remaining = generation_token_cap - len(delivered)
            round_start = time.perf_counter()
            decision_start = time.perf_counter()
            context_selector = getattr(controller, "select_arm_with_context", None)
            selected_arm = (
                context_selector(draft_session, remaining)
                if callable(context_selector) else controller.select_arm()
            )
            controller_select_ms = (time.perf_counter() - decision_start) * 1000.0
            effective_k = min(selected_arm, max(remaining - 1, 0))
            cap_censored = effective_k != selected_arm

            token_decider = getattr(controller, "decide_draft_token", None)
            draft_options = {"token_decider": token_decider} if callable(token_decider) else {}
            draft_result = draft_session.draft(effective_k, **draft_options)
            token_available_start = time.perf_counter()
            verification = verify_target_session(
                server,
                session_id,
                draft_result["draft_ids"],
                one_way_delay_ms,
            )
            token_available_ms = (
                time.perf_counter() - token_available_start
            ) * 1000.0
            committed_ids = [
                int(value) for value in verification["committed_token_ids"]
            ]
            delivered_this_round = []
            eos_seen = False
            answer_seen = False
            for token_id in committed_ids:
                if token_id in tokenizer_eos_ids(tokenizer):
                    eos_seen = True
                    break
                if len(delivered) + len(delivered_this_round) >= generation_token_cap:
                    break
                delivered_this_round.append(token_id)
                if answer_stop:
                    from hardware.answer_stopping import token_answer_complete
                    if token_answer_complete(tokenizer, delivered + delivered_this_round):
                        answer_seen = True
                        break
            if delivered_this_round:
                delivered.extend(delivered_this_round)
                delivery_elapsed_ms = (time.perf_counter() - request_start) * 1000.0
                if ttft_ms is None:
                    ttft_ms = delivery_elapsed_ms
                last_token_latency_ms = delivery_elapsed_ms

            commit_result = draft_session.commit(
                int(verification["n_accepted"]), committed_ids
            )
            feedback_elapsed_ms = (time.perf_counter() - round_start) * 1000.0
            eos_terminal = eos_seen or verification.get("stop_reason") == "eos"
            draft_shortened = len(draft_result["draft_ids"]) != effective_k
            policy_stopped = bool(draft_result.get("draft_stopped_by_policy", False))
            arm_censored = cap_censored or eos_terminal or draft_shortened or answer_seen
            censor_reason = (
                "task_answer" if answer_seen else
                "terminal_eos" if eos_terminal else
                "generation_cap" if cap_censored else
                "policy_selected_shorter_draft" if policy_stopped else
                "draft_terminated_early" if draft_shortened else None
            )
            update_start = time.perf_counter()
            updated, skipped_reason = controller.update(
                selected_arm,
                elapsed_ms=feedback_elapsed_ms,
                delivered_tokens=len(delivered_this_round),
                censored=arm_censored,
                censor_reason=censor_reason,
            )
            controller_update_ms = (time.perf_counter() - update_start) * 1000.0
            policy_time_ms = float(draft_result.get("draft_policy_time_ms", 0.0))
            controller_ms = controller_select_ms + controller_update_ms + policy_time_ms
            total_controller_ms += controller_ms
            round_cycle_ms = (time.perf_counter() - round_start) * 1000.0
            total_request_bytes += int(verification.get("_request_payload_bytes", 0))
            total_response_bytes += int(verification.get("_response_payload_bytes", 0))
            cloud_energy_values.append(verification.get("cloud_energy_j"))
            cloud_peak_memory_mb = max(
                cloud_peak_memory_mb,
                float(verification.get("cloud_memory_mb_start") or 0.0),
                float(verification.get("cloud_memory_mb_end") or 0.0),
                float(verification.get("cloud_process_peak_allocated_mb") or 0.0),
            )
            round_rows.append(
                {
                    "round": round_index,
                    "selected_arm": selected_arm,
                    "effective_k": effective_k,
                    "actual_draft_length": len(draft_result["draft_ids"]),
                    "arm_censored": arm_censored,
                    "censor_reason": censor_reason,
                    "feedback_updated": updated,
                    "feedback_skipped_reason": skipped_reason,
                    "accepted_draft_tokens": int(verification["n_accepted"]),
                    "committed_model_tokens": len(committed_ids),
                    "delivered_tokens": len(delivered_this_round),
                    "controller_time_ms": controller_ms,
                    "controller_select_time_ms": controller_select_ms,
                    "controller_update_time_ms": controller_update_ms,
                    "feedback_elapsed_ms": feedback_elapsed_ms,
                    "draft_time_ms": draft_result["draft_time_ms"] - policy_time_ms,
                    "draft_time_including_policy_ms": draft_result["draft_time_ms"],
                    "draft_policy_time_ms": draft_result.get("draft_policy_time_ms", 0.0),
                    "draft_stopped_by_policy": draft_result.get("draft_stopped_by_policy", False),
                    "lazy_final_forward": lazy_final_forward,
                    "draft_forward_calls": draft_result.get("draft_forward_calls"),
                    "draft_tokens_pending_forward": draft_result.get("draft_tokens_pending_forward"),
                    "draft_commit_forward_tokens": commit_result.get("draft_commit_forward_tokens"),
                    "draft_commit_forward_calls": commit_result.get("draft_commit_forward_calls"),
                    "draft_execution_backend": draft_result.get("draft_execution_backend", "dynamic"),
                    "draft_graph_replays": draft_result.get("draft_graph_replays", 0),
                    "draft_commit_graph_replays": commit_result.get("draft_commit_graph_replays", 0),
                    "verify_rpc_ms": token_available_ms,
                    "numerical_guard_triggered": verification.get("numerical_guard_triggered", False),
                    "canonical_rebuild_tokens": verification.get("canonical_rebuild_tokens", 0),
                    "cloud_verify_time_ms": verification["verify_time_ms"],
                    "draft_commit_time_ms": commit_result["draft_commit_time_ms"],
                    "round_cycle_ms": round_cycle_ms,
                    "draft_kv_tokens_before": draft_result["draft_kv_tokens_before"],
                    "draft_kv_tokens_after_draft": draft_result[
                        "draft_kv_tokens_after_draft"
                    ],
                    "draft_kv_tokens_after_commit": commit_result[
                        "draft_kv_tokens_after_commit"
                    ],
                    "target_kv_tokens_before": verification[
                        "target_kv_tokens_before"
                    ],
                    "target_kv_tokens_after": verification[
                        "target_kv_tokens_after"
                    ],
                    "request_payload_bytes": verification.get(
                        "_request_payload_bytes", 0
                    ),
                    "response_payload_bytes": verification.get(
                        "_response_payload_bytes", 0
                    ),
                    "cloud_energy_j": verification.get("cloud_energy_j"),
                    "cloud_memory_mb_end": verification.get(
                        "cloud_memory_mb_end"
                    ),
                    "committed_token_ids_json": json.dumps(
                        committed_ids, separators=(",", ":")
                    ),
                }
            )
            round_index += 1
            if answer_seen:
                stop_reason = "answer"
                break
            if eos_terminal:
                stop_reason = "eos"
                break
        completion_ms = (time.perf_counter() - request_start) * 1000.0
    finally:
        cleanup_start = time.perf_counter()
        try:
            try:
                close_result = close_target_session(server, session_id)
            except requests.RequestException as exc:
                close_result = {"closed": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            cleanup_ms = (time.perf_counter() - cleanup_start) * 1000.0
            if callable(getattr(draft_session, "close", None)):
                draft_session.close()

    e2e_ms = float(completion_ms)
    wall_end_ns = time.time_ns()
    monotonic_end_ns = time.monotonic_ns()
    draft_state = draft_session.execution_state() if callable(getattr(draft_session, "execution_state", None)) else None
    request = {
        "method": controller.method,
        "wall_start_ns": wall_start_ns,
        "wall_end_ns": wall_end_ns,
        "monotonic_start_ns": monotonic_start_ns,
        "monotonic_end_ns": monotonic_end_ns,
        "initial_context_tokens": len(context_ids),
        "generation_token_cap": generation_token_cap,
        "generated_tokens": len(delivered),
        "stop_reason": stop_reason,
        "rounds": round_index,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "completion_latency_ms": e2e_ms,
        "last_token_latency_ms": last_token_latency_ms,
        "cleanup_time_ms": cleanup_ms,
        "request_wall_ms": (time.perf_counter() - request_start) * 1000.0,
        "e2e_timing_scope": "request_start_through_generation_completion_before_cleanup",
        "draft_cuda_timing": "prefill_draft_and_commit_synchronize_their_cuda_device",
        "lazy_final_forward": lazy_final_forward,
        "draft_execution_backend": draft_state["backend"] if draft_state is not None else "dynamic",
        "graph_state": draft_state,
        "decode_ms_per_token": e2e_ms / len(delivered) if delivered else None,
        "throughput_tokens_s": (
            len(delivered) / (e2e_ms / 1000.0) if e2e_ms > 0 else None
        ),
        "controller_time_ms": total_controller_ms,
        "request_payload_bytes": total_request_bytes,
        "response_payload_bytes": total_response_bytes,
        "target_prefill_time_ms": target_start["prefill_time_ms"],
        "target_prefill_rpc_ms": target_start["_network_and_server_ms"],
        "edge_prefill_time_ms": edge_prefill_ms,
        "parallel_prefill_wall_ms": parallel_prefill_ms,
        "target_session_closed": bool(close_result["closed"]),
        "cloud_energy_j": (
            float(sum(float(value) for value in cloud_energy_values))
            if all(value is not None for value in cloud_energy_values)
            else None
        ),
        "cloud_peak_memory_mb": cloud_peak_memory_mb,
        "delivered_token_ids_json": json.dumps(delivered, separators=(",", ":")),
        "delivered_token_ids_sha256": token_ids_sha256(delivered),
    }
    return request, round_rows


def run_direct_target_request(
    server: str,
    context_ids: Sequence[int],
    generation_token_cap: int,
    one_way_delay_ms: float = 0.0,
    answer_stop: bool = False,
) -> Dict[str, object]:
    """Measure a reasonable one-connection streaming direct-target baseline."""

    if generation_token_cap <= 0:
        raise ValueError("generation_token_cap must be positive")
    wall_start_ns = time.time_ns()
    monotonic_start_ns = time.monotonic_ns()
    request_start = time.perf_counter()
    payload = {
        "context_ids": [int(value) for value in context_ids],
        "max_new_tokens": generation_token_cap,
        "answer_stop": answer_stop,
    }
    serialize_start = time.perf_counter()
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
    if one_way_delay_ms > 0:
        time.sleep(one_way_delay_ms / 1000.0)
    response = requests.post(
        f"{server.rstrip('/')}/generate/stream",
        data=body,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    if one_way_delay_ms > 0:
        time.sleep(one_way_delay_ms / 1000.0)

    delivered: List[int] = []
    ttft_ms: Optional[float] = None
    last_token_latency_ms: Optional[float] = None
    done: Optional[Dict[str, object]] = None
    response_bytes = 0
    try:
        for line in response.iter_lines(chunk_size=1):
            if not line:
                continue
            response_bytes += len(line) + 1
            event = json.loads(line)
            if event.get("event") == "token":
                token_latency_ms = (time.perf_counter() - request_start) * 1000.0
                if ttft_ms is None:
                    ttft_ms = token_latency_ms
                last_token_latency_ms = token_latency_ms
                delivered.append(int(event["token_id"]))
            elif event.get("event") == "done":
                done = event
            else:
                raise RuntimeError(f"unknown direct-stream event: {event}")
        if done is None:
            raise RuntimeError("direct target stream ended without a done event")
        if int(done["generated_tokens"]) != len(delivered):
            raise RuntimeError("direct target count disagrees with delivered token events")
        e2e_ms = (time.perf_counter() - request_start) * 1000.0
    finally:
        cleanup_start = time.perf_counter()
        response.close()
        cleanup_ms = (time.perf_counter() - cleanup_start) * 1000.0
    return {
        "method": "direct_target",
        "wall_start_ns": wall_start_ns,
        "wall_end_ns": time.time_ns(),
        "monotonic_start_ns": monotonic_start_ns,
        "monotonic_end_ns": time.monotonic_ns(),
        "initial_context_tokens": len(context_ids),
        "generation_token_cap": generation_token_cap,
        "generated_tokens": len(delivered),
        "stop_reason": done["stop_reason"],
        "rounds": 1,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "completion_latency_ms": e2e_ms,
        "last_token_latency_ms": last_token_latency_ms,
        "cleanup_time_ms": cleanup_ms,
        "request_wall_ms": (time.perf_counter() - request_start) * 1000.0,
        "e2e_timing_scope": "request_start_through_generation_completion_before_cleanup",
        "decode_ms_per_token": e2e_ms / len(delivered) if delivered else None,
        "throughput_tokens_s": (
            len(delivered) / (e2e_ms / 1000.0) if e2e_ms > 0 else None
        ),
        "controller_time_ms": 0.0,
        "request_payload_bytes": len(body),
        "response_payload_bytes": response_bytes,
        "json_serialize_ms": serialize_ms,
        "server_prefill_time_ms": done["prefill_time_ms"],
        "server_total_ms": done["server_total_ms"],
        "cloud_energy_j": done.get("cloud_energy_j"),
        "cloud_process_peak_allocated_mb": done.get(
            "cloud_process_peak_allocated_mb"
        ),
        "delivered_token_ids_json": json.dumps(delivered, separators=(",", ":")),
        "delivered_token_ids_sha256": token_ids_sha256(delivered),
    }
