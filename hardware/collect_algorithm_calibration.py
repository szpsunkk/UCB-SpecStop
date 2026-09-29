"""Collect complete paired calibration cells with real cached request primitives.

Only frozen algorithm_calibration plans are accepted. Target continuations
create calibration contexts; these are not evaluation prompts or controller
feedback. Each arm receives a fresh target session and draft KV prefill at the
same context. N covers features through draft, verification, and commit; both
prefills and session cleanup are recorded separately. No raw table is changed.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import shutil
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware.probe_prefix_consistency import sha256_file, _write_manifest


def validate_plan(plan: dict, plan_path: Path):
    if plan.get("schema_version") != 1 or plan.get("status") != "frozen" or plan.get("phase") != "algorithm_calibration":
        raise ValueError("plan requires schema_version=1, status=frozen, phase=algorithm_calibration")
    if not plan.get("measurement_session"):
        raise ValueError("measurement_session is required")
    offsets, arms = plan.get("contexts_offsets", []), plan.get("arms", [])
    arm_minimum = 0 if plan.get("allow_zero_arm") is True else 1
    for name, values, minimum in (("contexts_offsets", offsets, 0), ("arms", arms, arm_minimum)):
        if not values or len(set(values)) != len(values) or any(isinstance(x, bool) or not isinstance(x, int) or x < minimum for x in values):
            raise ValueError(f"{name} must contain unique integers >= {minimum}")
    if 0 not in offsets:
        raise ValueError("contexts_offsets must include the original prompt offset 0")
    for name, default, minimum in (("seed", None, 0), ("repeats", 1, 1)):
        value = plan.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    conditions = plan.get("conditions", [])
    if not conditions or len({str(row.get("condition_id")) for row in conditions}) != len(conditions):
        raise ValueError("conditions must be nonempty with unique IDs")
    for condition in conditions:
        if not condition.get("condition_id") or not str(condition.get("server", "")).startswith(("http://", "https://")):
            raise ValueError("every condition requires condition_id and HTTP server")
        delay = float(condition.get("extra_one_way_delay_ms", 0))
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("extra_one_way_delay_ms must be finite and nonnegative")
    path = Path(plan["prompts_file"]).expanduser()
    path = path.resolve() if path.is_absolute() else (plan_path.parent / path).resolve()
    if plan.get("prompts_sha256") and sha256_file(path) != plan["prompts_sha256"]:
        raise ValueError("prompt source differs from the frozen hash")
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts or len({str(row["prompt_id"]) for row in prompts}) != len(prompts):
        raise ValueError("calibration JSONL requires unique prompt IDs")
    for row in prompts:
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("every prompt record requires nonempty prompt text")
        if "eval" in str(row.get("split", "")).lower():
            raise ValueError("evaluation prompts cannot enter algorithm calibration")
    return path, prompts


def measure_arm(
    server, model, tokenizer, context_ids, arm, one_way_delay_ms=0.0,
    session_factory=None, start_fn=None, verify_fn=None, close_fn=None,
    feature_fn=None, sync_fn=None,
    lazy_final_forward=False,
    collect_token_features=False,
    gate_stop_cost=False, gate_max_arm=10,
    collect_initial_features=True,
) -> dict:
    """Measure a single fresh cached cycle; injectable services support offline tests."""

    if any(callback is None for callback in (session_factory, start_fn, verify_fn, close_fn)):
        from hardware.closed_loop import (
            GreedyDraftSession, start_target_session, verify_target_session, close_target_session,
        )
        session_factory = session_factory or GreedyDraftSession
        start_fn = start_fn or start_target_session
        verify_fn = verify_fn or verify_target_session
        close_fn = close_fn or close_target_session
    if feature_fn is None or sync_fn is None:
        from hardware.content_features import extract_draft_features, synchronize_device
        feature_fn = feature_fn or extract_draft_features
        sync_fn = sync_fn or synchronize_device
    result, session_id, draft_session = {}, None, None
    acquisition_start = time.perf_counter()
    try:
        start_time = time.perf_counter()
        target = start_fn(server, context_ids, one_way_delay_ms)
        target_prefill_rpc_ms = (time.perf_counter() - start_time) * 1000
        session_id = str(target["session_id"])
        start_time = time.perf_counter()
        draft_options = {"lazy_final_forward": True} if lazy_final_forward else {}
        draft_session = session_factory(model, tokenizer, context_ids, **draft_options)
        sync_fn(draft_session.device)
        edge_prefill_ms = (time.perf_counter() - start_time) * 1000
        if int(target["target_kv_tokens"]) != draft_session.kv_tokens or draft_session.kv_tokens != len(context_ids):
            raise RuntimeError("draft and target prefill lengths must equal the fixed context length")
        # Prefill is complete before this boundary. Feature computation is
        # available before selection and must be charged to content controllers.
        cycle_start = time.perf_counter()
        features = feature_fn(draft_session) if collect_initial_features else {}
        feature_time_ms = (time.perf_counter() - cycle_start) * 1000 if collect_initial_features else 0.0
        token_features = []
        def observe_token(position, logits, context_tokens):
            from hardware.content_features import extract_content_features
            feature_start = time.perf_counter()
            observed = extract_content_features(logits, context_tokens)
            token_features.append({"token_position": position, **observed,
                                   "feature_wall_ms": (time.perf_counter() - feature_start) * 1000.0})
            return not (gate_stop_cost and position > arm)
        draft_options = {"token_decider": observe_token} if collect_token_features else {}
        if gate_stop_cost and (not collect_token_features or arm > gate_max_arm):
            raise ValueError("gate cost requires per-token features and a supported maximum length")
        requested_draft = arm + 1 if gate_stop_cost and arm < gate_max_arm else arm
        draft_result = draft_session.draft(requested_draft, **draft_options)
        sync_fn(draft_session.device)
        verify_start = time.perf_counter()
        verification = verify_fn(server, session_id, draft_result["draft_ids"], one_way_delay_ms)
        verify_rpc_ms = (time.perf_counter() - verify_start) * 1000
        committed = [int(token) for token in verification["committed_token_ids"]]
        accepted = int(verification["n_accepted"])
        delivered = []
        eos_seen = False
        for token in committed:
            if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
                eos_seen = True
                break
            delivered.append(token)
        commit_result = draft_session.commit(accepted, committed)
        sync_fn(draft_session.device)
        complete_cycle_ms = (time.perf_counter() - cycle_start) * 1000
        if draft_session.kv_tokens != len(context_ids) + len(committed):
            raise RuntimeError("draft cache after commit disagrees with committed token count")
        if int(verification["target_kv_tokens_after"]) != draft_session.kv_tokens:
            raise RuntimeError("target and draft cache lengths differ after commit")
        terminal_eos = eos_seen or verification.get("stop_reason") == "eos"
        shortened = len(draft_result["draft_ids"]) != arm
        censored = terminal_eos or shortened or not delivered
        result.update({
            "status": "ok", "N_ms": complete_cycle_ms, "A": len(delivered),
            "total_round_time_ms": complete_cycle_ms,
            "feedback_elapsed_ms": complete_cycle_ms,
            "accepted_total": len(delivered),
            "accepted_draft_tokens": accepted,
            "committed_model_tokens": len(committed),
            "delivered_tokens": len(delivered),
            "draft_ids": draft_result["draft_ids"],
            "committed_token_ids": committed,
            "draft_stopped_before_requested_length": shortened,
            "terminal_eos": terminal_eos,
            "arm_censored": censored, "full_round": not censored,
            "warm_eligible": not censored,
            "censor_reason": "terminal_eos" if terminal_eos else "draft_terminated_early" if shortened else "zero_delivered_tokens" if not delivered else None,
            "feature_extraction_time_ms": feature_time_ms + float(draft_result.get("draft_policy_time_ms", 0.0)),
            "initial_feature_extraction_time_ms": feature_time_ms,
            "draft_token_features": token_features,
            "gate_stop_cost": gate_stop_cost,
            "draft_feature_time_ms": float(draft_result.get("draft_policy_time_ms", 0.0)),
            "draft_time_ms": draft_result.get("draft_time_ms"),
            "lazy_final_forward": lazy_final_forward,
            "draft_forward_calls": draft_result.get("draft_forward_calls"),
            "draft_commit_forward_calls": commit_result.get("draft_commit_forward_calls"),
            "draft_execution_backend": draft_result.get("draft_execution_backend", "dynamic"),
            "draft_graph_replays": draft_result.get("draft_graph_replays", 0),
            "draft_commit_graph_replays": commit_result.get("draft_commit_graph_replays", 0),
            "verify_rpc_ms": verify_rpc_ms,
            "draft_commit_time_ms": commit_result.get("draft_commit_time_ms"),
            "target_prefill_rpc_wall_ms": target_prefill_rpc_ms,
            "target_prefill_time_ms": target.get("prefill_time_ms"),
            "edge_prefill_wall_ms": edge_prefill_ms,
            "target_start_response": target,
            "verification_response": verification,
            "draft_kv_tokens_after_commit": draft_session.kv_tokens,
            **features,
        })
    finally:
        if session_id is not None:
            close_start = time.perf_counter()
            try:
                closed = close_fn(server, session_id)
                result["target_session_closed"] = bool(closed.get("closed"))
                result["target_close_response"] = closed
            except Exception as exc:
                result["target_session_closed"] = False
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["cleanup_time_ms"] = (time.perf_counter() - close_start) * 1000
        result["acquisition_wall_ms"] = (time.perf_counter() - acquisition_start) * 1000
        if callable(getattr(draft_session, "close", None)):
            draft_session.close()
        if callable(getattr(draft_session, "execution_state", None)):
            result["graph_state"] = draft_session.execution_state()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--draft-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()
    plan_path = args.plan.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prompt_path, prompts = validate_plan(plan, plan_path)
    output = args.out_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan_path, output / "frozen_plan.json")
    shutil.copy2(prompt_path, output / "prompts_snapshot.jsonl")
    manifest = {
        "schema_version": 1, "phase": "algorithm_calibration", "completed": False,
        "measurement_session": plan["measurement_session"],
        "started_wall_time_ns": time.time_ns(), "row_count": 0,
        "plan_sha256": sha256_file(plan_path), "prompts_sha256": sha256_file(prompt_path),
        "planned_arm_rows": len(prompts) * len(plan["contexts_offsets"]) * len(plan["conditions"]) * len(plan["arms"]) * plan.get("repeats", 1),
        "timing_scope": "N includes feature extraction through cached draft/verify/commit, with device completion; excludes prefills, cleanup, controller update, and a learned selection calculation",
        "features_available": "draft next_logits before any selected-arm drafting",
        "yield_definition": "A is delivered tokens excluding EOS; committed_model_tokens also records EOS when committed",
        "eligibility": "all rows retained; warm_eligible only marks complete nonterminal positive-yield arm cycles",
        "context_scope": "calibration-only target continuation, then identical fresh prefills per arm; no continuous-policy evaluation",
        "draft_model": args.draft_model,
        "hostname": socket.gethostname(), "python": platform.python_version(),
    }
    manifest_path = output / "manifest.json"
    _write_manifest(manifest_path, manifest)
    rows_path, contexts_path = output / "rows.jsonl", output / "contexts.jsonl"
    try:
        import torch
        import transformers
        import hardware.run_tmc_revision as tmc
        from hardware.closed_loop import run_direct_target_request, token_ids_sha256
        import hardware.closed_loop as closed_loop
        import hardware.content_features as content_features

        hw = tmc.hw
        continuation_server = plan.get("target_continuation_server", plan["conditions"][0]["server"])
        servers = list(dict.fromkeys([continuation_server] + [row["server"] for row in plan["conditions"]]))
        pings = {server: tmc.probe_server(server) for server in servers}
        for server, ping in pings.items():
            if ping.get("status") != "ok" or not ping.get("session_verification"):
                raise RuntimeError(f"server {server} lacks the cached session protocol: {ping}")
            if plan.get("expected_target_model_basename") and Path(str(ping.get("model"))).name != plan["expected_target_model_basename"]:
                raise RuntimeError("target model differs from frozen plan")
            if plan.get("expected_target_dtype") and ping.get("target_dtype") != plan["expected_target_dtype"]:
                raise RuntimeError("target precision differs from frozen plan")
            if plan.get("require_synchronized_target_responses") and ping.get("session_response_cuda_synchronized") is not True:
                raise RuntimeError("target must complete CUDA work before measured session responses")
            for field, value in plan.get("expected_target_execution", {}).items():
                if ping.get(field) != value:
                    raise RuntimeError(f"target execution differs for {field}")
        if len({(str(ping.get("model")), str(ping.get("target_dtype"))) for ping in pings.values()}) != 1:
            raise RuntimeError("all conditions must share the same target model and dtype")
        hw.load_model(args.draft_model, allow_download=args.allow_download)
        gc.collect()
        torch.cuda.empty_cache()
        if args.draft_dtype != "float16":
            hw._model.to(dtype=getattr(torch, args.draft_dtype))
        termination = tmc.configure_draft_termination_for_target(hw._tok, hw._model, pings[continuation_server])
        graph_engine = None
        graph_sources = {}
        session_factory = None
        if plan.get("draft_backend", "dynamic") == "cuda_graph":
            import hardware.graph_draft as graph_draft
            graph_engine = graph_draft.DraftGraphEngine(hw._model, max_cache_len=int(plan.get("graph_max_cache_len", 512)))
            def session_factory(model, tokenizer, context_ids, **kwargs):
                return graph_draft.GraphDraftSession(graph_engine, tokenizer, context_ids,
                                                     max_new_tokens=max(plan["arms"])+1, **kwargs)
            graph_sources[str(Path(graph_draft.__file__).resolve())] = sha256_file(Path(graph_draft.__file__).resolve())
        torch.manual_seed(plan["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(plan["seed"])
        manifest.update({
            "server_pings": pings, "termination": termination,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "draft_dtype": str(next(hw._model.parameters()).dtype),
            "draft_device": str(hw._device), "cuda": torch.version.cuda,
            "source_sha256": {
                str(Path(module.__file__).resolve()): sha256_file(Path(module.__file__).resolve())
                for module in (hw, tmc, closed_loop, content_features)
            },
            "collector_source_sha256": sha256_file(Path(__file__).resolve()),
            "draft_backend": plan.get("draft_backend", "dynamic"),
            "graph_setup": graph_engine.setup_metrics if graph_engine is not None else None,
            "graph_source_sha256": graph_sources,
        })
        _write_manifest(manifest_path, manifest)
        rng = np.random.default_rng(plan["seed"])
        maximum_offset = max(plan["contexts_offsets"])
        with rows_path.open("x", encoding="utf-8") as rows, contexts_path.open("x", encoding="utf-8") as contexts:
            def write_row(row):
                rows.write(json.dumps(row, sort_keys=True) + "\n")
                rows.flush()
                manifest["row_count"] += 1

            for prompt_position in rng.permutation(len(prompts)):
                prompt = prompts[int(prompt_position)]
                initial = hw._tok(prompt["prompt"], return_tensors="pt").input_ids[0].tolist()
                preparation_start = time.perf_counter()
                continuation_request = (
                    run_direct_target_request(continuation_server, initial, maximum_offset)
                    if maximum_offset else {"delivered_token_ids_json": "[]", "stop_reason": "not_requested"}
                )
                continuation = json.loads(continuation_request["delivered_token_ids_json"])
                context_record = {
                    "prompt_id": prompt["prompt_id"], "phase": "algorithm_calibration",
                    "initial_context_ids": initial, "target_continuation_ids": continuation,
                    "target_continuation_request": continuation_request,
                    "preparation_wall_ms": (time.perf_counter() - preparation_start) * 1000,
                }
                contexts.write(json.dumps(context_record, sort_keys=True) + "\n")
                contexts.flush()
                cells = [
                    (offset, repetition, condition)
                    for offset in plan["contexts_offsets"]
                    for repetition in range(plan.get("repeats", 1))
                    for condition in plan["conditions"]
                ]
                for position in rng.permutation(len(cells)):
                    offset, repetition, condition = cells[int(position)]
                    context = initial + continuation[:offset]
                    cell_id = f"{prompt['prompt_id']}:{offset}:{repetition}:{condition['condition_id']}"
                    for arm_order, arm_position in enumerate(rng.permutation(len(plan["arms"]))):
                        arm = plan["arms"][int(arm_position)]
                        row = {
                            "phase": "algorithm_calibration", "cell_id": cell_id,
                            "measurement_session": plan["measurement_session"],
                            "prompt_id": prompt["prompt_id"], "context_offset": offset,
                            "repetition": repetition, "condition_id": condition["condition_id"],
                            "server": condition["server"], "arm_evaluation_order": arm_order,
                            "k_selected": arm, "context_ids": context,
                            "context_token_ids_sha256": token_ids_sha256(context),
                            "context_tokens": len(context), "started_wall_time_ns": time.time_ns(),
                        }
                        if offset > len(continuation):
                            row.update({"status": "unreachable_context_offset", "warm_eligible": False, "reason": "target_continuation_ended_before_planned_offset", "actual_continuation_tokens": len(continuation)})
                            write_row(row)
                            continue
                        try:
                            row.update(measure_arm(
                                condition["server"], hw._model, hw._tok, context, arm,
                                float(condition.get("extra_one_way_delay_ms", 0)),
                                lazy_final_forward=bool(plan.get("lazy_final_forward", False)),
                                collect_token_features=bool(plan.get("collect_token_features", False)),
                                collect_initial_features=bool(plan.get("collect_initial_features", True)),
                                gate_stop_cost=bool(plan.get("gate_stop_cost", False)),
                                gate_max_arm=max(plan["arms"]),
                                session_factory=session_factory,
                            ))
                        except Exception as exc:
                            row.update({"status": "error", "warm_eligible": False, "error": f"{type(exc).__name__}: {exc}"})
                            write_row(row)
                            raise
                        row["finished_wall_time_ns"] = time.time_ns()
                        write_row(row)
                        if not row.get("target_session_closed", False):
                            raise RuntimeError("target session cleanup failed; partial archive retained")
        manifest["completed"] = True
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_wall_time_ns"] = time.time_ns()
        manifest["output_sha256"] = {
            path.name: sha256_file(path) for path in (rows_path, contexts_path) if path.exists()
        }
        _write_manifest(manifest_path, manifest)
    print(json.dumps({"out_dir": str(output), "completed": manifest["completed"], "rows": manifest["row_count"]}))


if __name__ == "__main__":
    main()
