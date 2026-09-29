"""Frozen CGP comparison using actual continuous edge/target KV sessions.

Cold learners get no calibration. Warm learners get identical calibration.
The fixed deployment reference always uses calibration, charged separately.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from hardware.cgp_controller import COLD_METHODS, WARM_METHODS, make_controller
from hardware.closed_loop import run_speculative_request, run_direct_target_request
from hardware.graph_draft import DraftGraphEngine, GraphDraftSession
from hardware.run_closed_loop_experiment import JsonlSink, _atomic_json, flatten_jsonl_to_csv
import hardware.run_tmc_revision as tmc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_request(result, rounds, initial_length):
    state = result["graph_state"]
    if not result["target_session_closed"]:
        raise RuntimeError("request cache cleanup or graph backend mismatch")
    if state is not None:
        if not state["session_closed"] or state["backend"] != "cuda_graph_static_cache":
            raise RuntimeError("graph backend cleanup mismatch")
    elif result["draft_execution_backend"] != "dynamic":
        raise RuntimeError("unexpected draft backend")
    length = initial_length
    for row in rounds:
        if row["draft_kv_tokens_before"] != length or row["target_kv_tokens_before"] != length:
            raise RuntimeError("prefill/prefix cache mismatch before round")
        length += row["committed_model_tokens"]
        if row["draft_kv_tokens_after_commit"] != length or row["target_kv_tokens_after"] != length:
            raise RuntimeError("cache mismatch after round commit")
    if state is not None and (state["final_logical_kv_tokens"] != length or state["engine_position"] != length):
        raise RuntimeError("final draft cache mismatch")
    if sum(row["delivered_tokens"] for row in rounds) != result["generated_tokens"]:
        raise RuntimeError("delivered token accounting mismatch")


def main(controller_factory=make_controller, cold_methods=COLD_METHODS, warm_methods=WARM_METHODS):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "frozen" or plan.get("protocol") != "cgp_continuous_20260922":
        raise ValueError("requires frozen CGP plan")
    methods = list(cold_methods if plan["information_budget"] == "cold" else warm_methods)
    if plan["methods"] != methods:
        raise ValueError("retain all declared comparison methods")
    for name, expected in plan["code_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("source mismatch: " + name)
    inputs = {}
    for key in ("prompts", "calibration"):
        path = plan_path.parent / plan[key + "_file"]
        if digest(path) != plan[key + "_sha256"]:
            raise ValueError("input mismatch: " + key)
        inputs[key] = path
    prompts = [json.loads(line) for line in inputs["prompts"].read_text().splitlines() if line.strip()]
    cal = json.loads(inputs["calibration"].read_text())
    if set(p["prompt_id"] for p in prompts) & set(cal["calibration_prompt_ids"]):
        raise ValueError("request prompts overlap calibration")
    if set(hashlib.sha256(p["prompt"].encode()).hexdigest() for p in prompts) & set(cal["calibration_prompt_text_sha256"]):
        raise ValueError("request prompt text overlaps calibration")
    if plan["phase"] == "evaluation" and any(p["split"] != "closed_loop_evaluation" for p in prompts):
        raise ValueError("evaluation requires untouched allocated prompts")
    if cal["runtime_launcher_sha256"] != plan["runtime_launcher_sha256"]:
        raise ValueError("calibration runtime differs")
    if Path(args.draft_model).name != "Qwen2.5-0.5B":
        raise ValueError("unexpected draft model")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, out / "frozen_plan.json")
    for key, path in inputs.items():
        shutil.copyfile(path, out / (key + path.suffix))
    manifest = {"completed": False, "phase": plan["phase"], "measurement_session": plan["measurement_session"],
                "information_budget": plan["information_budget"], "plan_sha256": digest(plan_path),
                "hostname": socket.gethostname(), "python": platform.python_version(),
                "started_wall_time_ns": time.time_ns(), "code_sha256": plan["code_sha256"],
                "runtime_launcher_sha256": plan["runtime_launcher_sha256"],
                "timing_scope": "completion includes prefill, selection, drafting, verification, updates and commit; cleanup also recorded",
                "sampling_scope": "one edge/cloud pair; separately restarted measurement sessions"}
    _atomic_json(out / "run_manifest.json", manifest)
    sinks = {key: JsonlSink(out / (key + ".jsonl")) for key in
             ("request_metrics", "round_metrics", "output_equality_audit", "failures")}
    telemetry = tmc.TegrastatsSampler(out / "edge_tegrastats.csv", 200)
    mismatches = 0
    try:
        pings = {}
        for condition in plan["conditions"]:
            ping = tmc.probe_server(condition["server"])
            tmc.validate_server_capabilities(ping, plan["code_sha256"]["hardware/cloud_server.py"],
                                             require_request_window_telemetry=True, require_session_protocol=True)
            for field, expected in plan["expected_target_execution"].items():
                if ping.get(field) != expected:
                    raise ValueError("unexpected target execution: " + field)
            pings[condition["condition_id"]] = ping
        manifest["server_pings"] = pings
        tmc.hw.load_model(args.draft_model, allow_download=False)
        import torch
        import transformers
        gc.collect()
        torch.cuda.empty_cache()
        manifest.update(torch=torch.__version__, transformers=transformers.__version__,
                        draft_backend=plan["draft_backend"],
                        device_model=Path('/proc/device-tree/model').read_text().strip('\x00'))
        manifest["termination"] = tmc.configure_draft_termination_for_target(tmc.hw._tok, tmc.hw._model, next(iter(pings.values())))
        manifest["draft_dtype"] = str(next(tmc.hw._model.parameters()).dtype)
        if manifest["draft_dtype"] != "torch.float16":
            raise ValueError("unexpected draft dtype")
        engine = None
        if plan["draft_backend"] == "cuda_graph":
            engine = DraftGraphEngine(tmc.hw._model, max_cache_len=512)
        elif plan["draft_backend"] != "dynamic":
            raise ValueError("unknown draft backend")
        manifest["graph_setup"] = engine.setup_metrics if engine is not None else {
            "backend": "dynamic", "actual_enabled": False, "fallback_used": False}
        # Uniform setup warm-up uses calibration text, never evaluation prompts.
        from hardware.closed_loop import ServiceController
        warm_prompt = json.loads((plan_path.parent / "calibration-prompts.jsonl").read_text().splitlines()[0])["prompt"]
        warm_context = tmc.hw._tok(warm_prompt, return_tensors="pt").input_ids[0].tolist()
        warm_factory = None
        if engine is not None:
            def warm_factory(model, tokenizer, context_ids, **kwargs):
                return GraphDraftSession(engine, tokenizer, context_ids, max_new_tokens=8, **kwargs)
        warm_result, warm_rounds = run_speculative_request(
            plan["conditions"][0]["server"], tmc.hw._model, tmc.hw._tok, warm_context,
            ServiceController("fixed", list(range(1, 11)), fixed_arm=1), 8,
            lazy_final_forward=True, draft_session_factory=warm_factory)
        validate_request(warm_result, warm_rounds, len(warm_context))
        _atomic_json(out / "setup_warmup.json", {"result": warm_result, "rounds": warm_rounds,
                     "scope": "calibration prompt, outside evaluation timer, controller discarded"})
        controllers = {(c["condition_id"], m): controller_factory(m, cal["conditions"][c["condition_id"]])
                       for c in plan["conditions"] for m in methods if m != "direct_target"}
        _atomic_json(out / "run_manifest.json", manifest)
        telemetry.start()
        rng = np.random.default_rng(plan["seed"])
        blocks = [(c, cap, p) for c in plan["conditions"] for cap in plan["generation_token_caps"] for p in prompts]
        for block_number, block_index in enumerate(rng.permutation(len(blocks))):
            condition, cap, prompt = blocks[int(block_index)]
            context = tmc.hw._tok(prompt["prompt"], return_tensors="pt").input_ids[0].tolist()
            cid = condition["condition_id"]
            identity = {"measurement_session": plan["measurement_session"], "phase": plan["phase"],
                        "condition_id": cid, "prompt_id": prompt["prompt_id"], "prompt_stratum": prompt["stratum"],
                        "prompt_tokens": len(context), "generation_token_cap": cap,
                        "block_execution_order": block_number,
                        "block_key": f"{plan['measurement_session']}|{cid}|{cap}|{prompt['prompt_id']}"}
            results = []
            for order, method_index in enumerate(rng.permutation(len(methods))):
                method = methods[int(method_index)]
                common = {**identity, "method": method, "method_execution_order": order}
                print(f"{plan['measurement_session']} block {block_number + 1}/{len(blocks)} {cid} {method}", flush=True)
                try:
                    if method == "direct_target":
                        result = run_direct_target_request(condition["server"], context, cap)
                        rounds = []
                    else:
                        controller = controllers[(cid, method)]
                        if hasattr(controller, "begin_request"):
                            controller.begin_request()
                        factory = None
                        if engine is not None:
                            def factory(model, tokenizer, context_ids, **kwargs):
                                return GraphDraftSession(engine, tokenizer, context_ids, max_new_tokens=cap, **kwargs)
                        result, rounds = run_speculative_request(condition["server"], tmc.hw._model, tmc.hw._tok,
                            context, controller, cap, lazy_final_forward=True, draft_session_factory=factory)
                        if hasattr(controller, "decisions"):
                            if len(rounds) != len(controller.decisions):
                                raise RuntimeError("decision/round audit count mismatch")
                            for row, decision in zip(rounds, controller.decisions):
                                row.update(decision)
                    sinks["request_metrics"].write({**result, **common})
                    for row in rounds:
                        sinks["round_metrics"].write({**row, **common})
                    if method != "direct_target":
                        validate_request(result, rounds, len(context))
                    results.append({**result, **common})
                except Exception as exc:
                    sinks["failures"].write({**common, "error": f"{type(exc).__name__}: {exc}"})
                    raise
            equal = len({(r["delivered_token_ids_sha256"], r["generated_tokens"], r["stop_reason"]) for r in results}) == 1
            mismatches += int(not equal)
            sinks["output_equality_audit"].write({**identity, "all_outputs_equal": equal,
                "outputs": {r["method"]: {k: r[k] for k in ("delivered_token_ids_sha256", "generated_tokens", "stop_reason")} for r in results}})
            _atomic_json(out / "controller_state.json", {f"{c}::{m}": obj.state_dict() for (c, m), obj in controllers.items()})
        manifest["completed"] = True
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        telemetry.stop()
        for key, sink in sinks.items():
            sink.close()
            flatten_jsonl_to_csv(out / (key + ".jsonl"), out / (key + ".csv"))
        manifest.update(finished_wall_time_ns=time.time_ns(), output_mismatch_blocks=mismatches,
                        request_rows=sinks["request_metrics"].rows, round_rows=sinks["round_metrics"].rows,
                        failure_rows=sinks["failures"].rows, telemetry_error=telemetry.error)
        manifest["valid_for_performance_comparison"] = manifest["completed"] and mismatches == 0 and sinks["failures"].rows == 0
        manifest["output_sha256"] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "run_manifest.json"}
        _atomic_json(out / "run_manifest.json", manifest)
    print(json.dumps({"completed": manifest["completed"], "mismatches": mismatches, "requests": manifest["request_rows"]}), flush=True)
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
