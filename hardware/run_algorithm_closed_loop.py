"""Run a frozen, calibrated controller comparison on actual edge hardware.

This runner is separate from the historical six-method revision protocol.
All policies share calibration data and use the same transport endpoint in
each condition. The content model stays frozen throughout evaluation.
"""

from __future__ import annotations

import argparse
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

from hardware.closed_loop import ServiceController, run_direct_target_request, run_speculative_request
from hardware.run_closed_loop_experiment import JsonlSink, _atomic_json, flatten_jsonl_to_csv
from hardware.content_features import extract_draft_features
from hardware.content_features import extract_content_features
import hardware.run_tmc_revision as tmc
from src.content_ratio import ContentRatioModel
from src.prefix_hazard import PrefixHazardModel


METHODS = ("sn", "per_round_ucb", "ratio_regularized", "ratio_residual",
           "content_ratio", "content_simple", "fixed", "direct_target")
METHODS_V2 = METHODS + ("prefix_hazard", "prefix_top1", "prefix_entropy")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(plan_path, path):
    value = Path(path)
    return value if value.is_absolute() else (plan_path.parent / value).resolve()


class FrozenContentController:
    def __init__(self, method, model):
        self.method = method
        self.model = model
        self.rounds_selected = 0
        self.decisions = []

    def select_arm_with_context(self, draft_session, remaining_tokens):
        features = extract_draft_features(draft_session)
        arm = self.model.choose(features)
        self.rounds_selected += 1
        self.decisions.append({**features, "remaining_tokens_before_selection": remaining_tokens})
        return arm

    def update(self, selected_arm, elapsed_ms, delivered_tokens, censored=False, censor_reason=None):
        return False, "frozen_content_model"

    def state_dict(self):
        return {"method": self.method, "rounds_selected": self.rounds_selected,
                "model": self.model.state, "evaluation_outcome_updates": 0}


class FrozenPrefixController(FrozenContentController):
    """Use only the features reached during actual sequential drafting."""

    def select_arm_with_context(self, draft_session, remaining_tokens):
        self.gate = self.model.new_round(max_draft_tokens=min(self.model.k_max, max(remaining_tokens - 1, 0)))
        self.rounds_selected += 1
        self.active_decisions = []
        self.decisions.append({"remaining_tokens_before_selection": remaining_tokens,
                               "prefix_decisions": self.active_decisions})
        return self.model.k_max

    def decide_draft_token(self, position, next_logits, context_tokens):
        features = extract_content_features(next_logits, context_tokens)
        proceed = self.gate(position, features)
        self.active_decisions.append({**features, **self.gate.decisions[-1]})
        return proceed

    def update(self, *args, **kwargs):
        return False, "frozen_prefix_hazard_model"


def make_controller(method, condition_calibration):
    cal = condition_calibration
    if method in ("prefix_hazard", "prefix_top1", "prefix_entropy"):
        return FrozenPrefixController(method, PrefixHazardModel(cal["gate_models"][method]))
    if method in ("content_ratio", "content_simple"):
        return FrozenContentController(method, ContentRatioModel(cal["content_models"][method]))
    params = cal["controller_parameters"]
    controller = ServiceController(
        method=method, actual_arms=cal["arms"],
        beta=float(params.get(method + "_beta", 1.0)),
        prior_strength=float(params.get("ratio_regularized_prior_strength", 0.0)),
        fixed_arm=int(cal["fixed_arm"]) if method == "fixed" else None,
        prefix_compatible=False,
    )
    if method != "fixed":
        controller.warm_start(np.asarray(cal["costs"]), np.asarray(cal["yields"]), arms=cal["arms"])
    return controller


def validate_calibration_execution(plan, calibration):
    """Reject incompatible frozen costs before server access or model loading."""
    revision = plan.get("execution_revision", 1)
    backend = plan.get("draft_backend", "dynamic")
    if type(revision) is not int or revision not in (1, 2, 3):
        raise ValueError("unsupported execution revision")
    if backend != ("cuda_graph" if revision == 3 else "dynamic"):
        raise ValueError("draft backend differs from its execution revision")
    contract = calibration.get("execution_contract")
    if contract is None and revision == 1:
        return  # Historical revision-1 calibration predates this contract.
    if not isinstance(contract, dict):
        raise ValueError("calibration execution contract is required")
    if calibration.get("schema_version") != 1 or calibration.get("phase") != "frozen_algorithm_calibration":
        raise ValueError("calibration must be a frozen schema-1 calibration")
    if contract.get("execution_revision") != revision or contract.get("draft_backend") != backend:
        raise ValueError("calibration execution revision or backend differs from the plan")
    if plan.get("lazy_final_forward") is not True:
        raise ValueError("calibrated revision-2/3 execution requires lazy final forwarding")
    planned_conditions = {condition["condition_id"] for condition in plan["conditions"]}
    if set(calibration.get("conditions", {})) != planned_conditions:
        raise ValueError("calibration conditions differ from the request plan")
    if plan.get("include_prefix_hazard", False):
        for condition in planned_conditions:
            models = calibration["conditions"][condition].get("gate_models", {})
            if any(not isinstance(models.get(method), dict) or not models[method]
                   for method in ("prefix_hazard", "prefix_top1", "prefix_entropy")):
                raise ValueError(f"calibration lacks the planned prefix models: {condition}")
    if backend == "dynamic":
        if (contract.get("graph_max_cache_len") is not None or
                contract.get("graph_source_sha256") != {} or contract.get("graph_setup_contract") is not None):
            raise ValueError("dynamic calibration contains graph execution metadata")
        return

    capacity = plan.get("graph_max_cache_len")
    if type(capacity) is not int or capacity < 2 or contract.get("graph_max_cache_len") != capacity:
        raise ValueError("calibration graph cache capacity differs from the plan")
    if plan.get("draft_dtype") != "torch.float16":
        raise ValueError("graph calibration requires a float16 draft model")
    graph_source = (plan.get("code_sha256") or {}).get("hardware/graph_draft.py")
    if (not isinstance(graph_source, str) or len(graph_source) != 64 or
            any(character not in "0123456789abcdef" for character in graph_source)):
        raise ValueError("frozen plan lacks a valid graph source hash")
    if contract.get("graph_source_sha256") != {"hardware/graph_draft.py": graph_source}:
        raise ValueError("calibration graph source differs from the frozen request source")
    expected_setup = {
        "backend": "cuda_graph_static_cache", "actual_enabled": True,
        "capture_success": True, "fallback_used": False, "batch_size": 1,
        "max_cache_len": capacity, "dtype": "float16", "attention": "sdpa", "capture_count": 1,
    }
    setup = contract.get("graph_setup_contract")
    if not isinstance(setup, dict) or set(setup) != set(expected_setup):
        raise ValueError("calibration graph setup contract is incomplete")
    for field, expected in expected_setup.items():
        if type(setup[field]) is not type(expected) or setup[field] != expected:
            raise ValueError(f"calibration graph setup differs from the required execution: {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--telemetry-interval-ms", type=int, default=200)
    args = parser.parse_args()
    args.offload_folder = args.max_cpu_mem_gb = args.max_gpu_mem_gb = None
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "frozen" or plan.get("schema_version") != 1:
        raise ValueError("a frozen schema-1 algorithm experiment plan is required")
    if plan.get("phase") not in ("development", "evaluation"):
        raise ValueError("phase must distinguish development from evaluation")
    protocol_v3 = plan.get("protocol") == "algorithm_continuous_request_v3_20260921"
    if not protocol_v3 and plan.get("protocol") != "algorithm_continuous_request_20260921":
        raise ValueError("this runner does not implement the historical revision protocol")
    if Path(args.draft_model).name != plan["draft_model_basename"]:
        raise ValueError("draft model differs from the frozen plan")
    methods = plan["methods"]
    expected_methods = METHODS_V2 if plan.get("include_prefix_hazard", False) else METHODS
    if protocol_v3:
        from hardware.run_algorithm_v3 import METHODS_V3, make_v3_controller
        expected_methods = METHODS_V3
    if set(methods) != set(expected_methods) or len(methods) != len(expected_methods):
        raise ValueError("all eight frozen methods must be retained")
    prompt_path = resolve(plan_path, plan["prompts_file"])
    calibration_path = resolve(plan_path, plan["calibration_file"])
    if file_hash(prompt_path) != plan["prompts_sha256"] or file_hash(calibration_path) != plan["calibration_sha256"]:
        raise ValueError("frozen input hash mismatch")
    prompts = [json.loads(line) for line in prompt_path.read_text().splitlines() if line.strip()]
    if plan["phase"] == "evaluation" and any(p["split"] != "closed_loop_evaluation" for p in prompts):
        raise ValueError("evaluation requires the untouched closed-loop prompt split")
    if len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("duplicate prompt IDs")
    calibration = json.loads(calibration_path.read_text())
    validate_calibration_execution(plan, calibration)
    calibration_ids = set(str(value) for value in calibration["calibration_prompt_ids"])
    calibration_texts = set(calibration["calibration_prompt_text_sha256"])
    if calibration_ids.intersection(str(p["prompt_id"]) for p in prompts):
        raise ValueError("request evaluation overlaps calibration prompt IDs")
    if calibration_texts.intersection(hashlib.sha256(p["prompt"].encode()).hexdigest() for p in prompts):
        raise ValueError("request evaluation overlaps calibration prompt text")
    conditions = plan["conditions"]
    if any(float(c.get("extra_one_way_delay_ms", 0)) != 0 for c in conditions):
        raise ValueError("use the common TCP transport; do not add client sleep")
    if len({c["condition_id"] for c in conditions}) != len(conditions):
        raise ValueError("duplicate condition IDs")
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("refusing to overwrite existing hardware results")
    out.mkdir(parents=True, exist_ok=True)
    for source, name in ((plan_path, "frozen_plan.json"), (prompt_path, "prompts_snapshot.jsonl"),
                         (calibration_path, "calibration_snapshot.json")):
        shutil.copyfile(source, out / name)
    sources = [Path(__file__).resolve(), ROOT / "hardware/closed_loop.py", ROOT / "hardware/cloud_server.py",
               ROOT / "hardware/content_features.py", ROOT / "src/content_ratio.py",
               ROOT / "src/structured_bandits.py", ROOT / "src/regularized_ratio.py",
               ROOT / "src/baselines.py", ROOT / "hardware/kv_cache.py",
               ROOT / "hardware/run_tmc_revision.py", ROOT / "hardware/run_revised_experiments.py",
               ROOT / "hardware/run_closed_loop_experiment.py", ROOT / "hardware/delay_proxy.py"]
    if plan.get("execution_revision", 1) >= 2:
        sources += [ROOT / "hardware/target_precision.py", ROOT / "src/prefix_hazard.py"]
    if plan.get("draft_backend", "dynamic") == "cuda_graph":
        sources += [ROOT / "hardware/graph_draft.py"]
    if protocol_v3:
        sources += [ROOT / name for name in (
            "hardware/run_algorithm_v3.py", "src/horizon_ratio.py", "src/core.py",
            "src/config.py", "src/plot_style.py", "src/__init__.py",
        )]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
    if hashes != plan["code_sha256"]:
        raise ValueError("deployed code differs from the frozen plan")
    pings = {}
    for condition in conditions:
        ping = tmc.probe_server(condition["server"])
        tmc.validate_server_capabilities(ping, file_hash(ROOT / "hardware/cloud_server.py"),
                                         require_request_window_telemetry=True, require_session_protocol=True)
        if Path(ping["model"]).name != plan["target_model_basename"] or ping["target_dtype"] != plan["target_dtype"]:
            raise ValueError("target model or precision differs from plan")
        for field, value in plan.get("expected_target_execution", {}).items():
            if ping.get(field) != value:
                raise ValueError(f"target execution differs for {field}")
        pings[condition["condition_id"]] = ping
    manifest = {"completed": False, "protocol": plan["protocol"], "phase": plan["phase"],
                "measurement_session": plan["measurement_session"], "plan_sha256": file_hash(plan_path),
                "prompts_sha256": file_hash(prompt_path), "calibration_sha256": file_hash(calibration_path),
                "client_hostname": socket.gethostname(), "python": platform.python_version(),
                "started_wall_time_ns": time.time_ns(), "code_sha256": hashes, "server_ping": pings,
                "information": "same calibration for all methods; content models frozen; online methods retain state across requests",
                "timing": "completion includes prefill, control, draft, verify, commit; cleanup separately recorded",
                "sampling": "same-day separately started measurement session; not independent network deployments"}
    manifest["execution_revision"] = plan.get("execution_revision", 1)
    manifest["lazy_final_forward"] = bool(plan.get("lazy_final_forward", False))
    _atomic_json(out / "run_manifest.json", manifest)
    tmc.load_draft_model(args)
    actual_dtype = str(next(tmc.hw._model.parameters()).dtype)
    if actual_dtype != plan["draft_dtype"]:
        raise ValueError("draft precision differs from the frozen plan")
    manifest["draft_runtime"] = {"model": args.draft_model, "dtype": actual_dtype,
                                 "device": str(next(tmc.hw._model.parameters()).device)}
    manifest["tokenizer_alignment"] = tmc.configure_draft_termination_for_target(tmc.hw._tok, tmc.hw._model, next(iter(pings.values())))
    graph_engine = None
    if plan.get("draft_backend", "dynamic") == "cuda_graph":
        from hardware.graph_draft import DraftGraphEngine, GraphDraftSession
        graph_engine = DraftGraphEngine(tmc.hw._model, max_cache_len=int(plan.get("graph_max_cache_len", 512)))
    manifest["draft_backend"] = plan.get("draft_backend", "dynamic")
    manifest["graph_setup"] = graph_engine.setup_metrics if graph_engine is not None else None
    _atomic_json(out / "run_manifest.json", manifest)
    factory = make_v3_controller if protocol_v3 else make_controller
    controllers, controller_setup = {}, {}
    for condition in conditions:
        condition_id = condition["condition_id"]
        for method in methods:
            if method == "direct_target":
                continue
            setup_start = time.perf_counter()
            controllers[(condition_id, method)] = factory(method, calibration["conditions"][condition_id])
            controller_setup[condition_id + "::" + method] = (time.perf_counter() - setup_start) * 1000
    manifest["controller_setup_ms"] = controller_setup
    manifest["controller_setup_scope"] = "one-time calibration loading and policy construction; outside request timing"
    _atomic_json(out / "run_manifest.json", manifest)
    rng = np.random.default_rng(int(plan["seed"]))
    blocks = [(c, int(cap), prompt) for c in conditions for cap in plan["generation_token_caps"] for prompt in prompts]
    order = rng.permutation(len(blocks))
    sinks = {name: JsonlSink(out / (name + ".jsonl")) for name in
             ("request_metrics", "round_metrics", "output_equality_audit", "failures")}
    telemetry = tmc.TegrastatsSampler(out / "edge_tegrastats.csv", args.telemetry_interval_ms)
    telemetry.start()
    completed = False
    mismatch_blocks = 0
    try:
        for block_number, block_idx in enumerate(order):
            condition, cap, prompt = blocks[int(block_idx)]
            context = tmc.hw._tok(prompt["prompt"], return_tensors="pt").input_ids[0].tolist()
            condition_id = condition["condition_id"]
            block_key = f"{plan['measurement_session']}|{condition_id}|{cap}|{prompt['prompt_id']}"
            common = {"block_key": block_key, "block_execution_order": block_number,
                      "measurement_session": plan["measurement_session"], "phase": plan["phase"],
                      "condition_id": condition_id, "prompt_id": str(prompt["prompt_id"]),
                      "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
                      "prompt_stratum": prompt["stratum"], "prompt_split": prompt["split"],
                      "prompt_tokens": len(context), "generation_token_cap": cap, "extra_one_way_delay_ms": 0.0}
            results = []
            for method_order, idx in enumerate(rng.permutation(len(methods))):
                method = methods[int(idx)]
                print(f"{plan['measurement_session']} block {block_number + 1}/{len(blocks)} {condition_id} {method}", flush=True)
                identity = {**common, "method": method, "method_execution_order": method_order}
                try:
                    if method == "direct_target":
                        result = run_direct_target_request(condition["server"], context, cap, 0.0)
                        rounds = []
                    else:
                        controller = controllers[(condition_id, method)]
                        begin_request = getattr(controller, "begin_request", None)
                        if callable(begin_request):
                            begin_request()
                        if isinstance(controller, FrozenContentController):
                            controller.decisions.clear()
                        execution_options = ({"lazy_final_forward": True}
                                             if plan.get("lazy_final_forward", False) else {})
                        if graph_engine is not None:
                            def graph_factory(model, tokenizer, context_ids, **kwargs):
                                return GraphDraftSession(graph_engine, tokenizer, context_ids,
                                                         max_new_tokens=cap, **kwargs)
                            execution_options["draft_session_factory"] = graph_factory
                        result, rounds = run_speculative_request(condition["server"], tmc.hw._model,
                                                                 tmc.hw._tok, context, controller, cap, 0.0,
                                                                 **execution_options)
                        if isinstance(controller, FrozenContentController) or protocol_v3 and hasattr(controller, "decisions"):
                            for row, decision in zip(rounds, controller.decisions):
                                row.update(decision)
                    sinks["request_metrics"].write({**result, **identity})
                    for row in rounds:
                        sinks["round_metrics"].write({**row, **identity})
                    # Keep the measured request, but do not continue after a
                    # failed cleanup: a retained target KV session changes the
                    # resource state seen by subsequent methods and blocks.
                    if method != "direct_target" and result.get("target_session_closed") is not True:
                        raise RuntimeError("target session cleanup failed; comparison stopped with raw measurements retained")
                    results.append({**result, **identity})
                except Exception as exc:
                    sinks["failures"].write({**identity, "exception": str(exc), "exception_type": type(exc).__name__})
                    raise
            hashes_equal = len({r["delivered_token_ids_sha256"] for r in results}) == 1
            lengths_equal = len({r["generated_tokens"] for r in results}) == 1
            stops_equal = len({r["stop_reason"] for r in results}) == 1
            mismatch_blocks += int(not (hashes_equal and lengths_equal and stops_equal))
            sinks["output_equality_audit"].write({**common, "methods": methods,
                "hashes_equal": hashes_equal,
                "lengths_equal": lengths_equal,
                "stop_reasons_equal": stops_equal,
                "hash_by_method": {r["method"]: r["delivered_token_ids_sha256"] for r in results}})
            _atomic_json(out / "controller_state.json", {f"{c}::{m}": obj.state_dict() for (c, m), obj in controllers.items()})
        completed = True
    finally:
        telemetry.stop()
        for sink in sinks.values():
            sink.close()
        for name in sinks:
            flatten_jsonl_to_csv(out / (name + ".jsonl"), out / (name + ".csv"))
        manifest.update({"completed": completed, "finished_wall_time_ns": time.time_ns(),
                         "request_rows": sinks["request_metrics"].rows,
                         "round_rows": sinks["round_metrics"].rows,
                         "output_audit_rows": sinks["output_equality_audit"].rows,
                         "failure_rows": sinks["failures"].rows,
                         "output_mismatch_blocks": mismatch_blocks,
                         "valid_for_performance_comparison": completed and mismatch_blocks == 0,
                         "expected_requests": len(blocks) * len(methods), "telemetry_error": telemetry.error})
        manifest["output_sha256"] = {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p.name != "run_manifest.json"}
        _atomic_json(out / "run_manifest.json", manifest)
    print(json.dumps({"completed": completed, "out_dir": str(out), "requests": manifest["request_rows"]}), flush=True)


if __name__ == "__main__":
    main()
