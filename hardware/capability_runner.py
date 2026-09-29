"""Paired complete-answer experiment using the original Table 4 controllers."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hardware.closed_loop import run_direct_target_request, run_speculative_request
from hardware.edge_only import run_edge_only_request
from hardware.graph_draft import DraftGraphEngine, GraphDraftSession
from hardware.run_cgp_closed_loop import validate_request
from hardware.run_closed_loop_experiment import JsonlSink, _atomic_json
from hardware.run_table4_closed_loop import make_table4_controller
from hardware.runtime_eos import model_eos_ids
import hardware.run_tmc_revision as tmc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--draft-model", required=True, help="Local Qwen2.5-0.5B model directory or cached model ID")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, expected in plan["code_sha256"].items():
        assert digest(ROOT / name) == expected, name
    protocol = args.plan.parent
    cal_path, prompt_path = protocol / "calibration-27.json", protocol / plan["prompts_file"]
    assert digest(cal_path) == plan["calibration_sha256"]
    assert digest(prompt_path) == plan["prompts_sha256"]
    calibration = json.loads(cal_path.read_text())
    prompts = [json.loads(s) for s in prompt_path.read_text().splitlines() if s]
    if not plan["measurement_session"].startswith("pilot"):
        assert not {p["prompt_id"] for p in prompts} & set(calibration["calibration_prompt_ids"])
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    _atomic_json(out / "frozen_plan.json", plan)
    manifest = dict(completed=False, measurement_session=plan["measurement_session"],
                    hostname=socket.gethostname(), python=platform.python_version(),
                    started_wall_ns=time.time_ns(), plan_sha256=digest(args.plan),
                    request_count=0, output_mismatches=0)
    _atomic_json(out / "run_manifest.json", manifest)
    sinks = {key: JsonlSink(out / (key + ".jsonl")) for key in
             ("request_metrics", "round_metrics", "output_equality", "failures")}
    telemetry = tmc.TegrastatsSampler(out / "edge_tegrastats.csv", 200)
    try:
        pings = {}
        for condition in plan["conditions"]:
            ping = tmc.probe_server(condition["server"])
            for field, expected in plan["expected_target_execution"].items():
                assert ping.get(field) == expected, field
            assert ping["server_entrypoint_sha256"] == plan["code_sha256"]["hardware/cloud_server.py"]
            pings[condition["condition_id"]] = ping
        tmc.hw.load_model(args.draft_model, allow_download=False)
        original_eos = model_eos_ids(tmc.hw._model, tmc.hw._tok)
        manifest["termination"] = tmc.configure_draft_termination_for_target(
            tmc.hw._tok, tmc.hw._model, next(iter(pings.values())))
        target_eos = next(iter(pings.values()))["generation_eos_token_ids"]
        tmc.hw._tok.specstop_eos_ids = tuple(target_eos)
        manifest["target_native_eos_ids"] = target_eos
        import torch
        import transformers
        manifest.update(torch=torch.__version__, transformers=transformers.__version__,
                        draft_dtype=str(next(tmc.hw._model.parameters()).dtype), server_pings=pings,
                        native_edge_eos=original_eos, graph_max_cache_len=plan["graph_max_cache_len"])
        engine = DraftGraphEngine(tmc.hw._model, max_cache_len=plan["graph_max_cache_len"])
        manifest["graph_setup"] = engine.setup_metrics
        controllers = {(c["condition_id"], m): make_table4_controller(m, calibration["conditions"][c["condition_id"]])
                       for c in plan["conditions"] for m in c["methods"] if m != "direct_target"}
        # Deterministic, non-evaluation warmup; never updates a learner.
        warm_context = tmc.hw._tok("Calculate 1 + 1. Answer:").input_ids
        run_edge_only_request(engine, warm_context, 8, original_eos)
        run_direct_target_request(plan["conditions"][0]["server"], warm_context, 8)
        _atomic_json(out / "run_manifest.json", manifest)
        telemetry.start()
        rng = random.Random(plan["seed"])
        rng.shuffle(prompts)
        cap = plan["generation_token_cap"]
        total = len(prompts) * (1 + sum(len(c["methods"]) for c in plan["conditions"]))
        for prompt_order, prompt in enumerate(prompts):
            context = tmc.hw._tok(prompt["prompt"]).input_ids
            assert len(context) + cap + 1 <= plan["graph_max_cache_len"]
            tasks = [(None, "edge_only")] + [(c, m) for c in plan["conditions"] for m in c["methods"]]
            rng.shuffle(tasks)
            outputs = {}
            for order, (condition, method) in enumerate(tasks):
                cid = "edge_only" if condition is None else condition["condition_id"]
                identity = dict(measurement_session=plan["measurement_session"], prompt_id=prompt["prompt_id"],
                    prompt_order=prompt_order, method=method, condition_id=cid, method_order=order,
                    block_key=f'{plan["measurement_session"]}|{cid}|{prompt["prompt_id"]}')
                print(json.dumps(dict(event="start", completed=manifest["request_count"], total=total, **identity)), flush=True)
                try:
                    if method == "edge_only":
                        result = run_edge_only_request(engine, context, cap, original_eos, tokenizer=tmc.hw._tok)
                        rounds = []
                    elif method == "direct_target":
                        result = run_direct_target_request(condition["server"], context, cap, answer_stop=True)
                        rounds = []
                    else:
                        controller = controllers[(cid, method)]
                        def factory(model, tokenizer, context_ids, **kwargs):
                            return GraphDraftSession(engine, tokenizer, context_ids, max_new_tokens=cap+1, **kwargs)
                        result, rounds = run_speculative_request(condition["server"], tmc.hw._model,
                            tmc.hw._tok, context, controller, cap, lazy_final_forward=True,
                            draft_session_factory=factory, answer_stop=True)
                        validate_request(result, rounds, len(context))
                    tokens = json.loads(result["delivered_token_ids_json"])
                    assert len(tokens) == result["generated_tokens"]
                    result["output_text"] = tmc.hw._tok.decode(tokens, skip_special_tokens=True)
                    sinks["request_metrics"].write(dict(result, **identity))
                    for row in rounds:
                        sinks["round_metrics"].write(dict(row, **identity))
                    outputs[(cid, method)] = (tokens, result["stop_reason"])
                    manifest["request_count"] += 1
                    _atomic_json(out / "run_manifest.json", manifest)
                    print(json.dumps(dict(event="complete", completed=manifest["request_count"], total=total,
                        tokens=len(tokens), stop_reason=result["stop_reason"], e2e_ms=result["completion_latency_ms"], **identity)), flush=True)
                except BaseException as exc:
                    sinks["failures"].write(dict(error=f"{type(exc).__name__}: {exc}", **identity))
                    raise
            reference = outputs[("lan_direct", "direct_target")]
            equal = all(value == reference for (cid, method), value in outputs.items() if method != "edge_only")
            sinks["output_equality"].write(dict(prompt_id=prompt["prompt_id"], equal=equal,
                checked_methods=len(outputs)-1, edge_matches_target=outputs[("edge_only", "edge_only")] == reference))
            if not equal:
                manifest["output_mismatches"] += 1
                raise RuntimeError("Exact target-output equality failed; preserve records and stop")
            _atomic_json(out / "controller_state.json", {f"{c}:{m}": obj.state_dict() for (c,m),obj in controllers.items()})
        manifest["completed"] = True
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        telemetry.stop()
        for sink in sinks.values():
            sink.close()
        manifest.update(finished_wall_ns=time.time_ns(), telemetry_error=telemetry.error,
                        request_rows=sinks["request_metrics"].rows, round_rows=sinks["round_metrics"].rows,
                        failure_rows=sinks["failures"].rows)
        _atomic_json(out / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
