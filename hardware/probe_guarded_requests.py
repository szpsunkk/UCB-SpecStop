"""Real edge regression for previously observed target arithmetic failures."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hardware.cgp_controller import make_controller
from hardware.closed_loop import ServiceController, run_speculative_request, run_direct_target_request
from hardware.graph_draft import DraftGraphEngine, GraphDraftSession
from hardware.run_cgp_closed_loop import validate_request
from hardware.run_closed_loop_experiment import JsonlSink, _atomic_json
import hardware.run_tmc_revision as tmc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScheduledController(ServiceController):
    def __init__(self, sequence):
        super().__init__("fixed", list(range(1, 11)), fixed_arm=sequence[-1])
        self.sequence = sequence
        self.position = 0

    def select_arm(self):
        value = self.sequence[min(self.position, len(self.sequence)-1)]
        self.position += 1
        return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt-files", nargs="+", type=Path, required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--old-rounds", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    records = {r["prompt_id"]: r for path in args.prompt_files
               for r in map(json.loads, path.read_text().splitlines())}
    prompts = [records[f"gsm8k-test-{i}"] for i in (1307, 258, 1173)]
    old_rounds = list(map(json.loads, args.old_rounds.read_text().splitlines()))
    schedules = {}
    for r in old_rounds:
        if r["prompt_id"] == "gsm8k-test-1307" and r["method"] in ("fixed", "cgp_full", "per_round_refresh"):
            key = r["condition_id"] + "::" + r["method"]
            schedules.setdefault(key, []).append(r["selected_arm"])
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"completed": False, "purpose": "previous-failure runtime regression",
                "formal_evaluation": False, "eligible_for_training": False,
                "algorithm_performance_inference": False, "started_wall_time_ns": time.time_ns(),
                "cap": 32, "schedules": schedules,
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in ROOT.rglob("*.py")},
                "prompts": prompts, "old_rounds_sha256": digest(args.old_rounds)}
    sinks = {name: JsonlSink(out / (name + ".jsonl")) for name in ("requests", "rounds", "audits")}
    mismatches = 0
    try:
        ping = tmc.probe_server("http://127.0.0.1:18080")
        tmc.validate_server_capabilities(ping, digest(ROOT / "hardware/cloud_server.py"),
            require_request_window_telemetry=True, require_session_protocol=True)
        if ping.get("numerical_guard_margin") != .1:
            raise ValueError("guarded runtime is required")
        manifest["ping"] = ping
        tmc.hw.load_model(args.draft_model, allow_download=False)
        manifest["termination"] = tmc.configure_draft_termination_for_target(tmc.hw._tok, tmc.hw._model, ping)
        engine = DraftGraphEngine(tmc.hw._model, max_cache_len=512)
        manifest["graph_setup"] = engine.setup_metrics
        _atomic_json(out / "manifest.json", manifest)
        rng = np.random.default_rng(2026092261)
        for server in ("http://127.0.0.1:18080", "http://127.0.0.1:18081"):
            for prompt in prompts:
                methods = ["k1", "k5", "k9", "k10", "cgp", "cgp_full", "direct_target"]
                if prompt["prompt_id"] == "gsm8k-test-1307":
                    methods += ["schedule::" + name for name in schedules]
                context = tmc.hw._tok(prompt["prompt"], return_tensors="pt").input_ids[0].tolist()
                outputs = {}
                for i in rng.permutation(len(methods)):
                    method = methods[int(i)]
                    def factory(model, tokenizer, context_ids, **kwargs):
                        return GraphDraftSession(engine, tokenizer, context_ids, max_new_tokens=32, **kwargs)
                    if method == "direct_target":
                        result, rounds = run_direct_target_request(server, context, 32), []
                    else:
                        if method.startswith("schedule::"):
                            controller = ScheduledController(schedules[method[len("schedule::"):]])
                        elif method.startswith("k"):
                            controller = ServiceController("fixed", list(range(1, 11)), fixed_arm=int(method[1:]))
                        else:
                            controller = make_controller(method, {})
                        result, rounds = run_speculative_request(server, tmc.hw._model, tmc.hw._tok,
                            context, controller, 32, lazy_final_forward=True, draft_session_factory=factory)
                        validate_request(result, rounds, len(context))
                    identity = {"server": server, "prompt_id": prompt["prompt_id"], "method": method}
                    sinks["requests"].write({**result, **identity})
                    for r in rounds:
                        sinks["rounds"].write({**r, **identity})
                    outputs[method] = (result["delivered_token_ids_sha256"], result["generated_tokens"], result["stop_reason"])
                    print(server, prompt["prompt_id"], method, "complete", flush=True)
                equal = len(set(outputs.values())) == 1
                mismatches += int(not equal)
                sinks["audits"].write({"server": server, "prompt_id": prompt["prompt_id"],
                                      "all_outputs_equal": equal, "outputs": outputs})
        manifest.update(completed=True, mismatch_blocks=mismatches)
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for sink in sinks.values():
            sink.close()
        manifest["finished_wall_time_ns"] = time.time_ns()
        manifest["output_sha256"] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"}
        _atomic_json(out / "manifest.json", manifest)
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
