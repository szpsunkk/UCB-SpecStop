"""Run frozen continuous-request controller experiments on a Jetson edge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hardware.closed_loop import (  # noqa: E402
    ServiceController,
    run_direct_target_request,
    run_speculative_request,
)
import hardware.run_tmc_revision as tmc  # noqa: E402

TegrastatsSampler = tmc.TegrastatsSampler
command_output = tmc.command_output
load_draft_model = tmc.load_draft_model
probe_server = tmc.probe_server
sha256_file = tmc.sha256_file
validate_server_capabilities = tmc.validate_server_capabilities
hw = tmc.hw


SUPPORTED_METHODS = (
    "sn",
    "lcb",
    "per_round_ucb",
    "ucbspec_throughput",
    "fixed",
    "direct_target",
)


class JsonlSink:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("x", encoding="utf-8")
        self.rows = 0

    def write(self, row: Dict[str, object]) -> None:
        self.handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self.handle.flush()
        self.rows += 1

    def close(self) -> None:
        self.handle.close()


def _atomic_json(path: Path, value: Dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_plan_path(plan_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (plan_path.parent / path).resolve()


def load_prompt_records(path: Path, measurement_session: str, phase: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid prompt JSONL line {line_number}: {exc}") from exc
        required = {"prompt_id", "prompt", "stratum", "split", "measurement_session"}
        missing = required.difference(record)
        if missing:
            raise ValueError(f"prompt line {line_number} is missing {sorted(missing)}")
        if not isinstance(record["prompt"], str) or not record["prompt"].strip():
            raise ValueError(f"prompt line {line_number} has an empty prompt")
        if record["stratum"] not in {"short", "medium", "long"}:
            raise ValueError(f"prompt line {line_number} has an invalid stratum")
        if str(record["measurement_session"]) != measurement_session:
            raise ValueError("prompt measurement_session does not match the plan")
        if phase == "evaluation" and record["split"] != "closed_loop_evaluation":
            raise ValueError("evaluation plans require closed_loop_evaluation prompts")
        records.append(record)
    if not records:
        raise ValueError("prompt manifest is empty")
    prompt_ids = [str(record["prompt_id"]) for record in records]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt IDs must be unique within a measurement session")
    strata = Counter(str(record["stratum"]) for record in records)
    if set(strata) != {"short", "medium", "long"}:
        raise ValueError("every prompt-length stratum must be represented")
    return records


def validate_plan(plan: Dict[str, object], plan_path: Path) -> tuple[Path, List[Dict[str, object]]]:
    if plan.get("schema_version") != 1:
        raise ValueError("closed-loop plan schema_version must be 1")
    if plan.get("status") != "frozen":
        raise ValueError("the experiment plan must be frozen before execution")
    phase = str(plan.get("phase"))
    if phase not in {"smoke", "pilot", "evaluation"}:
        raise ValueError("plan phase must be smoke, pilot, or evaluation")
    measurement_session = str(plan.get("measurement_session", ""))
    if not measurement_session:
        raise ValueError("measurement_session is required")
    prompt_path = _resolve_plan_path(plan_path, str(plan["prompts_file"]))
    records = load_prompt_records(prompt_path, measurement_session, phase)

    methods = [str(value) for value in plan.get("methods", [])]
    if len(set(methods)) != len(methods) or any(
        value not in SUPPORTED_METHODS for value in methods
    ):
        raise ValueError(f"methods must be unique members of {SUPPORTED_METHODS}")
    if phase == "evaluation" and set(methods) != set(SUPPORTED_METHODS):
        raise ValueError("evaluation plans must include all six frozen methods")

    speculative_arms = [int(value) for value in plan.get("speculative_arms", [])]
    if (
        not speculative_arms
        or any(value <= 0 for value in speculative_arms)
        or len(set(speculative_arms)) != len(speculative_arms)
    ):
        raise ValueError("speculative_arms must be unique positive integers")
    ucbspec_arms = [int(value) for value in plan.get("ucbspec_arms", [])]
    if 0 not in ucbspec_arms or any(value < 0 for value in ucbspec_arms):
        raise ValueError("ucbspec_arms must include the no-speculation arm 0")

    caps = [int(value) for value in plan.get("generation_token_caps", [])]
    if not caps or any(value <= 0 for value in caps):
        raise ValueError("generation_token_caps must be positive")
    conditions = plan.get("network_conditions", [])
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("network_conditions must be a non-empty list")
    condition_ids = []
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        condition_ids.append(condition_id)
        if float(condition["extra_one_way_delay_ms"]) < 0:
            raise ValueError("extra one-way delay must be nonnegative")
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("network condition IDs must be unique")

    if phase == "evaluation":
        expected_prompts = int(plan.get("prompts_per_session", 12))
        if len(records) != expected_prompts:
            raise ValueError(
                f"evaluation plan requires {expected_prompts} prompts, found {len(records)}"
            )
        if sorted(caps) != [32, 128]:
            raise ValueError("evaluation generation caps must be exactly 32 and 128")
        if len(conditions) != 2:
            raise ValueError("evaluation plan requires exactly two network conditions")
        if not plan.get("calibration_manifest_sha256"):
            raise ValueError("evaluation plan requires frozen calibration provenance")
        if not plan.get("expected_target_model_basename"):
            raise ValueError("evaluation plan must freeze the target model basename")
        if not plan.get("expected_target_dtype"):
            raise ValueError("evaluation plan must freeze the target precision")
        fixed_by_condition = plan.get("fixed_arm_by_condition", {})
        lcb_by_condition = plan.get("controller_parameters", {}).get(
            "lcb_l_max_ms_per_token_by_condition", {}
        )
        if set(fixed_by_condition) != set(condition_ids):
            raise ValueError("evaluation fixed arms must be frozen for every condition")
        if set(lcb_by_condition) != set(condition_ids):
            raise ValueError("evaluation LCB scales must be frozen for every condition")
        stratum_counts = Counter(str(record["stratum"]) for record in records)
        if len(set(stratum_counts.values())) != 1:
            raise ValueError("evaluation prompts must be balanced across length strata")
    return prompt_path, records


def make_controller(
    method: str,
    plan: Dict[str, object],
    condition_id: str,
) -> ServiceController:
    params = dict(plan.get("controller_parameters", {}))
    speculative_arms = tuple(int(value) for value in plan["speculative_arms"])
    if method == "ucbspec_throughput":
        arms = tuple(int(value) for value in plan["ucbspec_arms"])
    elif method == "fixed":
        arms = tuple(sorted(set(speculative_arms) | {0}))
    else:
        arms = speculative_arms
    fixed_by_condition = plan.get("fixed_arm_by_condition", {})
    lcb_by_condition = params.get("lcb_l_max_ms_per_token_by_condition", {})
    return ServiceController(
        method=method,
        actual_arms=arms,
        beta=float(params.get(f"{method}_beta", params.get("beta", 1.0))),
        horizon=int(params.get("horizon", 10000)),
        lcb_l_max_ms_per_token=float(lcb_by_condition.get(condition_id, 1000.0)),
        ucbspec_exploration_weight=float(
            params.get("ucbspec_exploration_weight", 6.0)
        ),
        fixed_arm=(
            int(fixed_by_condition[condition_id]) if method == "fixed" else None
        ),
    )


def flatten_jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> None:
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return
    pd.read_json(jsonl_path, lines=True).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--max-cpu-mem-gb", type=float, default=None)
    parser.add_argument("--max-gpu-mem-gb", type=float, default=None)
    parser.add_argument("--telemetry-interval-ms", type=int, default=100)
    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prompt_path, prompts = validate_plan(plan, plan_path)
    output = args.out_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan_path, output / "frozen_plan.json")
    shutil.copy2(prompt_path, output / "prompts_snapshot.jsonl")

    server_ping = probe_server(args.server)
    cloud_entrypoint = REPO_ROOT / "hardware" / "cloud_server.py"
    validate_server_capabilities(
        server_ping,
        sha256_file(cloud_entrypoint),
        require_request_window_telemetry=True,
        require_session_protocol=True,
    )
    if plan["phase"] == "evaluation":
        if Path(str(server_ping.get("model"))).name != str(
            plan["expected_target_model_basename"]
        ):
            raise RuntimeError("deployed target model does not match the frozen plan")
        if str(server_ping.get("target_dtype")) != str(plan["expected_target_dtype"]):
            raise RuntimeError("deployed target precision does not match the frozen plan")
    manifest: Dict[str, object] = {
        "schema_version": 1,
        "mode": "continuous_request_online_controller",
        "started_wall_time_ns": time.time_ns(),
        "completed": False,
        "client_hostname": socket.gethostname(),
        "client_tailscale_ipv4": command_output(["tailscale", "ip", "-4"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "-C", str(REPO_ROOT), "status", "--short"]),
        "server": args.server,
        "server_ping": server_ping,
        "draft_model": args.draft_model,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "prompts": str(prompt_path),
        "prompts_sha256": sha256_file(prompt_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "closed_loop_library_sha256": sha256_file(REPO_ROOT / "hardware" / "closed_loop.py"),
        "cloud_entrypoint_sha256": sha256_file(cloud_entrypoint),
        "kv_cache_sha256": sha256_file(REPO_ROOT / "hardware" / "kv_cache.py"),
        "measurement_session": plan["measurement_session"],
        "phase": plan["phase"],
        "feedback_scope": "selected arm only",
        "controller_lifecycle": "persistent within method and network condition across requests",
        "request_state": "fresh prompt KV per request; continuous KV within request",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _atomic_json(output / "run_manifest.json", manifest)

    load_draft_model(args)
    tokenizer_alignment = tmc.configure_draft_termination_for_target(
        hw._tok, hw._model, server_ping
    )
    manifest["tokenizer_alignment"] = tokenizer_alignment
    manifest["draft_runtime"] = {
        "tokenizer_vocab_size": len(hw._tok),
        "effective_tokenizer_eos_token_id": hw._tok.eos_token_id,
        "first_parameter_device": str(next(hw._model.parameters()).device),
        "first_parameter_dtype": str(next(hw._model.parameters()).dtype),
    }
    _atomic_json(output / "run_manifest.json", manifest)

    methods = [str(value) for value in plan["methods"]]
    conditions = list(plan["network_conditions"])
    controllers = {
        (str(condition["condition_id"]), method): make_controller(
            method, plan, str(condition["condition_id"])
        )
        for condition in conditions
        for method in methods
        if method != "direct_target"
    }
    blocks = [
        (condition, int(cap), prompt)
        for condition in conditions
        for cap in plan["generation_token_caps"]
        for prompt in prompts
    ]
    rng = np.random.default_rng(int(plan["randomization_seed"]))
    block_order = rng.permutation(len(blocks))

    request_sink = JsonlSink(output / "request_metrics.jsonl")
    round_sink = JsonlSink(output / "round_metrics.jsonl")
    audit_sink = JsonlSink(output / "output_equality_audit.jsonl")
    failure_sink = JsonlSink(output / "failures.jsonl")
    telemetry = TegrastatsSampler(
        output / "edge_tegrastats.csv", int(args.telemetry_interval_ms)
    )
    telemetry.start()
    completed = False
    try:
        for execution_order, block_index in enumerate(block_order):
            condition, generation_cap, prompt = blocks[int(block_index)]
            condition_id = str(condition["condition_id"])
            one_way_delay_ms = float(condition["extra_one_way_delay_ms"])
            tokenize_start = time.perf_counter()
            encoded = hw._tok(str(prompt["prompt"]), return_tensors="pt")
            context_ids = encoded.input_ids[0].tolist()
            tokenization_ms = (time.perf_counter() - tokenize_start) * 1000.0
            block_key = (
                f"{plan['measurement_session']}|{condition_id}|"
                f"{generation_cap}|{prompt['prompt_id']}"
            )
            method_order = [methods[int(index)] for index in rng.permutation(len(methods))]
            block_results = []
            for method_order_index, method in enumerate(method_order):
                common = {
                    "block_key": block_key,
                    "block_execution_order": execution_order,
                    "method_execution_order": method_order_index,
                    "measurement_session": plan["measurement_session"],
                    "phase": plan["phase"],
                    "condition_id": condition_id,
                    "extra_one_way_delay_ms": one_way_delay_ms,
                    "prompt_id": str(prompt["prompt_id"]),
                    "prompt_sha256": hashlib.sha256(
                        str(prompt["prompt"]).encode("utf-8")
                    ).hexdigest(),
                    "prompt_stratum": str(prompt["stratum"]),
                    "prompt_split": str(prompt["split"]),
                    "prompt_tokens": len(context_ids),
                    "prompt_tokenization_ms": tokenization_ms,
                    "retry_count": 0,
                }
                try:
                    if method == "direct_target":
                        request = run_direct_target_request(
                            args.server,
                            context_ids,
                            generation_cap,
                            one_way_delay_ms,
                        )
                        rounds: Sequence[Dict[str, object]] = []
                    else:
                        request, rounds = run_speculative_request(
                            args.server,
                            hw._model,
                            hw._tok,
                            context_ids,
                            controllers[(condition_id, method)],
                            generation_cap,
                            one_way_delay_ms,
                        )
                    request.update(common)
                    request["method"] = method
                    request_sink.write(request)
                    for round_row in rounds:
                        round_row.update(common)
                        round_row["method"] = method
                        round_sink.write(round_row)
                    block_results.append(request)
                    _atomic_json(
                        output / "controller_state.json",
                        {
                            f"{key[0]}::{key[1]}": controller.state_dict()
                            for key, controller in controllers.items()
                        },
                    )
                except Exception as exc:
                    failure_sink.write(
                        {
                            **common,
                            "method": method,
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                            "wall_time_ns": time.time_ns(),
                        }
                    )
                    raise

            hashes = [str(row["delivered_token_ids_sha256"]) for row in block_results]
            lengths = [int(row["generated_tokens"]) for row in block_results]
            stops = [str(row["stop_reason"]) for row in block_results]
            audit_sink.write(
                {
                    "block_key": block_key,
                    "measurement_session": plan["measurement_session"],
                    "condition_id": condition_id,
                    "generation_token_cap": generation_cap,
                    "prompt_id": str(prompt["prompt_id"]),
                    "methods": methods,
                    "hashes_equal": len(set(hashes)) == 1,
                    "lengths_equal": len(set(lengths)) == 1,
                    "stop_reasons_equal": len(set(stops)) == 1,
                    "hash_by_method": {
                        str(row["method"]): row["delivered_token_ids_sha256"]
                        for row in block_results
                    },
                }
            )
        completed = True
    finally:
        telemetry.stop()
        for sink in (request_sink, round_sink, audit_sink, failure_sink):
            sink.close()
        for stem in (
            "request_metrics",
            "round_metrics",
            "output_equality_audit",
            "failures",
        ):
            flatten_jsonl_to_csv(output / f"{stem}.jsonl", output / f"{stem}.csv")
        manifest.update(
            {
                "finished_wall_time_ns": time.time_ns(),
                "completed": completed,
                "request_rows": request_sink.rows,
                "round_rows": round_sink.rows,
                "output_audit_rows": audit_sink.rows,
                "failure_rows": failure_sink.rows,
                "telemetry_error": telemetry.error,
            }
        )
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "run_manifest.json":
                manifest[f"sha256::{path.name}"] = sha256_file(path)
        audit_csv = output / "output_equality_audit.csv"
        if audit_csv.exists() and audit_csv.stat().st_size:
            audit = pd.read_csv(audit_csv)
            manifest["integrity_audit"] = {
                "expected_requests": len(blocks) * len(methods),
                "actual_requests": request_sink.rows,
                "expected_blocks": len(blocks),
                "actual_blocks": audit_sink.rows,
                "all_output_hashes_equal": bool(audit["hashes_equal"].all()),
                "output_mismatch_blocks": int((~audit["hashes_equal"]).sum()),
                "complete_design": bool(
                    completed
                    and request_sink.rows == len(blocks) * len(methods)
                    and audit_sink.rows == len(blocks)
                    and failure_sink.rows == 0
                ),
            }
        _atomic_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
