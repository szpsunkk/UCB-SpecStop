"""Reviewer-driven hardware experiments for the TMC major revision.

This entry point adds three evidence paths without overwriting the historical
R1--R6 logs:

``replay-table``
    Measure every draft-length arm on paired prompts while replaying an
    application-level Tailscale RTT trace. The resulting potential-outcome
    table supports independent policy replays and honest dynamic-oracle
    diagnostics.

``request-metrics``
    Run complete fixed-token-cap speculative requests and record TTFT, E2E
    latency, throughput, payload bytes, serialization, memory, and energy.

``concurrency``
    Stress only the cloud verifier with prerecorded draft payloads. This mode
    is intentionally labelled as verifier concurrency rather than full-pipeline
    multi-client generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_revised_experiments as hw  # noqa: E402


TEGRA_FIELDS = (
    "wall_time_ns",
    "monotonic_ns",
    "ram_used_mb",
    "ram_total_mb",
    "swap_used_mb",
    "swap_total_mb",
    "gr3d_percent",
    "vdd_in_mw",
    "raw",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_vocab_sha256(tokenizer: Any) -> str:
    """Hash the token-to-ID mapping independently of tokenizer settings."""

    vocabulary = sorted(
        (str(token), int(token_id))
        for token, token_id in tokenizer.get_vocab().items()
    )
    payload = json.dumps(
        vocabulary, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configure_draft_termination_for_target(
    tokenizer: Any,
    model: Any,
    server_ping: Dict[str, object],
) -> Dict[str, object]:
    """Verify a shared vocabulary and use the target model's EOS semantics."""

    target_vocab_size = server_ping.get("tokenizer_vocab_size")
    target_vocab_sha256 = server_ping.get("tokenizer_vocab_sha256")
    target_eos_token_id = server_ping.get("tokenizer_eos_token_id")
    if not isinstance(target_vocab_size, int):
        raise RuntimeError("cloud verifier did not report a tokenizer vocabulary size")
    if not isinstance(target_vocab_sha256, str) or len(target_vocab_sha256) != 64:
        raise RuntimeError("cloud verifier did not report a tokenizer vocabulary hash")
    if not isinstance(target_eos_token_id, int):
        raise RuntimeError("cloud verifier did not report an integer EOS token ID")
    if len(tokenizer) != target_vocab_size:
        raise RuntimeError("edge and target tokenizer vocabulary sizes differ")

    draft_vocab_sha256 = tokenizer_vocab_sha256(tokenizer)
    if draft_vocab_sha256 != target_vocab_sha256:
        raise RuntimeError("edge and target token-to-ID mappings differ")
    if not 0 <= target_eos_token_id < len(tokenizer):
        raise RuntimeError("target EOS token ID is outside the shared vocabulary")

    configured_eos_token_id = tokenizer.eos_token_id
    configured_eos_token = (
        tokenizer.convert_ids_to_tokens(configured_eos_token_id)
        if configured_eos_token_id is not None
        else None
    )
    target_eos_token = tokenizer.convert_ids_to_tokens(target_eos_token_id)
    if target_eos_token is None:
        raise RuntimeError("target EOS token ID has no edge-tokenizer representation")

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        raise RuntimeError("draft model does not expose a generation configuration")
    configured_generation_eos = generation_config.eos_token_id
    configured_generation_pad = generation_config.pad_token_id
    tokenizer.eos_token_id = target_eos_token_id
    generation_config.eos_token_id = target_eos_token_id
    generation_config.pad_token_id = target_eos_token_id
    if tokenizer.eos_token_id != target_eos_token_id:
        raise RuntimeError("failed to apply target EOS semantics to the edge tokenizer")

    return {
        "vocabulary_match": True,
        "tokenizer_vocab_size": target_vocab_size,
        "tokenizer_vocab_sha256": draft_vocab_sha256,
        "configured_draft_eos_token_id": configured_eos_token_id,
        "configured_draft_eos_token": configured_eos_token,
        "configured_generation_eos_token_id": configured_generation_eos,
        "configured_generation_pad_token_id": configured_generation_pad,
        "effective_target_eos_token_id": target_eos_token_id,
        "effective_target_eos_token": target_eos_token,
        "eos_override_applied": configured_eos_token_id != target_eos_token_id,
    }


def command_output(command: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            list(command), stderr=subprocess.DEVNULL, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def parse_int_list(value: str) -> List[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def prepare_output_dir(path_value: str, allow_nonempty: bool = False) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not allow_nonempty:
        raise FileExistsError(
            f"Refusing to mix a new run with existing artifacts in {path}. "
            "Choose a fresh --out-dir."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def probe_server(server: str) -> Dict[str, object]:
    try:
        return requests.get(f"{server.rstrip('/')}/ping", timeout=10).json()
    except (requests.RequestException, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def validate_server_capabilities(
    server_ping: Dict[str, object],
    expected_entrypoint_sha256: str,
    require_request_window_telemetry: bool = False,
    require_session_protocol: bool = False,
) -> None:
    """Fail before a long run when the cloud protocol is stale or incomplete."""

    if "error" in server_ping:
        raise RuntimeError(f"Cloud verifier probe failed: {server_ping['error']}")
    protocol_version = server_ping.get("protocol_version")
    if not isinstance(protocol_version, int) or protocol_version < 2:
        raise RuntimeError(
            "Cloud verifier protocol_version>=2 is required for revision runs; "
            f"received {protocol_version!r}"
        )
    server_sha = server_ping.get("server_entrypoint_sha256")
    if server_sha != expected_entrypoint_sha256:
        raise RuntimeError(
            "Cloud verifier entrypoint does not match the deployed client copy: "
            f"server={server_sha!r}, expected={expected_entrypoint_sha256!r}"
        )
    verification_modes = server_ping.get("verification_modes", [])
    if "greedy_exact" not in verification_modes:
        raise RuntimeError("Cloud verifier does not advertise greedy_exact mode")
    if require_request_window_telemetry:
        if server_ping.get("request_window_telemetry") is not True:
            raise RuntimeError(
                "Request metrics require request-window cloud telemetry"
            )
        if server_ping.get("nvml_telemetry") is not True:
            raise RuntimeError("Request metrics require available NVML telemetry")
    if require_session_protocol:
        if protocol_version < 3:
            raise RuntimeError("Closed-loop runs require protocol_version>=3")
        for capability in (
            "session_verification",
            "session_no_speculation_arm",
            "direct_target_streaming",
        ):
            if server_ping.get(capability) is not True:
                raise RuntimeError(
                    f"Closed-loop runs require server capability {capability}"
                )


def probe_cloud_telemetry(
    server: str, reset_peak: bool = False
) -> Dict[str, object]:
    endpoint = "telemetry/reset-peak" if reset_peak else "telemetry"
    method = requests.post if reset_peak else requests.get
    try:
        response = method(f"{server.rstrip('/')}/{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def total_energy_delta_j(
    start: Dict[str, object], end: Dict[str, object]
) -> Optional[float]:
    try:
        start_j = float(start["total_energy_j"])
        end_j = float(end["total_energy_j"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(start_j) or not np.isfinite(end_j):
        return None
    if end_j < start_j:
        return None
    return end_j - start_j


def base_manifest(args: argparse.Namespace, mode: str) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parent.parent
    entrypoint = Path(__file__).resolve()
    prompts_path = Path(args.prompts).expanduser().resolve()
    server_ping = probe_server(args.server)
    return {
        "schema_version": 1,
        "mode": mode,
        "started_wall_time_ns": time.time_ns(),
        "client_hostname": socket.gethostname(),
        "client_tailscale_ipv4": command_output(["tailscale", "ip", "-4"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "git_status_short": command_output(
            ["git", "-C", str(repo_root), "status", "--short"]
        ),
        "entrypoint": str(entrypoint),
        "entrypoint_sha256": sha256_file(entrypoint),
        "server": args.server,
        "server_ping": server_ping,
        "draft_model": args.draft_model,
        "prompts": str(prompts_path),
        "prompts_sha256": sha256_file(prompts_path),
        "jetson_power_mode": command_output(["nvpmodel", "-q"]),
        "arguments": vars(args),
    }


class TegrastatsSampler:
    """Continuously record raw and parsed Jetson telemetry."""

    _ram = re.compile(r"RAM\s+(\d+)/(\d+)MB")
    _swap = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
    _gr3d = re.compile(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%")
    _power = re.compile(r"VDD_IN\s+(\d+(?:\.\d+)?)mW")

    def __init__(self, output: Path, interval_ms: int = 100, append: bool = False):
        self.output = output
        self.interval_ms = interval_ms
        self.append = append and output.exists() and output.stat().st_size > 0
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    @staticmethod
    def _match(pattern: re.Pattern, line: str, group: int = 1) -> Optional[float]:
        match = pattern.search(line)
        return float(match.group(group)) if match else None

    def start(self) -> None:
        binary = command_output(["which", "tegrastats"])
        if not binary:
            self.error = "tegrastats not found"
            return
        self.process = subprocess.Popen(
            [binary, "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def _consume(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            mode = "a" if self.append else "w"
            with self.output.open(mode, newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TEGRA_FIELDS)
                if not self.append:
                    writer.writeheader()
                handle.flush()
                for line in self.process.stdout:
                    line = line.strip()
                    ram = self._ram.search(line)
                    swap = self._swap.search(line)
                    writer.writerow(
                        {
                            "wall_time_ns": time.time_ns(),
                            "monotonic_ns": time.monotonic_ns(),
                            "ram_used_mb": float(ram.group(1)) if ram else None,
                            "ram_total_mb": float(ram.group(2)) if ram else None,
                            "swap_used_mb": float(swap.group(1)) if swap else None,
                            "swap_total_mb": float(swap.group(2)) if swap else None,
                            "gr3d_percent": self._match(self._gr3d, line),
                            "vdd_in_mw": self._match(self._power, line),
                            "raw": line,
                        }
                    )
                    handle.flush()
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(timeout=5)


class StreamingCsv:
    def __init__(self, path: Path, append: bool = False):
        self.path = path
        self.append = append and path.exists() and path.stat().st_size > 0
        self.handle = None
        self.writer = None
        self.fieldnames = None
        self.rows = 0
        if self.append:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                self.fieldnames = next(reader, None)
                self.rows = sum(1 for _ in reader)

    def write(self, row: Dict[str, object]) -> None:
        if self.writer is None:
            fieldnames = self.fieldnames or list(row.keys())
            if set(fieldnames) != set(row.keys()):
                raise ValueError(
                    f"CSV schema mismatch for {self.path}: existing={fieldnames}, "
                    f"new={list(row.keys())}"
                )
            mode = "a" if self.append else "w"
            self.handle = self.path.open(mode, newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
            if not self.append:
                self.writer.writeheader()
        self.writer.writerow(row)
        self.handle.flush()
        self.rows += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


def load_prompts(path_value: str) -> List[str]:
    path = Path(path_value).expanduser()
    if path.suffix.lower() == ".jsonl":
        prompts = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on prompt line {line_number}: {exc}"
                ) from exc
            prompt = record.get("prompt") if isinstance(record, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"prompt line {line_number} must contain a non-empty 'prompt' string"
                )
            prompts.append(prompt.strip())
    else:
        prompts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not prompts:
        raise ValueError("prompt file is empty")
    return prompts


def load_trace(path_value: str) -> pd.DataFrame:
    path = Path(path_value).expanduser().resolve()
    trace = pd.read_csv(path)
    required = {"sequence", "rtt_ms", "success"}
    missing = required.difference(trace.columns)
    if missing:
        raise ValueError(f"trace is missing columns: {sorted(missing)}")
    trace = trace[trace["success"].astype(int) == 1].copy()
    trace["rtt_ms"] = pd.to_numeric(trace["rtt_ms"], errors="coerce")
    trace = trace[np.isfinite(trace["rtt_ms"]) & (trace["rtt_ms"] >= 0)]
    if trace.empty:
        raise ValueError("trace has no successful finite RTT samples")
    return trace.reset_index(drop=True)


def balanced_trace_window_starts(
    trace_length: int, n_runs: int, n_rounds: int, seed: int
) -> tuple[List[int], bool]:
    """Choose deterministic, spread-out trace windows for seeded run replicates."""

    if trace_length <= 0 or n_runs <= 0 or n_rounds <= 0:
        raise ValueError("trace length, run count, and round count must be positive")
    if n_rounds > trace_length:
        raise ValueError("a replay run cannot be longer than the successful trace")

    rng = np.random.default_rng(seed + 0x5EED)
    if trace_length >= n_runs * n_rounds:
        starts = []
        for run_index in range(n_runs):
            segment_start = (run_index * trace_length) // n_runs
            segment_end = ((run_index + 1) * trace_length) // n_runs
            latest_start = segment_end - n_rounds
            starts.append(
                int(rng.integers(segment_start, latest_start + 1))
                if latest_start > segment_start
                else segment_start
            )
        return starts, True

    # Short traces cannot provide disjoint windows. Keep the fallback explicit
    # and deterministic so the manifest can disclose the overlap boundary.
    starts = [int(rng.integers(0, trace_length)) for _ in range(n_runs)]
    return starts, False


def load_draft_model(args: argparse.Namespace) -> None:
    hw.load_model(
        args.draft_model,
        allow_download=args.allow_download,
        offload_folder=args.offload_folder,
        max_cpu_mem_gb=args.max_cpu_mem_gb,
        max_gpu_mem_gb=args.max_gpu_mem_gb,
    )


def rejection_sampling_enabled(args: argparse.Namespace) -> bool:
    return args.verification_mode == "acceptance_ratio_proxy"


def run_replay_table(args: argparse.Namespace) -> None:
    output = prepare_output_dir(args.out_dir, allow_nonempty=args.resume)
    trace_path = Path(args.trace).expanduser().resolve()
    trace = load_trace(str(trace_path))
    prompts = load_prompts(args.prompts)
    trace_sha256 = sha256_file(trace_path)
    trace_manifest_path = trace_path.with_suffix(".manifest.json")
    trace_manifest = None
    if trace_manifest_path.exists():
        trace_manifest = json.loads(trace_manifest_path.read_text(encoding="utf-8"))
    manifest_path = output / "run_manifest.json"
    data_path = output / "potential_outcomes.csv"
    completed_keys = set()
    trace_window_starts, trace_windows_disjoint = balanced_trace_window_starts(
        len(trace), args.n_runs, args.n_rounds, args.seed
    )

    if args.resume:
        if not manifest_path.exists() or not data_path.exists():
            raise FileNotFoundError(
                "--resume requires both run_manifest.json and potential_outcomes.csv"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "mode": "paired_real_trace_potential_outcomes",
            "trace_sha256": trace_sha256,
            "server": args.server,
            "draft_model": args.draft_model,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Cannot resume: manifest {key}={manifest.get(key)!r}, "
                    f"requested {value!r}"
                )
        current_server_ping = probe_server(args.server)
        validate_server_capabilities(
            current_server_ping,
            sha256_file(Path(__file__).resolve().parent / "cloud_server.py"),
        )
        previous_server_model = manifest.get("server_ping", {}).get("model")
        if current_server_ping.get("model") != previous_server_model:
            raise ValueError(
                "Cannot resume with a different cloud target model: "
                f"manifest={previous_server_model!r}, "
                f"current={current_server_ping.get('model')!r}"
            )
        previous_args = manifest.get("arguments", {})
        for key in ("seed", "n_runs", "n_rounds", "k_max"):
            if previous_args.get(key) != getattr(args, key):
                raise ValueError(
                    f"Cannot resume: manifest {key}={previous_args.get(key)!r}, "
                    f"requested {getattr(args, key)!r}"
                )
        if manifest.get("trace_window_starts") != trace_window_starts:
            raise ValueError(
                "Cannot resume with a different trace-window allocation: "
                f"manifest={manifest.get('trace_window_starts')!r}, "
                f"requested={trace_window_starts!r}"
            )
        previous = pd.read_csv(
            data_path,
            usecols=["independent_run", "round", "k_selected"],
        )
        if previous.duplicated(["independent_run", "round", "k_selected"]).any():
            raise ValueError("Cannot resume a replay table containing duplicate cells")
        completed_keys = {
            (int(row.independent_run), int(row.round), int(row.k_selected))
            for row in previous.itertuples(index=False)
        }
        manifest.setdefault("resume_events", []).append(
            {
                "wall_time_ns": time.time_ns(),
                "previous_rows": len(completed_keys),
            }
        )
    else:
        manifest = base_manifest(args, "paired_real_trace_potential_outcomes")
        validate_server_capabilities(
            manifest["server_ping"],
            sha256_file(Path(__file__).resolve().parent / "cloud_server.py"),
        )
        manifest.update(
            {
                "trace": str(trace_path),
                "trace_sha256": trace_sha256,
                "trace_manifest": (
                    str(trace_manifest_path) if trace_manifest is not None else None
                ),
                "trace_manifest_sha256": (
                    sha256_file(trace_manifest_path)
                    if trace_manifest is not None
                    else None
                ),
                "trace_kind": (
                    trace_manifest.get("trace_kind")
                    if trace_manifest is not None
                    else "unspecified_rtt"
                ),
                "trace_transport": (
                    trace_manifest.get("transport")
                    if trace_manifest is not None
                    else None
                ),
                "trace_interpretation_boundary": (
                    trace_manifest.get("interpretation_boundary")
                    if trace_manifest is not None
                    else None
                ),
                "successful_trace_samples": len(trace),
                "trace_window_starts": trace_window_starts,
                "trace_window_length": int(args.n_rounds),
                "trace_windows_disjoint": trace_windows_disjoint,
                "trace_window_strategy": (
                    "Seeded start within each of n_runs disjoint contiguous "
                    "trace segments."
                    if trace_windows_disjoint
                    else "Seeded circular windows; source trace is too short "
                    "to guarantee cross-run disjointness."
                ),
                "pairing": (
                    "Every arm uses the same prompt, trace RTT, and verifier seed "
                    "within an independent-run/round cell; arm order is randomized."
                ),
            }
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    load_draft_model(args)
    if args.resume:
        live_baseline_rtt_ms = manifest.get("live_baseline_http_rtt_ms")
        if live_baseline_rtt_ms is None:
            raise ValueError("Cannot resume without live_baseline_http_rtt_ms")
        live_baseline_rtt_ms = float(live_baseline_rtt_ms)
    else:
        live_baseline_rtt_ms = hw.measure_rtt(args.server, n=30)
    manifest["live_baseline_http_rtt_ms"] = live_baseline_rtt_ms
    injection_baseline_rtt_ms = (
        float(args.replay_injection_baseline_rtt_ms)
        if args.replay_injection_baseline_rtt_ms is not None
        else live_baseline_rtt_ms
    )
    manifest["replay_injection_baseline_rtt_ms"] = injection_baseline_rtt_ms
    manifest["replay_injection_baseline_source"] = (
        "explicit_verify-path_pilot"
        if args.replay_injection_baseline_rtt_ms is not None
        else "live_ping_endpoint"
    )

    hw.set_delay(0.0)
    for warmup_k in sorted({1, args.k_max}):
        hw.run_round(
            args.server,
            prompts[0],
            warmup_k,
            run_id=f"warmup_k{warmup_k}",
            prompt_id=0,
            seed=-1,
            strategy="warmup",
            verify_seed=9_000_000 + warmup_k,
            rejection_sampling=rejection_sampling_enabled(args),
        )

    telemetry = TegrastatsSampler(
        output / "edge_tegrastats.csv",
        args.telemetry_interval_ms,
        append=args.resume,
    )
    rows = StreamingCsv(data_path, append=args.resume)
    telemetry.start()
    completed = False
    try:
        for run_index in range(args.n_runs):
            run_seed = args.seed + run_index
            rng = np.random.default_rng(run_seed)
            prompt_order = rng.permutation(len(prompts))
            trace_start = trace_window_starts[run_index]

            for round_index in range(args.n_rounds):
                prompt_id = int(prompt_order[round_index % len(prompt_order)])
                trace_row = trace.iloc[(trace_start + round_index) % len(trace)]
                trace_rtt_ms = float(trace_row["rtt_ms"])
                extra_one_way_ms = max(
                    0.0, (trace_rtt_ms - injection_baseline_rtt_ms) / 2.0
                )
                arm_order = rng.permutation(np.arange(1, args.k_max + 1))
                verify_seed = run_seed * 1_000_000 + round_index
                hw.set_delay(extra_one_way_ms)

                for order_index, k_value in enumerate(arm_order):
                    key = (run_index, round_index, int(k_value))
                    if key in completed_keys:
                        continue
                    record = hw.run_round(
                        args.server,
                        prompts[prompt_id],
                        int(k_value),
                        run_id=f"run{run_index:02d}_round{round_index:04d}",
                        prompt_id=prompt_id,
                        seed=run_seed,
                        strategy="potential_outcome",
                        state="measured_trace_replay",
                        verify_seed=verify_seed,
                        rejection_sampling=rejection_sampling_enabled(args),
                    )
                    record.update(
                        {
                            "independent_run": run_index,
                            "round": round_index,
                            "arm_evaluation_order": order_index,
                            "trace_sequence": int(trace_row["sequence"]),
                            "trace_rtt_ms": trace_rtt_ms,
                            # Retained for compatibility with the first HTTP
                            # replay schema and existing analysis artifacts.
                            "trace_http_rtt_ms": trace_rtt_ms,
                            "live_baseline_http_rtt_ms": live_baseline_rtt_ms,
                            "replay_injection_baseline_rtt_ms": (
                                injection_baseline_rtt_ms
                            ),
                            "replay_extra_one_way_ms": extra_one_way_ms,
                        }
                    )
                    # CSV schemas must not depend on which randomized arm is
                    # written first. Prefixes beyond the selected k were not
                    # drafted, so record them as unavailable rather than as
                    # rejected tokens.
                    for prefix in range(1, args.k_max + 1):
                        record.setdefault(
                            f"prefix_accept_indicator_{prefix}", None
                        )
                    rows.write(record)
        completed = True
    finally:
        telemetry.stop()
        rows.close()
        manifest.update(
            {
                "finished_wall_time_ns": time.time_ns(),
                "completed": completed,
                "potential_outcome_rows": rows.rows,
                "telemetry_error": telemetry.error,
            }
        )
        if data_path.exists():
            manifest["potential_outcomes_sha256"] = sha256_file(data_path)
            audit = pd.read_csv(data_path)
            expected_rows = args.n_runs * args.n_rounds * args.k_max
            keys = ["independent_run", "round", "k_selected"]
            replay_rtt_ms = 2.0 * pd.to_numeric(
                audit["d_eff_one_way_ms"], errors="coerce"
            )
            trace_rtt_column = (
                "trace_rtt_ms"
                if "trace_rtt_ms" in audit.columns
                else "trace_http_rtt_ms"
            )
            target_rtt_ms = pd.to_numeric(audit[trace_rtt_column], errors="coerce")
            planned_rtt_ms = injection_baseline_rtt_ms + 2.0 * pd.to_numeric(
                audit["replay_extra_one_way_ms"], errors="coerce"
            )
            planned_error_ms = planned_rtt_ms - target_rtt_ms
            observed_error_ms = replay_rtt_ms - target_rtt_ms
            manifest["integrity_audit"] = {
                "expected_rows": expected_rows,
                "actual_rows": len(audit),
                "duplicate_cells": int(audit.duplicated(keys).sum()),
                "unique_cells": int(audit[keys].drop_duplicates().shape[0]),
                "complete_grid": bool(
                    len(audit) == expected_rows
                    and not audit.duplicated(keys).any()
                ),
                "verification_modes": sorted(
                    audit["verification_mode"].dropna().astype(str).unique().tolist()
                ),
                "trace_rtt_column": trace_rtt_column,
                "trace_samples_clipped_to_zero_extra_delay_fraction": float(
                    np.mean(target_rtt_ms <= injection_baseline_rtt_ms)
                ),
                "planned_replay_rtt_error_ms_mean": float(planned_error_ms.mean()),
                "planned_replay_rtt_error_ms_p95_abs": float(
                    np.percentile(np.abs(planned_error_ms.dropna()), 95)
                ),
                "observed_replay_rtt_error_ms_mean": float(observed_error_ms.mean()),
                "observed_replay_rtt_error_ms_p95_abs": float(
                    np.percentile(np.abs(observed_error_ms.dropna()), 95)
                ),
            }
        telemetry_path = output / "edge_tegrastats.csv"
        if telemetry_path.exists():
            manifest["edge_tegrastats_sha256"] = sha256_file(telemetry_path)
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def select_stratified_prompt_ids(prompts: Sequence[str], per_stratum: int, seed: int):
    lengths = np.array([len(hw._tok(prompt).input_ids) for prompt in prompts], dtype=int)
    q1, q2 = np.quantile(lengths, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.where(lengths <= q1, "short", np.where(lengths <= q2, "medium", "long"))
    rng = np.random.default_rng(seed)
    selected = []
    for label in ("short", "medium", "long"):
        candidates = np.where(labels == label)[0]
        if len(candidates) == 0:
            continue
        replace = len(candidates) < per_stratum
        for prompt_id in rng.choice(candidates, size=per_stratum, replace=replace):
            selected.append((int(prompt_id), label, int(lengths[prompt_id])))
    return selected, {"short_max": float(q1), "medium_max": float(q2)}


def run_token_request(
    server: str,
    prompt: str,
    k_value: int,
    generation_tokens: int,
    verify_seed: int,
    rejection_sampling: bool,
) -> Dict[str, object]:
    encoded = hw._tok(prompt, return_tensors="pt")
    context_ids = encoded.input_ids.to(hw._device)
    attention = encoded.attention_mask.to(hw._device)
    initial_context_tokens = int(context_ids.shape[1])
    committed: List[int] = []
    round_latencies = []
    total_request_bytes = 0
    total_response_bytes = 0
    total_serialize_ms = 0.0
    total_deserialize_ms = 0.0
    total_cloud_energy_j = 0.0
    cloud_energy_available = True
    cloud_peak_memory_mb = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cloud_start = probe_cloud_telemetry(server, reset_peak=True)
    wall_start_ns = time.time_ns()
    monotonic_start_ns = time.monotonic_ns()
    request_start = time.perf_counter()
    ttft_ms = None
    round_index = 0

    while len(committed) < generation_tokens:
        remaining = generation_tokens - len(committed)
        effective_k = min(k_value, remaining)
        round_start = time.perf_counter()
        if rejection_sampling:
            draft_ids, draft_lps = hw._draft_with_logprobs(
                context_ids, attention, effective_k
            )
        else:
            draft_ids = hw._draft_only(context_ids, attention, effective_k)
            draft_lps = None
        response = hw.verify(
            server,
            context_ids[0].tolist(),
            draft_ids,
            draft_lps,
            seed=verify_seed + round_index,
        )
        accepted = int(response["n_accepted"])
        new_tokens = list(draft_ids[:accepted]) + [int(response["bonus_token_id"])]
        new_tokens = new_tokens[:remaining]
        committed.extend(new_tokens)

        appended = torch.tensor([new_tokens], device=context_ids.device, dtype=context_ids.dtype)
        context_ids = torch.cat([context_ids, appended], dim=1)
        attention = torch.ones_like(context_ids)
        round_latencies.append((time.perf_counter() - round_start) * 1000.0)
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - request_start) * 1000.0

        total_request_bytes += int(response.get("_request_payload_bytes", 0))
        total_response_bytes += int(response.get("_response_payload_bytes", 0))
        total_serialize_ms += float(response.get("_serialize_ms", 0.0))
        total_deserialize_ms += float(response.get("_deserialize_ms", 0.0))
        energy = response.get("cloud_energy_j")
        if energy is None:
            cloud_energy_available = False
        else:
            total_cloud_energy_j += float(energy)
        for key in ("cloud_memory_mb_start", "cloud_memory_mb_end"):
            value = response.get(key)
            if value is not None:
                cloud_peak_memory_mb = max(cloud_peak_memory_mb, float(value))
        cloud_peak_memory_mb = max(
            cloud_peak_memory_mb,
            float(response.get("cloud_process_peak_allocated_mb", 0.0)),
        )
        round_index += 1

    e2e_ms = (time.perf_counter() - request_start) * 1000.0
    monotonic_end_ns = time.monotonic_ns()
    wall_end_ns = time.time_ns()
    cloud_end = probe_cloud_telemetry(server)
    cloud_request_energy_j = total_energy_delta_j(cloud_start, cloud_end)
    cloud_verifier_energy_j = (
        total_cloud_energy_j if cloud_energy_available else None
    )
    if cloud_request_energy_j is not None:
        cloud_energy_j = cloud_request_energy_j
        cloud_energy_measurement = "request_window_nvml_total_energy"
    elif cloud_verifier_energy_j is not None:
        cloud_energy_j = cloud_verifier_energy_j
        cloud_energy_measurement = "summed_verifier_windows_nvml_total_energy"
    else:
        cloud_energy_j = None
        cloud_energy_measurement = "unavailable"
    for snapshot in (cloud_start, cloud_end):
        for key in ("memory_mb", "process_peak_allocated_mb"):
            value = snapshot.get(key)
            if value is not None:
                cloud_peak_memory_mb = max(cloud_peak_memory_mb, float(value))
    edge_peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024.0**2)
        if torch.cuda.is_available()
        else 0.0
    )
    token_bytes = np.asarray(committed, dtype=np.int64).tobytes()
    return {
        "wall_start_ns": wall_start_ns,
        "wall_end_ns": wall_end_ns,
        "monotonic_start_ns": monotonic_start_ns,
        "monotonic_end_ns": monotonic_end_ns,
        "initial_context_tokens": initial_context_tokens,
        "generation_token_cap": generation_tokens,
        "generated_tokens": len(committed),
        "k": k_value,
        "verification_mode": response.get(
            "verification_mode", "legacy_unspecified"
        ),
        "rounds": round_index,
        "ttft_ms": float(ttft_ms),
        "e2e_ms": e2e_ms,
        "throughput_tokens_s": len(committed) / (e2e_ms / 1000.0),
        "round_latency_p50_ms": float(np.percentile(round_latencies, 50)),
        "round_latency_p95_ms": float(np.percentile(round_latencies, 95)),
        "round_latency_p99_ms": float(np.percentile(round_latencies, 99)),
        "request_payload_bytes": total_request_bytes,
        "response_payload_bytes": total_response_bytes,
        "json_serialize_ms": total_serialize_ms,
        "json_deserialize_ms": total_deserialize_ms,
        "edge_process_peak_allocated_mb": edge_peak_memory_mb,
        "cloud_peak_memory_mb": cloud_peak_memory_mb,
        "cloud_energy_j": cloud_energy_j,
        "cloud_energy_per_token_j": (
            cloud_energy_j / len(committed) if cloud_energy_j is not None else None
        ),
        "cloud_energy_measurement": cloud_energy_measurement,
        "cloud_request_window_energy_j": cloud_request_energy_j,
        "cloud_verifier_window_energy_j": cloud_verifier_energy_j,
        "cloud_request_window_power_w_start": cloud_start.get("power_w"),
        "cloud_request_window_power_w_end": cloud_end.get("power_w"),
        "cloud_request_window_memory_mb_start": cloud_start.get("memory_mb"),
        "cloud_request_window_memory_mb_end": cloud_end.get("memory_mb"),
        "generated_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
    }


def run_request_metrics(args: argparse.Namespace) -> None:
    output = prepare_output_dir(args.out_dir, allow_nonempty=args.resume)
    prompts = load_prompts(args.prompts)
    manifest_path = output / "run_manifest.json"
    data_path = output / "request_metrics.csv"
    completed_keys = set()
    if args.resume:
        if not manifest_path.exists() or not data_path.exists():
            raise FileNotFoundError(
                "--resume requires both run_manifest.json and request_metrics.csv"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "mode": "request_level_system_metrics",
            "server": args.server,
            "draft_model": args.draft_model,
            "prompts_sha256": sha256_file(Path(args.prompts).expanduser().resolve()),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Cannot resume: manifest {key}={manifest.get(key)!r}, "
                    f"requested {value!r}"
                )
        current_server_ping = probe_server(args.server)
        previous_server_model = manifest.get("server_ping", {}).get("model")
        if current_server_ping.get("model") != previous_server_model:
            raise ValueError(
                "Cannot resume with a different cloud target model: "
                f"manifest={previous_server_model!r}, "
                f"current={current_server_ping.get('model')!r}"
            )
        previous_args = manifest.get("arguments", {})
        for key in (
            "seed",
            "n_runs",
            "requests_per_stratum",
            "k_values",
            "generation_lengths",
            "verification_mode",
        ):
            if previous_args.get(key) != getattr(args, key):
                raise ValueError(
                    f"Cannot resume: manifest {key}={previous_args.get(key)!r}, "
                    f"requested {getattr(args, key)!r}"
                )
        previous = pd.read_csv(
            data_path,
            usecols=["independent_run", "condition_id"],
        )
        if previous.duplicated(["independent_run", "condition_id"]).any():
            raise ValueError("Cannot resume request metrics containing duplicate conditions")
        completed_keys = {
            (int(row.independent_run), int(row.condition_id))
            for row in previous.itertuples(index=False)
        }
        manifest.setdefault("resume_events", []).append(
            {
                "wall_time_ns": time.time_ns(),
                "previous_rows": len(completed_keys),
            }
        )
    else:
        manifest = base_manifest(args, "request_level_system_metrics")
    current_server_ping = probe_server(args.server)
    expected_server_sha = sha256_file(Path(__file__).resolve().parent / "cloud_server.py")
    validate_server_capabilities(
        current_server_ping,
        expected_server_sha,
        require_request_window_telemetry=True,
    )
    manifest["server_ping"] = current_server_ping
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    load_draft_model(args)
    first_parameter = next(hw._model.parameters())
    device_map = getattr(hw._model, "hf_device_map", None)
    manifest["draft_model_runtime"] = {
        "first_parameter_device": str(first_parameter.device),
        "first_parameter_dtype": str(first_parameter.dtype),
        "hf_device_map": (
            {str(key): str(value) for key, value in device_map.items()}
            if isinstance(device_map, dict)
            else None
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    selected, strata = select_stratified_prompt_ids(
        prompts, args.requests_per_stratum, args.seed
    )
    manifest["prompt_stratum_token_boundaries"] = strata
    manifest["tail_latency_sample_unit"] = "complete speculative generation request"
    manifest["live_baseline_http_rtt_ms"] = hw.measure_rtt(args.server, n=30)
    manifest["network_condition"] = (
        "Live HTTP over Tailscale on the edge device's active Wi-Fi path; "
        "no software delay was added."
    )

    hw.set_delay(0.0)
    run_token_request(
        args.server,
        prompts[selected[0][0]],
        args.k_values[0],
        4,
        8_000_000,
        rejection_sampling_enabled(args),
    )

    telemetry = TegrastatsSampler(
        output / "edge_tegrastats.csv",
        args.telemetry_interval_ms,
        append=args.resume,
    )
    rows = StreamingCsv(data_path, append=args.resume)
    telemetry.start()
    completed = False
    try:
        for independent_run in range(args.n_runs):
            run_seed = args.seed + independent_run
            conditions = [
                (generation_tokens, prompt_position, prompt_info, k_value)
                for generation_tokens in args.generation_lengths
                for prompt_position, prompt_info in enumerate(selected)
                for k_value in args.k_values
            ]
            rng = np.random.default_rng(run_seed)
            for execution_order, condition_index in enumerate(
                rng.permutation(len(conditions))
            ):
                key = (independent_run, int(condition_index))
                if key in completed_keys:
                    continue
                generation_tokens, prompt_position, prompt_info, k_value = conditions[
                    int(condition_index)
                ]
                prompt_id, stratum, prompt_tokens = prompt_info
                result = run_token_request(
                    args.server,
                    prompts[prompt_id],
                    k_value,
                    generation_tokens,
                    verify_seed=(
                        run_seed * 1_000_000
                        + generation_tokens * 1_000
                        + prompt_position * 10
                    ),
                    rejection_sampling=rejection_sampling_enabled(args),
                )
                result.update(
                    {
                        "independent_run": independent_run,
                        "seed": run_seed,
                        "condition_id": int(condition_index),
                        "condition_order": execution_order,
                        "prompt_id": prompt_id,
                        "prompt_stratum": stratum,
                        "prompt_tokens_precomputed": prompt_tokens,
                    }
                )
                rows.write(result)
        completed = True
    finally:
        telemetry.stop()
        rows.close()
        manifest.update(
            {
                "finished_wall_time_ns": time.time_ns(),
                "completed": completed,
                "request_rows": rows.rows,
                "telemetry_error": telemetry.error,
            }
        )
        for name in ("request_metrics.csv", "edge_tegrastats.csv"):
            path = output / name
            if path.exists():
                manifest[f"{path.stem}_sha256"] = sha256_file(path)
        request_path = output / "request_metrics.csv"
        if request_path.exists():
            audit = pd.read_csv(request_path)
            condition_keys = ["independent_run", "condition_id"]
            pair_keys = [
                "independent_run",
                "prompt_id",
                "generation_token_cap",
            ]
            hash_counts = audit.groupby(pair_keys)[
                "generated_token_ids_sha256"
            ].nunique()
            expected_rows = (
                args.n_runs
                * len(selected)
                * len(args.generation_lengths)
                * len(args.k_values)
            )
            manifest["integrity_audit"] = {
                "expected_rows": expected_rows,
                "actual_rows": len(audit),
                "duplicate_conditions": int(
                    audit.duplicated(condition_keys).sum()
                ),
                "complete_design": bool(
                    len(audit) == expected_rows
                    and not audit.duplicated(condition_keys).any()
                ),
                "verification_modes": sorted(
                    audit["verification_mode"].dropna().astype(str).unique().tolist()
                ),
                "greedy_output_hash_equal_across_k": bool((hash_counts == 1).all()),
                "greedy_output_mismatch_groups": int((hash_counts != 1).sum()),
                "cloud_energy_available_fraction": float(
                    audit["cloud_energy_j"].notna().mean()
                ),
                "cloud_energy_measurements": sorted(
                    audit["cloud_energy_measurement"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                ),
            }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def build_payload(
    prompt: str, k_value: int, seed: int, rejection_sampling: bool
) -> Dict[str, object]:
    encoded = hw._tok(prompt, return_tensors="pt")
    ids = encoded.input_ids.to(hw._device)
    attention = encoded.attention_mask.to(hw._device)
    if rejection_sampling:
        draft_ids, draft_lps = hw._draft_with_logprobs(ids, attention, k_value)
    else:
        draft_ids = hw._draft_only(ids, attention, k_value)
        draft_lps = None
    payload = {
        "context_ids": ids[0].tolist(),
        "draft_ids": draft_ids,
        "seed": seed,
    }
    if draft_lps is not None:
        payload["draft_log_probs"] = draft_lps
    return payload


def send_payload(server: str, payload: Dict[str, object], request_id: int):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    start_ns = time.time_ns()
    start = time.perf_counter()
    response = requests.post(
        f"{server.rstrip('/')}/verify",
        data=body,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    end_ns = time.time_ns()
    response.raise_for_status()
    data = response.json()
    return {
        "request_id": request_id,
        "wall_start_ns": start_ns,
        "wall_end_ns": end_ns,
        "latency_ms": latency_ms,
        "request_payload_bytes": len(body),
        "response_payload_bytes": len(response.content),
        "accepted_total": int(data["n_accepted"]) + 1,
        "verification_mode": data.get("verification_mode", "legacy_unspecified"),
        "server_verify_ms": float(data["verify_time_ms"]),
        "cloud_memory_mb_end": data.get("cloud_memory_mb_end"),
        "cloud_process_peak_allocated_mb": data.get(
            "cloud_process_peak_allocated_mb"
        ),
    }


def run_concurrency(args: argparse.Namespace) -> None:
    output = prepare_output_dir(args.out_dir)
    prompts = load_prompts(args.prompts)
    manifest = base_manifest(args, "cloud_verifier_concurrency")
    current_server_ping = probe_server(args.server)
    expected_server_sha = sha256_file(Path(__file__).resolve().parent / "cloud_server.py")
    validate_server_capabilities(current_server_ping, expected_server_sha)
    manifest["server_ping"] = current_server_ping
    manifest["scope_boundary"] = (
        "Draft payloads are prerecorded sequentially on the edge. Reported "
        "concurrency measures the cloud verifier and Tailscale RPC path, not "
        "concurrent edge drafting or a full serving scheduler. Clients are "
        "closed-loop, so no separate open-loop offered load is claimed."
    )
    manifest["live_baseline_http_rtt_ms"] = hw.measure_rtt(args.server, n=30)
    manifest["network_condition"] = (
        "Live HTTP over Tailscale on the edge device's active Wi-Fi path; "
        "no software delay was added."
    )
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    load_draft_model(args)
    rng = np.random.default_rng(args.seed)
    prompt_ids = rng.choice(
        len(prompts), size=args.payload_bank_size, replace=len(prompts) < args.payload_bank_size
    )
    payloads = [
        build_payload(
            prompts[int(prompt_id)],
            args.concurrency_k,
            args.seed + index,
            rejection_sampling_enabled(args),
        )
        for index, prompt_id in enumerate(prompt_ids)
    ]
    send_payload(args.server, payloads[0], -1)

    rows = StreamingCsv(output / "concurrency_requests.csv")
    summary_rows = []
    completed = False
    try:
        for independent_run in range(args.n_runs):
            run_rng = np.random.default_rng(args.seed + independent_run)
            level_order = run_rng.permutation(args.concurrency_levels)
            for condition_order, concurrency_value in enumerate(level_order):
                concurrency = int(concurrency_value)
                suite_start = time.perf_counter()
                results = []
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = {
                        executor.submit(
                            send_payload,
                            args.server,
                            payloads[index % len(payloads)],
                            index,
                        ): index
                        for index in range(args.concurrency_requests)
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        result["independent_run"] = independent_run
                        result["condition_order"] = condition_order
                        result["concurrency"] = concurrency
                        rows.write(result)
                        results.append(result)
                wall_s = time.perf_counter() - suite_start
                latencies = np.array([row["latency_ms"] for row in results])
                accepted = sum(int(row["accepted_total"]) for row in results)
                summary_rows.append(
                    {
                        "independent_run": independent_run,
                        "condition_order": condition_order,
                        "concurrency": concurrency,
                        "requests": len(results),
                        "wall_s": wall_s,
                        "achieved_requests_s": len(results) / wall_s,
                        "accepted_tokens_s": accepted / wall_s,
                        "latency_p50_ms": float(np.percentile(latencies, 50)),
                        "latency_p95_ms": float(np.percentile(latencies, 95)),
                        "latency_p99_ms": float(np.percentile(latencies, 99)),
                    }
                )
        completed = True
    finally:
        rows.close()
        pd.DataFrame(summary_rows).to_csv(output / "concurrency_summary.csv", index=False)
        manifest.update(
            {
                "finished_wall_time_ns": time.time_ns(),
                "completed": completed,
                "request_rows": rows.rows,
            }
        )
        for name in ("concurrency_requests.csv", "concurrency_summary.csv"):
            path = output / name
            if path.exists():
                manifest[f"{path.stem}_sha256"] = sha256_file(path)
        request_path = output / "concurrency_requests.csv"
        if request_path.exists():
            audit = pd.read_csv(request_path)
            expected_rows = (
                args.n_runs
                * len(args.concurrency_levels)
                * args.concurrency_requests
            )
            manifest["integrity_audit"] = {
                "expected_rows": expected_rows,
                "actual_rows": len(audit),
                "duplicate_requests": int(
                    audit.duplicated(
                        ["independent_run", "concurrency", "request_id"]
                    ).sum()
                ),
                "complete_design": bool(
                    len(audit) == expected_rows
                    and not audit.duplicated(
                        ["independent_run", "concurrency", "request_id"]
                    ).any()
                ),
                "verification_modes": sorted(
                    audit["verification_mode"].dropna().astype(str).unique().tolist()
                ),
            }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompts", default="hardware/prompts.txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--max-cpu-mem-gb", type=float, default=None)
    parser.add_argument("--max-gpu-mem-gb", type=float, default=None)
    parser.add_argument("--telemetry-interval-ms", type=int, default=100)
    parser.add_argument(
        "--verification-mode",
        choices=("greedy_exact", "acceptance_ratio_proxy"),
        default="greedy_exact",
        help=(
            "greedy_exact preserves target greedy decoding; acceptance_ratio_proxy "
            "matches the historical selected-token p/q acceptance harness."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    replay = subparsers.add_parser("replay-table")
    add_common_arguments(replay)
    replay.add_argument("--trace", required=True)
    replay.add_argument("--n-runs", type=int, default=10)
    replay.add_argument("--n-rounds", type=int, default=100)
    replay.add_argument("--k-max", type=int, default=10)
    replay.add_argument(
        "--replay-injection-baseline-rtt-ms",
        type=float,
        default=None,
        help=(
            "Optional verify-path RTT baseline used only to compute software "
            "delay injection. The live /ping RTT is still measured and logged."
        ),
    )
    replay.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible interrupted replay table in --out-dir.",
    )

    request_metrics = subparsers.add_parser("request-metrics")
    add_common_arguments(request_metrics)
    request_metrics.add_argument("--n-runs", type=int, default=3)
    request_metrics.add_argument("--requests-per-stratum", type=int, default=3)
    request_metrics.add_argument("--k-values", type=parse_int_list, default=[1, 4, 8])
    request_metrics.add_argument(
        "--generation-lengths", type=parse_int_list, default=[32, 128]
    )
    request_metrics.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible interrupted request metrics in --out-dir.",
    )

    concurrency = subparsers.add_parser("concurrency")
    add_common_arguments(concurrency)
    concurrency.add_argument(
        "--concurrency-levels", type=parse_int_list, default=[1, 2, 4, 8]
    )
    concurrency.add_argument("--n-runs", type=int, default=3)
    concurrency.add_argument("--concurrency-requests", type=int, default=64)
    concurrency.add_argument("--concurrency-k", type=int, default=4)
    concurrency.add_argument("--payload-bank-size", type=int, default=16)

    args = parser.parse_args()
    if args.telemetry_interval_ms <= 0:
        parser.error("--telemetry-interval-ms must be positive")
    if hasattr(args, "n_runs") and args.n_runs <= 0:
        parser.error("--n-runs must be positive")
    if (
        hasattr(args, "replay_injection_baseline_rtt_ms")
        and args.replay_injection_baseline_rtt_ms is not None
        and args.replay_injection_baseline_rtt_ms < 0
    ):
        parser.error("--replay-injection-baseline-rtt-ms must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "replay-table":
        run_replay_table(args)
    elif args.mode == "request-metrics":
        run_request_metrics(args)
    elif args.mode == "concurrency":
        run_concurrency(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
