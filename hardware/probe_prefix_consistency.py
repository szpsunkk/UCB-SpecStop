"""Targeted real remeasurement of cross-length greedy-prefix consistency.

This is a diagnostic, not policy evaluation or a latency benchmark. It keeps
the original outcome tables untouched. Each prompt/repetition fixes context
tokens and tests both independently generated drafts and truncations of one
common maximum-length draft, using the existing /verify endpoint in both cases.
Importing this module or requesting --help never loads a model or contacts a
server. Actual collection occurs only when the CLI is explicitly run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HISTORICAL_PROMPTS_SHA256 = "97756fb891177639d5ba19921df5e72ff4033dd953335eb7cacb274c320db947"
# Controls are the first three Wi-Fi calibration cells, run 0, rounds 0..2.
DEFAULT_PROMPT_IDS = (144, 340, 15, 462, 363)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_difference(left: Sequence[int], right: Sequence[int]) -> Optional[int]:
    """First zero-based difference, including unequal sequence lengths."""

    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _target_ids(response: dict) -> Optional[List[int]]:
    diagnostics = response.get("diagnostics") or {}
    values = diagnostics.get("target_argmax_ids")
    return None if values is None else [int(value) for value in values]


def _validate_response(response: dict, draft: Sequence[int]) -> None:
    if response.get("verification_mode") != "greedy_exact":
        raise ValueError("prefix diagnostics require the greedy_exact /verify protocol")
    accepted = response.get("n_accepted")
    if isinstance(accepted, bool) or not isinstance(accepted, int):
        raise ValueError("verifier n_accepted must be an integer")
    if not 0 <= accepted <= len(draft):
        raise ValueError("verifier accepted length is outside the transmitted draft")
    bonus = response.get("bonus_token_id")
    if isinstance(bonus, bool) or not isinstance(bonus, int) or bonus < 0:
        raise ValueError("greedy /verify must return a nonnegative bonus token ID")


def probe_cell(
    prompt_id: int,
    repetition: int,
    context_ids: Sequence[int],
    kmax: int,
    seed: int,
    draft_fn: Callable[[List[int], int], List[int]],
    verify_fn: Callable[[List[int], List[int], int], dict],
    emit: Callable[[dict], None],
) -> List[dict]:
    """Measure one fixed-context cell; callbacks enable offline protocol tests."""

    context = [int(token) for token in context_ids]
    if not context or any(token < 0 for token in context):
        raise ValueError("context must contain nonnegative token IDs")
    if kmax < 1:
        raise ValueError("kmax must be positive")
    rng = np.random.default_rng(seed)
    common_start = time.perf_counter()
    common_draft = list(draft_fn(context.copy(), kmax))
    common_draft_wall_ms = (time.perf_counter() - common_start) * 1000
    if len(common_draft) > kmax or any(token < 0 for token in common_draft):
        raise ValueError("draft generator returned an invalid maximum-length draft")
    summaries = []
    for mode in ("independent_drafts", "common_draft_prefixes"):
        records = []
        arm_order = rng.permutation(np.arange(1, kmax + 1)).tolist()
        for order, arm in enumerate(arm_order):
            start = time.perf_counter()
            draft = (
                list(draft_fn(context.copy(), arm))
                if mode == "independent_drafts" else common_draft[:arm]
            )
            draft_wall_ms = (time.perf_counter() - start) * 1000
            if len(draft) > arm or any(token < 0 for token in draft):
                raise ValueError("draft generator returned invalid token IDs or length")
            call_start_ns = time.time_ns()
            start = time.perf_counter()
            response = verify_fn(context.copy(), draft.copy(), seed)
            verify_rpc_wall_ms = (time.perf_counter() - start) * 1000
            _validate_response(response, draft)
            accepted = int(response["n_accepted"])
            record = {
                "prompt_id": int(prompt_id),
                "repetition": int(repetition),
                "mode": mode,
                "cell_seed": int(seed),
                "arm_evaluation_order": order,
                "requested_k": int(arm),
                "actual_draft_length": len(draft),
                "draft_stopped_before_requested_length": len(draft) < arm,
                "context_ids": context,
                "draft_ids": draft,
                "common_maximum_draft_ids": common_draft,
                "draft_matches_common_prefix": draft == common_draft[:arm],
                "first_draft_difference_index": first_difference(draft, common_draft[:arm]),
                "accepted_draft_length": accepted,
                "accepted_total": accepted + 1,
                "committed_token_ids": draft[:accepted] + [response["bonus_token_id"]],
                "target_argmax_ids": _target_ids(response),
                "server_diagnostics_present": bool(response.get("diagnostics")),
                "response": response,
                "verify_started_wall_time_ns": call_start_ns,
                "draft_wall_ms": draft_wall_ms,
                "common_draft_generation_wall_ms": common_draft_wall_ms,
                "verify_rpc_wall_ms": verify_rpc_wall_ms,
                "timing_scope": "instrumented diagnostic only; not a performance comparison",
            }
            # Persist each actual response immediately; interrupted runs retain
            # observations without being presented as completed diagnostics.
            emit(record)
            records.append(record)
        largest = next(row for row in records if row["requested_k"] == kmax)
        target_reference = largest["target_argmax_ids"]
        comparisons = []
        for row in sorted(records, key=lambda item: item["requested_k"]):
            expected = min(largest["accepted_total"], row["requested_k"] + 1)
            target = row["target_argmax_ids"]
            target_difference = None
            if target is not None and target_reference is not None:
                target_difference = first_difference(target, target_reference[:len(target)])
            comparisons.append({
                "requested_k": row["requested_k"],
                "actual_draft_length": row["actual_draft_length"],
                "accepted_total": row["accepted_total"],
                "expected_from_mode_maximum": expected,
                "prefix_compatible": row["accepted_total"] == expected,
                "draft_matches_common_prefix": row["draft_matches_common_prefix"],
                "target_comparison_available": target is not None and target_reference is not None,
                "first_target_difference_from_mode_maximum": target_difference,
            })
        summaries.append({
            "prompt_id": int(prompt_id),
            "repetition": int(repetition),
            "mode": mode,
            "cell_seed": int(seed),
            "arm_order": arm_order,
            "prefix_compatible": all(row["prefix_compatible"] for row in comparisons),
            "drafts_share_common_prefix": all(row["draft_matches_common_prefix"] for row in comparisons),
            "all_requested_lengths_generated": all(row["actual_draft_length"] == row["requested_k"] for row in comparisons),
            "comparisons": comparisons,
        })
    return summaries


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_probe(
    prompts: Sequence[str], prompt_ids: Sequence[int], kmax: int, repeats: int,
    seed: int, out_dir: Path, tokenize_fn: Callable[[str], List[int]],
    draft_fn: Callable[[List[int], int], List[int]],
    verify_fn: Callable[[List[int], List[int], int], dict],
    environment: dict,
) -> dict:
    """Write a fresh diagnostic archive, preserving incomplete-run provenance."""

    if kmax < 1 or repeats < 1 or seed < 0:
        raise ValueError("kmax/repeats must be positive and seed nonnegative")
    if not prompt_ids or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt_ids must be a nonempty list without duplicates")
    if any(index < 0 or index >= len(prompts) for index in prompt_ids):
        raise ValueError("prompt_id is outside the zero-based input prompt list")
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("choose an empty output directory; diagnostics never overwrite prior runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    records_path = out_dir / "probe_records.jsonl"
    summary_path = out_dir / "cell_summaries.jsonl"
    manifest = {
        "schema_version": 1,
        "purpose": "targeted_prefix_consistency_diagnostic",
        "formal_evaluation": False,
        "latency_benchmark": False,
        "started_wall_time_ns": time.time_ns(),
        "completed": False,
        "protocol": {
            "prompt_ids": [int(index) for index in prompt_ids],
            "prompt_id_convention": "zero-based position among nonempty prompt records",
            "kmax": kmax, "repeats": repeats, "seed": seed,
            "modes": ["independent_drafts", "common_draft_prefixes"],
            "verification": "existing greedy_exact /verify, return_diagnostics=True",
            "context": "identical token IDs for every arm and both modes within each cell",
            "termination": "actual draft lengths are recorded; early termination is not padded",
        },
        "environment": environment,
        "record_count": 0,
        "summary_count": 0,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _write_manifest(manifest_path, manifest)
    try:
        with records_path.open("x", encoding="utf-8") as records, summary_path.open("x", encoding="utf-8") as summaries:
            def emit(record):
                records.write(json.dumps(record, sort_keys=True) + "\n")
                records.flush()
                manifest["record_count"] += 1

            for prompt_id in prompt_ids:
                context = tokenize_fn(prompts[prompt_id])
                for repetition in range(repeats):
                    cell_seed = seed + int(prompt_id) * repeats + repetition
                    result = probe_cell(
                        int(prompt_id), repetition, context, kmax, cell_seed,
                        draft_fn, verify_fn, emit,
                    )
                    for summary in result:
                        summaries.write(json.dumps(summary, sort_keys=True) + "\n")
                        manifest["summary_count"] += 1
                    summaries.flush()
        manifest["completed"] = True
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_wall_time_ns"] = time.time_ns()
        manifest["output_sha256"] = {
            path.name: sha256_file(path)
            for path in (records_path, summary_path) if path.exists()
        }
        _write_manifest(manifest_path, manifest)
    return manifest


class RequestsVerifier:
    def __init__(self, session, server: str, timeout: float) -> None:
        self.session, self.server, self.timeout = session, server.rstrip("/"), timeout

    def __call__(self, context: List[int], draft: List[int], seed: int) -> dict:
        response = self.session.post(
            self.server + "/verify",
            json={
                "context_ids": context, "draft_ids": draft, "seed": seed,
                "return_diagnostics": True,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompts", type=Path, default=ROOT / "hardware/prompts.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=int, nargs="+", default=list(DEFAULT_PROMPT_IDS))
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--expected-prompts-sha256", default=HISTORICAL_PROMPTS_SHA256)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--align-target-eos", action="store_true")
    parser.add_argument("--draft-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()
    prompt_path = args.prompts.expanduser().resolve()
    prompt_hash = sha256_file(prompt_path)
    if prompt_hash != args.expected_prompts_sha256:
        raise ValueError("prompt SHA256 does not match the explicitly requested source")
    raw_prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [json.loads(line)["prompt"] for line in raw_prompts] if prompt_path.suffix == ".jsonl" else raw_prompts
    if args.k_max < 1 or args.repeats < 1 or args.seed < 0 or args.timeout <= 0:
        raise ValueError("invalid k-max, repeats, seed, or timeout")
    if any(index < 0 or index >= len(prompts) for index in args.prompt_ids):
        raise ValueError("prompt_id is outside the prompt file")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("choose a fresh output directory")

    # Lazy imports: offline tests and --help do not initialize CUDA or models.
    import requests
    import torch
    import transformers
    import hardware.run_tmc_revision as tmc

    hw = tmc.hw
    with requests.Session() as session:
        ping_response = session.get(args.server.rstrip("/") + "/ping", timeout=10)
        ping_response.raise_for_status()
        server_ping = ping_response.json()
        hw.load_model(args.draft_model, allow_download=args.allow_download)
        if args.draft_dtype != "float16":
            hw._model.to(dtype=getattr(torch, args.draft_dtype))
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        vocab_hash = tmc.tokenizer_vocab_sha256(hw._tok)
        target_vocab_hash = server_ping.get("tokenizer_vocab_sha256")
        if target_vocab_hash is not None and vocab_hash != target_vocab_hash:
            raise ValueError("draft and target token-to-ID mappings differ")
        termination = {
            "align_target_eos_requested": args.align_target_eos,
            "draft_tokenizer_eos_token_id": hw._tok.eos_token_id,
            "generation_eos_token_id": hw._model.generation_config.eos_token_id,
            "generation_pad_token_id": hw._model.generation_config.pad_token_id,
            "target_eos_token_id": server_ping.get("tokenizer_eos_token_id"),
        }
        if args.align_target_eos:
            termination["alignment"] = tmc.configure_draft_termination_for_target(hw._tok, hw._model, server_ping)

        def tokenize(prompt):
            return hw._tok(prompt, return_tensors="pt").input_ids[0].tolist()

        def draft(context, k):
            ids = torch.tensor([context], device=hw._device)
            return hw._draft_only(ids, torch.ones_like(ids), k)

        environment = {
            "server": args.server,
            "server_ping": server_ping,
            "draft_model": args.draft_model,
            "draft_dtype": str(next(hw._model.parameters()).dtype),
            "draft_device": str(hw._device),
            "draft_tokenizer_vocab_sha256": vocab_hash,
            "vocabulary_comparison_available": target_vocab_hash is not None,
            "termination": termination,
            "prompts_file": str(prompt_path),
            "prompts_sha256": prompt_hash,
            "matches_historical_prompts": prompt_hash == HISTORICAL_PROMPTS_SHA256,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "source_sha256": {
                str(Path(module.__file__).resolve()): sha256_file(Path(module.__file__).resolve())
                for module in (hw, tmc)
            },
        }
        manifest = collect_probe(
            prompts, args.prompt_ids, args.k_max, args.repeats, args.seed,
            args.out_dir, tokenize, draft,
            RequestsVerifier(session, args.server, args.timeout), environment,
        )
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "completed": manifest["completed"], "records": manifest["record_count"]}))


if __name__ == "__main__":
    main()
