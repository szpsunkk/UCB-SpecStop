"""Estimate exact calibration moments and propose a prospective RTT grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from hardware.run_tmc_revision import sha256_file
except ModuleNotFoundError:  # Direct execution from the hardware directory.
    from run_tmc_revision import sha256_file


def paired_moments(frame: pd.DataFrame, arms: Sequence[int]) -> tuple[Dict[int, float], Dict[int, float]]:
    means = frame.groupby("k_selected").agg(
        mean_n_ms=("total_round_time_ms", "mean"),
        mean_a=("accepted_total", "mean"),
    )
    if set(means.index.astype(int)) != set(arms):
        raise ValueError("calibration data do not contain every configured arm")
    m = {int(arm): float(means.loc[arm, "mean_n_ms"]) for arm in arms}
    b = {int(arm): float(means.loc[arm, "mean_a"]) for arm in arms}
    return m, b


def pair_crossing(m_i: float, b_i: float, m_j: float, b_j: float) -> float:
    return (m_j * b_i - m_i * b_j) / (b_j - b_i)


def lower_envelope_transitions(
    m: Dict[int, float], b: Dict[int, float], arms: Sequence[int]
) -> List[Dict[str, float]]:
    candidates = {0.0}
    for left_index, left in enumerate(arms):
        for right in arms[left_index + 1 :]:
            if b[right] <= b[left]:
                continue
            crossing = pair_crossing(m[left], b[left], m[right], b[right])
            if np.isfinite(crossing) and crossing >= 0:
                candidates.add(float(crossing))
    ordered = sorted(candidates)

    def best_arm(added_rtt_ms: float) -> int:
        costs = {arm: (m[arm] + added_rtt_ms) / b[arm] for arm in arms}
        return min(arms, key=lambda arm: (costs[arm], arm))

    current = best_arm(0.0)
    transitions: List[Dict[str, float]] = []
    for crossing in ordered[1:]:
        epsilon = max(1e-6, abs(crossing) * 1e-9)
        after = best_arm(crossing + epsilon)
        if after == current:
            continue
        exact = pair_crossing(m[current], b[current], m[after], b[after])
        if exact >= 0 and np.isfinite(exact):
            transitions.append(
                {
                    "from_arm": int(current),
                    "to_arm": int(after),
                    "added_rtt_crossing_ms": float(exact),
                }
            )
            current = after
    return transitions


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    arms: Sequence[int],
    resamples: int,
    seed: int,
    yield_increment_tolerance: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = sorted(frame["measurement_session"].astype(str).unique())
    rows = []
    for bootstrap in range(resamples):
        pieces = []
        sampled_sessions = rng.choice(sessions, size=len(sessions), replace=True)
        for sampled_slot, session in enumerate(sampled_sessions):
            session_frame = frame[frame["measurement_session"].astype(str) == session]
            prompt_ids = session_frame["prompt_id"].astype(str).unique()
            sampled_prompts = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
            for prompt_slot, prompt_id in enumerate(sampled_prompts):
                cell = session_frame[
                    session_frame["prompt_id"].astype(str) == str(prompt_id)
                ].copy()
                cell["bootstrap_session_slot"] = sampled_slot
                cell["bootstrap_prompt_slot"] = prompt_slot
                pieces.append(cell)
        sample = pd.concat(pieces, ignore_index=True)
        m, b = paired_moments(sample, arms)
        for left_index, left in enumerate(arms):
            for right in arms[left_index + 1 :]:
                increment = b[right] - b[left]
                if increment <= 0:
                    status = "absent_nonpositive_yield_increment"
                    crossing = None
                elif increment <= yield_increment_tolerance:
                    status = "unstable_near_zero_yield_increment"
                    crossing = pair_crossing(m[left], b[left], m[right], b[right])
                else:
                    crossing = pair_crossing(m[left], b[left], m[right], b[right])
                    status = "finite" if crossing >= 0 else "absent_below_baseline"
                rows.append(
                    {
                        "bootstrap": bootstrap,
                        "left_arm": left,
                        "right_arm": right,
                        "yield_increment": increment,
                        "crossing_status": status,
                        "added_rtt_crossing_ms": crossing,
                    }
                )
    return pd.DataFrame(rows)


def proposed_grid(
    transitions: Iterable[Dict[str, float]], minimum_margin_ms: float
) -> List[float]:
    values = {0.0}
    for transition in transitions:
        crossing = float(transition["added_rtt_crossing_ms"])
        margin = max(minimum_margin_ms, 0.2 * crossing)
        for value in (max(0.0, crossing - margin), crossing, crossing + margin):
            values.add(float(5.0 * round(value / 5.0)))
    return sorted(values)


def finite_quantile(values: Iterable[object], quantile: float) -> float:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return float(numeric.quantile(quantile)) if len(numeric) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--yield-increment-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-grid-margin-ms", type=float, default=25.0)
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if args.yield_increment_tolerance <= 0 or args.minimum_grid_margin_ms <= 0:
        raise ValueError("tolerances must be positive")

    output = args.out_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    frames = []
    manifests = []
    prompt_ids_by_session = {}
    for run_dir_value in args.run_dir:
        run_dir = run_dir_value.expanduser().resolve()
        manifest_path = run_dir / "run_manifest.json"
        data_path = run_dir / "potential_outcomes.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("phase") != "exact_calibration" or not manifest.get("completed"):
            raise ValueError(f"{run_dir} is not a completed exact calibration run")
        if not manifest.get("integrity_audit", {}).get("complete_design"):
            raise ValueError(f"{run_dir} failed its complete-design audit")
        frame = pd.read_csv(data_path)
        if set(frame["planned_added_rtt_ms"].astype(float)) != {0.0}:
            raise ValueError("calibration data must contain only the baseline RTT")
        session = str(manifest["measurement_session"])
        ids = set(frame["prompt_id"].astype(str))
        if any(ids.intersection(previous) for previous in prompt_ids_by_session.values()):
            raise ValueError("calibration sessions reuse prompt IDs")
        prompt_ids_by_session[session] = ids
        frames.append(frame)
        manifests.append(
            {
                "run_dir": str(run_dir),
                "manifest_sha256": sha256_file(manifest_path),
                "data_sha256": sha256_file(data_path),
                "measurement_session": session,
                "server_model": manifest["server_ping"].get("model"),
                "draft_model": manifest.get("draft_model"),
                "live_baseline_http_rtt_ms": manifest.get(
                    "live_baseline_http_rtt_ms"
                ),
            }
        )
    if len(frames) < 2:
        raise ValueError("prospective calibration requires multiple measurement sessions")
    if len({item["server_model"] for item in manifests}) != 1:
        raise ValueError("target model differs across calibration sessions")
    if len({item["draft_model"] for item in manifests}) != 1:
        raise ValueError("draft model differs across calibration sessions")

    frame = pd.concat(frames, ignore_index=True)
    arms = sorted(frame["k_selected"].astype(int).unique())
    cell_counts = frame.groupby(["measurement_session", "prompt_id"])[
        "k_selected"
    ].nunique()
    if not (cell_counts == len(arms)).all():
        raise ValueError("not every calibration prompt contains all arms")
    m, b = paired_moments(frame, arms)
    transitions = lower_envelope_transitions(m, b, arms)
    summary = pd.DataFrame(
        [
            {
                "k": arm,
                "M_k_ms": m[arm],
                "B_k_tokens": b[arm],
                "baseline_cost_ms_per_token": m[arm] / b[arm],
            }
            for arm in arms
        ]
    )
    summary.to_csv(output / "calibration_moments.csv", index=False)
    pd.DataFrame(transitions).to_csv(output / "lower_envelope_transitions.csv", index=False)

    bootstrap = hierarchical_bootstrap(
        frame,
        arms,
        args.bootstrap_resamples,
        args.seed,
        args.yield_increment_tolerance,
    )
    bootstrap.to_csv(output / "bootstrap_pair_crossings.csv", index=False)
    crossing_summary = (
        bootstrap.groupby(["left_arm", "right_arm"])
        .agg(
            resamples=("bootstrap", "count"),
            finite_fraction=(
                "crossing_status",
                lambda values: float(np.mean(np.asarray(values) == "finite")),
            ),
            unstable_fraction=(
                "crossing_status",
                lambda values: float(
                    np.mean(np.asarray(values) == "unstable_near_zero_yield_increment")
                ),
            ),
            absent_fraction=(
                "crossing_status",
                lambda values: float(
                    np.mean([str(value).startswith("absent") for value in values])
                ),
            ),
            finite_crossing_p025_ms=(
                "added_rtt_crossing_ms",
                lambda values: finite_quantile(values, 0.025),
            ),
            finite_crossing_median_ms=(
                "added_rtt_crossing_ms",
                lambda values: finite_quantile(values, 0.5),
            ),
            finite_crossing_p975_ms=(
                "added_rtt_crossing_ms",
                lambda values: finite_quantile(values, 0.975),
            ),
        )
        .reset_index()
    )
    crossing_summary.to_csv(output / "bootstrap_pair_crossing_summary.csv", index=False)

    grid = proposed_grid(transitions, args.minimum_grid_margin_ms)
    proposal = {
        "schema_version": 1,
        "status": "analysis_complete_not_frozen",
        "added_rtt_ms": grid,
        "grid_selection_rule": (
            "Baseline plus each lower-envelope crossing rounded to 5 ms and "
            "points below/above by max(25 ms, 20% of the crossing)."
        ),
        "all_supported_arms_used": True,
        "lower_envelope_transitions": transitions,
        "near_zero_yield_increment_tolerance": args.yield_increment_tolerance,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
    }
    (output / "evaluation_grid_proposal.json").write_text(
        json.dumps(proposal, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "status": "calibration_analysis_complete",
        "inputs": manifests,
        "arms": arms,
        "measurement_sessions": sorted(prompt_ids_by_session),
        "prompt_count": len(set().union(*prompt_ids_by_session.values())),
        "calibration_effective_rtt_ms_mean": float(
            (2.0 * frame["d_eff_one_way_ms"].astype(float)).mean()
        ),
        "calibration_live_ping_rtt_ms_mean": float(
            np.mean(
                [
                    item["live_baseline_http_rtt_ms"]
                    for item in manifests
                    if item["live_baseline_http_rtt_ms"] is not None
                ]
            )
        ),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "yield_increment_tolerance": args.yield_increment_tolerance,
        "minimum_grid_margin_ms": args.minimum_grid_margin_ms,
    }
    for name in (
        "calibration_moments.csv",
        "lower_envelope_transitions.csv",
        "bootstrap_pair_crossings.csv",
        "bootstrap_pair_crossing_summary.csv",
        "evaluation_grid_proposal.json",
    ):
        manifest[f"sha256::{name}"] = sha256_file(output / name)
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
