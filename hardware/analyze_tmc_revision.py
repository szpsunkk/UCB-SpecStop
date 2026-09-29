"""Analyze reviewer-driven TMC hardware artifacts without fabricating results."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baselines import (  # noqa: E402
    BanditSpecUCB,
    BanditSpecThroughputUCB,
    DisplayedRatioLCB,
    EXP3Ratio,
    EXP3Spec,
    PerRoundRatioUCB,
    UCBSpecStop,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ci95_t(values: Sequence[float]) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(array.mean())
    if len(array) == 1:
        return mean, float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))
    return mean, mean - half, mean + half


def ratio_of_sums(frame: pd.DataFrame) -> float:
    return float(frame["total_round_time_ms"].sum() / frame["accepted_total"].sum())


def longest_true_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_rtt_trace(frame: pd.DataFrame) -> Dict[str, object]:
    required = {"sequence", "utc_start", "rtt_ms", "success"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"RTT trace is missing {sorted(missing)}")

    sequence = pd.to_numeric(frame["sequence"], errors="coerce")
    success = pd.to_numeric(frame["success"], errors="coerce") == 1
    rtt = pd.to_numeric(frame["rtt_ms"], errors="coerce")
    finite_rtt = np.isfinite(rtt) & (rtt >= 0)
    valid = success & finite_rtt
    successful_rtt = rtt[valid].reset_index(drop=True)
    if successful_rtt.empty:
        raise ValueError("RTT trace contains no successful finite samples")

    finite_sequence = sequence.dropna().astype(int)
    sequence_span = (
        int(finite_sequence.max() - finite_sequence.min() + 1)
        if len(finite_sequence)
        else 0
    )
    timestamps = pd.to_datetime(frame["utc_start"], utc=True, errors="coerce")
    timestamp_ordered = bool(timestamps.dropna().is_monotonic_increasing)
    elapsed_s = (
        float((timestamps.max() - timestamps.min()).total_seconds())
        if timestamps.notna().sum() >= 2
        else np.nan
    )
    row: Dict[str, object] = {
        "rows": int(len(frame)),
        "successful_rows": int(success.sum()),
        "failed_rows": int((~success).sum()),
        "valid_successful_rtt_rows": int(valid.sum()),
        "success_fraction": float(success.mean()),
        "success_rows_without_valid_rtt": int((success & ~finite_rtt).sum()),
        "failed_rows_with_valid_rtt": int((~success & finite_rtt).sum()),
        "duplicate_sequence_rows": int(sequence.duplicated().sum()),
        "missing_sequence_rows": max(0, sequence_span - finite_sequence.nunique()),
        "sequence_monotone": bool(sequence.dropna().is_monotonic_increasing),
        "timestamp_monotone": timestamp_ordered,
        "sample_span_s": elapsed_s,
        "longest_consecutive_failure_rows": longest_true_run((~success).tolist()),
        "rtt_mean_ms": float(successful_rtt.mean()),
        "rtt_std_ms": float(successful_rtt.std(ddof=1)),
        "rtt_min_ms": float(successful_rtt.min()),
        "rtt_p50_ms": float(np.percentile(successful_rtt, 50)),
        "rtt_p95_ms": float(np.percentile(successful_rtt, 95)),
        "rtt_p99_ms": float(np.percentile(successful_rtt, 99)),
        "rtt_max_ms": float(successful_rtt.max()),
        "rtt_lag1_autocorrelation": (
            float(successful_rtt.autocorr(lag=1))
            if len(successful_rtt) >= 3 and successful_rtt.std(ddof=1) > 0
            else np.nan
        ),
    }
    for source in ("schedule_lag_ms", "probe_wall_ms"):
        if source not in frame.columns:
            continue
        values = pd.to_numeric(frame[source], errors="coerce").dropna()
        row[f"{source}_mean"] = float(values.mean())
        row[f"{source}_p95"] = float(np.percentile(values, 95))
        row[f"{source}_max"] = float(values.max())
    return row


def arm_run_metrics(
    frame: pd.DataFrame, group_keys: Sequence[str]
) -> pd.DataFrame:
    """Aggregate arm measurements within each independent run."""

    mean_columns = {
        "total_round_time_ms": "mean_round_time_ms",
        "draft_time_ms": "mean_draft_time_ms",
        "server_verify_split_ms": "mean_server_verify_ms",
        "d_eff_one_way_ms": "mean_d_eff_one_way_ms",
        "accepted_total": "mean_accepted_total",
        "request_payload_bytes": "mean_request_payload_bytes",
        "response_payload_bytes": "mean_response_payload_bytes",
        "json_serialize_ms": "mean_json_serialize_ms",
        "json_deserialize_ms": "mean_json_deserialize_ms",
        "prompt_tokens": "mean_prompt_tokens",
    }
    rows = []
    for key_values, group in frame.groupby(list(group_keys), dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(group_keys, key_values))
        row["rounds"] = len(group)
        row["cost_ratio_of_sums_ms_per_token"] = ratio_of_sums(group)
        for source, target in mean_columns.items():
            if source in group.columns:
                row[target] = float(pd.to_numeric(group[source], errors="coerce").mean())
        if "cloud_energy_j" in group.columns:
            energy = pd.to_numeric(group["cloud_energy_j"], errors="coerce")
            row["cloud_energy_j_per_accepted_token"] = (
                float(energy.sum() / group["accepted_total"].sum())
                if energy.notna().all()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_run_metrics(
    run_metrics: pd.DataFrame, group_keys: Sequence[str]
) -> pd.DataFrame:
    """Report run-level means and t intervals without treating rounds as iid."""

    excluded = set(group_keys) | {"independent_run", "rounds"}
    metric_columns = [
        column
        for column in run_metrics.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(run_metrics[column])
    ]
    rows = []
    for key_values, group in run_metrics.groupby(list(group_keys), dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(group_keys, key_values))
        row["independent_runs"] = int(group["independent_run"].nunique())
        row["rounds_total"] = int(group["rounds"].sum())
        for metric in metric_columns:
            mean, low, high = ci95_t(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def validate_potential_outcomes(frame: pd.DataFrame) -> List[int]:
    required = {
        "independent_run",
        "round",
        "k_selected",
        "total_round_time_ms",
        "accepted_total",
        "trace_http_rtt_ms",
        "verification_mode",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"potential-outcome table is missing {sorted(missing)}")
    arms = sorted(frame["k_selected"].astype(int).unique().tolist())
    counts = frame.groupby(["independent_run", "round"])["k_selected"].nunique()
    if not (counts == len(arms)).all():
        bad = counts[counts != len(arms)]
        raise ValueError(f"incomplete arm grid in {len(bad)} run/round cells")
    if frame.duplicated(["independent_run", "round", "k_selected"]).any():
        raise ValueError("duplicate potential outcomes for the same run/round/arm")
    verification_modes = sorted(
        frame["verification_mode"].dropna().astype(str).unique().tolist()
    )
    if len(verification_modes) != 1:
        raise ValueError(
            "potential-outcome analysis requires exactly one verification mode; "
            f"found {verification_modes}"
        )
    return arms


def apply_fixed_rtt_offset(frame: pd.DataFrame, rtt_ms: float) -> pd.DataFrame:
    """Apply an explicitly counterfactual common RTT shift to paired outcomes."""

    if rtt_ms < 0:
        raise ValueError("fixed RTT offset must be nonnegative")
    shifted = frame.copy()
    if rtt_ms == 0:
        return shifted
    shifted["total_round_time_ms"] = pd.to_numeric(
        shifted["total_round_time_ms"], errors="raise"
    ) + float(rtt_ms)
    if "trace_http_rtt_ms" in shifted.columns:
        shifted["trace_http_rtt_ms"] = pd.to_numeric(
            shifted["trace_http_rtt_ms"], errors="raise"
        ) + float(rtt_ms)
    if "d_eff_one_way_ms" in shifted.columns:
        shifted["d_eff_one_way_ms"] = pd.to_numeric(
            shifted["d_eff_one_way_ms"], errors="raise"
        ) + float(rtt_ms) / 2.0
    return shifted


def arm_population_costs(frame: pd.DataFrame, arms: Sequence[int]) -> Dict[int, float]:
    return {
        int(k): ratio_of_sums(frame[frame["k_selected"] == k]) for k in arms
    }


def adjacent_arm_crossings(
    frame: pd.DataFrame, arms: Sequence[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate added common-RTT crossings for adjacent arms by run."""

    rows = []
    for run_id, run_frame in frame.groupby("independent_run", sort=True):
        for left, right in zip(arms[:-1], arms[1:]):
            left_frame = run_frame[run_frame["k_selected"] == left]
            right_frame = run_frame[run_frame["k_selected"] == right]
            if len(left_frame) != len(right_frame) or left_frame.empty:
                raise ValueError(
                    f"unpaired crossing cells for run={run_id}, arms={left},{right}"
                )
            left_n = float(left_frame["total_round_time_ms"].sum())
            right_n = float(right_frame["total_round_time_ms"].sum())
            left_a = float(left_frame["accepted_total"].sum())
            right_a = float(right_frame["accepted_total"].sum())
            rounds = len(left_frame)
            denominator = rounds * (right_a - left_a)
            crossing = (
                (right_n * left_a - left_n * right_a) / denominator
                if denominator != 0
                else np.nan
            )
            rows.append(
                {
                    "independent_run": int(run_id),
                    "left_k": int(left),
                    "right_k": int(right),
                    "paired_rounds": int(rounds),
                    "added_rtt_crossing_ms": crossing,
                }
            )
    per_run = pd.DataFrame(rows)
    summary_rows = []
    for (left, right), group in per_run.groupby(["left_k", "right_k"], sort=True):
        mean, low, high = ci95_t(group["added_rtt_crossing_ms"])
        summary_rows.append(
            {
                "left_k": int(left),
                "right_k": int(right),
                "independent_runs": int(group["independent_run"].nunique()),
                "added_rtt_crossing_ms_mean": mean,
                "added_rtt_crossing_ms_ci95_low": low,
                "added_rtt_crossing_ms_ci95_high": high,
            }
        )
    return per_run, pd.DataFrame(summary_rows)


def prefix_acceptance_metrics(
    frame: pd.DataFrame, max_arm: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize prefix survival once per paired cell using the largest arm."""

    source = frame[frame["k_selected"] == max_arm].copy()
    rows = []
    for run_id, group in source.groupby("independent_run", sort=True):
        previous_survival = 1.0
        for prefix in range(1, max_arm + 1):
            column = f"prefix_accept_indicator_{prefix}"
            if column not in group.columns:
                raise ValueError(f"missing {column} for prefix acceptance audit")
            survival = float(pd.to_numeric(group[column], errors="raise").mean())
            conditional = (
                survival / previous_survival if previous_survival > 0 else np.nan
            )
            rows.append(
                {
                    "independent_run": int(run_id),
                    "prefix_position": prefix,
                    "paired_cells": len(group),
                    "prefix_survival_probability": survival,
                    "conditional_acceptance_probability": conditional,
                }
            )
            previous_survival = survival
    per_run = pd.DataFrame(rows)
    summary_rows = []
    for prefix, group in per_run.groupby("prefix_position", sort=True):
        row = {
            "prefix_position": int(prefix),
            "independent_runs": int(group["independent_run"].nunique()),
            "paired_cells_total": int(group["paired_cells"].sum()),
        }
        for metric in (
            "prefix_survival_probability",
            "conditional_acceptance_probability",
        ):
            mean, low, high = ci95_t(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    return per_run, pd.DataFrame(summary_rows)


def greedy_prefix_nesting_audit(
    frame: pd.DataFrame, max_arm: int
) -> Dict[str, object]:
    """Audit finite-precision cross-k prefix consistency without altering data."""

    cells = ["independent_run", "round"]
    largest = frame[frame["k_selected"] == max_arm].set_index(cells)[
        "accepted_draft_len"
    ]
    cell_index = pd.MultiIndex.from_frame(frame[cells])
    reference = pd.to_numeric(largest.reindex(cell_index), errors="coerce").to_numpy()
    selected = pd.to_numeric(frame["k_selected"], errors="coerce").to_numpy()
    observed = pd.to_numeric(
        frame["accepted_draft_len"], errors="coerce"
    ).to_numpy()
    expected = np.minimum(selected, reference)
    discordant = ~np.isclose(observed, expected, rtol=0.0, atol=0.0)
    discordant_cells = frame.loc[discordant, cells].drop_duplicates()
    total_cells = int(frame[cells].drop_duplicates().shape[0])
    return {
        "total_run_round_cells": total_cells,
        "discordant_run_round_cells": int(len(discordant_cells)),
        "discordant_cell_fraction": float(len(discordant_cells) / total_cells),
        "definition": (
            "For each run-round cell, compare every arm's accepted draft length "
            "with min(k, L_at_Kmax). Discordance is retained as finite-precision "
            "measurement behavior; per-arm outcomes are never overwritten."
        ),
    }


def state_edges(values: Sequence[float], states: int) -> np.ndarray:
    edges = np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, states + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    for index in range(1, len(edges) - 1):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], np.inf)
    return edges


def assign_states(values: Sequence[float], edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], np.asarray(values, dtype=float), side="right")


def state_aware_arm_map(
    calibration: pd.DataFrame, arms: Sequence[int], edges: np.ndarray
) -> Dict[int, int]:
    data = calibration.copy()
    data["network_state"] = assign_states(data["trace_http_rtt_ms"], edges)
    mapping = {}
    for state, group in data.groupby("network_state"):
        costs = arm_population_costs(group, arms)
        mapping[int(state)] = min(costs, key=costs.get)
    fallback = min(arm_population_costs(calibration, arms), key=arm_population_costs(calibration, arms).get)
    for state in range(len(edges) - 1):
        mapping.setdefault(state, int(fallback))
    return mapping


def state_arm_population_costs(
    calibration: pd.DataFrame, arms: Sequence[int], edges: np.ndarray
) -> Dict[Tuple[int, int], float]:
    """Estimate each state-arm ratio from calibration rows only."""

    data = calibration.copy()
    data["network_state"] = assign_states(data["trace_http_rtt_ms"], edges)
    costs: Dict[Tuple[int, int], float] = {}
    for state in range(len(edges) - 1):
        for arm in arms:
            group = data[
                (data["network_state"] == state)
                & (data["k_selected"].astype(int) == int(arm))
            ]
            if group.empty:
                raise ValueError(
                    f"no calibration observations for state={state}, arm={arm}"
                )
            costs[(state, int(arm))] = ratio_of_sums(group)
    return costs


def aggregate_contextual_oracle(
    calibration: pd.DataFrame, arms: Sequence[int], edges: np.ndarray
) -> Tuple[Dict[int, int], float, pd.DataFrame]:
    """Optimize the empirical aggregate ratio over state-to-arm mappings.

    Per-state ratio minimization is generally not equivalent to minimizing the
    ratio after aggregating numerator and denominator across states.  The
    Dinkelbach transform separates over states, so the product policy class can
    be optimized exactly without enumerating ``len(arms) ** n_states`` maps.
    """

    data = calibration.copy()
    data["network_state"] = assign_states(data["trace_http_rtt_ms"], edges)
    context_rows = data.drop_duplicates(["independent_run", "round"])
    occupancy = (
        context_rows["network_state"]
        .value_counts(normalize=True)
        .reindex(range(len(edges) - 1), fill_value=0.0)
        .sort_index()
    )

    moment_rows = []
    for state in range(len(edges) - 1):
        for arm in arms:
            group = data[
                (data["network_state"] == state)
                & (data["k_selected"].astype(int) == int(arm))
            ]
            if group.empty:
                raise ValueError(
                    f"no calibration observations for state={state}, arm={arm}"
                )
            moment_rows.append(
                {
                    "network_state": state,
                    "k": int(arm),
                    "state_occupancy": float(occupancy.loc[state]),
                    "mean_round_time_ms": float(group["total_round_time_ms"].mean()),
                    "mean_accepted_total": float(group["accepted_total"].mean()),
                }
            )
    moments = pd.DataFrame(moment_rows)

    def minimize_transformed(lam: float):
        mapping: Dict[int, int] = {}
        transformed = 0.0
        for state, group in moments.groupby("network_state", sort=True):
            score = group["mean_round_time_ms"] - lam * group["mean_accepted_total"]
            selected_index = score.idxmin()
            selected = group.loc[selected_index]
            mapping[int(state)] = int(selected["k"])
            transformed += float(selected["state_occupancy"] * score.loc[selected_index])
        return transformed, mapping

    low = 0.0
    high = float(
        (moments["mean_round_time_ms"] / moments["mean_accepted_total"]).max()
    )
    if minimize_transformed(low)[0] < 0 or minimize_transformed(high)[0] > 0:
        raise RuntimeError("could not bracket aggregate contextual oracle ratio")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if minimize_transformed(middle)[0] > 0:
            low = middle
        else:
            high = middle
    _, mapping = minimize_transformed(0.5 * (low + high))

    selected = moments[
        moments.apply(
            lambda row: int(row["k"]) == mapping[int(row["network_state"])],
            axis=1,
        )
    ]
    numerator = float(
        (selected["state_occupancy"] * selected["mean_round_time_ms"]).sum()
    )
    denominator = float(
        (selected["state_occupancy"] * selected["mean_accepted_total"]).sum()
    )
    cost = numerator / denominator
    moments["aggregate_oracle_selected"] = moments.apply(
        lambda row: int(row["k"]) == mapping[int(row["network_state"])], axis=1
    )
    return mapping, cost, moments


def make_policy(
    name: str,
    k_max: int,
    beta: float,
    seed: int,
    horizon: int,
    displayed_l_max: float,
    exp3_cost_scale: float,
):
    rng = np.random.default_rng(seed)
    if name == "ucb_specstop_displayed_lcb":
        algorithm = DisplayedRatioLCB(
            k_max=k_max,
            horizon=horizon,
            l_max=displayed_l_max,
            beta=beta,
        )
        select = lambda t: algorithm.select_arm(t)
        update = lambda k, row: algorithm.update(
            k, float(row.total_round_time_ms), float(row.accepted_total)
        )
    elif name == "ucb_specstop_self_normalized":
        algorithm = UCBSpecStop(k_max=k_max, beta=beta)
        select = lambda t: algorithm.select_arm(t)
        update = lambda k, row: algorithm.update(
            k, float(row.total_round_time_ms), float(row.accepted_total)
        )
    elif name == "per_round_ratio_ucb":
        algorithm = PerRoundRatioUCB(k_max=k_max, beta=beta)
        select = lambda t: algorithm.select_arm(t)
        update = lambda k, row: algorithm.update(
            k, float(row.total_round_time_ms), float(row.accepted_total)
        )
    elif name == "banditspec_ucb":
        algorithm = BanditSpecUCB(k_max=k_max, delta=0.05)
        select = lambda t: algorithm.select_arm(t)
        update = lambda k, row: algorithm.update(k, float(row.accepted_total))
    elif name == "banditspec_throughput_public":
        algorithm = BanditSpecThroughputUCB(
            k_max=k_max,
            exploration_weight=6.0,
            common_bonus=True,
        )
        select = lambda t: algorithm.select_arm(t)
        update = lambda k, row: algorithm.update(
            k,
            float(row.accepted_total),
            float(row.total_round_time_ms),
            time_scale_to_seconds=1e-3,
        )
    elif name == "exp3spec":
        algorithm = EXP3Spec(k_max=k_max)
        select = lambda t: algorithm.select_arm(t, rng)
        update = lambda k, row: algorithm.update(k, float(row.accepted_total))
    elif name == "exp3_ratio":
        algorithm = EXP3Ratio(k_max=k_max)
        select = lambda t: algorithm.select_arm(t, rng)
        update = lambda k, row: algorithm.update(
            k,
            float(row.total_round_time_ms),
            float(row.accepted_total),
            cost_scale=exp3_cost_scale,
        )
    else:
        raise ValueError(name)
    return select, update


def make_contextual_policy(
    name: str,
    k_max: int,
    context_count: int,
    beta: float,
    horizon: int,
    displayed_l_max: float,
):
    """Create one independent online learner per observable context state."""

    algorithms = []
    for _ in range(context_count):
        if name == "contextual_ucb_specstop_displayed_lcb":
            algorithm = DisplayedRatioLCB(
                k_max=k_max,
                horizon=horizon,
                l_max=displayed_l_max,
                beta=beta,
                context_count=context_count,
            )
        elif name == "contextual_ucb_specstop_self_normalized":
            algorithm = UCBSpecStop(k_max=k_max, beta=beta)
        elif name == "contextual_per_round_ratio_ucb":
            algorithm = PerRoundRatioUCB(k_max=k_max, beta=beta)
        elif name == "contextual_banditspec_ucb":
            algorithm = BanditSpecUCB(k_max=k_max, delta=0.05)
        else:
            raise ValueError(name)
        algorithms.append(algorithm)

    state_visits = np.zeros(context_count, dtype=int)

    def select(state: int) -> int:
        state_visits[state] += 1
        return int(algorithms[state].select_arm(int(state_visits[state])))

    def update(state: int, arm: int, numerator: float, denominator: float) -> None:
        algorithm = algorithms[state]
        if name == "contextual_banditspec_ucb":
            algorithm.update(arm, denominator)
        else:
            algorithm.update(arm, numerator, denominator)

    return select, update


def analyze_policies(args: argparse.Namespace) -> None:
    if args.n_max_guard_factor < 1.0:
        raise ValueError("--n-max-guard-factor must be at least 1")
    if args.added_rtt_ms < 0:
        raise ValueError("--added-rtt-ms must be nonnegative")
    input_path = Path(args.potential_outcomes).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = apply_fixed_rtt_offset(pd.read_csv(input_path), args.added_rtt_ms)
    arms = validate_potential_outcomes(frame)
    runs = [int(value) for value in sorted(frame["independent_run"].astype(int).unique())]
    if len(runs) < 3:
        raise ValueError("at least three independent runs are required")
    calibration_count = max(1, int(math.floor(len(runs) * args.calibration_fraction)))
    if calibration_count >= len(runs):
        calibration_count = len(runs) - 1
    calibration_runs = runs[:calibration_count]
    evaluation_runs = runs[calibration_count:]
    calibration = frame[frame["independent_run"].isin(calibration_runs)].copy()
    evaluation = frame[frame["independent_run"].isin(evaluation_runs)].copy()

    per_arm_run = arm_run_metrics(frame, ["independent_run", "k_selected"])
    per_arm_summary = summarize_run_metrics(per_arm_run, ["k_selected"])
    crossing_run, crossing_summary = adjacent_arm_crossings(frame, arms)
    prefix_run, prefix_summary = prefix_acceptance_metrics(frame, max(arms))
    context_boundaries = None
    per_context_run = pd.DataFrame()
    per_context_summary = pd.DataFrame()
    context_crossing_run = pd.DataFrame()
    context_crossing_summary = pd.DataFrame()
    if "prompt_tokens" in frame.columns:
        context_rows = frame.drop_duplicates(["independent_run", "round"])
        q1, q2 = np.quantile(
            pd.to_numeric(context_rows["prompt_tokens"], errors="coerce"),
            [1.0 / 3.0, 2.0 / 3.0],
        )
        context_boundaries = {"short_max_tokens": float(q1), "medium_max_tokens": float(q2)}
        frame["prompt_stratum"] = np.where(
            frame["prompt_tokens"] <= q1,
            "short",
            np.where(frame["prompt_tokens"] <= q2, "medium", "long"),
        )
        per_context_run = arm_run_metrics(
            frame, ["independent_run", "prompt_stratum", "k_selected"]
        )
        per_context_summary = summarize_run_metrics(
            per_context_run, ["prompt_stratum", "k_selected"]
        )
        context_crossing_runs = []
        context_crossing_summaries = []
        for stratum, group in frame.groupby("prompt_stratum", sort=True):
            stratum_run, stratum_summary = adjacent_arm_crossings(group, arms)
            stratum_run.insert(1, "prompt_stratum", stratum)
            stratum_summary.insert(0, "prompt_stratum", stratum)
            context_crossing_runs.append(stratum_run)
            context_crossing_summaries.append(stratum_summary)
        context_crossing_run = pd.concat(context_crossing_runs, ignore_index=True)
        context_crossing_summary = pd.concat(
            context_crossing_summaries, ignore_index=True
        )

    payload_slope_rows = []
    if "mean_request_payload_bytes" in per_arm_run.columns:
        for run_id, group in per_arm_run.groupby("independent_run"):
            slope, intercept = np.polyfit(
                group["k_selected"].to_numpy(dtype=float),
                group["mean_request_payload_bytes"].to_numpy(dtype=float),
                deg=1,
            )
            payload_slope_rows.append(
                {
                    "independent_run": int(run_id),
                    "payload_bytes_per_additional_draft_token": float(slope),
                    "payload_intercept_bytes": float(intercept),
                }
            )
    payload_scaling = pd.DataFrame(payload_slope_rows)

    population_costs = arm_population_costs(calibration, arms)
    fixed_oracle_arm = min(population_costs, key=population_costs.get)
    fixed_oracle_cost = population_costs[fixed_oracle_arm]
    edges = state_edges(calibration["trace_http_rtt_ms"], args.context_states)
    per_state_ratio_map = state_aware_arm_map(calibration, arms, edges)
    aggregate_contextual_map, aggregate_contextual_cost, contextual_cells = (
        aggregate_contextual_oracle(calibration, arms, edges)
    )
    contextual_cells["per_state_ratio_selected"] = contextual_cells.apply(
        lambda row: int(row["k"])
        == per_state_ratio_map[int(row["network_state"])],
        axis=1,
    )
    calibration_n_max = float(calibration["total_round_time_ms"].max())
    displayed_n_max = calibration_n_max * args.n_max_guard_factor
    displayed_l_max = displayed_n_max * (1.0 + max(arms) + 1.0)
    evaluation_n_max = float(evaluation["total_round_time_ms"].max())

    policy_names = [
        "ucb_specstop_displayed_lcb",
        "ucb_specstop_self_normalized",
        "per_round_ratio_ucb",
        "banditspec_throughput_public",
        "banditspec_ucb",
        "exp3spec",
        "exp3_ratio",
        "independent_fixed_oracle",
        "calibrated_aggregate_contextual_oracle",
        "calibrated_per_state_ratio_oracle",
        "per_round_hindsight_diagnostic",
    ]
    trajectory_rows = []
    run_rows = []

    for run_id in evaluation_runs:
        run_frame = evaluation[evaluation["independent_run"] == run_id]
        round_ids = sorted(run_frame["round"].astype(int).unique())
        lookup = {
            (int(row.round), int(row.k_selected)): row
            for row in run_frame.itertuples(index=False)
        }

        for policy_index, policy in enumerate(policy_names):
            if policy in {
                "ucb_specstop_displayed_lcb",
                "ucb_specstop_self_normalized",
                "per_round_ratio_ucb",
                "banditspec_throughput_public",
                "banditspec_ucb",
                "exp3spec",
                "exp3_ratio",
            }:
                select, update = make_policy(
                    policy,
                    max(arms),
                    args.beta,
                    args.seed + run_id * 100 + policy_index,
                    len(round_ids),
                    displayed_l_max,
                    displayed_n_max,
                )
            else:
                select = update = None

            selected_rows = []
            cumulative_pseudo_regret = 0.0
            for time_index, round_id in enumerate(round_ids, start=1):
                candidates = [lookup[(round_id, arm)] for arm in arms]
                trace_rtt = float(candidates[0].trace_http_rtt_ms)
                if policy == "independent_fixed_oracle":
                    arm = int(fixed_oracle_arm)
                elif policy == "calibrated_aggregate_contextual_oracle":
                    state = int(assign_states([trace_rtt], edges)[0])
                    arm = int(aggregate_contextual_map[state])
                elif policy == "calibrated_per_state_ratio_oracle":
                    state = int(assign_states([trace_rtt], edges)[0])
                    arm = int(per_state_ratio_map[state])
                elif policy == "per_round_hindsight_diagnostic":
                    arm = int(
                        min(
                            candidates,
                            key=lambda row: float(row.total_round_time_ms)
                            / float(row.accepted_total),
                        ).k_selected
                    )
                else:
                    arm = int(select(time_index))
                row = lookup[(round_id, arm)]
                if update is not None:
                    update(arm, row)
                cumulative_pseudo_regret += population_costs[arm] - fixed_oracle_cost
                selected_rows.append(row)
                trajectory_rows.append(
                    {
                        "independent_run": run_id,
                        "policy": policy,
                        "round": round_id,
                        "k_selected": arm,
                        "trace_http_rtt_ms": trace_rtt,
                        "total_round_time_ms": float(row.total_round_time_ms),
                        "accepted_total": float(row.accepted_total),
                        "realized_round_cost_ms_per_token": (
                            float(row.total_round_time_ms) / float(row.accepted_total)
                        ),
                        "population_arm_cost_ms_per_token": population_costs[arm],
                        "pseudo_regret_increment": population_costs[arm]
                        - fixed_oracle_cost,
                        "cumulative_pseudo_regret": cumulative_pseudo_regret,
                    }
                )

            chosen = pd.DataFrame(
                {
                    "total_round_time_ms": [float(row.total_round_time_ms) for row in selected_rows],
                    "accepted_total": [float(row.accepted_total) for row in selected_rows],
                    "round_cost": [
                        float(row.total_round_time_ms) / float(row.accepted_total)
                        for row in selected_rows
                    ],
                    "k": [int(row.k_selected) for row in selected_rows],
                    "cloud_energy_j": [
                        getattr(row, "cloud_energy_j", np.nan) for row in selected_rows
                    ],
                }
            )
            realized_cost = ratio_of_sums(chosen)
            run_rows.append(
                {
                    "independent_run": run_id,
                    "policy": policy,
                    "n_rounds": len(chosen),
                    "cost_ratio_of_sums_ms_per_token": realized_cost,
                    "theorem_aligned_pseudo_regret": cumulative_pseudo_regret,
                    "pseudo_regret_per_round": cumulative_pseudo_regret / len(chosen),
                    "throughput_tokens_s": chosen["accepted_total"].sum()
                    / (chosen["total_round_time_ms"].sum() / 1000.0),
                    "round_cost_p50_ms_per_token": float(np.percentile(chosen["round_cost"], 50)),
                    "round_cost_p95_ms_per_token": float(np.percentile(chosen["round_cost"], 95)),
                    "round_cost_p99_ms_per_token": float(np.percentile(chosen["round_cost"], 99)),
                    "mean_selected_k": float(chosen["k"].mean()),
                    "cloud_energy_j_per_accepted_token": (
                        float(chosen["cloud_energy_j"].sum() / chosen["accepted_total"].sum())
                        if chosen["cloud_energy_j"].notna().all()
                        else np.nan
                    ),
                }
            )

    run_results = pd.DataFrame(run_rows)
    paired_fixed_cost = (
        run_results[run_results["policy"] == "independent_fixed_oracle"]
        .set_index("independent_run")["cost_ratio_of_sums_ms_per_token"]
    )
    run_results["paired_fixed_oracle_cost_ms_per_token"] = run_results[
        "independent_run"
    ].map(paired_fixed_cost)
    run_results["realized_excess_vs_paired_fixed_oracle_pct"] = 100.0 * (
        run_results["cost_ratio_of_sums_ms_per_token"]
        - run_results["paired_fixed_oracle_cost_ms_per_token"]
    ) / run_results["paired_fixed_oracle_cost_ms_per_token"]
    run_results["paired_difference_ms_per_token"] = (
        run_results["cost_ratio_of_sums_ms_per_token"]
        - run_results["paired_fixed_oracle_cost_ms_per_token"]
    )
    trajectories = pd.DataFrame(trajectory_rows)
    summary_rows = []
    metrics = [
        "cost_ratio_of_sums_ms_per_token",
        "paired_difference_ms_per_token",
        "realized_excess_vs_paired_fixed_oracle_pct",
        "theorem_aligned_pseudo_regret",
        "pseudo_regret_per_round",
        "throughput_tokens_s",
        "round_cost_p95_ms_per_token",
        "round_cost_p99_ms_per_token",
        "mean_selected_k",
        "cloud_energy_j_per_accepted_token",
    ]
    for policy, group in run_results.groupby("policy", sort=False):
        row = {"policy": policy, "independent_runs": group["independent_run"].nunique()}
        for metric in metrics:
            mean, low, high = ci95_t(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)

    pd.DataFrame(
        [
            {"k": arm, "population_cost_ms_per_token": population_costs[arm]}
            for arm in arms
        ]
    ).to_csv(output / "independent_arm_costs.csv", index=False)
    contextual_cells.to_csv(output / "contextual_oracle_cells.csv", index=False)
    per_arm_run.to_csv(output / "arm_run_metrics.csv", index=False)
    per_arm_summary.to_csv(output / "arm_summary.csv", index=False)
    crossing_run.to_csv(output / "adjacent_arm_crossing_run.csv", index=False)
    crossing_summary.to_csv(output / "adjacent_arm_crossing_summary.csv", index=False)
    prefix_run.to_csv(output / "prefix_acceptance_run.csv", index=False)
    prefix_summary.to_csv(output / "prefix_acceptance_summary.csv", index=False)
    if not per_context_run.empty:
        per_context_run.to_csv(output / "arm_context_run_metrics.csv", index=False)
        per_context_summary.to_csv(output / "arm_context_summary.csv", index=False)
        context_crossing_run.to_csv(
            output / "context_adjacent_arm_crossing_run.csv", index=False
        )
        context_crossing_summary.to_csv(
            output / "context_adjacent_arm_crossing_summary.csv", index=False
        )
    if not payload_scaling.empty:
        payload_scaling.to_csv(output / "payload_scaling.csv", index=False)
    run_results.to_csv(output / "policy_run_results.csv", index=False)
    trajectories.to_csv(output / "policy_trajectories.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "policy_summary.csv", index=False)
    metadata = {
        "analysis": "policy_replay",
        "analysis_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "potential_outcomes": str(input_path),
        "potential_outcomes_sha256": sha256_file(input_path),
        "arguments": vars(args),
        "verification_mode": str(frame["verification_mode"].iloc[0]),
        "fixed_rtt_sensitivity": {
            "added_rtt_ms": float(args.added_rtt_ms),
            "interpretation": (
                "A common deterministic RTT offset was added to each paired "
                "round numerator and network-delay field before policy replay. "
                "This is a model-aligned counterfactual sensitivity analysis, "
                "not an additional measured wireless trace."
                if args.added_rtt_ms > 0
                else "No counterfactual RTT offset was applied."
            ),
        },
        "calibration_runs": [int(value) for value in calibration_runs],
        "evaluation_runs": [int(value) for value in evaluation_runs],
        "prompt_stratum_token_boundaries": context_boundaries,
        "fixed_oracle_arm": int(fixed_oracle_arm),
        "fixed_oracle_population_cost_ms_per_token": fixed_oracle_cost,
        "displayed_lcb_scale": {
            "calibration_n_max_ms": calibration_n_max,
            "n_max_guard_factor": args.n_max_guard_factor,
            "displayed_n_max_ms": displayed_n_max,
            "displayed_l_max_ms": displayed_l_max,
            "evaluation_n_max_ms": evaluation_n_max,
            "evaluation_exceeded_displayed_n_max": bool(
                evaluation_n_max > displayed_n_max
            ),
            "exp3_ratio_loss_scale_ms_per_token": displayed_n_max,
            "definition": (
                "L_max = N_max/B_min + N_max*A_max/B_min^2 with "
                "B_min=1 and A_max=K_max+1"
            ),
            "boundary": (
                "N_max is fixed from calibration with a predeclared guard "
                "factor; it is an empirical deployment scale, not a certified "
                "physical bound."
            ),
        },
        "realized_excess_definition": (
            "Within each evaluation run, compare the policy ratio of sums with "
            "the independently calibrated fixed arm evaluated on the same paired "
            "prompt and trace stream."
        ),
        "context_state_edges_ms": edges.tolist(),
        "aggregate_contextual_oracle": {
            "definition": (
                "Exact optimizer of the empirical aggregate ratio over the "
                "calibration state-to-arm policy class via Dinkelbach separation"
            ),
            "arm_map": {
                int(state): int(arm)
                for state, arm in aggregate_contextual_map.items()
            },
            "calibration_cost_ms_per_token": aggregate_contextual_cost,
            "oracle_voi_ms_per_token": fixed_oracle_cost - aggregate_contextual_cost,
            "oracle_voi_pct_of_blind": 100.0
            * (fixed_oracle_cost - aggregate_contextual_cost)
            / fixed_oracle_cost,
        },
        "per_state_ratio_oracle": {
            "definition": (
                "Diagnostic map formed by minimizing the ratio separately in "
                "each state; it need not optimize the aggregate ratio"
            ),
            "arm_map": {
                int(state): int(arm) for state, arm in per_state_ratio_map.items()
            },
        },
        "pseudo_regret_definition": "sum_t [C_hat(k_t)-C_hat(k*)] with C_hat estimated only on calibration runs",
        "hindsight_boundary": "per_round_hindsight_diagnostic is unattainable and is not a theorem comparator",
        "banditspec_length_baseline": {
            "source_repository": "https://github.com/sail-sg/BanditSpec",
            "source_commit": "4e6dd869a3a1677ca62ee8055713cec439f26ccc",
            "public_function": "ucb_length_spec_generate",
            "reward": "committed tokens in one round divided by complete round wall-clock seconds",
            "warmup": "cycle arms while t < 2*K, matching the public implementation",
            "confidence": "common scalar bonus, matching the public implementation",
            "adaptation": "batch size one; edge-cloud round time; arms 1..K; canonical no-speculation arm unavailable in frozen-input tables",
        },
        "greedy_prefix_nesting_audit": greedy_prefix_nesting_audit(
            frame, max(arms)
        ),
    }
    if not payload_scaling.empty:
        slope_mean, slope_low, slope_high = ci95_t(
            payload_scaling["payload_bytes_per_additional_draft_token"]
        )
        metadata["payload_scaling"] = {
            "bytes_per_additional_draft_token_mean": slope_mean,
            "bytes_per_additional_draft_token_ci95_low": slope_low,
            "bytes_per_additional_draft_token_ci95_high": slope_high,
            "replicate_unit": "independent_run",
        }
    output_files = [
        "independent_arm_costs.csv",
        "contextual_oracle_cells.csv",
        "arm_run_metrics.csv",
        "arm_summary.csv",
        "adjacent_arm_crossing_run.csv",
        "adjacent_arm_crossing_summary.csv",
        "prefix_acceptance_run.csv",
        "prefix_acceptance_summary.csv",
        "policy_run_results.csv",
        "policy_trajectories.csv",
        "policy_summary.csv",
    ]
    if not per_context_run.empty:
        output_files.extend(
            [
                "arm_context_run_metrics.csv",
                "arm_context_summary.csv",
                "context_adjacent_arm_crossing_run.csv",
                "context_adjacent_arm_crossing_summary.csv",
            ]
        )
    if not payload_scaling.empty:
        output_files.append("payload_scaling.csv")
    metadata["output_sha256"] = {
        name: sha256_file(output / name)
        for name in output_files
    }
    (output / "policy_analysis_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def estimate_transition(states: np.ndarray, n_states: int, laplace: float = 0.0):
    counts = np.full((n_states, n_states), laplace, dtype=float)
    for left, right in zip(states[:-1], states[1:]):
        counts[int(left), int(right)] += 1.0
    row_sum = counts.sum(axis=1, keepdims=True)
    for index in range(n_states):
        if row_sum[index, 0] == 0:
            counts[index, index] = 1.0
    transition = counts / counts.sum(axis=1, keepdims=True)
    occupancy = np.bincount(states, minlength=n_states).astype(float)
    occupancy /= occupancy.sum()
    return transition, counts, occupancy


def stochastic_monotonicity_violation(transition: np.ndarray) -> float:
    cdf = np.cumsum(transition, axis=1)
    # For increasingly worse states, row CDFs must be non-increasing.
    return float(max(0.0, np.max(np.diff(cdf, axis=0))))


def solve_dp(
    round_cost: np.ndarray,
    token_yield: np.ndarray,
    state_delay: np.ndarray,
    transition: np.ndarray,
    pi0: np.ndarray,
    tolerance: float = 1e-9,
):
    k_max = len(round_cost) - 1

    def evaluate(lam: float):
        value = np.full((k_max + 1, len(state_delay)), np.nan)
        gamma = np.full_like(value, np.inf)
        g = np.full_like(value, np.nan)
        for n in range(1, k_max + 1):
            g[n] = round_cost[n] + 2.0 * state_delay - lam * token_yield[n]
        value[k_max] = g[k_max]
        for n in range(k_max - 1, 0, -1):
            continuation = transition @ value[n + 1]
            gamma[n] = continuation - g[n]
            value[n] = np.minimum(g[n], continuation)
        return float(pi0 @ value[1]), value, gamma, g

    low = 0.0
    f_low = evaluate(low)[0]
    high = max(1.0, float(np.nanmax(round_cost[1:] / token_yield[1:])))
    f_high = evaluate(high)[0]
    while f_high > 0 and high < 1e9:
        high *= 2.0
        f_high = evaluate(high)[0]
    if f_low < 0 or f_high > 0:
        raise RuntimeError("could not bracket the Dinkelbach root")
    for _ in range(100):
        middle = 0.5 * (low + high)
        f_middle = evaluate(middle)[0]
        if abs(f_middle) <= tolerance:
            low = high = middle
            break
        if f_middle > 0:
            low = middle
        else:
            high = middle
    root = 0.5 * (low + high)
    f_root, value, gamma, g = evaluate(root)
    return root, f_root, value, gamma, g, evaluate


def evaluate_threshold_policy(
    thresholds: Sequence[int],
    round_cost: np.ndarray,
    token_yield: np.ndarray,
    state_delay: np.ndarray,
    transition: np.ndarray,
    pi0: np.ndarray,
) -> float:
    """Evaluate one state-dependent draft-depth threshold policy.

    State indices are ordered from lower to higher delay.  A threshold value
    ``h_s`` means stop at the first depth ``n >= h_s`` in state ``s``.  The
    final arm is always a forced stop, matching :func:`solve_dp`.
    """

    threshold_array = np.asarray(thresholds, dtype=int)
    k_max = len(round_cost) - 1
    n_states = len(state_delay)
    if threshold_array.shape != (n_states,):
        raise ValueError("one threshold is required for each state")
    if np.any(threshold_array < 1) or np.any(threshold_array > k_max):
        raise ValueError("thresholds must lie between 1 and K_max")

    numerator = np.full((k_max + 1, n_states), np.nan)
    denominator = np.full_like(numerator, np.nan)
    numerator[k_max] = round_cost[k_max] + 2.0 * state_delay
    denominator[k_max] = token_yield[k_max]
    for depth in range(k_max - 1, 0, -1):
        stop = depth >= threshold_array
        continue_numerator = transition @ numerator[depth + 1]
        continue_denominator = transition @ denominator[depth + 1]
        numerator[depth] = np.where(
            stop, round_cost[depth] + 2.0 * state_delay, continue_numerator
        )
        denominator[depth] = np.where(
            stop, token_yield[depth], continue_denominator
        )
    expected_numerator = float(pi0 @ numerator[1])
    expected_denominator = float(pi0 @ denominator[1])
    return expected_numerator / expected_denominator


def stopping_distribution(
    stop_decisions: np.ndarray,
    transition: np.ndarray,
    pi0: np.ndarray,
) -> Dict[str, object]:
    """Return reach and stopping probabilities for a finite Markov policy."""

    stop = np.asarray(stop_decisions, dtype=bool).copy()
    n_states = transition.shape[0]
    if stop.ndim != 2 or stop.shape[1] != n_states:
        raise ValueError("stop_decisions must have shape (K_max, n_states)")
    stop[-1, :] = True
    active = np.asarray(pi0, dtype=float).copy()
    reachable = []
    stopping = []
    for depth in range(stop.shape[0]):
        reachable.append(active.copy())
        stop_mass = active * stop[depth]
        stopping.append(float(stop_mass.sum()))
        active = (active * (~stop[depth])) @ transition
    stopping_array = np.asarray(stopping, dtype=float)
    if not np.isclose(stopping_array.sum(), 1.0, atol=1e-8):
        raise ValueError("finite-horizon stopping probabilities do not sum to one")
    depths = np.arange(1, len(stopping_array) + 1, dtype=float)
    return {
        "p_tau_gt_1": float(1.0 - stopping_array[0]),
        "expected_tau": float(depths @ stopping_array),
        "stop_depth_probabilities": stopping_array.tolist(),
        "reachable_state_depth_probabilities": np.asarray(reachable).tolist(),
    }


def best_ordered_threshold_policy(
    round_cost: np.ndarray,
    token_yield: np.ndarray,
    state_delay: np.ndarray,
    transition: np.ndarray,
    pi0: np.ndarray,
) -> Tuple[float, Tuple[int, ...]]:
    """Enumerate the best delay-ordered threshold policy.

    Higher-delay states may continue to at least as large a draft depth, so
    the threshold vector is nondecreasing in the ordered state index.
    """

    k_max = len(round_cost) - 1
    best_cost = float("inf")
    best_thresholds: Tuple[int, ...] = tuple()
    for thresholds in itertools.combinations_with_replacement(
        range(1, k_max + 1), len(state_delay)
    ):
        cost = evaluate_threshold_policy(
            thresholds,
            round_cost,
            token_yield,
            state_delay,
            transition,
            pi0,
        )
        if cost < best_cost:
            best_cost = cost
            best_thresholds = tuple(int(value) for value in thresholds)
    return best_cost, best_thresholds


def check_gamma(gamma: np.ndarray, tolerance: float = 1e-7) -> Dict[str, object]:
    finite = gamma[1:-1]
    depth_delta = np.diff(gamma[1:], axis=0)
    state_delta = np.diff(finite, axis=1) if finite.shape[1] > 1 else np.empty((0, 0))
    condition_i = bool(np.all(depth_delta >= -tolerance))
    gamma_state_sufficient = bool(np.all(state_delta <= tolerance)) if state_delta.size else True
    stop = gamma[1:] >= -tolerance
    upper_interval = True
    for column in stop.T:
        if np.any(np.diff(column.astype(int)) < 0):
            upper_interval = False
            break
    ordered_region = True
    for row in stop:
        if np.any(np.diff(row.astype(int)) > 0):
            ordered_region = False
            break
    thresholds = []
    for state in range(stop.shape[1]):
        indices = np.where(stop[:, state])[0]
        thresholds.append(int(indices[0] + 1) if len(indices) else int(stop.shape[0]))
    return {
        "condition_i_depth_monotone": condition_i,
        "condition_ii_ordered_stopping_region": ordered_region,
        "stop_region_upper_interval": upper_interval,
        "ordered_threshold_policy": upper_interval and ordered_region,
        "gamma_nonincreasing_in_state_sufficient": gamma_state_sufficient,
        "min_depth_gamma_increment": float(np.nanmin(depth_delta)),
        "max_state_gamma_increment": float(np.nanmax(state_delta)) if state_delta.size else 0.0,
        "thresholds": thresholds,
    }


def empirical_g_b(frame: pd.DataFrame, k_max: int) -> Tuple[np.ndarray, np.ndarray]:
    round_cost = np.full(k_max + 1, np.nan)
    token_yield = np.full(k_max + 1, np.nan)
    for k in range(1, k_max + 1):
        group = frame[frame["k_selected"] == k]
        if group.empty:
            raise ValueError(f"no rows for arm {k}")
        # The replay design makes total time approximately G(k)+trace RTT.
        # Subtract the paired application-level trace RTT to estimate G(k).
        round_cost[k] = float(
            np.mean(group["total_round_time_ms"] - group["trace_http_rtt_ms"])
        )
        token_yield[k] = float(group["accepted_total"].mean())
    return round_cost, token_yield


def cell_arm_g_b_tables(
    frame: pd.DataFrame, arms: Sequence[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned paired cell-by-arm tables for cost and token yield."""

    cell_index = ["independent_run", "round"]
    arm_order = [int(arm) for arm in arms]
    cell_g = frame.assign(
        _g_component=(
            pd.to_numeric(frame["total_round_time_ms"], errors="raise")
            - pd.to_numeric(frame["trace_http_rtt_ms"], errors="raise")
        )
    ).pivot(index=cell_index, columns="k_selected", values="_g_component")
    cell_b = frame.pivot(
        index=cell_index, columns="k_selected", values="accepted_total"
    )
    cell_g = cell_g.reindex(columns=arm_order)
    cell_b = cell_b.reindex(index=cell_g.index, columns=arm_order)
    if cell_g.isna().any().any() or cell_b.isna().any().any():
        raise ValueError("incomplete cell-by-arm matrix for Markov bootstrap")
    return cell_g, cell_b


def cell_arm_g_b_matrices(
    frame: pd.DataFrame, arms: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired cell-by-arm NumPy matrices for invariant checks."""

    cell_g, cell_b = cell_arm_g_b_tables(frame, arms)
    return cell_g.to_numpy(dtype=float), cell_b.to_numpy(dtype=float)


def hierarchical_paired_cell_indices(
    index: pd.MultiIndex, rng: np.random.Generator
) -> np.ndarray:
    """Resample runs, then paired round cells within each sampled run."""

    run_values = np.asarray(index.get_level_values("independent_run"))
    runs = np.unique(run_values)
    sampled_positions = []
    for sampled_run in rng.choice(runs, size=len(runs), replace=True):
        positions = np.flatnonzero(run_values == sampled_run)
        sampled_positions.extend(
            rng.choice(positions, size=len(positions), replace=True).tolist()
        )
    return np.asarray(sampled_positions, dtype=int)


def block_bootstrap_indices(length: int, block: int, rng: np.random.Generator):
    result = []
    while len(result) < length:
        start = int(rng.integers(0, length))
        result.extend((start + offset) % length for offset in range(block))
    return np.asarray(result[:length], dtype=int)


def hierarchical_block_paired_cell_indices(
    index: pd.MultiIndex, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample run clusters, then circular blocks of paired cells per run."""

    if block_length < 1:
        raise ValueError("block_length must be positive")
    run_values = np.asarray(index.get_level_values("independent_run"))
    runs = np.unique(run_values)
    sampled_positions = []
    for sampled_run in rng.choice(runs, size=len(runs), replace=True):
        positions = np.flatnonzero(run_values == sampled_run)
        local = block_bootstrap_indices(len(positions), block_length, rng)
        sampled_positions.extend(positions[local].tolist())
    return np.asarray(sampled_positions, dtype=int)


def analyze_markov(args: argparse.Namespace) -> None:
    if args.added_rtt_ms < 0:
        raise ValueError("--added-rtt-ms must be nonnegative")
    potential_path = Path(args.potential_outcomes).expanduser().resolve()
    trace_path = Path(args.trace).expanduser().resolve()
    potential = apply_fixed_rtt_offset(
        pd.read_csv(potential_path), args.added_rtt_ms
    )
    arms = validate_potential_outcomes(potential)
    k_max = max(arms)
    trace = pd.read_csv(trace_path)
    trace = trace[trace["success"].astype(int) == 1].copy()
    trace["rtt_ms"] = pd.to_numeric(trace["rtt_ms"], errors="coerce")
    trace = trace[np.isfinite(trace["rtt_ms"])].reset_index(drop=True)
    trace["rtt_ms"] += float(args.added_rtt_ms)
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    round_cost, token_yield = empirical_g_b(potential, k_max)

    certificate_rows = []
    instance_rows = []
    transition_rows = []
    bootstrap_rows = []
    model_parameter_rows = []
    bootstrap_parameter_rows = []
    stress_rows = []
    rng = np.random.default_rng(args.seed)

    for n_states in args.state_counts:
        edges = state_edges(trace["rtt_ms"], n_states)
        states = assign_states(trace["rtt_ms"], edges)
        transition, counts, pi0 = estimate_transition(states, n_states)
        state_rtt = np.array(
            [trace.loc[states == state, "rtt_ms"].mean() for state in range(n_states)]
        )
        state_delay = state_rtt / 2.0
        root, f_root, value, gamma, g, evaluate = solve_dp(
            round_cost, token_yield, state_delay, transition, pi0
        )
        check = check_gamma(gamma)
        stop_stats = stopping_distribution(gamma[1:] >= -1e-7, transition, pi0)
        threshold_cost, threshold_vector = best_ordered_threshold_policy(
            round_cost, token_yield, state_delay, transition, pi0
        )
        threshold_gap_pct = 100.0 * (threshold_cost - root) / root
        monotonicity_violation = stochastic_monotonicity_violation(transition)
        trace_rtt = trace["rtt_ms"].to_numpy()
        lag1_autocorrelation = (
            float(np.corrcoef(trace_rtt[:-1], trace_rtt[1:])[0, 1])
            if len(trace) > 2
            else np.nan
        )
        instance_rows.append(
            {
                "states": n_states,
                "lambda_star": root,
                "dinkelbach_residual": f_root,
                "stochastic_monotonicity_max_cdf_violation": monotonicity_violation,
                "stochastic_monotonicity_diagnostic_pass": bool(
                    monotonicity_violation <= 1e-7
                ),
                "trace_rtt_lag1_autocorrelation": lag1_autocorrelation,
                "best_ordered_threshold_cost_ms_per_token": threshold_cost,
                "ordered_threshold_gap_pct": threshold_gap_pct,
                "ordered_threshold_exact_match": bool(
                    abs(threshold_cost - root) <= 1e-7 * max(1.0, abs(root))
                ),
                "best_ordered_thresholds": ";".join(map(str, threshold_vector)),
                **check,
                "thresholds": ";".join(map(str, check["thresholds"])),
                "state_mean_rtt_ms": ";".join(f"{value:.6f}" for value in state_rtt),
                "state_edges_ms": ";".join(f"{value:.6f}" for value in edges[1:-1]),
                "p_tau_gt_1": stop_stats["p_tau_gt_1"],
                "expected_tau": stop_stats["expected_tau"],
                "stop_depth_probabilities": ";".join(
                    f"{value:.9f}" for value in stop_stats["stop_depth_probabilities"]
                ),
            }
        )
        model_parameter_rows.append(
            {
                "states": int(n_states),
                "k_max": int(k_max),
                "solver_tolerance": 1e-9,
                "lambda_star": float(root),
                "dinkelbach_residual": float(f_root),
                "round_cost_g_ms": round_cost[1:].tolist(),
                "token_yield_b": token_yield[1:].tolist(),
                "state_delay_d_ms": state_delay.tolist(),
                "initial_distribution_pi0": pi0.tolist(),
                "transition_matrix_p": transition.tolist(),
                "state_edges_ms": edges[1:-1].tolist(),
                "unrestricted_stop_decisions": (
                    gamma[1:] >= -1e-7
                ).astype(int).tolist(),
                **stop_stats,
            }
        )
        for source in range(n_states):
            for target in range(n_states):
                transition_rows.append(
                    {
                        "states": n_states,
                        "source_state": source,
                        "target_state": target,
                        "transition_count": counts[source, target],
                        "transition_probability": transition[source, target],
                        "source_occupancy": pi0[source],
                        "source_mean_rtt_ms": state_rtt[source],
                    }
                )

        lambda_grid = np.linspace(max(0.0, 0.5 * root), 1.5 * root, args.lambda_grid)
        if not np.any(np.isclose(lambda_grid, root)):
            lambda_grid = np.sort(np.append(lambda_grid, root))
        for lam in lambda_grid:
            _, _, current_gamma, _ = evaluate(float(lam))
            current_check = check_gamma(current_gamma)
            for depth in range(1, k_max + 1):
                for state in range(n_states):
                    certificate_rows.append(
                        {
                            "states": n_states,
                            "lambda": float(lam),
                            "is_lambda_star": bool(np.isclose(lam, root)),
                            "depth": depth,
                            "state": state,
                            "gamma": current_gamma[depth, state],
                            "stop": bool(current_gamma[depth, state] >= -1e-7),
                            "condition_i_depth_monotone": current_check[
                                "condition_i_depth_monotone"
                            ],
                            "condition_ii_ordered_stopping_region": current_check[
                                "condition_ii_ordered_stopping_region"
                            ],
                        }
                    )

        cell_g_table, cell_b_table = cell_arm_g_b_tables(potential, arms)
        cell_g_values = cell_g_table.to_numpy(dtype=float)
        cell_b_values = cell_b_table.to_numpy(dtype=float)
        for bootstrap in range(args.bootstrap):
            sampled_trace = trace.iloc[
                block_bootstrap_indices(len(trace), args.block_length, rng)
            ].reset_index(drop=True)
            boot_edges = state_edges(sampled_trace["rtt_ms"], n_states)
            boot_states = assign_states(sampled_trace["rtt_ms"], boot_edges)
            boot_transition, _, boot_pi0 = estimate_transition(boot_states, n_states)
            boot_monotonicity_violation = stochastic_monotonicity_violation(
                boot_transition
            )
            boot_rtt = np.array(
                [
                    sampled_trace.loc[boot_states == state, "rtt_ms"].mean()
                    for state in range(n_states)
                ]
            )
            cell_indices = hierarchical_paired_cell_indices(cell_g_table.index, rng)
            boot_g = np.full(k_max + 1, np.nan)
            boot_b = np.full(k_max + 1, np.nan)
            boot_g[np.asarray(arms, dtype=int)] = cell_g_values[cell_indices].mean(
                axis=0
            )
            boot_b[np.asarray(arms, dtype=int)] = cell_b_values[cell_indices].mean(
                axis=0
            )
            try:
                boot_root, boot_residual, _, boot_gamma, _, _ = solve_dp(
                    boot_g, boot_b, boot_rtt / 2.0, boot_transition, boot_pi0
                )
                boot_check = check_gamma(boot_gamma)
                boot_stop_stats = stopping_distribution(
                    boot_gamma[1:] >= -1e-7,
                    boot_transition,
                    boot_pi0,
                )
                boot_threshold_cost, boot_threshold_vector = (
                    best_ordered_threshold_policy(
                        boot_g,
                        boot_b,
                        boot_rtt / 2.0,
                        boot_transition,
                        boot_pi0,
                    )
                )
                boot_threshold_gap_pct = (
                    100.0 * (boot_threshold_cost - boot_root) / boot_root
                )
                error = ""
            except Exception as exc:
                boot_root = np.nan
                boot_threshold_cost = np.nan
                boot_threshold_gap_pct = np.nan
                boot_threshold_vector = tuple()
                boot_check = {
                    "condition_i_depth_monotone": False,
                    "condition_ii_ordered_stopping_region": False,
                    "stop_region_upper_interval": False,
                    "ordered_threshold_policy": False,
                    "gamma_nonincreasing_in_state_sufficient": False,
                    "min_depth_gamma_increment": np.nan,
                    "max_state_gamma_increment": np.nan,
                }
                boot_residual = np.nan
                boot_stop_stats = {
                    "p_tau_gt_1": np.nan,
                    "expected_tau": np.nan,
                    "stop_depth_probabilities": [],
                    "reachable_state_depth_probabilities": [],
                }
                error = f"{type(exc).__name__}: {exc}"
            bootstrap_rows.append(
                {
                    "states": n_states,
                    "bootstrap": bootstrap,
                    "lambda_star": boot_root,
                    "stochastic_monotonicity_max_cdf_violation": (
                        boot_monotonicity_violation
                    ),
                    "stochastic_monotonicity_diagnostic_pass": bool(
                        boot_monotonicity_violation <= 1e-7
                    ),
                    "best_ordered_threshold_cost_ms_per_token": (
                        boot_threshold_cost
                    ),
                    "ordered_threshold_gap_pct": boot_threshold_gap_pct,
                    "ordered_threshold_exact_match": bool(
                        np.isfinite(boot_threshold_gap_pct)
                        and abs(boot_threshold_gap_pct) <= 1e-7
                    ),
                    "best_ordered_thresholds": ";".join(
                        map(str, boot_threshold_vector)
                    ),
                    **boot_check,
                    "p_tau_gt_1": boot_stop_stats["p_tau_gt_1"],
                    "expected_tau": boot_stop_stats["expected_tau"],
                    "stop_depth_probabilities": ";".join(
                        f"{value:.9f}"
                        for value in boot_stop_stats["stop_depth_probabilities"]
                    ),
                    "error": error,
                }
            )
            bootstrap_parameter_rows.append(
                {
                    "states": int(n_states),
                    "bootstrap": int(bootstrap),
                    "k_max": int(k_max),
                    "solver_tolerance": 1e-9,
                    "lambda_star": (
                        float(boot_root) if np.isfinite(boot_root) else None
                    ),
                    "dinkelbach_residual": (
                        float(boot_residual)
                        if np.isfinite(boot_residual)
                        else None
                    ),
                    "round_cost_g_ms": boot_g[1:].tolist(),
                    "token_yield_b": boot_b[1:].tolist(),
                    "state_delay_d_ms": (boot_rtt / 2.0).tolist(),
                    "initial_distribution_pi0": boot_pi0.tolist(),
                    "transition_matrix_p": boot_transition.tolist(),
                    "best_ordered_thresholds": list(boot_threshold_vector),
                    "ordered_threshold_gap_pct": (
                        float(boot_threshold_gap_pct)
                        if np.isfinite(boot_threshold_gap_pct)
                        else None
                    ),
                    **boot_stop_stats,
                    "error": error,
                }
            )

        iid_projection = np.tile(pi0, (n_states, 1))
        iid_root, _, _, iid_gamma, _, _ = solve_dp(
            round_cost, token_yield, state_delay, iid_projection, pi0
        )
        iid_check = check_gamma(iid_gamma)
        stress_rows.append(
            {
                "states": n_states,
                "stress_kind": "iid_occupancy_row_projection",
                "strength": 1.0,
                "lambda_star": iid_root,
                "stochastic_monotonicity_max_cdf_violation": stochastic_monotonicity_violation(
                    iid_projection
                ),
                **iid_check,
                "thresholds": ";".join(map(str, iid_check["thresholds"])),
            }
        )
        for strength in np.linspace(0.0, 1.0, args.stress_steps):
            unordered = (1.0 - strength) * transition + strength * transition[::-1]
            unordered /= unordered.sum(axis=1, keepdims=True)
            stress_root, _, _, stress_gamma, _, _ = solve_dp(
                round_cost, token_yield, state_delay, unordered, pi0
            )
            stress_check = check_gamma(stress_gamma)
            stress_rows.append(
                {
                    "states": n_states,
                    "stress_kind": "unordered_row_reversal",
                    "strength": strength,
                    "lambda_star": stress_root,
                    "stochastic_monotonicity_max_cdf_violation": stochastic_monotonicity_violation(
                        unordered
                    ),
                    **stress_check,
                    "thresholds": ";".join(map(str, stress_check["thresholds"])),
                }
            )
            # Action/depth-dependent transitions increasingly push later draft
            # depths toward the worse-state uniform tail. This deliberately
            # leaves the exogenous-state proposition's model class.
            depth_transitions = []
            for depth in range(1, k_max):
                worse_tail = np.zeros_like(transition)
                worse_tail[:, -1] = 1.0
                depth_strength = strength * depth / max(k_max - 1, 1)
                matrix = (1.0 - depth_strength) * transition + depth_strength * worse_tail
                depth_transitions.append(matrix / matrix.sum(axis=1, keepdims=True))

            def solve_action_dependent():
                def evaluate(lam):
                    v = np.full((k_max + 1, n_states), np.nan)
                    gam = np.full_like(v, np.inf)
                    gg = np.full_like(v, np.nan)
                    for depth in range(1, k_max + 1):
                        gg[depth] = (
                            round_cost[depth]
                            + 2.0 * state_delay
                            - lam * token_yield[depth]
                        )
                    v[k_max] = gg[k_max]
                    for depth in range(k_max - 1, 0, -1):
                        continuation = depth_transitions[depth - 1] @ v[depth + 1]
                        gam[depth] = continuation - gg[depth]
                        v[depth] = np.minimum(gg[depth], continuation)
                    return float(pi0 @ v[1]), gam

                low, high = 0.0, max(1.0, root * 2.0)
                while evaluate(high)[0] > 0:
                    high *= 2.0
                for _ in range(80):
                    middle = 0.5 * (low + high)
                    if evaluate(middle)[0] > 0:
                        low = middle
                    else:
                        high = middle
                action_root = 0.5 * (low + high)
                return action_root, evaluate(action_root)[1]

            action_root, action_gamma = solve_action_dependent()
            action_check = check_gamma(action_gamma)
            stress_rows.append(
                {
                    "states": n_states,
                    "stress_kind": "action_depth_dependent_transition",
                    "strength": strength,
                    "lambda_star": action_root,
                    "stochastic_monotonicity_max_cdf_violation": np.nan,
                    **action_check,
                    "thresholds": ";".join(map(str, action_check["thresholds"])),
                }
            )

    certificate_frame = pd.DataFrame(certificate_rows)
    certificate_frame.to_csv(output / "markov_gamma_certificate.csv", index=False)
    lambda_checks = certificate_frame.drop_duplicates(["states", "lambda"])
    lambda_summary = (
        lambda_checks.groupby("states")
        .agg(
            lambda_grid_points=("lambda", "count"),
            condition_i_pass_fraction=("condition_i_depth_monotone", "mean"),
            condition_ii_pass_fraction=(
                "condition_ii_ordered_stopping_region",
                "mean",
            ),
        )
        .reset_index()
    )
    lambda_summary.to_csv(output / "markov_lambda_summary.csv", index=False)
    pd.DataFrame(instance_rows).to_csv(output / "markov_instance_summary.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(output / "markov_transition_matrices.csv", index=False)
    (output / "markov_model_parameters.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in model_parameter_rows),
        encoding="utf-8",
    )
    (output / "markov_bootstrap_parameters.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in bootstrap_parameter_rows
        ),
        encoding="utf-8",
    )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame.to_csv(output / "markov_bootstrap_runs.csv", index=False)
    bootstrap_summary = (
        bootstrap_frame.groupby("states")
        .agg(
            bootstrap_resamples=("bootstrap", "count"),
            condition_i_pass_fraction=("condition_i_depth_monotone", "mean"),
            condition_ii_pass_fraction=("condition_ii_ordered_stopping_region", "mean"),
            stop_region_upper_interval_pass_fraction=(
                "stop_region_upper_interval",
                "mean",
            ),
            ordered_threshold_policy_pass_fraction=(
                "ordered_threshold_policy",
                "mean",
            ),
            stochastic_monotonicity_diagnostic_pass_fraction=(
                "stochastic_monotonicity_diagnostic_pass",
                "mean",
            ),
            gamma_state_sufficient_pass_fraction=(
                "gamma_nonincreasing_in_state_sufficient",
                "mean",
            ),
            lambda_star_mean=("lambda_star", "mean"),
            lambda_star_std=("lambda_star", "std"),
            p_tau_gt_1_mean=("p_tau_gt_1", "mean"),
            expected_tau_mean=("expected_tau", "mean"),
            nontrivial_policy_fraction=("p_tau_gt_1", lambda x: float((x > 0.01).mean())),
        )
        .reset_index()
    )
    bootstrap_summary.to_csv(output / "markov_bootstrap_summary.csv", index=False)
    robustness_rows = []
    for n_states, group in bootstrap_frame.groupby("states", sort=True):
        failed = group[
            (~group["condition_i_depth_monotone"].astype(bool))
            | (~group["stochastic_monotonicity_diagnostic_pass"].astype(bool))
        ].copy()
        gaps = pd.to_numeric(failed["ordered_threshold_gap_pct"], errors="coerce")
        gaps = gaps[np.isfinite(gaps)]
        robustness_rows.append(
            {
                "states": int(n_states),
                "assumption_failure_resamples": int(len(failed)),
                "valid_gap_resamples": int(len(gaps)),
                "ordered_threshold_exact_match_fraction": float(
                    failed.loc[gaps.index, "ordered_threshold_exact_match"].mean()
                )
                if len(gaps)
                else np.nan,
                "ordered_threshold_gap_mean_pct": float(gaps.mean())
                if len(gaps)
                else np.nan,
                "ordered_threshold_gap_median_pct": float(gaps.median())
                if len(gaps)
                else np.nan,
                "ordered_threshold_gap_p95_pct": float(np.percentile(gaps, 95))
                if len(gaps)
                else np.nan,
                "ordered_threshold_gap_max_pct": float(gaps.max())
                if len(gaps)
                else np.nan,
            }
        )
    pd.DataFrame(robustness_rows).to_csv(
        output / "markov_robustness_summary.csv", index=False
    )
    pd.DataFrame(stress_rows).to_csv(output / "markov_stress_tests.csv", index=False)
    output_names = (
        "markov_gamma_certificate.csv",
        "markov_lambda_summary.csv",
        "markov_instance_summary.csv",
        "markov_transition_matrices.csv",
        "markov_bootstrap_runs.csv",
        "markov_bootstrap_summary.csv",
        "markov_robustness_summary.csv",
        "markov_stress_tests.csv",
        "markov_model_parameters.jsonl",
        "markov_bootstrap_parameters.jsonl",
    )
    manifest = {
        "analysis": "bounded_horizon_markov_certificate",
        "analysis_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "potential_outcomes": str(potential_path),
        "potential_outcomes_sha256": sha256_file(potential_path),
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "arguments": vars(args),
        "verification_mode": str(potential["verification_mode"].iloc[0]),
        "fixed_rtt_sensitivity": {
            "added_rtt_ms": float(args.added_rtt_ms),
            "interpretation": (
                "Common deterministic RTT sensitivity applied to both paired "
                "outcomes and trace states; not an additional measured trace."
                if args.added_rtt_ms > 0
                else "No counterfactual RTT offset was applied."
            ),
        },
        "trace_transition_cadence": {
            "median_sampling_interval_s": (
                float(np.median(np.diff(trace["monotonic_start_s"])))
                if "monotonic_start_s" in trace.columns and len(trace) > 1
                else None
            ),
            "interpretation": (
                "The estimated transition matrix is trace-quantized at the RTT "
                "sampling cadence; it is not asserted to equal a per-token "
                "within-round transition kernel."
            ),
        },
        "calibration_bootstrap": (
            "Hierarchical paired bootstrap: independent runs are resampled "
            "first, then round cells within each sampled run; all arms share "
            "the same sampled cell indices."
        ),
        "output_sha256": {
            name: sha256_file(output / name) for name in output_names
        },
    }
    (output / "markov_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def analyze_horizon(args: argparse.Namespace) -> None:
    """Run paired global and contextual policy stress tests over long horizons."""

    if args.n_resamples < 1:
        raise ValueError("--n-resamples must be positive")
    if args.context_states < 1:
        raise ValueError("--context-states must be positive")
    if args.fixed_map_bootstrap < 1:
        raise ValueError("--fixed-map-bootstrap must be positive")
    if args.fixed_map_block_length < 1:
        raise ValueError("--fixed-map-block-length must be positive")
    horizons = sorted(set(int(value) for value in args.horizons))
    if not horizons or horizons[0] < 1:
        raise ValueError("--horizons must contain positive integers")
    input_path = Path(args.potential_outcomes).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = apply_fixed_rtt_offset(pd.read_csv(input_path), args.added_rtt_ms)
    arms = validate_potential_outcomes(frame)
    runs = [int(value) for value in sorted(frame["independent_run"].unique())]
    calibration_count = max(1, int(math.floor(len(runs) * args.calibration_fraction)))
    calibration_count = min(calibration_count, len(runs) - 1)
    calibration_runs = runs[:calibration_count]
    evaluation_runs = runs[calibration_count:]
    calibration = frame[frame["independent_run"].isin(calibration_runs)].copy()
    evaluation = frame[frame["independent_run"].isin(evaluation_runs)].copy()

    population_costs = arm_population_costs(calibration, arms)
    fixed_arm = min(population_costs, key=population_costs.get)
    edges = state_edges(calibration["trace_http_rtt_ms"], args.context_states)
    per_state_map = state_aware_arm_map(calibration, arms, edges)
    contextual_map, contextual_cost, contextual_cells = aggregate_contextual_oracle(
        calibration, arms, edges
    )
    state_costs = state_arm_population_costs(calibration, arms, edges)
    calibration_n_max = float(calibration["total_round_time_ms"].max())
    displayed_n_max = calibration_n_max * args.n_max_guard_factor
    displayed_l_max = displayed_n_max * (1.0 + max(arms) + 1.0)

    cell_index = ["independent_run", "round"]
    numerator = evaluation.pivot(
        index=cell_index, columns="k_selected", values="total_round_time_ms"
    ).reindex(columns=arms)
    denominator = evaluation.pivot(
        index=cell_index, columns="k_selected", values="accepted_total"
    ).reindex(index=numerator.index, columns=arms)
    rtt = (
        evaluation.drop_duplicates(cell_index)
        .set_index(cell_index)
        .reindex(numerator.index)["trace_http_rtt_ms"]
    )
    if numerator.isna().any().any() or denominator.isna().any().any():
        raise ValueError("incomplete paired evaluation cells")
    arm_to_column = {int(arm): index for index, arm in enumerate(arms)}
    numerator_values = numerator.to_numpy(dtype=float)
    denominator_values = denominator.to_numpy(dtype=float)
    rtt_values = rtt.to_numpy(dtype=float)

    evaluation_states = assign_states(rtt_values, edges).astype(int)
    fixed_column = arm_to_column[int(fixed_arm)]
    contextual_arms = np.asarray(
        [contextual_map[int(state)] for state in evaluation_states], dtype=int
    )
    contextual_columns = np.asarray(
        [arm_to_column[int(arm)] for arm in contextual_arms], dtype=int
    )
    cell_rows = np.arange(len(numerator_values), dtype=int)
    global_numerator = numerator_values[:, fixed_column]
    global_denominator = denominator_values[:, fixed_column]
    contextual_numerator = numerator_values[cell_rows, contextual_columns]
    contextual_denominator = denominator_values[cell_rows, contextual_columns]

    def fixed_map_costs(indices: np.ndarray) -> Tuple[float, float]:
        global_cost = float(
            global_numerator[indices].sum() / global_denominator[indices].sum()
        )
        contextual_fixed_cost = float(
            contextual_numerator[indices].sum()
            / contextual_denominator[indices].sum()
        )
        return global_cost, contextual_fixed_cost

    global_heldout_cost, contextual_heldout_cost = fixed_map_costs(cell_rows)
    heldout_bootstrap_rows = []
    heldout_rng = np.random.default_rng(args.seed + 91_973)
    for bootstrap in range(args.fixed_map_bootstrap):
        sampled = hierarchical_block_paired_cell_indices(
            numerator.index, args.fixed_map_block_length, heldout_rng
        )
        global_cost, contextual_fixed_cost = fixed_map_costs(sampled)
        gain = global_cost - contextual_fixed_cost
        heldout_bootstrap_rows.append(
            {
                "bootstrap": bootstrap,
                "global_fixed_cost_ms_per_token": global_cost,
                "contextual_map_cost_ms_per_token": contextual_fixed_cost,
                "contextual_minus_global_cost_ms_per_token": -gain,
                "global_minus_contextual_gain_ms_per_token": gain,
                "global_minus_contextual_gain_pct": 100.0 * gain / global_cost,
            }
        )
    heldout_bootstrap = pd.DataFrame(heldout_bootstrap_rows)
    heldout_bootstrap.to_csv(
        output / "contextual_fixed_heldout_bootstrap.csv", index=False
    )
    heldout_gain = global_heldout_cost - contextual_heldout_cost
    heldout_gain_samples = heldout_bootstrap[
        "global_minus_contextual_gain_ms_per_token"
    ].to_numpy(dtype=float)
    heldout_gain_pct_samples = heldout_bootstrap[
        "global_minus_contextual_gain_pct"
    ].to_numpy(dtype=float)
    heldout_map_audit = {
        "global_fixed_cost_ms_per_token": global_heldout_cost,
        "contextual_map_cost_ms_per_token": contextual_heldout_cost,
        "global_minus_contextual_gain_ms_per_token": heldout_gain,
        "global_minus_contextual_gain_pct": (
            100.0 * heldout_gain / global_heldout_cost
        ),
        "gain_ci95_low_ms_per_token": float(
            np.quantile(heldout_gain_samples, 0.025)
        ),
        "gain_ci95_high_ms_per_token": float(
            np.quantile(heldout_gain_samples, 0.975)
        ),
        "gain_ci95_low_pct": float(
            np.quantile(heldout_gain_pct_samples, 0.025)
        ),
        "gain_ci95_high_pct": float(
            np.quantile(heldout_gain_pct_samples, 0.975)
        ),
        "bootstrap_probability_positive_gain": float(
            np.mean(heldout_gain_samples > 0.0)
        ),
        "bootstrap_resamples": int(args.fixed_map_bootstrap),
        "block_length_rounds": int(args.fixed_map_block_length),
        "uncertainty": (
            "Conditional paired hierarchical bootstrap: evaluation runs are "
            "resampled as clusters, followed by circular block resampling of "
            "paired round cells within each selected run."
        ),
    }

    global_policies = {
        "per_round_ratio_ucb",
        "ucb_specstop_self_normalized",
        "ucb_specstop_displayed_lcb",
        "banditspec_throughput_public",
        "banditspec_ucb",
    }
    contextual_policies = {
        "contextual_per_round_ratio_ucb",
        "contextual_ucb_specstop_self_normalized",
        "contextual_ucb_specstop_displayed_lcb",
        "contextual_banditspec_ucb",
    }
    policy_names = [
        "calibration_selected_global_fixed",
        "calibration_selected_contextual_fixed",
        "per_round_ratio_ucb",
        "contextual_per_round_ratio_ucb",
        "ucb_specstop_self_normalized",
        "contextual_ucb_specstop_self_normalized",
        "ucb_specstop_displayed_lcb",
        "contextual_ucb_specstop_displayed_lcb",
        "banditspec_throughput_public",
        "banditspec_ucb",
        "contextual_banditspec_ucb",
    ]
    rows = []
    pull_rows = []
    rng = np.random.default_rng(args.seed)
    max_horizon = max(horizons)
    for resample in range(args.n_resamples):
        sampled_cells = rng.integers(0, len(numerator), size=max_horizon)
        for horizon in horizons:
            cell_sequence = sampled_cells[:horizon]
            states = assign_states(rtt_values[cell_sequence], edges).astype(int)
            fixed_column = arm_to_column[int(fixed_arm)]
            fixed_cost = float(
                numerator_values[cell_sequence, fixed_column].sum()
                / denominator_values[cell_sequence, fixed_column].sum()
            )
            contextual_arms = np.asarray(
                [contextual_map[int(state)] for state in states], dtype=int
            )
            contextual_columns = np.asarray(
                [arm_to_column[int(arm)] for arm in contextual_arms], dtype=int
            )
            contextual_fixed_cost = float(
                numerator_values[cell_sequence, contextual_columns].sum()
                / denominator_values[cell_sequence, contextual_columns].sum()
            )

            for policy_index, policy in enumerate(policy_names):
                if policy in global_policies:
                    select, update = make_policy(
                        policy,
                        max(arms),
                        args.beta,
                        args.seed + resample * 1000 + horizon + policy_index,
                        horizon,
                        displayed_l_max,
                        displayed_n_max,
                    )
                    contextual_select = contextual_update = None
                elif policy in contextual_policies:
                    contextual_select, contextual_update = make_contextual_policy(
                        policy,
                        max(arms),
                        args.context_states,
                        args.beta,
                        horizon,
                        displayed_l_max,
                    )
                    select = update = None
                else:
                    select = update = contextual_select = contextual_update = None

                selected_numerator = 0.0
                selected_denominator = 0.0
                selected_arm_sum = 0.0
                contextual_pseudo_regret = 0.0
                pull_counts = np.zeros(
                    (args.context_states, len(arms)), dtype=int
                )
                for time_index, (cell, state) in enumerate(
                    zip(cell_sequence, states), start=1
                ):
                    if policy == "calibration_selected_global_fixed":
                        arm = int(fixed_arm)
                    elif policy == "calibration_selected_contextual_fixed":
                        arm = int(contextual_map[int(state)])
                    elif policy in contextual_policies:
                        arm = int(contextual_select(int(state)))
                    else:
                        arm = int(select(time_index))

                    column = arm_to_column[arm]
                    current_n = float(numerator_values[cell, column])
                    current_b = float(denominator_values[cell, column])
                    if policy in contextual_policies:
                        contextual_update(int(state), arm, current_n, current_b)
                    elif update is not None:
                        row = SimpleNamespace(
                            total_round_time_ms=current_n,
                            accepted_total=current_b,
                        )
                        update(arm, row)
                    selected_numerator += current_n
                    selected_denominator += current_b
                    selected_arm_sum += arm
                    pull_counts[int(state), column] += 1
                    contextual_pseudo_regret += (
                        state_costs[(int(state), arm)]
                        - state_costs[(int(state), int(per_state_map[int(state)]))]
                    )

                cost = selected_numerator / selected_denominator
                rows.append(
                    {
                        "resample": resample,
                        "horizon": horizon,
                        "policy": policy,
                        "cost_ratio_of_sums_ms_per_token": cost,
                        "paired_global_fixed_cost_ms_per_token": fixed_cost,
                        "paired_contextual_fixed_cost_ms_per_token": contextual_fixed_cost,
                        "excess_vs_paired_fixed_pct": 100.0
                        * (cost - fixed_cost)
                        / fixed_cost,
                        "excess_vs_paired_contextual_fixed_pct": 100.0
                        * (cost - contextual_fixed_cost)
                        / contextual_fixed_cost,
                        "contextual_pseudo_regret": contextual_pseudo_regret,
                        "contextual_pseudo_regret_per_round": (
                            contextual_pseudo_regret / horizon
                        ),
                        "mean_selected_k": selected_arm_sum / horizon,
                    }
                )
                for state in range(args.context_states):
                    state_total = int(pull_counts[state].sum())
                    for arm_index, arm in enumerate(arms):
                        count = int(pull_counts[state, arm_index])
                        pull_rows.append(
                            {
                                "resample": resample,
                                "horizon": horizon,
                                "policy": policy,
                                "network_state": state,
                                "k_selected": int(arm),
                                "pull_count": count,
                                "state_visit_count": state_total,
                                "within_state_pull_fraction": (
                                    count / state_total if state_total else np.nan
                                ),
                            }
                        )

    run_frame = pd.DataFrame(rows)
    pull_frame = pd.DataFrame(pull_rows)
    run_frame.to_csv(output / "horizon_resample_runs.csv", index=False)
    pull_frame.to_csv(output / "horizon_state_arm_pulls.csv", index=False)
    summary_rows = []
    for (horizon, policy), group in run_frame.groupby(
        ["horizon", "policy"], sort=False
    ):
        row = {
            "horizon": int(horizon),
            "policy": policy,
            "n_resamples": int(group["resample"].nunique()),
            "mean_selected_k": float(group["mean_selected_k"].mean()),
        }
        for source, prefix in (
            ("excess_vs_paired_fixed_pct", "excess"),
            (
                "excess_vs_paired_contextual_fixed_pct",
                "contextual_reference_excess",
            ),
            ("contextual_pseudo_regret_per_round", "contextual_regret_per_round"),
        ):
            mean, low, high = ci95_t(group[source])
            row[f"{prefix}_mean_pct" if "excess" in prefix else f"{prefix}_mean"] = mean
            row[f"{prefix}_ci95_low_pct" if "excess" in prefix else f"{prefix}_ci95_low"] = low
            row[f"{prefix}_ci95_high_pct" if "excess" in prefix else f"{prefix}_ci95_high"] = high
            row[f"{prefix}_median_pct" if "excess" in prefix else f"{prefix}_median"] = float(
                group[source].median()
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "horizon_summary.csv", index=False)
    manifest = {
        "analysis": "paired_global_contextual_horizon_stress_test",
        "analysis_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "potential_outcomes": str(input_path),
        "potential_outcomes_sha256": sha256_file(input_path),
        "arguments": vars(args),
        "calibration_runs": calibration_runs,
        "evaluation_runs": evaluation_runs,
        "paired_cell_pool_size": int(len(numerator)),
        "policies": policy_names,
        "context": {
            "information_structure": "oracle-context paired replay",
            "source": (
                "The exogenous trace RTT attached to the current replay cell is "
                "revealed before arm selection. This is an information-value "
                "stress test, not an implementation of RTT acquisition. A deployed "
                "controller would require a previous-round estimate or a separately "
                "costed pre-round probe. No current-round arm outcome is used to "
                "define the state."
            ),
            "calibration_only_edges_ms": edges.tolist(),
            "calibration_internal_edges_ms": [
                float(value) for value in edges[1:-1]
            ],
            "calibration_state_occupancy": {
                str(int(state)): float(group["state_occupancy"].iloc[0])
                for state, group in contextual_cells.groupby(
                    "network_state", sort=True
                )
            },
            "global_fixed_arm": int(fixed_arm),
            "global_fixed_calibration_cost_ms_per_token": float(
                population_costs[fixed_arm]
            ),
            "per_state_ratio_map": {
                str(state): int(arm) for state, arm in per_state_map.items()
            },
            "aggregate_ratio_map": {
                str(state): int(arm) for state, arm in contextual_map.items()
            },
            "aggregate_ratio_calibration_cost_ms_per_token": float(
                contextual_cost
            ),
            "aggregate_ratio_calibration_gain_ms_per_token": float(
                population_costs[fixed_arm] - contextual_cost
            ),
            "aggregate_ratio_calibration_gain_pct_of_global": float(
                100.0
                * (population_costs[fixed_arm] - contextual_cost)
                / population_costs[fixed_arm]
            ),
            "heldout_fixed_map_audit": heldout_map_audit,
            "map_optimization": (
                "Exact empirical aggregate-ratio optimization by Dinkelbach "
                "separation over all state-to-arm mappings; exact ties select "
                "the smallest arm. This calibration optimizer is not a "
                "population oracle."
            ),
        },
        "confidence_modes": {
            "lcb": (
                "beta is dimensionless because the confidence width already "
                "contains the calibration-derived L_max scale"
            ),
            "sn": (
                "beta has numerator-time units; beta=1.0 denotes 1 ms in this "
                "replay because total_round_time_ms is the numerator"
            ),
        },
        "comparators": {
            "aggregate_system_cost": (
                "ratio of summed round time to summed committed tokens"
            ),
            "contextual_pseudo_regret": (
                "sum of calibration-estimated statewise cost gaps relative to "
                "the per-state ratio minimizer"
            ),
        },
        "boundary": (
            "Each synthetic sequence samples paired evaluation cells with "
            "replacement. This isolates finite-horizon behavior under the "
            "empirical outcome distribution and is not a new continuous "
            "hardware run or a validation of temporal independence."
        ),
        "output_sha256": {
            "horizon_resample_runs.csv": sha256_file(
                output / "horizon_resample_runs.csv"
            ),
            "horizon_state_arm_pulls.csv": sha256_file(
                output / "horizon_state_arm_pulls.csv"
            ),
            "horizon_summary.csv": sha256_file(output / "horizon_summary.csv"),
            "contextual_fixed_heldout_bootstrap.csv": sha256_file(
                output / "contextual_fixed_heldout_bootstrap.csv"
            ),
        },
    }
    (output / "horizon_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def integrate_edge_energy(requests: pd.DataFrame, telemetry: pd.DataFrame) -> pd.DataFrame:
    result = requests.copy()
    telemetry = telemetry.copy()
    telemetry["monotonic_ns"] = pd.to_numeric(telemetry["monotonic_ns"], errors="coerce")
    telemetry["vdd_in_mw"] = pd.to_numeric(telemetry["vdd_in_mw"], errors="coerce")
    telemetry["ram_used_mb"] = pd.to_numeric(telemetry["ram_used_mb"], errors="coerce")
    power = telemetry.dropna(subset=["monotonic_ns", "vdd_in_mw"]).sort_values(
        "monotonic_ns"
    )
    power_seconds = power["monotonic_ns"].to_numpy(dtype=float) / 1e9
    power_watts = power["vdd_in_mw"].to_numpy(dtype=float) / 1000.0
    energy_values = []
    peak_ram_values = []
    samples_values = []
    for row in result.itertuples(index=False):
        window = telemetry[
            (telemetry["monotonic_ns"] >= row.monotonic_start_ns)
            & (telemetry["monotonic_ns"] <= row.monotonic_end_ns)
        ].dropna(subset=["vdd_in_mw"])
        samples_values.append(len(window))
        peak_ram_values.append(float(window["ram_used_mb"].max()) if len(window) else np.nan)
        if len(power_seconds):
            start_s = float(row.monotonic_start_ns) / 1e9
            end_s = float(row.monotonic_end_ns) / 1e9
            inside = (power_seconds > start_s) & (power_seconds < end_s)
            seconds = np.concatenate(([start_s], power_seconds[inside], [end_s]))
            watts = np.interp(seconds, power_seconds, power_watts)
            energy_values.append(float(np.trapz(watts, seconds)))
        else:
            energy_values.append(np.nan)
    result["edge_energy_j"] = energy_values
    result["edge_energy_per_token_j"] = result["edge_energy_j"] / result["generated_tokens"]
    result["edge_peak_ram_mb"] = peak_ram_values
    result["edge_telemetry_samples"] = samples_values
    return result


def aggregate_request_run_metrics(
    enriched: pd.DataFrame, keys: Sequence[str]
) -> pd.DataFrame:
    """Aggregate sequential requests without averaging per-request ratios."""

    rows = []
    group_columns = list(keys) + ["independent_run"]
    for key_values, group in enriched.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(group_columns, key_values))
        generated_tokens = pd.to_numeric(
            group["generated_tokens"], errors="coerce"
        ).sum()
        e2e_ms = pd.to_numeric(group["e2e_ms"], errors="coerce")
        row.update(
            {
                "requests": len(group),
                "generated_tokens": float(generated_tokens),
                "ttft_ms": float(
                    pd.to_numeric(group["ttft_ms"], errors="coerce").mean()
                ),
                "e2e_ms": float(e2e_ms.mean()),
                "throughput_tokens_s": float(
                    generated_tokens / (e2e_ms.sum() / 1000.0)
                ),
                "edge_peak_ram_mb": float(group["edge_peak_ram_mb"].max()),
                "edge_process_peak_allocated_mb": float(
                    group["edge_process_peak_allocated_mb"].max()
                ),
                "cloud_peak_memory_mb": float(group["cloud_peak_memory_mb"].max()),
            }
        )
        for source, target in (
            ("edge_energy_j", "edge_energy_per_token_j"),
            ("cloud_energy_j", "cloud_energy_per_token_j"),
        ):
            values = pd.to_numeric(group[source], errors="coerce")
            row[target] = (
                float(values.sum() / generated_tokens)
                if values.notna().all() and generated_tokens > 0
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_request_run_metrics(
    enriched: pd.DataFrame,
    per_run: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Summarize independent-run request metrics and raw request tails."""

    metrics = [
        "ttft_ms",
        "e2e_ms",
        "throughput_tokens_s",
        "edge_energy_per_token_j",
        "cloud_energy_per_token_j",
        "edge_peak_ram_mb",
        "edge_process_peak_allocated_mb",
        "cloud_peak_memory_mb",
    ]
    summary_rows = []
    for key_values, group in per_run.groupby(list(keys), dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        raw = enriched
        for key, value in zip(keys, key_values):
            raw = raw[raw[key] == value]
        row = dict(zip(keys, key_values))
        row["independent_runs"] = group["independent_run"].nunique()
        row["request_samples"] = len(raw)
        for metric in metrics:
            mean, low, high = ci95_t(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        for metric in ("ttft_ms", "e2e_ms"):
            row[f"{metric}_p50_across_requests"] = float(
                np.percentile(raw[metric], 50)
            )
            row[f"{metric}_p95_across_requests"] = float(
                np.percentile(raw[metric], 95)
            )
            row[f"{metric}_p99_across_requests"] = float(
                np.percentile(raw[metric], 99)
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def analyze_system(args: argparse.Namespace) -> None:
    requests_path = Path(args.request_metrics).expanduser().resolve()
    telemetry_path = Path(args.edge_telemetry).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    requests_frame = pd.read_csv(requests_path)
    telemetry = pd.read_csv(telemetry_path)
    enriched = integrate_edge_energy(requests_frame, telemetry)
    enriched.to_csv(output / "request_metrics_with_edge_energy.csv", index=False)

    keys = ["k", "prompt_stratum", "generation_token_cap"]
    per_run = aggregate_request_run_metrics(enriched, keys)
    per_run.to_csv(output / "request_metrics_per_run.csv", index=False)
    summary = summarize_request_run_metrics(enriched, per_run, keys)
    summary.to_csv(output / "request_metrics_summary.csv", index=False)

    overall_keys = ["k", "generation_token_cap"]
    overall_per_run = aggregate_request_run_metrics(enriched, overall_keys)
    overall_per_run.to_csv(
        output / "request_metrics_overall_per_run.csv", index=False
    )
    overall_summary = summarize_request_run_metrics(
        enriched, overall_per_run, overall_keys
    )
    overall_summary.to_csv(
        output / "request_metrics_overall_summary.csv", index=False
    )
    output_names = (
        "request_metrics_with_edge_energy.csv",
        "request_metrics_per_run.csv",
        "request_metrics_summary.csv",
        "request_metrics_overall_per_run.csv",
        "request_metrics_overall_summary.csv",
    )
    manifest = {
        "analysis": "request_level_system_metrics",
        "analysis_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "request_metrics": str(requests_path),
        "request_metrics_sha256": sha256_file(requests_path),
        "edge_telemetry": str(telemetry_path),
        "edge_telemetry_sha256": sha256_file(telemetry_path),
        "arguments": vars(args),
        "edge_energy_integration": (
            "Full request monotonic-clock window with linear interpolation "
            "of tegrastats VDD_IN power at both request boundaries."
        ),
        "output_sha256": {
            name: sha256_file(output / name) for name in output_names
        },
    }
    (output / "system_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def analyze_concurrency(args: argparse.Namespace) -> None:
    requests_path = Path(args.requests).expanduser().resolve()
    run_summary_path = Path(args.run_summary).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    requests_frame = pd.read_csv(requests_path)
    run_summary = pd.read_csv(run_summary_path)
    required_request = {"independent_run", "concurrency", "latency_ms"}
    required_summary = {
        "independent_run",
        "concurrency",
        "achieved_requests_s",
        "accepted_tokens_s",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    }
    if missing := required_request.difference(requests_frame.columns):
        raise ValueError(f"concurrency requests are missing {sorted(missing)}")
    if missing := required_summary.difference(run_summary.columns):
        raise ValueError(f"concurrency run summary is missing {sorted(missing)}")

    metrics = [
        "achieved_requests_s",
        "accepted_tokens_s",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    ]
    rows = []
    rng = np.random.default_rng(args.seed)
    for concurrency, group in run_summary.groupby("concurrency", sort=True):
        raw = requests_frame[requests_frame["concurrency"] == concurrency]
        row = {
            "concurrency": int(concurrency),
            "independent_runs": int(group["independent_run"].nunique()),
            "request_samples": len(raw),
            "request_latency_p50_ms": float(np.percentile(raw["latency_ms"], 50)),
            "request_latency_p95_ms": float(np.percentile(raw["latency_ms"], 95)),
            "request_latency_p99_ms": float(np.percentile(raw["latency_ms"], 99)),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            sample_indices = rng.integers(
                0, len(values), size=(args.bootstrap_resamples, len(values))
            )
            bootstrap_means = values[sample_indices].mean(axis=1)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci95_low"] = float(
                np.quantile(bootstrap_means, 0.025)
            )
            row[f"{metric}_ci95_high"] = float(
                np.quantile(bootstrap_means, 0.975)
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "concurrency_summary_ci.csv", index=False)
    manifest = {
        "analysis": "cloud_verifier_concurrency",
        "analysis_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "requests": str(requests_path),
        "requests_sha256": sha256_file(requests_path),
        "run_summary": str(run_summary_path),
        "run_summary_sha256": sha256_file(run_summary_path),
        "scope_boundary": (
            "Cloud verifier plus wireless RPC under closed-loop clients; not "
            "full-pipeline concurrent edge drafting or an open-loop scheduler"
        ),
        "uncertainty": (
            "Conditional percentile bootstrap interval over seeded run clusters "
            "within one hardware session "
            f"({args.bootstrap_resamples} resamples, seed={args.seed})."
        ),
        "output_sha256": {
            "concurrency_summary_ci.csv": sha256_file(
                output / "concurrency_summary_ci.csv"
            )
        },
    }
    (output / "concurrency_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def analyze_trace(args: argparse.Namespace) -> None:
    trace_path = Path(args.trace).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(trace_path)
    summary = pd.DataFrame([summarize_rtt_trace(frame)])
    summary_path = output / "trace_summary.csv"
    summary.to_csv(summary_path, index=False)

    source_manifest = trace_path.with_suffix(".manifest.json")
    manifest = {
        "analysis": "rtt_trace_integrity_and_distribution",
        "analysis_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "source_manifest": str(source_manifest) if source_manifest.exists() else None,
        "source_manifest_sha256": (
            sha256_file(source_manifest) if source_manifest.exists() else None
        ),
        "output_sha256": {"trace_summary.csv": sha256_file(summary_path)},
        "statistical_unit": "one successful RTT probe",
        "lag1_boundary": (
            "Lag-one autocorrelation is computed after removing failed probes; "
            "it is a trace diagnostic, not a per-token transition estimate."
        ),
    }
    (output / "trace_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    policies = subparsers.add_parser("policies")
    policies.add_argument("--potential-outcomes", required=True)
    policies.add_argument("--out-dir", required=True)
    policies.add_argument("--calibration-fraction", type=float, default=0.3)
    policies.add_argument("--context-states", type=int, default=5)
    policies.add_argument("--beta", type=float, default=1.0)
    policies.add_argument("--n-max-guard-factor", type=float, default=1.25)
    policies.add_argument("--added-rtt-ms", type=float, default=0.0)
    policies.add_argument("--seed", type=int, default=20260818)

    markov = subparsers.add_parser("markov")
    markov.add_argument("--potential-outcomes", required=True)
    markov.add_argument("--trace", required=True)
    markov.add_argument("--out-dir", required=True)
    markov.add_argument("--state-counts", type=parse_int_list, default=[3, 5])
    markov.add_argument("--lambda-grid", type=int, default=21)
    markov.add_argument("--bootstrap", type=int, default=200)
    markov.add_argument("--block-length", type=int, default=20)
    markov.add_argument("--stress-steps", type=int, default=11)
    markov.add_argument("--added-rtt-ms", type=float, default=0.0)
    markov.add_argument("--seed", type=int, default=20260818)

    horizon = subparsers.add_parser("horizon")
    horizon.add_argument("--potential-outcomes", required=True)
    horizon.add_argument("--out-dir", required=True)
    horizon.add_argument(
        "--horizons", type=parse_int_list, default=[100, 500, 1000, 5000]
    )
    horizon.add_argument("--n-resamples", type=int, default=100)
    horizon.add_argument("--calibration-fraction", type=float, default=0.3)
    horizon.add_argument("--context-states", type=int, default=3)
    horizon.add_argument("--beta", type=float, default=1.0)
    horizon.add_argument("--n-max-guard-factor", type=float, default=1.25)
    horizon.add_argument("--fixed-map-bootstrap", type=int, default=10000)
    horizon.add_argument("--fixed-map-block-length", type=int, default=20)
    horizon.add_argument("--added-rtt-ms", type=float, default=0.0)
    horizon.add_argument("--seed", type=int, default=20260824)

    system = subparsers.add_parser("system")
    system.add_argument("--request-metrics", required=True)
    system.add_argument("--edge-telemetry", required=True)
    system.add_argument("--out-dir", required=True)

    concurrency = subparsers.add_parser("concurrency")
    concurrency.add_argument("--requests", required=True)
    concurrency.add_argument("--run-summary", required=True)
    concurrency.add_argument("--out-dir", required=True)
    concurrency.add_argument("--bootstrap-resamples", type=int, default=10000)
    concurrency.add_argument("--seed", type=int, default=20260827)

    trace = subparsers.add_parser("trace")
    trace.add_argument("--trace", required=True)
    trace.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "policies":
        analyze_policies(args)
    elif args.mode == "markov":
        analyze_markov(args)
    elif args.mode == "horizon":
        analyze_horizon(args)
    elif args.mode == "system":
        analyze_system(args)
    elif args.mode == "concurrency":
        analyze_concurrency(args)
    elif args.mode == "trace":
        analyze_trace(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
