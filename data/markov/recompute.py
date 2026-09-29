"""Read-only numerical and provenance checks for the supplementary audit.

Packet inputs and additional local archive inputs are recorded separately.
Only this audit directory receives generated outputs. No hardware is used.
"""

import csv
import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ARCHIVE = OUT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_model(record):
    g = np.asarray(record["round_cost_g_ms"], dtype=float)
    b = np.asarray(record["token_yield_b"], dtype=float)
    d = np.asarray(record["state_delay_d_ms"], dtype=float)
    p = np.asarray(record["transition_matrix_p"], dtype=float)
    pi = np.asarray(record["initial_distribution_pi0"], dtype=float)
    terminal = g[:, None] + 2 * d[None, :]
    horizon, states = terminal.shape

    def backward(lam):
        stop_cost = terminal - lam * b[:, None]
        values = stop_cost[-1].copy()
        gamma = np.full((horizon, states), np.inf)
        for depth in range(horizon - 2, -1, -1):
            continuation = p @ values
            gamma[depth] = continuation - stop_cost[depth]
            values = np.minimum(continuation, stop_cost[depth])
        return float(pi @ values), gamma

    lo, hi = 0.0, float(np.max(terminal / b[:, None]) + 1.0)
    for _ in range(65):
        mid = (lo + hi) / 2
        residual, gamma = backward(mid)
        if abs(residual) <= 1e-10:
            break
        if residual > 0:
            lo = mid
        else:
            hi = mid
    stop = gamma >= -1e-7
    mass = pi.copy()
    numerator = denominator = 0.0
    probabilities = []
    for depth in range(horizon):
        exited = mass * stop[depth]
        probabilities.append(float(exited.sum()))
        numerator += float(exited @ terminal[depth])
        denominator += float(exited.sum() * b[depth])
        mass = (mass - exited) @ p
    policy_cost = numerator / denominator

    # Evaluate all ordered threshold vectors in parallel, with no archived
    # selection decisions or costs used in their construction.
    thresholds = np.asarray(list(itertools.combinations_with_replacement(
        range(1, horizon + 1), states
    )))
    mass = np.broadcast_to(pi, thresholds.shape).copy()
    numerators = np.zeros(len(thresholds))
    denominators = np.zeros(len(thresholds))
    for depth in range(horizon):
        exited = mass * (thresholds <= depth + 1)
        numerators += exited @ terminal[depth]
        denominators += exited.sum(axis=1) * b[depth]
        mass = (mass - exited) @ p
    costs = numerators / denominators
    best = int(np.argmin(costs))
    q = np.diff(np.concatenate(([1.0], b)))
    valid = bool(q[0] <= 1 + 1e-10 and q.min() >= -1e-10
                 and np.diff(q).max() <= 1e-10)
    condition_i = bool(np.all(np.diff(gamma, axis=0) >= -1e-7))
    condition_ii = bool(np.all(np.diff(stop.astype(int), axis=1) <= 0))
    return {
        "root": mid,
        "root_residual": residual,
        "policy_cost": policy_cost,
        "best_threshold_cost": float(costs[best]),
        "prefix_valid": valid,
        "condition_i": condition_i,
        "condition_ii": condition_ii,
        "p_continue": 1 - probabilities[0],
        "max_q_increase": float(np.diff(q).max()),
        "restriction_loss_fraction": float(costs[best] / policy_cost - 1),
    }


def markov_checks():
    source = ARCHIVE / "markov_all_inputs_outputs.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    totals = Counter()
    invalid = []
    mismatches = []
    maximum = Counter()
    for index, record in enumerate(records):
        result = solve_model(record)
        is_boot = record["kind"] == "bootstrap"
        totals[record["kind"]] += 1
        for key in ("prefix_valid", "condition_i", "condition_ii"):
            if result[key] != record[key]:
                mismatches.append([index, key, result[key], record[key]])
        errors = {
            "policy_cost_error": abs(result["policy_cost"] - record["policy_cost_ms_per_token"]),
            "threshold_cost_error": abs(result["best_threshold_cost"] - record["threshold_cost_ms_per_token"]),
            "root_residual": abs(result["root_residual"]),
            "relative_restriction_loss": abs(result["restriction_loss_fraction"]),
        }
        for key, value in errors.items():
            maximum[key] = max(maximum[key], value)
        if errors["policy_cost_error"] > 1e-7 or errors["threshold_cost_error"] > 1e-7:
            mismatches.append([index, "cost", errors])
        if is_boot:
            both = result["condition_i"] and result["condition_ii"]
            cont = result["p_continue"] > .05
            totals["both_pass"] += int(both)
            totals["continuation_group"] += int(cont)
            totals["invalid_prefix"] += int(not result["prefix_valid"])
            totals["invalid_and_both_pass"] += int(not result["prefix_valid"] and both)
            totals["invalid_and_continuation"] += int(not result["prefix_valid"] and cont)
            if not result["prefix_valid"]:
                invalid.append({"index": index, "condition": record["condition"],
                                "bootstrap": record.get("bootstrap"),
                                "max_q_increase": result["max_q_increase"]})
        if (index + 1) % 400 == 0:
            print(f"Recomputed {index + 1}/{len(records)} archived Markov inputs", flush=True)
    return {"scope": "full-precision inputs packaged with the supplement",
            "counts": dict(totals), "maximum_errors": dict(maximum),
            "mismatches": mismatches, "invalid_details": invalid}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT.parents[1] / "outputs/markov")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = markov_checks()
    (args.out_dir / "recomputation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("counts", "maximum_errors", "mismatches")}, indent=2))
    assert not result["mismatches"]
