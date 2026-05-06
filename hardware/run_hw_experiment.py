"""
run_hw_experiment.py — H1–H5: full hardware validation suite.

Run on the Jetson. The 3090 server must be running cloud_server.py first.
tc netem is configured either locally (on Jetson outgoing interface) or via SSH.

Experiments:
  H1/E1  —  k* sweep: empirical vs theoretical, verify log scaling (review.md E1)
  H2/E2  —  Phase transition: fine-grained sweep near d_c (review.md E2)
  H3/E3  —  All-strategy comparison + UCB-SpecStop online regret (review.md E3+E4)
  H4/E5  —  Markov channel: good/bad switching, VOI measurement (review.md E5)

Prerequisite: run measure_params.py (H0) first to produce params_measured.json.

Usage (all experiments):
  python run_hw_experiment.py --all \\
      --draft-model Qwen/Qwen2.5-0.5B \\
      --server http://192.168.1.100:8000 \\
      --iface eth0 \\
      --prompts prompts.txt \\
      --n-prompts 200

Usage (single experiment):
  python run_hw_experiment.py --exp h1 --draft-model Qwen/Qwen2.5-0.5B ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baselines import (
    UCBSpecStop, PerRoundRatioUCB, EXP3Ratio, EpsilonGreedyRatio,
    confidence_stop_policy, greedy_policy,
)
from src.core import B, C, compute_kstar, dc_theory
from src.plot_style import apply_ieee_style, save_figure

OUT_DIR = Path(__file__).parent.parent / "outputs" / "hardware"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_draft_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Hardware helpers (shared)
# ---------------------------------------------------------------------------

def load_draft_model(name: str):
    global _draft_model, _tokenizer
    print(f"[HW] Loading draft model: {name}  device={_device}")
    _tokenizer = AutoTokenizer.from_pretrained(name)
    _draft_model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16
    ).to(_device)
    _draft_model.eval()
    print("[HW] Draft model loaded.")


def generate_draft(input_ids: torch.Tensor, k: int) -> list[int]:
    with torch.no_grad():
        out = _draft_model.generate(
            input_ids, max_new_tokens=k, do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    return out[0, input_ids.shape[1]:].tolist()


def generate_draft_with_log_probs(
    input_ids: torch.Tensor, k: int
) -> tuple[list[int], list[float]]:
    with torch.no_grad():
        out = _draft_model.generate(
            input_ids, max_new_tokens=k, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    token_ids = out.sequences[0, input_ids.shape[1]:].tolist()
    log_probs = [
        torch.log_softmax(score[0], dim=-1)[tok].item()
        for score, tok in zip(out.scores, token_ids)
    ]
    return token_ids, log_probs


def verify_on_cloud(
    server: str, context_ids: list[int], draft_ids: list[int],
    draft_log_probs: Optional[list[float]] = None,
) -> dict:
    payload: dict = {"context_ids": context_ids, "draft_ids": draft_ids}
    if draft_log_probs is not None:
        payload["draft_log_probs"] = draft_log_probs
    resp = requests.post(f"{server}/verify", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def measure_rtt(server: str, n: int = 50) -> float:
    """Estimate one-way delay (ms) = median RTT / 2."""
    rtts = []
    for _ in range(n):
        t0 = time.perf_counter()
        requests.get(f"{server}/ping", timeout=5)
        rtts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(rtts)) / 2.0


def set_netem(iface: str, delay_ms: int, jitter_ms: int = 0,
              dist: str = "", server_user: str = "") -> None:
    """Configure tc netem locally or via SSH."""
    netem_script = Path(__file__).parent / "setup_netem.sh"
    dist_arg = [dist] if dist else []

    if server_user:
        # Configure on the remote server (SSH)
        cmd = ["ssh", server_user,
               f"sudo bash -s change {iface} {delay_ms} {jitter_ms} " +
               (" ".join(dist_arg))]
        subprocess.run(cmd, check=True, timeout=10)
    else:
        # Configure locally (Jetson outgoing interface)
        args = ["sudo", "bash", str(netem_script), "change",
                iface, str(delay_ms), str(jitter_ms)] + dist_arg
        subprocess.run(args, check=True, timeout=10)
    time.sleep(0.3)


def run_one_round(
    server: str, prompt: str, k: int, rejection_sampling: bool = True,
) -> dict:
    """Execute one draft-verify round; return timing and acceptance metrics."""
    input_ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(_device)
    context_ids = input_ids[0].tolist()

    t0 = time.perf_counter()
    if rejection_sampling:
        draft_ids, draft_lp = generate_draft_with_log_probs(input_ids, k)
    else:
        draft_ids = generate_draft(input_ids, k)
        draft_lp = None
    t_draft = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    resp = verify_on_cloud(server, context_ids, draft_ids, draft_lp)
    t_comm = (time.perf_counter() - t1) * 1000.0

    n_t = t_draft + t_comm
    a_t = resp["n_accepted"] + 1           # bonus token always included

    return {
        "k": k, "N_t": n_t, "A_t": a_t,
        "n_accepted": resp["n_accepted"],
        "t_draft_ms": t_draft,
        "t_comm_ms": t_comm,
        "verify_time_ms": resp["verify_time_ms"],
    }


def aggregate(records: list[dict]) -> dict:
    costs = [r["N_t"] / r["A_t"] for r in records]
    return {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "n_rounds": len(records),
        "mean_accepted": float(np.mean([r["n_accepted"] for r in records])),
    }


# ---------------------------------------------------------------------------
# H1/E1 — k* sweep: verify log(d) scaling
# ---------------------------------------------------------------------------

def experiment_h1_kstar_sweep(
    server: str, prompts: list[str], alpha: float, cd: float, cv: float,
    delays: list[int], iface: str, server_user: str, k_max: int,
    rejection_sampling: bool,
) -> None:
    print("\n=== H1/E1: k* sweep (log-scaling validation) ===")
    rows = []
    d_c = dc_theory(alpha, cd, cv)
    print(f"  Theoretical d_c = {d_c:.2f} ms,  alpha={alpha:.3f}, "
          f"cd={cd:.2f}, cv={cv:.2f}")

    for delay_ms in delays:
        set_netem(iface, delay_ms, server_user=server_user)
        rtt = measure_rtt(server)
        print(f"\n  d={delay_ms}ms  measured_rtt={rtt:.1f}ms")

        # Theory prediction
        k_theory = compute_kstar(alpha, cd, cv, float(delay_ms), k_max)
        c_theory = C(k_theory, float(delay_ms), alpha, cd, cv)

        # Empirical sweep: for each k, run n_sweep rounds and measure avg cost
        n_sweep = min(30, len(prompts))
        sweep_costs = {}
        for k in range(1, k_max + 1):
            records = [
                run_one_round(server, prompts[i % len(prompts)], k, rejection_sampling)
                for i in range(n_sweep)
            ]
            sweep_costs[k] = np.mean([r["N_t"] / r["A_t"] for r in records])

        k_empirical = int(min(sweep_costs, key=sweep_costs.get))
        c_empirical = sweep_costs[k_empirical]

        rows.append({
            "delay_ms": delay_ms,
            "k_theory": k_theory,
            "k_empirical": k_empirical,
            "C_theory": float(c_theory),
            "C_empirical": float(c_empirical),
            "delta_k": abs(k_empirical - k_theory),
            "rtt_ms": rtt,
        })
        print(f"  k_theory={k_theory}, k_empirical={k_empirical}, "
              f"delta={abs(k_empirical - k_theory)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "h1_kstar_sweep.csv", index=False)

    # Figure: empirical and theoretical k* vs log(d)
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(delays, df["k_theory"], "r--o", markersize=5, label=r"Theory $k^*(d)$")
    ax.plot(delays, df["k_empirical"], "b-s", markersize=5, label=r"Empirical $\hat{k}^*$")
    ax.set_xscale("log")
    ax.set_xlabel("Network delay $d$ (ms, log scale)")
    ax.set_ylabel("Optimal draft length $k^*$")
    ax.legend(frameon=True)
    save_figure(fig, OUT_DIR / "fig_h1_kstar_sweep")
    plt.close(fig)
    print(f"\n  Saved fig_h1_kstar_sweep.pdf, h1_kstar_sweep.csv")


# ---------------------------------------------------------------------------
# H2/E2 — Phase transition: fine-grained sweep near d_c
# ---------------------------------------------------------------------------

def experiment_h2_phase_transition(
    server: str, prompts: list[str], alpha: float, cd: float, cv: float,
    iface: str, server_user: str, k_max: int, rejection_sampling: bool,
) -> None:
    print("\n=== H2/E2: Phase transition near d_c ===")
    d_c = dc_theory(alpha, cd, cv)
    print(f"  Theoretical d_c = {d_c:.2f} ms")

    # Fine-grained delay sweep around d_c
    fine_delays = sorted(set(
        [max(1, int(d_c * f)) for f in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]]
    ))
    # Clamp to at least 1 ms (tc netem minimum)
    fine_delays = [max(1, d) for d in fine_delays]

    rows = []
    n_sweep = min(20, len(prompts))
    for delay_ms in fine_delays:
        set_netem(iface, delay_ms, server_user=server_user)
        k_theory = compute_kstar(alpha, cd, cv, float(delay_ms), k_max)

        sweep_costs = {}
        for k in range(1, min(k_max, 10) + 1):
            records = [
                run_one_round(server, prompts[i % len(prompts)], k, rejection_sampling)
                for i in range(n_sweep)
            ]
            sweep_costs[k] = np.mean([r["N_t"] / r["A_t"] for r in records])

        k_empirical = int(min(sweep_costs, key=sweep_costs.get))
        rows.append({"delay_ms": delay_ms, "k_theory": k_theory,
                     "k_empirical": k_empirical})
        print(f"  d={delay_ms}ms: k_theory={k_theory}, k_empirical={k_empirical}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "h2_phase_transition.csv", index=False)

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axvline(d_c, color="gray", linestyle=":", label=r"$d_c$ (theory)")
    ax.plot(df["delay_ms"], df["k_theory"], "r--o", markersize=5,
            label=r"Theory $k^*(d)$")
    ax.plot(df["delay_ms"], df["k_empirical"], "b-s", markersize=5,
            label=r"Empirical $\hat{k}^*$")
    ax.set_xlabel("Network delay $d$ (ms)")
    ax.set_ylabel("Optimal draft length $k^*$")
    ax.legend(frameon=True)
    save_figure(fig, OUT_DIR / "fig_h2_phase_transition")
    plt.close(fig)
    print("  Saved fig_h2_phase_transition.pdf, h2_phase_transition.csv")


# ---------------------------------------------------------------------------
# H3/E3+E4 — All-strategy comparison + UCB-SpecStop regret curve
# ---------------------------------------------------------------------------

def _run_strategy_all_prompts(
    strategy_name: str, prompts: list[str], server: str,
    alpha: float, cd: float, cv: float, d_mean: float,
    k_max: int, beta: float, rejection_sampling: bool,
) -> list[dict]:
    rng = np.random.default_rng(42)

    # Strategy setup
    if strategy_name.startswith("fixed"):
        k_fixed = int(strategy_name[5:])
        get_k = lambda t, last_rtt: k_fixed
        update = lambda k, n, a: None
    elif strategy_name == "greedy":
        k_fixed = greedy_policy(alpha, cd, cv, k_max)
        get_k = lambda t, last_rtt: k_fixed
        update = lambda k, n, a: None
    elif strategy_name == "specdec_pp":
        k_fixed = confidence_stop_policy(alpha, p_min=0.3, k_max=k_max)
        get_k = lambda t, last_rtt: k_fixed
        update = lambda k, n, a: None
    elif strategy_name == "oracle":
        k_fixed = compute_kstar(alpha, cd, cv, d_mean, k_max)
        get_k = lambda t, last_rtt: k_fixed
        update = lambda k, n, a: None
    elif strategy_name == "ucb":
        alg = UCBSpecStop(k_max=k_max, beta=beta)
        get_k = lambda t, last_rtt: alg.select_arm(t)
        update = lambda k, n, a: alg.update(k, n, a)
    elif strategy_name == "naive_ucb":
        alg = PerRoundRatioUCB(k_max=k_max, beta=beta)
        get_k = lambda t, last_rtt: alg.select_arm(t)
        update = lambda k, n, a: alg.update(k, n, a)
    elif strategy_name == "exp3":
        alg = EXP3Ratio(k_max=k_max)
        get_k = lambda t, last_rtt: alg.select_arm(t, rng)
        update = lambda k, n, a: alg.update(k, n, a)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    records = []
    for t, prompt in enumerate(prompts, start=1):
        last_rtt = records[-1]["t_comm_ms"] if records else None
        k = get_k(t, last_rtt)
        r = run_one_round(server, prompt, k, rejection_sampling)
        update(k, r["N_t"], r["A_t"])
        r["strategy"] = strategy_name
        r["round"] = t
        records.append(r)

    return records


def experiment_h3_strategy_compare(
    server: str, prompts: list[str], alpha: float, cd: float, cv: float,
    delays: list[int], iface: str, server_user: str, k_max: int, beta: float,
    rejection_sampling: bool, n_regret_rounds: int,
) -> None:
    print("\n=== H3/E3+E4: All-strategy comparison + UCB regret ===")

    strategies = [
        "fixed1", "fixed3", "fixed5", "fixed7", "fixed10",
        "greedy", "specdec_pp", "oracle",
        "naive_ucb", "ucb",
    ]

    all_rows = []
    for delay_ms in delays:
        print(f"\n  Delay = {delay_ms} ms")
        set_netem(iface, delay_ms, server_user=server_user)
        k_oracle = compute_kstar(alpha, cd, cv, float(delay_ms), k_max)
        c_oracle = float(C(k_oracle, float(delay_ms), alpha, cd, cv))

        for strat in strategies:
            print(f"    {strat}...", end=" ", flush=True)
            records = _run_strategy_all_prompts(
                strat, prompts, server, alpha, cd, cv,
                float(delay_ms), k_max, beta, rejection_sampling,
            )
            agg = aggregate(records)
            agg["strategy"] = strat
            agg["delay_ms"] = delay_ms
            agg["theory_oracle_cost"] = c_oracle
            all_rows.append(agg)
            print(f"{agg['mean_cost']:.2f} ms/token")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "h3_strategy_compare.csv", index=False)

    # Table II: per-delay comparison
    pivot = df.pivot(index="strategy", columns="delay_ms", values="mean_cost")
    pivot.to_csv(OUT_DIR / "table_ii_hw_comparison.csv")
    print(f"\n  Table II saved to {OUT_DIR}/table_ii_hw_comparison.csv")

    # Figure: bar chart per delay
    apply_ieee_style()
    fig, axes = plt.subplots(1, len(delays), figsize=(7.2, 2.6), sharey=False)
    for ax, d in zip(axes, delays):
        sub = df[df["delay_ms"] == d].set_index("strategy")
        vals = sub["mean_cost"].to_numpy()
        errs = sub["std_cost"].to_numpy()
        labels = sub.index.tolist()
        short = [s.replace("fixed", "k=").replace("specdec_pp", "Spec++")
                  .replace("naive_ucb", "NvUCB").replace("oracle", "Oracle")
                  .replace("greedy", "Greedy").replace("ucb", "Ours")
                 for s in labels]
        ax.bar(range(len(labels)), vals, yerr=errs, capsize=2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=6)
        ax.set_title(f"$d$={d} ms", fontsize=8)
        if d == delays[0]:
            ax.set_ylabel("ms/token")
    save_figure(fig, OUT_DIR / "fig_h3_strategy_compare")
    plt.close(fig)

    # E3: Regret curve — UCB-SpecStop vs NaiveUCB vs EXP3 at fixed delay
    d_regret = delays[len(delays) // 2]   # middle delay
    print(f"\n  E3: Regret curve at d={d_regret}ms, T={n_regret_rounds} rounds")
    set_netem(iface, d_regret, server_user=server_user)
    k_oracle = compute_kstar(alpha, cd, cv, float(d_regret), k_max)
    c_oracle_val = float(C(k_oracle, float(d_regret), alpha, cd, cv))

    regret_prompts = [prompts[i % len(prompts)] for i in range(n_regret_rounds)]
    regret_data = {}
    for strat in ("ucb", "naive_ucb", "exp3"):
        records = _run_strategy_all_prompts(
            strat, regret_prompts, server, alpha, cd, cv,
            float(d_regret), k_max, beta, rejection_sampling,
        )
        costs = np.array([r["N_t"] / r["A_t"] for r in records])
        cumulative_regret = np.cumsum(costs - c_oracle_val)
        regret_data[strat] = cumulative_regret

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ts = np.arange(1, n_regret_rounds + 1)
    labels = {"ucb": "UCB-SpecStop (Ours)", "naive_ucb": "NaiveUCB (B6)",
              "exp3": "EXP3-Ratio"}
    styles = {"ucb": "b-", "naive_ucb": "r--", "exp3": "g-."}
    for name, cum_reg in regret_data.items():
        ax.plot(ts, cum_reg, styles[name], label=labels[name], linewidth=1.5)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Cumulative regret (ms)")
    ax.legend(frameon=True, fontsize=7)
    save_figure(fig, OUT_DIR / "fig_h3_regret_curve")
    plt.close(fig)
    print("  Saved fig_h3_regret_curve.pdf")

    # Save regret data
    np.savez(OUT_DIR / "h3_regret_data.npz", **{k: v for k, v in regret_data.items()})
    print("  Saved fig_h3_strategy_compare.pdf, table_ii_hw_comparison.csv")


# ---------------------------------------------------------------------------
# H4/E5 — Markov channel: VOI measurement
# ---------------------------------------------------------------------------

def experiment_h4_markov(
    server: str, prompts: list[str], alpha: float, cd: float, cv: float,
    iface: str, server_user: str, k_max: int, beta: float,
    rejection_sampling: bool, n_rounds: int,
    d_good: int = 5, d_bad: int = 80,
    p_good_to_bad: float = 0.1, p_bad_to_good: float = 0.1,
) -> None:
    print(f"\n=== H4/E5: Markov channel VOI (good={d_good}ms, bad={d_bad}ms) ===")

    # We simulate Markov switching via subprocess: launch markov_netem.py
    # as background process. Here we run the switching inline for simplicity.
    rng_markov = np.random.default_rng(0)
    state = "good"  # initial state

    k_good = compute_kstar(alpha, cd, cv, float(d_good), k_max)
    k_bad  = compute_kstar(alpha, cd, cv, float(d_bad), k_max)
    c_oracle_good = float(C(k_good, float(d_good), alpha, cd, cv))
    c_oracle_bad  = float(C(k_bad, float(d_bad), alpha, cd, cv))
    print(f"  Oracle: k_good={k_good}, k_bad={k_bad}")

    # UCB-SpecStop (no state observation — blind to channel state)
    ucb_blind = UCBSpecStop(k_max=k_max, beta=beta)
    # UCB-SpecStop (contextual — two separate UCBs, one per channel state)
    ucb_good = UCBSpecStop(k_max=k_max, beta=beta)
    ucb_bad  = UCBSpecStop(k_max=k_max, beta=beta)

    records_blind = []
    records_ctx   = []

    for t in range(1, n_rounds + 1):
        prompt = prompts[t % len(prompts)]

        # Transition Markov state
        if state == "good":
            if rng_markov.random() < p_good_to_bad:
                state = "bad"
        else:
            if rng_markov.random() < p_bad_to_good:
                state = "good"

        delay_ms = d_good if state == "good" else d_bad
        set_netem(iface, delay_ms, server_user=server_user)

        # Blind strategy
        k_blind = ucb_blind.select_arm(t)
        r_blind = run_one_round(server, prompt, k_blind, rejection_sampling)
        ucb_blind.update(k_blind, r_blind["N_t"], r_blind["A_t"])
        records_blind.append({**r_blind, "state": state})

        # Contextual strategy
        ucb_ctx = ucb_good if state == "good" else ucb_bad
        k_ctx = ucb_ctx.select_arm(t)
        r_ctx = run_one_round(server, prompt, k_ctx, rejection_sampling)
        ucb_ctx.update(k_ctx, r_ctx["N_t"], r_ctx["A_t"])
        records_ctx.append({**r_ctx, "state": state})

        if t % 100 == 0:
            c_blind = np.mean([r["N_t"]/r["A_t"] for r in records_blind[-100:]])
            c_ctx   = np.mean([r["N_t"]/r["A_t"] for r in records_ctx[-100:]])
            print(f"  t={t}: blind={c_blind:.2f}, contextual={c_ctx:.2f}  state={state}")

    c_blind_avg = np.mean([r["N_t"]/r["A_t"] for r in records_blind])
    c_ctx_avg   = np.mean([r["N_t"]/r["A_t"] for r in records_ctx])
    voi = (c_blind_avg - c_ctx_avg) / c_blind_avg * 100.0

    print(f"\n  Blind:       {c_blind_avg:.3f} ms/token")
    print(f"  Contextual:  {c_ctx_avg:.3f} ms/token")
    print(f"  VOI gain:    {voi:.2f}%")

    result = {
        "blind_mean_cost": c_blind_avg,
        "contextual_mean_cost": c_ctx_avg,
        "voi_pct": voi,
        "d_good": d_good, "d_bad": d_bad,
        "p_good_to_bad": p_good_to_bad, "p_bad_to_good": p_bad_to_good,
    }
    (OUT_DIR / "h4_markov_voi.json").write_text(json.dumps(result, indent=2))

    # Regret curves for blind vs contextual
    oracle_cost = (c_oracle_good + c_oracle_bad) / 2.0  # rough mixed oracle
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ts = np.arange(1, n_rounds + 1)
    ax.plot(ts, np.cumsum([r["N_t"]/r["A_t"] - oracle_cost for r in records_blind]),
            "r--", label="UCB-SpecStop (Blind)", linewidth=1.5)
    ax.plot(ts, np.cumsum([r["N_t"]/r["A_t"] - oracle_cost for r in records_ctx]),
            "b-", label="UCB-SpecStop (Contextual)", linewidth=1.5)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Cumulative regret (ms)")
    ax.legend(frameon=True, fontsize=7)
    save_figure(fig, OUT_DIR / "fig_h4_markov_regret")
    plt.close(fig)
    print("  Saved h4_markov_voi.json, fig_h4_markov_regret.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Experiment selection
    parser.add_argument("--exp", choices=["h1", "h2", "h3", "h4", "all"],
                        default="all")
    # Hardware
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--server", default="http://192.168.1.100:8000")
    parser.add_argument("--iface", default="eth0",
                        help="Network interface to configure with tc netem")
    parser.add_argument("--server-user", default="",
                        help="SSH user@host for remote netem (empty = configure locally)")
    # Params
    parser.add_argument("--params", default=None,
                        help="Path to params_measured.json from H0")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--cd", type=float, default=15.0)
    parser.add_argument("--cv", type=float, default=3.0)
    # Prompts
    parser.add_argument("--prompts", default="prompts.txt")
    parser.add_argument("--n-prompts", type=int, default=200)
    # Algorithm
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rejection-sampling", action="store_true", default=True)
    parser.add_argument("--no-rejection-sampling", dest="rejection_sampling",
                        action="store_false")
    # Experiment-specific
    parser.add_argument("--delays", type=int, nargs="+",
                        default=[5, 10, 30, 50, 100],
                        help="Delay values for H1/H3 sweep (ms)")
    parser.add_argument("--n-regret-rounds", type=int, default=1000,
                        help="Rounds for E3 regret curve")
    parser.add_argument("--d-good", type=int, default=5,
                        help="H4 good-state delay (ms)")
    parser.add_argument("--d-bad", type=int, default=80,
                        help="H4 bad-state delay (ms)")
    parser.add_argument("--n-markov-rounds", type=int, default=500,
                        help="Rounds for H4 Markov experiment")
    args = parser.parse_args()

    # Load params
    alpha, cd, cv = args.alpha, args.cd, args.cv
    if args.params:
        with open(args.params) as f:
            p = json.load(f)
        alpha, cd, cv = p["alpha_fit"], p["cd_ms"], p["cv_ms"]
        print(f"[HW] Loaded params: alpha={alpha:.3f}, cd={cd:.2f}ms, cv={cv:.2f}ms")

    load_draft_model(args.draft_model)
    prompts = Path(args.prompts).read_text().strip().split("\n")[: args.n_prompts]
    print(f"[HW] Using {len(prompts)} prompts")

    run_h1 = args.exp in ("h1", "all")
    run_h2 = args.exp in ("h2", "all")
    run_h3 = args.exp in ("h3", "all")
    run_h4 = args.exp in ("h4", "all")

    if run_h1:
        experiment_h1_kstar_sweep(
            args.server, prompts, alpha, cd, cv,
            args.delays, args.iface, args.server_user,
            args.k_max, args.rejection_sampling,
        )
    if run_h2:
        experiment_h2_phase_transition(
            args.server, prompts, alpha, cd, cv,
            args.iface, args.server_user, args.k_max, args.rejection_sampling,
        )
    if run_h3:
        experiment_h3_strategy_compare(
            args.server, prompts, alpha, cd, cv,
            args.delays, args.iface, args.server_user,
            args.k_max, args.beta, args.rejection_sampling, args.n_regret_rounds,
        )
    if run_h4:
        experiment_h4_markov(
            args.server, prompts, alpha, cd, cv,
            args.iface, args.server_user, args.k_max, args.beta,
            args.rejection_sampling, args.n_markov_rounds,
            d_good=args.d_good, d_bad=args.d_bad,
        )


if __name__ == "__main__":
    main()
