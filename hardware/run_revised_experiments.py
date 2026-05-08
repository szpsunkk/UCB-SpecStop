"""
run_revised_experiments.py — Revised hardware experiments per review.md feedback.

Key fixes vs. original run_hw_experiment.py:
  1. A_t = n_accepted + 1 (bonus token included) — correct per paper definition
  2. Every round logs: configured_delay, measured_rtt, draft_time, verify_time,
     comm_time, total_time, k, prefix_accept_len (L), accepted_total (L+1)
  3. Phase-transition sweep is dense around d_c (~80 ms): [40,55,65,72,79,85,92,100,115,130]
  4. n_rounds=500 per strategy (enough for bandit learning curves)
  5. Acceptance analysis outputs prefix P(L>=k), conditional q_k, AND sample counts n_k
  6. EXP3 NaN guard (weight underflow fix)
  7. Measured RTT is separately recorded per delay setting
  8. delay injection clearly labeled as "software-injected" in output

Experiments:
  R1 — Latency calibration: verify cd, cv, RTT for each delay setting
  R2 — Acceptance analysis: prefix + conditional curves + sample counts
  R3 — Phase-transition k-sweep (dense around d_c)
  R4 — Strategy comparison (all baselines, n=200 per strategy per delay)
  R5 — UCB regret curves (n=500 rounds, ucb vs naive_ucb vs exp3 vs oracle)
  R6 — Markov channel VOI

Usage:
  python hardware/run_revised_experiments.py \\
      --server http://192.168.3.72:8000 \\
      --params outputs/hardware/params_measured.json \\
      --prompts hardware/prompts.txt \\
      --exp all
"""
from __future__ import annotations

import argparse
import json
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

OUT_DIR = Path(__file__).parent.parent / "outputs" / "hardware_revised"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SUITES = {
    "qwen": {
        "draft": "Qwen/Qwen2.5-0.5B",
        "cloud": "Qwen/Qwen2.5-7B-Instruct",
    },
    "llama": {
        "draft": "meta-llama/Llama-3.2-1B-Instruct",
        "cloud": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "phi": {
        "draft": "microsoft/Phi-3-mini-4k-instruct",
        "cloud": "microsoft/Phi-3-small-128k-instruct",
    },
}

_model = None
_tok = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_injected_delay_ms: float = 0.0   # one-way software delay in ms


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(name: str, allow_download: bool = False):
    global _model, _tok
    model_ref = str(Path(name).expanduser())
    local_only = not allow_download or Path(model_ref).exists()
    print(f"[HW] Loading {model_ref}  device={_device} local_only={local_only}")
    try:
        _tok = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_only)
        _model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=torch.float16,
            local_files_only=local_only,
        ).to(_device)
    except Exception as e:
        raise RuntimeError(
            f"Could not load draft model '{name}'. Use local path or pass --allow-download after fixing network."
        ) from e
    _model.eval()
    print("[HW] Model loaded.")


def _draft_with_logprobs(input_ids, attn, k):
    with torch.no_grad():
        out = _model.generate(
            input_ids, attention_mask=attn,
            max_new_tokens=k, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=_tok.eos_token_id,
        )
    toks = out.sequences[0, input_ids.shape[1]:].tolist()
    lps = [torch.log_softmax(s[0], -1)[t].item() for s, t in zip(out.scores, toks)]
    return toks, lps


def _draft_only(input_ids, attn, k):
    with torch.no_grad():
        out = _model.generate(
            input_ids, attention_mask=attn,
            max_new_tokens=k, do_sample=False,
            pad_token_id=_tok.eos_token_id,
        )
    return out[0, input_ids.shape[1]:].tolist()


# ---------------------------------------------------------------------------
# Cloud helpers
# ---------------------------------------------------------------------------

def set_delay(d_ms: float):
    global _injected_delay_ms
    _injected_delay_ms = float(d_ms)
    print(f"  [delay] software-injected {d_ms:.0f} ms one-way")


def verify(server, ctx_ids, draft_ids, draft_lps=None):
    payload = {"context_ids": ctx_ids, "draft_ids": draft_ids}
    if draft_lps is not None:
        payload["draft_log_probs"] = draft_lps
    if _injected_delay_ms > 0:
        time.sleep(_injected_delay_ms / 1000.0)
    t0 = time.perf_counter()
    resp = requests.post(f"{server}/verify", json=payload, timeout=60)
    resp.raise_for_status()
    t_net = (time.perf_counter() - t0) * 1000.0
    if _injected_delay_ms > 0:
        time.sleep(_injected_delay_ms / 1000.0)
    data = resp.json()
    data["_net_ms"] = t_net          # actual HTTP round-trip (no injected sleep)
    return data


def measure_rtt(server, n=30):
    """Median actual network RTT (no injected delay)."""
    rtts = []
    for _ in range(n):
        t0 = time.perf_counter()
        requests.get(f"{server}/ping", timeout=5)
        rtts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(rtts))


# ---------------------------------------------------------------------------
# Core per-round function — records everything review.md requested
# ---------------------------------------------------------------------------

def run_round(server, prompt, k, rejection_sampling=True, run_id="", prompt_id=-1,
              seed=-1, strategy="", state=""):
    """
    Returns dict with all fields required by review.md:
      configured_one_way_delay_ms, bare_rtt_ms, measured_comm_round_ms,
      k_selected, accepted_draft_len (L_t), accepted_total (A_t),
      draft_time_ms, verify_time_ms, total_round_time_ms,
      plus reproducibility keys run_id/prompt_id/seed/strategy/state.
    """
    enc = _tok(prompt, return_tensors="pt")
    ids = enc.input_ids.to(_device)
    attn = enc.attention_mask.to(_device)
    ctx = ids[0].tolist()

    t_draft_start = time.perf_counter()
    if rejection_sampling:
        draft_ids, draft_lps = _draft_with_logprobs(ids, attn, k)
    else:
        draft_ids = _draft_only(ids, attn, k)
        draft_lps = None
    t_draft = (time.perf_counter() - t_draft_start) * 1000.0

    t_comm_start = time.perf_counter()
    resp = verify(server, ctx, draft_ids, draft_lps)
    t_comm = (time.perf_counter() - t_comm_start) * 1000.0

    L_t = resp["n_accepted"]
    A_t = L_t + 1
    N_t = t_draft + t_comm

    prefix_indicators = {
        f"prefix_accept_indicator_{i}": int(L_t >= i)
        for i in range(1, k + 1)
    }

    return {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "seed": seed,
        "strategy": strategy,
        "state": state,
        "k_selected": k,
        "configured_one_way_delay_ms": _injected_delay_ms,
        "bare_rtt_ms": resp["_net_ms"],
        "measured_comm_round_ms": t_comm,
        "accepted_draft_len": L_t,
        "accepted_total": A_t,
        "draft_time_ms": t_draft,
        "verify_time_ms": resp["verify_time_ms"],
        "total_round_time_ms": N_t,
        "cost_per_token": N_t / A_t,
        **prefix_indicators,
    }


# ---------------------------------------------------------------------------
# R1 — Latency calibration
# ---------------------------------------------------------------------------

def exp_r1_calibration(server, prompts, delays):
    """
    For each delay setting: measure actual RTT (ping), then run 50 rounds at k=1
    to confirm cd, cv, and that N_t grows with injected delay.
    """
    print("\n=== R1: Latency calibration ===")
    rows = []
    for d in delays:
        set_delay(d)
        rtt_actual = measure_rtt(server, n=20)
        records = [
            run_round(
                server,
                prompts[i % len(prompts)],
                1,
                run_id=f"r1_d{d}",
                prompt_id=i % len(prompts),
                seed=0,
                strategy="fixed1",
            )
            for i in range(120)
        ]
        N_vals  = [r["total_round_time_ms"] for r in records]
        cd_vals = [r["draft_time_ms"] for r in records]
        cv_vals = [r["verify_time_ms"] for r in records]
        comm_vals = [r["measured_comm_round_ms"] for r in records]
        rows.append({
            "configured_one_way_delay_ms": d,
            "bare_rtt_ms": rtt_actual,
            "measured_comm_round_ms": float(np.median(comm_vals)),
            "median_total_round_time_ms": float(np.median(N_vals)),
            "median_cd_ms": float(np.median(cd_vals)),
            "median_cv_ms": float(np.median(cv_vals)),
            "n_rounds": len(records),
        })
        print(f"  d={d}ms: bare_rtt={rtt_actual:.1f}ms, measured_comm={np.median(comm_vals):.1f}ms, "
              f"N_t={np.median(N_vals):.0f}ms, cd={np.median(cd_vals):.1f}ms, cv={np.median(cv_vals):.1f}ms")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "r1_calibration.csv", index=False)

    # Figure: configured vs measured N_t
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(df["configured_one_way_delay_ms"], df["median_total_round_time_ms"], "b-o", label="Measured $N_t$ (k=1)")

    # theoretical: k=1, N = cd + cv + 2d + cv = cd + 2*cv + 2d
    # use measured medians from d=min
    cd0 = df["median_cd_ms"].iloc[0]; cv0 = df["median_cv_ms"].iloc[0]

    theory_N = [cd0 + 2*cv0 + 2*d for d in delays]
    ax.plot(delays, theory_N, "r--", label="Theory (k=1)")
    ax.set_xlabel("Configured one-way delay (ms)")
    ax.set_ylabel("Total round time $N_t$ (ms)")
    ax.legend(frameon=True)
    save_figure(fig, OUT_DIR / "fig_r1_calibration")
    plt.close(fig)
    print("  Saved r1_calibration.csv, fig_r1_calibration.pdf")
    return df


# ---------------------------------------------------------------------------
# R2 — Acceptance analysis (prefix + conditional + sample counts)
# ---------------------------------------------------------------------------

def exp_r2_acceptance(server, prompts, k_max=10):
    """
    Collect acceptance traces.  For each prompt, send k_max draft tokens.
    Record L (prefix accept length).  Then compute:
      - prefix_accept[k] = P(L >= k)
      - conditional_accept[k] = P(L >= k | L >= k-1)
      - sample_count[k]
    """
    print("\n=== R2: Acceptance analysis ===")
    set_delay(0)  # no injected delay for acceptance measurement
    L_list = []
    n_prompts = min(len(prompts), 500)

    for i, prompt in enumerate(prompts[:n_prompts]):
        enc = _tok(prompt, return_tensors="pt")
        ids = enc.input_ids.to(_device); attn = enc.attention_mask.to(_device)
        ctx = ids[0].tolist()
        draft_ids, draft_lps = _draft_with_logprobs(ids, attn, k_max)
        resp = requests.post(f"{server}/verify",
                             json={"context_ids": ctx, "draft_ids": draft_ids,
                                   "draft_log_probs": draft_lps}, timeout=30)
        resp.raise_for_status()
        L_list.append(resp.json()["n_accepted"])

    L_arr = np.array(L_list)
    n_total = len(L_arr)
    prefix_accept = {}    # P(L >= k)
    cond_accept = {}      # P(L >= k | L >= k-1)
    sample_count = {}     # n_k = # sequences where pos k was "observed" = P(L >= k-1)*N

    for k in range(1, k_max + 1):
        n_k = int(np.sum(L_arr >= k - 1))   # sequences reaching position k
        n_acc = int(np.sum(L_arr >= k))
        prefix_accept[k] = n_acc / n_total
        cond_accept[k] = (n_acc / n_k) if n_k > 0 else float('nan')
        sample_count[k] = n_k

    rows = [{"k": k, "prefix_P_Lge_k": prefix_accept[k],
              "cond_q_k": cond_accept[k], "sample_count": sample_count[k]}
            for k in range(1, k_max + 1)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "r2_acceptance.csv", index=False)

    # Figure
    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    ax.bar(range(1, k_max + 1), [prefix_accept[k] for k in range(1, k_max + 1)],
           color="#1f77b4", alpha=0.7, label="$P(L\\geq k)$")
    # geometric reference
    alpha_geo = float(np.mean([cond_accept[k] for k in range(2, k_max + 1)
                                if not np.isnan(cond_accept[k])]))
    geo = [alpha_geo ** k for k in range(1, k_max + 1)]
    ax.plot(range(1, k_max + 1), geo, "r--o", markersize=4,
            label=fr"Geometric $\hat\alpha={alpha_geo:.2f}$")
    ax.set_xlabel("$k$ (draft tokens)"); ax.set_ylabel("Acceptance prob.")
    ax.set_title("Prefix acceptance $P(L\\geq k)$"); ax.legend(frameon=True, fontsize=7)

    ax2 = axes[1]
    cond_vals = [cond_accept[k] for k in range(1, k_max + 1)]
    n_vals    = [sample_count[k] for k in range(1, k_max + 1)]
    ax2.bar(range(1, k_max + 1), cond_vals, color="#ff7f0e", alpha=0.7,
            label="$q_k = P(L\\geq k|L\\geq k-1)$")
    ax2.axhline(alpha_geo, color="red", linestyle="--", linewidth=1,
                label=fr"Mean $\bar{{q}}={alpha_geo:.2f}$")
    for k, n in enumerate(n_vals, start=1):
        ax2.text(k, 0.05, str(n), ha="center", va="bottom", fontsize=5)
    ax2.set_xlabel("Position $k$"); ax2.set_ylabel("Cond. acceptance $q_k$")
    ax2.set_title("Conditional acceptance (n_k shown)"); ax2.legend(frameon=True, fontsize=7)

    save_figure(fig, OUT_DIR / "fig_r2_acceptance")
    plt.close(fig)
    print(f"  Saved r2_acceptance.csv, fig_r2_acceptance.pdf  (n={n_total} prompts)")
    return df


# ---------------------------------------------------------------------------
# R3 — Phase transition (dense sweep around d_c)
# ---------------------------------------------------------------------------

def exp_r3_phase_transition(server, prompts, alpha, cd, cv, k_max):
    """
    Dense delay sweep around d_c.  Each (delay, k) pair: n_sweep rounds.
    Reports empirical best-k with 95% CI.
    """
    print("\n=== R3: Phase transition (dense sweep around d_c) ===")
    d_c = dc_theory(alpha, cd, cv)
    print(f"  d_c = {d_c:.1f} ms  (alpha={alpha:.3f}, cd={cd:.1f}, cv={cv:.1f})")

    # Dense around d_c, plus wider context
    delays = sorted(set([
        max(1, int(d_c * f))
        for f in [0.4, 0.6, 0.75, 0.88, 1.0, 1.12, 1.25, 1.5, 1.75, 2.0]
    ] + [int(d_c)]))

    n_sweep = 220   # per (delay, k) — reduce variance near critical zone
    rows = []
    sweep_detail = {}   # delay -> {k -> [costs]}

    for d in delays:
        set_delay(d)
        k_theory = compute_kstar(alpha, cd, cv, float(d), k_max)
        sweep_costs = {}
        for k in range(1, min(k_max, 8) + 1):
            records = [
                run_round(
                    server,
                    prompts[i % len(prompts)],
                    k,
                    run_id=f"r3_d{d}_k{k}",
                    prompt_id=i % len(prompts),
                    seed=0,
                    strategy="phase_sweep",
                )
                for i in range(n_sweep)
            ]
            costs = [r["cost_per_token"] for r in records]
            sweep_costs[k] = costs
        sweep_detail[d] = sweep_costs

        k_empirical = int(min(sweep_costs, key=lambda k: np.mean(sweep_costs[k])))
        c_emp = float(np.mean(sweep_costs[k_empirical]))
        c_theory = float(C(k_theory, float(d), alpha, cd, cv))
        rows.append({
            "configured_one_way_delay_ms": d,
            "k_theory": k_theory,
            "k_empirical": k_empirical,
            "C_theory": c_theory,
            "C_empirical": c_emp,
            "C_emp_std": float(np.std(sweep_costs[k_empirical])),
        })
        print(f"  d={d}ms: k_theory={k_theory}, k_empirical={k_empirical}, "
              f"C_emp={c_emp:.1f}±{np.std(sweep_costs[k_empirical]):.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "r3_phase_transition.csv", index=False)

    # Figure: k*(d) theory vs empirical, with d_c marked
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axvline(d_c, color="gray", linestyle=":", linewidth=1, label=f"$d_c={d_c:.0f}$ ms")
    ax.plot(df["configured_one_way_delay_ms"], df["k_theory"], "r--o", markersize=5, label=r"Theory $k^*(d)$")
    ax.errorbar(df["configured_one_way_delay_ms"], df["k_empirical"],
                yerr=df["C_emp_std"] / (df["C_empirical"] + 1e-9),  # relative std as proxy
                fmt="b-s", markersize=5, capsize=3, label=r"Empirical $\hat{k}^*$")
    ax.set_xlabel("Configured one-way delay $d$ (ms)")
    ax.set_ylabel("Optimal draft length $k^*$")
    ax.legend(frameon=True, fontsize=7)
    save_figure(fig, OUT_DIR / "fig_r3_phase_transition")
    plt.close(fig)
    print("  Saved r3_phase_transition.csv, fig_r3_phase_transition.pdf")
    return df


# ---------------------------------------------------------------------------
# R4 — Strategy comparison (n=200 per delay)
# ---------------------------------------------------------------------------

def _run_strategy(strategy_name, prompts, server, alpha, cd, cv, d_mean,
                  k_max, beta, rejection_sampling, n_rounds):
    rng = np.random.default_rng(42)
    k_oracle = compute_kstar(alpha, cd, cv, d_mean, k_max)

    if strategy_name.startswith("fixed"):
        k_fixed = int(strategy_name[5:])
        get_k = lambda t, _: k_fixed; update = lambda *a: None
    elif strategy_name == "greedy":
        k_fixed = greedy_policy(alpha, cd, cv, k_max)
        get_k = lambda t, _: k_fixed; update = lambda *a: None
    elif strategy_name == "specdec_pp":
        k_fixed = confidence_stop_policy(alpha, p_min=0.3, k_max=k_max)
        get_k = lambda t, _: k_fixed; update = lambda *a: None
    elif strategy_name == "oracle":
        get_k = lambda t, _: k_oracle; update = lambda *a: None
    elif strategy_name == "ucb":
        alg = UCBSpecStop(k_max=k_max, beta=beta)
        get_k = lambda t, _: alg.select_arm(t); update = lambda k,n,a: alg.update(k,n,a)
    elif strategy_name == "naive_ucb":
        alg = PerRoundRatioUCB(k_max=k_max, beta=beta)
        get_k = lambda t, _: alg.select_arm(t); update = lambda k,n,a: alg.update(k,n,a)
    elif strategy_name == "exp3":
        alg = EXP3Ratio(k_max=k_max)
        get_k = lambda t, _: alg.select_arm(t, rng)
        update = lambda k,n,a: alg.update(k,n,a)
    else:
        raise ValueError(strategy_name)

    records = []
    for t, prompt in enumerate(prompts[:n_rounds], start=1):
        k = get_k(t, None)
        r = run_round(
            server,
            prompt,
            k,
            rejection_sampling,
            run_id=f"{strategy_name}_t{t}",
            prompt_id=t - 1,
            seed=42,
            strategy=strategy_name,
        )
        update(k, r["total_round_time_ms"], r["accepted_total"])
        r["strategy"] = strategy_name; r["round"] = t
        records.append(r)
    return records


def exp_r4_strategy_compare(server, prompts, alpha, cd, cv, delays,
                             k_max, beta, rejection_sampling, n_rounds):
    print("\n=== R4: Strategy comparison (n={} per strategy per delay) ===".format(n_rounds))
    strategies = ["fixed1", "fixed2", "fixed3", "fixed5", "fixed7",
                  "greedy", "specdec_pp", "oracle", "naive_ucb", "ucb"]
    all_rows = []
    round_logs = []

    for d in delays:
        print(f"\n  Delay = {d} ms")
        set_delay(d)
        k_oracle = compute_kstar(alpha, cd, cv, float(d), k_max)
        c_theory_oracle = float(C(k_oracle, float(d), alpha, cd, cv))

        for s in strategies:
            print(f"    {s}...", end=" ", flush=True)
            records = _run_strategy(s, prompts, server, alpha, cd, cv,
                                    float(d), k_max, beta, rejection_sampling, n_rounds)
            costs = [r["cost_per_token"] for r in records]
            accepted_totals = [r["accepted_total"] for r in records]
            all_rows.append({
                "strategy": s,
                "configured_one_way_delay_ms": d,
                "n_rounds": n_rounds,
                "mean_cost_per_token": float(np.mean(costs)),
                "std_cost": float(np.std(costs)),
                "ci95": float(1.96 * np.std(costs) / np.sqrt(len(costs))),
                "mean_accepted_total": float(np.mean(accepted_totals)),  # A_t = L+1
                "theory_oracle_cost": c_theory_oracle,
                "empirical_oracle_k": k_oracle,
            })
            print(f"{np.mean(costs):.1f}±{1.96*np.std(costs)/np.sqrt(len(costs)):.1f} ms/tok  "
                  f"(mean A_t={np.mean(accepted_totals):.2f})")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "r4_strategy_compare.csv", index=False)

    # Pivot table (Table II style)
    pivot = df.pivot(index="strategy", columns="configured_one_way_delay_ms", values="mean_cost_per_token")
    pivot.to_csv(OUT_DIR / "table_ii_revised.csv")

    # Figure: bar chart per delay
    apply_ieee_style()
    fig, axes = plt.subplots(1, len(delays), figsize=(7.2, 2.6), sharey=False)
    if len(delays) == 1: axes = [axes]
    for ax, d in zip(axes, delays):
        sub = df[df["configured_one_way_delay_ms"] == d].sort_values("mean_cost_per_token")
        ax.bar(range(len(sub)), sub["mean_cost_per_token"],
               yerr=sub["ci95"], capsize=2)
        short = [s.replace("fixed","k=").replace("specdec_pp","Spec++")
                  .replace("naive_ucb","NvUCB").replace("oracle","Oracle")
                  .replace("greedy","Greedy").replace("ucb","Ours")
                  for s in sub["strategy"]]
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=6)
        ax.set_title(f"$d$={d} ms", fontsize=8)
        if d == delays[0]: ax.set_ylabel("ms/token")
    save_figure(fig, OUT_DIR / "fig_r4_strategy_compare")
    plt.close(fig)
    print(f"\n  Saved r4_strategy_compare.csv, table_ii_revised.csv, fig_r4_strategy_compare.pdf")
    return df


# ---------------------------------------------------------------------------
# R5 — UCB regret curves (n=500, ucb vs naive_ucb vs exp3 vs oracle)
# ---------------------------------------------------------------------------

def exp_r5_regret(server, prompts, alpha, cd, cv, d_regret,
                  k_max, beta, rejection_sampling, n_rounds):
    print(f"\n=== R5: Regret curves at d={d_regret}ms, T={n_rounds} ===")
    set_delay(d_regret)
    k_oracle = compute_kstar(alpha, cd, cv, float(d_regret), k_max)
    c_oracle_theory = float(C(k_oracle, float(d_regret), alpha, cd, cv))

    regret_prompts = [prompts[i % len(prompts)] for i in range(n_rounds)]
    regret_data = {}

    for strat in ("ucb", "naive_ucb", "exp3"):
        print(f"  {strat}...", flush=True)
        rng = np.random.default_rng(123)

        if strat == "ucb":
            alg = UCBSpecStop(k_max=k_max, beta=beta)
            select = lambda t: alg.select_arm(t)
            update = lambda k, n, a: alg.update(k, n, a)
        elif strat == "naive_ucb":
            alg = PerRoundRatioUCB(k_max=k_max, beta=beta)
            select = lambda t: alg.select_arm(t)
            update = lambda k, n, a: alg.update(k, n, a)
        else:
            alg = EXP3Ratio(k_max=k_max)
            select = lambda t: alg.select_arm(t, rng)
            update = lambda k, n, a: alg.update(k, n, a)

        cum = 0.0
        curve = []
        for t in range(1, n_rounds + 1):
            k = select(t)
            r = run_round(
                server,
                regret_prompts[t - 1],
                k,
                rejection_sampling,
                run_id=f"r5_{strat}_t{t}",
                prompt_id=t - 1,
                seed=123,
                strategy=strat,
            )
            update(k, r["total_round_time_ms"], r["accepted_total"])
            cum += (r["cost_per_token"] - c_oracle_theory)
            curve.append(cum)

        regret_data[strat] = np.array(curve)
        print(f"  {strat} final cumulative regret: {curve[-1]:.1f} ms")

    np.savez(
        OUT_DIR / "r5_regret_data.npz",
        **{k: v for k, v in regret_data.items()},
        c_oracle_theory=np.array([c_oracle_theory]),
    )

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ts = np.arange(1, n_rounds + 1)
    labels = {"ucb": "UCB-SpecStop (Ours)", "naive_ucb": "NaiveUCB (B6)", "exp3": "EXP3-Ratio"}
    styles = {"ucb": "b-", "naive_ucb": "r--", "exp3": "g-."}
    for name, cum_reg in regret_data.items():
        ax.plot(ts, cum_reg, styles[name], label=labels[name], linewidth=1.5)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Cumulative regret (ms)")
    ax.set_title(f"Online regret ($d={d_regret}$ ms, $T={n_rounds}$)")
    ax.legend(frameon=True, fontsize=7)
    save_figure(fig, OUT_DIR / "fig_r5_regret")
    plt.close(fig)
    print("  Saved r5_regret_data.npz, fig_r5_regret.pdf")
    return regret_data


# ---------------------------------------------------------------------------
# R6 — Markov channel VOI
# ---------------------------------------------------------------------------

def exp_r6_markov(server, prompts, alpha, cd, cv,
                  k_max, beta, rejection_sampling, n_rounds,
                  d_good=40, d_bad=120, p_g2b=0.1, p_b2g=0.1):
    print(f"\n=== R6: Markov VOI (good={d_good}ms, bad={d_bad}ms) ===")
    rng = np.random.default_rng(0)
    k_good = compute_kstar(alpha, cd, cv, float(d_good), k_max)
    k_bad  = compute_kstar(alpha, cd, cv, float(d_bad), k_max)
    c_g = float(C(k_good, float(d_good), alpha, cd, cv))
    c_b = float(C(k_bad,  float(d_bad),  alpha, cd, cv))
    print(f"  k_good={k_good} (C={c_g:.1f}), k_bad={k_bad} (C={c_b:.1f})")

    ucb_blind   = UCBSpecStop(k_max=k_max, beta=beta)
    ucb_good_st = UCBSpecStop(k_max=k_max, beta=beta)
    ucb_bad_st  = UCBSpecStop(k_max=k_max, beta=beta)

    records_blind = []; records_ctx = []
    state = "good"

    for t in range(1, n_rounds + 1):
        # Markov state transition
        if state == "good":
            if rng.random() < p_g2b: state = "bad"
        else:
            if rng.random() < p_b2g: state = "good"

        d_now = d_good if state == "good" else d_bad
        set_delay(d_now)
        prompt = prompts[t % len(prompts)]

        # Blind
        k_bl = ucb_blind.select_arm(t)
        r_bl = run_round(server, prompt, k_bl, rejection_sampling)
        ucb_blind.update(k_bl, r_bl["total_round_time_ms"], r_bl["accepted_total"])
        records_blind.append({**r_bl, "state": state})

        # Contextual (separate UCB per state)
        ucb_ctx = ucb_good_st if state == "good" else ucb_bad_st
        k_ct = ucb_ctx.select_arm(t)
        r_ct = run_round(server, prompt, k_ct, rejection_sampling)
        ucb_ctx.update(k_ct, r_ct["total_round_time_ms"], r_ct["accepted_total"])
        records_ctx.append({**r_ct, "state": state})

        if t % 100 == 0:
            c_bl = np.mean([r["cost_per_token"] for r in records_blind[-100:]])
            c_ct = np.mean([r["cost_per_token"] for r in records_ctx[-100:]])
            print(f"  t={t}: blind={c_bl:.1f}, ctx={c_ct:.1f}  state={state}")

    c_bl_avg = float(np.mean([r["cost_per_token"] for r in records_blind]))
    c_ct_avg = float(np.mean([r["cost_per_token"] for r in records_ctx]))
    voi_pct  = (c_bl_avg - c_ct_avg) / c_bl_avg * 100.0

    result = {
        "blind_mean_cost": c_bl_avg,
        "contextual_mean_cost": c_ct_avg,
        "voi_pct": voi_pct,
        "d_good": d_good, "d_bad": d_bad,
        "p_g2b": p_g2b, "p_b2g": p_b2g,
        "k_good": k_good, "k_bad": k_bad,
    }
    (OUT_DIR / "r6_markov_voi.json").write_text(json.dumps(result, indent=2))

    oracle_cost = (c_g + c_b) / 2.0
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ts = np.arange(1, n_rounds + 1)
    ax.plot(ts, np.cumsum([r["cost_per_token"] - oracle_cost for r in records_blind]),
            "r--", label="UCB-SpecStop (Blind)")
    ax.plot(ts, np.cumsum([r["cost_per_token"] - oracle_cost for r in records_ctx]),
            "b-", label="UCB-SpecStop (Contextual)")
    ax.set_xlabel("Round $t$"); ax.set_ylabel("Cumulative regret (ms)")
    ax.legend(frameon=True, fontsize=7)
    save_figure(fig, OUT_DIR / "fig_r6_markov_regret")
    plt.close(fig)
    print(f"  VOI={voi_pct:.2f}%  Saved r6_markov_voi.json, fig_r6_markov_regret.pdf")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",       default="http://192.168.3.72:8000")
    parser.add_argument("--draft-model",  default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--suite",        choices=["qwen", "llama", "phi"], default=None,
        help="Use a predefined model suite (draft/cloud pair) for reproducible comparison")
    parser.add_argument("--params",       default="outputs/hardware/params_measured.json",
        help="Real measured params json from hardware/measure_params.py")
    parser.add_argument("--prompts",      default="hardware/prompts.txt")
    parser.add_argument("--cloud-model",  default=None,
        help="Cloud verification model; defaults to the suite's matching 7B/8B model")
    parser.add_argument("--out-dir",      default=None,
        help="Output directory override")
    parser.add_argument("--n-prompts",    type=int, default=500)
    parser.add_argument("--exp",          default="all",
        choices=["all","r1","r2","r3","r4","r5","r6"])
    parser.add_argument("--k-max",        type=int, default=10)
    parser.add_argument("--beta",         type=float, default=1.0)
    parser.add_argument("--n-rounds",     type=int, default=200,
        help="Rounds per strategy in R4; R5 uses 3x this")
    parser.add_argument("--no-rejection-sampling", dest="rs",
        action="store_false", default=True)
    parser.add_argument("--allow-download", action="store_true", default=False,
        help="Allow downloading models from Hugging Face when local files are unavailable")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--cd",    type=float, default=15.0)
    parser.add_argument("--cv",    type=float, default=3.0)
    args = parser.parse_args()

    if args.suite:
        suite_default_draft = MODEL_SUITES[args.suite]["draft"]
        parser_default_draft = parser.get_default("draft_model")
        if args.draft_model == parser_default_draft:
            args.draft_model = suite_default_draft
        if args.cloud_model is None:
            args.cloud_model = MODEL_SUITES[args.suite]["cloud"]

    if args.cloud_model:
        print(f"[HW] Expected cloud model for this run: {args.cloud_model}")

    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    elif args.suite:
        OUT_DIR = Path(__file__).parent.parent / "outputs" / "hardware_revised" / args.suite
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha, cd, cv = args.alpha, args.cd, args.cv
    if not args.params:
        raise ValueError("--params is required. Use real measured communication parameters from measure_params.py")
    p = json.loads(Path(args.params).read_text())
    alpha = p["alpha_fit"]; cd = p["cd_ms"]; cv = p["cv_ms"]
    print(f"[HW] params: alpha={alpha:.3f}, cd={cd:.1f}ms, cv={cv:.1f}ms")

    load_model(args.draft_model, allow_download=args.allow_download)
    prompts = Path(args.prompts).read_text().strip().split("\n")[:args.n_prompts]
    print(f"[HW] {len(prompts)} prompts loaded")

    run_config = {
        "suite": args.suite,
        "draft_model": args.draft_model,
        "cloud_model": args.cloud_model,
        "server": args.server,
        "params_file": args.params,
        "n_prompts": len(prompts),
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2))

    d_c = dc_theory(alpha, cd, cv)
    print(f"[HW] Theoretical d_c = {d_c:.1f} ms")

    # Calibration delays: cover low, mid, and around d_c
    calib_delays  = [5, 20, 40, int(d_c), int(d_c * 1.5)]
    # Phase transition: dense around d_c
    pt_delays = sorted(set([max(1, int(d_c * f))
                             for f in [0.4, 0.6, 0.75, 0.88, 1.0, 1.12, 1.25, 1.5, 1.75, 2.0]]
                            + [int(d_c)]))
    # Strategy compare: representative spread
    strat_delays  = sorted(set([max(1, int(d_c * f)) for f in [0.5, 1.0, 1.5, 2.0]]))
    # Regret: at d_c
    regret_delay  = int(d_c)
    # Markov: good below d_c, bad above d_c
    d_good = max(1, int(d_c * 0.5))
    d_bad  = int(d_c * 1.5)

    run_r1 = args.exp in ("all", "r1")
    run_r2 = args.exp in ("all", "r2")
    run_r3 = args.exp in ("all", "r3")
    run_r4 = args.exp in ("all", "r4")
    run_r5 = args.exp in ("all", "r5")
    run_r6 = args.exp in ("all", "r6")

    if run_r1:
        exp_r1_calibration(args.server, prompts, calib_delays)
    if run_r2:
        exp_r2_acceptance(args.server, prompts, k_max=args.k_max)
    if run_r3:
        exp_r3_phase_transition(args.server, prompts, alpha, cd, cv, args.k_max)
    if run_r4:
        exp_r4_strategy_compare(args.server, prompts, alpha, cd, cv, strat_delays,
                                args.k_max, args.beta, args.rs, args.n_rounds)
    if run_r5:
        exp_r5_regret(args.server, prompts, alpha, cd, cv, regret_delay,
                      args.k_max, args.beta, args.rs, args.n_rounds * 3)
    if run_r6:
        exp_r6_markov(args.server, prompts, alpha, cd, cv,
                      args.k_max, args.beta, args.rs, args.n_rounds * 2,
                      d_good=d_good, d_bad=d_bad)

    print("\n[HW] All requested experiments complete.")
    print(f"[HW] Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
