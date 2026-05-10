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

def load_model(
    name: str,
    allow_download: bool = False,
    offload_folder: str | None = None,
    max_cpu_mem_gb: float | None = None,
    max_gpu_mem_gb: float | None = None,
):
    global _model, _tok
    model_ref = str(Path(name).expanduser())
    local_only = not allow_download or Path(model_ref).exists()
    print(f"[HW] Loading {model_ref}  device={_device} local_only={local_only}"
          f"{'  offload=' + offload_folder if offload_folder else ''}")

    kwargs = dict(
        torch_dtype=torch.float16,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if offload_folder:
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
        max_mem = {}
        if max_cpu_mem_gb is not None:
            max_mem["cpu"] = f"{max_cpu_mem_gb}GiB"
        if torch.cuda.is_available() and max_gpu_mem_gb is not None:
            max_mem[0] = f"{max_gpu_mem_gb}GiB"
        kwargs.update(
            device_map="auto",
            offload_folder=offload_folder,
            offload_state_dict=True,
            max_memory=max_mem or None,
        )
    else:
        kwargs.update(device_map={"": _device})

    try:
        _tok = AutoTokenizer.from_pretrained(
            model_ref, local_files_only=local_only, trust_remote_code=True
        )
        _model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
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


def verify(server, ctx_ids, draft_ids, draft_lps=None, seed=None):
    payload = {"context_ids": ctx_ids, "draft_ids": draft_ids}
    if draft_lps is not None:
        payload["draft_log_probs"] = draft_lps
    if seed is not None:
        payload["seed"] = int(seed)
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
              seed=-1, strategy="", state="", verify_seed=None):
    """
    Returns dict with all fields required by review.md:
      configured_one_way_delay_ms, bare_rtt_ms, measured_comm_round_ms,
      k_selected, accepted_draft_len (L_t), accepted_total (A_t),
      draft_time_ms, verify_time_ms, total_round_time_ms,
      plus reproducibility keys run_id/prompt_id/seed/strategy/state.

    `verify_seed`: when set, forwarded to /verify so rejection sampling and
    bonus sampling are deterministic for this (prompt, draft_ids) pair.
    Use the same seed across strategies for the same prompt to enable paired
    comparison (R4) and remove cloud-side sampling noise.
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
    resp = verify(server, ctx, draft_ids, draft_lps, seed=verify_seed)
    t_comm = (time.perf_counter() - t_comm_start) * 1000.0

    L_t = resp["n_accepted"]
    A_t = L_t + 1
    N_t = t_draft + t_comm

    # Server-side timing breakdown (review B1). Backwards-compatible: old
    # cloud_server doesn't set these fields; .get falls back to 0.0.
    server_recv_ms = float(resp.get("server_recv_to_verify_start_ms", 0.0))
    verify_split_ms = float(resp.get("verify_split_ms", 0.0))
    pack_split_ms = float(resp.get("pack_split_ms", 0.0))
    server_total_ms = server_recv_ms + verify_split_ms + pack_split_ms
    if server_total_ms <= 0.0:
        # Old server: fall back to total verify_time_ms for the subtraction
        server_total_ms = float(resp["verify_time_ms"])

    # Review D#4: oracle formulas must be fed the *measured* communication
    # delay, not the configured injection. d_eff is half of (round-trip minus
    # server-side processing), which is the one-way network term the closed-
    # form cost model assumes.
    d_eff_one_way_ms = max(0.0, (t_comm - server_total_ms) / 2.0)

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
        "server_recv_to_verify_start_ms": server_recv_ms,
        "server_verify_split_ms": verify_split_ms,
        "server_pack_split_ms": pack_split_ms,
        "server_total_ms": server_total_ms,
        "d_eff_one_way_ms": d_eff_one_way_ms,
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

def _state_path() -> Path:
    return OUT_DIR / "calibrated_state.json"


def _load_state() -> dict:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    _state_path().write_text(json.dumps(state, indent=2))


def cost_ratio_of_sums(records: list) -> float:
    """
    Estimator the paper actually optimises (review D#1, B3):
        Ĉ(k,d) = Σ_r T_r(k,d) / Σ_r A_r(k,d)
    NOT mean(T_r/A_r). For a fair strategy comparison every cost figure must
    use this aggregator; cost_per_token (mean of ratios) is kept only as a
    sanity column.
    """
    if not records:
        return float("nan")
    sum_n = sum(r["total_round_time_ms"] for r in records)
    sum_a = sum(r["accepted_total"] for r in records)
    return float(sum_n / sum_a) if sum_a > 0 else float("nan")


def mean_d_eff(records: list) -> float:
    if not records:
        return float("nan")
    return float(np.mean([r.get("d_eff_one_way_ms", 0.0) for r in records]))


def exp_r1_calibration(server, prompts, delays, ks=(1, 2, 3, 5, 7, 10), n_per_cell=300):
    """
    Multi-k system calibration (review §B1).

    For each (delay, k): n_per_cell rounds of `run_round`. Logs per-cell
    medians of cd_per_token, cv_per_token, server_total, comm round, d_eff.
    Aggregates into per-k cd/cv curves and a delay-vs-comm-round table.

    Sanity check (review §B1 last item): T_e2e ≈ T_draft + T_rpc must hold per
    round; if absolute relative error > 5% on cell median the row is flagged.
    """
    print(f"\n=== R1: Multi-k calibration (k={list(ks)}, delays={list(delays)}, n={n_per_cell}/cell) ===")
    rows = []

    for d in delays:
        set_delay(d)
        rtt_actual = measure_rtt(server, n=20)

        for k in ks:
            cd_per_tok_vals = []
            cv_per_tok_vals = []
            comm_vals = []
            server_total_vals = []
            d_eff_vals = []
            n_total_vals = []
            cost_per_round_for_sanity = []
            sum_n = 0.0
            sum_a = 0.0
            for i in range(n_per_cell):
                r = run_round(
                    server,
                    prompts[i % len(prompts)],
                    k,
                    run_id=f"r1_d{d}_k{k}",
                    prompt_id=i % len(prompts),
                    seed=0,
                    strategy=f"calib_k{k}",
                    verify_seed=50_000 + (i % len(prompts)),
                )
                # cd / cv per token at this k
                cd_per_tok_vals.append(r["draft_time_ms"] / max(k, 1))
                cv_per_tok_vals.append(r["server_verify_split_ms"] / max(k + 1, 1)
                                       if r["server_verify_split_ms"] > 0
                                       else r["verify_time_ms"] / max(k + 1, 1))
                comm_vals.append(r["measured_comm_round_ms"])
                server_total_vals.append(r["server_total_ms"])
                d_eff_vals.append(r["d_eff_one_way_ms"])
                n_total_vals.append(r["total_round_time_ms"])
                cost_per_round_for_sanity.append(r["cost_per_token"])
                sum_n += r["total_round_time_ms"]
                sum_a += r["accepted_total"]

            # T_e2e = T_draft + T_rpc identity check (review §B1)
            # We measured t_draft and t_comm independently; total_round_time_ms
            # is their sum by construction, so the identity holds exactly per
            # round. The diagnostic value here is whether server_total_ms is
            # meaningfully smaller than measured_comm_round_ms (i.e. there is
            # detectable network time), and that comm-round - server_total - 2d
            # ≈ 0 — see d_eff column.
            rows.append({
                "configured_one_way_delay_ms": d,
                "k": k,
                "n_rounds": n_per_cell,
                "bare_rtt_ms": rtt_actual,
                "median_cd_per_token_ms": float(np.median(cd_per_tok_vals)),
                "median_cv_per_token_ms": float(np.median(cv_per_tok_vals)),
                "median_comm_round_ms": float(np.median(comm_vals)),
                "median_server_total_ms": float(np.median(server_total_vals)),
                "median_d_eff_one_way_ms": float(np.median(d_eff_vals)),
                "median_total_round_time_ms": float(np.median(n_total_vals)),
                "cost_ratio_of_sums": float(sum_n / sum_a) if sum_a > 0 else float("nan"),
                "mean_cost_per_token_sanity": float(np.mean(cost_per_round_for_sanity)),
            })
            print(f"  d={d}ms k={k}: comm={np.median(comm_vals):.1f}, "
                  f"server={np.median(server_total_vals):.1f}, "
                  f"d_eff={np.median(d_eff_vals):.1f}, "
                  f"cd/tok={np.median(cd_per_tok_vals):.2f}, "
                  f"cv/tok={np.median(cv_per_tok_vals):.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "r1_calibration.csv", index=False)

    # Per-k summary (review B1 wants k-dependent cd, cv)
    cd_per_k = {int(k): float(df[df.k == k]["median_cd_per_token_ms"].median()) for k in ks}
    cv_per_k = {int(k): float(df[df.k == k]["median_cv_per_token_ms"].median()) for k in ks}
    cd_calibrated = float(np.mean(list(cd_per_k.values())))
    cv_calibrated = float(np.mean(list(cv_per_k.values())))
    rtt_baseline = float(df.loc[df["configured_one_way_delay_ms"].idxmin(), "bare_rtt_ms"])

    state = _load_state()
    state.update({
        "cd_calibrated_ms": cd_calibrated,
        "cv_calibrated_ms": cv_calibrated,
        "cd_per_k_calibrated": cd_per_k,
        "cv_per_k_calibrated": cv_per_k,
        "bare_rtt_baseline_ms": rtt_baseline,
        "r1_delays": list(map(float, delays)),
        "r1_ks": list(map(int, ks)),
    })
    _save_state(state)
    print(f"  -> calibrated cd={cd_calibrated:.2f}ms/tok (per-k: {cd_per_k}), "
          f"cv={cv_calibrated:.2f}ms/tok (per-k: {cv_per_k})")

    # Figure: configured delay vs measured comm round, faceted by k
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for k in ks:
        sub = df[df.k == k].sort_values("configured_one_way_delay_ms")
        ax.plot(sub["configured_one_way_delay_ms"], sub["median_comm_round_ms"],
                marker="o", markersize=4, linewidth=1.0, label=f"k={k}")
    ax.set_xlabel("Configured one-way delay (ms)")
    ax.set_ylabel("Median comm round (ms)")
    ax.legend(frameon=True, fontsize=6, ncol=2)
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

    # Persist empirical prefix curve so R3/R6 can derive empirical-B(k) without
    # re-running R2 or assuming geometric acceptance (review M1, M4).
    state = _load_state()
    state["prefix_P_Lge_k"] = {str(k): float(prefix_accept[k]) for k in range(1, k_max + 1)}
    state["cond_q_k"] = {
        str(k): (None if np.isnan(cond_accept[k]) else float(cond_accept[k]))
        for k in range(1, k_max + 1)
    }
    state["q1_observed"] = float(cond_accept[1]) if not np.isnan(cond_accept[1]) else None
    state["alpha_geo_kge2"] = float(np.mean([
        cond_accept[k] for k in range(2, k_max + 1)
        if not np.isnan(cond_accept[k])
    ]))
    state["n_acceptance_prompts"] = int(n_total)
    _save_state(state)

    # Figure (M2): keep the geometric reference but make its scope explicit.
    # Left: prefix P(L>=k) — bars only, NO geometric overlay (used to mislead).
    # Right: conditional q_k — geometric mean is computed over k>=2 and the
    # k=1 outlier is annotated, matching what the paper's Assumption 1 covers.
    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    alpha_geo = float(np.mean([cond_accept[k] for k in range(2, k_max + 1)
                                if not np.isnan(cond_accept[k])]))

    ax = axes[0]
    ax.bar(range(1, k_max + 1), [prefix_accept[k] for k in range(1, k_max + 1)],
           color="#1f77b4", alpha=0.7, label="$P(L\\geq k)$ (empirical)")
    ax.set_xlabel("$k$ (draft tokens)"); ax.set_ylabel("Prefix acceptance")
    ax.set_title("Prefix acceptance $P(L\\geq k)$"); ax.legend(frameon=True, fontsize=7)

    ax2 = axes[1]
    cond_vals = [cond_accept[k] for k in range(1, k_max + 1)]
    n_vals    = [sample_count[k] for k in range(1, k_max + 1)]
    ax2.bar(range(1, k_max + 1), cond_vals, color="#ff7f0e", alpha=0.7,
            label="$q_k = P(L\\geq k|L\\geq k-1)$")
    ax2.axhline(alpha_geo, color="red", linestyle="--", linewidth=1,
                label=fr"Mean over $k\geq 2$: $\bar q={alpha_geo:.2f}$")
    if not np.isnan(cond_accept[1]):
        ax2.annotate(f"$q_1={cond_accept[1]:.2f}$ (outlier)",
                     xy=(1, cond_accept[1]),
                     xytext=(2.3, max(0.05, cond_accept[1] - 0.1)),
                     fontsize=6, color="darkred",
                     arrowprops=dict(arrowstyle="-", color="darkred", lw=0.6))
    for k, n in enumerate(n_vals, start=1):
        ax2.text(k, 0.04, str(n), ha="center", va="bottom", fontsize=5)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Position $k$"); ax2.set_ylabel("Cond. acceptance $q_k$")
    ax2.set_title("Conditional acceptance ($n_k$ shown)"); ax2.legend(frameon=True, fontsize=7)

    save_figure(fig, OUT_DIR / "fig_r2_acceptance")
    plt.close(fig)
    print(f"  Saved r2_acceptance.csv, fig_r2_acceptance.pdf  (n={n_total} prompts)")
    return df


# ---------------------------------------------------------------------------
# R3 — Phase transition (dense sweep around d_c)
# ---------------------------------------------------------------------------

def _empirical_kstar(prefix_arr: np.ndarray, d: float, cd: float, cv: float, k_max: int) -> tuple[int, float]:
    """
    Pick k that minimises C_emp(k,d) using empirical prefix:
      B_emp(k) = 1 + sum_{j=1..k} P(L>=j)   (1 is the bonus token)
      C_emp(k,d) = (k*(cd+cv) + 2d + cv) / B_emp(k)
    Returns (k*, C_emp(k*)).
    """
    ks = np.arange(1, k_max + 1)
    pref = prefix_arr[:k_max]
    B_emp = 1.0 + np.cumsum(pref)
    numer = ks * (cd + cv) + 2.0 * d + cv
    costs = numer / B_emp
    idx = int(np.argmin(costs))
    return int(ks[idx]), float(costs[idx])


def exp_r3_phase_transition(server, prompts, alpha, cd, cv, k_max,
                             delays_override=None, ks_override=None,
                             n_per_cell=300):
    """
    R3 hardware cost curve and empirical oracle (review §B3).

    For each delay × k cell, runs n_per_cell rounds against the SAME prompt
    pool and the SAME verify_seed schedule, so fixed1 and fixed10 see paired
    cloud-side RNG. Cost is reported as ratio-of-sums (review D#1).

    Three k* curves are produced (review B4):
      - k_theory_geometric:    closed-form C(k,d,alpha,cd,cv) with d = configured
      - k_theory_calibrated:   closed-form with d = measured d_eff median
      - k_theory_empirical:    C with B_emp(k) (R2 prefix) and d = d_eff
      - k_empirical_oracle:    argmin_k Ĉ(k,d) on the actual sweep data
    """
    print("\n=== R3: Phase transition / hardware cost curve ===")

    state = _load_state()
    cd_used = state.get("cd_calibrated_ms", cd)
    cv_used = state.get("cv_calibrated_ms", cv)
    if "cd_calibrated_ms" in state:
        print(f"  using R1-calibrated cd={cd_used:.2f}ms/tok, cv={cv_used:.2f}ms/tok")
    else:
        print(f"  [warn] no R1 calibration, using params cd={cd:.2f}, cv={cv:.2f}")

    prefix_dict = state.get("prefix_P_Lge_k")
    if prefix_dict:
        prefix_arr = np.array([prefix_dict[str(k)] for k in range(1, k_max + 1)])
        have_empirical_prefix = True
    else:
        prefix_arr = np.array([alpha ** k for k in range(1, k_max + 1)])
        have_empirical_prefix = False
        print("  [warn] no R2 prefix; falling back to geometric P(L>=k)=alpha^k")

    d_c_geom = dc_theory(alpha, cd_used, cv_used)
    delays = list(delays_override) if delays_override is not None else \
             [0, 5, 20, 40, 55, 83, 111, 150]   # review §B3
    ks = list(ks_override) if ks_override is not None else \
         [1, 2, 3, 4, 5, 7, 10]                  # review §D#3
    print(f"  d_c (geometric, calibrated) = {d_c_geom:.1f} ms")
    print(f"  delays = {delays}, ks = {ks}, n/cell = {n_per_cell}")

    rows = []
    cell_records = {}   # (d, k) -> list of round records (for paired Ĉ)

    for d in delays:
        set_delay(d)
        for k in ks:
            recs = [
                run_round(
                    server,
                    prompts[i % len(prompts)],
                    k,
                    run_id=f"r3_d{d}_k{k}",
                    prompt_id=i % len(prompts),
                    seed=0,
                    strategy="phase_sweep",
                    verify_seed=10_000 + (i % len(prompts)),
                )
                for i in range(n_per_cell)
            ]
            cell_records[(d, k)] = recs
            c_emp = cost_ratio_of_sums(recs)
            d_eff_med = float(np.median([r["d_eff_one_way_ms"] for r in recs]))
            sample_a = [r["accepted_total"] for r in recs]
            rows.append({
                "configured_one_way_delay_ms": d,
                "k": k,
                "n_rounds": n_per_cell,
                "median_d_eff_one_way_ms": d_eff_med,
                "median_comm_round_ms": float(np.median([r["measured_comm_round_ms"] for r in recs])),
                "median_server_total_ms": float(np.median([r["server_total_ms"] for r in recs])),
                "cost_ratio_of_sums": c_emp,
                "mean_accepted_total": float(np.mean(sample_a)),
                "C_theory_geometric_d_cfg": float(C(k, float(d), alpha, cd_used, cv_used)),
                "C_theory_geometric_d_eff": float(C(k, d_eff_med, alpha, cd_used, cv_used)),
            })

    df = pd.DataFrame(rows)

    # Per-delay empirical oracle (argmin Ĉ) and theory oracles.
    summary_rows = []
    for d in delays:
        sub = df[df.configured_one_way_delay_ms == d]
        d_eff_med = float(sub["median_d_eff_one_way_ms"].median())

        # Empirical oracle: smallest ratio-of-sums cost
        idx_min = int(sub["cost_ratio_of_sums"].idxmin())
        k_emp_oracle = int(df.loc[idx_min, "k"])
        c_emp_oracle = float(df.loc[idx_min, "cost_ratio_of_sums"])

        # Theory oracle (closed form, configured delay)
        ks_arr = np.array(ks)
        c_theory_arr = np.array([C(int(k), float(d), alpha, cd_used, cv_used) for k in ks_arr])
        k_theory_geo = int(ks_arr[np.argmin(c_theory_arr)])

        # Calibrated geometric oracle: same closed form, but d=d_eff
        c_calib_arr = np.array([C(int(k), d_eff_med, alpha, cd_used, cv_used) for k in ks_arr])
        k_theory_calib = int(ks_arr[np.argmin(c_calib_arr)])

        # Empirical-prefix theory oracle: B_emp(k) with d=d_eff
        if have_empirical_prefix:
            B_emp_full = 1.0 + np.cumsum(prefix_arr)
            c_emp_th_arr = np.array([
                (k * (cd_used + cv_used) + 2.0 * d_eff_med + cv_used) / B_emp_full[k - 1]
                for k in ks_arr
            ])
            k_theory_emp = int(ks_arr[np.argmin(c_emp_th_arr)])
        else:
            k_theory_emp = k_theory_calib

        summary_rows.append({
            "configured_one_way_delay_ms": d,
            "median_d_eff_one_way_ms": d_eff_med,
            "k_theory_geometric": k_theory_geo,
            "k_theory_calibrated": k_theory_calib,
            "k_theory_empirical": k_theory_emp,
            "k_empirical_oracle": k_emp_oracle,
            "cost_empirical_oracle": c_emp_oracle,
        })
        print(f"  d={d}ms: d_eff={d_eff_med:.1f}, k_geo={k_theory_geo}, "
              f"k_calib={k_theory_calib}, k_emp_th={k_theory_emp}, "
              f"k_emp_oracle={k_emp_oracle}, C_emp={c_emp_oracle:.1f}")

    df.to_csv(OUT_DIR / "r3_phase_transition.csv", index=False)
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "r3_phase_summary.csv", index=False)

    # Persist empirical k*(d) for downstream reuse
    state["empirical_kstar_per_delay"] = {
        str(int(r["configured_one_way_delay_ms"])): int(r["k_empirical_oracle"])
        for r in summary_rows
    }
    _save_state(state)

    # Figure (a): four k* curves
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.axvline(d_c_geom, color="gray", linestyle=":", linewidth=1,
               label=f"$d_c$(geom)$={d_c_geom:.0f}$ ms")
    ax.plot(df_sum["configured_one_way_delay_ms"], df_sum["k_theory_geometric"],
            "r--o", markersize=5, label=r"Theory $k^*$ (geom, $d_{cfg}$)")
    ax.plot(df_sum["configured_one_way_delay_ms"], df_sum["k_theory_calibrated"],
            "m:^", markersize=5, label=r"Theory $k^*$ (geom, $d_{eff}$)")
    if have_empirical_prefix:
        ax.plot(df_sum["configured_one_way_delay_ms"], df_sum["k_theory_empirical"],
                "g-.s", markersize=5, label=r"Theory $k^*$ (emp $B(k)$)")
    ax.plot(df_sum["configured_one_way_delay_ms"], df_sum["k_empirical_oracle"],
            "bD", markersize=6, label=r"Empirical oracle $\hat{k}^*$")
    ax.set_xlabel("Configured one-way delay $d$ (ms)")
    ax.set_ylabel("Optimal draft length $k^*$")
    ax.set_ylim(bottom=0.5)
    ax.legend(frameon=True, fontsize=6, loc="best")
    save_figure(fig, OUT_DIR / "fig_r3_phase_transition")
    plt.close(fig)

    # Figure (b): U-shaped cost curve per delay
    fig2, ax2 = plt.subplots(figsize=(3.6, 2.7))
    cmap = plt.cm.viridis
    for i, d in enumerate(delays):
        sub = df[df.configured_one_way_delay_ms == d].sort_values("k")
        ax2.plot(sub["k"], sub["cost_ratio_of_sums"],
                 marker="o", markersize=4, linewidth=1.0,
                 color=cmap(i / max(1, len(delays) - 1)),
                 label=f"d={d}ms")
    ax2.set_xlabel("Draft length $k$")
    ax2.set_ylabel(r"$\hat{C}(k,d)$ ms/token")
    ax2.legend(frameon=True, fontsize=5, ncol=2)
    save_figure(fig2, OUT_DIR / "fig_r3_cost_curves")
    plt.close(fig2)
    print("  Saved r3_phase_transition.csv, r3_phase_summary.csv, fig_r3_phase_transition.pdf, fig_r3_cost_curves.pdf")
    return df


# ---------------------------------------------------------------------------
# R4 — Strategy comparison (n=200 per delay)
# ---------------------------------------------------------------------------

def _empirical_oracle_k(prefix_arr: np.ndarray, d_eff: float, cd: float, cv: float, k_max: int) -> int:
    """argmin_k Ĉ(k,d) using empirical B(k) and effective d (review B4 oracle 3)."""
    ks = np.arange(1, k_max + 1)
    B_emp = 1.0 + np.cumsum(prefix_arr[:k_max])
    costs = (ks * (cd + cv) + 2.0 * d_eff + cv) / B_emp
    return int(ks[np.argmin(costs)])


def _run_strategy(strategy_name, prompts, server, alpha, cd, cv, d_mean,
                  k_max, beta, rejection_sampling, n_rounds,
                  verify_seed_base=20_000,
                  prefix_arr=None,
                  empirical_kstar_for_d=None):
    """
    Paired-prompt replay: every strategy walks the same `prompts[:n_rounds]`
    in the same order, and forwards `verify_seed = verify_seed_base + prompt_id`
    to the cloud so rejection sampling and bonus sampling become deterministic
    per (prompt, draft_ids) pair (review M3, M6).

    Strategies (review §B5):
      fixed1, fixed2, fixed3, fixed4, fixed5, fixed7, fixed10,
      greedy, specdec_pp,
      theory_oracle              (closed form, paper default α/cd/cv)
      calibrated_geometric_oracle (closed form, R1 cd/cv, configured d)
      empirical_oracle           (R2 prefix + R1 cd/cv + measured d_eff)
      naive_ucb, ucb
    """
    rng = np.random.default_rng(42)

    # Paper defaults from CLAUDE.md / experiment.md
    PAPER_ALPHA, PAPER_CD, PAPER_CV = 0.7, 1.0, 0.5

    if strategy_name.startswith("fixed"):
        k_fixed = int(strategy_name[5:])
        get_k = lambda t, _: k_fixed; update = lambda *a: None
    elif strategy_name == "greedy":
        k_fixed = greedy_policy(alpha, cd, cv, k_max)
        get_k = lambda t, _: k_fixed; update = lambda *a: None
    elif strategy_name == "specdec_pp":
        k_fixed = confidence_stop_policy(alpha, p_min=0.3, k_max=k_max)
        get_k = lambda t, _: k_fixed; update = lambda *a: None
    elif strategy_name == "theory_oracle":
        k_fixed = compute_kstar(PAPER_ALPHA, PAPER_CD, PAPER_CV, d_mean, k_max)
        get_k = lambda t, _: k_fixed; update = lambda *a: None
    elif strategy_name == "calibrated_geometric_oracle":
        k_fixed = compute_kstar(alpha, cd, cv, d_mean, k_max)
        get_k = lambda t, _: k_fixed; update = lambda *a: None
    elif strategy_name == "empirical_oracle":
        # If R3 already produced an empirical k* for this delay, prefer it
        # (true sweep argmin); otherwise fall back to prefix-based optimum.
        if empirical_kstar_for_d is not None:
            k_fixed = int(empirical_kstar_for_d)
        elif prefix_arr is not None:
            k_fixed = _empirical_oracle_k(prefix_arr, d_mean, cd, cv, k_max)
        else:
            # Last resort: same as calibrated geometric
            k_fixed = compute_kstar(alpha, cd, cv, d_mean, k_max)
        get_k = lambda t, _: k_fixed; update = lambda *a: None
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
        prompt_id = t - 1
        r = run_round(
            server,
            prompt,
            k,
            rejection_sampling,
            run_id=f"{strategy_name}_t{t}",
            prompt_id=prompt_id,
            seed=42,
            strategy=strategy_name,
            verify_seed=verify_seed_base + prompt_id,
        )
        update(k, r["total_round_time_ms"], r["accepted_total"])
        r["strategy"] = strategy_name; r["round"] = t
        records.append(r)
    return records


def exp_r4_strategy_compare(server, prompts, alpha, cd, cv, delays,
                             k_max, beta, rejection_sampling, n_rounds):
    """
    R4 strategy comparison (review §B5).

    Cost is reported as ratio-of-sums Ĉ = ΣT/ΣA (review D#1). Three oracles
    are reported side-by-side (review B4):
      * theory_oracle: paper defaults α=0.7, cd=1, cv=0.5
      * calibrated_geometric_oracle: closed form with R1 cd/cv
      * empirical_oracle: R3 sweep argmin if available, else R2-prefix optimum

    Paired prompts and per-prompt verify seeds remove cloud-side sampling
    noise (review M3, M6).
    """
    print(f"\n=== R4: Strategy comparison (n={n_rounds} per strategy per delay) ===")

    state = _load_state()
    cd_used = state.get("cd_calibrated_ms", cd)
    cv_used = state.get("cv_calibrated_ms", cv)
    if "cd_calibrated_ms" in state:
        print(f"  using R1-calibrated cd={cd_used:.2f}ms/tok, cv={cv_used:.2f}ms/tok")

    prefix_dict = state.get("prefix_P_Lge_k")
    prefix_arr = (np.array([prefix_dict[str(k)] for k in range(1, k_max + 1)])
                  if prefix_dict else None)
    emp_kstar_per_delay = state.get("empirical_kstar_per_delay", {})

    strategies = ["fixed1", "fixed2", "fixed3", "fixed4", "fixed5", "fixed7", "fixed10",
                  "greedy", "specdec_pp",
                  "theory_oracle", "calibrated_geometric_oracle", "empirical_oracle",
                  "naive_ucb", "ucb"]
    all_rows = []

    for d_idx, d in enumerate(delays):
        print(f"\n  Delay = {d} ms")
        set_delay(d)
        seed_base = 20_000 + 1_000 * d_idx
        emp_k = emp_kstar_per_delay.get(str(int(d)))

        for s in strategies:
            print(f"    {s}...", end=" ", flush=True)
            records = _run_strategy(
                s, prompts, server, alpha, cd_used, cv_used,
                float(d), k_max, beta, rejection_sampling, n_rounds,
                verify_seed_base=seed_base,
                prefix_arr=prefix_arr,
                empirical_kstar_for_d=emp_k,
            )
            cost_ros = cost_ratio_of_sums(records)
            costs = [r["cost_per_token"] for r in records]
            accepted_totals = [r["accepted_total"] for r in records]
            d_eff_med = float(np.median([r["d_eff_one_way_ms"] for r in records]))

            all_rows.append({
                "strategy": s,
                "configured_one_way_delay_ms": d,
                "median_d_eff_one_way_ms": d_eff_med,
                "n_rounds": n_rounds,
                "cost_ratio_of_sums": cost_ros,
                "mean_cost_per_token_sanity": float(np.mean(costs)),
                "std_cost_per_token": float(np.std(costs)),
                "ci95_cost_per_token": float(1.96 * np.std(costs) / np.sqrt(len(costs))),
                "mean_accepted_total": float(np.mean(accepted_totals)),
                "k_used": records[0]["k_selected"] if s.startswith("fixed") or s.endswith("_oracle") or s in ("greedy", "specdec_pp") else None,
                "verify_seed_base": seed_base,
            })
            print(f"Ĉ={cost_ros:.1f} ms/tok  (mean A_t={np.mean(accepted_totals):.2f})")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "r4_strategy_compare.csv", index=False)

    # Pivot for table
    pivot = df.pivot(index="strategy", columns="configured_one_way_delay_ms",
                     values="cost_ratio_of_sums")
    pivot.to_csv(OUT_DIR / "table_ii_revised.csv")

    # Sanity: at every delay, empirical_oracle must have the lowest cost.
    sanity_failures = []
    for d in delays:
        sub = df[df.configured_one_way_delay_ms == d]
        emp_cost = float(sub[sub.strategy == "empirical_oracle"]["cost_ratio_of_sums"].iloc[0])
        cheaper = sub[sub.cost_ratio_of_sums < emp_cost - 1e-6]
        if not cheaper.empty:
            cheaper_strats = list(cheaper["strategy"])
            sanity_failures.append((d, cheaper_strats, emp_cost))
    if sanity_failures:
        print("\n  [sanity] Empirical oracle is NOT the lowest-cost strategy at:")
        for d, strats, ec in sanity_failures:
            print(f"    d={d}ms (Ĉ_emp_oracle={ec:.1f}): cheaper => {strats}")
        print("  This is review §D#5: oracle must be cost-min on the same data.")
    else:
        print("\n  [sanity] empirical_oracle is the lowest-cost strategy at every delay (D#5 OK).")

    # Figure: bar chart per delay (review request — slim view).
    # CSV keeps all 14 strategies for the appendix; the figure only shows the
    # 7 baselines required by review §B5 plus a single `fixed_best` bar that
    # represents the best fixed-k at each delay (so the reader sees one line
    # for "tuned fixed", not seven). UCB-SpecStop is highlighted.
    apply_ieee_style()
    PLOT_STRATS = ["fixed_best", "greedy", "specdec_pp",
                   "theory_oracle", "calibrated_geometric_oracle",
                   "empirical_oracle", "naive_ucb", "ucb"]
    SHORT = {
        "fixed_best": "FixBest",
        "greedy": "Greedy",
        "specdec_pp": "Spec++",
        "theory_oracle": "ThOr",
        "calibrated_geometric_oracle": "GeoOr",
        "empirical_oracle": "EmpOr",
        "naive_ucb": "NvUCB",
        "ucb": "Ours",
    }
    fig, axes = plt.subplots(1, len(delays), figsize=(8.4, 3.0), sharey=False)
    if len(delays) == 1: axes = [axes]
    fixed_best_rows = []
    for ax, d in zip(axes, delays):
        sub_all = df[df.configured_one_way_delay_ms == d]
        # Pick best fixed-k at this delay (lowest cost_ratio_of_sums)
        fixed_sub = sub_all[sub_all.strategy.str.startswith("fixed")]
        fb_row = fixed_sub.loc[fixed_sub["cost_ratio_of_sums"].idxmin()].copy()
        fb_k_used = int(fb_row["k_used"])
        fb_cost = float(fb_row["cost_ratio_of_sums"])
        fb_ci = float(fb_row["ci95_cost_per_token"])
        fixed_best_rows.append({
            "configured_one_way_delay_ms": d,
            "k_best_fixed": fb_k_used,
            "cost_best_fixed": fb_cost,
        })

        plot_data = []
        for s in PLOT_STRATS:
            if s == "fixed_best":
                plot_data.append((SHORT[s] + f" (k={fb_k_used})", fb_cost, fb_ci, s))
            else:
                row = sub_all[sub_all.strategy == s]
                if row.empty: continue
                plot_data.append((SHORT[s], float(row["cost_ratio_of_sums"].iloc[0]),
                                  float(row["ci95_cost_per_token"].iloc[0]), s))
        # sort by cost so reader sees ranking
        plot_data.sort(key=lambda x: x[1])
        labels = [p[0] for p in plot_data]
        costs  = [p[1] for p in plot_data]
        cis    = [p[2] for p in plot_data]
        # highlight Ours
        colors = ["#d62728" if p[3] == "ucb" else "#1f77b4" for p in plot_data]
        ax.bar(range(len(plot_data)), costs, yerr=cis, capsize=2, color=colors)
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
        ax.set_title(f"$d$={d} ms", fontsize=8)
        if d == delays[0]: ax.set_ylabel(r"$\hat{C}$ (ms/token)")
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "fig_r4_strategy_compare")
    plt.close(fig)

    # Persist fixed_best summary so the table in the paper can cite it.
    pd.DataFrame(fixed_best_rows).to_csv(OUT_DIR / "r4_fixed_best.csv", index=False)
    print(f"  Saved r4_strategy_compare.csv (full 14 strats), r4_fixed_best.csv,")
    print(f"        table_ii_revised.csv, fig_r4_strategy_compare.pdf (8 strats)")
    return df


# ---------------------------------------------------------------------------
# R5 — UCB regret curves (n=500, ucb vs naive_ucb vs exp3 vs oracle)
# ---------------------------------------------------------------------------

def exp_r5_regret(server, prompts, alpha, cd, cv, d_regret,
                  k_max, beta, rejection_sampling, n_rounds):
    """
    R5 online regret (review §B5 / D#6).

    Reports cumulative regret of UCB-SpecStop, NaiveUCB (per-round-ratio), and
    EXP3-Ratio against an oracle baseline. The oracle reference cost is the
    *empirical* cost-min over the same arm set (review D#5): we run a quick
    n_oracle_probe rollout per arm to pick the empirical k* on this trace,
    rather than using the closed form.

    Per-round log (r5_round_log.csv) records t / strategy / k / T_r / A_r /
    S_N[k] / S_A[k] / estimated_C[k] / oracle_C / instant_regret /
    cumulative_regret — direct match to review §B5 spec.
    """
    print(f"\n=== R5: Regret curves at d={d_regret}ms, T={n_rounds} ===")

    state = _load_state()
    cd_used = state.get("cd_calibrated_ms", cd)
    cv_used = state.get("cv_calibrated_ms", cv)
    if "cd_calibrated_ms" in state:
        print(f"  using R1-calibrated cd={cd_used:.2f}ms/tok, cv={cv_used:.2f}ms/tok")

    set_delay(d_regret)

    # Oracle: probe each arm and pick empirical argmin Ĉ(k,d_regret) on a
    # short paired rollout (review D#5).
    n_oracle_probe = max(60, n_rounds // 50)
    print(f"  Probing empirical oracle ({n_oracle_probe} rounds/arm)...")
    arm_set = list(range(1, k_max + 1))
    oracle_costs = {}
    for k_probe in arm_set:
        recs = [
            run_round(
                server,
                prompts[i % len(prompts)],
                k_probe,
                rejection_sampling,
                run_id=f"r5_probe_k{k_probe}_t{i}",
                prompt_id=i % len(prompts),
                seed=999,
                strategy="oracle_probe",
                verify_seed=70_000 + (i % len(prompts)),
            )
            for i in range(n_oracle_probe)
        ]
        oracle_costs[k_probe] = cost_ratio_of_sums(recs)
    k_oracle = int(min(oracle_costs, key=oracle_costs.get))
    c_oracle = float(oracle_costs[k_oracle])
    c_oracle_theory = float(C(k_oracle, float(d_regret), alpha, cd_used, cv_used))
    print(f"  Empirical oracle: k={k_oracle}, Ĉ={c_oracle:.1f} ms/tok  "
          f"(theory @ this k: {c_oracle_theory:.1f}); per-arm: "
          + ", ".join(f"k{k}:{oracle_costs[k]:.1f}" for k in arm_set))

    regret_prompts = [prompts[i % len(prompts)] for i in range(n_rounds)]
    regret_data = {}
    arm_pull_history = {}
    diagnostics_lines = []
    round_log_rows = []   # for r5_round_log.csv (review §B5)

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
        pulls = np.zeros(k_max, dtype=int)
        pulls_over_time = np.zeros((n_rounds, k_max), dtype=int)
        # ratio-of-sums regret accumulators (review D#1)
        sum_T = 0.0
        sum_A = 0.0

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
                verify_seed=30_000 + (t - 1),
            )
            T_r = r["total_round_time_ms"]
            A_r = r["accepted_total"]
            update(k, T_r, A_r)
            sum_T += T_r
            sum_A += A_r

            # instantaneous regret (per round) using realised T_r/A_r minus
            # oracle C; cumulative on that.
            inst_regret = (T_r / A_r) - c_oracle
            cum += inst_regret
            curve.append(cum)
            pulls[k - 1] += 1
            pulls_over_time[t - 1] = pulls

            # Snapshot internal state for the round log
            if strat == "ucb":
                s_n_k = float(alg.s_n[k - 1])
                s_a_k = float(alg.s_a[k - 1])
                est_C = (s_n_k / s_a_k) if s_a_k > 0 else float("nan")
            elif strat == "naive_ucb":
                s_n_k = float("nan")
                s_a_k = float(alg.t_k[k - 1])
                est_C = (alg.sum_ratio[k - 1] / alg.t_k[k - 1]) if alg.t_k[k - 1] > 0 else float("nan")
            else:
                s_n_k = float("nan"); s_a_k = float("nan"); est_C = float("nan")

            round_log_rows.append({
                "t": t,
                "strategy": strat,
                "k": k,
                "T_r": T_r,
                "A_r": A_r,
                "S_N_k": s_n_k,
                "S_A_k": s_a_k,
                "estimated_C_k": est_C,
                "oracle_C": c_oracle,
                "instant_regret": inst_regret,
                "cumulative_regret": cum,
                "running_C_ratio_of_sums": (sum_T / sum_A) if sum_A > 0 else float("nan"),
            })

            if t % 100 == 0:
                hist = " ".join(f"k{kk+1}={pulls[kk]}" for kk in range(k_max))
                diag_line = f"  [{strat}] t={t}: {hist}"
                if strat == "ucb":
                    safe_a = np.where(alg.s_a > 0, alg.s_a, np.nan)
                    idx = alg.s_n / safe_a - alg.beta * np.sqrt(alg.t_k * np.log(t)) / safe_a
                    diag_line += "  idx=[" + ",".join(
                        ("nan" if np.isnan(v) else f"{v:.2f}") for v in idx
                    ) + "]"
                elif strat == "naive_ucb":
                    safe_t = np.where(alg.t_k > 0, alg.t_k, np.nan)
                    mean_r = alg.sum_ratio / safe_t
                    idx = mean_r - alg.beta * np.sqrt(np.log(t) / safe_t)
                    diag_line += "  idx=[" + ",".join(
                        ("nan" if np.isnan(v) else f"{v:.2f}") for v in idx
                    ) + "]"
                print(diag_line)
                diagnostics_lines.append(diag_line)

        regret_data[strat] = np.array(curve)
        arm_pull_history[strat] = pulls_over_time
        print(f"  {strat} final cumulative regret: {curve[-1]:.1f} ms; "
              f"final pulls=[{','.join(str(int(x)) for x in pulls)}]; "
              f"running Ĉ={(sum_T/sum_A):.1f}")

    (OUT_DIR / "r5_arm_pull_diagnostics.txt").write_text("\n".join(diagnostics_lines))
    pd.DataFrame(round_log_rows).to_csv(OUT_DIR / "r5_round_log.csv", index=False)

    np.savez(
        OUT_DIR / "r5_regret_data.npz",
        c_oracle=np.array([c_oracle]),
        c_oracle_theory=np.array([c_oracle_theory]),
        oracle_k=np.array([k_oracle]),
        d_regret=np.array([d_regret]),
        cd_used_ms=np.array([cd_used]),
        cv_used_ms=np.array([cv_used]),
        oracle_per_arm_costs=np.array([oracle_costs[k] for k in arm_set]),
        **{f"regret_{k}": v for k, v in regret_data.items()},
        **{f"pulls_{k}": v for k, v in arm_pull_history.items()},
    )

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ts = np.arange(1, n_rounds + 1)
    labels = {"ucb": "UCB-SpecStop (Ours)", "naive_ucb": "NaiveUCB (B6)", "exp3": "EXP3-Ratio"}
    styles = {"ucb": "b-", "naive_ucb": "r--", "exp3": "g-."}
    for name, cum_reg in regret_data.items():
        ax.plot(ts, cum_reg, styles[name], label=labels[name], linewidth=1.5)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(f"Cumulative regret vs empirical oracle (ms)")
    ax.set_title(f"Online regret ($d={d_regret}$ ms, $T={n_rounds}$)")
    ax.legend(frameon=True, fontsize=7)
    save_figure(fig, OUT_DIR / "fig_r5_regret")
    plt.close(fig)

    # Stacked area of cumulative arm pull share
    fig2, axes2 = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, (name, pulls_t) in zip(axes2, arm_pull_history.items()):
        cum = pulls_t.astype(float)
        share = cum / np.maximum(cum.sum(axis=1, keepdims=True), 1)
        ax.stackplot(ts, share.T, labels=[f"k={kk+1}" for kk in range(k_max)])
        ax.set_title(labels[name], fontsize=7)
        ax.set_xlabel("Round $t$")
        ax.set_ylim(0, 1)
    axes2[0].set_ylabel("Cumulative pull share")
    axes2[-1].legend(frameon=True, fontsize=5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save_figure(fig2, OUT_DIR / "fig_r5_arm_pulls")
    plt.close(fig2)

    # log-log slope figure (review §B5 item 2)
    fig3, ax3 = plt.subplots(figsize=(3.5, 2.6))
    ts_log = np.maximum(ts, 1)
    for name, cum_reg in regret_data.items():
        clipped = np.maximum(cum_reg, 1e-3)
        ax3.loglog(ts_log, clipped, styles[name], label=labels[name], linewidth=1.5)
    # Reference √(T log T) line (anchored at t=10)
    ref = np.sqrt(ts_log * np.log(np.maximum(ts_log, 2)))
    ref *= regret_data["ucb"][9] / max(ref[9], 1e-3)
    ax3.loglog(ts_log, ref, "k:", linewidth=1, label=r"$\sqrt{T\log T}$ ref")
    ax3.set_xlabel("Round $t$ (log)")
    ax3.set_ylabel("Cumulative regret (ms, log)")
    ax3.legend(frameon=True, fontsize=6)
    save_figure(fig3, OUT_DIR / "fig_r5_regret_loglog")
    plt.close(fig3)

    print("  Saved r5_regret_data.npz, r5_round_log.csv, fig_r5_regret.pdf, "
          "fig_r5_regret_loglog.pdf, fig_r5_arm_pulls.pdf, r5_arm_pull_diagnostics.txt")
    return regret_data


# ---------------------------------------------------------------------------
# Extra figures: R4-with-oracle bar chart + R5 convergence vs round.
# Reads only csv (r4_strategy_compare.csv, r5_round_log.csv); does not re-run.
# Called automatically when both R4 and R5 ran in the same invocation, and
# also exposed as a standalone helper for post-hoc replotting.
# ---------------------------------------------------------------------------

_EXTRA_PLOT_STRATS = ["fixed_best", "greedy", "specdec_pp",
                      "theory_oracle", "calibrated_geometric_oracle",
                      "empirical_oracle", "naive_ucb", "ucb"]
_EXTRA_SHORT = {
    "fixed_best": "FixBest", "greedy": "Greedy", "specdec_pp": "SpecDec++",
    "theory_oracle": "TheoryOra", "calibrated_geometric_oracle": "CalibOra",
    "empirical_oracle": "EmpOra", "naive_ucb": "Naive-UCB", "ucb": "Ours",
}
_STRAT_COLORS = {
    "fixed_best":                  "#1f77b4",
    "greedy":                      "#8c564b",
    "specdec_pp":                  "#9467bd",
    "theory_oracle":               "#17becf",
    "calibrated_geometric_oracle": "#bcbd22",
    "empirical_oracle":            "#2ca02c",
    "naive_ucb":                   "#ff7f0e",
    "ucb":                         "#d62728",
    "exp3":                        "#7f7f7f",
}


def _fig_r4_with_oracle(r4_path: Path, out_png: Path, out_pdf: Path) -> None:
    df = pd.read_csv(r4_path)
    delays = sorted(df["configured_one_way_delay_ms"].unique())

    rows = []
    for d in delays:
        sub = df[df["configured_one_way_delay_ms"] == d]
        fixed_sub = sub[sub["strategy"].str.startswith("fixed")]
        if not fixed_sub.empty:
            fb = fixed_sub.loc[fixed_sub["cost_ratio_of_sums"].idxmin()].copy()
            fb["strategy"] = "fixed_best"
            fb["k_label"] = int(fb["k_used"])
            rows.append(fb)
        for s in [x for x in _EXTRA_PLOT_STRATS if x != "fixed_best"]:
            mr = sub[sub["strategy"] == s]
            if not mr.empty:
                r = mr.iloc[0].copy()
                r["k_label"] = "" if pd.isna(r.get("k_used")) else \
                    (int(r["k_used"]) if r["k_used"] == r["k_used"] else "")
                rows.append(r)
    plot_df = pd.DataFrame(rows)

    n_strats = len(_EXTRA_PLOT_STRATS)
    width = 0.8 / n_strats
    x = np.arange(len(delays))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for j, s in enumerate(_EXTRA_PLOT_STRATS):
        sub = plot_df[plot_df["strategy"] == s]
        cost, ks = [], []
        for d in delays:
            row = sub[sub["configured_one_way_delay_ms"] == d]
            if row.empty:
                cost.append(np.nan); ks.append("")
            else:
                cost.append(float(row["cost_ratio_of_sums"].iloc[0]))
                k_lab = row["k_label"].iloc[0] if "k_label" in row.columns else ""
                ks.append(str(k_lab) if k_lab not in (None, "", float("nan")) else "")
        color = _STRAT_COLORS.get(s, "#1f77b4")
        ax.bar(x + j * width, cost, width, label=_EXTRA_SHORT[s],
               color=color, edgecolor="black", linewidth=0.4)
        if s == "fixed_best":
            for bi, k in enumerate(ks):
                if k:
                    ax.text(x[bi] + j * width, cost[bi], f"k={k}",
                            ha="center", va="bottom", fontsize=7)

    for i, d in enumerate(delays):
        sub = df[(df["configured_one_way_delay_ms"] == d) &
                 (df["strategy"] == "empirical_oracle")]
        if sub.empty:
            continue
        oracle = float(sub["cost_ratio_of_sums"].iloc[0])
        ax.hlines(oracle, x[i] - 0.05, x[i] + 0.8 - width / 2,
                  colors="#d62728", linestyles="--", linewidth=1.5, zorder=4)
        ucb_row = df[(df["configured_one_way_delay_ms"] == d) &
                     (df["strategy"] == "ucb")]
        if not ucb_row.empty:
            ucb_cost = float(ucb_row["cost_ratio_of_sums"].iloc[0])
            gap_pct = 100.0 * (ucb_cost - oracle) / oracle
            ax.annotate(f"{gap_pct:+.1f}%",
                        xy=(x[i] + _EXTRA_PLOT_STRATS.index("ucb") * width,
                            ucb_cost),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color="#d62728",
                        fontweight="bold")

    ax.set_xticks(x + (n_strats - 1) * width / 2)
    ax.set_xticklabels([f"d={d} ms" for d in delays])
    ax.set_ylabel("Per-token cost (ms/token), ratio-of-sums")
    ax.set_title("R4: strategy comparison, with empirical oracle (red dashed) per delay")
    h, lab = ax.get_legend_handles_labels()
    h.append(plt.Line2D([0], [0], color="#d62728", linestyle="--", lw=1.5))
    lab.append("EmpOracle (line)")
    ax.legend(h, lab, ncol=5, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, -0.12))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}")
    print(f"[plot] wrote {out_pdf}")


def _fig_convergence(r5_path: Path, r4_path: Path, out_png: Path, out_pdf: Path,
                     fixed_ref_delay: int) -> None:
    r5 = pd.read_csv(r5_path)
    r4 = pd.read_csv(r4_path)

    oracle_C = float(r5[r5["strategy"] == "ucb"]["oracle_C"].iloc[0])

    fixed_rows = r4[(r4["strategy"].str.startswith("fixed")) &
                    (r4["configured_one_way_delay_ms"] == fixed_ref_delay)].copy()
    if fixed_rows.empty:
        # Fallback: pick the R4 delay closest to where R5 ran.
        avail = sorted(r4["configured_one_way_delay_ms"].unique())
        fixed_ref_delay = min(avail, key=lambda d: abs(d - oracle_C))
        fixed_rows = r4[(r4["strategy"].str.startswith("fixed")) &
                        (r4["configured_one_way_delay_ms"] == fixed_ref_delay)].copy()
    fixed_rows["k_used"] = fixed_rows["k_used"].astype(int)
    fixed_rows = fixed_rows.sort_values("k_used")

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    style = {
        "ucb":      dict(color=_STRAT_COLORS["ucb"],      lw=2.4,            label="Ours (UCB-SpecStop)"),
        "naive_ucb":dict(color=_STRAT_COLORS["naive_ucb"],lw=1.6, ls="--",   label="Naive-UCB (mean of N/A)"),
        "exp3":     dict(color=_STRAT_COLORS["exp3"],     lw=1.4, ls=":",    label="EXP3"),
    }
    for s, st in style.items():
        sub = r5[r5["strategy"] == s]
        if sub.empty:
            continue
        ax.plot(sub["t"].values, sub["running_C_ratio_of_sums"].values, **st, zorder=5)

    ax.axhline(oracle_C, color="black", lw=1.5, ls="-",
               label=f"Empirical oracle = {oracle_C:.1f} ms/token", zorder=6)

    n_fix = len(fixed_rows)
    cmap = plt.cm.Greys(np.linspace(0.35, 0.85, max(n_fix, 1)))
    t_max = int(r5["t"].max())
    for i, (_, fr) in enumerate(fixed_rows.iterrows()):
        c = float(fr["cost_ratio_of_sums"]); k = int(fr["k_used"])
        ax.axhline(c, color=cmap[i], lw=1.0, ls="--", alpha=0.85, zorder=2)
        ax.text(t_max * 1.005, c, f"fixed k={k}: {c:.0f}",
                va="center", ha="left", fontsize=7, color=cmap[i])

    ax.set_xlabel("Round t")
    ax.set_ylabel("Running per-token cost (ms/token)")
    ax.set_title(f"R5 convergence at d ≈ d_c "
                 f"(reference fixed-k from R4 @ d={fixed_ref_delay} ms)")
    ax.set_xscale("log")
    ax.set_xlim(1, t_max * 1.05)
    # Honest y-range: cover every plotted line + oracle + fixed-k references.
    # Early-round spikes (e.g. EXP3 at t=1) push the upper bound up; that's the
    # point — the user wants to see the full trajectory, not a clipped window.
    series_min = []
    series_max = []
    for s in style:
        sub = r5[r5["strategy"] == s]
        if not sub.empty:
            v = sub["running_C_ratio_of_sums"].values
            series_min.append(float(np.nanmin(v)))
            series_max.append(float(np.nanmax(v)))
    if n_fix:
        series_min.append(float(fixed_rows["cost_ratio_of_sums"].min()))
        series_max.append(float(fixed_rows["cost_ratio_of_sums"].max()))
    series_min.append(oracle_C)
    series_max.append(oracle_C)
    y_lo = min(series_min)
    y_hi = max(series_max)
    pad = 0.04 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ucb_tail = r5[r5["strategy"] == "ucb"].tail(1)
    if not ucb_tail.empty:
        ucb_final = float(ucb_tail["running_C_ratio_of_sums"].iloc[0])
        gap_pct = 100.0 * (ucb_final - oracle_C) / oracle_C
        # Place the annotation roughly mid-axis so it stays visible even when
        # the y-range is large (early-round transients can stretch it 4-5x
        # beyond the converged band).
        y_mid = 0.5 * (y_lo + y_hi)
        ax.annotate(f"Ours @ t={t_max}: {ucb_final:.1f}\n= oracle {gap_pct:+.2f}%",
                    xy=(t_max, ucb_final),
                    xytext=(t_max * 0.18, y_mid),
                    textcoords="data",
                    ha="left", fontsize=9, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}")
    print(f"[plot] wrote {out_pdf}")


def render_extra_figures(out_dir: Path, regret_delay: int,
                         r3_grid: list = None) -> None:
    """Build fig_r4_with_oracle and fig_convergence_vs_round from csvs.

    Skips silently if either input csv is missing — that lets users re-run
    only some experiments without crashing the orchestration.

    fixed_ref_delay snaps regret_delay onto the R4/R3 grid so the convergence
    figure's reference fixed-k bars come from the run where d is closest to
    where R5 was driven.
    """
    r4_csv = out_dir / "r4_strategy_compare.csv"
    r5_csv = out_dir / "r5_round_log.csv"
    if not r4_csv.exists() or not r5_csv.exists():
        print(f"[plot] skipping extra figures: need both {r4_csv.name} and "
              f"{r5_csv.name} in {out_dir}")
        return

    grid = r3_grid or [0, 5, 20, 40, 55, 83, 111, 150]
    fixed_ref_delay = min(grid, key=lambda d: abs(d - regret_delay))
    _fig_r4_with_oracle(r4_csv,
                        out_dir / "fig_r4_with_oracle.png",
                        out_dir / "fig_r4_with_oracle.pdf")
    _fig_convergence(r5_csv, r4_csv,
                     out_dir / "fig_convergence_vs_round.png",
                     out_dir / "fig_convergence_vs_round.pdf",
                     fixed_ref_delay=fixed_ref_delay)


# ---------------------------------------------------------------------------
# R6 — Markov channel VOI
# ---------------------------------------------------------------------------

def exp_r6_markov(server, prompts, alpha, cd, cv,
                  k_max, beta, rejection_sampling, n_rounds,
                  d_good=40, d_bad=120, p_g2b=0.1, p_b2g=0.1):
    """
    Markov two-state VOI experiment.

    Reports two oracle baselines so the VOI claim is internally consistent
    with R2/R3/R4 (review M4):
      - geometric oracle: compute_kstar with alpha_fit
      - empirical oracle: argmin C_emp(k,d) using R2 prefix and R1 cd/cv
    Both per-state oracle costs are written to r6_markov_voi.json. The
    "VOI" reported is the realised cost gap between blind and contextual UCB
    in this trace, not a model-derived constant.
    """
    print(f"\n=== R6: Markov VOI (good={d_good}ms, bad={d_bad}ms) ===")
    state_data = _load_state()
    cd_used = state_data.get("cd_calibrated_ms", cd)
    cv_used = state_data.get("cv_calibrated_ms", cv)
    if "cd_calibrated_ms" in state_data:
        print(f"  using R1-calibrated cd={cd_used:.1f}ms, cv={cv_used:.1f}ms")

    rng = np.random.default_rng(0)
    k_good_geo = compute_kstar(alpha, cd_used, cv_used, float(d_good), k_max)
    k_bad_geo  = compute_kstar(alpha, cd_used, cv_used, float(d_bad),  k_max)
    c_g_geo = float(C(k_good_geo, float(d_good), alpha, cd_used, cv_used))
    c_b_geo = float(C(k_bad_geo,  float(d_bad),  alpha, cd_used, cv_used))

    prefix_dict = state_data.get("prefix_P_Lge_k")
    if prefix_dict:
        prefix_arr = np.array([prefix_dict[str(k)] for k in range(1, k_max + 1)])
        k_good_emp, c_g_emp = _empirical_kstar(prefix_arr, float(d_good), cd_used, cv_used, k_max)
        k_bad_emp,  c_b_emp = _empirical_kstar(prefix_arr, float(d_bad),  cd_used, cv_used, k_max)
        print(f"  geometric oracle: k_good={k_good_geo} (C={c_g_geo:.1f}), k_bad={k_bad_geo} (C={c_b_geo:.1f})")
        print(f"  empirical oracle: k_good={k_good_emp} (C={c_g_emp:.1f}), k_bad={k_bad_emp} (C={c_b_emp:.1f})")
    else:
        prefix_arr = None
        k_good_emp = k_bad_emp = None
        c_g_emp = c_b_emp = None
        print(f"  [warn] no R2 prefix; only geometric oracle reported")
        print(f"  geometric oracle: k_good={k_good_geo} (C={c_g_geo:.1f}), k_bad={k_bad_geo} (C={c_b_geo:.1f})")

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
        prompt_id = t % len(prompts)

        # Paired comparison: blind and contextual see the same cloud RNG state
        # for this prompt. Strategy difference is only in arm choice.
        verify_seed = 40_000 + prompt_id

        # Blind
        k_bl = ucb_blind.select_arm(t)
        r_bl = run_round(server, prompt, k_bl, rejection_sampling,
                         prompt_id=prompt_id, strategy="blind",
                         verify_seed=verify_seed)
        ucb_blind.update(k_bl, r_bl["total_round_time_ms"], r_bl["accepted_total"])
        records_blind.append({**r_bl, "state": state})

        # Contextual (separate UCB per state)
        ucb_ctx = ucb_good_st if state == "good" else ucb_bad_st
        k_ct = ucb_ctx.select_arm(t)
        r_ct = run_round(server, prompt, k_ct, rejection_sampling,
                         prompt_id=prompt_id, strategy="contextual",
                         verify_seed=verify_seed)
        ucb_ctx.update(k_ct, r_ct["total_round_time_ms"], r_ct["accepted_total"])
        records_ctx.append({**r_ct, "state": state})

        if t % 100 == 0:
            c_bl = cost_ratio_of_sums(records_blind[-100:])
            c_ct = cost_ratio_of_sums(records_ctx[-100:])
            print(f"  t={t}: blind={c_bl:.1f}, ctx={c_ct:.1f}  state={state}")

    c_bl_avg = cost_ratio_of_sums(records_blind)
    c_ct_avg = cost_ratio_of_sums(records_ctx)
    voi_pct  = (c_bl_avg - c_ct_avg) / c_bl_avg * 100.0

    result = {
        "blind_cost_ratio_of_sums": c_bl_avg,
        "contextual_cost_ratio_of_sums": c_ct_avg,
        "voi_pct": voi_pct,
        "d_good": d_good, "d_bad": d_bad,
        "p_g2b": p_g2b, "p_b2g": p_b2g,
        "k_good_geometric": k_good_geo, "k_bad_geometric": k_bad_geo,
        "C_good_geometric": c_g_geo, "C_bad_geometric": c_b_geo,
        "k_good_empirical": k_good_emp, "k_bad_empirical": k_bad_emp,
        "C_good_empirical": c_g_emp, "C_bad_empirical": c_b_emp,
        "cd_used_ms": cd_used, "cv_used_ms": cv_used,
        "alpha_fit": alpha,
        "used_empirical_prefix": prefix_arr is not None,
    }
    (OUT_DIR / "r6_markov_voi.json").write_text(json.dumps(result, indent=2))

    # Use empirical oracle when available (more conservative VOI baseline)
    if c_g_emp is not None and c_b_emp is not None:
        oracle_cost = (c_g_emp + c_b_emp) / 2.0
        oracle_label = "Empirical oracle"
    else:
        oracle_cost = (c_g_geo + c_b_geo) / 2.0
        oracle_label = "Geometric oracle"

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ts = np.arange(1, n_rounds + 1)
    # Per-round contribution to ratio-of-sums regret: (T_r - oracle * A_r) so
    # cumulative sum / cumulative A = running Ĉ - oracle_cost.
    blind_inst = [r["total_round_time_ms"] - oracle_cost * r["accepted_total"] for r in records_blind]
    ctx_inst   = [r["total_round_time_ms"] - oracle_cost * r["accepted_total"] for r in records_ctx]
    ax.plot(ts, np.cumsum(blind_inst), "r--", label="UCB-SpecStop (Blind)")
    ax.plot(ts, np.cumsum(ctx_inst), "b-", label="UCB-SpecStop (Contextual)")
    ax.set_xlabel("Round $t$"); ax.set_ylabel(f"Cumulative regret vs {oracle_label} (ms)")
    ax.legend(frameon=True, fontsize=7)
    save_figure(fig, OUT_DIR / "fig_r6_markov_regret")
    plt.close(fig)
    print(f"  VOI={voi_pct:.2f}%  (Ĉ_blind={c_bl_avg:.1f}, Ĉ_ctx={c_ct_avg:.1f})")
    print(f"  Saved r6_markov_voi.json, fig_r6_markov_regret.pdf")
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
    parser.add_argument("--params",       default=None,
        help="Real measured params json from hardware/measure_params.py. "
             "If omitted, defaults to outputs/hardware/params_<suite>.json when --suite is given, "
             "else outputs/hardware/params_measured.json")
    parser.add_argument("--prompts",      default="hardware/prompts.txt")
    parser.add_argument("--cloud-model",  default=None,
        help="Cloud verification model; defaults to the suite's matching 7B/8B model")
    parser.add_argument("--out-dir",      default=None,
        help="Output directory override")
    parser.add_argument("--n-prompts",    type=int, default=500)
    parser.add_argument("--exp",          default="all",
        help="Comma-separated subset of {r1,r2,r3,r4,r5,r6} or 'all'. "
             "Examples: --exp all, --exp r4, --exp r2,r3, --exp r5,r6")
    parser.add_argument("--k-max",        type=int, default=10)
    parser.add_argument("--beta",         type=float, default=1.0)
    parser.add_argument("--n-rounds",     type=int, default=200,
        help="Rounds per strategy in R4; R5 uses 3x this")
    parser.add_argument("--no-rejection-sampling", dest="rs",
        action="store_false", default=True)
    parser.add_argument("--allow-download", action="store_true", default=False,
        help="Allow downloading models from Hugging Face when local files are unavailable")
    parser.add_argument("--offload-folder", default=None,
        help="Spill weights to this directory when RAM is short (e.g. /home/jetson/offload).")
    parser.add_argument("--max-cpu-mem-gb", type=float, default=None,
        help="Cap CPU memory budget for accelerate (e.g. 1.0). Only used with --offload-folder.")
    parser.add_argument("--max-gpu-mem-gb", type=float, default=None,
        help="Cap GPU memory budget for accelerate (e.g. 3.0). On Jetson unified memory, this caps the resident slice.")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--cd",    type=float, default=15.0)
    parser.add_argument("--cv",    type=float, default=3.0)
    args = parser.parse_args()

    if args.suite:
        suite_default_draft = MODEL_SUITES[args.suite]["draft"]
        parser_default_draft = parser.get_default("draft_model")
        if args.draft_model == parser_default_draft:
            # Offline-Jetson fallback: if the HF cache for this repo is a stub
            # (refs/ exists but snapshots/ is empty), and a complete local copy
            # exists at ~/local/models/<basename>, prefer the local path.
            repo_basename = suite_default_draft.split("/")[-1]
            local_fallback = Path.home() / "local" / "models" / repo_basename
            org, name = suite_default_draft.split("/")
            snap = (Path.home() / ".cache" / "huggingface" / "hub" /
                    f"models--{org}--{name}" / "snapshots")
            cache_ok = snap.is_dir() and any(snap.iterdir())
            if (not cache_ok and local_fallback.is_dir()
                    and (local_fallback / "config.json").exists()):
                print(f"[HW] HF cache for {suite_default_draft} is incomplete; "
                      f"using local copy at {local_fallback}")
                args.draft_model = str(local_fallback)
            else:
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
        # Auto-resolve: prefer suite-specific params; fall back to legacy path.
        repo_root = Path(__file__).resolve().parent.parent
        if args.suite:
            cand = repo_root / "outputs" / "hardware" / f"params_{args.suite}.json"
        else:
            cand = repo_root / "outputs" / "hardware" / "params_measured.json"
        if not cand.exists():
            raise FileNotFoundError(
                f"--params not given and {cand} not found. Run hardware/measure_params.py "
                f"first, or pass --params <file> explicitly."
            )
        args.params = str(cand)
        print(f"[HW] auto-selected params: {args.params}")
    p = json.loads(Path(args.params).read_text())
    alpha = p["alpha_fit"]; cd = p["cd_ms"]; cv = p["cv_ms"]
    print(f"[HW] params: alpha={alpha:.3f}, cd={cd:.1f}ms, cv={cv:.1f}ms")

    load_model(
        args.draft_model,
        allow_download=args.allow_download,
        offload_folder=args.offload_folder,
        max_cpu_mem_gb=args.max_cpu_mem_gb,
        max_gpu_mem_gb=args.max_gpu_mem_gb,
    )
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

    # Calibration delays: review §B1 / §D — full set so cd/cv per k can be
    # estimated robustly. R1 sweeps every k in {1,2,3,5,7,10}.
    calib_delays  = [0, 5, 10, 20, 40, 55, 83, 111, 150]
    calib_ks      = [1, 2, 3, 5, 7, 10]
    calib_n_per_cell = 300
    # Phase transition: dense around d_c
    pt_delays = sorted(set([max(1, int(d_c * f))
                             for f in [0.4, 0.6, 0.75, 0.88, 1.0, 1.12, 1.25, 1.5, 1.75, 2.0]]
                            + [int(d_c)]))
    # Strategy compare: must be a SUBSET of R3 grid, so empirical_oracle hits
    # the cached `empirical_kstar_per_delay` key directly (review D#5).
    # R3 grid is [0, 5, 20, 40, 55, 83, 111, 150]; pick a 4-point spread that
    # straddles d_c on both sides.
    R3_GRID = [0, 5, 20, 40, 55, 83, 111, 150]
    strat_delays = [20, 55, 111, 150]
    assert set(strat_delays).issubset(R3_GRID), \
        f"R4 delays {strat_delays} must be subset of R3 grid {R3_GRID}"
    # Regret: at d_c
    regret_delay  = int(d_c)
    # Markov: good below d_c, bad above d_c
    d_good = max(1, int(d_c * 0.5))
    d_bad  = int(d_c * 1.5)

    requested = {tag.strip().lower() for tag in args.exp.split(",") if tag.strip()}
    valid = {"all", "r1", "r2", "r3", "r4", "r5", "r6"}
    bad = requested - valid
    if bad:
        raise ValueError(f"--exp got unknown tags: {sorted(bad)}. Valid: {sorted(valid)}")
    run_all = "all" in requested
    run_r1 = run_all or "r1" in requested
    run_r2 = run_all or "r2" in requested
    run_r3 = run_all or "r3" in requested
    run_r4 = run_all or "r4" in requested
    run_r5 = run_all or "r5" in requested
    run_r6 = run_all or "r6" in requested

    if run_r1:
        exp_r1_calibration(args.server, prompts, calib_delays,
                           ks=calib_ks, n_per_cell=calib_n_per_cell)
    if run_r2:
        exp_r2_acceptance(args.server, prompts, k_max=args.k_max)
    if run_r3:
        exp_r3_phase_transition(args.server, prompts, alpha, cd, cv, args.k_max)
    if run_r4:
        # Review D#6: 200 rounds is too short for UCB to converge. Use 1000
        # for R4 (regret payoff still belongs to R5 with T=5000).
        n_r4 = max(args.n_rounds, 1000)
        exp_r4_strategy_compare(args.server, prompts, alpha, cd, cv, strat_delays,
                                args.k_max, args.beta, args.rs, n_r4)
    if run_r5:
        # Review §D#6: ≥5000 rounds. Default 5000; allow CLI override.
        r5_T = max(5000, int(args.n_rounds * 25))
        exp_r5_regret(args.server, prompts, alpha, cd, cv, regret_delay,
                      args.k_max, args.beta, args.rs, r5_T)
    if run_r6:
        exp_r6_markov(args.server, prompts, alpha, cd, cv,
                      args.k_max, args.beta, args.rs, args.n_rounds * 2,
                      d_good=d_good, d_bad=d_bad)

    # Auto-render the two extra figures whenever both R4 and R5 ran in this
    # invocation. They read csv only, so they're cheap and idempotent.
    if run_r4 and run_r5:
        render_extra_figures(OUT_DIR, regret_delay=regret_delay,
                             r3_grid=R3_GRID)

    print("\n[HW] All requested experiments complete.")
    print(f"[HW] Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
