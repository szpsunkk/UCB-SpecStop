"""
edge_client.py — Run on the Jetson Orin Nano Super (edge side).

Loads the draft model locally. For each generation step:
  1. Draft k tokens with the local model (optionally extracting log-probs).
  2. Send to cloud_server for verification (HTTP POST /verify).
  3. Receive n_accepted + bonus token.
  4. Record (N_t, A_t) for strategy evaluation.

Supported strategies (B1–B6 + Ours):
  fixed1/3/5/7/10   — B1: fixed k
  greedy            — B2: d=0 optimal (requires --alpha --cd --cv)
  sled              — B3: SLED-style timeout adaptive
  specdec_pp        — B4: SpecDec++ probability threshold
  oracle            — B5: oracle optimal k (requires known --delay)
  naive_ucb         — B6: UCB1 on mean(N/A), biased estimator
  ucb               — Ours: UCB-SpecStop, ratio-of-sums estimator
  exp3              — EXP3-Ratio adversarial bandit

Usage:
  python edge_client.py \\
      --draft-model Qwen/Qwen2.5-0.5B \\
      --server http://192.168.1.100:8000 \\
      --strategy ucb \\
      --delay 50 \\
      --prompts prompts.txt \\
      --output results_ucb_d50.json \\
      --rejection-sampling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baselines import (
    UCBSpecStop, PerRoundRatioUCB, EXP3Ratio, EpsilonGreedyRatio,
    confidence_stop_policy, greedy_policy, oracle_mean_policy,
)
from src.core import compute_kstar

STRATEGIES = (
    "ucb", "naive_ucb", "exp3", "epsilon_greedy",
    "greedy", "specdec_pp", "oracle",
    "sled",
    "fixed1", "fixed3", "fixed5", "fixed7", "fixed10",
)

_draft_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Draft model helpers
# ---------------------------------------------------------------------------

def load_draft_model(model_name: str):
    global _draft_model, _tokenizer
    print(f"Loading draft model: {model_name}  device={_device}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _draft_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(_device)
    _draft_model.eval()
    print("Draft model loaded.")


def generate_draft(input_ids: torch.Tensor, k: int) -> list[int]:
    with torch.no_grad():
        out = _draft_model.generate(
            input_ids,
            max_new_tokens=k,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    return out[0, input_ids.shape[1]:].tolist()


def generate_draft_with_log_probs(
    input_ids: torch.Tensor, k: int
) -> tuple[list[int], list[float]]:
    """Generate k tokens; return (token_ids, per-token log-probs under draft model)."""
    with torch.no_grad():
        out = _draft_model.generate(
            input_ids,
            max_new_tokens=k,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    token_ids = out.sequences[0, input_ids.shape[1]:].tolist()
    log_probs = []
    for score, tok in zip(out.scores, token_ids):
        lp = torch.log_softmax(score[0], dim=-1)[tok].item()
        log_probs.append(lp)
    return token_ids, log_probs


def measure_cd(n_warmup: int = 10, n_measure: int = 200) -> float:
    """Measure per-token draft latency (cd) in ms via 1-token generation."""
    dummy = _tokenizer("Hello", return_tensors="pt").input_ids.to(_device)
    for _ in range(n_warmup):
        generate_draft(dummy, 1)
    latencies = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        generate_draft(dummy, 1)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(latencies))


# ---------------------------------------------------------------------------
# Cloud communication
# ---------------------------------------------------------------------------

def verify_on_cloud(
    server: str,
    context_ids: list[int],
    draft_ids: list[int],
    draft_log_probs: Optional[list[float]] = None,
) -> dict:
    payload: dict = {"context_ids": context_ids, "draft_ids": draft_ids}
    if draft_log_probs is not None:
        payload["draft_log_probs"] = draft_log_probs
    resp = requests.post(f"{server}/verify", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def measure_rtt(server: str, n: int = 100) -> float:
    """Estimate one-way delay (ms) = median RTT / 2."""
    rtts = []
    for _ in range(n):
        t0 = time.perf_counter()
        requests.get(f"{server}/ping", timeout=5)
        rtts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(rtts)) / 2.0


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

class SLEDStrategy:
    """B3: SLED-style timeout — decrement k on slow round, increment on fast."""
    def __init__(self, k_init: int = 5, k_max: int = 20, timeout_ms: float = 100.0):
        self.k = k_init
        self.k_max = k_max
        self.timeout_ms = timeout_ms

    def select(self, last_rtt_ms: Optional[float] = None) -> int:
        if last_rtt_ms is not None:
            if last_rtt_ms > self.timeout_ms:
                self.k = max(1, self.k - 1)
            else:
                self.k = min(self.k_max, self.k + 1)
        return self.k


def make_strategy(args, alpha: float, cd: float, cv: float, d_mean: float):
    name = args.strategy
    k_max = args.k_max
    rng = np.random.default_rng(args.seed)

    if name.startswith("fixed"):
        k = int(name[5:])
        return ("fixed", k)
    if name == "greedy":
        k = greedy_policy(alpha, cd, cv, k_max)
        return ("fixed", k)
    if name == "specdec_pp":
        k = confidence_stop_policy(alpha, p_min=args.p_min, k_max=k_max)
        return ("fixed", k)
    if name == "oracle":
        k = compute_kstar(alpha, cd, cv, d_mean, k_max)
        return ("fixed", k)
    if name == "sled":
        return ("sled", SLEDStrategy(k_init=args.k_init, k_max=k_max,
                                     timeout_ms=args.sled_timeout))
    if name == "ucb":
        return ("ucb", UCBSpecStop(k_max=k_max, beta=args.beta))
    if name == "naive_ucb":
        return ("naive_ucb", PerRoundRatioUCB(k_max=k_max, beta=args.beta))
    if name == "exp3":
        return ("exp3", EXP3Ratio(k_max=k_max))
    if name == "epsilon_greedy":
        return ("epsilon_greedy", EpsilonGreedyRatio(k_max=k_max, epsilon=args.epsilon))
    raise ValueError(f"Unknown strategy: {name}")


def get_k(strategy_tuple, t: int, last_rtt_ms: Optional[float],
          rng: np.random.Generator) -> int:
    kind, obj = strategy_tuple
    if kind == "fixed":
        return obj
    if kind == "sled":
        return obj.select(last_rtt_ms)
    if kind == "ucb":
        return obj.select_arm(t)
    if kind == "naive_ucb":
        return obj.select_arm(t)
    if kind == "exp3":
        return obj.select_arm(t, rng)
    if kind == "epsilon_greedy":
        return obj.select_arm(t, rng)
    raise ValueError(kind)


def update_strategy(strategy_tuple, k: int, n_t: float, a_t: float,
                    rng: np.random.Generator):
    kind, obj = strategy_tuple
    if kind in ("ucb", "naive_ucb", "epsilon_greedy"):
        obj.update(k, n_t, a_t)
    elif kind == "exp3":
        obj.update(k, n_t, a_t)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args):
    prompts = Path(args.prompts).read_text().strip().split("\n")
    if args.n_prompts:
        prompts = prompts[: args.n_prompts]

    # Load measured params if available
    alpha, cd, cv = args.alpha, args.cd, args.cv
    if args.params:
        import json as _json
        p = _json.loads(Path(args.params).read_text())
        alpha = p["alpha_fit"]
        cd = p["cd_ms"]
        cv = p["cv_ms"]
        print(f"Loaded params: alpha={alpha:.3f}, cd={cd:.2f}ms, cv={cv:.2f}ms")

    strategy = make_strategy(args, alpha, cd, cv, args.delay)
    rng = np.random.default_rng(args.seed)

    results = []
    last_rtt_ms: Optional[float] = None

    for t, prompt in enumerate(prompts, start=1):
        k = get_k(strategy, t, last_rtt_ms, rng)
        input_ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(_device)
        context_ids = input_ids[0].tolist()

        t_draft_start = time.perf_counter()
        if args.rejection_sampling:
            draft_ids, draft_log_probs = generate_draft_with_log_probs(input_ids, k)
        else:
            draft_ids = generate_draft(input_ids, k)
            draft_log_probs = None
        t_draft = (time.perf_counter() - t_draft_start) * 1000.0

        t_comm_start = time.perf_counter()
        resp = verify_on_cloud(args.server, context_ids, draft_ids, draft_log_probs)
        t_comm = (time.perf_counter() - t_comm_start) * 1000.0
        last_rtt_ms = t_comm

        n_accepted = resp["n_accepted"]
        a_t = float(n_accepted + 1)      # bonus token always included
        n_t = t_draft + t_comm           # total wall-clock cost this round

        update_strategy(strategy, k, n_t, a_t, rng)

        record = {
            "round": t,
            "k": k,
            "N_t_ms": n_t,
            "A_t": a_t,
            "n_accepted": n_accepted,
            "t_draft_ms": t_draft,
            "t_comm_ms": t_comm,
            "verify_time_ms": resp["verify_time_ms"],
        }
        results.append(record)

        if t % 50 == 0:
            recent = results[-50:]
            avg_cost = sum(r["N_t_ms"] / r["A_t"] for r in recent) / len(recent)
            print(f"  round={t:4d}  k={k:2d}  A={a_t:.0f}  "
                  f"avg_cost={avg_cost:.2f} ms/token")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} rounds to {args.output}")

    avg_cost = sum(r["N_t_ms"] / r["A_t"] for r in results) / len(results)
    print(f"Overall avg cost: {avg_cost:.3f} ms/token")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--server", default="http://192.168.1.100:8000")
    parser.add_argument("--strategy", choices=STRATEGIES, default="ucb")
    parser.add_argument("--delay", type=float, default=50.0,
                        help="Nominal mean delay (ms) for oracle/greedy")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--cd", type=float, default=15.0,
                        help="Measured cd (ms/token) from H0")
    parser.add_argument("--cv", type=float, default=3.0,
                        help="Measured cv (ms/token) from H0")
    parser.add_argument("--params", default=None,
                        help="Path to params_measured.json (H0 output); overrides alpha/cd/cv")
    parser.add_argument("--prompts", default="prompts.txt")
    parser.add_argument("--n-prompts", type=int, default=None)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="UCB exploration coefficient")
    parser.add_argument("--p-min", type=float, default=0.3,
                        help="SpecDec++ threshold p_min")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon for epsilon-greedy")
    parser.add_argument("--sled-timeout", type=float, default=100.0,
                        help="SLED timeout threshold (ms)")
    parser.add_argument("--k-init", type=int, default=5,
                        help="SLED initial k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rejection-sampling", action="store_true",
                        help="Use full rejection sampling (sends draft log-probs to server)")
    parser.add_argument("--measure-cd", action="store_true")
    parser.add_argument("--measure-rtt", action="store_true")
    args = parser.parse_args()

    load_draft_model(args.draft_model)

    if args.measure_cd:
        cd = measure_cd()
        print(f"Measured cd = {cd:.2f} ms/token")
        return

    if args.measure_rtt:
        d = measure_rtt(args.server)
        print(f"Estimated one-way delay = {d:.2f} ms")
        return

    import json
    run_experiment(args)


if __name__ == "__main__":
    main()
