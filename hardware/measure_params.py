"""
measure_params.py — H0: measure cd, cv, alpha on real hardware.

Run on the Jetson after edge_client.py is set up, with the 3090 server running.

Produces:
  outputs/hardware/params_measured.json   — cd, cv, alpha, per-position alpha
  outputs/hardware/fig_alpha_per_pos.pdf  — Fig. 7 (Assumption 1 validation)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plot_style import apply_ieee_style, save_figure

OUT_DIR = Path(__file__).parent.parent / "outputs" / "hardware"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str):
    global _model, _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(name)
    _model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16
    ).to(_device)
    _model.eval()


def measure_cd(n: int = 500) -> float:
    dummy = _tokenizer("The quick brown fox", return_tensors="pt").input_ids.to(_device)
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            _model.generate(dummy, max_new_tokens=1, do_sample=False,
                            pad_token_id=_tokenizer.eos_token_id)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(latencies))


def measure_alpha_per_position(server: str, prompts: list[str], k_max: int = 10) -> dict:
    """
    For each prompt, generate k_max draft tokens, send to cloud for verification,
    collect per-position accept/reject outcomes.
    """
    position_accepts: dict[int, list[int]] = {i: [] for i in range(k_max)}

    for prompt in prompts:
        input_ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(_device)
        context_ids = input_ids[0].tolist()

        with torch.no_grad():
            out = _model.generate(
                input_ids, max_new_tokens=k_max, do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
            )
        draft_ids = out[0, len(context_ids):].tolist()

        resp = requests.post(
            f"{server}/verify",
            json={"context_ids": context_ids, "draft_ids": draft_ids},
            timeout=30,
        ).json()

        n_acc = resp["n_accepted"]
        for i in range(k_max):
            if i < n_acc:
                position_accepts[i].append(1)
            elif i == n_acc:
                position_accepts[i].append(0)
            # positions beyond rejection point: not observed

    per_pos_alpha = {
        i: float(np.mean(v)) for i, v in position_accepts.items() if len(v) > 0
    }
    return per_pos_alpha


def measure_cv(server: str, n: int = 200) -> float:
    dummy = _tokenizer("The quick brown fox", return_tensors="pt").input_ids.to(_device)
    context_ids = dummy[0].tolist()
    with torch.no_grad():
        draft_ids = _model.generate(
            dummy, max_new_tokens=5, do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )[0, len(context_ids):].tolist()

    times = []
    for _ in range(n):
        resp = requests.post(
            f"{server}/verify",
            json={"context_ids": context_ids, "draft_ids": draft_ids[:1]},
            timeout=10,
        ).json()
        times.append(resp["verify_time_ms"])
    return float(np.median(times))


def plot_alpha_per_pos(per_pos: dict, alpha_fit: float):
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    positions = sorted(per_pos.keys())
    measured = [per_pos[p] for p in positions]
    geometric = [alpha_fit ** (p + 1) for p in positions]

    ax.bar(positions, measured, label="Empirical", color="#1f77b4", alpha=0.7)
    ax.plot(positions, geometric, "r--o", markersize=4,
            label=fr"Geometric $\hat{{\alpha}}={alpha_fit:.2f}$")
    ax.set_xlabel("Draft token position")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Per-position acceptance (Assumption 1 validation)")
    ax.legend(frameon=True)

    save_figure(fig, OUT_DIR / "fig_alpha_per_pos")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--server", default="http://192.168.1.100:8000")
    parser.add_argument("--prompts", default="prompts.txt")
    parser.add_argument("--k-max", type=int, default=10)
    args = parser.parse_args()

    prompts = Path(args.prompts).read_text().strip().split("\n")

    print("Loading draft model...")
    load_model(args.draft_model)

    print("Measuring cd (draft latency)...")
    cd = measure_cd()
    print(f"  cd = {cd:.2f} ms/token")

    print("Measuring cv (verification latency)...")
    cv = measure_cv(args.server)
    print(f"  cv = {cv:.2f} ms/token")

    print(f"Measuring per-position alpha on {len(prompts)} prompts...")
    per_pos = measure_alpha_per_position(args.server, prompts, args.k_max)
    alpha_fit = float(np.mean(list(per_pos.values())))
    print(f"  alpha (mean) = {alpha_fit:.3f}")
    for pos, a in sorted(per_pos.items()):
        print(f"  pos {pos}: {a:.3f}  (geometric: {alpha_fit**(pos+1):.3f})")

    results = {
        "cd_ms": cd,
        "cv_ms": cv,
        "alpha_fit": alpha_fit,
        "per_position_alpha": {str(k): v for k, v in per_pos.items()},
    }
    out_file = OUT_DIR / "params_measured.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_file}")

    plot_alpha_per_pos(per_pos, alpha_fit)
    print(f"Saved Fig. 7 (alpha validation) to {OUT_DIR}/fig_alpha_per_pos.pdf")


if __name__ == "__main__":
    main()
