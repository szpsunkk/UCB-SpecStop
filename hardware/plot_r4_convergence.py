"""Standalone helper to (re-)render the two extra hardware figures.

The plotting functions live in run_revised_experiments.py — this script is a
thin wrapper for the post-hoc workflow ("I have csvs from a finished run, just
redraw the figures without re-running anything").

Reads:
    outputs/hardware_revised/<suite>/r4_strategy_compare.csv
    outputs/hardware_revised/<suite>/r5_round_log.csv

Writes:
    fig_r4_with_oracle.{png,pdf}
    fig_convergence_vs_round.{png,pdf}

Usage:
    python hardware/plot_r4_convergence.py --suite qwen
    python hardware/plot_r4_convergence.py --suite llama --regret-delay 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

# run_revised_experiments lives next to this file; reuse its renderer.
import run_revised_experiments as rre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="qwen")
    ap.add_argument("--regret-delay", type=int, default=None,
                    help="d (ms) used in R5. We snap to the closest delay in "
                         "the R3 grid to pick fixed-k reference lines. "
                         "Default: read oracle_C from r5_round_log.csv and "
                         "best-effort guess (matches R4 grid by default).")
    ap.add_argument("--fixed-ref-delay", type=int, default=None,
                    help="Override which R4 delay's fixed-k cells become the "
                         "horizontal reference lines. Useful when the auto "
                         "snap picks the wrong column.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "outputs" / "hardware_revised" / args.suite
    if not out_dir.exists():
        raise SystemExit(f"missing suite dir: {out_dir}")

    if args.fixed_ref_delay is not None:
        # Bypass the snap: feed the override straight through by tricking the
        # R3 grid to be a single value.
        rre.render_extra_figures(out_dir,
                                 regret_delay=args.fixed_ref_delay,
                                 r3_grid=[args.fixed_ref_delay])
    else:
        regret_delay = args.regret_delay
        if regret_delay is None:
            # Default to 55 ms: it's the closest R3-grid delay to d_c for both
            # the qwen and llama suites in our hardware setup. Override with
            # --regret-delay if your run shifted.
            regret_delay = 55
        rre.render_extra_figures(out_dir, regret_delay=regret_delay)


if __name__ == "__main__":
    main()
