"""CPU reproduction tasks; no model download or hardware calls."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_data import verify


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["table4", "exact-replay", "markov", "figures", "simulations", "all"])
    parser.add_argument("--quick", action="store_true", help="Reduced Monte Carlo samples for simulation smoke checks")
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("Run without -O: evidence audits use assertions")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    print("Verified %d archived files" % verify(), flush=True)
    tasks = [args.task] if args.task != "all" else ["table4", "exact-replay", "markov", "figures", "simulations"]
    for task in tasks:
        print("Running " + task, flush=True)
        if task == "table4":
            run("-m", "experiments.table4.analyze")
            run("scripts/check_results.py")
            run("scripts/build_table4.py")
        elif task == "exact-replay":
            for label, source, offset in [("wifi", "wifi", 0), ("plus150", "plus150", 0),
                                          ("remote", "remote", 0), ("counterfactual350", "wifi", 350)]:
                run("hardware/analyze_tmc_revision.py", "policies", "--potential-outcomes",
                    "data/exact_replay/" + source + ".csv", "--out-dir", "outputs/exact_replay/" + label,
                    "--added-rtt-ms", str(offset))
        elif task == "markov":
            run("data/markov/recompute.py")
        elif task == "figures":
            run("data/figure_s1/plot_s1.py")
            run("data/figure_s7/replot.py")
        else:
            if args.quick:
                os.environ.update(REGRET_T_ROUNDS="100", REGRET_N_TRIALS="3", ROBUSTNESS_SAMPLES="1000")
            subprocess.run(["bash", "run_all.sh"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
