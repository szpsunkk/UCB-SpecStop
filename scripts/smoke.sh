#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export REGRET_T_ROUNDS=100
export REGRET_N_TRIALS=3
export ROBUSTNESS_SAMPLES=1000
bash run_all.sh
python -m pytest experiments/table4/test_analysis.py tests/test_release.py -q
python scripts/verify_data.py
python -m experiments.table4.analyze
python scripts/check_results.py
