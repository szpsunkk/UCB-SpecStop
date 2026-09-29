#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT_DIR"

python -m src.sim_s1_phase_transition
python -m src.sim_s2_latency_table
python -m src.sim_s3_regret
python -m src.sim_s4_voi
python -m src.sim_s5_robustness

echo "All simulation experiments completed. Outputs are under outputs/."
