#!/usr/bin/env bash
# run_all_suites.sh — Run experiments for all three draft/cloud model pairs.
#
# Usage on Jetson:
#   chmod +x hardware/run_all_suites.sh
#   ./hardware/run_all_suites.sh
#
# Before running:
#   - Cloud server must already be started for each suite (see comments below)
#   - measure_params.py must have been run for each model pair to produce params
#
# Environment variables you can override:
#   SERVER     — cloud server address       (default: http://192.168.3.72:8000)
#   PROMPTS    — path to prompts file       (default: hardware/prompts.txt)
#   N_PROMPTS  — number of prompts to use   (default: 500)
#   N_ROUNDS   — rounds per strategy in R4  (default: 200)
#   EXP        — which experiments to run   (default: all)

set -euo pipefail

SERVER=${SERVER:-"http://192.168.3.72:8000"}
PROMPTS=${PROMPTS:-"hardware/prompts.txt"}
N_PROMPTS=${N_PROMPTS:-500}
N_ROUNDS=${N_ROUNDS:-200}
EXP=${EXP:-"all"}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# Each suite needs its own measured-params file (run measure_params.py for each)
declare -A PARAMS=(
    [qwen]="outputs/hardware/params_qwen.json"
    [llama]="outputs/hardware/params_llama.json"
    [phi]="outputs/hardware/params_phi.json"
)

# Cloud models — you must restart cloud_server.py with the right model before each suite
declare -A CLOUD_MODEL=(
    [qwen]="Qwen/Qwen2.5-7B-Instruct"
    [llama]="meta-llama/Llama-3.1-8B-Instruct"
    [phi]="microsoft/Phi-3-small-128k-instruct"
)

SUITES=(qwen llama phi)

for SUITE in "${SUITES[@]}"; do
    log "==============================="
    log "Starting suite: $SUITE"
    log "  Draft model : see MODEL_SUITES[$SUITE] in run_revised_experiments.py"
    log "  Cloud model : ${CLOUD_MODEL[$SUITE]}"
    log "  Params file : ${PARAMS[$SUITE]}"
    log "==============================="

    PARAMS_FILE="${PARAMS[$SUITE]}"
    [[ -f "$PARAMS_FILE" ]] || die \
        "Params file not found: $PARAMS_FILE
         Run:  python hardware/measure_params.py \\
                   --draft-model <draft> --server $SERVER \\
                   --prompts $PROMPTS \\
               then rename outputs/hardware/params_measured.json to $PARAMS_FILE"

    log "Confirm the cloud server is running ${CLOUD_MODEL[$SUITE]}.  Press ENTER to continue."
    read -r

    python hardware/run_revised_experiments.py \
        --suite        "$SUITE" \
        --server       "$SERVER" \
        --params       "$PARAMS_FILE" \
        --prompts      "$PROMPTS" \
        --n-prompts    "$N_PROMPTS" \
        --n-rounds     "$N_ROUNDS" \
        --exp          "$EXP"

    log "Suite $SUITE complete.  Outputs: outputs/hardware_revised/$SUITE/"
done

log "All suites complete."
