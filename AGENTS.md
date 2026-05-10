# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the simulation code used for S1-S5. Core math lives in `src/core.py`, shared defaults and output paths in `src/config.py`, and each experiment entry point follows the `sim_sN_*` naming pattern. `hardware/` contains the Jetson/cloud experiment stack, including `run_revised_experiments.py`, `cloud_server.py`, deployment notes, and offline validation. Generated artifacts belong under `outputs/` (`figures/`, `tables/`, `raw/`, `hardware_revised/`). Keep paper notes and reviewer docs at the repository root.

## Build, Test, and Development Commands
Create a local environment and install simulation dependencies with `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Run the full simulation pipeline with `bash run_all.sh`, or execute one study at a time, for example `python -m src.sim_s3_regret`. Use reduced settings for quick checks: `REGRET_T_ROUNDS=2000 REGRET_N_TRIALS=100 ROBUSTNESS_SAMPLES=20000 bash run_all.sh`. For hardware work, install `pip install -r hardware/requirements_hw.txt` and validate logic offline with `python hardware/validate_hw.py` or `python -m pytest hardware/validate_hw.py -v`.

## Coding Style & Naming Conventions
This repository is Python-first. Use 4-space indentation, PEP 8 spacing, and type hints where the surrounding code already uses them. Prefer `snake_case` for modules, functions, variables, and output filenames; use `PascalCase` for classes such as `UCBSpecStop`. Keep new experiment scripts consistent with existing names like `sim_s4_voi.py` or `plot_r4_convergence.py`. There is no committed formatter or linter config, so keep imports tidy and avoid introducing style-only churn.

## Testing Guidelines
There is no separate `tests/` package; `hardware/validate_hw.py` is the main invariant and smoke-test suite. Run it before changing bandit logic, verification flow, or hardware interfaces. For simulation edits, run the affected `python -m src...` module and confirm outputs land in the expected `outputs/` subdirectory. Preserve deterministic seeds unless a change explicitly studies randomness.

## Commit & Pull Request Guidelines
Recent history uses short subjects (`v1`, `v2`), but contributors should write descriptive, imperative commit messages such as `hardware: tighten R4 delay checks`. Pull requests should state which experiment paths were touched, list the commands used for validation, and note any regenerated figures, tables, or large output files. Avoid committing raw output blobs unless they are required to reproduce a paper result.
