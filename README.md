# UCB-SpecStop: reproducible experiments

Code for *Optimal Stopping Theory for Speculative Decoding Under Communication
Constraints*. This release accompanies the September 2026 revision. It includes
the frozen answer-completion runtime, recorded experiment inputs, offline
analysis, simulation entry points, and hardware instructions.

The previous `main` branch and `v1.0` tag are retained. This publication branch
has a new root commit so that local research files and previous deployment
metadata are not inherited into its history.

## Quick start (CPU only)

Use Python 3.12 on Linux; no GPU or model weights are needed for these commands.

```bash
git clone --branch reproducibility-20260929 https://github.com/szpsunkk/UCB-SpecStop.git
cd UCB-SpecStop
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/reproduce.py table4
```

This validates the public archive, re-scores all 32 questions, checks all 576
requests and exact target-output equality, recomputes the eight paired session
contrasts with the original Holm family, and checks the results against the
preserved reference. Results go to `outputs/table4/`; the archived inputs are
not overwritten. Read `table4_compact.md` for the current core-table layout and
`table4_all_metrics.csv` for all measured metrics.

## Reproduction routes

| Task | Command | Inputs and scope |
| --- | --- | --- |
| Latest Table 4 | `python scripts/reproduce.py table4` | 576 recorded requests, 8 restarted sessions, 32 GSM8K questions |
| Cold-start exact replay | `python scripts/reproduce.py exact-replay` | Three measured potential-outcome grids; +350 ms is a constructed offset |
| Conditional Markov extension | `python scripts/reproduce.py markov` | Independent solution of 8 nominal and 1,600 archived bootstrap models |
| Supplement Figures S1 and S7 | `python scripts/reproduce.py figures` | Preserved crossing summaries and historical curves |
| Original S1–S5 simulations | `python scripts/reproduce.py simulations` | Synthetic model; historical figure names, not current manuscript numbering |
| Fast simulation check | `python scripts/reproduce.py simulations --quick` | Reduced sample sizes; not publication-scale evidence |
| CPU smoke suite | `bash scripts/smoke.sh` | Simulation, statistical tests, archive integrity, Table 4 numerical regression |
| New hardware acquisition | [Hardware guide](docs/HARDWARE.md) | Jetson edge, CUDA target server, separately downloaded model weights |

`python scripts/reproduce.py all --quick` runs every CPU route with reduced
simulation settings. The Markov and exact-replay routes still use all their
archived inputs. [Reproduction details](docs/REPRODUCIBILITY.md) explain seeds,
metrics, original environments, and what is not included.

## Algorithms and interpretation

`src/baselines.py` contains UCB-SpecStop's ratio-of-sums SN implementation,
the separately scaled `DisplayedRatioLCB`, per-round UCB, throughput-UCBSpec
adaptations, and EXP3 controls. The SN heuristic does not inherit the LCB
guarantee. Table 4 uses SN with shared calibration; it does not use the
exploratory CGP variants also required by the frozen runtime's import graph.

The latest Table 4 compares B1–B9, UCB-SpecStop, edge-only generation, and
target-only streaming. The SN mean costs are 69.77 and 100.15 ms/token on the
two recorded paths, versus 83.42 and 128.76 for per-round UCB. Strong fixed and
throughput controls remain competitive, and target-only is faster in this
testbed. Reproduction preserves these outcomes and all prespecified contrasts.
See [method mapping](docs/METHODS.md) for exact implementation names and
baseline adaptation boundaries.

## Repository layout

```text
src/                   Algorithms, cost model, and synthetic simulations
hardware/              Frozen edge/cloud runtime and offline validation
experiments/table4/    Scoring and session-level statistical analysis
data/table4/           Sanitized request/round records, plans, and references
data/exact_replay/     Historical frozen-input potential outcomes
data/markov/           Full-precision inputs and independent solver
data/figure_s1/        Crossing summaries and plot script
data/figure_s7/        Historical curves and plot script
scripts/               Reproduction, verification, and deployment-plan tools
docs/                  Protocol, provenance, and hardware instructions
outputs/               Generated results (ignored by Git)
```

## Data, privacy, and software terms

Deployment addresses, host identities, private paths, SSH configuration,
credentials, model caches, paper submissions, and reviewer correspondence are
excluded or sanitized. `data/SHA256SUMS.json` authenticates the published bytes;
`docs/source_provenance.json` records their original-source digests. Historical
hashes embedded in archived plans describe the acquisition snapshot, not the
portable release. Newly prepared hardware plans bind to the current code.

This update does not assign a new software license to the original project,
which did not contain one. Third-party material retains its own terms; see
[THIRD_PARTY.md](THIRD_PARTY.md). Model weights are not redistributed. Newly
collected hardware logs may contain deployment metadata; inspect and sanitize
them before publishing. The privacy audit covers this branch, not older
branches retained for continuity.
