# Reproduction scope and numerical conventions

## Latest answer-completion Table 4

The acquisition code comes from the frozen September 28 EOS-corrected snapshot,
not the later mutable research workspace. This release preserves the 8 formal
sessions and excludes pilot and pre-correction records. There are 72 requests
per session and 576 in total. Each session has four distinct questions; the 32
questions are drawn from GSM8K test. Exact speculative and direct-target outputs
are checked token-by-token, including termination. Edge-only is a separate small
model run and has different outputs.

The token cap is 512, graph-cache length is 1,024, and the common calibration has
27 observations per depth over depths 1–10. Calibration comes from the earlier
experiment and is not refit on evaluation questions. Keep this transfer scope
when interpreting performance. All online methods receive identical calibration.
Aliases selecting the same fixed depth reuse a measurement; they are not extra
replicates. Calibration acquisition, model loading, and graph setup are outside
request latency.

Each cost is the sum of completion time divided by the sum of delivered tokens
within a session, followed by an equal-weight mean over eight sessions. It is
not the average of round ratios. Statistical uncertainty uses paired session
differences with 7 degrees of freedom and one Holm family of eight tests.
Quality uses 32 distinct questions, not 544 duplicate exact-method requests.
The scoring function and its strict sensitivity analysis are retained verbatim.
The paired quality bootstrap uses 50,000 resamples and seed 2026092802.
Full prespecification: [STATISTICS.md](STATISTICS.md).

The `table4_compact` reduction is `100 * (B7_mean - method_mean) / B7_mean`.
`mean_paired_cost_reduction_pct` in the contrast JSON uses per-session reductions
and is intentionally a different estimand. Verification RPC contains server
compute time. RPC minus reported server verification time is a protocol
residual, not pure network latency. Target-only has no verification RPC.

The +150 ms path adds 75 ms in each direction through the provided proxy. It
is not a second physical wireless deployment. The supplied measurement tables
remain descriptive of the original testbed.

## Original runtime and current validation environment

Original edge manifests record Python 3.10.12 and the model/graph configuration;
original target manifests record PyTorch 2.4.1+cu121, Transformers 4.46.3 and an
RTX 3090 Ti. Edge hardware is Jetson Orin Nano Super. Preserve vendor-compatible
Jetson PyTorch rather than installing a desktop wheel. Per-session sanitized
`run_manifest.json` files retain the recorded library versions and numeric
settings. Portable CPU dependencies are pinned in `requirements.txt` and a
fully resolved CPU environment is archived in `docs/requirements-cpu-lock.txt`.
Use `python -m pip install -r docs/requirements-cpu-lock.txt` to install that
exact CPU dependency set instead of resolving transitive dependencies afresh.

The draft is Qwen2.5-0.5B Base, the target Qwen2.5-7B-Instruct. The acquisition
did not independently record the model weight revision. Tokenizer hashes and
model configuration checks are available, but cannot substitute for weight
commit IDs. Fresh hardware timing and possibly token decisions can vary with
weights, CUDA, GPU model and numerics. Do not present an offline reanalysis as
a new hardware acquisition.

## Historical experiments

The three `data/exact_replay/*.csv` inputs each contain frozen-input potential
outcomes for 10 seeded runs. The default policy analysis splits calibration
and evaluation by run (fraction 0.3), with seed 20260818. These are within-session
runs, not 10 independent hardware sessions. Remote is an ICMP-delay replay,
not a remote HTTP target service. `counterfactual350` adds 350 ms to the Wi-Fi
round-cost numerator; it is not a new measurement.

The 1,608 full-precision Markov inputs retain all 30 prefix-invalid bootstrap
arrays. Independent recomputation must recover 401 both-condition passes,
297 bootstrap instances with continuation probability above 0.05, and all 30
invalid cases. Matching ordered thresholds and unrestricted costs in these
archived examples does not establish unconditional threshold optimality.

Figure S1 uses the original four k=1/k=2 crossing summaries across all 10 runs,
including the three calibration runs. These intervals are retrospective.
Figure S7 reproduces historical acceptance-proxy curves; the score has units
ms/token and is not accumulated elapsed time or population pseudo-regret.

S1–S5 simulations use synthetic geometric yields and shared defaults in
`src/config.py` (seed 42). S3 defaults to 10,000 rounds and 1,000 trials; S5 uses
100,000 samples per cell. Environment overrides in the quick command reduce
these settings. S4 reports a statewise objective diagnostic; it must not be
interpreted as aggregate oracle value under a common additive delay.

## Included and excluded material

The release includes latest Table 4 raw request/round observations and original
GSM8K test data, three historical exact-replay grids, Markov inputs, S1/S7 curve
data, the frozen runtime dependency graph and relevant offline tests. Historical
simulation output is regenerated rather than checked in.

It does not bundle all exploratory controller campaigns, full energy telemetry,
concurrency data, all model/platform variants, every paper figure, model weights,
or private paper/reviewer files. Scripts with additional runtime dependencies
remain for compatibility; their presence is not a claim that every historical
study can be reconstructed from this release alone. The current table's request
and communication metrics are reproducible from the included inputs; resource
telemetry metrics are outside this release's verified scope.

## Integrity after sanitization

Raw request and round measurements are preserved. Addresses, local paths and
host identities in deployment manifests are replaced. Hash links affected by
that replacement are recomputed for the public archive, while original source
digests are retained in `docs/source_provenance.json`. No claim is made that
sanitized metadata are byte-identical to the original metadata.

`python scripts/verify_data.py` checks all published input bytes. The default
reproduction wrapper runs it before analysis. `scripts/check_results.py` compares
all four published numerical reference files with relative and absolute
tolerances of 1e-10, except CI endpoints allow 1e-8 ms absolute error. SciPy
1.10 versus 1.13 changes the t(7) quantile slightly; observed CI drift is below
3.3e-9 ms and has no effect on printed values or conclusions.
A fresh hardware campaign will have new timings and should
not be compared with these values as an exact numerical regression test.
