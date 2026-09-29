# Release validation — 2026-09-29

Validated in a newly created Python 3.12 environment, with the pinned CPU
requirements and optional PyTorch 2.4.1+cpu / Transformers 4.46.3 runtime.
No GPU inference requests or new hardware measurements were performed during
this code-publication task.

| Check | Result |
| --- | --- |
| Package dependency check | No broken requirements |
| Statistical and release tests | 8 passed |
| Offline runtime tests (core, KV, EOS, graph mocks, proxy, precision and capability) | 115 passed |
| S1–S5 smoke simulation | Completed with 100 rounds, 3 trials, 1,000 robustness samples |
| Latest Table 4 | All 576 requests audited; all four numerical reference files matched |
| Table 4 inference | Eight original contrasts retained; scoring and Holm results reproduced |
| Raw measurement preservation | 19 request/round or replay input files byte-identical to source |
| Exact frozen-input replay | Three measured grids and one +350 ms counterfactual completed |
| Markov independent recomputation | All 1,608 models solved; no archived classification/cost mismatches |
| Markov boundary counts | 401 both-condition passes, 297 continuation cases, 30 prefix-invalid bootstrap inputs retained |
| Supplement plot generation | S1 and S7 generated from included source data |
| Hardware CLI and plan preparation | Help commands, source compilation and fresh-plan checks passed |
| Deployment/credential scan | No unresolved findings; checksum and upstream commit detections classified separately |

The Markov maximum cost discrepancy is below 1.2e-13 ms/token. Table 4 CI
endpoints differ from the historical SciPy version by less than 3.3e-9 ms;
reported values and inferential decisions are unchanged. The regression checker
allows 1e-8 ms absolute error for CI endpoints only.

The original proxy shutdown order blocked on active connections under Python
3.12. Cancelling handlers before waiting for server closure fixes the tested
shutdown path without changing the forwarding-delay algorithm. A legacy fake
model's EOS fixture was also synchronized with its generation configuration.

GitHub workflows are provided for CPU reproduction and offline runtime tests.
Local passing checks do not claim a new hardware deployment or exact timing on
another device. Detailed scope is in REPRODUCIBILITY.md; security review scope
and classified scanner findings are in SECURITY_REVIEW.json.
