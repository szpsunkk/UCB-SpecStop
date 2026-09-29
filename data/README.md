# Published input data

Run `python scripts/verify_data.py` before analysis. `SHA256SUMS.json` records
the published bytes. Original-source digests and changes needed for portability
are tracked separately in `docs/source_provenance.json`.

| Directory | Unit and contents |
| --- | --- |
| `table4/hardware_run/capability-01..08` | Request metrics, round metrics, output-equality checks, execution manifest and frozen plan for each formal session |
| `table4/protocol` | Eight prompt blocks and plans; common 27-cell calibration; static-selector aliases |
| `table4/expected` | Four numerical reference files for regression checks |
| `table4/gsm8k_test.jsonl` | Original public test split, indexed by `source_index` in `gold_answers.json` |
| `exact_replay` | Three complete frozen-input potential-outcome grids; each row identifies seeded run, round and measured draft depth |
| `markov` | 8 nominal and 1,600 bootstrap model records; all inputs and archived solver outputs |
| `figure_s1` | Full crossing summaries; the plot selects the original four k=1/k=2 rows |
| `figure_s7` | Full 5,000-round historical proxy curves, including all three displayed strategies |

In Table 4 request records, `measurement_session`, `condition_id`, `method`,
and `prompt_id` form the unique key. `completion_latency_ms` and
`generated_tokens` define the cost. `delivered_token_ids_json` and `stop_reason`
support exact output checks. Round records use the same key and preserve
selected depth, feedback censoring, time components and delivered token counts.
Units are encoded in column/field names. Empty verification fields for direct
target and edge-only are not zeros; those methods have no verification RPC.

The archived plan's `code_sha256` values refer to the original measurement
snapshot. Public portability changes are described in the provenance manifest.
Use `scripts/prepare_hardware.py` to generate new plans for the public code;
do not execute an archived plan after manually bypassing its code-hash check.

Published request/round JSONL files and all three exact-replay CSVs are
byte-identical to the original measurements (19 files). Deployment metadata
files were sanitized and their public hash links were refreshed. Full energy
telemetry, model weights and private machine configuration are not included.
