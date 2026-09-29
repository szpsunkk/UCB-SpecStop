# 2026-09-29 reproducibility release

- Selected the EOS-corrected frozen runtime actually used by the latest Table 4.
- Included formal request/round records, shared calibration, scoring, reference
  statistics, and B1–B9 label mapping.
- Added CPU commands for Table 4, exact replay, Markov recomputation and S1/S7.
- Preserved S1–S5 simulation algorithms and deterministic defaults; documented
  their historical protocol and objective boundaries.
- Replaced deployment identifiers with public placeholders and CLI configuration.
- Added explicit dependency files, fresh-plan generation, integrity checks,
  numerical regression checks, offline tests and GitHub Actions.
- Fixed proxy shutdown ordering for Python 3.12, which waits for live
  connections before completing server closure. Forwarding delays are unchanged.
  Updated the old fake-model test fixture to set the authoritative native EOS.
- Kept all outputs outside the archived inputs. Excluded paper submissions,
  reviewer correspondence, local device automation, model weights and caches.
- Preserved the existing remote main branch and historical release tag. This
  branch starts a separate history containing only curated public files.

The algorithm rules, experimental observations, comparator tuning, scoring
function and statistical family were not changed to improve reported results.
