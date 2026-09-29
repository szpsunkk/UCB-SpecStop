# Termination correction: this campaign is diagnostic, not the final capability result

During collection, inspection of the locally installed target's `generation_config.json` confirmed two native EOS IDs, `[151645, 151643]`. The inherited Table 4 runtime checked only `tokenizer.eos_token_id` (151645). This is consistent across the inherited cloud/distributed protocol but unsuitable for a comparison advertised as native answer completion against edge-only, which stops at its native end-of-text token.

The issue is directly observed in archived target-only output for `gsm8k-test-1117`: token 151643 occurs at zero-based position 224, but the inherited generator continues until the 512-token cap. This can add unrelated continuation, alter answer extraction and overcharge latency. It also explains why an explicit numeric-answer stop alone was insufficient.

The running campaign was stopped by sending SIGINT only to the verified, task-owned edge measurement child (PID 732368). The supervisor and campaign driver perform their normal archival/cleanup. Its interruption record is retained and must not be described as an algorithm execution failure in the corrected campaign. All completed and partial raw data, the old source snapshot, and frozen manifests remain intact.

A separate corrected campaign will rerun every method, both paths and all 32 originally selected questions. No question, arm rule, calibration, comparison, statistical test or generation budget is chosen using observed performance. Both native target EOS tokens will be honored in direct generation, verification, draft termination and delivery; EOS tokens will not count as delivered answer tokens. The original paper and its experiment code are not changed.

The final report must disclose this implementation correction and distinguish the pre-correction requests from the fully repeated evaluation. Do not combine unaffected old rows with corrected rows or select a favorable subset.
