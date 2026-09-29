# Method mapping

Table 4 is generated from `experiments/table4/analyze.py`, with current paper
labels applied by `scripts/build_table4.py`. Internally `sn_warm` remains the
paper's practical UCB-SpecStop SN variant.

| Paper label | Internal identifier | Rule |
| --- | --- | --- |
| B1 Fixed k=5 | `fixed_k5` | Constant depth 5 |
| B2 Greedy k=1 | `greedy` | Constant depth 1 in this calibration |
| B3 SpecDec++ adaptation | `specdec_pp_threshold` | Acceptance-threshold adaptation with p_min=0.3; k=7 here |
| B4 Idealized selector | `b4_idealized` | Geometric yield, idealized constant cost parameters |
| B5 Constant-cost geometric | `b5_calibrated_geometric` | Fitted scalar drafting/verification costs and geometric yield |
| B6 Calibration-fixed | `b6_calibrated_fixed` | Minimum cost/yield from the measured calibration grid |
| B7 Per-round UCB | `per_round_ucb_warm` | Mean of per-round cost/yield ratios plus exploration |
| B8 Throughput-UCBSpec (common bonus) | `ucbspec_release_warm` | Throughput adaptation; common exploration term |
| B9 Throughput-UCBSpec (arm bonus) | `ucbspec_arm_bonus_warm` | Throughput adaptation; arm-specific exploration term |
| UCB-SpecStop | `sn_warm` | Ratio-of-sums SN index, beta=1, shared calibration |
| Target-only streaming | `direct_target` | Target autoregression streamed in one request; no edge drafting |
| Edge-only | `edge_only` | Draft model autoregression with no target calls |

B3 is not a reproduction of SpecDec++'s trained acceptance predictor. B8/B9
are adaptations to this feedback protocol, not end-to-end BanditSpec service
reproductions. B4/B5 are model-based controls. B5's definition here differs from
the older measured-cost geometric control; avoid reusing its name across the
two protocols without specifying the cost model.

Relevant sources: [speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html),
[SpecDec++](https://arxiv.org/abs/2405.19715),
[BanditSpec](https://proceedings.mlr.press/v267/hou25h.html), and
[UCB1](https://doi.org/10.1023/A:1013689704352).

All methods and parameters are retained, including weaker and stronger
comparators. The release neither retunes baselines nor selects observations to
change rankings. `src/cgp_*`, prefix predictors and structured learners are
included because the archived runtime imports them; they are exploratory and
are not the method labelled UCB-SpecStop in the latest core table.
