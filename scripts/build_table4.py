"""Build the current compact Table 4 columns from regenerated measurements."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("edge_only", "Edge-only (0.5B)"),
    ("direct_target", "Target-only streaming (7B)"),
    ("fixed_k5", "B1: Fixed k=5"),
    ("greedy", "B2: Greedy (k=1)"),
    ("specdec_pp_threshold", "B3: SpecDec++ adaptation"),
    ("b4_idealized", "B4: Idealized selector"),
    ("b5_calibrated_geometric", "B5: Constant-cost geometric"),
    ("b6_calibrated_fixed", "B6: Calibration-fixed"),
    ("per_round_ucb_warm", "B7: Per-round UCB"),
    ("ucbspec_release_warm", "B8: Throughput-UCBSpec (common bonus)"),
    ("ucbspec_arm_bonus_warm", "B9: Throughput-UCBSpec (arm bonus)"),
    ("sn_warm", "UCB-SpecStop"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/table4")
    args = parser.parse_args()
    with (args.out_dir / "table4_all_metrics.csv").open(newline="") as handle:
        rows = {(r["condition"], r["method"]): r for r in csv.DictReader(handle)}
    output = []
    for method, label in METHODS:
        values = {"method": label}
        for condition in ("lan_direct", "lan_plus150"):
            row = rows[(condition, method)]
            cost = float(row["cost_ms_token"])
            baseline = float(rows[(condition, "per_round_ucb_warm")]["cost_ms_token"])
            values[condition + "_cost_ms_token"] = cost
            values[condition + "_reduction_pct"] = "" if method == "edge_only" else 100 * (1 - cost / baseline)
            values[condition + "_calls"] = float(row["generation_interactions"])
        output.append(values)
    with (args.out_dir / "table4_compact.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    lines = ["| Method | Wi-Fi cost | Reduction % | Calls | Proxy cost | Reduction % | Calls |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in output:
        cells = [row["method"]]
        for condition in ("lan_direct", "lan_plus150"):
            cells += ["%.2f" % row[condition + "_cost_ms_token"],
                      "—" if row[condition + "_reduction_pct"] == "" else "%+.2f" % row[condition + "_reduction_pct"],
                      "%.1f" % row[condition + "_calls"]]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Cost is the equal-weight mean of eight session time/token ratios (ms/token).",
              "Reduction uses the ratio of these means relative to B7, not the mean of paired percentages.",
              "Calls count generation interactions; they are not pure network latency.",
              "Edge-only differs in output quality, so its reduction is not reported.",
              "See latency_contrasts.json for all eight prespecified tests and Holm correction.", ""]
    (args.out_dir / "table4_compact.md").write_text("\n".join(lines))
    print("Wrote " + str(args.out_dir / "table4_compact.md"))


if __name__ == "__main__":
    main()
