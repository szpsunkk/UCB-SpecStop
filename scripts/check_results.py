"""Compare regenerated Table 4 results with the preserved numerical reference."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(actual, expected, path="result"):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(path + ": keys differ")
        for name in expected:
            compare(actual[name], expected[name], path + "." + name)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(path + ": lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, "%s[%d]" % (path, index))
    else:
        # SciPy 1.10 -> 1.13 changes the t(7) quantile in the ninth decimal.
        # Keep strict checks on estimates/p-values; allow 1e-8 ms for CI bounds.
        ci_bound = path.rsplit(".", 1)[-1] in {"ci_low", "ci_high", "cost_ci_low", "cost_ci_high"}
        try:
            same = math.isclose(float(actual), float(expected), rel_tol=1e-10,
                                abs_tol=1e-8 if ci_bound else 1e-10)
        except (ValueError, TypeError):
            same = actual == expected
        if not same:
            raise AssertionError(path + ": value differs")


def read(path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/table4")
    args = parser.parse_args()
    expected = ROOT / "data/table4/expected"
    for reference in sorted(expected.iterdir()):
        compare(read(args.out_dir / reference.name), read(reference), reference.name)
        print("PASS " + reference.name)


if __name__ == "__main__":
    main()
