"""Synthetic diagnostic for ratio-of-expectations objective consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    probabilities: Sequence[float]
    numerator_by_arm: Dict[int, Sequence[float]]
    yield_by_arm: Dict[int, Sequence[float]]


SCENARIOS = (
    Scenario(
        name="ranking_reversal",
        probabilities=(0.5, 0.5),
        numerator_by_arm={1: (10.0, 10.0), 2: (12.0, 12.0)},
        yield_by_arm={1: (1.0, 2.0), 2: (1.0, 3.0)},
    ),
    Scenario(
        name="aligned_ranking",
        probabilities=(0.5, 0.5),
        numerator_by_arm={1: (10.0, 10.0), 2: (11.0, 11.0)},
        yield_by_arm={1: (1.0, 2.0), 2: (1.0, 3.0)},
    ),
    Scenario(
        name="near_tie_ratio_of_expectations",
        probabilities=(0.5, 0.5),
        numerator_by_arm={1: (10.0, 10.0), 2: (13.3, 13.3)},
        yield_by_arm={1: (1.0, 2.0), 2: (1.0, 3.0)},
    ),
    Scenario(
        name="nearly_flat_yield",
        probabilities=(0.95, 0.05),
        numerator_by_arm={1: (10.0, 10.0), 2: (10.6, 10.6)},
        yield_by_arm={1: (1.0, 2.0), 2: (1.0, 3.0)},
    ),
)


def validate_scenario(scenario: Scenario) -> None:
    probabilities = np.asarray(scenario.probabilities, dtype=float)
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError(f"{scenario.name}: probabilities must sum to one")
    if np.any(probabilities < 0):
        raise ValueError(f"{scenario.name}: probabilities must be nonnegative")
    first = np.asarray(scenario.yield_by_arm[1], dtype=float)
    second = np.asarray(scenario.yield_by_arm[2], dtype=float)
    if np.any((first < 1) | (first > 2)):
        raise ValueError(f"{scenario.name}: arm-1 yield violates prefix bounds")
    if np.any((second < first) | (second > 3)):
        raise ValueError(f"{scenario.name}: arm-2 yield violates prefix nesting")


def population_rows(scenarios: Sequence[Scenario]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        validate_scenario(scenario)
        probabilities = np.asarray(scenario.probabilities, dtype=float)
        metrics = {}
        for arm in sorted(scenario.numerator_by_arm):
            numerator = np.asarray(scenario.numerator_by_arm[arm], dtype=float)
            token_yield = np.asarray(scenario.yield_by_arm[arm], dtype=float)
            ratio_of_expectations = float(
                (probabilities @ numerator) / (probabilities @ token_yield)
            )
            expectation_of_ratio = float(
                probabilities @ (numerator / token_yield)
            )
            metrics[arm] = (ratio_of_expectations, expectation_of_ratio)
            rows.append(
                {
                    "scenario": scenario.name,
                    "arm": arm,
                    "ratio_of_expectations": ratio_of_expectations,
                    "expectation_of_ratio": expectation_of_ratio,
                }
            )
        roe_best = min(metrics, key=lambda arm: metrics[arm][0])
        eor_best = min(metrics, key=lambda arm: metrics[arm][1])
        for row in rows[-len(metrics) :]:
            row["ratio_of_expectations_best_arm"] = roe_best
            row["expectation_of_ratio_best_arm"] = eor_best
    return pd.DataFrame(rows)


def finite_sample_rows(
    scenarios: Sequence[Scenario],
    sample_sizes: Sequence[int],
    replications: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    population = population_rows(scenarios)
    rows = []
    for scenario in scenarios:
        probabilities = np.asarray(scenario.probabilities, dtype=float)
        truth = population[population["scenario"] == scenario.name].iloc[0]
        for sample_size in sample_sizes:
            ratio_errors = 0
            mean_ratio_errors = 0
            for _ in range(replications):
                outcomes = rng.choice(
                    len(probabilities), size=sample_size, p=probabilities
                )
                ratio_estimates = {}
                mean_ratio_estimates = {}
                for arm in sorted(scenario.numerator_by_arm):
                    numerator = np.asarray(
                        scenario.numerator_by_arm[arm], dtype=float
                    )[outcomes]
                    token_yield = np.asarray(
                        scenario.yield_by_arm[arm], dtype=float
                    )[outcomes]
                    ratio_estimates[arm] = float(
                        numerator.sum() / token_yield.sum()
                    )
                    mean_ratio_estimates[arm] = float(
                        np.mean(numerator / token_yield)
                    )
                ratio_errors += int(
                    min(ratio_estimates, key=ratio_estimates.get)
                    != int(truth["ratio_of_expectations_best_arm"])
                )
                mean_ratio_errors += int(
                    min(mean_ratio_estimates, key=mean_ratio_estimates.get)
                    != int(truth["expectation_of_ratio_best_arm"])
                )
            rows.append(
                {
                    "scenario": scenario.name,
                    "sample_size": sample_size,
                    "replications": replications,
                    "ratio_of_sums_selection_error": ratio_errors / replications,
                    "mean_of_ratios_selection_error": (
                        mean_ratio_errors / replications
                    ),
                }
            )
    return pd.DataFrame(rows)


def latex_table(population: pd.DataFrame) -> str:
    labels = {
        "ranking_reversal": "Ranking reversal",
        "aligned_ranking": "Aligned ranking",
        "near_tie_ratio_of_expectations": "Near RoE tie",
        "nearly_flat_yield": "Nearly flat yield",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Synthetic objective-consistency diagnostic. RoE is the ratio of expectations; EoR is the expectation of the per-round ratio. Values are exact population costs, and lower is better.}",
        r"\label{supp:tab:objective_consistency}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Scenario & RoE$_1$ & RoE$_2$ & EoR$_1$ & EoR$_2$\\",
        r"\midrule",
    ]
    for scenario, group in population.groupby("scenario", sort=False):
        by_arm = group.set_index("arm")
        lines.append(
            f"{labels[scenario]} & "
            f"{by_arm.loc[1, 'ratio_of_expectations']:.3f} & "
            f"{by_arm.loc[2, 'ratio_of_expectations']:.3f} & "
            f"{by_arm.loc[1, 'expectation_of_ratio']:.3f} & "
            f"{by_arm.loc[2, 'expectation_of_ratio']:.3f}\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", default="outputs/revision2/objective_consistency"
    )
    parser.add_argument("--latex-table")
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    population = population_rows(SCENARIOS)
    finite_sample = finite_sample_rows(
        SCENARIOS,
        sample_sizes=(20, 100, 1000),
        replications=args.replications,
        seed=args.seed,
    )
    population_path = output / "population_objectives.csv"
    finite_path = output / "finite_sample_selection.csv"
    population.to_csv(population_path, index=False)
    finite_sample.to_csv(finite_path, index=False)

    if args.latex_table:
        latex_path = Path(args.latex_table).expanduser().resolve()
        latex_path.write_text(latex_table(population), encoding="utf-8")
    else:
        latex_path = None

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "analysis": "objective_consistency",
        "seed": args.seed,
        "replications": args.replications,
        "sample_sizes": [20, 100, 1000],
        "scope": "synthetic estimator-target diagnostic, not hardware evidence",
        "prefix_constraints": "A_1 in [1,2], A_2 in [A_1,3] for every outcome",
        "outputs": {
            str(population_path): digest(population_path),
            str(finite_path): digest(finite_path),
        },
    }
    if latex_path is not None:
        manifest["outputs"][str(latex_path)] = digest(latex_path)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
