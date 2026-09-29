from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
RAW_DIR = OUTPUT_DIR / "raw"


@dataclass(frozen=True)
class BaseConfig:
    alpha: float = 0.7
    cd: float = 1.0
    cv: float = 0.5
    k_max: int = 20
    seed: int = 42


@dataclass(frozen=True)
class RegretConfig:
    t_rounds: int = 10000
    n_trials: int = 1000
    beta: float = 1.0
