from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, out_base: Path, dpi: int = 300) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
