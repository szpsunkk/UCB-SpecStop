"""Replot Fig. S7 from the verified historical curves; no new observations.

Adapted from hardware/plot_submission1_revision.py's historical_r5 and save.
Dependencies: matplotlib, numpy, pandas. All 5,000 points per curve are kept.
The legacy reference_score_ms column has units ms/token, not elapsed ms.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT.parents[1] / 'outputs/figures/fig_supp_historical_score.pdf')
args = parser.parse_args()
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': True, 'axes.spines.right': True, 'axes.linewidth': 0.6,
    'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 160/72))
fig.subplots_adjust(left=0.145, right=0.99, bottom=39/160, top=0.98, wspace=0.60)
checks = []
for ax, suite, title, letter in zip(axes, ['qwen', 'llama'], ['Qwen', 'LLaMA'], 'ab'):
    frame = pd.read_csv(ROOT / (suite + '_r5_curves.csv'))
    for policy, label, color, style in [
        ('ucb', 'UCB-SpecStop (SN)', '#B14442', '-'),
        ('naive_ucb', 'Naive-UCB', '#205D91', '--'),
        ('exp3', 'EXP3', '#676767', '-.'),
    ]:
        rows = frame[frame.strategy == policy].sort_values('t')
        x, y = rows.t.to_numpy(), rows.reference_score_ms.to_numpy()/1000
        assert np.array_equal(x, np.arange(1, 5001)) and np.isfinite(y).all()
        ax.plot(x, y, color=color, lw=1, ls=style, label=label)
        checks.append({'suite': suite, 'policy': policy, 'points': len(x),
                       'endpoint_s_per_token': float(y[-1])})
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.set(xscale='log', xlabel='Round', ylabel='Reference score (s/token)', xlim=(1,7000))
    ax.set_xticks([1,100,5000])
    ax.set_xticklabels(['1','100','5000'])
    ax.get_xticklabels()[-1].set_ha('right')
    ax.tick_params(axis='both', which='major', pad=2, labelsize=7.5)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 3
    ax.annotate('('+letter+') '+title, (0.5,0), xycoords='axes fraction',
                xytext=(0,-29), textcoords='offset points', fontsize=7.5,
                ha='center', va='top', annotation_clip=False)
    legend = ax.legend(loc='best', ncol=1, fontsize=7, frameon=True,
        facecolor='none', edgecolor='#777777', framealpha=1, fancybox=False,
        borderpad=0.25, handlelength=1.05, columnspacing=0.55, handletextpad=0.35,
        labelspacing=0.2, markerscale=0.8, borderaxespad=0.4)
    legend.get_frame().set_linewidth(0.45)
fig.canvas.draw()
args.output.parent.mkdir(parents=True, exist_ok=True)
require_matplotlib_panel_alignment(fig, json_out=str(args.output.with_suffix('.alignment.json')),
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(args.output)
fig.savefig(args.output.with_suffix('.svg'))
fig.savefig(args.output.with_suffix('.png'), dpi=300)
plt.close(fig)
print(json.dumps(checks, indent=2))
