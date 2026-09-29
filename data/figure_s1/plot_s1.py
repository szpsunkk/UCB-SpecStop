"""Compact presentation of the existing additional-RTT crossing summaries.

Figure contract: show the retrospective k=1/k=2 tie estimate overall and
by prompt length, including all four existing means and 95% t intervals.
Single-panel quantitative comparison; ten seeded runs per summary, including
the three calibration runs. No test or estimate is recomputed here.
The two full upstream summary tables are preserved unchanged. Selecting
left_k=1/right_k=2 preserves the original Fig. S1 scope (4 of 36 rows).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parent


def load_rows():
    rows, inputs = [], {}
    for filename, overall in [('adjacent_arm_crossing_summary.csv', True),
                              ('context_adjacent_arm_crossing_summary.csv', False)]:
        path = ROOT / filename
        inputs[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if int(row['left_k']) == 1 and int(row['right_k']) == 2:
                    row['prompt_stratum'] = 'overall' if overall else row['prompt_stratum']
                    rows.append(row)
    order = ['overall', 'short', 'medium', 'long']
    assert sorted(row['prompt_stratum'] for row in rows) == sorted(order)
    rows.sort(key=lambda row: order.index(row['prompt_stratum']))
    fields = ['added_rtt_crossing_ms_mean', 'added_rtt_crossing_ms_ci95_low',
              'added_rtt_crossing_ms_ci95_high']
    values = np.array([[float(row[key]) for key in fields] for row in rows])
    assert np.isfinite(values).all()
    assert np.all(values[:, 1] <= values[:, 0]) and np.all(values[:, 0] <= values[:, 2])
    assert all(int(row['independent_runs']) == 10 for row in rows)
    return rows, values, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parents[1] / 'outputs/figures/fig_supp_crossings.pdf')
    args = parser.parse_args()
    rows, values, inputs = load_rows()
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.linewidth': 0.6, 'text.color': '#262626',
        'axes.edgecolor': '#777777', 'xtick.color': '#454545',
    })
    fig, ax = plt.subplots(figsize=(3.5, 1.72))
    fig.subplots_adjust(left=0.19, right=0.655, bottom=0.255, top=0.83)
    ax.set_xlim(200, 620)
    ax.set_ylim(-0.52, 3.52)
    ax.set_xticks([200, 400, 600])
    ax.set_yticks([])
    ax.set_xlabel('Additional RTT (ms)', labelpad=4)
    ax.tick_params(axis='x', direction='out', length=3, width=0.6, pad=2)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    positions = np.arange(4)[::-1]
    # Row labels and numbers use one shared y transform, so they align exactly.
    row_transform = transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for i, (row, (center, low, high), y) in enumerate(zip(rows, values, positions)):
        color = '#205D91' if i == 0 else '#505A63'
        marker = 'D' if i == 0 else 'o'
        ax.errorbar(center, y, xerr=[[center-low], [high-center]], fmt=marker,
                    color=color, markersize=4.2 if i == 0 else 3.5,
                    markeredgewidth=0.7, elinewidth=1.0, capsize=2.2, capthick=0.8)
        fig.text(0.025, y, row['prompt_stratum'].capitalize(),
                 transform=row_transform, ha='left', va='center', fontsize=8,
                 color=color, fontweight='bold' if i == 0 else 'normal')
        fig.text(0.985, y, f'{center:.1f} [{low:.1f}, {high:.1f}]',
                 transform=row_transform, ha='right', va='center', fontsize=7,
                 color=color, fontweight='bold' if i == 0 else 'normal')
    fig.text(0.025, 0.925, 'Prompt group', fontsize=7, color='#505A63', va='center')
    fig.text(0.985, 0.925, 'Mean [95% CI]', fontsize=7, color='#505A63',
             va='center', ha='right')
    # The separator is below the headings and clear of all estimates.
    fig.add_artist(plt.Line2D([0.025,0.985], [0.872,0.872],
                   transform=fig.transFigure, color='#B8BEC3', linewidth=0.5))
    fig.canvas.draw()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(fig,
        json_out=str(args.output.with_suffix('.alignment.json')), strict=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix('.svg'))
    fig.savefig(args.output.with_suffix('.png'), dpi=300)
    fig.savefig(args.output.with_suffix('.tiff'), dpi=600, pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    report = {'inputs_sha256': inputs, 'source_rows': 36, 'displayed_rows': 4,
              'scope': 'Only the original k=1/k=2 crossing; no group or run excluded',
              'center': 'Existing mean across ten seeded runs',
              'interval': 'Existing marginal 95% Student-t interval',
              'calibration_included': True, 'new_statistical_analysis': False,
              'rows': [{**row, 'display': [float(v) for v in value]}
                       for row,value in zip(rows,values)]}
    args.output.with_suffix('.data-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output), 'values': values.tolist()},indent=2))


if __name__ == '__main__':
    main()
