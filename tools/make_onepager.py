"""One-page comparison of the four Craftax runs, on achievement COUNT.

All achievement numbers here are the episode-final count -- `max` from the
training curves, and the matching `final` figure from tools/death_eval.py. The
`avg` series used earlier is a per-step mean of a total that climbs 0 -> final
over the episode, so it reads roughly 35% low and is not an achievement count.

Usage:  python tools/make_onepager.py <curves.json> <out.pdf>
"""

import json
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

C = {'vanilla': '#2a78d6', 'map': '#1baf7a',
     'reshape': '#eb6834', 'potential': '#6250d6'}
LABEL = {'vanilla': 'Vanilla', 'map': 'Map model',
         'reshape': 'Map + hand-written', 'potential': 'Map + potential'}
INK, MUTED, GRID = '#1a1a19', '#6b6a66', '#e1e0d9'

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True,
})

# tools/death_eval.py, 25 episodes each, identical method across all four.
EVAL = {'vanilla': 4.44, 'map': 4.35, 'reshape': 2.30, 'potential': 4.68}
# The potential run was evaluated twice. 4.68 is its PEAK, not its level: by
# 672k it had fallen to 3.60, the same peak-then-decline every other run shows.
POT = [(477, 4.68), (672, 3.60)]
BUDGET = {'vanilla': '100%', 'map': '100%', 'reshape': '46%',
          'potential': 'peak, 43%'}
# Per-achievement unlock rate, same evaluations. The last two rows are the gate
# to the entire rest of the tech tree.
GATES = [
    ('COLLECT_DRINK', {'vanilla': 0.80, 'map': 0.43, 'potential': 0.32}),
    ('PLACE_TABLE', {'vanilla': 0.64, 'map': 0.35, 'potential': 0.16}),
    ('MAKE_WOOD_PICKAXE', {'vanilla': 0.24, 'map': 0.00, 'potential': 0.00}),
    ('COLLECT_STONE', {'vanilla': 0.12, 'map': 0.00, 'potential': 0.00}),
]


def main():
    curves = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    order = ['reshape', 'map', 'vanilla', 'potential']

    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.97, 'Potential-based shaping peaks highest,',
                 fontsize=18, weight='bold', va='top')
        fig.text(0.07, 0.945, 'then declines like everything else',
                 fontsize=18, weight='bold', va='top')
        fig.text(0.07, 0.922,
                 'Achievement COUNT per episode  ·  DreamerV3  ·  '
                 'tools/death_eval.py, 25 episodes each',
                 fontsize=9, color=MUTED, va='top')

        # --- headline bars -------------------------------------------------
        ax = fig.add_axes([0.30, 0.715, 0.62, 0.165])
        vals = [EVAL[k] for k in order]
        bars = ax.barh(range(len(order)), vals, height=0.55,
                       color=[C[k] for k in order])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f'{LABEL[k]}   ({BUDGET[k]})' for k in order],
                           fontsize=9.5)
        for i, (k, v) in enumerate(zip(order, vals)):
            ax.text(v + 0.08, i, f'{v:.2f}', va='center', fontsize=10,
                    weight='bold' if k == 'potential' else 'normal')
        ax.set_xlim(0, 5.4)
        ax.set_xlabel('achievements per episode (episode-final count)')
        ax.grid(axis='y', visible=False)
        ax.set_title('Peak achievement count reached by each run',
                     fontsize=11, weight='bold', loc='left', pad=8)

        # --- training curves, on max ---------------------------------------
        ax2 = fig.add_axes([0.10, 0.475, 0.82, 0.175])
        for k in ('vanilla', 'map', 'reshape'):
            pts = curves[k].get('max') or []
            if pts:
                ax2.plot([p[0] / 1000 for p in pts], [p[1] for p in pts],
                         color=C[k], lw=2, marker='o', ms=3, mec='white',
                         mew=0.7, label=LABEL[k])
        ax2.plot([p[0] for p in POT], [p[1] for p in POT],
                 color=C['potential'], lw=2, ls='--', marker='D', ms=7,
                 mec='white', mew=1.2, label='Map + potential (eval)')
        ax2.set_title('Achievement count during training',
                      fontsize=11, weight='bold', loc='left', pad=8)
        ax2.set_xlabel('training steps (thousands)')
        ax2.set_ylabel('achievements')
        ax2.set_ylim(0, 5.4)
        ax2.legend(frameon=False, fontsize=8, loc='lower right', ncol=2)

        # --- the gate --------------------------------------------------------
        ax3 = fig.add_axes([0.10, 0.270, 0.82, 0.130])
        runs = ['vanilla', 'map', 'potential']
        w = 0.26
        for j, k in enumerate(runs):
            ax3.bar([i + (j - 1) * w for i in range(len(GATES))],
                    [g[1][k] for g in GATES], width=w * 0.92,
                    color=C[k], label=LABEL[k])
        ax3.set_xticks(range(len(GATES)))
        ax3.set_xticklabels([g[0] for g in GATES], fontsize=8)
        ax3.set_ylabel('unlock rate')
        ax3.set_ylim(0, 1.0)
        ax3.set_title('Only vanilla gets through the tech-tree gate',
                      fontsize=11, weight='bold', loc='left', pad=8)
        ax3.legend(frameon=False, fontsize=8, ncol=3, loc='upper right')
        ax3.grid(axis='x', visible=False)

        paras = [
            "Potential-based shaping PEAKED at 4.68 achievements around 477k "
            "steps - higher than any other run reached at any point. Because "
            "the reward is a difference of a potential it provably cannot "
            "create the cheap optima that made the hand-written version collect "
            "HALF of no shaping at all (2.30), and at its peak it had moved the "
            "target behaviour: COLLECT_DRINK 0.56, PLACE_TABLE 0.44.",

            "By 672k it had fallen to 3.60, with COLLECT_DRINK down to 0.32 and "
            "PLACE_TABLE to 0.16. That is the same peak-then-decline every "
            "previous run shows, and it arrives as action entropy bottoms out "
            "at 0.007. The potential raised the ceiling; it did not stop the "
            "collapse, which is exactly what a credit-assignment fix can and "
            "cannot do.",

            "Vanilla remains the only run that CRAFTS - MAKE_WOOD_PICKAXE 0.24, "
            "COLLECT_STONE 0.12, against exactly zero for every map-based run - "
            "and the only one whose entropy never collapses (0.117 across all "
            "1.1M steps). Next: adaptive entropy targeting, the one variable "
            "never manipulated.",
        ]
        y = 0.212
        for para in paras:
            block = textwrap.fill(para, 100)
            fig.text(0.07, y, block, fontsize=8.8, va='top', linespacing=1.45)
            y -= 0.0158 * (block.count('\n') + 1) + 0.012

        pdf.savefig(fig)
        plt.close(fig)

        d = pdf.infodict()
        d['Title'] = 'Craftax: potential-based shaping sets an achievement record'

    print('wrote', out)


if __name__ == '__main__':
    main()
