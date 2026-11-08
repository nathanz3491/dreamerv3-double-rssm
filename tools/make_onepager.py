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

C = {'vanilla': '#2a78d6', 'map': '#1baf7a', 'reshape': '#eb6834',
     'potential': '#6250d6', 'map_buggy': '#9fd9c2', 'pot_buggy': '#b3aae8'}
LABEL = {'vanilla': 'Vanilla', 'map': 'Map model (fixed)',
         'map_buggy': 'Map model (bug)', 'pot_buggy': 'Map + potential (bug)',
         'reshape': 'Map + hand-written', 'potential': 'Map + potential (fixed)'}
INK, MUTED, GRID = '#1a1a19', '#6b6a66', '#e1e0d9'

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True,
})

# tools/death_eval.py, 25 episodes each, identical method across all four.
EVAL = {'vanilla': 4.44, 'map_buggy': 4.35, 'reshape': 2.30,
        'pot_buggy': 3.87, 'map': 5.03, 'potential': 5.80}
# Every run at 1.1M, tools/death_eval.py, 25-30 episodes, one method.
POT = None
BUDGET = {k: '100%' for k in EVAL} | {'reshape': '46%'}
# Per-achievement unlock rate, same evaluations. The last two rows are the gate
# to the entire rest of the tech tree.
GATES = [
    ('COLLECT_DRINK', {'vanilla': 0.80, 'map': 0.70, 'potential': 0.60}),
    ('PLACE_TABLE', {'vanilla': 0.64, 'map': 0.63, 'potential': 0.90}),
    ('MAKE_WOOD_PICKAXE', {'vanilla': 0.24, 'map': 0.07, 'potential': 0.30}),
    ('COLLECT_STONE', {'vanilla': 0.12, 'map': 0.00, 'potential': 0.20}),
    ('MAKE_WOOD_SWORD', {'vanilla': 0.00, 'map': 0.00, 'potential': 0.10}),
]


def main():
    curves = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    order = ['reshape', 'pot_buggy', 'map_buggy', 'vanilla', 'map', 'potential']

    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.97, 'A one-flag bug fix, then potential shaping:',
                 fontsize=18, weight='bold', va='top')
        fig.text(0.07, 0.945, '4.44 -> 5.03 -> 5.80 achievements',
                 fontsize=18, weight='bold', va='top')
        fig.text(0.07, 0.922,
                 'Achievement COUNT per episode  ·  DreamerV3  ·  '
                 'tools/death_eval.py, 25-30 episodes each',
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
        ax.set_xlim(0, 6.6)
        ax.set_xlabel('achievements per episode (episode-final count)')
        ax.grid(axis='y', visible=False)
        ax.set_title('Achievement count at 1.1M steps, all runs',
                     fontsize=11, weight='bold', loc='left', pad=8)

        # --- training curves, on max ---------------------------------------
        ax2 = fig.add_axes([0.10, 0.475, 0.82, 0.175])
        for k in ('vanilla', 'map', 'reshape'):
            pts = curves[k].get('max') or []
            if pts:
                ax2.plot([p[0] / 1000 for p in pts], [p[1] for p in pts],
                         color=C[k], lw=2, marker='o', ms=3, mec='white',
                         mew=0.7, label=LABEL[k])
        for k, lab in (('map', 'Map fixed (eval)'),
                       ('potential', 'Map + potential fixed (eval)')):
            ax2.scatter([1100], [EVAL[k]], color=C[k], s=90, zorder=5,
                        marker='D', edgecolor='white', linewidth=1.3, label=lab)
        ax2.set_title('Achievement count during training',
                      fontsize=11, weight='bold', loc='left', pad=8)
        ax2.set_xlabel('training steps (thousands)')
        ax2.set_ylabel('achievements')
        ax2.set_ylim(0, 6.4)
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
        ax3.set_title('The tech-tree gate, finally opened',
                      fontsize=11, weight='bold', loc='left', pad=8)
        ax3.legend(frameon=False, fontsize=8, ncol=3, loc='upper right')
        ax3.grid(axis='x', visible=False)

        paras = [
            "imag_shift True corrupted the policy gradient. The imagination "
            "rollout PICKS actions reading the crop frozen at its start, while "
            "the loss SCORES them reading the shifted crop - and imag_loss is "
            "REINFORCE, so it takes logpi of actions drawn from a different "
            "distribution. The objective ran to -61 where vanilla sat at 0.00. "
            "Turning the flag off is worth +0.68 on otherwise identical configs.",

            "Potential-based shaping then adds +0.77 on top. They compose "
            "because they fix different things: the flag restores a working "
            "policy gradient, and PHI - weighted heaviest on MAKE_WOOD_PICKAXE "
            "- gives it somewhere to point. Under the bug the same shaping "
            "scored 3.87, below doing nothing at all.",

            "The tech-tree gate opened. MAKE_WOOD_PICKAXE 0.30 and "
            "COLLECT_STONE 0.20 both beat vanilla; MAKE_WOOD_SWORD and "
            "PLACE_STONE appear for the first time in any run. Not fixed: "
            "episode length 278 sits inside the random-policy band and "
            "COLLECT_DRINK fell to 0.60. This bought depth, not survival.",
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
