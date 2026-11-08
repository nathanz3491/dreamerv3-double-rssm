"""Render the Craftax map-model progress report to PDF.

Reads the binned curves dumped from the three completed runs plus the measured
diagnostics and writes a two-page PDF with matplotlib: headline charts, then
supporting measurements with the findings as a caption.

Usage:  python tools/make_report.py <runs.json> <out.pdf>
"""

import json
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Categorical hues, fixed order, never cycled. Ink stays in greys so the marks
# carry identity and the text does not compete with them.
C = {'vanilla': '#2a78d6', 'map': '#1baf7a', 'reshape': '#eb6834'}
LABEL = {'vanilla': 'Vanilla DreamerV3',
         'map': 'Two-RSSM map model',
         'reshape': 'Map + hand-written reward'}
INK, MUTED, GRID = '#1a1a19', '#6b6a66', '#e1e0d9'

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
})


def xy(runs, name, key):
    pts = runs[name].get(key) or []
    return [p[0] / 1000 for p in pts], [p[1] for p in pts]


def main():
    runs = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    A = 'epstats/log/achievements/avg'

    with PdfPages(out) as pdf:
        # ---- page 1: title + the two headline charts ----------------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.965, 'Craftax: a spatial memory model,',
                 fontsize=19, weight='bold', va='top')
        fig.text(0.08, 0.938, 'and two reward functions that backfired',
                 fontsize=19, weight='bold', va='top')
        fig.text(0.08, 0.912,
                 'Three completed 1.1M-step runs  ·  DreamerV3 on Craftax-Symbolic-v1'
                 '  ·  RTX 4090', fontsize=9.5, color=MUTED, va='top')

        ax = fig.add_axes([0.10, 0.60, 0.84, 0.26])
        for r in ('vanilla', 'map', 'reshape'):
            x, y = xy(runs, r, A)
            ax.plot(x, y, color=C[r], lw=2, label=LABEL[r],
                    marker='o', ms=3.5, mec='white', mew=0.8)
        ax.set_title('Achievements per episode (unshaped count)',
                     fontsize=11, weight='bold', loc='left', pad=8)
        ax.set_xlabel('training steps (thousands)')
        ax.set_ylabel('achievements')
        ax.set_ylim(0, 4.2)
        ax.legend(frameon=False, fontsize=8.5, loc='lower right')

        ax2 = fig.add_axes([0.10, 0.26, 0.84, 0.24])
        for r in ('vanilla', 'map', 'reshape'):
            x, y = xy(runs, r, 'train/rand/action')
            ax2.plot(x, y, color=C[r], lw=2, label=LABEL[r],
                     marker='o', ms=3.5, mec='white', mew=0.8)
        ax2.set_title('Action entropy  —  the finding that reframed everything',
                      fontsize=11, weight='bold', loc='left', pad=8)
        ax2.set_xlabel('training steps (thousands)')
        ax2.set_ylabel('entropy (1.0 = uniform over 43)')
        ax2.set_yscale('log')
        ax2.legend(frameon=False, fontsize=8.5)

        fig.text(0.10, 0.20, textwrap.fill(
            'Vanilla never collapses: it holds entropy near 0.12 for the whole run '
            'and is still rising at 1.1M. Both map runs fall to 0.004 - a policy '
            'emitting essentially one action per state out of 43. The map buys '
            '+0.8 achievements AND causes the collapse; the achievement peak at '
            '400k lands exactly where entropy bottoms out.', 96),
            fontsize=9, va='top', linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: supporting charts ------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
        fig.subplots_adjust(top=0.91, bottom=0.40, hspace=0.30, wspace=0.30)
        fig.text(0.08, 0.985, 'Supporting measurements', fontsize=17,
                 weight='bold', va='top')

        ax = axes[0][0]
        for r in ('vanilla', 'map', 'reshape'):
            x, y = xy(runs, r, 'episode/length')
            ax.plot(x, y, color=C[r], lw=2, label=LABEL[r])
        ax.axhspan(261, 284, color=MUTED, alpha=0.13)
        ax.set_title('Episode length   (band = random policy)',
                     fontsize=10, weight='bold', loc='left')
        ax.set_xlabel('steps (k)'); ax.set_ylabel('steps survived')
        ax.legend(frameon=False, fontsize=7.2, loc='lower center')

        ax = axes[0][1]
        for r in ('map', 'reshape'):
            x, y = xy(runs, r, 'train/map/gate')
            ax.plot(x, [abs(v) for v in y], color=C[r], lw=2, label=LABEL[r])
        ax.set_title('|map gate| — is the actor reading the map?',
                     fontsize=10, weight='bold', loc='left')
        ax.set_xlabel('steps (k)'); ax.set_ylabel('|gate|')
        ax.legend(frameon=False, fontsize=7.2, loc='upper left')

        ax = axes[1][0]
        for r in ('map', 'reshape'):
            x, y = xy(runs, r, 'train/map/posacc')
            ax.plot(x, y, color=C[r], lw=2, label=LABEL[r])
        ax.axhline(1 / 144, color=MUTED, ls='--', lw=1.2)
        ax.set_title('RSSM-2 position accuracy', fontsize=10,
                     weight='bold', loc='left')
        ax.set_xlabel('steps (k)')
        ax.set_ylabel('accuracy   (dashed = chance, 0.007)')
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7.2, loc='lower right')

        ax = axes[1][1]
        causes = ['thirst', 'mob / lava', 'hunger', 'energy']
        counts = [25, 11, 11, 3]
        bars = ax.barh(causes[::-1], counts[::-1], color='#2a78d6', height=0.55)
        bars[-1].set_color('#eb6834')
        ax.set_title('Cause of death (50 episodes)', fontsize=10,
                     weight='bold', loc='left')
        ax.set_xlabel('episodes')
        ax.grid(axis='y', visible=False)

        # Carries what the prose pages used to say, since the report is two
        # pages: the map works, survival did not move, and the probe that
        # reframed the whole problem.
        paras = [
            "The map model itself works. RSSM-2 predicts the agent's coarse "
            "cell at 0.83-0.91 against a chance of 0.007, and the learned gate "
            "controlling how much map reaches the actor rose from 0.095 to "
            "0.526 without a single reversal across 1.1M steps - the actor "
            "chose to read it more heavily, right to the end.",

            "Survival never moved: all three runs sit inside the random-policy "
            "band. Thirst causes half of all deaths, and the agent dies with a "
            "mean of 2.45 of 9 drink still in the tank while knowing exactly "
            "where the water is. The failure is not that it cannot find water; "
            "it is that a second drink pays nothing, so water is worth nothing.",

            "A probe then placed the trained agent beside its own crafting "
            "table, holding five wood, meters full - a state where one keypress "
            "yields a pickaxe, verified in 10 of 10 seeds. It pressed craft "
            "ZERO times, using 6 of 43 available actions. No reward attached to "
            "crafting can be collected, because the action is never emitted.",

            "Next: a potential-based reward over tech-tree prerequisites, which "
            "provably cannot create the cheap optima that made the hand-written "
            "version collect half the achievements of no shaping at all - "
            "paired with an exploration fix, since better credit assignment "
            "cannot make an unsampled action fire.",
        ]
        y = 0.345
        for para in paras:
            block = textwrap.fill(para, 96)
            fig.text(0.08, y, block, fontsize=9, va='top', linespacing=1.5)
            y -= 0.0165 * (block.count('\n') + 1) + 0.013
        pdf.savefig(fig)
        plt.close(fig)

        d = pdf.infodict()
        d['Title'] = 'Craftax two-RSSM map model: progress report'
        d['Subject'] = 'Vanilla vs map model vs hand-written reward shaping'

    print('wrote', out)


if __name__ == '__main__':
    main()
