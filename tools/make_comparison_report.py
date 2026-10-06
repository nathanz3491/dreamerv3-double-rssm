"""Comparison report: every trained run, every achievement, one PDF.

Reads docs/eval_results.json -- one entry per evaluated checkpoint, each the
output of tools/death_eval.py (episode-final achievement count, per-achievement
unlock rate, cause of death) -- and writes docs/comparison-report.pdf.

  python tools/make_comparison_report.py
"""

import json
import pathlib
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / 'docs' / 'eval_results.json'
DIAG = ROOT / 'docs' / 'diagnosis.json'           # tools/diagnose_run.py
OUT = ROOT / 'docs' / 'comparison-report.pdf'

# Reference palette (dataviz skill, light mode): recessive ink, one sequential
# blue ramp for magnitude, the first three categorical slots for death causes.
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE, BLUE_DARK, WAKE = '#2a78d6', '#184f95', '#eb6834'
CAUSE_COLORS = {'drink': '#2a78d6', 'food': '#eb6834', 'mob / lava': '#1baf7a',
                'energy': '#eda100'}
RAMP = LinearSegmentedColormap.from_list('blue', [
    '#f3f7fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95',
    '#0d366b'])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
    'axes.spines.top': False, 'axes.spines.right': False,
})
A4 = (8.27, 11.69)


def page(pdf, title, subtitle=None):
  fig = plt.figure(figsize=A4)
  fig.text(0.07, 0.955, title, fontsize=15, weight='bold')
  if subtitle:
    wrap(fig, 0.07, 0.935, subtitle, width=110, size=8.5, color=INK2,
         line=0.014)
  return fig


def wrap(fig, x, y, text, width=100, size=9, color=INK, line=0.0155, **kw):
  """Write wrapped paragraphs top-down; returns the y below the last line."""
  for para in text.split('\n'):
    lines = textwrap.wrap(para, width) or ['']
    for ln in lines:
      fig.text(x, y, ln, fontsize=size, color=color, va='top', **kw)
      y -= line
  return y


def main():
  data = json.loads(DATA.read_text(encoding='utf-8'))
  runs = [r for r in data['runs'] if r.get('final')]
  labels = [r['label'] for r in runs]

  with PdfPages(OUT) as pdf:
    # ---------------------------------------------------------- 1. summary
    fig = page(pdf, 'Craftax runs compared: achievements and what unlocks them',
               data['subtitle'])
    y = wrap(fig, 0.07, 0.90, data['summary'], width=104)

    cols = ['run', 'eps', 'ach.', 'w/o wake', 'life', 'table', 'pickaxe',
            'sword', 'stone', 'drink']
    rows = []
    for r in runs:
      rt = r['rates']
      rows.append([r['label'], str(r['episodes']), f"{r['achievements']:.2f}",
                   f"{r['achievements'] - r['wake_up']:.2f}",
                   f"{r['lifespan']:.0f}" if r.get('lifespan') else '-']
                  + [f"{100 * rt.get(k, 0):.0f}%" for k in (
                      'PLACE_TABLE', 'MAKE_WOOD_PICKAXE', 'MAKE_WOOD_SWORD',
                      'COLLECT_STONE', 'COLLECT_DRINK')])
    # Height follows the row count, so the note below never collides with a
    # table that has grown by a run.
    th = 0.0235 * (len(rows) + 1)
    ax = fig.add_axes([0.07, y - 0.02 - th, 0.86, th])
    ax.axis('off')
    tab = ax.table(cellText=rows, colLabels=cols, loc='upper left',
                   cellLoc='center', colLoc='center',
                   colWidths=[0.34, 0.05, 0.06, 0.09, 0.06, 0.07, 0.08, 0.07,
                              0.07, 0.07])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.55)
    for (i, j), cell in tab.get_celld().items():
      cell.set_edgecolor(GRID)
      cell.set_facecolor(SURFACE)
      if j == 0:
        cell._loc = 'left'
        cell.set_text_props(ha='left')
      if i == 0:
        cell.set_text_props(weight='bold', color=INK2)
      if i == len(rows):                         # newest run
        cell.set_text_props(weight='bold')
    wrap(fig, 0.07, y - 0.05 - th, data['table_note'], width=125, size=7.5,
         color=MUTED, line=0.012)
    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------------- 2. totals + the tech spine
    fig = page(pdf, 'Sleep inflates the map runs; the manager is real progress',
               'Mean episode-final achievements per run, split into WAKE_UP and '
               'the rest; unlock rate of the early tech-tree rungs')
    ax = fig.add_axes([0.33, 0.60, 0.59, 0.27])
    ypos = np.arange(len(runs))[::-1]
    rest = np.array([r['achievements'] - r['wake_up'] for r in runs])
    wake = np.array([r['wake_up'] for r in runs])
    ax.barh(ypos, rest, height=0.62, color=BLUE, edgecolor=SURFACE,
            linewidth=2, label='everything else')
    ax.barh(ypos, wake, left=rest, height=0.62, color=WAKE, edgecolor=SURFACE,
            linewidth=2, label='WAKE_UP (sleep, then wake)')
    for yy, a, b in zip(ypos, rest, wake):
      ax.text(a + b + 0.06, yy, f'{a + b:.2f}', va='center', fontsize=8.5)
      ax.text(a - 0.08, yy, f'{a:.2f}', va='center', ha='right', fontsize=7.5,
              color='#ffffff')
    ax.axvline(runs[0]['achievements'], color=MUTED, linewidth=1,
               linestyle=(0, (3, 3)))
    ax.text(runs[0]['achievements'], -0.75, ' vanilla', fontsize=7.5,
            color=MUTED, va='center')
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max(rest + wake) * 1.15)
    ax.set_xlabel('achievements per episode (episode-final count)')
    ax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False,
              fontsize=8)
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    spine = data['spine']
    ax = fig.add_axes([0.33, 0.08, 0.59, 0.42])
    M = np.array([[r['rates'].get(k, 0.0) for k in spine] for r in runs])
    im = ax.imshow(M, cmap=RAMP, vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
      for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f'{100 * v:.0f}', ha='center', va='center', fontsize=8,
                color='#ffffff' if v > 0.55 else INK)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(spine)))
    ax.set_xticklabels([s.replace('_', ' ').lower() for s in spine],
                       rotation=35, ha='right', fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
      s.set_visible(False)
    fig.text(0.33, 0.515, 'Unlock rate (% of episodes), early tech tree in '
             'dependency order', fontsize=9, weight='bold')
    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------------- 3. every achievement
    names = [n for n in data['achievement_order']
             if any(r['rates'].get(n, 0) > 0 for r in runs)]
    never = len(data['achievement_order']) - len(names)
    fig = page(pdf, 'Every achievement any run unlocked',
               f'% of evaluation episodes in which each achievement fired. '
               f'The other {never} of 67 were never unlocked by any run.')
    ax = fig.add_axes([0.30, 0.05, 0.64, 0.75])
    M = np.array([[r['rates'].get(n, 0.0) for r in runs] for n in names])
    ax.imshow(M, cmap=RAMP, vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
      for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f'{100 * v:.0f}' if v > 0 else '·', ha='center',
                va='center', fontsize=7.5,
                color='#ffffff' if v > 0.55 else (INK if v > 0 else MUTED))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace('_', ' ').lower() for n in names],
                       fontsize=8)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([r['short'] for r in runs], rotation=60, ha='left',
                       fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
      s.set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)

    # ---------------------------------------- 4. death + checkpoint variance
    fig = page(pdf, 'Survival never moved',
               'Cause of death (share of evaluation episodes) and lifespan; '
               'and how much one run moves between its own checkpoints')
    cruns = [r for r in runs if r.get('causes')]
    ax = fig.add_axes([0.40, 0.56, 0.52, 0.31])
    ypos = np.arange(len(cruns))[::-1]
    left = np.zeros(len(cruns))
    for cause, col in CAUSE_COLORS.items():
      share = np.array([r['causes'].get(cause, 0) / sum(r['causes'].values())
                        for r in cruns])
      ax.barh(ypos, share, left=left, height=0.62, color=col,
              edgecolor=SURFACE, linewidth=2, label=cause)
      for yy, l0, s in zip(ypos, left, share):
        if s >= 0.12:
          ax.text(l0 + s / 2, yy, f'{100 * s:.0f}%', ha='center', va='center',
                  fontsize=7.5, color='#ffffff')
      left += share
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['label']}  ({r['lifespan']:.0f} steps)"
                        for r in cruns], fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
    ax.legend(ncol=4, loc='lower left', bbox_to_anchor=(0, 1.01), frameon=False,
              fontsize=8)
    ax.spines['left'].set_visible(False)
    wrap(fig, 0.40, 0.528, 'Random policy lives 261-284 steps; doing nothing '
         'lives 333. Map + potential (bug) has no recorded causes.', width=75,
         size=7.5, color=MUTED, line=0.012)

    ck = data['checkpoints']
    ax = fig.add_axes([0.30, 0.10, 0.62, 0.33])
    xs = np.arange(len(ck['points']))
    ys = [p['achievements'] for p in ck['points']]
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker='o', markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    for x, p in zip(xs, ck['points']):
      ax.text(x, p['achievements'] + 0.08, f"{p['achievements']:.2f}",
              ha='center', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([p['label'] for p in ck['points']], fontsize=8)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.5)
    ax.set_ylabel('achievements per episode')
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(0.30, 0.46, ck['title'], fontsize=9, weight='bold')
    wrap(fig, 0.30, 0.448, ck['note'], width=78, size=7.5, color=INK2,
         line=0.013)
    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------------- 5. training curves
    import plot_training_curves as curves
    fig = curves.figure(size=A4)
    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------- 6. where the best run falls short
    diag = {d['run']: d for d in json.loads(DIAG.read_text(encoding='utf-8'))}
    fig = page(pdf, 'Where B2 falls short',
               'Last 150k training steps of each run (~500 episodes), decoded '
               'from replay by tools/diagnose_run.py')
    rungs = ['COLLECT_WOOD', 'PLACE_TABLE', 'MAKE_WOOD_PICKAXE',
             'COLLECT_STONE', 'PLACE_FURNACE', 'MAKE_STONE_PICKAXE',
             'COLLECT_IRON']
    series = [('expB_mask', 'B2 masked', '#2a78d6'),
              ('expB_input', 'B1 told', '#1baf7a'),
              ('honest_map', 'honest map', '#eb6834')]
    ax = fig.add_axes([0.10, 0.58, 0.62, 0.30])
    xs = np.arange(len(rungs))
    for run, label, colour in series:
      ys = [100 * diag[run]['funnel'][r]['all'] for r in rungs]
      ax.plot(xs, ys, color=colour, linewidth=2, marker='o', markersize=7,
              markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    b2 = [100 * diag['expB_mask']['funnel'][r]['all'] for r in rungs]
    for x, y in zip(xs, b2):
      ax.text(x, y + 4, f'{y:.0f}%', ha='center', fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([r.replace('_', ' ').lower() for r in rungs],
                       rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('% of episodes reaching the rung')
    ax.set_ylim(-3, 108)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    sp = diag['expB_mask']['stone_pickaxe']
    de = diag['expB_mask']['deaths']
    pct = lambda v: f'{100 * v:.0f}%'
    text = (
        "1. The stone pickaxe is the wall. B2 reaches the furnace in "
        f"{pct(diag['expB_mask']['funnel']['PLACE_FURNACE']['all'])} of episodes "
        "but makes the stone pickaxe in "
        f"{pct(diag['expB_mask']['funnel']['MAKE_STONE_PICKAXE']['all'])}. Of the "
        f"{sp['episodes']} episodes that collected stone, {pct(sp['held_both'])} "
        "held wood and stone at the same time, but only "
        f"{pct(sp['both_at_table'])} ever stood next to a table while holding "
        f"both, and {pct(sp['made'])} made the pickaxe. Two gaps: it rarely "
        "brings the ingredients back to a table (logistics), and when it does "
        "the chance lasts one step -- the single log usually goes into the wood "
        "sword or it walks off (measured in docs/b2-casebook.pdf).\n"
        "2. Table to wood pickaxe leaks. "
        f"{pct(1 - diag['expB_mask']['funnel']['MAKE_WOOD_PICKAXE']['given_previous'])}"
        " of the episodes that place a table never make the wood pickaxe. At a "
        "table holding wood it crafts every time, but every one of these "
        "episodes placed the table with its last two logs, and later logs go "
        "into another table rather than back to the first "
        "(docs/b2-casebook.pdf).\n"
        "3. Survival. Training-window lifespan "
        f"{de['lifespan_mean']:.0f} (median {de['lifespan_median']:.0f}) against "
        "261-284 for a random policy. Thirst is still the first meter to run "
        f"out in {pct(de['causes'].get('drink ran out', 0))} of deaths and "
        f"hunger in {pct(de['causes'].get('food ran out', 0))}: Craftax pays "
        "for the first drink only, so topping up is never rewarded. A hostile "
        f"mob is adjacent at {pct(de['at_death'].get('hostile', 0))} of deaths; "
        "it dies in lava in "
        f"{pct(de['at_death'].get('lava', 0))} (the others: under 1%), the price "
        "of exploring further. Deaths in its sleep are rare "
        f"({pct(de['at_death'].get('sleeping', 0))}).\n"
        "4. Wasted presses. 43% of presses still do nothing (experiment A): DO "
        "with nothing in front (26%), NOOP and walking into walls. Masking only "
        "removes the special actions; these basic ones are never masked.")
    wrap(fig, 0.07, 0.47, text, width=108, size=8.6, line=0.0148)
    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------------- 7. interpretation
    fig = page(pdf, 'What the results say, and what would settle the rest')
    y = 0.90
    for block in data['interpretation']:
      fig.text(0.07, y, block['heading'], fontsize=10.5, weight='bold',
               va='top')
      y = wrap(fig, 0.07, y - 0.022, block['body'], width=104) - 0.012
    pdf.savefig(fig)
    plt.close(fig)

  print('wrote', OUT)


if __name__ == '__main__':
  main()
