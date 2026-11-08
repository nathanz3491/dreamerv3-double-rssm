"""All training curves in one figure, from tools/curves_from_replay.py output.

Top: achievements per training episode (episode-final count, the same measure
death_eval reports), averaged in 50k-step bins. Bottom: episode length. The
value printed beside each run is the mean over its last 100k steps -- a single
bin is too noisy to quote. Runs listed with a colour are drawn in it; runs
without one are grey context, since ten categorical colours would be
unreadable.

  python tools/plot_training_curves.py   # reads docs/curves/, writes docs/
"""

import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CURVES = ROOT / 'docs' / 'curves'
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
CONTEXT = '#c4c3bd'
BIN = 50_000
TAIL = 100_000

# (run, label, colour or None for grey context, z-order)
# This branch lists the plain baselines and the masked family's runs. Every
# family run reads the game's action rules (counted as cheating); its label
# says how.
RUNS = [
    ('vanilla', 'vanilla DreamerV3 (old code)', '#52514e', 3),
    ('abl_plain', 'plain DreamerV3 (fixed code)', '#8a8984', 4),
    ('abl_plain_s1', 'plain DreamerV3 (fixed code), seed 1', None, 2),
    ('expB_input', 'B1: rule flags as input (rules)', '#1baf7a', 5),
    ('expB3', 'B3: learned mask (rules in setup)', '#eb6834', 6),
    ('expB_mask', 'B2: rule mask (old code)', None, 2),
    ('expB_mask_fix', 'B2: rule mask, fixed RSSM-2', '#2a78d6', 7),
    ('expB_mask_fix_s1', 'B2 fixed, seed 1', None, 2),
    ('expB_mask_fix_s2', 'B2 fixed, seed 2', None, 2),
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def binned(c, key):
  end = np.asarray(c['end'], float)
  val = np.asarray(c[key], float)
  edges = np.arange(0, end.max() + BIN, BIN)
  idx = np.digitize(end, edges) - 1
  xs, ys = [], []
  for b in range(len(edges) - 1):
    m = idx == b
    if m.sum() >= 3:
      xs.append((edges[b] + BIN / 2) / 1e6)
      ys.append(val[m].mean())
  return np.array(xs), np.array(ys)


def figure(size=(11, 10)):
  """The two-panel figure; also drawn as a page of the comparison report."""
  fig, (top, bot) = plt.subplots(
      2, 1, figsize=size, sharex=True,
      gridspec_kw=dict(height_ratios=[1.7, 1], hspace=0.08))
  fig.subplots_adjust(left=0.07, right=0.68, top=0.90, bottom=0.07)
  fig.text(0.07, 0.955, 'Training curves: every run', fontsize=15,
           weight='bold')
  fig.text(0.07, 0.93, "Rebuilt from each run's replay; 50k-step bins; value "
           '= mean of the last 100k steps.', fontsize=8.5,
           color=INK2)

  ends = []
  for run, label, colour, z in RUNS:
    path = CURVES / f'{run}.json'
    if not path.exists():
      continue
    c = json.loads(path.read_text())
    for ax, key in ((top, 'ach'), (bot, 'len')):
      x, y = binned(c, key)
      ax.plot(x, y, color=colour or CONTEXT, linewidth=2.2 if colour else 1.2,
              zorder=z)
    end = np.asarray(c['end'])
    tail = np.asarray(c['ach'], float)[end > end.max() - TAIL].mean()
    x, y = binned(c, 'ach')
    ends.append([tail, x[-1], y[-1], label, colour])

  # Every run labelled in the right margin, spaced so none collide; a leader
  # joins each label to where its line actually ends.
  # A run that stopped early is labelled where it stops; a leader to the
  # margin would cross the whole chart.
  for tail, lx, ly, label, colour in [e for e in ends if e[1] < 0.9]:
    top.text(lx + 0.02, ly, f'{label}  {tail:.1f} (stopped at {lx:.1f}M)',
             fontsize=7.5, color=MUTED, va='center')
  # Placed top-down, each label pushed below the one above it: the best runs
  # cluster near the top, and pushing upward would run past the title.
  ends = sorted((e for e in ends if e[1] >= 0.9), key=lambda e: -e[0])
  placed = []
  for tail, lx, ly, label, colour in ends:
    yy = min(tail, placed[-1] - 0.22) if placed else tail
    placed.append(yy)
    top.annotate(
        f'{label}  {tail:.1f}', xy=(lx, ly), xytext=(1.13, yy),
        textcoords=('data', 'data'), fontsize=8.5 if colour else 7.5,
        color=INK if colour else MUTED, va='center',
        weight='bold' if colour else 'normal', annotation_clip=False,
        arrowprops=dict(arrowstyle='-', color=colour or CONTEXT,
                        linewidth=1.4 if colour else 0.8,
                        shrinkA=0, shrinkB=2))

  top.set_ylabel('achievements per episode')
  bot.set_ylabel('episode length (steps)')
  bot.set_xlabel('environment steps (millions)')
  bot.axhspan(261, 284, color=GRID, alpha=0.7, zorder=0)
  bot.annotate('random policy (261-284)', xy=(1.1, 272), xytext=(1.13, 272),
               fontsize=7.5, color=MUTED, va='center', annotation_clip=False)
  for ax in (top, bot):
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1.1)

  return fig


def main():
  fig = figure()
  for ext in ('png', 'pdf'):
    fig.savefig(ROOT / 'docs' / f'training-curves.{ext}', dpi=150)
  print('wrote docs/training-curves.png and .pdf')


if __name__ == '__main__':
  main()
