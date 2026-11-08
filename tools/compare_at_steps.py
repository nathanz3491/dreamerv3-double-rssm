"""Achievements per training episode at matched env steps, for live checks.

Reads the curve JSONs written by tools/curves_from_replay.py (achievements
per training episode, which every run logs the same way; the training return
includes each run's own shaping and cannot be compared). Each cell is the mean
achievement count of the training episodes that ended in the `window` env
steps before the checkpoint. A group may list several seeds; its cell is the
mean over seeds, with the standard error over seeds when there are several.

Rows default to the shortest group's latest step plus the 4 checkpoints
before it (`--earlier`), so a run still training is compared at the same
steps with its own recent trend. They come from a fixed grid (25k, 50k, 100k,
then every 100k) once it has enough points, else evenly spaced. Each row
averages the episodes since the previous row (at most `window` steps).

  python tools/compare_at_steps.py "vanilla=docs/curves/vanilla.json" \
      "plain=docs/curves/abl_plain.json,docs/curves/abl_plain_s1.json"
"""

import argparse
import json
import pathlib

import numpy as np

FIXED = (25e3, 50e3, 100e3, 200e3, 300e3, 400e3, 500e3, 600e3, 700e3, 800e3,
         900e3, 1000e3, 1100e3)


def load(path):
  d = json.loads(pathlib.Path(path).read_text())
  return np.asarray(d['end'], float), np.asarray(d['ach'], float)


def at(curve, step, window):
  end, ach = curve
  sel = (end <= step) & (end > step - window)
  return float(ach[sel].mean()) if sel.any() else float('nan')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('groups', nargs='+', help='name=curve.json[,seed2.json...]')
  ap.add_argument('--window', type=float, default=50e3)
  ap.add_argument('--steps', default='', help='comma list; default automatic')
  ap.add_argument('--earlier', type=int, default=4,
                  help='checkpoints shown before the latest step')
  ap.add_argument('--all', action='store_true',
                  help='every fixed checkpoint up to the latest step')
  args = ap.parse_args()
  groups = {}
  for spec in args.groups:
    name, paths = spec.split('=', 1)
    groups[name] = [load(p) for p in paths.split(',')]
  reach = {n: min(c[0][-1] for c in cs) for n, cs in groups.items()}
  shortest = min(reach.values())
  if args.steps:
    steps = [float(s) for s in args.steps.split(',')]
  else:
    # The shortest group's latest step and `earlier` checkpoints before it:
    # from the fixed grid once it has enough, else evenly spaced.
    grid = [s for s in FIXED if s < shortest - 5e3]
    if args.all:
      steps = grid + [shortest]
    elif len(grid) >= args.earlier:
      steps = grid[-args.earlier:] + [shortest]
    else:
      n = args.earlier + 1
      steps = [round(shortest * (i + 1) / n, -2) for i in range(n)]
  gaps = np.diff([0.0] + steps)

  names = list(groups)
  print('| env steps | ' + ' | '.join(names) + ' |')
  print('|---|' + '---|' * len(names))
  for s, gap in zip(steps, gaps):
    # The window never reaches back past the previous row, so rows stay
    # independent.
    w = min(args.window, gap) if not args.steps else min(args.window, s)
    cells = []
    for n in names:
      v = np.array([at(c, s, w) for c in groups[n]])
      if np.isnan(v).all():
        cells.append('–')
      elif len(v) > 1:
        v = v[~np.isnan(v)]
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        cells.append(f'{v.mean():.2f} ± {se:.2f}')
      else:
        cells.append(f'{v[0]:.2f}')
    print(f'| {s / 1e3:.0f}k | ' + ' | '.join(cells) + ' |')
  print(f'\nmean achievements per training episode over the env steps since the'
        f' previous row (at most {args.window / 1e3:.0f}k); ± is the SE over seeds')


if __name__ == '__main__':
  main()
