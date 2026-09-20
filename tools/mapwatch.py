"""Progress readout for a joint RSSM-1 + RSSM-2 + actor-critic Craftax run.

Reads metrics.jsonl from a logdir and prints the latest value of every metric
that matters, grouped, each against the baseline it should be beaten against.
Chance levels are printed inline because several of these numbers look fine in
isolation and are actually at chance (cf. the silent optimiser-registration bug:
position CE sat at ln 144 while displaying as 0.63).

Usage:  python mapwatch.py <logdir> [--history N] [--every SECONDS]
"""

import argparse
import json
import math
import os
import pathlib
import subprocess
import time

TOTAL_STEPS = 1_100_000

# (key, label, baseline text) -- baseline is what "no better than nothing" is.
GROUPS = [
    ('Task', [
        ('episode/score', 'score', 'random ~1.6 achievements'),
        ('episode/length', 'length', 'random 261-284, do-nothing 333'),
    ]),
    ('RSSM-1  (world model)', [
        ('train/loss/dyn', 'dyn KL', 'free_nats 1.0'),
        ('train/loss/rep', 'rep KL', 'free_nats 1.0'),
        ('train/loss/vector', 'obs recon', 'lower is better'),
        ('train/loss/rew', 'reward', 'lower is better'),
        ('train/loss/con', 'continue', 'lower is better'),
    ]),
    ('RSSM-2  (map model)', [
        ('train/map/bce_cell', 'map BCE/cell', f'chance ln2 = {math.log(2):.3f}'),
        ('train/map/posacc', 'position acc', 'chance 1/144 = 0.007'),
        ('train/map/posce', 'position CE', f'chance ln144 = {math.log(144):.2f}'),
        ('train/map/gate', 'ACTOR GATE', 'magnitude only; sign is arbitrary'),
    ]),
    ('Reward (shaped: potential over the tech tree)', [
        ('episode/score', 'episode return', 'stock craftax ~= achievements'),
        ('train/loss/rew', 'reward head', 'lower = reward is predictable'),
    ]),
    ('Actor-critic', [
        ('train/loss/policy', 'policy', ''),
        ('train/loss/value', 'value', ''),
        ('train/loss/repval', 'replay value', ''),
        ('train/rand/action', 'action entropy', '1.0 = uniform over 43'),
    ]),
    ('Throughput', [
        ('fps/policy', 'env FPS', ''),
        ('fps/train', 'train FPS', ''),
        ('replay/replay_ratio', 'replay ratio', 'target 512'),
    ]),
]


def load(logdir):
  path = pathlib.Path(logdir) / 'metrics.jsonl'
  if not path.exists():
    raise SystemExit(f'no metrics yet at {path}')
  rows = []
  for line in path.read_text().splitlines():
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      pass                      # a half-flushed final line during a live run
  return rows


def latest(rows, key):
  for row in reversed(rows):
    if key in row:
      return row['step'], row[key]
  return None, None


def series(rows, key, n):
  vals = [(r['step'], r[key]) for r in rows if key in r]
  return vals[-n:]


def achievements(rows):
  keys = sorted({k for r in rows for k in r if 'achievement' in k.lower()})
  out = []
  for k in keys:
    _, v = latest(rows, k)
    if v:
      out.append((k.split('/')[-1], v))
  return sorted(out, key=lambda kv: -kv[1])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logdir')
  ap.add_argument('--history', type=int, default=0,
                  help='also print the last N points of each map metric')
  ap.add_argument('--every', type=float, default=0,
                  help='refresh every N seconds instead of printing once')
  args = ap.parse_args()

  while True:
    rows = load(args.logdir)
    step = max((r['step'] for r in rows), default=0)
    pct = 100 * step / TOTAL_STEPS
    _, fps = latest(rows, 'fps/policy')
    eta = ''
    if fps:
      hrs = (TOTAL_STEPS - step) / fps / 3600
      eta = f'  ETA {hrs:.1f}h'
    print(f'\n{"=" * 62}')
    # A metrics file keeps its last value forever, so reading it says
    # nothing about whether the run is alive. Check the process and the
    # file mtime before quoting an ETA -- this tool once reported
    # "62.7%, ETA 4.2h" from a run OOM-killed four hours earlier.
    # Bracket the first character so the pattern cannot match the command line
    # that invoked this tool. `pgrep -f main.py.*vanilla` run from inside an
    # `ssh host '...vanilla...'` wrapper matches that wrapper and reports a
    # finished run as RUNNING -- which it did, three times, before this fix.
    name = pathlib.Path(args.logdir).name
    pattern = 'main.py.*[%s]%s' % (name[0], name[1:])
    alive = subprocess.run(
        ['pgrep', '-f', pattern], capture_output=True).returncode == 0
    age = time.time() - (
        pathlib.Path(args.logdir) / 'metrics.jsonl').stat().st_mtime
    print(f'step {step:,} / {TOTAL_STEPS:,}   ({pct:.1f}%)'
          + (eta if alive else ''))
    if alive:
      print(f'status  RUNNING   (last metric {age / 60:.0f} min ago)')
    else:
      print(f'status  *** NOT RUNNING *** - no process; last metric '
            f'{age / 3600:.1f}h ago. Numbers below are FINAL, not live.')
    print('=' * 62)

    for title, items in GROUPS:
      print(f'\n{title}')
      for key, label, base in items:
        _, val = latest(rows, key)
        if val is None:
          continue
        note = f'   [{base}]' if base else ''
        if key == 'train/map/gate':
          # The actor's first layer absorbs a sign flip for free, so only |gate|
          # says whether the map is being read. Trend matters more than level.
          hist = [abs(float(v)) for _, v in series(rows, key, 99)]
          arrow = ''
          if len(hist) > 1:
            arrow = ' rising' if hist[-1] > hist[0] else ' FLAT/FALLING'
          print(f'  {label:16s} {abs(float(val)):10.4f}{note}')
          print(f'  {"|gate| trend":16s} {hist[0]:.4f} -> {hist[-1]:.4f}{arrow}')
          continue
        print(f'  {label:16s} {float(val):10.4f}{note}')

    ach = achievements(rows)
    if ach:
      print('\nAchievements (rate)')
      for name, val in ach[:15]:
        print(f'  {name:28s} {float(val):.3f}')

    print(chr(10) + 'Trend  (first -> last, over the whole run)')
    for key, label in (
        ('epstats/log/achievements/avg', 'achievements'),
        ('episode/length', 'episode length'),
        ('train/map/gate', '|map gate|'),
        ('train/rand/action', 'action entropy'),
        ('train/map/posacc', 'map position acc')):
      pts = series(rows, key, 10 ** 9)
      if len(pts) < 2:
        continue
      vals = [abs(v) if 'gate' in key else v for _, v in pts]
      half = max(1, len(vals) // 4)
      first, last = sum(vals[:half]) / half, sum(vals[-half:]) / half
      arrow = 'up  ' if last > first * 1.02 else (
          'DOWN' if last < first * 0.98 else 'flat')
      print(f'  {label:18s} {first:8.4f} -> {last:8.4f}   {arrow}')

    # Peak matters as much as the last value: both prior runs peaked early and
    # then declined, which a single latest-value readout hides completely.
    pts = series(rows, 'epstats/log/achievements/avg', 10 ** 9)
    if len(pts) > 3:
      pk_step, pk = max(pts, key=lambda p: p[1])
      cur = pts[-1][1]
      print(chr(10) + f'  achievements PEAK {pk:.3f} at {pk_step / 1000:.0f}k'
            f'   now {cur:.3f}'
            + ('   <-- declining since peak' if cur < pk * 0.97 else ''))

    if args.history:
      print('\nHistory')
      for key in ('train/map/bce_cell', 'train/map/posacc', 'train/map/gate',
                  'episode/score'):
        pts = series(rows, key, args.history)
        if pts:
          txt = ' '.join(f'{v:.3f}' for _, v in pts)
          print(f'  {key.split("/")[-1]:14s} {txt}')

    if not args.every:
      break
    time.sleep(args.every)


if __name__ == '__main__':
  main()
