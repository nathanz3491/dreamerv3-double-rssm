"""Training curves rebuilt from the replay chunks a run leaves on disk.

Achievement logging was off for most runs (env.craftax.logs False), so their
metrics hold only the episode return, which includes each run's own reward
shaping and so cannot be compared across runs. But every replay chunk stores
`ach` -- the achievements newly unlocked at each step -- and `is_first`, and the
chunks of a run cover all of its training. That is enough to recover, for every
training episode, its final achievement count and its length: the same
episode-final measure `death_eval` reports, here over the course of training.

Writes <out>/<run>.json with one entry per training episode:
  end   env step at which the episode ended
  ach   achievements unlocked in the episode
  len   episode length in steps

Run on the box:  python tools/curves_from_replay.py ~/logdir/run1 ~/logdir/run2 ...
A run resumed from a checkpoint keeps the chunks written between that
checkpoint and the crash, so its step axis runs long by that gap (~1%).
"""

import argparse
import json
import pathlib

import numpy as np


def curve(logdir):
  chunks = sorted((pathlib.Path(logdir).expanduser() / 'replay').glob('*.npz'))
  ends, achs, lens = [], [], []
  step, start, unlocked = 0, None, None
  for path in chunks:
    with np.load(path) as z:
      ach, first = z['ach'] > 0.5, z['is_first']
    for t in range(len(first)):
      if first[t]:
        if start is not None:
          ends.append(step)
          achs.append(int(unlocked.sum()))
          lens.append(step - start)
        start, unlocked = step, np.zeros(ach.shape[1], bool)
      if unlocked is not None:
        unlocked |= ach[t]
      step += 1
  return dict(end=ends, ach=achs, len=lens, steps=step, chunks=len(chunks))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logdirs', nargs='+')
  ap.add_argument('--out', default='~/logdir/curves')
  args = ap.parse_args()
  out = pathlib.Path(args.out).expanduser()
  out.mkdir(parents=True, exist_ok=True)
  for logdir in args.logdirs:
    name = pathlib.Path(logdir).expanduser().name
    c = curve(logdir)
    (out / f'{name}.json').write_text(json.dumps(c))
    print(f'{name}: {c["chunks"]} chunks, {c["steps"]} steps, '
          f'{len(c["ach"])} episodes, last 50 eps mean '
          f'{np.mean(c["ach"][-50:]):.2f} achievements')


if __name__ == '__main__':
  main()
