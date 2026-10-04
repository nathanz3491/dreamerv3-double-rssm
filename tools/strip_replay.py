"""Shrink a finished run for archiving: drop the latents cached in its replay.

With replay_context on, every replay chunk also stores the agent's latent
state at each step (dyn/deter, dyn/stoch, map/deter2, ...) so training can
resume a sequence mid-episode. Those are random-looking floats that barely
compress: dyn/deter alone is ~89% of a run's disk, ~7.6 of 9.3 GB. Nothing
after training reads them -- curves_from_replay, diagnose_run and
episode_cases need only observations, actions and achievements -- so a finished
run keeps everything else and shrinks to ~1.4 GB (replay ~0.8 GB, checkpoint
~0.7 GB). The cost: the run can no longer be resumed from its replay.

Guards, so a live run is never touched:
  - refuses if any process is training on that logdir;
  - refuses a run whose last logged step is short of its config's run.steps
    (a crashed run may still need resuming) unless --allow-incomplete.

  python tools/strip_replay.py ~/logdir/vanilla ~/logdir/expB_mask
  python tools/strip_replay.py --wait ~/logdir/mgr ~/logdir/mgr2   # queued
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import subprocess
import time
import zipfile

LATENT = ('dyn/', 'enc/', 'dec/', 'map/')    # replay_context entry prefixes


def training(logdir):
  """Is a process training on this logdir right now?"""
  out = subprocess.run(['pgrep', '-f', '--', f'--logdir {logdir} '],
                       capture_output=True, text=True).stdout.split()
  return [p for p in out if int(p) != os.getpid()]


def finished(logdir):
  """(done, last step, target): did the run reach its configured run.steps?"""
  target = None
  for line in (logdir / 'config.yaml').read_text().splitlines():
    if line.strip().startswith('steps:') and target is None:
      target = float(line.split(':', 1)[1])
  last = 0
  metrics = logdir / 'metrics.jsonl'
  if metrics.exists():
    with metrics.open() as f:
      for line in f:
        last = max(last, json.loads(line).get('step', 0))
  return target is not None and last >= 0.999 * target, last, target


def strip_chunk(path):
  """Rewrite one .npz without the latent members. Returns bytes saved."""
  before = path.stat().st_size
  with zipfile.ZipFile(path) as zin:
    keep = [i for i in zin.infolist()
            if not i.filename.startswith(LATENT)]
    if len(keep) == len(zin.infolist()):
      return 0                                   # already stripped
    tmp = path.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=1) as zout:
      for info in keep:
        zout.writestr(info.filename, zin.read(info.filename))
  os.replace(tmp, path)
  return before - path.stat().st_size


def strip(logdir, allow_incomplete, workers):
  logdir = pathlib.Path(logdir).expanduser().resolve()
  if training(logdir):
    print(f'SKIP {logdir.name}: a process is training on it')
    return False
  done, last, target = finished(logdir)
  if not done and not allow_incomplete:
    print(f'SKIP {logdir.name}: stopped at step {last} of {target} '
          '(pass --allow-incomplete if it will never be resumed)')
    return False
  chunks = sorted((logdir / 'replay').glob('*.npz'))
  with cf.ProcessPoolExecutor(workers) as pool:
    saved = sum(pool.map(strip_chunk, chunks, chunksize=8))
  print(f'{logdir.name}: {len(chunks)} chunks, freed {saved / 1e9:.2f} GB')
  return True


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logdirs', nargs='+')
  ap.add_argument('--allow-incomplete', action='store_true')
  ap.add_argument('--workers', type=int, default=6)
  ap.add_argument('--wait', action='store_true',
                  help='poll until each run has started and finished, then '
                       'strip it (for runs still queued or training)')
  args = ap.parse_args()
  for d in args.logdirs:
    logdir = pathlib.Path(d).expanduser().resolve()
    while args.wait:
      if (logdir / 'config.yaml').exists() and not training(logdir):
        if finished(logdir)[0]:
          break
        print(f'{logdir.name} stopped short of its target; not stripping')
        break
      time.sleep(300)
    strip(logdir, args.allow_incomplete, args.workers)


if __name__ == '__main__':
  main()
