"""How much of the 12x12 coarse map does the agent actually observe?

That fraction is exactly how much supervision survives if RSSM-2 stops being
graded against cells it never saw. Reports the three candidate masks:

  causal          seen so far at time t        -- strictest, no lookahead
  window-final    seen by the end of the 64-step training window (free: the
                  loss already slices per window)
  episode-final   seen by the end of the episode -- best, needs a back-fill

Also splits the coarse map by plane, because "never observed" costs nothing on
a plane that is empty there anyway.
"""

import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, '.')

COARSE, CELL = 12, 4
OBS_H, OBS_W = 9, 11
WINDOW = 64


def build(argv, logdir):
  import elements
  import ruamel.yaml as yaml
  from dreamerv3 import main as m
  cfg_path = pathlib.Path(m.__file__).parent / 'configs.yaml'
  cfgs = yaml.YAML(typ='safe').load(cfg_path.read_text(encoding='utf-8'))
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(cfgs['defaults'])
  for name in parsed.configs:
    config = config.update(cfgs[name])
  return elements.Flags(config).parse(other + [f'--logdir={logdir}'])


def stamp(mask, pos):
  """Mark the coarse cells the 9x11 window overlaps. Mirrors update_seen."""
  y, x = int(pos[0]), int(pos[1])
  cy0, cy1 = max(0, (y - OBS_H // 2) // CELL), min(11, (y + OBS_H // 2) // CELL)
  cx0, cx1 = max(0, (x - OBS_W // 2) // CELL), min(11, (x + OBS_W // 2) // CELL)
  if cy0 <= cy1 and cx0 <= cx1:
    mask[cy0:cy1 + 1, cx0:cx1 + 1] = True


def main():
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--episodes', type=int, default=8)
  ap.add_argument('--max-steps', type=int, default=600)
  known, rest = ap.parse_known_args()

  config = build(rest, known.logdir)
  import jax
  from dreamerv3 import main as dv3main

  agent = dv3main.make_agent(config)
  env = dv3main.make_env(config, 0)
  ckdir = pathlib.Path(known.logdir) / 'ckpt'
  target = ckdir / (ckdir / 'latest').read_text().strip()
  agent.load(pickle.loads((target / 'agent.pkl').read_bytes()))

  causal, winfinal, epfinal, lens = [], [], [], []

  for ep in range(known.episodes):
    obs = env.step({'action': np.zeros((), np.int32),
                    'reset': np.ones((), bool)})
    carry = agent.init_policy(batch_size=1)
    run = np.zeros((COARSE, COARSE), bool)
    per_step, windows, cur = [], [], np.zeros((COARSE, COARSE), bool)
    step = 0
    while step < known.max_steps:
      batched = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith('log/')}
      carry, act, _ = agent.policy(carry, batched, mode='probe')
      with jax.transfer_guard('allow'):
        pos = np.asarray(env._state.player_position).reshape(2)
      stamp(run, pos)
      stamp(cur, pos)
      per_step.append(run.sum())
      obs = env.step({'action': np.int32(np.asarray(act['action'])[0]),
                      'reset': np.zeros((), bool)})
      step += 1
      if step % WINDOW == 0:
        windows.append(run.sum())      # seen by the end of this window
        cur = np.zeros((COARSE, COARSE), bool)
      if bool(obs['is_last']):
        break
    if step % WINDOW:
      windows.append(run.sum())
    causal.append(np.mean(per_step))
    winfinal.append(np.mean(windows))
    epfinal.append(run.sum())
    lens.append(step)

  n = COARSE * COARSE
  print(f'\n{known.episodes} episodes, mean length {np.mean(lens):.0f}')
  print(f'coarse cells observed, out of {n}:')
  print(f'  causal        {np.mean(causal):6.1f}  ({np.mean(causal)/n:5.1%})'
        '   supervision if graded only on what it had seen at time t')
  print(f'  window-final  {np.mean(winfinal):6.1f}  '
        f'({np.mean(winfinal)/n:5.1%})   free: seen by end of the 64-step window')
  print(f'  episode-final {np.mean(epfinal):6.1f}  '
        f'({np.mean(epfinal)/n:5.1%})   best: needs an episode-end back-fill')
  print(f'  per-episode episode-final: {epfinal}')


if __name__ == '__main__':
  main()
