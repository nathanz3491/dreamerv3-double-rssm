"""Quantify what a trained Craftax agent actually spends its actions on.

Written to check four things seen by eye in the pygame viewer: that most
presses are actions whose preconditions can never hold, that the state usually
does not change at all, that the agent ignores water it is standing next to,
and that it walks into blocks.

  python action_audit.py --logdir . --episodes 8 --configs craftax size50m ...
"""

import argparse
import collections
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, '.')

WATER, TREE, STONE = 3, 5, 4
# Actions the agent can never complete on floor 1 with the tech it reaches:
# potions, magic, enchanting, attribute points, and the diamond/iron tiers it
# has no ore for. Counting them separately turns "it presses nonsense" into a
# number.
DEAD = set(range(20, 24)) | {24, 26, 27} | set(range(29, 43))


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


def snapshot(state, np_):
  inv = state.inventory
  vec = [getattr(inv, f) for f in inv.__dataclass_fields__]
  return (tuple(np_.asarray(state.player_position).reshape(2).tolist()),
          tuple(float(np_.asarray(v).reshape(-1)[0]) for v in vec),
          int(np_.asarray(state.achievements).sum()))


def main():
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--episodes', type=int, default=8)
  ap.add_argument('--max-steps', type=int, default=600)
  known, rest = ap.parse_known_args()

  config = build(rest, known.logdir)
  import jax
  from dreamerv3 import main as dv3main
  from craftax.craftax.constants import Action, Achievement, DIRECTIONS

  agent = dv3main.make_agent(config)
  env = dv3main.make_env(config, 0)
  ckdir = pathlib.Path(known.logdir) / 'ckpt'
  target = ckdir / (ckdir / 'latest').read_text().strip()
  agent.load(pickle.loads((target / 'agent.pkl').read_bytes()))

  names = {a.value: a.name for a in Action}
  dirs = np.asarray(DIRECTIONS)

  counts = collections.Counter()
  inert = dead = total = blocked = 0
  ent_sum = 0.0
  # opportunity -> taken
  opp = collections.Counter()
  took = collections.Counter()
  eps_ach, eps_len = [], []

  for ep in range(known.episodes):
    obs = env.step({'action': np.zeros((), np.int32),
                    'reset': np.ones((), bool)})
    carry = agent.init_policy(batch_size=1)
    with jax.transfer_guard('allow'):
      prev = snapshot(env._state, np)
    step = 0
    while step < known.max_steps:
      batched = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith('log/')}
      carry, act, out = agent.policy(carry, batched, mode='probe')
      a = int(np.asarray(act['action'])[0])
      p = np.asarray(out['policy_prob'])[0]
      ent_sum += float(-(p * np.log(p + 1e-9)).sum() / np.log(len(p)))

      with jax.transfer_guard('allow'):
        st = env._state
        blocks = np.asarray(st.map)
        if blocks.ndim == 3:
          blocks = blocks[int(st.player_level)]
        pos = np.asarray(st.player_position).reshape(2).astype(int)
        face = pos + dirs[int(np.asarray(st.player_direction))]
        drink = float(st.player_drink)
      inb = (0 <= face[0] < blocks.shape[0] and 0 <= face[1] < blocks.shape[1])
      facing = int(blocks[face[0], face[1]]) if inb else -1
      # An opportunity is: the thing is directly in front of you, so one DO
      # press collects it. Thirsty is < 9 so topping up still counts.
      if facing == WATER and drink < 9:
        opp['water'] += 1
        took['water'] += (a == Action.DO.value)
      if facing == TREE:
        opp['tree'] += 1
        took['tree'] += (a == Action.DO.value)
      if facing == STONE:
        opp['stone'] += 1
        took['stone'] += (a == Action.DO.value)

      obs = env.step({'action': np.int32(a), 'reset': np.zeros((), bool)})
      step += 1
      total += 1
      counts[a] += 1
      dead += a in DEAD
      with jax.transfer_guard('allow'):
        now = snapshot(env._state, np)
      if now == prev:
        inert += 1
        if 1 <= a <= 4:
          blocked += 1
      prev = now
      if bool(obs['is_last']):
        break
    with jax.transfer_guard('allow'):
      eps_ach.append(int(np.asarray(env._state.achievements).sum()))
    eps_len.append(step)

  print(f'\n{known.episodes} episodes, {total} steps')
  print(f'achievements (episode-final): mean {np.mean(eps_ach):.2f}  '
        f'per-ep {eps_ach}')
  print(f'episode length: mean {np.mean(eps_len):.0f}')
  print(f'mean normalised policy entropy: {ent_sum / total:.3f}')
  print(f'\nimpossible actions (potions/magic/enchant/attrib/diamond): '
        f'{dead / total:.1%} of presses')
  print(f'steps where NOTHING changed (pos, inventory, achievements): '
        f'{inert / total:.1%}')
  print(f'  of which a move key was pressed and the agent did not move: '
        f'{blocked / total:.1%}  (walking into a wall)')

  print('\nopportunity taken (thing directly in front, DO would collect it)')
  for k in ('water', 'tree', 'stone'):
    if opp[k]:
      print(f'  {k:<6} faced {opp[k]:>5} times, pressed DO '
            f'{took[k] / opp[k]:>6.1%}')
    else:
      print(f'  {k:<6} never faced')

  print('\ntop 15 actions')
  for a, n in counts.most_common(15):
    flag = '  <- impossible' if a in DEAD else ''
    print(f'  {names.get(a, a):<22} {n / total:>6.1%}{flag}')


if __name__ == '__main__':
  main()
