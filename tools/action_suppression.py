"""Experiment A: is low entropy skill, or are valid actions being suppressed?

Rolls a trained agent and, at every step, asks the GAME which of the 43 actions
would do anything: the state is copied, every action is stepped with the same
random key, and an action counts as valid if the result differs from doing
nothing. That is exact ground truth for the state the agent is in. It is used
here only to measure the policy -- nothing flows back into training -- so it may
read the game state freely (see docs/entropy-and-action-suppression.md).

Reports, per run:

  * valid-action mass      probability the policy puts on actions with an effect
  * entropy                conditional H(A|s) and marginal H(E_s[pi]), normalised
                           by ln 43; the gap I(A;s) separates "certain but
                           different in different states" from "one key always"
  * entropy within valid   the policy renormalised over the actions that work
  * opportunities          probability and rank of the right action where it is
                           available -- above all the two crafts at a table
  * suppression ratio      mean probability of each action where it is valid
                           vs where it is not; near 1 means the policy is not
                           conditioning on whether the action can work

Run from the dreamerv3 repo root, with the training flags of the checkpoint:

  python tools/action_suppression.py --logdir ~/logdir/honest_map --episodes 20 \
      --configs craftax size50m --env.craftax.mapmodel True \
      --env.craftax.survival potential --agent.mapmodel.enabled True \
      --agent.mapmodel.to_actor True --agent.mapmodel.imag_shift False \
      --agent.imag_length 15
"""

import argparse
import collections
import functools
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, '.')

N_ACT = 43
TREE, STONE, WATER, TABLE = 5, 4, 3, 11
CLOSE = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

# Actions whose suppression the report breaks out: the tech tree, placing, and
# the keys that are nearly always useless (potions, magic, attributes).
WATCH = ('DO', 'PLACE_TABLE', 'MAKE_WOOD_PICKAXE', 'MAKE_WOOD_SWORD',
         'PLACE_STONE', 'PLACE_FURNACE', 'MAKE_STONE_PICKAXE',
         'MAKE_STONE_SWORD', 'PLACE_PLANT', 'SLEEP', 'DRINK_POTION_GREEN',
         'LEVEL_UP_INTELLIGENCE', 'MAKE_DIAMOND_ARMOUR')


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


def load_agent(agent, logdir):
  ckdir = pathlib.Path(logdir).expanduser() / 'ckpt'
  target = ckdir / (ckdir / 'latest').read_text().strip()
  agent.load(pickle.loads((target / 'agent.pkl').read_bytes()))
  return target


def make_oracle(env):
  """jit(key, state) -> (43,) bool: does each action change the game state?"""
  import jax
  import jax.numpy as jnp
  cenv, params = env._env, env._params

  def nxt(key, state, a):
    return cenv.step(key, state, a, params)[1]

  @jax.jit
  def effects(key, state):
    acts = jnp.arange(N_ACT)
    after = jax.vmap(lambda a: nxt(key, state, a))(acts)
    base = nxt(key, state, 0)
    diff = jax.tree.map(
        lambda n, b: (n != b[None]).reshape(N_ACT, -1).any(-1), after, base)
    return functools.reduce(jnp.logical_or, jax.tree.leaves(diff))

  return effects


def situation(state):
  """Labels for the step, read from the true state (measurement only)."""
  blocks = np.asarray(state.map)
  level = int(np.asarray(state.player_level))
  if blocks.ndim == 3:
    blocks = blocks[level]
  y, x = (int(v) for v in np.asarray(state.player_position).reshape(2))
  near = set()
  for dy, dx in CLOSE:
    if 0 <= y + dy < blocks.shape[0] and 0 <= x + dx < blocks.shape[1]:
      near.add(int(blocks[y + dy, x + dx]))
  d = int(np.asarray(state.player_direction))
  fy, fx = y + DIRS.get(d, (0, 0))[0], x + DIRS.get(d, (0, 0))[1]
  facing = (int(blocks[fy, fx]) if 0 <= fy < blocks.shape[0]
            and 0 <= fx < blocks.shape[1] else -1)
  inv = state.inventory
  wood, pick = int(np.asarray(inv.wood)), int(np.asarray(inv.pickaxe))
  tags = []
  if level != 0:
    return ['below surface']
  if TABLE in near and wood >= 1:
    tags.append('at table with wood')
  if facing == TREE:
    tags.append('facing tree')
  if facing == WATER and float(np.asarray(state.player_drink)) < 9:
    tags.append('facing water, thirsty')
  if facing == STONE and pick >= 1:
    tags.append('facing stone, has pickaxe')
  if not near & {TREE, STONE, WATER, TABLE}:
    tags.append('open ground')
  return tags or ['other']


def entropy(p, axis=-1):
  p = np.clip(p, 1e-12, 1.0)
  return -(p * np.log(p)).sum(axis)


def main():
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--episodes', type=int, default=20)
  ap.add_argument('--max-steps', type=int, default=1200)
  ap.add_argument('--out', default=None)
  known, rest = ap.parse_known_args()

  config = build(rest, known.logdir)
  import jax
  from dreamerv3 import main as dv3main
  from craftax.craftax.constants import Action

  names = [a.name for a in sorted(Action, key=lambda a: a.value)]
  agent = dv3main.make_agent(config)
  env = dv3main.make_env(config, 0)
  ck = load_agent(agent, known.logdir)
  print('loaded', ck)
  effects = make_oracle(env)
  # Separate from the env's own stream, so probing never changes the episode.
  # Created and split under the allow scope: dreamerv3 installs a global
  # transfer guard and a PRNG key is a host-to-device transfer.
  with jax.transfer_guard('allow'):
    key = jax.random.PRNGKey(12345)

  P, V, A, E, S = [], [], [], [], []
  for ep in range(known.episodes):
    obs = env.step({'action': np.zeros((), np.int32), 'reset': np.ones((), bool)})
    carry = agent.init_policy(batch_size=1)
    for step in range(known.max_steps):
      batched = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith('log/')}
      carry, act, out = agent.policy(carry, batched, mode='probe')
      a = int(np.asarray(act['action'])[0])
      with jax.transfer_guard('allow'):
        key, sub = jax.random.split(key)
        state = env._state
        valid = np.asarray(effects(sub, state))
        tags = situation(state)
      P.append(np.asarray(out['policy_prob'])[0])
      V.append(valid)
      A.append(a)
      E.append(bool(valid[a]))
      S.append(tags)
      obs = env.step({'action': np.int32(a), 'reset': np.zeros((), bool)})
      if bool(obs['is_last']):
        break
    print(f'  ep {ep + 1}/{known.episodes}  {step + 1} steps', flush=True)

  P, V, A, E = np.stack(P), np.stack(V), np.array(A), np.array(E)
  n = len(A)
  lnA = np.log(N_ACT)
  res = dict(logdir=str(known.logdir), checkpoint=str(ck), steps=n,
             episodes=known.episodes)

  # --- mass and entropies ----------------------------------------------------
  res['valid_mass'] = float((P * V).sum(-1).mean())
  res['pressed_with_effect'] = float(E.mean())
  res['entropy_conditional'] = float(entropy(P).mean() / lnA)
  res['entropy_marginal'] = float(entropy(P.mean(0)) / lnA)
  res['mutual_information'] = res['entropy_marginal'] - res['entropy_conditional']
  nv = V.sum(-1)
  ok = nv >= 2
  Pv = np.where(V, P, 0)[ok]
  Pv = Pv / np.maximum(Pv.sum(-1, keepdims=True), 1e-12)
  res['entropy_within_valid'] = float((entropy(Pv) / np.log(nv[ok])).mean())
  res['valid_actions_mean'] = float(nv.mean())
  wasted = collections.Counter(names[a] for a, e in zip(A, E) if not e)
  res['top_no_effect_presses'] = {k: v / n for k, v in wasted.most_common(8)}

  # --- by situation ----------------------------------------------------------
  by = collections.defaultdict(list)
  for i, tags in enumerate(S):
    for t in tags:
      by[t].append(i)
  res['situations'] = {}
  for t, idx in sorted(by.items(), key=lambda kv: -len(kv[1])):
    idx = np.array(idx)
    res['situations'][t] = dict(
        share=len(idx) / n,
        entropy=float(entropy(P[idx]).mean() / lnA),
        valid_mass=float((P[idx] * V[idx]).sum(-1).mean()),
        pressed_with_effect=float(E[idx].mean()))

  # --- opportunities ---------------------------------------------------------
  def opp(tag, action):
    a = names.index(action)
    idx = np.array([i for i in by.get(tag, []) if V[i, a]])
    if not len(idx):
      return dict(states=0)
    ranks = (P[idx] > P[idx, a][:, None]).sum(-1) + 1
    return dict(states=int(len(idx)), prob=float(P[idx, a].mean()),
                median_rank=float(np.median(ranks)),
                pressed=float((A[idx] == a).mean()))

  res['opportunities'] = {
      'MAKE_WOOD_PICKAXE @ table with wood': opp('at table with wood',
                                                 'MAKE_WOOD_PICKAXE'),
      'MAKE_WOOD_SWORD @ table with wood': opp('at table with wood',
                                               'MAKE_WOOD_SWORD'),
      'DO @ facing tree': opp('facing tree', 'DO'),
      'DO @ facing water, thirsty': opp('facing water, thirsty', 'DO'),
      'DO @ facing stone, has pickaxe': opp('facing stone, has pickaxe', 'DO'),
  }

  # --- suppression ratio -----------------------------------------------------
  res['suppression'] = {}
  for name in WATCH:
    a = names.index(name)
    on, off = V[:, a], ~V[:, a]
    res['suppression'][name] = dict(
        valid_share=float(on.mean()),
        prob_when_valid=float(P[on, a].mean()) if on.any() else None,
        prob_when_invalid=float(P[off, a].mean()) if off.any() else None)

  report(res)
  out = pathlib.Path(known.out or (
      pathlib.Path(known.logdir).expanduser().parent /
      f'{pathlib.Path(known.logdir).name}_suppression.json'))
  out.write_text(json.dumps(res, indent=1), encoding='utf-8')
  print('wrote', out)


def report(r):
  pct = lambda v: '   -' if v is None else f'{100 * v:5.1f}%'
  print(f"\n{r['logdir']}: {r['steps']} steps over {r['episodes']} episodes")
  print(f"  valid-action mass        {pct(r['valid_mass'])}   "
        f"(avg {r['valid_actions_mean']:.1f} of 43 actions do something)")
  print(f"  presses with an effect   {pct(r['pressed_with_effect'])}")
  print(f"  entropy H(A|s)           {r['entropy_conditional']:.3f}   "
        f"within valid {r['entropy_within_valid']:.3f}")
  print(f"  entropy H(E pi)          {r['entropy_marginal']:.3f}   "
        f"I(A;s) {r['mutual_information']:.3f}")
  print('  no-effect presses: ' + ', '.join(
      f'{k} {100 * v:.1f}%' for k, v in r['top_no_effect_presses'].items()))
  print('\n  situation                       share  entropy  valid mass  effect')
  for t, s in r['situations'].items():
    print(f"  {t:<30} {pct(s['share'])}   {s['entropy']:.3f}     "
          f"{pct(s['valid_mass'])}  {pct(s['pressed_with_effect'])}")
  print('\n  opportunity                          states   prob   rank  pressed')
  for k, o in r['opportunities'].items():
    if not o['states']:
      print(f'  {k:<36} {0:>6}')
      continue
    print(f"  {k:<36} {o['states']:>6}  {pct(o['prob'])}  "
          f"{o['median_rank']:>4.0f}  {pct(o['pressed'])}")
  print('\n  action                 valid in   p | valid   p | invalid')
  for k, s in r['suppression'].items():
    print(f"  {k:<22} {pct(s['valid_share'])}   {pct(s['prob_when_valid'])}"
          f"    {pct(s['prob_when_invalid'])}")


if __name__ == '__main__':
  main()
