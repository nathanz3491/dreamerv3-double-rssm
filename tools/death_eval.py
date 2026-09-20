"""In-game diagnosis of a TRAINED agent: what it does, and what kills it.

Reports per-achievement unlock RATES alongside cause of death. The aggregate
count says the agent got worse; only the per-achievement breakdown says whether
it stopped attempting something, became unreliable at it, or started dying
before it could get there.

death_diagnostic.py answers this for a random policy. This one loads a real
checkpoint and drives the env by hand so it can read the privileged EnvState at
the moment of death -- Craftax's observation does not expose which necessity
ran out, only the agent's own (possibly wrong) belief about it.

Run from the dreamerv3 repo root:
  python tools/death_eval.py --logdir ~/logdir/map_stage5 --episodes 30 \
      --configs craftax size50m --env.craftax.mapmodel True \
      --agent.mapmodel.enabled True --agent.mapmodel.to_actor True
Extra flags are passed through to the config exactly as for main.py, so the
agent is rebuilt with the same architecture the checkpoint was trained with.
"""

import argparse
import collections
import sys

import elements
import jax
import numpy as np

sys.path.insert(0, '.')


def build(argv):
  from dreamerv3 import main as dv3main
  configs = elements.Path(dv3main.__file__).parent / 'configs.yaml'
  configs = elements.Config(elements.Path(configs).read_text(mode='r'), 'yaml') \
      if False else None
  # Reuse main.py's own parsing so a run's flags mean the same thing here.
  import ruamel.yaml as yaml
  cfgs = yaml.YAML(typ='safe').load(
      (elements.Path(dv3main.__file__).parent / 'configs.yaml').read())
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(cfgs['defaults'])
  for name in parsed.configs:
    config = config.update(cfgs[name])
  config = elements.Flags(config).parse(other)
  return config


def main():
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--episodes', type=int, default=30)
  ap.add_argument('--max-steps', type=int, default=1200)
  ap.add_argument('--greedy', action='store_true',
                  help='eval mode; default is the training policy')
  known, rest = ap.parse_known_args()

  config = build(rest + [f'--logdir={known.logdir}'])
  from dreamerv3 import main as dv3main
  agent = dv3main.make_agent(config)
  env = dv3main.make_env(config, 0)

  cp = elements.Checkpoint(elements.Path(known.logdir) / 'ckpt')
  cp.agent = agent
  cp.load(keys=['agent'])

  mode = 'eval' if known.greedy else 'train'
  carry = agent.init_policy(batch_size=1)
  rows, causes = [], collections.Counter()
  try:
    from craftax.craftax.constants import Achievement
    names = [a.name for a in Achievement]
  except Exception:
    names = None

  for ep in range(known.episodes):
    obs = env.step({'action': np.zeros((), np.int32), 'reset': np.ones((), bool)})
    carry = agent.init_policy(batch_size=1)
    first_zero, step = {}, 0
    restores, prev_meters, prev_pos = {}, None, None
    ach_trace = []
    moved, reward_trace = 0, 0.0
    while step < known.max_steps:
      batched = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith('log/')}
      carry, act, _ = agent.policy(carry, batched, mode=mode)
      action = {k: np.asarray(v)[0] for k, v in act.items()}
      action['reset'] = np.zeros((), bool)
      obs = env.step(action)
      step += 1
      # The env scopes its own transfers; these reads are ours, so they need
      # their own escape from the guard (cf. 70a89b8).
      with jax.transfer_guard('allow'):
        st = env._state
        meters = dict(food=float(st.player_food), drink=float(st.player_drink),
                      energy=float(st.player_energy))
        pos = tuple(np.asarray(st.player_position).reshape(-1).tolist())
      for name, val in meters.items():
        if val <= 0 and name not in first_zero:
          first_zero[name] = step
      # Running achievement total, sampled every step. Averaging THIS over the
      # episode reproduces `epstats/log/achievements/avg`, which is a per-step
      # mean of a quantity that climbs 0 -> final and so lands ~35% below the
      # final count. Tracking both keeps eval numbers comparable to the curves.
      with jax.transfer_guard('allow'):
        ach_trace.append(int(np.asarray(env._state.achievements).sum()))
      # Count restore EVENTS the way the shaping does: a meter rising while it
      # was already low. This is what the `restore` term actually pays for, so
      # it is what a reward-farming loop would show up in.
      if prev_meters:
        for name, val in meters.items():
          if val > prev_meters[name] and prev_meters[name] <= 3.0:
            restores[name] = restores.get(name, 0) + 1
      prev_meters = meters
      moved += int(pos != prev_pos)
      prev_pos = pos
      reward_trace += float(obs['reward'])
      if bool(obs['is_last']):
        break

    with jax.transfer_guard('allow'):
      st = env._state
      hp = float(st.player_health)
      terminal = dict(
          food=float(st.player_food), drink=float(st.player_drink),
          energy=float(st.player_energy))
      # Read the bitmap off the state, not obs['ach']: the latter counts unlock
      # EVENTS per step, this is what the agent ended the episode holding.
      unlocked = np.asarray(st.achievements, bool).reshape(-1)
    if first_zero:
      cause = min(first_zero, key=first_zero.get)
      causes[f'{cause} ran out first'] += 1
    else:
      causes['no necessity hit zero -> mob damage or fall/lava'] += 1
    rows.append(dict(
        steps=step, health=hp, **terminal,
        ach=int(unlocked.sum()), unlocked=unlocked,
        ach_perstep=float(np.mean(ach_trace)) if ach_trace else 0.0,
        restores=dict(restores), moved=moved, reward=reward_trace,
        zero=dict(first_zero)))
    print(f'  ep {ep + 1:>3}/{known.episodes}  {step:>4} steps  '
          f'{rows[-1]["ach"]} ach  hp {hp:.0f}  '
          f'food {rows[-1]["food"]:.0f} drink {rows[-1]["drink"]:.0f} '
          f'energy {rows[-1]["energy"]:.0f}', flush=True)

  L = np.array([r['steps'] for r in rows])
  print(f'\n=== {known.episodes} episodes, {"GREEDY" if known.greedy else "TRAINING"} '
        f'policy from {known.logdir} ===')
  print(f'lifespan   mean {L.mean():6.1f}   median {np.median(L):6.1f}'
        f'   min {L.min()}   max {L.max()}')
  print(f'achieve    final {np.mean([r["ach"] for r in rows]):.2f}'
        f'   per-step mean {np.mean([r["ach_perstep"] for r in rows]):.2f}'
        f'   (the latter matches epstats/log/achievements/avg)')
  print(f'terminal   health {np.mean([r["health"] for r in rows]):.2f}'
        f'   food {np.mean([r["food"] for r in rows]):.2f}'
        f'   drink {np.mean([r["drink"] for r in rows]):.2f}'
        f'   energy {np.mean([r["energy"] for r in rows]):.2f}')
  for key in ('drink', 'food', 'energy'):
    vals = [r['zero'][key] for r in rows if key in r['zero']]
    if vals:
      print(f'{key:<8} hit zero in {len(vals):>3}/{len(rows)} eps, '
            f'first at mean step {np.mean(vals):.0f}')
  mat = np.stack([r['unlocked'] for r in rows])          # (episodes, n_ach)
  rate = mat.mean(0)
  labels = names if (names and len(names) == len(rate)) else [
      f'ach_{i:02d}' for i in range(len(rate))]
  print('\nper-achievement unlock rate:')
  order = np.argsort(-rate)
  for i in order:
    if rate[i] == 0:
      continue
    print(f'  {labels[i]:<26} {rate[i]:5.2f}  ' + '#' * int(round(rate[i] * 30)))
  dead = [labels[i] for i in order if rate[i] == 0]
  print(f'  -- never unlocked ({len(dead)}): {", ".join(dead)}')

  # Reward economics from the AGENT's point of view: points per step of effort.
  # A term with a high rate is what the policy will chase, regardless of how
  # small its share of the episode total looks.
  eps = len(rows)
  n_drink = sum(r['restores'].get('drink', 0) for r in rows) / eps
  n_food = sum(r['restores'].get('food', 0) for r in rows) / eps
  n_energy = sum(r['restores'].get('energy', 0) for r in rows) / eps
  steps = L.mean()
  ach_mean = np.mean([r['ach'] for r in rows])
  print('\nreward economics (per episode, and per step of effort):')
  print(f'  restore events   drink {n_drink:5.1f}   food {n_food:5.1f}'
        f'   energy {n_energy:5.1f}')
  restore_pts = 0.3 * (n_drink + n_food + n_energy)
  alive_pts = 0.005 * steps
  print(f'  {"restore @0.3":<18} {restore_pts:7.2f} pts')
  print(f'  {"alive @0.005/step":<18} {alive_pts:7.2f} pts')
  print(f'  {"achievements @1.0":<18} {ach_mean:7.2f} pts')
  print(f'  moved on {np.mean([r["moved"] for r in rows]) / steps:.0%} of steps'
        f'   mean shaped return {np.mean([r["reward"] for r in rows]):.2f}')

  print('\ncause of death:')
  for cause, n in causes.most_common():
    print(f'  {n:>3}  {cause}')
  print('\nreference: neglect ceiling (do nothing) = 333 steps; '
        'random policy = 261-284')


if __name__ == '__main__':
  main()
