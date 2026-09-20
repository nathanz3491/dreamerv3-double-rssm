"""Watch a trained Craftax agent play, one step at a time.

Renders the agent's own 9x11 view as ASCII beside its vitals, the action it
chose, the actor's top action probabilities, its entropy, and
the achievements as they unlock. Paced so a human can follow it.

Run (CPU is fine, and keeps the GPU free for training):
  python tools/watch_agent.py --logdir ~/logdir/map_pot_fixed --episodes 5 \
      --delay 1.5 --configs craftax size50m \
      --env.craftax.mapmodel True --env.craftax.survival potential \
      --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
      --agent.mapmodel.imag_shift False --agent.imag_length 15 \
      --jax.platform cpu
"""

import argparse
import sys
import time

import elements
import jax
import numpy as np

sys.path.insert(0, '.')

# One glyph per block type. Chosen so terrain recedes and the things the agent
# is supposed to care about stand out.
GLYPH = {
    0: ' ', 1: '.', 2: ',', 3: '~', 4: '#', 5: 'T', 6: '"', 7: '.',
    8: 'c', 9: 'i', 10: 'D', 11: 'C', 12: 'F', 13: ':', 14: '!',
    15: 'p', 16: 'P', 17: '#', 19: '#', 20: '^', 21: 'S', 22: 'R', 27: '#',
    28: 'T', 29: 't',
}
LEGEND = ('~water  #stone  Ttree  ,grass  :sand  !LAVA  C table  F furnace  '
          'c coal  i iron  D diamond  p plant  @you')

BAR = '#'


def build(argv):
  import ruamel.yaml as yaml
  from dreamerv3 import main as m
  cfgs = yaml.YAML(typ='safe').load(
      (elements.Path(m.__file__).parent / 'configs.yaml').read())
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(cfgs['defaults'])
  for name in parsed.configs:
    config = config.update(cfgs[name])
  return elements.Flags(config).parse(other)


def view(state, half_h=4, half_w=5):
  """The agent's own window, as text, with the agent at the centre."""
  blocks = np.asarray(state.map)
  if blocks.ndim == 3:
    blocks = blocks[int(state.player_level)]
  y, x = np.asarray(state.player_position).reshape(2).astype(int)
  rows = []
  for dy in range(-half_h, half_h + 1):
    row = ''
    for dx in range(-half_w, half_w + 1):
      yy, xx = y + dy, x + dx
      if dy == 0 and dx == 0:
        row += '@'
      elif 0 <= yy < blocks.shape[0] and 0 <= xx < blocks.shape[1]:
        row += GLYPH.get(int(blocks[yy, xx]), '?')
      else:
        row += ' '
    rows.append(row)
  return rows


def bar(value, limit=9, width=9):
  n = int(round(width * max(0.0, min(1.0, value / limit))))
  return BAR * n + '-' * (width - n)


def main():
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--episodes', type=int, default=5)
  ap.add_argument('--delay', type=float, default=1.5)
  ap.add_argument('--max-steps', type=int, default=600)
  ap.add_argument('--topk', type=int, default=4)
  known, rest = ap.parse_known_args()

  config = build(rest + [f'--logdir={known.logdir}'])
  from dreamerv3 import main as dv3main
  from craftax.craftax.constants import Action, Achievement
  agent = dv3main.make_agent(config)
  env = dv3main.make_env(config, 0)
  cp = elements.Checkpoint(elements.Path(known.logdir) / 'ckpt')
  cp.agent = agent
  cp.load(keys=['agent'])

  anames = [a.name for a in Achievement]
  actnames = {a.value: a.name for a in Action}

  for ep in range(known.episodes):
    obs = env.step({'action': np.zeros((), np.int32), 'reset': np.ones((), bool)})
    carry = agent.init_policy(batch_size=1)
    seen, total, step = set(), 0.0, 0
    print('\n' + '=' * 64)
    print(f'EPISODE {ep + 1} / {known.episodes}')
    print('=' * 64)

    while step < known.max_steps:
      batched = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith('log/')}
      carry, act, out = agent.policy(carry, batched, mode='probe')
      a = int(np.asarray(act['action'])[0])
      probs = np.asarray(out['policy_prob'])[0] if 'policy_prob' in out else None
      # Normalised so 1.0 is uniform over all actions, matching the
      # train/rand/action metric the runs are compared on.
      if probs is None:
        ent = float('nan')
      else:
        ent = float(-(probs * np.log(probs + 1e-9)).sum() / np.log(len(probs)))

      obs = env.step({'action': np.int32(a), 'reset': np.zeros((), bool)})
      step += 1
      total += float(obs['reward'])
      with jax.transfer_guard('allow'):
        st = env._state
        hp = float(st.player_health)
        food, drink = float(st.player_food), float(st.player_drink)
        energy = float(st.player_energy)
        unlocked = np.asarray(st.achievements, bool).reshape(-1)
      rows = view(st)

      new = {anames[i] for i in np.nonzero(unlocked)[0]} - seen
      seen |= new

      print(f'\nstep {step:>4}   action {actnames.get(a, a):<18}'
            f'   reward {float(obs["reward"]):+7.3f}   return {total:+8.3f}')
      stats = [f'health {bar(hp)} {hp:.0f}', f'food   {bar(food)} {food:.0f}',
               f'drink  {bar(drink)} {drink:.0f}',
               f'energy {bar(energy)} {energy:.0f}']
      for i, row in enumerate(rows):
        right = stats[i - 1] if 1 <= i <= len(stats) else ''
        if i == 0:
          right = f'achievements {int(unlocked.sum())}   entropy {ent:.3f}'
        if i == 6 and probs is not None:
          top = np.argsort(-probs)[:known.topk]
          right = 'policy: ' + '  '.join(
              f'{actnames.get(int(j), j)} {probs[j]:.0%}' for j in top)
        print(f'   |{row}|   {right}')
      if new:
        print(f'   *** UNLOCKED: {", ".join(sorted(new))}')
      if bool(obs['is_last']):
        cause = 'died' if hp <= 0 else 'ended'
        print(f'\n   {cause} at step {step} with {int(unlocked.sum())} '
              f'achievements, return {total:+.2f}')
        break
      time.sleep(known.delay)

  print('\n' + LEGEND)


if __name__ == '__main__':
  main()
