"""Does the agent ever press craft, when crafting is trivially available?

The plateau has two candidate causes and they need different fixes:

  exploration   MAKE_WOOD_PICKAXE is never SAMPLED, so its reward is
                unreachable and no reward design can help.
  credit        it is sampled but not reinforced, so reward design matters.

This tells them apart by removing every other obstacle. The agent is dropped
into a hand-built state: standing next to its own crafting table, wood already
in inventory, meters full. From there `MAKE_WOOD_PICKAXE` is a single keypress
with its precondition satisfied. If the action still never appears, the
bottleneck is exploration, full stop.

Run (CPU is plenty):
  python tools/craft_probe.py --logdir <logdir> --trials 200 \
      --configs craftax size50m --env.craftax.mapmodel True \
      --agent.mapmodel.enabled True --agent.mapmodel.to_actor True
"""

import argparse
import collections
import dataclasses
import sys

import elements
import jax
import numpy as np

sys.path.insert(0, '.')


def build(argv):
  """Reuse main.py's own parsing so flags mean the same thing as in training."""
  import ruamel.yaml as yaml
  from dreamerv3 import main as dv3main
  cfgs = yaml.YAML(typ='safe').load(
      (elements.Path(dv3main.__file__).parent / 'configs.yaml').read())
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(cfgs['defaults'])
  for name in parsed.configs:
    config = config.update(cfgs[name])
  return elements.Flags(config).parse(other)


def rig_state(state, wood=5):
  """Put a crafting table beside the agent and wood in the bag.

  Craftax requires being WITHIN ONE TILE of a table to craft, so the table goes
  on the tile the agent faces-adjacent; meters are topped up so nothing else
  competes for the agent's attention during the probe.
  """
  from craftax.craftax.constants import BlockType
  blocks = np.array(state.map)
  y, x = np.asarray(state.player_position).reshape(2).astype(int)
  level = int(state.player_level)

  # Clear a 3x3 around the agent to GRASS so no terrain blocks movement or
  # hides the table, then place the table directly north.
  def put(arr, yy, xx, val):
    if 0 <= yy < arr.shape[-2] and 0 <= xx < arr.shape[-1]:
      arr[yy, xx] = val

  plane = blocks[level] if blocks.ndim == 3 else blocks
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      put(plane, y + dy, x + dx, BlockType.GRASS.value)
  put(plane, y - 1, x, BlockType.CRAFTING_TABLE.value)
  if blocks.ndim == 3:
    blocks[level] = plane
  else:
    blocks = plane

  # Preserve each field's dtype: Craftax mixes int32 (inventory) and float32
  # (health), and jax.lax.scan inside the step function rejects a carry whose
  # dtypes changed, so a plain np.int32(9) here fails at the next step.
  like = lambda ref, val: np.asarray(val, np.asarray(ref).dtype)
  inv = dataclasses.replace(
      state.inventory, wood=like(state.inventory.wood, wood))
  return dataclasses.replace(
      state, map=blocks, inventory=inv,
      player_food=like(state.player_food, 9),
      player_drink=like(state.player_drink, 9),
      player_energy=like(state.player_energy, 9),
      player_health=like(state.player_health, 9))


def main():
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--trials', type=int, default=200)
  ap.add_argument('--steps', type=int, default=40,
                  help='steps to run from each rigged state')
  ap.add_argument('--greedy', action='store_true')
  known, rest = ap.parse_known_args()

  config = build(rest + [f'--logdir={known.logdir}'])
  from dreamerv3 import main as dv3main
  from craftax.craftax.constants import Action
  agent = dv3main.make_agent(config)
  env = dv3main.make_env(config, 0)

  cp = elements.Checkpoint(elements.Path(known.logdir) / 'ckpt')
  cp.agent = agent
  cp.load(keys=['agent'])

  craft = Action.MAKE_WOOD_PICKAXE.value
  names = {a.value: a.name for a in Action}
  mode = 'eval' if known.greedy else 'train'

  taken = collections.Counter()
  crafted = 0
  craft_pressed = 0

  for trial in range(known.trials):
    env.step({'action': np.zeros((), np.int32), 'reset': np.ones((), bool)})
    # Rig the state, then hand it back through the env's own snapshot API --
    # it regenerates the observation under a transfer guard, which hand-built
    # numpy fields would otherwise trip.
    with jax.transfer_guard('allow'):
      state, prev_ach = env.save_state()
      state = jax.device_put(rig_state(state))
      before = bool(np.asarray(state.inventory.pickaxe) > 0)
    obs = env.reset_to((state, prev_ach))
    carry = agent.init_policy(batch_size=1)
    pressed = False
    for _ in range(known.steps):
      batched = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith('log/')}
      carry, act, _ = agent.policy(carry, batched, mode=mode)
      a = int(np.asarray(act['action'])[0])
      taken[a] += 1
      if a == craft:
        pressed = True
      obs = env.step({'action': np.int32(a), 'reset': np.zeros((), bool)})
      if bool(obs['is_last']):
        break
    with jax.transfer_guard('allow'):
      after = bool(np.asarray(env._state.inventory.pickaxe) > 0)
    craft_pressed += int(pressed)
    crafted += int(after and not before)
    if (trial + 1) % 25 == 0:
      print(f'  {trial + 1}/{known.trials}  pressed craft in '
            f'{craft_pressed} trials, actually crafted {crafted}', flush=True)

  total = sum(taken.values())
  print(f'\n=== {known.trials} rigged trials x {known.steps} steps, '
        f'{"GREEDY" if known.greedy else "TRAINING"} policy ===')
  print('state: crafting table one tile north, 5 wood, all meters full\n')
  print(f'MAKE_WOOD_PICKAXE pressed in   {craft_pressed}/{known.trials} trials')
  print(f'pickaxe actually obtained in   {crafted}/{known.trials} trials')
  print(f'\naction distribution over {total} steps (top 12):')
  for a, n in taken.most_common(12):
    print(f'  {names.get(a, a):<24} {n / total:6.1%}  {n}')
  print(f'\ndistinct actions used: {len(taken)} of {len(names)}')
  if craft_pressed == 0:
    print('\nVERDICT: craft is NEVER sampled even with the precondition met.')
    print('The bottleneck is EXPLORATION -- no reward attached to crafting can')
    print('be collected, because the event never occurs. Reward design is')
    print('premature; raise actent (or otherwise restore exploration) first.')
  elif crafted == 0:
    print('\nVERDICT: craft is sampled but never succeeds -- check the rig '
          '(adjacency/inventory) before drawing conclusions.')
  else:
    print('\nVERDICT: the agent CAN craft from a handed-to-it state. The '
          'failure is reaching that state on its own: credit assignment over '
          'the setup sequence. Potential-based shaping over the tech tree is '
          'the right tool.')


if __name__ == '__main__':
  main()
