"""Tests for the observation-only validity checker.

The one that matters is the comparison with the game itself: for thousands of
states -- ordinary play, plus states seeded with random inventories, tables,
furnaces, enchantment tables, potions, mana and XP so every rule is exercised --
the checker, reading only the observation vector, must agree with what the game
actually does when each action is pressed. Skipped where Craftax is absent.
"""

import inspect

import numpy as np

from dreamerv3 import craftax_valid as V


def test_reads_nothing_but_the_observation():
  assert list(inspect.signature(V.valid_actions).parameters) == ['vec']


def test_basic_actions_are_never_masked():
  vec = np.zeros(8268, np.float32)
  out = V.valid_actions(vec)
  assert out[list(V.BASIC)].all()


def _craftax():
  try:
    import jax
    from embodied.envs.craftax import Craftax
  except Exception:
    return None
  return jax, Craftax


def seeded_states(n=600, seed=0):
  """Real states, many with random inventories and stations placed nearby."""
  import jax
  import jax.numpy as jnp
  from embodied.envs.craftax import Craftax
  env = Craftax(seed=seed)
  rng = np.random.default_rng(seed)
  env.step({'action': np.zeros((), np.int32), 'reset': np.ones((), bool)})
  out = []
  with jax.transfer_guard('allow'):
    for t in range(n):
      obs = env.step({'action': np.int32(rng.integers(0, 43)),
                      'reset': np.zeros((), bool)})
      if obs['is_last']:
        env.step({'action': np.zeros((), np.int32), 'reset': np.ones((), bool)})
      s = env._state
      if rng.random() < 0.7:
        inv = s.inventory
        r = lambda hi: jnp.asarray(int(rng.integers(0, hi)), jnp.int32)
        s = s.replace(
            inventory=inv.replace(
                wood=r(5), stone=r(5), coal=r(5), iron=r(5), diamond=r(5),
                sapphire=r(3), ruby=r(3), sapling=r(3), torches=r(3),
                arrows=r(3), books=r(3), pickaxe=r(5), sword=r(5), bow=r(2),
                potions=jnp.asarray(rng.integers(0, 2, 6), jnp.int32),
                armour=jnp.asarray(rng.integers(0, 3, 4), jnp.int32)),
            player_mana=jnp.asarray(float(rng.integers(0, 12)), jnp.float32),
            player_xp=jnp.asarray(int(rng.integers(0, 3)), jnp.int32),
            player_dexterity=jnp.asarray(int(rng.integers(1, 6)), jnp.int32),
            player_strength=jnp.asarray(int(rng.integers(1, 6)), jnp.int32),
            player_intelligence=jnp.asarray(int(rng.integers(1, 6)), jnp.int32),
            learned_spells=jnp.asarray(rng.random(2) < 0.5),
            player_direction=jnp.asarray(int(rng.integers(1, 5)), jnp.int32))
        y, x = (int(v) for v in np.asarray(s.player_position))
        lvl = int(s.player_level)
        m = s.map
        for block in (11, 12, 30, 31):           # table, furnace, enchant x2
          if rng.random() < 0.4:
            dy, dx = V._AROUND[int(rng.integers(0, 8))]
            if 0 <= y + dy < 48 and 0 <= x + dx < 48:
              m = m.at[lvl, y + dy, x + dx].set(block)
        s = s.replace(map=m)
      out.append(s)
  return env, out


def agreement(n=600, seed=0):
  """Per-action agreement between the checker and the game, actions 6..42."""
  import jax
  import sys
  sys.path.insert(0, 'tools')
  from action_suppression import make_oracle
  env, states = seeded_states(n, seed)
  oracle = make_oracle(env)
  hits = np.zeros(V.N_ACTIONS)
  tot = np.zeros(V.N_ACTIONS)
  wrong = {}
  with jax.transfer_guard('allow'):
    key = jax.random.PRNGKey(seed)
    for s in states:
      key, sub = jax.random.split(key)
      truth = np.asarray(oracle(sub, s))
      vec = np.asarray(env._get_obs_fn(s), np.float32)
      got = V.valid_actions(vec)
      for a in range(6, V.N_ACTIONS):
        tot[a] += 1
        hits[a] += got[a] == truth[a]
        if got[a] != truth[a]:
          wrong.setdefault(a, []).append((bool(got[a]), bool(truth[a])))
  return hits, tot, wrong


def test_agrees_with_the_game():
  if _craftax() is None:
    return
  hits, tot, wrong = agreement()
  rate = hits[6:] / tot[6:]
  bad = {a: (round(float(hits[a] / tot[a]), 3), wrong[a][:3])
         for a in range(6, V.N_ACTIONS) if hits[a] / tot[a] < 0.99}
  print('worst per-action agreement', float(rate.min()), 'mean', float(rate.mean()))
  assert not bad, bad
