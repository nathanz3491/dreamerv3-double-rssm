"""Tests for the manager's goal progress, read from the observation alone.

The game check: over real states seeded with random inventories and stations
(the same states test_craftax_valid uses), a goal reads as done exactly when
the true game state says so -- inventory from state.inventory, "near" from the
true map around the true position. Skipped where Craftax is absent.
"""

import inspect

import numpy as np

from dreamerv3 import craftax_goals as G
from dreamerv3 import craftax_potential as P
from dreamerv3 import test_craftax_valid as TV


def test_reads_nothing_but_the_observation():
  assert list(inspect.signature(G.progress).parameters) == ['vec']


def test_shape_range_and_none():
  out = G.progress(np.zeros(8268, np.float32))
  assert out.shape == (G.N_GOALS,) and out.dtype == np.float32
  assert out[0] == 0.0
  assert ((out >= 0) & (out <= 1)).all()


def test_matches_the_game():
  if TV._craftax() is None:
    return
  import jax
  env, states = TV.seeded_states(300, seed=1)
  done = {name: 0 for name in G.GOALS[1:]}
  with jax.transfer_guard('allow'):
    for s in states:
      vec = np.asarray(env._get_obs_fn(s), np.float32)
      got = G.progress(vec)
      assert got[0] == 0.0
      inv = s.inventory
      y, x = (int(v) for v in np.asarray(s.player_position))
      blocks = np.asarray(s.map[int(s.player_level)])
      near = {int(blocks[y + dy, x + dx]) for dy, dx in P.CLOSE
              if 0 <= y + dy < 48 and 0 <= x + dx < 48}
      for i, name in enumerate(G.GOALS[1:], 1):
        kind, *what = G._DONE[name]
        if kind == 'near':
          truth = what[0] in near
        else:
          truth = int(np.asarray(getattr(inv, what[0]))) >= what[1]
        assert (got[i] == 1.0) == truth, (name, got[i], truth)
        assert truth or got[i] <= P.PREREQ_CEIL
        done[name] += truth
  # The seeding must actually exercise both outcomes for every goal.
  assert all(0 < v < len(states) for v in done.values()), done
