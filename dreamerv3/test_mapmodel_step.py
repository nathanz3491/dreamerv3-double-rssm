"""RSSM-2's per-step logic: movement deltas, episode resets, train == act.

The step function (Agent._map_step) is tested with a stand-in for the GRU, a
fixed linear map, so every property is exact: what matters here is which
inputs reach which state and when, not what the network learns.

Run: python -m pytest dreamerv3/test_mapmodel_step.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3 import agent as agentlib
from dreamerv3 import mapmodel as mapmod
import embodied.jax.nets as nn

f32 = jnp.float32
D, F, M, TICK = 4, 4, 2, 8


class _MapModel:

  @staticmethod
  def tick(deter2, x):
    # Depends on the previous state and on the window's mean features, so a
    # leak of either an old episode or a future step shows up exactly.
    return 0.5 * f32(deter2) + f32(x)[:, :D]


class _Stub:
  _map_tick = TICK
  mapmodel = _MapModel()


def step(carry, feat, move, reset):
  return agentlib.Agent._map_step(_Stub(), carry, feat, move, reset)


def initial(B, deter2=None, count=0.0):
  return nn.cast(dict(
      deter2=jnp.zeros((B, D), f32) if deter2 is None else f32(deter2),
      featsum=jnp.zeros((B, F), f32), movesum=jnp.zeros((B, M), f32),
      count=jnp.full((B, 1), count, f32), n=jnp.zeros((B, 1), f32)))


def online(carry, feats, moves, resets):
  """The acting path: one call per env step."""
  outs, closes = [], []
  for t in range(feats.shape[1]):
    carry, closed, _ = step(carry, feats[:, t], moves[:, t], resets[:, t])
    outs.append(f32(carry['deter2']))
    closes.append(closed)
  return jnp.stack(outs, 1), jnp.stack(closes, 1)


def batch(carry, feats, moves, resets):
  """The training path: the same step, scanned over time as in Agent.loss."""
  def body(c, xs):
    c, closed, _ = step(c, *xs)
    return c, (f32(c['deter2']), closed)
  swap = lambda x: jnp.swapaxes(x, 0, 1)
  _, (outs, closes) = jax.lax.scan(
      body, carry, (swap(feats), swap(moves), swap(resets)))
  return swap(outs), swap(closes)


def trajectory(B=3, T=40, seed=0, resets_at=((0, 0), (1, 0), (1, 13), (2, 0),
                                             (2, 21))):
  rng = np.random.default_rng(seed)
  feats = f32(rng.normal(size=(B, T, F)))
  moves = f32(rng.normal(size=(B, T, M)))
  resets = np.zeros((B, T), bool)
  for b, t in resets_at:
    resets[b, t] = True
  return feats, moves, jnp.asarray(resets)


# --- A: movement deltas ---------------------------------------------------------
@pytest.mark.parametrize('action', range(43))
def test_only_the_four_moves_move(action):
  expected = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action, (0, 0))
  got = np.asarray(mapmod.action_deltas(jnp.array([action])))[0]
  assert tuple(got) == expected


# --- B: a new episode starts from an empty state --------------------------------
def test_reset_mid_window_clears_the_old_state_at_once():
  carry = initial(1, deter2=np.full((1, D), 7.0), count=3.0)
  feat, move = jnp.ones((1, F), f32), jnp.ones((1, M), f32)
  carry, closed, _ = step(carry, feat, move, jnp.array([True]))
  assert not bool(closed[0])
  np.testing.assert_array_equal(np.asarray(f32(carry['deter2'])), 0.0)
  assert float(carry['count'][0, 0]) == 1.0


def test_new_episode_does_not_depend_on_the_old_one():
  feats, moves, resets = trajectory(B=2, T=24, resets_at=((0, 0), (1, 0)))
  dirty = initial(2, deter2=np.full((2, D), 5.0), count=6.0)
  dirty = {**dirty, **nn.cast(dict(
      featsum=jnp.full((2, F), 3.0, f32), movesum=jnp.full((2, M), 3.0, f32),
      n=jnp.full((2, 1), 6.0, f32)))}
  out_dirty, _ = online(dirty, feats, moves, resets)
  out_clean, _ = online(initial(2), feats, moves, resets)
  np.testing.assert_allclose(np.asarray(out_dirty), np.asarray(out_clean),
                             atol=1e-2)


# --- C: training sees exactly what acting saw, and nothing later ----------------
def test_batch_path_equals_online_path():
  feats, moves, resets = trajectory()
  out_on, close_on = online(initial(3), feats, moves, resets)
  out_b, close_b = batch(initial(3), feats, moves, resets)
  np.testing.assert_array_equal(np.asarray(close_on), np.asarray(close_b))
  np.testing.assert_allclose(np.asarray(out_on), np.asarray(out_b), atol=1e-2)


def test_windows_close_every_tick_steps_from_the_episode_start():
  feats, moves, resets = trajectory()
  _, closes = online(initial(3), feats, moves, resets)
  closes = np.asarray(closes)
  # Row 1 restarts at step 13: closes at 7, then 13 + 7 = 20, 28, 36.
  assert list(np.flatnonzero(closes[1])) == [7, 20, 28, 36]
  # Row 2 restarts at step 21: closes at 7, 15, then 28, 36.
  assert list(np.flatnonzero(closes[2])) == [7, 15, 28, 36]


@pytest.mark.parametrize('t', [0, 5, 7, 8, 19, 30])
def test_no_step_sees_its_future(t):
  feats, moves, resets = trajectory()
  base, _ = batch(initial(3), feats, moves, resets)
  later = feats.at[:, t + 1:].add(100.0)
  pert, _ = batch(initial(3), later, moves, resets)
  np.testing.assert_array_equal(np.asarray(base[:, :t + 1]),
                                np.asarray(pert[:, :t + 1]))


def test_state_holds_between_closes():
  feats, moves, resets = trajectory(B=1, T=24, resets_at=((0, 0),))
  out, closes = online(initial(1), feats, moves, resets)
  out, closes = np.asarray(out[0]), np.asarray(closes[0])
  np.testing.assert_array_equal(out[:7], 0.0)       # nothing closed yet
  np.testing.assert_array_equal(out[7:15], np.broadcast_to(out[7], (8, D)))
  assert not np.allclose(out[15], out[7])


def test_resumed_chunk_closes_its_first_window_on_time():
  # A chunk resumed at window step 5 (count restored, partial sums lost)
  # closes after 3 more steps, averaging the 3 steps it has.
  feats = f32(np.arange(1, 4 * F + 1).reshape(1, 4, F))
  moves = jnp.ones((1, 4, M), f32)
  carry = initial(1, count=5.0)
  carry, closed, inp = step(carry, feats[:, 0], moves[:, 0], jnp.array([False]))
  carry, closed, inp = step(carry, feats[:, 1], moves[:, 1], jnp.array([False]))
  assert not bool(closed[0])
  carry, closed, inp = step(carry, feats[:, 2], moves[:, 2], jnp.array([False]))
  assert bool(closed[0])
  np.testing.assert_allclose(np.asarray(inp[0, :F]),
                             np.asarray(feats[0, :3].mean(0)), rtol=1e-2)
  # Move sum rescaled from the 3 steps seen to the window's 8.
  np.testing.assert_allclose(np.asarray(inp[0, F:F + M]), 8.0)
