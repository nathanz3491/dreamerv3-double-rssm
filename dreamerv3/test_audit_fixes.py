"""Fixes from the 2026-10-11 audit (round 2): fixed evaluation worlds, the
first shaped step, the goal codebook on empty batches, goal starts at replay
boundaries, greedy acting.

Run: python -m pytest dreamerv3/test_audit_fixes.py
The env tests need Craftax installed.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3 import agent as agentlib
from dreamerv3 import goalcodes
from embodied.jax import outs as jouts

f32 = jnp.float32


# --- B05: the world an episode starts in depends only on its seed --------------
def _env(**kw):
  pytest.importorskip('craftax')
  from embodied.envs.craftax import Craftax
  return Craftax(**kw)


def _reset(env):
  return env.step({'action': np.zeros((), np.int32), 'reset': np.ones((), bool)})


def _act(env, a):
  return env.step({'action': np.asarray(a, np.int32),
                   'reset': np.zeros((), bool)})


def test_reseeded_world_ignores_how_long_earlier_episodes_ran():
  a, b = _env(), _env()
  for env, n in ((a, 3), (b, 40)):          # earlier episodes of different length
    env.reseed(100)
    _reset(env)
    for _ in range(n):
      _act(env, 0)
  a.reseed(7)
  b.reseed(7)
  oa, ob = _reset(a), _reset(b)
  np.testing.assert_array_equal(oa['vector'], ob['vector'])


def test_different_seeds_give_different_worlds():
  env = _env()
  env.reseed(1)
  o1 = _reset(env)['vector']
  env.reseed(2)
  o2 = _reset(env)['vector']
  assert not np.array_equal(o1, o2)


# --- B07 and 6.4: the first step is shaped; raw reward is kept apart -----------
def test_first_step_is_shaped_from_the_start_state():
  env = _env(survival='potential')
  env.reseed(3)
  _reset(env)
  phi0 = env._prev_phi
  assert phi0 is not None
  with jax.transfer_guard('allow'):
    assert phi0 == pytest.approx(env._potential(env._state))
  obs = _act(env, 0)
  with jax.transfer_guard('allow'):
    phi1 = env._potential(env._state)
  shaping = float(obs['reward']) - env.raw_reward
  assert shaping == pytest.approx(env._phi_gamma * phi1 - phi0, abs=1e-5)


def test_raw_reward_equals_reward_without_shaping():
  env = _env(survival='none')
  env.reseed(4)
  _reset(env)
  for a in (5, 5, 1, 5):
    obs = _act(env, a)
    assert float(obs['reward']) == pytest.approx(env.raw_reward)


# --- B08: a batch with no valid change leaves the codebook untouched -----------
def _book_after(z, idx, valid, dead=0.02):
  def fn(z, idx, valid):
    book = goalcodes.GoalBook(codes=4, dim=3, dead=dead, name='book')
    before = (book.read(), book.share.read(), book.total.read())
    book.update(z, idx, valid)
    after = (book.read(), book.share.read(), book.total.read())
    return before, after
  state = nj.init(fn)({}, z, idx, valid, seed=0)
  _, (before, after) = nj.pure(fn)(state, z, idx, valid, seed=1)
  return before, after


def test_empty_batch_leaves_the_book_unchanged():
  z = f32(np.random.default_rng(0).normal(size=(6, 3)))
  z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
  before, after = _book_after(z, jnp.zeros(6, jnp.int32), jnp.zeros(6, f32))
  for x, y in zip(before, after):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert np.isfinite(np.asarray(y)).all()


def test_dead_codes_revive_only_onto_valid_changes():
  rng = np.random.default_rng(1)
  z = f32(rng.normal(size=(8, 3)))
  z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
  valid = f32([0, 0, 1, 0, 0, 0, 0, 0])       # only change 2 is valid
  # A threshold above every code's share after one update: all four are
  # revived, and each must land on the one valid change.
  _, (emb, share, _) = _book_after(
      z, jnp.zeros(8, jnp.int32), valid, dead=0.3)
  np.testing.assert_allclose(np.asarray(share), 0.25)
  for c in range(4):
    np.testing.assert_allclose(np.asarray(emb)[c], np.asarray(z[2]), atol=1e-6)


# --- B06: a goal set before the replay window restarts consistently ------------
def test_goal_start_inside_the_window():
  tpos = jnp.arange(10, 14)[None, :]
  src, phase = agentlib.goal_start(tpos, jnp.array([[3, 4, 5, 6]]))
  assert src.tolist() == [[7, 7, 7, 7]]
  assert phase.tolist() == [[3, 4, 5, 6]]


def test_goal_set_before_the_window_restarts_at_its_first_step():
  tpos = jnp.arange(0, 4)[None, :]
  src, phase = agentlib.goal_start(tpos, jnp.array([[10, 11, 12, 13]]))
  assert src.tolist() == [[0, 0, 0, 0]]
  assert phase.tolist() == [[0, 1, 2, 3]]         # counted from that state


# --- B09: greedy acting takes the argmax ---------------------------------------
class _Greedy:
  _greedy = True


def test_greedy_choose_is_the_argmax_whatever_the_seed():
  dist = jouts.Categorical(jnp.array([[0.1, 2.0, -1.0, 1.9]]))
  assert int(agentlib.Agent._choose(_Greedy(), dist)[0]) == 1
