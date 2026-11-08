"""Fixes from the 2026-10-11 audit (round 2) that apply to this branch:
fixed evaluation worlds (B05), the first shaped step (B07), raw reward kept
apart (6.4), greedy acting (B09). The RSSM-2 fixes (B01-B04) are tested in
test_mapmodel_step.py.

Run: python -m pytest dreamerv3/test_audit_fixes.py
The env tests need Craftax installed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3 import agent as agentlib
from embodied.jax import outs as jouts


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


# --- B09: greedy acting takes the argmax ---------------------------------------
class _Greedy:
  _greedy = True


def test_greedy_choose_is_the_argmax_whatever_the_seed():
  dist = jouts.Categorical(jnp.array([[0.1, 2.0, -1.0, 1.9]]))
  assert int(agentlib.Agent._choose(_Greedy(), dist)[0]) == 1
