"""The potential shaping on the market branch (env.craftax.survival potential).

The env sends the shaping as its own key, ``shape``, and leaves ``reward``
(the game reward plus the quota) untouched, so the market's score Q-head can
learn the score alone. Needs Craftax.
"""

import numpy as np
import pytest

from dreamerv3 import craftax_potential as P


def _run(env, steps, seed=0):
  rng = np.random.default_rng(seed)
  out = [env.step({'action': np.int32(0), 'reset': np.ones((), bool)})]
  for _ in range(steps):
    act = np.int32(rng.integers(0, env.act_space['action'].high))
    out.append(env.step({'action': act, 'reset': np.zeros((), bool)}))
  return out


def test_shaping_is_its_own_key_and_leaves_the_reward_alone():
  pytest.importorskip('craftax')
  from embodied.envs.craftax import Craftax
  quota = dict(enabled=True, k=1.0, margin=2.0, lnom=50, init=7.0, rate=0.05,
               min_obs=5)
  plain = Craftax(seed=3, quota=quota)
  shaped = Craftax(seed=3, quota=quota, survival='potential')
  assert 'shape' not in plain.obs_space
  assert shaped.obs_space['shape'].dtype == np.float32
  a, b = _run(plain, 400), _run(shaped, 400)
  for x, y in zip(a, b):
    assert set(y) == set(x) | {'shape'}
    assert x['reward'] == y['reward']
    np.testing.assert_array_equal(x['vector'], y['vector'])
    assert y['shape'].dtype == np.float32
  assert b[0]['shape'] == 0.0                    # the reset step pays nothing
  assert any(y['shape'] != 0.0 for y in b)


def test_shaping_is_gamma_phi_next_minus_phi_and_hands_phi_back_at_death():
  pytest.importorskip('craftax')
  import jax
  from embodied.envs.craftax import Craftax
  env = Craftax(seed=5, survival='potential')
  _run(env, 1, seed=1)
  rng = np.random.default_rng(2)
  for _ in range(3000):
    with jax.transfer_guard('allow'):
      before = P.potential(env._state, env._ach_names, env._phi_scale)
    act = np.int32(rng.integers(0, env.act_space['action'].high))
    obs = env.step({'action': act, 'reset': np.zeros((), bool)})
    if obs['is_first']:
      continue                                   # a reset step, no shaping
    with jax.transfer_guard('allow'):
      after = P.potential(env._state, env._ach_names, env._phi_scale)
    want = -before if obs['is_terminal'] else env._phi_gamma * after - before
    assert obs['shape'] == pytest.approx(want, abs=1e-5)


def test_reset_to_carries_the_shape_key():
  pytest.importorskip('craftax')
  from embodied.envs.craftax import Craftax
  env = Craftax(seed=3, survival='potential')
  _run(env, 20)
  obs = env.reset_to(env.save_state())
  assert set(obs) == set(env.obs_space)
  assert obs['shape'] == 0.0
