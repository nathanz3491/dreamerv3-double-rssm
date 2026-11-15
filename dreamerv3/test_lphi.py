"""The learned potential (agent.py: readiness_labels, potential_shaping).

Needs JAX (runs on the box).

Run: python -m pytest dreamerv3/test_lphi.py
"""

import pathlib

import numpy as np
import pytest

jax = pytest.importorskip('jax')
pytest.importorskip('ninjax')
pytest.importorskip('chex')
import jax.numpy as jnp  # noqa: E402

from dreamerv3 import agent as agentlib  # noqa: E402

f32 = jnp.float32


def _labels(T=10, A=3, unlocks=(), firsts=(0,), horizon=3):
  ach = np.zeros((1, T, A), np.float32)
  for t, k in unlocks:
    ach[0, t, k] = 1.0
  first = np.zeros((1, T), bool)
  first[0, list(firsts)] = True
  y, ok = agentlib.readiness_labels(jnp.asarray(ach), jnp.asarray(first),
                                    horizon)
  return np.asarray(y)[0], np.asarray(ok)[0]


def test_label_marks_the_steps_before_an_unlock():
  y, ok = _labels(unlocks=[(5, 1)])
  # Achievement 1 unlocks at step 5: steps 2, 3, 4 see it within 3 steps.
  np.testing.assert_array_equal(y[:, 1], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
  assert (y[:, [0, 2]] == 0).all()
  # The unlock step itself does not predict its own unlock.
  assert y[5, 1] == 0


def test_windows_off_the_end_are_masked():
  _, ok = _labels(horizon=3)
  np.testing.assert_array_equal(ok, [1] * 7 + [0] * 3)


def test_unlocks_in_the_next_episode_do_not_count():
  # A new episode starts at 4; an unlock at 5 belongs to it.
  y, _ = _labels(unlocks=[(5, 0)], firsts=(0, 4))
  np.testing.assert_array_equal(y[:, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])


def test_horizon_is_clamped_to_the_sequence():
  y, ok = _labels(T=4, unlocks=[(3, 2)], horizon=32)
  assert ok.tolist() == [1, 0, 0, 0]
  assert y[0, 2] == 1


def _shape(phi, live=1.0, gamma=0.9, first=None):
  phi = jnp.asarray(np.asarray(phi, np.float32)[None])
  if first is not None:
    first = jnp.asarray(np.asarray(first, bool)[None])
  live = jnp.asarray(np.asarray(live, np.float32))
  return np.asarray(agentlib.potential_shaping(phi, live, gamma, first))[0]


def test_shaping_is_gamma_phi_next_minus_phi():
  out = _shape([0.0, 0.5, 1.0])
  np.testing.assert_allclose(out, [0.0, 0.45, 0.9 * 1.0 - 0.5], atol=1e-6)


def test_a_death_hands_phi_back():
  out = _shape([0.5, 0.8], live=np.array([[1.0, 0.0]]))
  np.testing.assert_allclose(out, [0.0, -0.5], atol=1e-6)


def test_no_shaping_across_an_episode_start():
  out = _shape([0.5, 0.2, 0.7], first=[True, True, False])
  np.testing.assert_allclose(out, [0.0, 0.0, 0.9 * 0.7 - 0.2], atol=1e-6)


def test_shaping_telescopes_with_gamma_one():
  phi = np.random.default_rng(0).random(12).astype(np.float32)
  out = _shape(phi, gamma=1.0)
  assert out.sum() == pytest.approx(phi[-1] - phi[0], abs=1e-5)


def test_lphi_is_off_by_default_and_has_a_loss_scale():
  yaml = pytest.importorskip('yaml')
  path = pathlib.Path(__file__).with_name('configs.yaml')
  cfg = yaml.safe_load(path.read_text(encoding='utf-8'))['defaults']['agent']
  assert cfg['lphi']['enabled'] is False
  assert cfg['lphi']['ceil'] == 0.25
  assert 'rdy' in cfg['loss_scales']
  # An untrained head (logit 0) starts each achievement near 1%.
  p0 = 1.0 / (1.0 + np.exp(-cfg['lphi']['bias0']))
  assert 0.005 < p0 < 0.02
