"""Looped actor (patch/looped-actor): nets.MLP(loops=n).

Needs JAX (runs on the box).

Run: python -m pytest dreamerv3/test_looped_actor.py
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
nj = pytest.importorskip('ninjax')
import jax.numpy as jnp  # noqa: E402

from embodied.jax import nets  # noqa: E402


def _run(loops, params=None, x=None, gate=None):
  net = nets.MLP(3, 16, loops, name='mlp')
  x = jnp.asarray(np.random.default_rng(0).normal(size=(4, 8)), jnp.float32)\
      if x is None else x
  if params is None:
    params = nj.init(net)({}, x, seed=0)
  if gate is not None:
    params = {**params, 'mlp/loopgate': jnp.full((16,), gate, jnp.float32)}
  _, out = nj.pure(net)(params, x)
  return params, x, out


def test_closed_gate_gives_the_unlooped_network():
  params3, x, out3 = _run(3)
  assert 'mlp/loopgate' in params3
  plain = {k: v for k, v in params3.items() if k != 'mlp/loopgate'}
  _, _, out1 = _run(1, plain, x)
  np.testing.assert_array_equal(np.asarray(out3, np.float32),
                                np.asarray(out1, np.float32))


def test_only_new_parameter_is_the_gate():
  params3, _, _ = _run(3)
  params1, _, _ = _run(1)
  assert set(params3) - set(params1) == {'mlp/loopgate'}
  assert params3['mlp/loopgate'].shape == (16,)


def test_open_gate_changes_the_output():
  params3, x, out3 = _run(3)
  _, _, opened = _run(3, params3, x, gate=0.5)
  assert not np.allclose(np.asarray(out3, np.float32),
                         np.asarray(opened, np.float32))


def test_gate_gets_a_gradient_while_closed():
  params3, x, _ = _run(3)
  net = nets.MLP(3, 16, 3, name='mlp')

  def loss(params):
    _, out = nj.pure(net)(params, x)
    return jnp.square(out.astype(jnp.float32)).sum()

  grad = jax.grad(loss)(params3)['mlp/loopgate']
  assert float(jnp.abs(grad).sum()) > 0


def test_looping_needs_a_middle_layer():
  with pytest.raises(AssertionError):
    nets.MLP(1, 16, 3, name='bad')
