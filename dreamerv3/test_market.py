"""The market agent's valuation (agent.py: _alive, _profit, _q_loss, _price).

The methods are called on a stand-in for the agent with fixed return scales,
so every property is exact. Needs JAX (they run on the box).

Run: python -m pytest dreamerv3/test_market.py
"""

import functools
import types

import pytest

jax = pytest.importorskip('jax')
pytest.importorskip('ninjax')
pytest.importorskip('chex')
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from dreamerv3 import agent as agentlib  # noqa: E402

f32 = jnp.float32
H = 333


class _Norm:
  """A normaliser with a fixed scale and no update."""

  def __init__(self, scale=1.0):
    self.scale = scale

  def stats(self):
    return f32(0.0), f32(self.scale)

  def __call__(self, x, update):
    return self.stats()


def _agent(contdisc=True, lam=0.95, sscale=1.0, lscale=1.0, rscale=1.0):
  a = types.SimpleNamespace(
      config=types.SimpleNamespace(
          contdisc=contdisc, horizon=H,
          imag_loss=types.SimpleNamespace(lam=lam)),
      qsnorm=_Norm(sscale), qlnorm=_Norm(lscale), retnorm=_Norm(rscale),
      _price0=0.25)
  a._profit = functools.partial(agentlib.Agent._profit, a)
  return a


def _profit(a, qs, ql, price):
  return np.asarray(agentlib.Agent._profit(
      a, f32(qs), f32(ql), f32(price)), np.float32)


# --- economic profit ---------------------------------------------------------
def test_profit_is_value_minus_the_best_other_action():
  # The design's example: Q = [10, 2, 1] gives [+8, -8, -9].
  a = _agent()
  got = _profit(a, [[10., 2., 1.]], [[0., 0., 0.]], [1.0])
  np.testing.assert_allclose(got, [[8., -8., -9.]], atol=1e-2)


def test_profit_prices_survival_at_p():
  a = _agent()
  got = _profit(a, [[0., 0., 0.]], [[1., 2., 3.]], [0.5])
  np.testing.assert_allclose(got, [[-1., -0.5, 0.5]], atol=1e-2)


def test_profit_of_tied_best_actions_is_zero():
  a = _agent()
  got = _profit(a, [[3., 3., 1.]], [[0., 0., 0.]], [1.0])
  np.testing.assert_allclose(got, [[0., 0., -2.]], atol=1e-2)


def test_profit_is_in_units_of_the_return_scales():
  a = _agent(sscale=4.0, lscale=2.0, rscale=2.0)
  # Q = 4 * [1, 0] + 0.5 * 2 * [0, 1] = [4, 1]; profit [3, -3] / 2.
  got = _profit(a, [[1., 0.]], [[0., 1.]], [0.5])
  np.testing.assert_allclose(got, [[1.5, -1.5]], atol=1e-2)


def test_profit_takes_leading_dims_and_stops_gradients():
  a = _agent()
  rng = np.random.default_rng(0)
  qs = f32(rng.normal(size=(2, 3, 5, 7)))       # (N, I, G, actions)
  ql = f32(rng.normal(size=(2, 3, 5, 7)))
  price = f32(rng.uniform(size=(2, 3)))         # (N, I)
  out = agentlib.Agent._profit(a, qs, ql, price)
  assert out.shape == qs.shape
  best = np.asarray(out).max(-1)
  assert (best >= 0).all()
  g = jax.grad(lambda x: agentlib.Agent._profit(a, x, ql, price).sum())(qs)
  assert float(jnp.abs(g).max()) == 0.0


# --- survival reward -----------------------------------------------------------
def test_alive_is_one_while_the_agent_continues_and_zero_at_death():
  a = _agent(contdisc=True)
  con = f32([[1 - 1 / H, 0.0, 0.5 * (1 - 1 / H)]])
  got = np.asarray(agentlib.Agent._alive(a, con))
  np.testing.assert_allclose(got, [[1.0, 0.0, 0.5]], atol=1e-6)
  a = _agent(contdisc=False)
  got = np.asarray(agentlib.Agent._alive(a, f32([[1.0, 0.0]])))
  np.testing.assert_allclose(got, [[1.0, 0.0]], atol=1e-6)


def test_fixed_price_is_price0_everywhere():
  a = _agent()
  obs = {'is_first': jnp.zeros((2, 5), bool)}
  got = np.asarray(agentlib.Agent._price(a, obs))
  assert got.shape == (2, 5) and (got == 0.25).all()


# --- the Q loss ----------------------------------------------------------------
def _qloss(a, qs, ql, act, rew, alive, con):
  price = jnp.full((qs.shape[0],), 0.25, f32)
  return agentlib.Agent._q_loss(
      a, qs, ql, {'action': jnp.asarray(act, jnp.int32)}, rew, alive, con,
      price, True)


def test_q_loss_targets_the_return_of_the_action_taken():
  # contdisc (no discount), lam 0, no death: ret_t = r_{t+1} + Q(s_{t+1}, a_{t+1}). Both heads predict
  # 0.5 for the taken action 0, so the score error is (r - 0)^2 with r = 2 and
  # the life error 1^2: 5 on every step.
  a = _agent(lam=0.0)
  N, T = 2, 4
  qs = jnp.zeros((N, T, 3)).at[..., 0].set(0.5)
  ql = jnp.zeros((N, T, 3)).at[..., 0].set(0.5)
  loss, mets = _qloss(a, qs, ql, np.zeros((N, T)), jnp.full((N, T), 2.0),
                      jnp.ones((N, T)), jnp.ones((N, T)))
  assert loss.shape == (N, T - 1)
  np.testing.assert_allclose(np.asarray(loss), 5.0, atol=1e-5)
  assert {'qscore', 'qlife', 'price', 'alive', 'profit_best'} <= set(mets)


def test_q_loss_only_touches_the_action_taken():
  a = _agent(lam=0.0)
  N, T = 1, 3
  act = np.array([[1, 2, 0]])
  qs = jnp.zeros((N, T, 3))
  ql = jnp.zeros((N, T, 3))
  grad = jax.grad(lambda q: _qloss(a, q, ql, act, jnp.ones((N, T)),
                                   jnp.ones((N, T)), jnp.ones((N, T)))[0].sum())
  g = np.asarray(grad(qs))
  # Steps 0 and 1 train their taken action; the last state is only a
  # bootstrap (stop-gradient), and the other actions get nothing.
  assert g[0, 0, 1] != 0 and g[0, 1, 2] != 0
  assert g[0, 0, 0] == 0 and g[0, 0, 2] == 0 and g[0, 1, 0] == 0
  assert (g[0, 2] == 0).all()


def test_q_loss_weights_a_dream_by_its_survival():
  # A rollout that has already ended (con 0 from step 1) trains nothing after.
  a = _agent(lam=0.0)
  N, T = 1, 4
  con = f32([[1.0, 0.0, 0.0, 0.0]])
  qs = ql = jnp.zeros((N, T, 2))
  loss, _ = _qloss(a, qs, ql, np.zeros((N, T)), jnp.ones((N, T)),
                   jnp.ones((N, T)), con)
  loss = np.asarray(loss)
  assert loss[0, 0] > 0 and (loss[0, 1:] == 0).all()
