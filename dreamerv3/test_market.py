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
      _price0=0.25, _shift=False)
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


# --- the moving price ------------------------------------------------------------
from dreamerv3 import craftax_map as M  # noqa: E402
from dreamerv3 import market as K  # noqa: E402


class _Base:
  """The REINFORCE baseline, fixed, remembering what it was updated with."""

  def __init__(self, b=0.0):
    self.b, self.seen = b, []

  def stats(self):
    return f32(self.b), f32(1.0)

  def update(self, x):
    self.seen.append(float(x))


def _market(mu=(0.0, 0.0), base=0.0, **kw):
  cfg = dict(k=8, sigma=0.3, bound=2.3, dead=0.05, price0=0.01, eps_s=1.0,
             eps_d=1.0, a_s=1.0, a_d=0.0, window=4)
  cfg.update(kw)
  mu = f32(mu)
  a = types.SimpleNamespace(
      _mk=types.SimpleNamespace(**cfg), _greedy=True, _shift=True,
      mkt=lambda x, bdims: types.SimpleNamespace(
          pred=lambda: jnp.broadcast_to(mu, (*x.shape[:-1], 2))),
      mbase=_Base(base), _price0=cfg['price0'])
  for name in ('_market_mean', '_market_price'):
    setattr(a, name, functools.partial(getattr(agentlib.Agent, name), a))
  return a


def _carry(k=8, B=1):
  return {'mhist': jnp.zeros((B, k, M.OBS_LEN), f32),
          'mz': jnp.zeros((B, 2), f32), 'mleft': jnp.zeros((B,), jnp.int32)}


def test_market_compares_with_the_observation_k_steps_ago():
  a = _market()
  rng = np.random.default_rng(0)
  T = 14
  vecs = rng.random((T, 1, M.OBS_LEN)).astype(np.float32)
  firsts = np.zeros(T, bool)
  firsts[0] = firsts[10] = True
  carry, start = _carry(), 0
  for t in range(T):
    if firsts[t]:
      start = t
    carry, out = agentlib.Agent._market_act(
        a, carry, {'vector': jnp.asarray(vecs[t])}, jnp.array([firsts[t]]))
    want = K.delta_env(vecs[t], vecs[max(start, t - 8)])
    np.testing.assert_allclose(np.asarray(out['mfeat']), want, atol=1e-5)


def test_market_price_is_the_baseline_when_nothing_changes():
  a = _market()
  v = jnp.asarray(np.random.default_rng(1).random((1, M.OBS_LEN)), f32)
  carry, out = agentlib.Agent._market_act(a, _carry(), {'vector': v},
                                          jnp.array([True]))
  np.testing.assert_allclose(np.asarray(out['pstar']), 0.01, rtol=1e-5)
  carry, out = agentlib.Agent._market_act(a, carry, {'vector': v},
                                          jnp.array([False]))
  np.testing.assert_allclose(np.asarray(out['pstar']), 0.01, rtol=1e-5)


def test_market_price_follows_the_draw():
  # Greedy: the draw is the net's mean. Demand shifted up raises the price.
  a = _market(mu=(0.0, 0.5))
  v = jnp.zeros((1, M.OBS_LEN), f32)
  _, out = agentlib.Agent._market_act(a, _carry(), {'vector': v},
                                      jnp.array([True]))
  sh = 2.3 * np.tanh(0.45)
  want = 0.01 * np.exp(sh / 2)
  np.testing.assert_allclose(np.asarray(out['pstar']), want, rtol=1e-4)
  np.testing.assert_allclose(np.asarray(out['mshift']), [[0.0, 0.5]])


def test_market_history_is_f32_and_keeps_its_shape():
  a = _market()
  v = jnp.ones((2, M.OBS_LEN), f32)
  carry, _ = agentlib.Agent._market_act(
      a, _carry(B=2), {'vector': v}, jnp.array([True, False]))
  assert carry['mhist'].shape == (2, 8, M.OBS_LEN)
  assert carry['mhist'].dtype == jnp.float32
  assert float(carry['mhist'][:, -1].min()) == 1.0


def _run_market(hold, firsts, T=10, seed=2):
  """Act T steps with a greedy net whose mean is the first two change
  features, so every step would draw something different. -> (z, mdraw)."""
  a = _market(hold=hold)
  a.mkt = lambda x, bdims: types.SimpleNamespace(pred=lambda: x[..., :2])
  vecs = np.random.default_rng(seed).random((T, 1, M.OBS_LEN)).astype(
      np.float32)
  carry, zs, draws = _carry(), [], []
  for t in range(T):
    carry, out = agentlib.Agent._market_act(
        a, carry, {'vector': jnp.asarray(vecs[t])}, jnp.array([t in firsts]))
    zs.append(np.asarray(out['mshift'])[0])
    draws.append(float(out['mdraw'][0]))
  return np.array(zs), np.array(draws)


def test_hold_one_draws_every_step():
  zs, draws = _run_market(hold=1, firsts=(0,))
  assert (draws == 1).all()
  assert (np.abs(np.diff(zs, axis=0)).sum(-1) > 0).all()


def test_a_draw_is_held_and_a_new_episode_draws_afresh():
  zs, draws = _run_market(hold=3, firsts=(0, 5))
  # Draws at 0 and 3; the episode starting at 5 draws (cutting the hold
  # short) and restarts the count, so the next draw is at 8.
  np.testing.assert_array_equal(draws, [1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
  for start, end in ((0, 3), (3, 5), (5, 8), (8, 10)):
    np.testing.assert_allclose(zs[start:end], zs[start:start + 1].repeat(
        end - start, 0))
  assert np.abs(zs[3] - zs[0]).sum() > 0


def test_market_loss_scores_only_the_drawing_steps():
  a = _market(window=3)
  obs = _loss_batch(ach_at=(2,), draws=(0, 3))
  loss, mets = agentlib.Agent._market_loss(a, obs, False)
  loss = np.asarray(loss)[0]
  # Both drawing steps have a full window, so both pull; held steps do not.
  assert (loss[[0, 3]] != 0).all() and (np.delete(loss, [0, 3]) == 0).all()
  # t=0's window holds the achievement (weight e), t=3's does not (weight 1).
  assert loss[0] / loss[3] == pytest.approx(np.e, rel=1e-4)
  assert float(mets['draw_frac']) == pytest.approx(2 / 8)
  assert float(mets['valid_frac']) == pytest.approx(2 / 8)


def _loss_batch(T=8, ach_at=(), firsts=(0,), z=(0.4, -0.2), draws=None):
  ach = np.zeros((1, T, 22), np.float32)
  for t in ach_at:
    ach[0, t, 0] = 1.0
  first = np.zeros((1, T), bool)
  first[0, list(firsts)] = True
  mdraw = np.ones((1, T), np.float32)
  if draws is not None:
    mdraw[:] = 0.0
    mdraw[0, list(draws)] = 1.0
  return {'mshift': jnp.broadcast_to(f32(z), (1, T, 2)),
          'mfeat': jnp.zeros((1, T, K.N_FEAT), f32),
          'ach': jnp.asarray(ach), 'is_first': jnp.asarray(first),
          'mdraw': jnp.asarray(mdraw)}


def test_market_loss_rewards_draws_followed_by_achievements():
  obs = _loss_batch(ach_at=(3,))          # windows (0,3], (1,4], (2,5] hold it
  def total(mu):
    a2 = _market(mu=mu, window=3, sigma=0.5)
    loss, _ = agentlib.Agent._market_loss(a2, obs, False)
    return loss.sum(), loss
  (val, loss), grad = jax.value_and_grad(total, has_aux=True)(f32([0.0, 0.0]))
  loss = np.asarray(loss)[0]
  # Every step with a full window pulls; the steps after t = 5 have none.
  assert (loss[:5] != 0).all() and (loss[5:] == 0).all()
  # Windows holding the achievement (t = 0..2) weigh e, the others 1.
  np.testing.assert_allclose(loss[:3] / loss[3], np.e, rtol=1e-4)
  # Descent moves the mean toward the draw (0.4, -0.2), so d loss / d mu
  # points away from it.
  g = np.asarray(grad)
  assert g[0] < 0 and g[1] > 0


def test_market_loss_weighs_draws_below_the_baseline_less_but_never_pushes():
  obs = _loss_batch(ach_at=())            # nothing follows; baseline expects 1
  a = _market(base=1.0, window=3, sigma=0.5)
  loss, mets = agentlib.Agent._market_loss(a, obs, True)
  assert (np.asarray(loss)[0, :5] > 0).all()
  assert float(mets['weight']) == pytest.approx(np.exp(-1.0), rel=1e-4)
  assert a.mbase.seen == [0.0]            # the baseline tracks the mean return
  assert float(mets['valid_frac']) == pytest.approx(5 / 8)
  # Even a draw that did worse than expected pulls the mean toward itself.
  def total(mu):
    loss, _ = agentlib.Agent._market_loss(
        _market(mu=mu, base=1.0, window=3, sigma=0.5), obs, False)
    return loss.sum()
  g = np.asarray(jax.grad(total)(f32([0.0, 0.0])))
  assert g[0] < 0 and g[1] > 0


def test_market_weights_are_capped():
  obs = _loss_batch(ach_at=(1, 2, 3))     # 3 achievements in t=0's window
  a = _market(window=3, sigma=0.5, awr_temp=0.1, awr_wmax=20.0)
  _, mets = agentlib.Agent._market_loss(a, obs, False)
  assert float(mets['weight']) <= 20.0 + 1e-4


def test_market_mean_stays_among_stale_draws():
  # The failure that ended market12_full's first run: replayed draws that did
  # worse than the baseline. Gradient descent on them must keep the mean
  # within the draws' range (REINFORCE pushed it off without bound).
  rng = np.random.default_rng(0)
  T = 16
  zs = rng.normal(0.0, 0.3, (1, T, 2)).astype(np.float32)
  obs = {**_loss_batch(T=T, ach_at=(5,)), 'mshift': jnp.asarray(zs)}
  def total(mu):
    loss, _ = agentlib.Agent._market_loss(
        _market(mu=mu, base=2.0, window=3, sigma=0.1), obs, False)
    return loss.sum()
  grad = jax.jit(jax.grad(total))
  mu = f32([1.0, -1.0])
  for _ in range(300):
    mu = mu - 1e-4 * grad(mu)
  lo, hi = zs[0, :T - 3].min(0), zs[0, :T - 3].max(0)
  assert np.isfinite(np.asarray(mu)).all()
  assert (np.asarray(mu) >= lo - 1e-3).all() and (np.asarray(mu) <= hi + 1e-3).all()


def test_market_mean_is_soft_capped():
  a = _market(mu_bound=1.5)
  for raw, want in ((0.01, 0.01), (100.0, 1.5), (-100.0, -1.5)):
    a.mkt = lambda x, bdims, r=raw: types.SimpleNamespace(
        pred=lambda: jnp.full((*x.shape[:-1], 2), r, f32))
    mu = np.asarray(agentlib.Agent._market_mean(a, jnp.zeros((1, K.N_FEAT)), 1))
    np.testing.assert_allclose(mu, want, rtol=1e-3, atol=1e-5)


def test_market_loss_skips_windows_into_the_next_episode():
  a = _market(window=3)
  obs = _loss_batch(ach_at=(1,), firsts=(0, 4))
  loss, mets = agentlib.Agent._market_loss(a, obs, False)
  ok = np.asarray(loss)[0] != 0
  # t = 1..3 would end at 4..6, in the new episode; t = 4, 5 start in it.
  assert not ok[1:4].any()


def test_market_loss_logs_the_price_and_quantity():
  a = _market(window=3)
  _, mets = agentlib.Agent._market_loss(a, _loss_batch(), False)
  sh = K.shifts(np.array([0.4, -0.2]), 2.3, 0.05)
  logp, logq = K.equilibrium(sh[0], sh[1], 0.01, 1.0, 1.0)
  assert float(mets['logp']) == pytest.approx(float(logp), abs=1e-4)
  assert float(mets['logq']) == pytest.approx(float(logq), abs=1e-4)


def test_the_stored_price_is_what_value_reads():
  a = _market()
  obs = {'is_first': jnp.zeros((2, 3), bool),
         'pstar': jnp.full((2, 3), 0.07, f32)}
  got = np.asarray(agentlib.Agent._price(a, obs))
  assert (got == np.float32(0.07)).all()


# --- the consumers read the price -----------------------------------------------------
def _with_price(a, mapfeat, price):
  return agentlib.Agent._with_price(a, mapfeat, f32(price))


def test_log_price_input_is_zero_at_the_baseline_and_appended_last():
  a = _agent()                                   # price0 0.25
  mf = jnp.ones((3, 5), jnp.bfloat16)
  out = _with_price(a, mf, [0.25, 0.5, 0.125])
  assert out.shape == (3, 6)
  np.testing.assert_array_equal(np.asarray(out[:, :5], np.float32), 1.0)
  np.testing.assert_allclose(np.asarray(out[:, 5], np.float32),
                             [0.0, np.log(2.0), -np.log(2.0)], atol=1e-2)


def test_log_price_input_takes_replay_and_rollout_shapes():
  a = _agent()
  # (B, K, D2) with (B, K) prices, as the replay value loss feeds it.
  out = _with_price(a, jnp.zeros((2, 4, 7), jnp.bfloat16), jnp.full((2, 4), 0.25))
  assert out.shape == (2, 4, 8) and float(jnp.abs(out[..., -1]).max()) == 0.0
  # (BK, D2) with (BK,) prices, one per rollout, as the imagination feeds it.
  out = _with_price(a, jnp.zeros((6, 7), jnp.bfloat16), jnp.full((6,), 0.5))
  assert out.shape == (6, 8)


def test_log_price_input_stops_the_gradient():
  a = _agent()
  g = jax.grad(lambda p: _with_price(
      a, jnp.zeros((2, 3), jnp.bfloat16), p).astype(f32).sum())(f32([0.3, 0.6]))
  assert float(jnp.abs(g).max()) == 0.0


def test_a_fixed_price_is_a_constant_input():
  # market without shift: every step reads price0, so the input is exactly 0
  # at acting, in the dream and in replay -- the same architecture as the
  # moving price, with nothing to read.
  a = _agent()
  a._shift = False
  obs = {'is_first': jnp.zeros((2, 5), bool)}
  price = agentlib.Agent._price(a, obs)
  out = _with_price(a, jnp.zeros((2, 5, 3), jnp.bfloat16), price)
  assert float(jnp.abs(out[..., -1]).max()) == 0.0


def test_the_price_input_exists_only_with_the_market():
  # With the market off the helper is never reached: every call is under a
  # market guard in agent.py.
  import inspect
  import re
  src = inspect.getsource(agentlib.Agent)
  lines = src.split('\n')
  for i, line in enumerate(lines):
    if '_with_price(' in line and 'def _with_price' not in line:
      window = '\n'.join(lines[max(0, i - 10):i])
      assert 'if self._market' in window, line
