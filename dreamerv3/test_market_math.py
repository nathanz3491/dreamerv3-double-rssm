"""The market's curves, sensing and training return (market.py). Pure numpy.

Run: python -m pytest dreamerv3/test_market_math.py
"""

import numpy as np
import pytest

from dreamerv3 import craftax_map as M
from dreamerv3 import market as K


def _vec(rng, scale=1.0):
  v = rng.random(M.OBS_LEN).astype(np.float32) * scale
  return v


# --- sensing ------------------------------------------------------------------
def test_layout_constants_match_the_observation():
  assert K.VIEW + K.N_NONVIEW == M.OBS_LEN == 8268
  assert K.N_NONVIEW == 51 and K.N_FEAT == 52


def test_no_change_is_no_signal():
  v = _vec(np.random.default_rng(0))
  f = K.delta_env(v, v)
  assert f.shape == (K.N_FEAT,)
  np.testing.assert_allclose(f, 0.0, atol=1e-6)


def test_nonview_is_a_plain_difference_and_view_one_minus_cosine():
  rng = np.random.default_rng(1)
  a, b = _vec(rng), _vec(rng)
  f = K.delta_env(a, b)
  np.testing.assert_allclose(f[:-1], a[K.VIEW:] - b[K.VIEW:], atol=1e-6)
  va, vb = a[:K.VIEW], b[:K.VIEW]
  cos = va @ vb / np.linalg.norm(va) / np.linalg.norm(vb)
  assert f[-1] == pytest.approx(1 - cos, abs=1e-5)


def test_view_feature_ignores_the_scale_of_the_view_and_the_nonview_part():
  rng = np.random.default_rng(2)
  a, b = _vec(rng), _vec(rng)
  c = b.copy()
  c[:K.VIEW] *= 3.0          # the same view, brighter
  c[K.VIEW:] += 5.0          # a different scalar block
  assert K.delta_env(a, c)[-1] == pytest.approx(K.delta_env(a, b)[-1], abs=1e-5)


def test_empty_views():
  z = np.zeros(M.OBS_LEN, np.float32)
  v = np.ones(M.OBS_LEN, np.float32)
  assert K.delta_env(z, z)[-1] == 0.0       # the same (dark) view
  assert K.delta_env(z, v)[-1] == 1.0       # nothing shared


def test_delta_env_takes_batches():
  rng = np.random.default_rng(3)
  a = np.stack([_vec(rng) for _ in range(4)])
  b = np.stack([_vec(rng) for _ in range(4)])
  f = K.delta_env(a, b)
  assert f.shape == (4, K.N_FEAT)
  np.testing.assert_allclose(f[2], K.delta_env(a[2], b[2]), atol=1e-6)


# --- shifts and curves ------------------------------------------------------------
def test_dead_zone_and_bound():
  z = np.array([[0.0, 0.04], [-0.05, 0.05]])
  np.testing.assert_array_equal(K.shifts(z, 2.3, 0.05), 0.0)
  big = K.shifts(np.array([100.0, -100.0]), 2.3, 0.05)
  np.testing.assert_allclose(big, [2.3, -2.3], atol=1e-6)
  assert K.shifts(np.array([0.5, 0.5]), 2.3, 0.05)[0] == pytest.approx(
      2.3 * np.tanh(0.45))
  # Odd: a negative draw shifts the other way by the same amount.
  assert K.shifts(np.array([-0.5, 0.0]), 2.3, 0.05)[0] == pytest.approx(
      -2.3 * np.tanh(0.45))


@pytest.mark.parametrize('eps', [(1.0, 1.0), (0.5, 2.0)])
def test_no_shift_prices_at_price0(eps):
  logp, _ = K.equilibrium(0.0, 0.0, 0.01, *eps)
  assert np.exp(logp) == pytest.approx(0.01)


def test_demand_up_or_supply_down_raises_the_price():
  base, _ = K.equilibrium(0.0, 0.0, 0.01, 1.0, 1.0)
  up_d, _ = K.equilibrium(0.0, 0.5, 0.01, 1.0, 1.0)
  down_s, _ = K.equilibrium(-0.5, 0.0, 0.01, 1.0, 1.0)
  up_s, _ = K.equilibrium(0.5, 0.0, 0.01, 1.0, 1.0)
  assert up_d > base and down_s > base and up_s < base
  assert up_d - base == pytest.approx(0.25)      # 0.5 / (eps_s + eps_d)


def test_the_curves_cross_at_the_price():
  eps_s, eps_d = 0.7, 1.3
  dls, dld = 0.4, -0.2
  logp, logq = K.equilibrium(dls, dld, 0.05, eps_s, eps_d, a_s=2.0, a_d=3.0)
  log_as, log_ad = np.log(2.0), np.log(3.0)
  assert logq == pytest.approx(log_as + dls + eps_s * logp)
  assert logq == pytest.approx(log_ad + dld - eps_d * logp)


def test_price_stays_inside_the_bounded_range():
  bound = 2.3
  lo, _ = K.equilibrium(bound, -bound, 0.01, 1.0, 1.0)
  hi, _ = K.equilibrium(-bound, bound, 0.01, 1.0, 1.0)
  assert np.exp(lo) == pytest.approx(0.01 * np.exp(-bound))
  assert np.exp(hi) == pytest.approx(0.01 * np.exp(bound))


# --- training ------------------------------------------------------------------------
def test_gaussian_logp_matches_the_density():
  z = np.array([[0.1, -0.2], [0.0, 0.0]])
  mu = np.array([[0.0, 0.0], [0.0, 0.0]])
  got = K.gauss_logp(z, mu, 0.3)
  one = lambda x: -0.5 * (x / 0.3) ** 2 - np.log(0.3) - 0.5 * np.log(2 * np.pi)
  np.testing.assert_allclose(got, [one(0.1) + one(-0.2), 2 * one(0.0)])
  # The score of the mean: d logp / d mu = (z - mu) / sigma^2.
  eps = 1e-5
  g = (K.gauss_logp(z, mu + [eps, 0], 0.3) - K.gauss_logp(z, mu, 0.3)) / eps
  np.testing.assert_allclose(g, z[:, 0] / 0.09, rtol=1e-3, atol=1e-3)


def test_window_returns_count_the_next_window_inside_the_episode():
  ach = np.zeros((1, 10), np.float32)
  ach[0, 3] = 1
  ach[0, 6] = 2
  first = np.zeros((1, 10), bool)
  first[0, 0] = True
  r, ok = K.window_returns(ach, first, 4)
  # t = 0: steps 1..4 hold the one at 3. t = 2: steps 3..6 hold 1 + 2.
  assert r[0, 0] == 1 and r[0, 2] == 3 and r[0, 3] == 2 and r[0, 4] == 2
  assert ok[0].tolist() == [True] * 6 + [False] * 4     # the last 4 lack a window


def test_a_window_that_runs_into_the_next_episode_is_not_scored():
  ach = np.zeros((1, 12), np.float32)
  first = np.zeros((1, 12), bool)
  first[0, [0, 7]] = True
  r, ok = K.window_returns(ach, first, 3)
  # A reset at step 7: the windows ending at step 7 or later that start before
  # it (t = 4, 5, 6) run into the new episode; t = 3 ends at step 6, and t = 7
  # starts inside the new episode.
  assert ok[0, :4].all()
  assert not ok[0, 4:7].any()
  assert ok[0, 7] and ok[0, 8]
