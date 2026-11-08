"""Tests for the slope quota (quota.py). Pure numpy: they run without JAX.

Run: python -m pytest dreamerv3/test_quota.py
"""

import math

import numpy as np
import pytest

from dreamerv3 import quota as Q


def _episode(q, gaps, tail=0, died=False):
  """Achievements after the given gaps, then `tail` quiet steps; returns the
  per-step rewards."""
  q.begin()
  out = []
  for g in gaps:
    for i in range(g):
      out.append(q.step(int(i == g - 1)))
  for i in range(tail):
    end = i == tail - 1
    out.append(q.step(0, terminal=end and died, last=end))
  return out


def test_window_before_statistics_is_init():
  q = Q.Quota(init=50, margin=2.0, min_obs=3)
  assert q.window(0) == 50 and q.window(7) == 50
  assert q.mean_gap(0) is None


def test_penalty_every_window_without_an_achievement():
  q = Q.Quota(k=1.0, init=10, lnom=1e9)
  q.begin()
  rews = [q.step(0) for _ in range(35)]
  # Windows close at steps 10, 20, 30.
  hits = [i + 1 for i, r in enumerate(rews) if r != 0]
  assert hits == [10, 20, 30]
  assert all(r == -1.0 for r in rews if r != 0)


def test_achievement_restarts_the_window_and_is_never_penalised():
  q = Q.Quota(k=1.0, init=10, lnom=1e9)
  q.begin()
  rews = [q.step(0) for _ in range(9)]
  rews.append(q.step(1))                      # step 10: an achievement, no pen
  rews += [q.step(0) for _ in range(9)]       # 9 more quiet: still inside
  assert all(r == 0 for r in rews)
  assert q.step(0) == -1.0                    # 10th quiet step after it


def test_window_follows_the_running_gap_per_count():
  q = Q.Quota(margin=2.0, init=1000, min_obs=3, rate=0.5, lnom=1e9)
  for _ in range(3):
    _episode(q, [10, 40], tail=1)
  assert q.mean_gap(0) == pytest.approx(10)
  assert q.mean_gap(1) == pytest.approx(40)
  assert q.window(0) == pytest.approx(20)
  assert q.window(1) == pytest.approx(80)
  # Count 2 has no statistics: the count below it, floored at init.
  assert q.window(2) == pytest.approx(1000)
  q2 = Q.Quota(margin=2.0, init=5, min_obs=3, rate=0.5, lnom=1e9)
  for _ in range(3):
    _episode(q2, [10, 40], tail=1)
  assert q2.window(2) == pytest.approx(80)


def test_running_mean_is_debiased_and_tracks_change():
  q = Q.Quota(min_obs=1, rate=0.1, lnom=1e9)
  _episode(q, [30], tail=1)
  assert q.mean_gap(0) == pytest.approx(30)   # one sample: exactly it
  for _ in range(100):
    _episode(q, [10], tail=1)
  assert q.mean_gap(0) == pytest.approx(10, abs=1e-3)


def test_too_few_observations_fall_back():
  q = Q.Quota(init=77, min_obs=4, rate=0.5, lnom=1e9)
  for _ in range(3):
    _episode(q, [10], tail=1)
  assert q.mean_gap(0) is None and q.window(0) == 77
  _episode(q, [10], tail=1)
  assert q.window(0) == pytest.approx(20)


def test_simultaneous_achievements_record_one_gap():
  q = Q.Quota(min_obs=1, lnom=1e9)
  q.begin()
  for _ in range(4):
    q.step(0)
  q.step(3)
  assert q.count == 3
  assert q.mean_gap(0) == pytest.approx(5)
  assert q.mean_gap(1) is None and q.mean_gap(2) is None


def test_death_charges_the_remaining_windows():
  q = Q.Quota(k=0.5, init=10, lnom=100)
  q.begin()
  rews = [q.step(0) for _ in range(24)]
  last = q.step(0, terminal=True, last=True)      # dies at t = 25
  assert sum(rews) == -1.0                        # windows at 10 and 20
  assert last == pytest.approx(-0.5 * math.ceil((100 - 25) / 10))


def test_dying_never_saves_penalty():
  """Over the same stretch up to L_nom, dying early costs at least as much as
  living on quietly."""
  for t_die in (1, 7, 33, 64, 99):
    q = Q.Quota(k=1.0, init=10, lnom=100)
    q.begin()
    died = sum(q.step(0, terminal=(i == t_die - 1), last=(i == t_die - 1))
               for i in range(t_die))
    q = Q.Quota(k=1.0, init=10, lnom=100)
    q.begin()
    lived = sum(q.step(0) for _ in range(100))
    assert died <= lived, (t_die, died, lived)


def test_no_death_charge_past_l_nom_or_on_timeout():
  q = Q.Quota(k=1.0, init=1000, lnom=10)
  q.begin()
  for _ in range(19):
    q.step(0)
  assert q.step(0, terminal=True, last=True) == 0.0
  q = Q.Quota(k=1.0, init=1000, lnom=100)
  q.begin()
  assert q.step(0, terminal=False, last=True) == 0.0   # timeout: no charge


def test_lnom_mean_is_the_running_lifespan():
  q = Q.Quota(k=1.0, init=10, lnom='mean', rate=0.5)
  assert q.nominal_life() is None
  q.begin()
  for _ in range(4):
    q.step(0)
  assert q.step(0, terminal=True, last=True) == 0.0   # first death: no L_nom
  assert q.nominal_life() == pytest.approx(5)
  _episode(q, [], tail=55)
  assert q.nominal_life() == pytest.approx((0.5 * 0.5 * 5 + 0.5 * 55) / 0.75)
  # A death at t = 3 now charges up to the running mean lifespan.
  q.begin()
  q.step(0)
  q.step(0)
  life = q.nominal_life()
  assert q.step(0, terminal=True, last=True) == pytest.approx(
      -math.ceil((life - 3) / 10))


def test_lnom_given_as_a_string_number():
  assert Q.Quota(lnom='3000').nominal_life() == 3000.0


def test_quota_never_pays_a_positive_reward():
  rng = np.random.default_rng(0)
  q = Q.Quota(k=1.0, init=20, rate=0.2, min_obs=2)
  for _ in range(30):
    q.begin()
    n = int(rng.integers(1, 300))
    for i in range(n):
      r = q.step(int(rng.random() < 0.02), terminal=(i == n - 1),
                 last=(i == n - 1))
      assert r <= 0.0


# --- in the env (needs Craftax) -----------------------------------------------
def test_env_adds_the_quota_to_the_reward_only():
  pytest.importorskip('craftax')
  from embodied.envs.craftax import Craftax
  cfg = dict(enabled=True, k=1.0, margin=2.0, lnom=50, init=7.0, rate=0.05,
             min_obs=5)
  env, ref = Craftax(seed=3, quota=cfg), Q.Quota(
      **{k: v for k, v in cfg.items() if k != 'enabled'})
  reset = {'action': np.int32(0), 'reset': np.ones((), bool)}
  obs = env.step(reset)
  ref.begin()
  assert obs['reward'] == 0.0
  pens = 0
  rng = np.random.default_rng(0)
  for _ in range(300):
    obs = env.step({'action': np.int32(rng.integers(0, 43)),
                    'reset': np.zeros((), bool)})
    want = ref.step(int(obs['ach'].sum()), terminal=obs['is_terminal'],
                    last=obs['is_last'])
    assert float(obs['reward']) == pytest.approx(env.raw_reward + want,
                                                 abs=1e-5)
    pens += want < 0
    if obs['is_last']:
      obs = env.step(reset)
      ref.begin()
  assert pens > 0


def test_env_without_the_quota_pays_the_raw_reward():
  pytest.importorskip('craftax')
  from embodied.envs.craftax import Craftax
  env = Craftax(seed=3, quota=dict(enabled=False, k=1.0))
  assert env._quota is None
  env.step({'action': np.int32(0), 'reset': np.ones((), bool)})
  for a in (5, 5, 1, 5):
    obs = env.step({'action': np.int32(a), 'reset': np.zeros((), bool)})
    assert float(obs['reward']) == env.raw_reward
