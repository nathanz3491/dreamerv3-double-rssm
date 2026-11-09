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


# --- quota v2 (LogQuota) -------------------------------------------------------
def _life(q, times, length, died=False):
  """One life of `length` steps with an achievement at each of `times`;
  returns the per-step rewards."""
  q.begin()
  out = []
  for t in range(1, length + 1):
    end = t == length
    out.append(q.step(times.count(t), terminal=end and died, last=end))
  return out


def _curve_times(a, c, kmax):
  """The step at which a life following k = a ln(1 + t/c) unlocks its k-th."""
  return [int(round(c * (math.exp(k / a) - 1))) for k in range(1, kmax + 1)]


def _fitted(q, a=2.0, c=10.0, kmax=5, n=None):
  times = _curve_times(a, c, kmax)
  for _ in range(n or q.min_lives):
    _life(q, times, times[-1] + 1)
  return times


def test_log_fit_recovers_a_known_curve():
  q = Q.LogQuota(min_lives=5, fit_lives=50)
  rng = np.random.default_rng(0)
  for a, c in [(2.24, 11.5), (1.5, 40.0), (3.0, 3.0)]:
    q = Q.LogQuota(min_lives=5, fit_lives=50)
    for _ in range(20):
      kmax = int(rng.integers(4, 9))
      times = _curve_times(a, c, kmax)
      _life(q, times, times[-1] + 5)
    assert q.a == pytest.approx(a, rel=0.1), (a, c, q.a, q.c)
    assert q.c == pytest.approx(c, rel=0.2), (a, c, q.a, q.c)
    t = np.array([10, 50, 200])
    want = a * np.log1p(t / c)
    got = [q.floor(x) / q.frac for x in t]
    assert got == pytest.approx(want, rel=0.1)


def test_log_no_floor_and_no_penalty_before_min_lives():
  q = Q.LogQuota(k=1.0, every=10, min_lives=5)
  times = _curve_times(2.0, 10.0, 5)
  for i in range(4):
    assert q.floor(100) is None
    rews = _life(q, times, 400, died=True)       # nothing slips through
    assert all(r == 0 for r in rews), i
  _life(q, times, 400, died=True)
  assert q.floor(100) is not None


def test_log_penalty_when_below_the_floor_and_none_when_ahead():
  q = Q.LogQuota(k=1.0, frac=0.5, every=20, min_lives=3)
  times = _fitted(q, a=2.0, c=10.0, kmax=6)
  # An idle life: below the floor at every check from step 20 on.
  rews = _life(q, [], 100)
  hits = [i + 1 for i, r in enumerate(rews) if r != 0]
  assert hits == [20, 40, 60, 80, 100]
  assert all(r == -1.0 for r in rews if r != 0)
  # A life that tracks the curve it was fitted on is never penalised.
  rews = _life(q, times, times[-1] + 1)
  assert all(r == 0 for r in rews)


def test_log_banking_an_early_burst_then_a_pause_stays_free():
  q = Q.LogQuota(k=1.0, frac=0.5, every=10, min_lives=3, death_cap=1.0)
  _fitted(q, a=2.0, c=10.0, kmax=6)
  floor_at_400 = q.floor(400)
  assert floor_at_400 < 4
  # Four achievements in the first 5 steps, then nothing for 395 steps.
  rews = _life(q, [1, 2, 3, 5], 400)
  assert all(r == 0 for r in rews)
  # The same pause with no burst pays.
  assert sum(_life(q, [], 400)) < 0


def test_log_achievement_on_the_check_step_counts():
  q = Q.LogQuota(k=1.0, frac=1.0, every=20, min_lives=3)
  _fitted(q, a=2.0, c=10.0, kmax=6)
  need = math.floor(q.floor(20)) + 1       # the count that clears F(20)
  rews = _life(q, [20] * need, 20)
  assert rews[-1] == 0.0


def test_log_death_charge_is_capped_and_only_below_the_floor():
  q = Q.LogQuota(k=0.5, frac=1.0, every=1000, death_cap=2.0, min_lives=3)
  _fitted(q, a=2.0, c=10.0, kmax=6)
  assert _life(q, [], 50, died=True)[-1] == pytest.approx(-1.0)  # -k x cap
  # No matter how long the life, the charge never grows past the cap.
  assert _life(q, [], 700, died=True)[-1] == pytest.approx(-1.0)
  # A timeout is not a death.
  assert _life(q, [], 50, died=False)[-1] == 0.0
  # Ahead of the floor at death: free.
  assert _life(q, [1, 2, 3, 4, 5, 6], 50, died=True)[-1] == 0.0


def test_log_memory_is_bounded_and_follows_the_agent():
  q = Q.LogQuota(fit_lives=10, min_lives=5, every=1000)
  _fitted(q, a=2.0, c=10.0, kmax=5, n=10)
  a_old = q.a
  _fitted(q, a=4.0, c=10.0, kmax=5, n=10)       # ten better lives replace it
  assert len(q._lives) == 10
  assert q.a > 1.5 * a_old


def test_log_unfinished_life_is_dropped_and_empty_lives_count():
  q = Q.LogQuota(min_lives=2, fit_lives=5, every=1000)
  q.begin()
  q.step(1)
  q.begin()                                    # no is_last: not recorded
  assert len(q._lives) == 0
  _life(q, [], 10)
  _life(q, [], 10)
  assert q.a == 0.0 and q.floor(100) == 0.0    # lives but no points: no floor
  assert _life(q, [], 100, died=True)[-1] == 0.0


def test_log_never_pays_a_positive_reward():
  rng = np.random.default_rng(1)
  q = Q.LogQuota(k=1.0, every=7, min_lives=3)
  for _ in range(30):
    q.begin()
    n = int(rng.integers(1, 300))
    for i in range(n):
      r = q.step(int(rng.random() < 0.05), terminal=(i == n - 1),
                 last=(i == n - 1))
      assert r <= 0.0


def test_make_picks_the_mode_and_ignores_the_other_modes_keys():
  cfg = dict(k=1.0, margin=2.0, lnom='mean', init=200.0, rate=0.05, min_obs=5,
             frac=0.5, every=100, death_cap=1.0, fit_lives=50, min_lives=5)
  assert type(Q.make(mode='window', **cfg)) is Q.Quota
  assert type(Q.make(**cfg)) is Q.Quota                    # window is default
  v2 = Q.make(mode='log', **cfg)
  assert type(v2) is Q.LogQuota and v2.every == 100 and v2.frac == 0.5
  with pytest.raises(ValueError):
    Q.make(mode='lin', **cfg)


def test_make_window_is_v1_exactly():
  """The reward stream of make(mode='window') equals a plain Quota's."""
  rng = np.random.default_rng(2)
  cfg = dict(k=1.0, margin=2.0, lnom=50, init=7.0, rate=0.05, min_obs=5)
  a, b = Q.make(mode='window', frac=0.9, every=3, **cfg), Q.Quota(**cfg)
  for _ in range(20):
    a.begin()
    b.begin()
    n = int(rng.integers(1, 200))
    for i in range(n):
      x = int(rng.random() < 0.05)
      kw = dict(terminal=(i == n - 1 and rng.random() < 0.5), last=i == n - 1)
      assert a.step(x, **kw) == b.step(x, **kw)


def test_info_logs_the_pieces_without_changing_the_reward():
  q = Q.Quota(k=0.5, init=10, lnom=100)
  q.begin()
  rews = [q.step(0) for _ in range(24)]
  assert q.info['pen'] == 0.0 and q.info['floor'] == 0.0
  r = q.step(0, terminal=True, last=True)
  want = math.ceil((100 - 25) / 10)
  assert q.info['pen'] == 0.0 and q.info['death'] == pytest.approx(0.5 * want)
  assert q.info['floor'] == 10.0 and r == pytest.approx(-0.5 * want)
  assert set(q.info) == set(Q.Quota.LOGS)
  # The windows at 10 and 20 show up as `pen`.
  q.begin()
  seen = [q.step(0) and q.info['pen'] for _ in range(20)]
  assert [i + 1 for i, p in enumerate(seen) if p] == [10, 20]


def test_log_info_has_the_fit_and_the_floor_at_the_end_of_the_life():
  q = Q.LogQuota(k=1.0, frac=0.5, every=50, min_lives=3, death_cap=1.0)
  _fitted(q, a=2.0, c=10.0, kmax=6)
  a, c = q.a, q.c
  rews = _life(q, [], 120, died=True)
  assert q.info['floor'] == pytest.approx(0.5 * a * math.log1p(120 / c))
  assert q.info['death'] == 1.0 and q.info['pen'] == 0.0   # 120 is no check
  assert q.info['a'] == a and q.info['c'] == c             # the fit in force
  assert rews[-1] == -1.0
  # Mid-life steps carry the floor as 0, and the penalty at a check.
  q.begin()
  for _ in range(49):
    q.step(0)
    assert q.info['floor'] == 0.0 and q.info['pen'] == 0.0
  assert q.step(0) == -1.0 and q.info['pen'] == 1.0
  assert set(q.info) == set(Q.LogQuota.LOGS)


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


def test_env_logs_the_quota_metrics_and_does_not_change_the_reward():
  pytest.importorskip('craftax')
  from embodied.envs.craftax import Craftax
  cfg = dict(enabled=True, mode='log', k=1.0, frac=0.5, every=5, death_cap=1.0,
             fit_lives=10, min_lives=1)
  logged = Craftax(seed=3, logs=True, quota=cfg)
  plain = Craftax(seed=3, logs=False, quota=cfg)
  keys = {f'log/quota_{n}' for n in Q.LogQuota.LOGS}
  assert keys <= set(logged.obs_space) and not keys & set(plain.obs_space)
  reset = {'action': np.int32(0), 'reset': np.ones((), bool)}
  rng = np.random.default_rng(0)
  for env in (logged, plain):
    env.step(reset)
  for _ in range(200):
    act = {'action': np.int32(rng.integers(0, 43)), 'reset': np.zeros((), bool)}
    a, b = logged.step(act), plain.step(act)
    assert keys <= set(a) and not keys & set(b)
    assert float(a['reward']) == float(b['reward'])
    assert float(a['log/quota_pen']) + float(a['log/quota_death']) >= 0
    if a['is_last']:
      assert logged.step(reset)['log/quota_pen'] == 0.0
      plain.step(reset)

