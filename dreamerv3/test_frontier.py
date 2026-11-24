"""Frontier dreaming: the archive (frontier.py) and the loss weighting.

The archive tests need only NumPy. The weighting test needs JAX (box).

Run: python -m pytest dreamerv3/test_frontier.py
"""

import numpy as np
import pytest

from dreamerv3 import frontier as frlib

OBS, ACT = ('vector',), ('action',)


def _archive(W=8, S=3, **kw):
  kw.setdefault('window', 10)
  kw.setdefault('pmax', 0.25)
  return frlib.FrontierArchive(W, S, OBS, ACT, **kw)


def _step(t, reward=0.0, first=False, last=False, terminal=False):
  return {
      'vector': np.full(2, t, np.float32),
      'action': np.int32(t % 5),
      'reward': np.float32(reward),
      'is_first': first, 'is_last': last, 'is_terminal': terminal,
  }


def _episode(arc, length, unlocks, start=0, worker=0, health=()):
  """Feed one episode; ``unlocks`` are the in-episode steps with reward 1."""
  for i in range(length):
    r = 1.0 if i in unlocks else (0.1 if i in health else 0.0)
    arc.observe(_step(start + i, r, first=(i == 0), last=(i == length - 1)),
                worker)


# --- levels and rarity -------------------------------------------------------

def test_small_rewards_are_not_level_steps():
  arc = _archive()
  _episode(arc, 20, unlocks=(), health=(3, 4, 5))
  assert arc.best == 0 and arc.size == 0 and list(arc.episodes) == [0]


def test_first_reach_of_a_level_is_an_event():
  arc = _archive()
  _episode(arc, 20, unlocks=(5,))
  assert arc.best == 1 and arc.events == 1 and arc.size == 1


def test_reward_on_the_first_step_does_not_count():
  arc = _archive()
  arc.observe(_step(0, 1.0, first=True))
  assert arc.best == 0


def test_reach_is_the_fraction_of_recent_episodes():
  arc = _archive(window=4)
  for unlocks in [(2,), (2, 4), (), ()]:
    _episode(arc, 10, unlocks)
  assert arc.reach(1) == pytest.approx(0.5)
  assert arc.reach(2) == pytest.approx(0.25)
  assert arc.reach(3) == 0.0


def test_mastered_levels_leave_the_frontier_but_the_best_stays():
  arc = _archive(window=4, pmax=0.25)
  for _ in range(4):
    _episode(arc, 10, unlocks=(2,))
  # Level 1 is reached by every episode, but it is also the best level.
  assert arc.is_frontier(1)
  _episode(arc, 10, unlocks=(2, 5))
  assert arc.best == 2
  assert not arc.is_frontier(1)     # reached by 4 of the last 4
  assert arc.is_frontier(2)
  assert arc.frontier_levels() == [2]


# --- windows -----------------------------------------------------------------

def test_window_puts_the_event_at_W_minus_S():
  W, S = 8, 3
  arc = _archive(W, S)
  _episode(arc, 20, unlocks=(10,))
  win = arc.levels[1][0]
  # Event at in-episode step 10 sits at position 5; positions 0..7 hold 5..12.
  np.testing.assert_array_equal(win['vector'][:, 0], np.arange(5, 13))
  assert win['reward'][W - S] == 1.0
  assert win['valid'].tolist() == [1.0] * W
  assert win['is_first'].tolist() == [True] + [False] * (W - 1)


def test_window_waits_for_the_steps_after_the_event():
  arc = _archive(8, 3)
  for i in range(12):
    arc.observe(_step(i, 1.0 if i == 10 else 0.0, first=(i == 0)))
  assert arc.size == 0               # needs steps 11 and 12
  arc.observe(_step(12))
  assert arc.size == 1


def test_episode_end_pads_and_masks_the_rest():
  W, S = 8, 3
  arc = _archive(W, S)
  _episode(arc, 11, unlocks=(10,))   # dies on the unlock step
  win = arc.levels[1][0]
  assert win['valid'].tolist() == [1.0] * (W - S + 1) + [0.0] * (S - 1)
  assert (win['vector'][W - S + 1:] == 0).all()


def test_steps_before_the_episode_start_are_masked():
  W, S = 8, 3
  arc = _archive(W, S)
  _episode(arc, 5, unlocks=(), start=100)          # an earlier episode
  _episode(arc, 10, unlocks=(2,), start=200)
  win = arc.levels[1][0]
  # Event at step 2 -> window covers steps -3..4 of the new episode.
  assert win['valid'].tolist() == [0.0] * 3 + [1.0] * 5
  assert win['is_first'][0] and win['is_first'][3]
  np.testing.assert_array_equal(win['vector'][3:, 0], np.arange(200, 205))


def test_workers_keep_separate_episodes():
  arc = _archive(8, 3)
  for i in range(12):
    arc.observe(_step(i, 1.0 if i == 6 else 0.0, first=(i == 0)), worker=0)
    arc.observe(_step(1000 + i, 0.0, first=(i == 0)), worker=1)
  win = arc.levels[1][0]
  assert (win['vector'][:, 0] < 1000).all()


# --- caps and sampling -------------------------------------------------------

def test_caps_drop_the_oldest_and_the_lowest_level():
  arc = _archive(8, 3, per_level=2, cap=3, pmax=1.0)
  for e in range(3):
    _episode(arc, 20, unlocks=(5,), start=100 * e)
  assert len(arc.levels[1]) == 2
  assert arc.levels[1][0]['vector'][0, 0] == 100      # episode 0 dropped
  _episode(arc, 20, unlocks=(5, 10), start=400)
  _episode(arc, 20, unlocks=(5, 10), start=500)
  assert arc.size == 3
  assert len(arc.levels[2]) == 2 and len(arc.levels[1]) == 1


def test_sample_is_none_without_frontier_windows():
  assert _archive().sample(4) is None


def test_sample_draws_levels_uniformly():
  arc = _archive(8, 3, pmax=1.0, per_level=50)
  for e in range(30):
    _episode(arc, 20, unlocks=(5,), start=100 * e)
  _episode(arc, 20, unlocks=(5, 10), start=9000)
  rows = arc.sample(2000)
  # Level 2 has one window, level 1 has 31; each level gets about half.
  share = np.mean(rows['reward'][:, 5] == 1.0)
  assert share == pytest.approx(1.0, abs=1e-6)  # every event step pays 1
  assert np.mean(np.asarray(arc.sampled_levels)[-1000:] == 2) == \
      pytest.approx(0.5, abs=0.07)


def test_attach_adds_rows_with_the_batch_dtypes():
  arc = _archive(8, 3)
  _episode(arc, 20, unlocks=(10,))
  batch = {
      'vector': np.zeros((4, 8, 2), np.float32),
      'action': np.zeros((4, 8), np.int32),
      'reward': np.zeros((4, 8), np.float32),
      'is_first': np.zeros((4, 8), bool),
      'is_terminal': np.zeros((4, 8), bool),
  }
  out = frlib.attach(batch, arc, OBS, ACT)
  assert out['fr/obs/vector'].shape == (4, 8, 2)
  assert out['fr/act/action'].dtype == np.int32
  assert out['fr/valid'].shape == (4, 8) and out['fr/valid'].sum() > 0
  # A batch of another length (report) gets zeros.
  short = {k: v[:, :5] for k, v in batch.items()}
  out = frlib.attach(short, arc, OBS, ACT)
  assert out['fr/valid'].sum() == 0 and out['fr/obs/vector'].shape[1] == 5


def test_save_and_load_round_trip(tmp_path):
  arc = _archive(8, 3, directory=str(tmp_path), pmax=1.0)
  for e in range(3):
    _episode(arc, 20, unlocks=(5, 10), start=100 * e)
  arc.save()
  new = _archive(8, 3, directory=str(tmp_path), pmax=1.0)
  new.load()
  assert new.best == arc.best and list(new.episodes) == list(arc.episodes)
  assert new.size == arc.size
  for L in arc.levels:
    for a, b in zip(arc.levels[L], new.levels[L]):
      for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_frontier_is_off_by_default():
  yaml = pytest.importorskip('yaml')
  import pathlib
  path = pathlib.Path(__file__).with_name('configs.yaml')
  cfg = yaml.safe_load(path.read_text(encoding='utf-8'))['defaults']['agent']
  assert cfg['frontier']['enabled'] is False
  assert cfg['frontier']['obs_keys'] == ['vector']


# --- loss weighting (JAX) ----------------------------------------------------

def test_weighting_gives_every_real_start_the_same_weight():
  jnp = pytest.importorskip('jax.numpy')
  from dreamerv3 import agent as agentlib
  B, K, F = 2, 3, 4
  per = jnp.arange(B * K + F, dtype=jnp.float32) + 1.0
  fmask = jnp.array([1.0, 0.0, 1.0, 0.0])
  out = agentlib.frontier_weighted(per, fmask, B, K)
  assert out.shape == (B, K)
  real = list(range(1, B * K + 1)) + [B * K + 1, B * K + 3]
  assert float(out.mean()) == pytest.approx(np.mean(real))
  # Without real frontier starts it is the normal losses unchanged.
  out = agentlib.frontier_weighted(per, jnp.zeros(F), B, K)
  np.testing.assert_allclose(out, per[:B * K].reshape(B, K))
