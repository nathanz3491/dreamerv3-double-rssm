"""Taste (taste.py) and its hooks on the manager: the fold matches the toy's
rule, evidence waits for the fold, the taste leaves untouched slots alone, the
relative taste the manager sees is neutral to a level shared by every goal,
and the manager's evidence is the return after each real choice minus its
expectation. (The agent hooks of taste over actions are tested on
family/taste.)

Run: python -m pytest dreamerv3/test_taste.py
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3 import agent as agentlib
from dreamerv3 import taste as tastelib

f32 = jnp.float32
KW = dict(n_act=4, basic=(0,), every=2, fresh=100.0, zscale=0.5, w0=0.1,
          alpha=0.4, rho=0.05, pos=4.0, neg=1.5, decay=0.5, lam=5.0,
          optimism=0.0, n0=20.0, stdrate=1.0)


def _run(batches, **kw):
  """Observe each (adv, act, weight) batch; return taste state after each."""
  cfg = {**KW, **kw}
  def fn(batches):
    t = tastelib.Taste(**cfg, name='taste')
    out = []
    for adv, act, weight, *steps in batches:
      t.observe(adv, act, weight, *steps)
      out.append(dict(w=t.w.read(), n=t.n.read(), eff=t.eff.read(),
                      bias=t.bias(), shown=t.shown()))
    return out
  state = nj.init(fn)({}, batches, seed=0)
  _, out = nj.pure(fn)(state, batches, seed=0)
  return jax.tree.map(np.asarray, out)


def _batch(adv, act, weight=None):
  adv, act = f32(adv)[None], jnp.asarray(act, jnp.int32)[None]
  weight = jnp.ones_like(adv) if weight is None else f32(weight)[None]
  return adv, act, weight


def test_fold_matches_the_toy_update_rule():
  # Two batches, folded after the second (every=2). stdrate 1 makes the
  # spread that of the last batch alone.
  b1 = _batch([1.0, -1.0, 2.0, -2.0], [1, 2, 1, 2])
  b2 = _batch([3.0, -3.0, 1.0, -1.0], [1, 2, 1, 2])
  out = _run([b1, b2])
  np.testing.assert_allclose(out[0]['w'], KW['w0'])     # no fold yet
  std2 = np.sqrt(np.mean(np.square([3.0, -3.0, 1.0, -1.0])))
  # Each batch is z-scored by the spread known when it arrived.
  std1 = np.sqrt(np.mean(np.square([1.0, -1.0, 2.0, -2.0])))
  seen1 = (1.0 / std1 + 2.0 / std1 + 3.0 / std2 + 1.0 / std2) / 4
  count = 100.0 * 4 / 8                                 # half the samples
  n = count
  q = n / (n + 20.0) * seen1 * 0.5
  w = np.clip(0.1 + 0.4 * 4.0 * max(q, 0) - 0.05 * (0.1 - 0.1), -1, 1)
  np.testing.assert_allclose(out[1]['n'][1], n, rtol=1e-5)
  np.testing.assert_allclose(out[1]['eff'][1], seen1, rtol=1e-5)
  np.testing.assert_allclose(out[1]['w'][1], w, rtol=1e-5)
  assert out[1]['w'][2] < KW['w0']                      # bad evidence: down
  np.testing.assert_allclose(out[1]['w'][3], KW['w0'])  # never taken: unchanged


def test_bad_news_moves_taste_less_than_good_news():
  b = _batch([2.0, -2.0], [1, 2])
  out = _run([b, b])
  up, down = out[1]['w'][1] - KW['w0'], KW['w0'] - out[1]['w'][2]
  assert up > 0 and down > 0
  np.testing.assert_allclose(up / down, KW['pos'] / KW['neg'], rtol=1e-4)


def test_bias_is_zero_on_basic_actions_and_before_any_evidence():
  b = _batch([2.0, -2.0, 1.0, 1.0], [0, 2, 0, 3])
  out = _run([b, b])
  np.testing.assert_allclose(out[0]['bias'], 0.0)       # n = 0: optimism 0
  assert out[1]['bias'][0] == 0.0                       # basic: untouched
  assert out[1]['bias'][3] > 0 and out[1]['bias'][2] < 0
  np.testing.assert_allclose(out[1]['bias'], 5.0 * out[1]['shown'] * [0, 1, 1, 1],
                             rtol=1e-6)


def test_faded_evidence_falls_back_to_optimism():
  b = _batch([2.0, -2.0], [1, 2])
  quiet = _batch([0.0, 0.0], [3, 3])
  out = _run([b, b] + [quiet] * 40, optimism=0.3)
  # Actions 1 and 2 stop being taken; their memory decays toward nothing and
  # what the policy sees returns toward optimism whatever the taste.
  for a in (1, 2):
    assert abs(out[-1]['shown'][a] - 0.3) < abs(out[1]['shown'][a] - 0.3)
    assert abs(out[-1]['shown'][a] - 0.3) < 1e-3


def test_zero_weight_steps_carry_no_evidence():
  b = _batch([5.0, -5.0, 1.0, -1.0], [1, 2, 1, 2], weight=[0, 0, 1, 1])
  same = _batch([1.0, -1.0], [1, 2])
  a, c = _run([b, b]), _run([same, same])
  np.testing.assert_allclose(a[1]['w'], c[1]['w'], rtol=1e-5)


def test_centered_evidence_ignores_a_constant_shift():
  # The fix for v2-taste: a critic that is off by the same amount everywhere
  # must not move taste. Only how each slot compares with the average does.
  # Two different batches with different slot mixes, so centering per batch
  # and centering per fold would give different answers.
  b1 = _batch([1.0, -1.0, 2.0, 0.0], [1, 2, 1, 3])
  b2 = _batch([0.5, 1.5, -2.0, -1.0], [1, 1, 2, 3])
  shift = lambda b: (b[0] - 3.0, b[1], b[2])
  a = _run([b1, b2], center=True)
  c = _run([shift(b1), shift(b2)], center=True)
  np.testing.assert_allclose(a[1]['w'], c[1]['w'], rtol=1e-5)
  # Centered per fold: slot 1's mean z over both batches beats the average.
  assert a[1]['w'][1] > KW['w0'] > a[1]['w'][2]


def test_uncentered_negative_evidence_sinks_the_most_taken_slot_most():
  # v2-taste's failure in miniature: every slot averages the same amount
  # below the critic. Uncentered, the slot taken most sinks most; centered,
  # none moves.
  b = _batch([-0.5, -1.5, -0.5, -1.5, -0.5, -1.5, -0.5, -1.5],
             [1, 1, 1, 1, 1, 1, 2, 2])
  plain = _run([b, b])[1]['w']
  centered = _run([b, b], center=True)[1]['w']
  assert plain[1] < plain[2] < KW['w0']
  np.testing.assert_allclose(centered, KW['w0'], atol=1e-6)


def test_counts_are_a_share_of_all_steps_when_evidence_is_sparse():
  # Manager choices: evidence at 2 of 8 steps, so 1/8 of the fresh env steps
  # go to each chosen goal, not half.
  adv, act, _ = _batch([1.0, 0, 0, 0, -1.0, 0, 0, 0], [1, 0, 0, 0, 2, 0, 0, 0])
  weight = f32([1, 0, 0, 0, 1, 0, 0, 0])[None]
  out = _run([(adv, act, weight, 8.0)] * 4, target='manager')
  # Fold 1: 2 choices of each goal in 16 steps.
  np.testing.assert_allclose(out[1]['n'][1:3], 100.0 * 2 / 16, rtol=1e-5)
  # Fold 2: the step count restarted, old evidence decayed by half.
  np.testing.assert_allclose(out[3]['n'][1:3], 0.5 * 12.5 + 12.5, rtol=1e-5)


def test_steps_are_required_exactly_for_the_manager_target():
  b = _batch([1.0], [1])
  with pytest.raises(AssertionError):
    _run([b], target='manager')
  with pytest.raises(AssertionError):
    _run([(*b, 1.0)])


def test_relative_taste_is_neutral_to_a_shared_level():
  # Optimism or drift shared by every tasted goal must not act against NONE
  # (slot 0): the manager's advantage cancels a shift over all slots, not
  # over 12 of 13.
  b = _batch([2.0, -2.0], [1, 2])
  out = _run([b, b], optimism=0.25)
  def rel(o):
    m = np.array([0, 1, 1, 1.0])
    return (o['shown'] - (o['shown'] * m).sum() / m.sum()) * m
  for o in out:
    r = rel(o)
    assert r[0] == 0.0 and abs(r[1:].mean()) < 1e-6
  np.testing.assert_allclose(rel(out[0]), 0.0, atol=1e-7)   # optimism only
  assert rel(out[1])[1] > 0 > rel(out[1])[2]


def test_relative_matches_the_module():
  def fn():
    t = tastelib.Taste(**{**KW, 'optimism': 0.3}, name='taste')
    t.w.write(f32([0.9, 0.5, -0.2, 0.1]))
    t.n.write(f32([50.0, 20.0, 60.0, 0.0]))
    return t.shown(), t.relative()
  state = nj.init(fn)({}, seed=0)
  _, (shown, rel) = nj.pure(fn)(state, seed=0)
  shown, rel = np.asarray(shown), np.asarray(rel)
  assert rel[0] == 0.0
  np.testing.assert_allclose(rel[1:], shown[1:] - shown[1:].mean(), rtol=1e-6)


class _GoalStub:
  _goals = 3
  _phases = 4
  _goalfeat = agentlib.Agent._goalfeat


def test_all_goals_input_matches_the_replay_critic_layout():
  # Replay feeds the critic [actor view, goal one-hot, phase one-hot, RSSM-2,
  # flags]; the all-goals input must be that, once per goal at phase 0.
  base = jnp.arange(2 * 5 * 3, dtype=f32).reshape(2, 5, 3)
  d2 = jnp.ones((2, 5, 2), f32) * 7
  flags = jnp.zeros((2, 5, 3), bool).at[:, :, 1].set(True)
  out = np.asarray(agentlib.Agent._all_goals_input(
      _GoalStub(), base, d2, flags), np.float32)
  assert out.shape == (2, 5, 3, 3 + 3 + 4 + 2 + 3)
  for g in range(3):
    want = np.concatenate([
        np.asarray(base), np.broadcast_to(np.eye(3)[g], (2, 5, 3)),
        np.broadcast_to(np.eye(4)[0], (2, 5, 4)), np.asarray(d2),
        np.asarray(flags, np.float32)], -1)
    np.testing.assert_allclose(out[:, :, g], want)


class _EvidenceStub:
  """Enough of Agent for _taste_mgr_evidence: Q(s, g) = g, an untrained
  manager (uniform over the goals not yet reached), a recording taste."""
  _goals = 3
  _phases = 4
  _goalfeat = agentlib.Agent._goalfeat
  _all_goals_input = agentlib.Agent._all_goals_input
  _mgr_input = agentlib.Agent._mgr_input
  _mgr_dist = agentlib.Agent._mgr_dist
  D = 2                                     # width of the actor's view

  def __init__(self, rows=256):
    self.calls = []
    stub = self

    class Taste:
      def observe(self, adv, act, weight, steps):
        stub.calls.append(jax.tree.map(np.asarray, (adv, act, weight, steps)))
        return {}
    self.taste = Taste()
    self.taste.rows = rows

  def val(self, x, bdims):
    D, G = self.D, self._goals
    g = (f32(x[..., D:D + G]) * jnp.arange(G)).sum(-1)

    class Head:
      pred = staticmethod(lambda: g)
    return {'game': Head()}

  class valnorm:
    stats = staticmethod(lambda: (0.0, 1.0))

  def mgr(self, x, bdims):
    class Out:
      logits = jnp.zeros((*x.shape[:bdims], 3), f32)
    return Out()


def test_manager_evidence_is_the_return_after_each_choice_minus_its_expectation():
  B, K, G = 2, 5, 3
  gphase = np.ones((B, K), np.int32)
  gphase[0, 1] = 0                    # choice at step 1 of row 0
  gphase[1, 3] = 0                    # choice at step 3 of row 1
  gphase[1, 4] = 0                    # last step: no following return, cut
  gphase[0, 3] = 0                    # an episode end: no evidence
  goal = np.zeros((B, K), np.int32)
  goal[0, 1], goal[1, 3], goal[0, 3] = 2, 1, 1
  last = np.zeros((B, K), bool)
  last[0, 3] = True
  phi = np.zeros((B, K, G), np.float32)
  phi[1, 3, 2] = 1.0                  # goal 2 already holds at the row-1 choice
  obs = dict(gphase=jnp.asarray(gphase), goal=jnp.asarray(goal),
             goalreach=jnp.zeros((B, K, G), f32), goalphi=jnp.asarray(phi))
  ret = jnp.arange(B * (K - 1), dtype=f32).reshape(B, K - 1) * 10
  base = jnp.zeros((B, K, _EvidenceStub.D), f32)
  deter2 = jnp.zeros((B, K, 2), f32)
  stub = _EvidenceStub()
  agentlib.Agent._taste_mgr_evidence(
      stub, base, deter2, obs, ret, jnp.asarray(last), K)
  adv, act, weight, steps = stub.calls[0]
  n = int(weight.sum())
  assert n == 2 and (weight[0, :n] == 1).all() and (weight[0, n:] == 0).all()
  # Row 0, step 1: ret[0, 1] = 10; uniform over goals 0-2, E[Q] = 1.
  # Row 1, step 3: ret[1, 3] = 70; goal 2 holds, uniform over 0-1, E[Q] = 0.5.
  np.testing.assert_allclose(adv[0, :n], [10 - 1.0, 70 - 0.5], rtol=1e-5)
  np.testing.assert_array_equal(act[0, :n], [2, 1])
  assert steps == B * (K - 1) - 1     # the episode-end step is not a step


def test_manager_evidence_drops_choices_beyond_the_row_cap():
  B, K = 1, 6
  obs = dict(gphase=jnp.zeros((B, K), jnp.int32),
             goal=jnp.ones((B, K), jnp.int32),
             goalreach=jnp.zeros((B, K, 3), f32),
             goalphi=jnp.zeros((B, K, 3), f32))
  ret = jnp.arange(K - 1, dtype=f32)[None] * 10
  stub = _EvidenceStub(rows=2)
  mets = agentlib.Agent._taste_mgr_evidence(
      stub, jnp.zeros((B, K, 2), f32), jnp.zeros((B, K, 2), f32), obs, ret,
      jnp.zeros((B, K), bool), K)
  adv, _, weight, _ = stub.calls[0]
  assert adv.shape == (1, 2) and weight.sum() == 2
  np.testing.assert_allclose(adv[0], [0 - 1.0, 10 - 1.0])  # the first two
  assert float(mets['choices']) == 5 and float(mets['choices_dropped']) == 3
