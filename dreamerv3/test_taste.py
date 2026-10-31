"""Taste over actions (taste.py): the fold matches the toy's rule, evidence
waits for the fold, the bias leaves basic actions alone, and the policy sees
the bias wherever it acts.

Run: python -m pytest dreamerv3/test_taste.py
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3 import agent as agentlib
from dreamerv3 import taste as tastelib
from embodied.jax import outs as jouts

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
    for adv, act, weight in batches:
      t.observe(adv, act, weight)
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


class _Stub:
  _taste = True
  _exempt_basic = True

  class taste:
    @staticmethod
    def bias():
      return f32([0.0, 0, 0, 0, 0, 0] + [2.0] + [0.0] * 36)


def test_masked_adds_the_taste_bias_with_and_without_a_mask():
  logits = jnp.zeros((2, 43), f32)
  policy = {'action': jouts.Categorical(logits)}
  plain = agentlib.Agent._masked(_Stub(), policy, None)
  np.testing.assert_allclose(np.asarray(plain['action'].logits)[:, 6], 2.0)
  np.testing.assert_allclose(np.asarray(plain['action'].logits)[:, 7], 0.0)
  valid = jnp.zeros((2, 43), bool).at[:, 6].set(True)
  masked = agentlib.Agent._masked(_Stub(), policy, valid)
  out = np.asarray(masked['action'].logits)
  np.testing.assert_allclose(out[:, 6], 2.0)
  assert (out[:, 7] < -1e3).all()                       # mask still applies
