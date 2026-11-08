"""The slope quota: at least one new achievement per window, or a penalty.

The market agent's floor on score (docs/design-market.md §4). Pure numpy and
no game knowledge: it counts achievements from ``obs['ach']`` (the ones newly
unlocked this step) and steps, nothing else. The env (embodied/envs/craftax.py,
``env.craftax.quota``) adds what ``step`` returns to the reward, so the quota
is part of the score good the world model learns to predict.

  window   N(n) = margin x the running mean of the steps from the n-th
           achievement of an episode to the (n+1)-th (from the episode start
           for n = 0), over past episodes. Late in life achievements come
           slower, so the window grows with the count n instead of demanding
           the impossible.
  penalty  -k each time a window of N(n) steps passes with no new
           achievement; the window restarts at every achievement and after
           every penalty.
  death    at death the windows left up to a nominal lifespan L_nom are
           charged as failed, -k x ceil((L_nom - t) / N(n)), so dying never
           saves penalty. L_nom is the running mean episode length ('mean'),
           or a fixed number.

Before statistics exist. A count n with fewer than ``min_obs`` recorded gaps
uses the window of the nearest lower count that has them, but never less than
``init`` steps; with no statistics at all the window is ``init``. So the first
episodes, and any count the agent has not yet reached often, get a lenient
window rather than a strict guess. With lnom 'mean', no death is charged
until the first episode has ended (there is no lifespan to charge up to).

Gaps. Only completed gaps are recorded: an episode that ends before its next
achievement adds nothing at that count. When several achievements unlock on
one step, one gap is recorded at the count before them; the zero gaps between
them say nothing about how long the next one takes.

The statistics live in the env process and start fresh with it (a resumed
run relearns them within a few episodes); they are not checkpointed.
"""

import math

import numpy as np


class Quota:

  def __init__(self, k=1.0, margin=2.0, lnom='mean', init=200.0, rate=0.05,
               min_obs=5, max_count=256):
    self.k = float(k)
    self.margin = float(margin)
    # Flags parse a number given for a string default as a string.
    self.lnom = 'mean' if str(lnom) == 'mean' else float(lnom)
    self.init = float(init)
    self.rate = float(rate)
    self.min_obs = int(min_obs)
    assert self.k >= 0 and self.margin > 0 and self.init > 0, self.__dict__
    assert 0 < self.rate <= 1, self.rate
    # Debiased EMA per count: mean = sums / wts once wts > 0.
    self._sums = np.zeros(max_count, np.float64)
    self._wts = np.zeros(max_count, np.float64)
    self._obs = np.zeros(max_count, np.int64)
    self._life_sum = 0.0
    self._life_wt = 0.0
    self.begin()

  # --- statistics -------------------------------------------------------------
  def _ema(self, sums, wts, idx, x):
    sums[idx] = (1 - self.rate) * sums[idx] + self.rate * x
    wts[idx] = (1 - self.rate) * wts[idx] + self.rate

  def mean_gap(self, n):
    """Running mean steps from the n-th achievement to the next, or None."""
    n = min(int(n), len(self._obs) - 1)
    if self._obs[n] < self.min_obs:
      return None
    return float(self._sums[n] / self._wts[n])

  def window(self, n):
    """N(n): steps allowed for the next achievement at count n."""
    n = min(int(n), len(self._obs) - 1)
    for m in range(n, -1, -1):
      gap = self.mean_gap(m)
      if gap is None:
        continue
      win = self.margin * max(gap, 1.0)
      return win if m == n else max(win, self.init)
    return self.init

  def nominal_life(self):
    """L_nom, or None while it is still unknown."""
    if self.lnom != 'mean':
      return self.lnom
    if self._life_wt <= 0:
      return None
    return self._life_sum / self._life_wt

  # --- one episode --------------------------------------------------------------
  def begin(self):
    """A new episode starts (the reset step, which takes no action)."""
    self.t = 0          # steps taken this episode
    self.count = 0      # achievements unlocked this episode
    self.gap = 0        # steps since the last achievement (or the start)
    self.since = 0      # steps into the current window

  def step(self, new, terminal=False, last=False):
    """One env step: ``new`` achievements unlocked by it. Returns the reward
    term (zero or negative) to add to this step's reward."""
    new = int(new)
    self.t += 1
    self.gap += 1
    self.since += 1
    pen = 0.0
    if new > 0:
      n = min(self.count, len(self._obs) - 1)
      self._ema(self._sums, self._wts, n, float(self.gap))
      self._obs[n] += 1
      self.count += new
      self.gap = 0
      self.since = 0
    elif self.since >= self.window(self.count):
      pen += self.k
      self.since = 0
    if terminal:
      pen += self.death_charge()
    if last or terminal:
      self._life_sum = (1 - self.rate) * self._life_sum + self.rate * self.t
      self._life_wt = (1 - self.rate) * self._life_wt + self.rate
    return -pen

  def death_charge(self):
    """k x ceil((L_nom - t) / N(n)): the windows a death would skip."""
    life = self.nominal_life()
    if life is None or life <= self.t:
      return 0.0
    return self.k * math.ceil((life - self.t) / self.window(self.count))
