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

Quota v2 (``LogQuota``, ``env.craftax.quota.mode log``) is below. v1 stays as
``mode window``, the default, so v1 runs reproduce exactly.

Logging, both modes. After every ``step`` the ``info`` dict holds the pieces of
the reward term just returned: ``pen`` (the periodic penalty, a positive
number), ``death`` (the death charge, positive), ``floor`` (on the last step
of a life only, else 0: the window N(n) in v1, F(t) in v2) and, in v2, ``a``
and ``c``, the fit in force during the life. The env emits them as
``log/quota_<name>``; they never enter the reward or an observation the agent
sees.
"""

import collections
import math

import numpy as np


class Quota:

  # The keys of ``info`` (the env's ``log/quota_*`` metrics).
  LOGS = ('pen', 'death', 'floor')

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
    self.info = dict.fromkeys(self.LOGS, 0.0)
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
    periodic, death = pen, 0.0
    if terminal:
      death = self.death_charge()
      pen += death
    if last or terminal:
      self._life_sum = (1 - self.rate) * self._life_sum + self.rate * self.t
      self._life_wt = (1 - self.rate) * self._life_wt + self.rate
    self.info = dict(
        pen=periodic, death=death,
        floor=self.window(self.count) if last or terminal else 0.0)
    return -pen

  def death_charge(self):
    """k x ceil((L_nom - t) / N(n)): the windows a death would skip."""
    life = self.nominal_life()
    if life is None or life <= self.t:
      return 0.0
    return self.k * math.ceil((life - self.t) / self.window(self.count))


class LogQuota:
  """Quota v2: a floor shaped like a logarithm, F(t) = frac x a x ln(1 + t/c).

  t is the step within the life. (a, c) are fitted to the agent's own recent
  lives: the unlock points (t_k, k) (the k-th achievement of a life at step
  t_k) of the last ``fit_lives`` completed lives, refitted by least squares at
  each life's end. The memory is bounded and lives on in the env process (not
  checkpointed). Until ``min_lives`` lives are recorded there is no floor and
  no penalty. A life with no achievement counts as a life but adds no points.

  check  every ``every`` steps of a life, if the count of achievements
         unlocked so far is below F(t), the reward gets -k. An achievement
         never restarts anything: being ahead of the floor banks credit, so
         there is no reason to hold achievements back (v1's per-window clock
         rewarded exactly that pacing).
  death  at death (terminal, not a timeout) the agent pays -k x ``death_cap``
         if its count is below F(t) at that step, else nothing. A capped,
         single charge instead of v1's lump of all the remaining windows. It
         is paid on top of a check that falls on the same step.

  fit    for each c on a log-spaced grid, the best a has a closed form,
         a = sum(k g) / sum(g^2) with g = ln(1 + t/c); the (a, c) with the
         lowest squared residual wins, then a finer grid between the winner's
         neighbours refines it. No scipy.
  """

  LOGS = ('pen', 'death', 'floor', 'a', 'c')

  def __init__(self, k=1.0, frac=0.5, every=100, death_cap=1.0,
               fit_lives=50, min_lives=5):
    self.k = float(k)
    self.frac = float(frac)
    self.every = int(every)
    self.death_cap = float(death_cap)
    self.fit_lives = int(fit_lives)
    self.min_lives = int(min_lives)
    assert self.k >= 0 and self.frac >= 0 and self.death_cap >= 0, self.__dict__
    assert self.every >= 1 and 1 <= self.min_lives <= self.fit_lives, (
        self.every, self.min_lives, self.fit_lives)
    # One (times, counts) pair of arrays per completed life.
    self._lives = collections.deque(maxlen=self.fit_lives)
    self._grid = np.geomspace(0.5, 500.0, 64)
    self.a = None
    self.c = None
    self.info = dict.fromkeys(self.LOGS, 0.0)
    self.begin()

  # --- the fit ----------------------------------------------------------------
  def _best(self, grid, t, k):
    """The (a, c, residual) with c in ``grid`` and the best closed-form a."""
    g = np.log1p(t[None, :] / grid[:, None])
    a = (g * k).sum(1) / (g * g).sum(1)
    res = ((k[None, :] - a[:, None] * g) ** 2).sum(1)
    i = int(np.argmin(res))
    return float(a[i]), float(grid[i]), float(res[i]), i

  def fit(self):
    """Refit (a, c) to the stored lives; None until min_lives are recorded."""
    if len(self._lives) < self.min_lives:
      self.a = self.c = None
      return
    t = np.concatenate([x[0] for x in self._lives])
    k = np.concatenate([x[1] for x in self._lives])
    if len(t) == 0:                    # no life unlocked anything: no floor
      self.a, self.c = 0.0, float(self._grid[0])
      return
    a, c, _, i = self._best(self._grid, t, k)
    lo = self._grid[max(i - 1, 0)]
    hi = self._grid[min(i + 1, len(self._grid) - 1)]
    a, c, _, _ = self._best(np.geomspace(lo, hi, 33), t, k)
    self.a, self.c = a, c

  def floor(self, t):
    """F(t), or None while there is no fit."""
    if self.a is None:
      return None
    return self.frac * self.a * math.log1p(t / self.c)

  # --- one life -----------------------------------------------------------------
  def begin(self):
    """A new life starts (the reset step, which takes no action). An
    unfinished life is dropped."""
    self.t = 0
    self.count = 0
    self._times = []
    self._counts = []

  def step(self, new, terminal=False, last=False):
    """One env step: ``new`` achievements unlocked by it. Returns the reward
    term (zero or negative) to add to this step's reward."""
    new = int(new)
    self.t += 1
    for _ in range(new):
      self.count += 1
      self._times.append(self.t)
      self._counts.append(self.count)
    floor = self.floor(self.t)
    pen = death = 0.0
    if floor is not None:
      if self.t % self.every == 0 and self.count < floor:
        pen = self.k
      if terminal and self.count < floor:
        death = self.k * self.death_cap
    self.info = dict(
        pen=pen, death=death,
        floor=(floor or 0.0) if last or terminal else 0.0,
        a=self.a or 0.0, c=self.c or 0.0)
    if last or terminal:
      self._lives.append((np.asarray(self._times, np.float64),
                          np.asarray(self._counts, np.float64)))
      self.fit()
    return -(pen + death)


def make(mode='window', **kw):
  """The quota for ``env.craftax.quota.mode``. Each mode takes its own keys
  from the config group and ignores the other's."""
  mode = str(mode)
  if mode == 'window':
    keys = ('k', 'margin', 'lnom', 'init', 'rate', 'min_obs')
    return Quota(**{k: v for k, v in kw.items() if k in keys})
  if mode == 'log':
    keys = ('k', 'frac', 'every', 'death_cap', 'fit_lives', 'min_lives')
    return LogQuota(**{k: v for k, v in kw.items() if k in keys})
  raise ValueError(f'quota mode must be window or log, got {mode!r}')
