"""Frontier dreaming: an archive of rarely reached progress moments.

See docs/design-frontier.md. The archive runs on the host. It watches every
env step (``observe``), records a window of steps around each moment where an
episode first reaches a rare progress level, and hands windows to the train
stream (``sample``). The agent re-encodes those windows with the current world
model and starts part of its imagined rollouts from them.

Progress level: the number of reward events (``reward >= rmin``) so far in the
episode. In Craftax these are the achievement unlocks. The level is derived
from the reward alone; ``obs['ach']`` is used only by the tests.

Window layout (length ``W``, the train batch's ``batch_length +
replay_context``): the event sits at position ``W - S`` and positions
``W - S .. W - 1`` (the event and the ``S - 1`` steps after it) are the
imagination starts. The ``W - S`` steps before rebuild the agent's memory of
how it got there. Position 0 is always marked ``is_first``.
"""

import collections
import io
import threading

import numpy as np


class FrontierArchive:

  def __init__(
      self, length, starts, obs_keys, act_keys, window=200, pmax=0.25,
      cap=256, per_level=32, rmin=0.5, seed=0):
    assert 0 < starts <= length, (starts, length)
    self.W = int(length)
    self.S = int(starts)
    self.obs_keys = tuple(obs_keys)
    self.act_keys = tuple(act_keys)
    self.pmax = float(pmax)
    self.cap = int(cap)
    self.per_level = int(per_level)
    self.rmin = float(rmin)
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()

    self.episodes = collections.deque(maxlen=int(window))  # max level each
    self.best = 0
    self.levels = collections.defaultdict(collections.deque)  # L -> windows
    self.workers = {}
    self.events = 0
    self.sampled_levels = collections.deque(maxlen=1000)

  # --- rarity ------------------------------------------------------------

  def reach(self, level):
    """Fraction of recent finished episodes that reached ``level``."""
    if not self.episodes:
      return 0.0
    eps = np.asarray(self.episodes)
    return float((eps >= level).mean())

  def is_frontier(self, level):
    if level < 1:
      return False
    return level >= self.best or self.reach(level) <= self.pmax

  # --- recording ---------------------------------------------------------

  def observe(self, tran, worker=0):
    with self.lock:
      self._observe(tran, worker)

  def _observe(self, tran, worker):
    w = self.workers.get(worker)
    if w is None:
      w = dict(n=0, ring=collections.deque(maxlen=self.W), start=0,
               level=0, pending=[])
      self.workers[worker] = w
    n = w['n']
    if tran['is_first']:
      # A new episode without a closing is_last: flush what is pending.
      self._flush(w, end=n - 1)
      w['start'] = n
      w['level'] = 0
    step = {k: np.array(tran[k]) for k in self.obs_keys + self.act_keys}
    step['reward'] = np.float32(tran['reward'])
    step['is_terminal'] = bool(tran['is_terminal'])
    w['ring'].append((n, step))

    if not tran['is_first'] and float(tran['reward']) >= self.rmin:
      w['level'] += 1
      level = w['level']
      if self.is_frontier(level):
        w['pending'].append((n, level))
        self.events += 1
      self.best = max(self.best, level)

    # Finalize events whose S - 1 following steps are now in the ring.
    done = [e for e in w['pending'] if n >= e[0] + self.S - 1]
    for event in done:
      self._store(w, event, end=n)
      w['pending'].remove(event)

    if tran['is_last']:
      self._flush(w, end=n)
      self.episodes.append(w['level'])
    w['n'] = n + 1

  def _flush(self, w, end):
    for event in w['pending']:
      self._store(w, event, end=end)
    w['pending'] = []

  def _store(self, w, event, end):
    """Build the window around ``event`` from the ring (steps up to ``end``)."""
    t, level = event
    first = t - (self.W - self.S)
    steps = dict(w['ring'])
    example = next(iter(steps.values()))
    win = {k: np.zeros((self.W, *np.shape(v)), np.asarray(v).dtype)
           for k, v in example.items()}
    win['is_first'] = np.zeros(self.W, bool)
    win['valid'] = np.zeros(self.W, np.float32)
    for p in range(self.W):
      c = first + p
      if c > end or c not in steps:
        continue  # padding: before the ring or after the episode ended
      for k, v in steps[c].items():
        win[k][p] = v
      win['valid'][p] = float(c >= w['start'])
      win['is_first'][p] = (c == w['start'])
    win['is_first'][0] = True
    store = self.levels[level]
    store.append(win)
    while len(store) > self.per_level:
      store.popleft()
    while self.size > self.cap:
      lowest = min(L for L, ws in self.levels.items() if ws)
      self.levels[lowest].popleft()

  @property
  def size(self):
    return sum(len(ws) for ws in self.levels.values())

  # --- sampling ----------------------------------------------------------

  def frontier_levels(self):
    return sorted(L for L, ws in self.levels.items()
                  if ws and self.is_frontier(L))

  def sample(self, batch):
    """Return ``batch`` windows as arrays of shape (batch, W, ...).

    ``valid`` is all zero when no frontier window exists yet; the agent then
    ignores the rows.
    """
    with self.lock:
      levels = self.frontier_levels()
      if not levels:
        return None
      wins = []
      for _ in range(batch):
        L = levels[self.rng.integers(len(levels))]
        ws = self.levels[L]
        wins.append(ws[self.rng.integers(len(ws))])
        self.sampled_levels.append(L)
    return {k: np.stack([w[k] for w in wins]) for k in wins[0]}

  def stats(self):
    with self.lock:
      levels = self.frontier_levels()
      out = {
          'size': self.size,
          'events': self.events,
          'best': self.best,
          'num_frontier_levels': len(levels),
          'lowest_frontier_level': min(levels) if levels else 0,
      }
      if self.sampled_levels:
        out['sampled_level'] = float(np.mean(self.sampled_levels))
      for L in range(1, self.best + 1):
        out[f'reach/{L:02d}'] = self.reach(L)
        out[f'windows/{L:02d}'] = len(self.levels.get(L, ()))
      return out

  # --- checkpointing (elements.Checkpoint protocol) ----------------------

  def save(self):
    """The archive as compressed npz bytes, for elements.Checkpoint.

    It is stored inside the checkpoint folder (`frontier.pkl`), so it is
    written, kept and cleaned up together with the rest of the checkpoint.
    """
    with self.lock:
      arrays = {
          'episodes': np.asarray(self.episodes, np.int32),
          'best': np.int32(self.best),
          'events': np.int64(self.events),
      }
      for L, ws in self.levels.items():
        for i, win in enumerate(ws):
          for k, v in win.items():
            arrays[f'w/{L}/{i}/{k}'] = v
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()

  def load(self, data):
    with self.lock, np.load(io.BytesIO(data)) as f:
      self.episodes.clear()
      self.episodes.extend(int(x) for x in f['episodes'])
      self.best = int(f['best'])
      self.events = int(f['events'])
      wins = collections.defaultdict(dict)
      for key in f.files:
        if key.startswith('w/'):
          _, L, i, k = key.split('/', 3)
          wins[(int(L), int(i))][k] = f[key]
      self.levels.clear()
      for (L, i) in sorted(wins):
        self.levels[L].append(wins[(L, i)])


def keys(obs_keys, act_keys):
  """Batch key for each window key."""
  out = {k: f'fr/obs/{k}' for k in obs_keys}
  out.update({k: f'fr/act/{k}' for k in act_keys})
  out.update({k: f'fr/{k}' for k in (
      'reward', 'is_first', 'is_terminal', 'valid')})
  return out


def attach(batch, archive, obs_keys, act_keys):
  """Add one frontier row per batch row to a train or report batch.

  Rows come from the archive when it has frontier windows of the batch's
  length; otherwise they are zeros with ``valid`` all zero, which the agent
  ignores. Report batches (a different length) always get zeros.
  """
  B, T = batch['is_first'].shape
  rows = None
  if archive is not None and archive.W == T:
    rows = archive.sample(B)
  names = keys(obs_keys, act_keys)
  if rows is None:
    like = {k: batch[k] for k in (*obs_keys, *act_keys)}
    like['reward'] = batch['reward']
    like['is_first'] = batch['is_first']
    like['is_terminal'] = batch['is_terminal']
    like['valid'] = np.zeros((B, T), np.float32)
    rows = {k: np.zeros_like(v) for k, v in like.items()}
  dtypes = {k: batch[k].dtype for k in (
      *obs_keys, *act_keys, 'reward', 'is_first', 'is_terminal')}
  dtypes['valid'] = np.float32
  batch = dict(batch)
  for k, name in names.items():
    batch[name] = np.asarray(rows[k]).astype(dtypes[k], copy=False)
  return batch
