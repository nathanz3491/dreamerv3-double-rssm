"""RSSM-2 -- the slow map model.

A second, coarser world model that ticks once per ``tick`` env steps and
predicts a 12x12x16 map of the whole level plus the agent's own coarse cell.
Where the agent has been it recalls; where it hasn't it predicts from the prior
that Craftax's biomes are clustered rather than random.

Two properties the design leans on (``design-map-model.md``):

* **Deterministic, not stochastic.** RSSM-1 needs a stochastic latent because it
  models one-step dynamics under uncertainty. RSSM-2 predicts static terrain, so
  a GRU plus two heads is enough -- no KL, no sampling.
* **Position is a TARGET, not an input.** The 8268-dim observation contains no
  absolute (x, y), so RSSM-2 must dead-reckon from its own movement. Asking it
  to predict its cell is what makes path integration actually get learned.

Gradient isolation is enforced by callers, not here: ``agent.py`` stop-gradients
``feat1`` on the way in (so the map loss never reaches RSSM-1) and the decoded
map on the way out (so policy gradients never reach RSSM-2).
"""

import jax
import jax.numpy as jnp
import numpy as np
import ninjax as nj
from jax import numpy as jnp  # noqa: F811  (kept explicit for readability)

import embodied.jax.nets as nn

f32 = jnp.float32


class MapModel(nj.Module):
  """Slow GRU over aggregated RSSM-1 features; decodes a coarse level map."""

  deter: int = 1024
  hidden: int = 512
  layers: int = 2
  coarse: int = 12
  planes: int = 16
  act: str = 'silu'
  norm: str = 'rms'

  def __init__(self, **kw):
    self.kw = dict(**kw)
    self.cells = self.coarse * self.coarse

  # --- state ----------------------------------------------------------------
  def initial(self, bsize):
    return nn.cast(dict(deter2=jnp.zeros([bsize, self.deter], f32)))

  @property
  def entry_space(self):
    """Stored in the replay context so a mid-episode chunk resumes the map.

    Without this, a 64-step batch would only ever give RSSM-2 eight ticks --
    far too few to learn a map that accumulates over hundreds of steps.
    """
    import elements
    import numpy as np
    # count: the step within the current window, so a chunk that resumes
    # mid-window closes its first window on the same step the actor did.
    return dict(deter2=elements.Space(np.float32, (self.deter,)),
                count=elements.Space(np.float32, (1,)))

  def truncate(self, entries, carry=None):
    return jax.tree.map(lambda x: x[:, -1], entries)

  # --- recurrence -----------------------------------------------------------
  def _gru(self, deter, x):
    # nets.Linear asserts its input is COMPUTE_DTYPE (bf16); unlike MLPHead we
    # build layers by hand, so casting is our job.
    x = nn.cast(jnp.concatenate([nn.cast(deter), nn.cast(x)], -1))
    for i in range(self.layers):
      x = self.sub(f'hid{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'hid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('gru', nn.Linear, 3 * self.deter, **self.kw)(x)
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    return update * cand + (1 - update) * deter

  def tick(self, deter2, x):
    """One window closes: (B, D2), (B, F) -> (B, D2).

    The only recurrence. Acting and training both call it from the same
    per-step function in agent.py, once per closed window, so the state the
    actor reads online is exactly the state the batch path trains on.
    """
    return self._gru(deter2, x)

  # --- decoding -------------------------------------------------------------
  def decode(self, deter2):
    """(..., deter) -> map logits (..., C, C, P) and position logits (..., C*C)."""
    x = nn.cast(deter2)
    for i in range(self.layers):
      x = self.sub(f'dec{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dec{i}norm', nn.Norm, self.norm)(x))
    mlogit = self.sub('decmap', nn.Linear, self.cells * self.planes, **self.kw)(x)
    mlogit = mlogit.reshape((*x.shape[:-1], self.coarse, self.coarse, self.planes))
    plogit = self.sub('decpos', nn.Linear, self.cells, **self.kw)(x)
    return mlogit, plogit

  def gate(self):
    """Scalar the actor's map input is multiplied by. Learned, initialised 0.

    This replaces a hand-tuned warm-up schedule. At 0 the actor is map-blind, so
    an untrained RSSM-2 cannot poison the policy while everything trains at
    once; the gate still receives gradient (d loss/d gate is non-zero even at 0),
    so the channel opens by itself exactly as fast as it starts paying. Logged
    as ``map/gate``: if it stays near zero the actor found the map useless, which
    is the crop-ablation test running continuously instead of once at the end.
    """
    return self.value('gate', jnp.zeros, (), f32)

  # --- losses ---------------------------------------------------------------
  def loss(self, deter2, map_target, pos_target, weight=None):
    """BCE over OBSERVED cells + cross-entropy over position.

    ``weight`` is (B, T, C, C) or (B, T, C, C, P). For terrain planes it is
    the fraction of each coarse cell the agent has actually observed
    (``obs['mapknown']``), so cells it has never seen contribute no gradient.

    This used to supervise every cell against the true map, on the reasoning that
    prediction of the unseen is the point. It is, but grading against terrain the
    agent had no way to observe teaches Craftax's world generator rather than
    inference, and leaves no held-out set: every cell the model was scored on, it
    had also studied. Prediction of the unseen is preserved instead by HINDSIGHT
    (``segment_last``) -- an early tick is graded against what the agent went
    on to discover, so every label is still something it saw with its own eyes.

    Normalised by weight mass and rescaled by the cell count rather than summed
    over all cells. At ~26% coverage a plain sum would shrink this loss ~4x
    against every other term and the head would quietly stop training; with
    ``weight`` all ones the value is identical to the old sum.
    """
    mlogit, plogit = self.decode(deter2)
    mlogit, plogit = f32(mlogit), f32(plogit)   # reduce in f32, not bf16
    tgt = f32(map_target)
    bce = jnp.maximum(mlogit, 0) - mlogit * tgt + jnp.log1p(jnp.exp(-jnp.abs(mlogit)))
    if weight is None:
      map_loss = bce.sum((-1, -2, -3))
    else:
      # (..., C, C) weights every plane of a cell alike; (..., C, C, P) lets
      # the caller weight planes separately.
      w = f32(weight)
      if w.ndim == bce.ndim - 1:
        w = w[..., None]
      w = jnp.broadcast_to(w, bce.shape)
      n = bce.shape[-1] * bce.shape[-2] * bce.shape[-3]
      map_loss = (bce * w).sum((-1, -2, -3)) / jnp.maximum(
          w.sum((-1, -2, -3)), 1e-3) * n
    onehot = jax.nn.one_hot(pos_target.astype(jnp.int32), self.cells)
    pos_loss = -(onehot * jax.nn.log_softmax(plogit, -1)).sum(-1)
    return map_loss, pos_loss


# --- helpers used by agent.py -------------------------------------------------
def segment_last(x, reset):
  """(B, T, ...), (B, T) -> each step carries the value at the LAST step of its
  episode segment within the batch.

  This is the hindsight target. The map mosaic only grows within an episode, so
  the last one holds everything the agent saw; grading step t against it asks
  the model to predict terrain it has not reached yet, and marks the answer once
  the agent gets there. Every label remains an observation -- just a later one.
  ``reset`` is True where a new episode begins; segments never borrow across it.
  """
  T = x.shape[1]
  shape = (-1,) + (1,) * (x.ndim - 2)
  outs = [None] * T
  outs[T - 1] = x[:, T - 1]
  for t in range(T - 2, -1, -1):
    outs[t] = jnp.where(reset[:, t + 1].reshape(shape), x[:, t], outs[t + 1])
  return jnp.stack(outs, 1)


def crop_egocentric(map12, cells, size=9):
  """(B, C, C, P), (B, S) -> (B, S, size, size, P) windows centred on the agent.

  Egocentric on purpose. With the agent at the centre, *position is the
  meaning*: "stone up-left" reads straight off the layout, so the actor never
  has to learn a coordinate transform -- and equally, no convolution is wanted
  downstream, since translation equivariance would deliberately blur "above me"
  into "below me" (design SS4).
  """
  B, C = map12.shape[0], map12.shape[1]
  r = size // 2
  # Pad by r so a cell at the edge still yields a full window; the pad offset
  # then cancels the -r of the crop origin, so cy/cx index the padded array
  # directly.
  pad = jnp.pad(map12, ((0, 0), (r, r), (r, r), (0, 0)))
  cy, cx = cells // C, cells % C                          # (B, S)
  rows = cy[:, :, None] + jnp.arange(size)[None, None, :]  # (B, S, size)
  cols = cx[:, :, None] + jnp.arange(size)[None, None, :]
  b = jnp.arange(B)[:, None, None, None]
  return pad[b, rows[:, :, :, None], cols[:, :, None, :]]


# Movement actions -> (dy, dx). Craftax Action: LEFT=1 RIGHT=2 UP=3 DOWN=4.
# numpy, not jnp: a module-level jnp.array lands on device at import time and
# then trips the transfer guard when traced as a constant (cf. 70a89b8).
_DELTAS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], np.int32)


def dead_reckon(cell0, actions, coarse, cell_tiles, shift=True):
  """(N,), (N, S) -> (N, S) coarse cells visited by an imagined action sequence.

  Imagination has no observations, so RSSM-2 cannot legitimately tick: ticking
  it on imagined features would let the agent invent terrain and then plan to
  harvest it. Instead the map is FROZEN and only the crop window moves -- which
  is also the honest semantics, since imagining does not teach you geography.

  Positions are integrated in tiles and rounded to the nearest cell, assuming
  the agent starts mid-cell (it has no sub-cell information to do better).
  """
  cy0, cx0 = cell0 // coarse, cell0 % coarse
  if not shift:
    S = actions.shape[1]
    return jnp.repeat(cell0[:, None], S, 1)
  deltas = action_deltas(actions)                       # (N, S, 2) tiles
  tiles = jnp.cumsum(deltas, 1) - deltas                # offset BEFORE the step
  cells = (tiles + cell_tiles // 2) // cell_tiles       # -> nearest cell
  cy = jnp.clip(cy0[:, None] + cells[..., 0], 0, coarse - 1)
  cx = jnp.clip(cx0[:, None] + cells[..., 1], 0, coarse - 1)
  return cy * coarse + cx


def action_deltas(actions):
  """Map discrete actions to (dy, dx); non-movement actions contribute zero.

  Used both for RSSM-2's tick input and for dead-reckoning the agent's position
  through imagination, where the map is frozen but the crop must still slide.
  Only LEFT/RIGHT/UP/DOWN (1-4) move. This once clipped the index to 0..4,
  which sent DO, SLEEP and every crafting action (5-42) to DOWN.
  """
  a = actions.astype(jnp.int32)
  idx = jnp.where((a >= 1) & (a <= 4), a, 0)
  return jnp.asarray(_DELTAS)[idx]
