"""RSSM-2 -- the slow map model.

A second, coarser world model that ticks once per ``tick`` env steps and
predicts a 12x12x13 map of the whole level plus the agent's own coarse cell.
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

import einops
import jax
import jax.numpy as jnp
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
  planes: int = 13
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
    return dict(deter2=elements.Space(np.float32, (self.deter,)))

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

  def observe(self, carry, ticks, reset):
    """Roll the map model over T2 ticks.

    ``ticks``  (B, T2, F)  aggregated RSSM-1 features + movement per window
    ``reset``  (B, T2)     episode/level boundaries -- the map does not survive
                           either, since each Craftax level is its own 48x48 map
    """
    deter = carry['deter2']
    outs = []
    for t in range(ticks.shape[1]):
      deter = nn.mask(deter, ~reset[:, t])
      deter = self._gru(deter, ticks[:, t])
      outs.append(deter)
    deter2 = jnp.stack(outs, 1)
    return dict(deter2=deter), dict(deter2=deter2), deter2

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

  # --- losses ---------------------------------------------------------------
  def loss(self, deter2, map_target, pos_target):
    """BCE over every cell (seen AND unseen) + cross-entropy over position.

    Supervising unseen cells is deliberate: masking to visited cells would mean
    the model never learns to predict the unseen, and prediction of the unseen
    is the whole point. One loss teaches both behaviours -- cells with evidence
    in the history get recalled, cells without get generated from the prior.
    """
    mlogit, plogit = self.decode(deter2)
    mlogit, plogit = f32(mlogit), f32(plogit)   # reduce in f32, not bf16
    tgt = f32(map_target)
    bce = jnp.maximum(mlogit, 0) - mlogit * tgt + jnp.log1p(jnp.exp(-jnp.abs(mlogit)))
    map_loss = bce.sum((-1, -2, -3))
    onehot = jax.nn.one_hot(pos_target.astype(jnp.int32), self.cells)
    pos_loss = -(onehot * jax.nn.log_softmax(plogit, -1)).sum(-1)
    return map_loss, pos_loss


# --- helpers used by agent.py -------------------------------------------------
def aggregate(feat1, moves, tick):
  """(B, T, F), (B, T, 2) -> (B, T2, F + 2 + 1) tick inputs, T2 = T // tick.

  Mean-pools RSSM-1 features over each window and sums the movement deltas.
  ``tick`` must divide ``batch_length`` (8 divides 64) so the reshape is exact.
  """
  B, T = feat1.shape[0], feat1.shape[1]
  assert T % tick == 0, (T, tick)
  T2 = T // tick
  feat = einops.reduce(feat1[:, :T2 * tick], 'b (t k) f -> b t f', 'mean', k=tick)
  move = einops.reduce(moves[:, :T2 * tick], 'b (t k) d -> b t d', 'sum', k=tick)
  steps = jnp.full((B, T2, 1), tick, f32)
  return nn.cast(jnp.concatenate([f32(feat), f32(move), steps], -1))


def last_of_window(x, tick):
  """(B, T, ...) -> (B, T2, ...) taking the final entry of each tick window."""
  T2 = x.shape[1] // tick
  return x[:, tick - 1: T2 * tick: tick]


def repeat_ticks(x, tick, T):
  """(B, T2, ...) -> (B, T, ...): hold each tick's value until the next one."""
  out = jnp.repeat(x, tick, axis=1)
  return out[:, :T]


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
_DELTAS = jnp.array(
    [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], jnp.int32)


def action_deltas(actions):
  """Map discrete actions to (dy, dx); non-movement actions contribute zero.

  Used both for RSSM-2's tick input and for dead-reckoning the agent's position
  through imagination, where the map is frozen but the crop must still slide.
  """
  idx = jnp.clip(actions.astype(jnp.int32), 0, 4)
  return _DELTAS[idx]
