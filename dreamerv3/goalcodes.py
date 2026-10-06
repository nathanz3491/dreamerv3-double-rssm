"""Learned goals for the manager (v2): a codebook of the changes the agent causes.

The v1 managers chose from 13 hand-written tech-tree milestones -- game
knowledge built into the architecture. Here the goals are learned instead,
the way B3 learns which actions do anything: from the agent's own experience,
with no list written by hand.

  change   delta = feat(t + window) - feat(t), how RSSM-1's latent moved over a
           window of steps, taken from replay (stop-gradient, so the codebook
           never reshapes the world model)
  encoder  delta -> 64 numbers on the unit sphere (GoalNet.encode)
  codes    16 unit vectors (GoalBook); each encoded change snaps to its
           nearest code, as in a VQ-VAE. Distinctive events -- "got wood",
           "placed something", "drank" -- should each claim a code without
           ever being named.
  decoder  quantised code -> delta, so the encoding has to keep what made the
           change distinctive (otherwise the encoder could map everything to
           one point and the codes would mean nothing)

Codes move by exponential moving average of the encodings assigned to them,
not by gradient, and a code that stops being used is moved onto a random
recent change, so all 16 stay alive. That state lives in GoalBook, which is
updated in the loss like the return normalisers and is deliberately not in
the optimizer's module list -- the optimizer would otherwise fight the
moving-average writes.

At use time a goal is "reached" when the change since the goal was set is
classified as the goal's code: that code is its nearest one (see progress).
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax.nets as nn

f32 = jnp.float32
sg = jax.lax.stop_gradient


def _unit(x):
  return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


class GoalNet(nj.Module):
  """Encoder and decoder between latent changes and the code space."""

  dim: int = 64
  units: int = 256

  def __init__(self, width):
    self.width = int(width)

  def encode(self, delta):
    x = nn.cast(delta)
    x = self.sub('enc0', nn.Linear, self.units)(x)
    x = nn.act('silu')(self.sub('enc0norm', nn.Norm, 'rms')(x))
    x = self.sub('enc1', nn.Linear, self.dim)(x)
    return _unit(f32(x))

  def decode(self, z):
    x = nn.cast(z)
    x = self.sub('dec0', nn.Linear, self.units)(x)
    x = nn.act('silu')(self.sub('dec0norm', nn.Norm, 'rms')(x))
    return f32(self.sub('dec1', nn.Linear, self.width)(x))


class GoalBook(nj.Module):
  """The codes themselves, moved by moving average rather than by gradient."""

  codes: int = 16
  dim: int = 64
  decay: float = 0.99
  dead: float = 0.02      # a code holding under this share of use is reset

  def __init__(self):
    rng = np.random.RandomState(0)
    init = rng.randn(self.codes, self.dim).astype(np.float32)
    init /= np.linalg.norm(init, axis=-1, keepdims=True)
    share = np.full((self.codes,), 1.0 / self.codes, np.float32)
    self.emb = nj.Variable(lambda: jnp.asarray(init), name='emb')
    self.share = nj.Variable(lambda: jnp.asarray(share), name='share')
    self.total = nj.Variable(lambda: jnp.asarray(init * share[:, None]),
                             name='total')

  def read(self):
    return self.emb.read()

  def update(self, z, idx, valid):
    """EMA towards the encodings assigned to each code; revive dead codes."""
    onehot = jax.nn.one_hot(idx, self.codes) * valid[:, None]
    n = jnp.maximum(valid.sum(), 1.0)
    share = self.decay * self.share.read() + (1 - self.decay) * onehot.sum(0) / n
    total = self.decay * self.total.read() + (1 - self.decay) * (
        onehot.T @ sg(z)) / n
    emb = _unit(total / jnp.maximum(share, 1e-6)[:, None])
    # Dead codes jump onto a random valid recent change.
    probs = valid / jnp.maximum(valid.sum(), 1.0)
    pick = jax.random.choice(nj.seed(), z.shape[0], (self.codes,), p=probs)
    dead = share < self.dead
    fresh = sg(z)[pick]
    emb = jnp.where(dead[:, None], fresh, emb)
    share = jnp.where(dead, 1.0 / self.codes, share)
    total = jnp.where(dead[:, None], fresh / self.codes, total)
    self.emb.write(emb)
    self.share.write(share)
    self.total.write(total)
    return dead.sum()


def vq_loss(net, book, delta, valid, beta, update):
  """(N, W) changes, (N,) mask -> (N,) loss and metrics. Updates the book."""
  z = net.encode(delta)
  emb = book.read()
  sims = z @ emb.T
  idx = jnp.argmax(sims, -1)
  q = emb[idx]
  recon = net.decode(z + sg(q - z))              # straight-through
  rec = ((recon - sg(f32(delta))) ** 2).mean(-1)
  commit = beta * ((z - sg(q)) ** 2).sum(-1)
  loss = (rec + commit) * valid
  mets = {}
  if update:
    mets['revived'] = book.update(z, idx, valid)
  use = (jax.nn.one_hot(idx, emb.shape[0]) * valid[:, None]).sum(0)
  use = use / jnp.maximum(use.sum(), 1.0)
  mets['perplexity'] = jnp.exp(-(use * jnp.log(use + 1e-8)).sum())
  mets['recon'] = (rec * valid).sum() / jnp.maximum(valid.sum(), 1.0)
  # How closely real changes match their nearest code: the yardstick for the
  # manager's code_reach threshold.
  mets['match'] = (sims.max(-1) * valid).sum() / jnp.maximum(valid.sum(), 1.0)
  return loss, mets


def progress(net, book, delta, goal, floor):
  """How far a change has gone towards a goal code, and whether it got there.

  (..., W) changes, (...) goals -> (progress, reached). Progress is the cosine
  between the encoded change and the goal's code. Reached means the change is
  CLASSIFIED as that code -- its nearest code is the goal -- and is not a
  near-zero change that happens to lean that way (cosine above `floor`). A
  fixed cosine threshold had to be calibrated: in the first debug run real
  changes matched their nearest code at only ~0.38.
  """
  z = net.encode(delta)
  sims = z @ book.read().T                                    # (..., K)
  prog = jnp.take_along_axis(sims, goal[..., None], -1)[..., 0]
  reached = (jnp.argmax(sims, -1) == goal) & (prog > floor)
  return prog, reached
