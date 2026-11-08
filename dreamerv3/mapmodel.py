"""RSSM-2 -- the slow recurrent model, used here as a memory.

A second, coarser recurrent model that ticks once per ``tick`` env steps on
the window's mean RSSM-1 latent and the actions taken in it. On this branch it
is trained with no game labels (agent.mapmodel.target memory): heads on its
state predict RSSM-1's latent a few ticks ahead and recall the observation a
few ticks back (agent.py, ``memfut`` / ``memrec``). The actor, the critic and
the manager read its state through a learned scalar gate.

Deterministic, not stochastic: a GRU is enough, with no KL and no sampling.
The module keeps the name ``mapmodel`` (and the config key) from the map
model it grew out of, so checkpoints and configs of the runs that used it
still load; the map decoder and the map labels live on family/rssm2.

Gradient isolation is enforced by callers, not here: ``agent.py`` stops
gradients on ``feat1`` on the way in (so RSSM-2's loss never reaches RSSM-1)
and on the state on the way out (so policy gradients never reach RSSM-2).
"""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax.nets as nn

f32 = jnp.float32


class MapModel(nj.Module):
  """Slow GRU over aggregated RSSM-1 features."""

  deter: int = 1024
  hidden: int = 512
  layers: int = 2
  act: str = 'silu'
  norm: str = 'rms'

  def __init__(self, **kw):
    self.kw = dict(**kw)

  # --- state ----------------------------------------------------------------
  def initial(self, bsize):
    return nn.cast(dict(deter2=jnp.zeros([bsize, self.deter], f32)))

  @property
  def entry_space(self):
    """Stored in the replay context so a mid-episode chunk resumes RSSM-2.

    Without this, a 64-step batch would only ever give RSSM-2 eight ticks.
    count is the step within the current window, so a chunk that resumes
    mid-window closes its first window on the same step the actor did.
    """
    return dict(deter2=elements.Space(np.float32, (self.deter,)),
                count=elements.Space(np.float32, (1,)))

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

  def gate(self):
    """Scalar the actor's RSSM-2 input is multiplied by. Learned, initialised 0.

    At 0 the actor ignores RSSM-2, so an untrained RSSM-2 cannot poison the
    policy while everything trains at once; the gate still receives gradient
    (d loss/d gate is non-zero even at 0), so the channel opens by itself as
    fast as it starts paying. Logged as ``memory/gate``.
    """
    return self.value('gate', jnp.zeros, (), f32)
