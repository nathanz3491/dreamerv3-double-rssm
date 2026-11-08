"""RSSM-2 -- the slow second world model, used here as a memory.

A deterministic GRU that ticks once per ``tick`` env steps on RSSM-1's mean
latent over the window plus the window's raw actions (agent.py, _map_step).
On this branch it is trained with no labels at all (agent.py, target
'memory'): from its state, memfut predicts RSSM-1's latent a few ticks ahead
and memrec recalls the observation a few ticks back. The actor and the
manager read its state through a learned gate.

It is still called the map model, and its params still live under
``mapmodel/``, because it began as one: on family/rssm2 the same GRU decodes a
coarse map of the level. The names are kept so old checkpoints load.

Deterministic, not stochastic: RSSM-1 needs a stochastic latent because it
models one-step dynamics under uncertainty; a slow summary of the episode so
far does not, so a GRU is enough -- no KL, no sampling.

Gradient isolation is enforced by callers, not here: ``agent.py``
stop-gradients ``feat1`` on the way in (so RSSM-2's loss never reaches RSSM-1)
and the state on the way out (so policy gradients never reach RSSM-2).
"""

import jax
import jax.numpy as jnp
import ninjax as nj

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

    Without this, a 64-step batch would only ever give RSSM-2 eight ticks --
    far too few for a memory that spans hundreds of steps.
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

  def gate(self):
    """Scalar the actor's RSSM-2 input is multiplied by. Learned, initialised 0.

    This replaces a hand-tuned warm-up schedule. At 0 the actor ignores
    RSSM-2, so an untrained memory cannot poison the policy while everything
    trains at once; the gate still receives gradient (d loss/d gate is
    non-zero even at 0), so the channel opens by itself exactly as fast as it
    starts paying. Logged as ``memory/gate``.
    """
    return self.value('gate', jnp.zeros, (), f32)
