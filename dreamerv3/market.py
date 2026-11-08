"""The market: sensing, curves and the training return (docs/design-market.md §3).

The price P* is where a supply curve and a demand curve for survival cross.
Both are log-linear (constant elasticity):

  supply   log Q = log A_s + eps_s log P
  demand   log Q = log A_d - eps_d log P
  cross    log P* = (log A_d - log A_s) / (eps_s + eps_d)

The baselines A_s, A_d are constants; a small network moves them. Every step it
reads how the observation changed over the last ``k`` steps and outputs the two
shifts, d log A_s and d log A_d, each through ``bound x tanh`` after a dead zone
(so that noise in the observation does not move the price).

Nothing here knows the game's rules. The observation is only split along its
layout (craftax_map: a 9x11 grid of tiles, then the scalar block), the way the
encoder sees it:

  non-view  the scalar block (vitals, inventory, light, ...): subtracted
            directly, x_t - x_{t-k};
  view      the local tile grid, centred on the agent: a raw difference would
            mostly measure "I walked", so it is one number, 1 - cos(view_t,
            view_{t-k}).

Pure numpy; every function takes ``xp`` (numpy or jax.numpy) so the tests run
without JAX and the agent runs jitted.
"""

import numpy as np

from dreamerv3 import craftax_map as M

VIEW = M.OBS_H * M.OBS_W * M.N_TILE      # length of the tile grid in the vector
N_NONVIEW = M.OBS_LEN - VIEW             # the scalar block that follows it
N_FEAT = N_NONVIEW + 1                   # non-view differences + 1 - cos(view)
EPS = 1e-6


def delta_env(vec, old, xp=np):
  """(..., OBS_LEN) now and k steps ago -> (..., N_FEAT) the change."""
  assert vec.shape[-1] == M.OBS_LEN, vec.shape
  nonview = vec[..., VIEW:] - old[..., VIEW:]
  now, then = vec[..., :VIEW], old[..., :VIEW]
  dot = (now * then).sum(-1)
  den = xp.sqrt((now * now).sum(-1) * (then * then).sum(-1))
  # Two empty views are the same view; one empty view shares nothing.
  both = ((now * now).sum(-1) + (then * then).sum(-1)) <= EPS
  cos = xp.where(den > EPS, dot / xp.maximum(den, EPS),
                 xp.where(both, 1.0, 0.0))
  return xp.concatenate([nonview, (1.0 - cos)[..., None]], -1)


def deadzone(x, dead, xp=np):
  """Zero within +-dead, shrunk toward zero by dead outside."""
  return xp.sign(x) * xp.maximum(xp.abs(x) - dead, 0.0)


def shifts(z, bound, dead, xp=np):
  """(..., 2) the network's draw -> (d log A_s, d log A_d), each in
  (-bound, bound), exactly 0 inside the dead zone."""
  return bound * xp.tanh(deadzone(z, dead, xp))


def log_baselines(price0, eps_s, eps_d, a_s=1.0, a_d=0.0):
  """(log A_s, log A_d). a_d <= 0 derives the demand baseline so that the
  curves cross at price0 with no shift."""
  log_as = float(np.log(a_s))
  log_ad = float(np.log(a_d)) if a_d > 0 else (
      log_as + (eps_s + eps_d) * float(np.log(price0)))
  return log_as, log_ad


def equilibrium(dls, dld, price0, eps_s, eps_d, a_s=1.0, a_d=0.0, xp=np):
  """Shifts (d log A_s, d log A_d) -> (log P*, log Q*)."""
  log_as, log_ad = log_baselines(price0, eps_s, eps_d, a_s, a_d)
  logp = ((log_ad + dld) - (log_as + dls)) / (eps_s + eps_d)
  logq = log_as + dls + eps_s * logp
  return logp, logq


def gauss_logp(z, mu, sigma, xp=np):
  """(..., 2) draw under N(mu, sigma^2): the log-probability of the pair."""
  d = (z - mu) / sigma
  return (-0.5 * d * d - xp.log(sigma) - 0.5 * np.log(2 * np.pi)).sum(-1)


def window_returns(ach, first, window, xp=np):
  """The REINFORCE return of the shifts drawn at each step.

  ach (B, T) achievements unlocked at each step, first (B, T) episode starts.
  Returns (R, ok), both (B, T): R[t] = achievements at steps t+1 .. t+window,
  ok[t] whether that whole window lies inside the sequence and inside one
  episode (a window that runs into the next episode, or off the end, is not
  scored).
  """
  T = ach.shape[1]
  assert 0 < window < T, (window, T)
  cs = xp.cumsum(ach, 1)
  ep = xp.cumsum(first.astype(ach.dtype), 1)
  pad = ((0, 0), (0, window))
  ret = cs[:, window:] - cs[:, :-window]
  ok = ep[:, window:] == ep[:, :-window]
  return xp.pad(ret, pad), xp.pad(ok, pad)

