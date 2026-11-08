"""Taste over actions: a fast, per-action prior on the actor.

Ported from the plastic-reward toy (../plastic-reward, docs/results-buttons.md
there), where a per-action taste added to the policy's logits helped in all
11 test worlds (+11 to +37 lifespan) and beat its evidence-shuffled control in
15-16 of 16 paired lifetimes. Lessons carried over from the toy:
  - taste is a logit BIAS, never a reward: a reward on a free action is
    farmed (the toy agent spammed a useless button and starved);
  - its evidence is the critic's ADVANTAGE at the steps where the action was
    really taken, so each state is its own baseline: a window-return-vs-quiet
    baseline made actions pressed in safe moments look good;
  - while evidence is thin the policy sees `optimism`, not the taste, so an
    avoided action drifts back and gets retried.

The rule's constants are the evolved champion `btn2adv` (birth taste +0.25,
good evidence weighted 3x bad, fast forgetting, loudness ~5), transplanted
unit for unit:
  time   the toy revised taste once per 4096 fresh env steps, 300 times a
         life. Here a fold every `every` train updates with `fresh` new env
         steps between folds (2000 updates = 4000 env steps at train_ratio 512
         with 16x64 batches), ~275 folds in a 1.1M-step run.
  counts replay samples each env step many times, so counts are re-spread:
         the window's `fresh` steps are shared across actions in proportion
         to how often each was sampled.
  scale  the toy judged q = conf * eff / 32 with eff in steps of survival and
         a per-step advantage spread of ~9.2 steps; here eff is in units of
         the running per-step advantage spread, times zscale = 9.2 / 32.

The slots need not be actions. With `agent.taste.target manager` they are
the manager's goals: the evidence is the real return after the manager chose
a goal, and the taste is an internal reward on that choice (agent.py,
_mgr_loss_shared) rather than a logit bias.

`center` compares each slot's evidence with the average over all slots in
the fold, the baseline of the research proposal. Without it v2-taste's
advantage averaged below zero for every action, so every press counted a
little against its button and the most-pressed ones were liked least
(docs/taste-prior.md, Result).
"""

import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
i32 = jnp.int32


class Taste(nj.Module):

  n_act: int = 43
  basic: tuple = ()        # actions taste never touches (actions='special')
  actions: str = 'special'
  every: int = 2000        # train updates per fold of evidence into taste
  fresh: float = 4000.0    # new env steps per fold
  zscale: float = 0.288
  w0: float = 0.25         # taste at birth, and where it relaxes back to
  alpha: float = 0.41      # how fast evidence moves taste
  rho: float = 0.014       # pull back toward w0
  pos: float = 4.45        # weight of good evidence
  neg: float = 1.43        # weight of bad evidence
  decay: float = 0.40      # per fold, how much old evidence is kept
  lam: float = 4.92        # logits per unit of taste (target actions)
  lam_mgr: float = 1.0     # retnorm scales per unit of taste (target manager)
  rows: int = 256          # replay rows read per update (target manager)
  optimism: float = 0.0    # what the policy sees where evidence has faded
  n0: float = 20.0         # evidence count at which confidence is 1/2
  wmax: float = 1.0
  stdrate: float = 0.01    # EMA rate of the per-step advantage spread
  center: bool = False     # evidence relative to the fold's average slot
  target: str = 'actions'  # read by agent.py: where the taste acts

  def __init__(self):
    assert self.actions in ('special', 'all'), self.actions
    assert self.target in ('actions', 'manager'), self.target
    A = self.n_act
    zeros = lambda: jnp.zeros((A,), f32)
    self.w = nj.Variable(lambda: jnp.full((A,), self.w0, f32), name='w')
    self.n = nj.Variable(zeros, name='n')
    self.eff = nj.Variable(zeros, name='eff')
    self.acc_sum = nj.Variable(zeros, name='acc_sum')
    self.acc_cnt = nj.Variable(zeros, name='acc_cnt')
    # Every step sampled since the last fold, with evidence or not: slots
    # filled only at some steps (manager decisions) get counts in proportion.
    # Over actions every step carries evidence, so the count of evidence is
    # the count of steps and the variable is left out: v2-taste checkpoints
    # keep loading.
    self._sparse = self.target == 'manager'
    if self._sparse:
      self.acc_steps = nj.Variable(jnp.zeros, (), f32, name='acc_steps')
    # Running mean, mean square and debias weight of the per-step advantage.
    self.moms = nj.Variable(jnp.zeros, (3,), f32, name='moms')
    self.tick = nj.Variable(jnp.zeros, (), i32, name='tick')

  def mask(self):
    m = jnp.ones((self.n_act,), f32)
    if self.actions == 'special' and self.basic:
      m = m.at[jnp.asarray(self.basic)].set(0.0)
    return m

  def shown(self):
    """Taste where evidence is fresh, `optimism` where it has faded."""
    n = self.n.read()
    conf = n / (n + self.n0)
    return conf * self.w.read() + (1.0 - conf) * self.optimism

  def bias(self):
    """(n_act,) logits added to the actor's; zero on untouched actions."""
    return self.lam * self.shown() * self.mask()

  def relative(self):
    """(n_act,) shown taste minus its mean over the tasted slots; 0 untouched.

    The manager's advantage is shift-invariant across all its goals, so a
    level shared by the tasted goals would only act against the untasted one
    (NONE): optimism 0.25 alone would be a permanent handicap on NONE larger
    than the game's own Q spread. Only how a goal compares with the others
    may move the manager.
    """
    m = self.mask()
    shown = self.shown()
    mean = (shown * m).sum() / jnp.maximum(m.sum(), 1.0)
    return (shown - mean) * m

  def observe(self, adv, act, weight, steps=None):
    """Fold one replay batch of advantages into the evidence.

    ``adv`` (B, T) external-reward advantage of each real step (lambda-return
    minus the critic's value), ``act`` (B, T) the action (or goal) taken
    there, ``weight`` (B, T) 1 where the step carries evidence. ``steps``
    (target manager only) is how many replay steps the batch covers, so a
    slot's count is its share of the fresh env steps. Every `every` calls the
    accumulated evidence is folded into memory and taste moves (the toy's
    reward.update). Returns metrics.
    """
    A = self.n_act
    adv, weight = f32(adv), f32(weight)
    assert (steps is not None) == self._sparse, (steps, self.target)
    total = weight.sum()
    has = total > 0
    norm = jnp.maximum(total, 1.0)
    batch = jnp.stack([(adv * weight).sum() / norm,
                       (adv ** 2 * weight).sum() / norm, 1.0])
    moms = (1 - self.stdrate) * self.moms.read() + self.stdrate * batch
    moms = jnp.where(has, moms, self.moms.read())
    self.moms.write(moms)
    mean, sq = moms[0] / jnp.maximum(moms[2], 1e-8), moms[1] / jnp.maximum(
        moms[2], 1e-8)
    std = jnp.sqrt(jnp.maximum(sq - mean ** 2, 1e-8))

    onehot = jax.nn.one_hot(act, A, dtype=f32) * weight[..., None]
    acc_sum = self.acc_sum.read() + (onehot * (adv / std)[..., None]).sum((0, 1))
    acc_cnt = self.acc_cnt.read() + onehot.sum((0, 1))
    if self._sparse:
      acc_steps = self.acc_steps.read() + f32(steps)
    else:
      acc_steps = acc_cnt.sum()
    tick = self.tick.read() + 1
    self.tick.write(tick)

    # Fold: the window's fresh env steps, shared by how often each was sampled.
    count = acc_cnt / jnp.maximum(acc_steps, 1.0) * self.fresh
    seen = acc_sum / jnp.maximum(acc_cnt, 1e-6)
    if self.center:
      seen -= acc_sum.sum() / jnp.maximum(acc_cnt.sum(), 1e-6)
    n_old = self.decay * self.n.read()
    n = n_old + count
    eff = jnp.where(n > 0, (n_old * self.eff.read() + count * seen)
                    / jnp.maximum(n, 1e-6), self.eff.read())
    q = n / (n + self.n0) * eff * self.zscale
    w = self.w.read()
    dw = (self.alpha * (self.pos * jax.nn.relu(q) - self.neg * jax.nn.relu(-q))
          - self.rho * (w - self.w0))
    w = jnp.clip(w + dw, -self.wmax, self.wmax)

    fold = tick % self.every == 0
    keep = lambda new, old: jnp.where(fold, new, old)
    self.w.write(keep(w, self.w.read()))
    self.n.write(keep(n, self.n.read()))
    self.eff.write(keep(eff, self.eff.read()))
    self.acc_sum.write(keep(jnp.zeros_like(acc_sum), acc_sum))
    self.acc_cnt.write(keep(jnp.zeros_like(acc_cnt), acc_cnt))
    if self._sparse:
      self.acc_steps.write(keep(jnp.zeros_like(acc_steps), acc_steps))

    m = self.mask()
    shown = self.shown()
    k = jnp.maximum(m.sum(), 1.0)
    return {
        'adv_std': std,
        'shown_mean': (shown * m).sum() / k,
        'shown_max': jnp.where(m > 0, shown, -jnp.inf).max(),
        'shown_min': jnp.where(m > 0, shown, jnp.inf).min(),
        'liked': ((shown > 0) * m).sum() / k,
        'conf_mean': ((self.n.read() / (self.n.read() + self.n0)) * m).sum() / k,
    }
