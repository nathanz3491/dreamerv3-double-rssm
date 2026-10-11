import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import craftax_map as cmap
from . import mapmodel as mapmod
from . import rssm
from . import craftax_features as cf
from . import craftax_valid as cvalid
from . import craftax_goals as cgoals
from . import goalcodes
from embodied.jax import outs as jouts

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)


def goal_start(tpos, gphase):
  """Where in the replay window the held goal was set, and its step count.

  ``tpos`` (1, K) window positions, ``gphase`` (B, K) steps since the goal was
  set. A goal set before the window began restarts at position 0 with the
  count measured from there, so start state and count always agree.
  """
  src = jnp.maximum(tpos - gphase, 0)
  return src, (tpos - src).astype(gphase.dtype)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Counter(nj.Module):
  """A step counter kept with the parameters, so acting can read it too."""

  def __init__(self):
    self.n = nj.Variable(jnp.zeros, (), i32, name='n')

  def read(self):
    return self.n.read()

  def inc(self):
    self.n.write(self.n.read() + 1)


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    # 'ach' is the reward-cause LABEL (multi-hot of newly-unlocked
    # achievements); it must not be fed to the encoder or reconstructed.
    # 'map12'/'mappos'/'mapseen'/'mapknown' are RSSM-2 TARGETS -- like 'ach'
    # they are supervision only and must never reach the encoder or decoder.
    # 'valid' (experiment B) is an INPUT only when valid.input is set; with
    # valid.mask alone it is the label for the feasibility head, like 'ach'.
    # 'goalphi' (two-level agent) is the goal-progress head's target, likewise;
    # 'goalreach' is read by the manager alone, never the encoder.
    exclude = ('is_first', 'is_last', 'is_terminal', 'reward', 'ach',
               'map12', 'mappos', 'mapseen', 'mapknown', 'goalphi',
               'goalreach')
    if not config.valid.input:
      exclude += ('valid',)
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    # --- map model (RSSM-2) ---------------------------------------------------
    # A second, slower world model holding a coarse map of the level. Off by
    # default: with mapmodel.enabled False this file behaves exactly as before.
    # v2: target 'memory' trains RSSM-2 with no map labels at all -- it
    # predicts RSSM-1's latent a few ticks ahead and recalls the observation a
    # few ticks back -- and is fed the raw actions instead of hand-computed
    # movement vectors. The actor then reads RSSM-2's state directly instead
    # of a map crop. No env targets are needed.
    self._map_memory = config.mapmodel.target == 'memory'
    assert config.mapmodel.target in ('map', 'memory'), config.mapmodel.target
    self._use_map = bool(config.mapmodel.enabled) and (
        self._map_memory or 'map12' in obs_space)
    self._n_act = int(act_space['action'].high)
    if self._use_map:
      mm = config.mapmodel
      self._map_tick = int(mm.tick)
      self._map_crop = int(mm.crop)
      self._map_coarse = int(mm.coarse)
      self._map_to_actor = bool(mm.to_actor)
      self._map_imag_shift = bool(mm.imag_shift)
      # 48x48 level over a 12x12 grid -> 4 tiles per cell. Needed to turn
      # imagined tile-sized steps into cell-sized ones when sliding the crop.
      self._map_cell_tiles = int(cmap.MAP_SIZE // mm.coarse)
      r = config.dyn[config.dyn.typ]
      self._feat1_dim = int(r.deter) + int(r.stoch) * int(r.classes)
      self.mapmodel = mapmod.MapModel(
          deter=int(mm.deter), hidden=int(mm.hidden), layers=int(mm.layers),
          coarse=int(mm.coarse), planes=int(mm.planes), name='mapmodel')
      if self._map_memory:
        assert not self._map_imag_shift, 'memory mode has no crop to slide'
        assert 'vector' in obs_space, 'memory recall reconstructs obs vector'
        self._mem_ahead = [int(h) for h in mm.ahead]
        self._mem_back = [int(h) for h in mm.back]
        fspace = elements.Space(np.float32, (self._feat1_dim,))
        vspace = elements.Space(np.float32, obs_space['vector'].shape)
        self.memfut = embodied.jax.MLPHead(
            {f'h{h}': fspace for h in self._mem_ahead},
            {f'h{h}': 'mse' for h in self._mem_ahead},
            **config.memhead, name='memfut')
        self.memrec = embodied.jax.MLPHead(
            {f'h{h}': vspace for h in self._mem_back},
            {f'h{h}': 'mse' for h in self._mem_back},
            **config.memhead, name='memrec')

    def actor2tensor(x, mapfeat=None):
      # Actor-only path. mapfeat deliberately does NOT go through feat2tensor:
      # that tensor also feeds 'rew' and 'con', and letting the map into the
      # world model would make a map error corrupt imagined reward -- and so
      # every plan built on it.
      base = self.feat2tensor(x)
      if mapfeat is None:
        return base
      mapfeat = nn.cast(mapfeat)
      if mapfeat.ndim == base.ndim - 1:       # one crop shared across the axis
        mapfeat = jnp.repeat(mapfeat[:, None], base.shape[-2], -2)
      return jnp.concatenate([base, mapfeat], -1)
    self.actor2tensor = actor2tensor

    def mapfeat(deter2, cells=None):
      """(N, D2), (N, S) -> (N, S, crop*crop*planes) actor-side map feature.

      ``cells`` are the coarse cells to centre each crop on; pass None to use
      RSSM-2's own predicted position. Also returns that prediction, which the
      imagination path needs as the origin to dead-reckon from.

      sg on the way OUT, mirroring the sg on feat1 going in: policy gradients
      must never reach RSSM-2, or the map stops being a description of the world
      and becomes whatever raises return this batch. The crop itself is pure
      indexing, so there is no parameter between them to train -- only the
      scalar gate, which is deliberately outside the sg.
      """
      if self._map_memory:
        # v2: RSSM-2's own state is the actor's memory input, gated as the
        # crop was. There is no position, so nothing to dead-reckon from.
        return nn.cast(sg(f32(deter2))[:, None] * self.mapmodel.gate()), None
      mlogit, plogit = self.mapmodel.decode(deter2)
      prob = jax.nn.sigmoid(f32(mlogit))                 # (N, C, C, P)
      pred = sg(jnp.argmax(f32(plogit), -1))             # (N,)
      cells = pred[:, None] if cells is None else cells
      crop = mapmod.crop_egocentric(prob, cells, self._map_crop)
      flat = crop.reshape((*crop.shape[:2], -1))         # (N, S, F)
      return nn.cast(sg(flat) * self.mapmodel.gate()), pred
    self.mapfeat = mapfeat

    # --- experiment B: what can the agent actually do? ------------------------
    # docs/entropy-and-action-suppression.md. Unmasked policy gradients push an
    # action down in every state where it does nothing, and shared weights
    # carry that into the rare states where it would work: the furnace was
    # never placed in 386 states where it could have been. With valid.mask an
    # impossible action gets zero probability, hence zero gradient, so its
    # logit is never pushed down where it cannot work.
    #   acting       mask = obs['valid'], computed from the observation alone
    #                (craftax_valid.valid_actions; exact against the game)
    #   imagination  mask = a learned head on the latent -- there is no
    #                observation inside a dream. The rollout and the loss read
    #                the same head on the same states, so they cannot disagree
    #                the way imag_shift's rollout and loss did. Acting reading
    #                the true flags instead only changes what reaches replay.
    self._use_valid = 'valid' in obs_space
    self._valid_mask = self._use_valid and bool(config.valid.mask)
    if self._valid_mask:
      n = int(act_space['action'].high)
      assert n == cvalid.N_ACTIONS, (n, cvalid.N_ACTIONS)
      self.feas = embodied.jax.MLPHead(
          elements.Space(bool, (n,), 0, 2), **config.validhead, name='feas')
      self._valid_threshold = float(config.valid.threshold)
    # B3: the same head, but its label is learned from experience instead of
    # transcribed from the game's rules, and it -- not obs['valid'] -- masks
    # the actor while acting. No game knowledge reaches the agent at test time.
    #   label    for each real step (s, a, s'): would NOOP have explained the
    #            outcome as well? evidence = KL(post' || prior(s, NOOP))
    #            - KL(post' || prior(s, a)), the world model's own judgement of
    #            how much better the action explains what really happened.
    #            Above valid.evidence nats, the action did something.
    #   partial  only the action actually taken gets a label each step.
    #   explore  a masked action is never taken, so a wrong "impossible" would
    #            never be corrected; with probability valid.explore a step
    #            ignores the mask.
    # obs['valid'] (the rules) is kept only as a measuring stick in metrics.
    self._valid_learned = self._valid_mask and bool(config.valid.learned)
    # v2: exempt_basic False learns the mask for NOOP, the moves and DO too;
    # reference 'random' drops the assumption that action 0 is NOOP.
    self._exempt_basic = bool(config.valid.exempt_basic)
    self._valid_reference = str(config.valid.reference)
    assert self._valid_reference in ('noop', 'random'), self._valid_reference
    # Fix after B3 (docs/entropy-and-action-suppression.md). B3's labels came
    # from the world model's own latent predictions while the head's loss
    # trained that same world model: it learned to pull the two predictions
    # apart, evidence climbed to 13.8 nats, 87% of "did something" labels were
    # wrong, and the run fell ~2 achievements below B2.
    #   detach  the head's loss no longer reaches the world model
    #   label   obs: compare DECODED observations, grounded in what really
    #           came next -- the action did something if the real next
    #           observation is explained clearly better after it than after
    #           the reference action: err_ref - err_real > margin nats.
    self._valid_detach = bool(config.valid.detach)
    self._valid_label = str(config.valid.label)
    assert self._valid_label in ('latent', 'obs'), self._valid_label
    self._valid_margin = float(config.valid.margin)
    #   warmup   an untrained world model judges almost nothing as having an
    #            effect (0.2-0.5% positive labels in the first few thousand
    #            steps). Masking on those labels would teach the head that
    #            everything is impossible, block nearly every action, and
    #            starve the head of the labels that could correct it. So the
    #            mask stays off -- acting and imagination both unmasked, as in
    #            vanilla -- for the first valid.warmup train updates while the
    #            head trains; at train_ratio 512 that is twice as many env
    #            steps.
    if self._valid_learned:
      self._valid_explore = float(config.valid.explore)
      self._valid_evidence = float(config.valid.evidence)
      self._valid_warmup = int(config.valid.warmup)
      self._valid_rule_mix = float(config.valid.rule_mix)
      self.feasclock = Counter(name='feasclock')

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)

    # --- two-level agent: a manager over tech-tree goals ----------------------
    # docs/design-manager.md. Every `every` steps the manager -- a second
    # actor-critic -- picks one of cgoals.GOALS from RSSM-1's state plus
    # RSSM-2's slow state; the actor sees the goal (and how far into its
    # 8-step segment it is) and is paid the change in that goal's progress on
    # top of the game reward. The manager is paid the game reward only, so it
    # learns which goal to set, not how to satisfy the goal reward.
    self._shared = False
    self._use_mgr = bool(config.manager.enabled)
    if self._use_mgr:
      assert self._use_map and self._map_to_actor, (
          'the manager reads RSSM-2: needs mapmodel.enabled and to_actor')
      # v2: goals 'learned' replaces the hand-written list with a codebook of
      # the changes the agent causes (goalcodes.py); the env emits nothing.
      self._learned_goals = config.manager.goals == 'learned'
      assert config.manager.goals in ('named', 'learned'), config.manager.goals
      if self._learned_goals:
        mc = config.manager
        self._goals = int(mc.codes)
        self._goal_names = [f'code{k:02d}' for k in range(self._goals)]
        self._code_window = int(mc.code_window)
        self._code_reach = float(mc.code_reach)
        self._code_beta = float(mc.code_beta)
        self.goalnet = goalcodes.GoalNet(
            self._feat1_dim, dim=int(mc.code_dim), name='goalnet')
        self.goalbook = goalcodes.GoalBook(
            codes=self._goals, dim=int(mc.code_dim),
            decay=float(mc.code_decay), name='goalbook')
      else:
        assert 'goalreach' in obs_space, 'needs env.craftax.goals_obs True'
        # 13 tech goals, or 15 with the v1.3 survival goals: whatever the env
        # emits progress for.
        self._goals = int(obs_space['goalphi'].shape[0])
        self._goal_names = cgoals.names(self._goals > cgoals.N_GOALS)
        assert len(self._goal_names) == self._goals, self._goals
      self._every = int(config.manager.every)
      # v1.2: one two-headed critic shared by both actors (game head for both,
      # goal head for the bottom actor only), goals held until reached.
      self._shared = config.manager.critic == 'shared'
      assert config.manager.critic in ('separate', 'shared'), (
          config.manager.critic)
      self._hold = int(config.manager.hold)
      assert self._shared == (self._hold > 0), (
          'v1.2 is critic=shared with hold > 0; v1/v1.1 is separate with 0')
      self._phases = self._hold if self._shared else self._every
      self._goal_weight = float(config.manager.goal_weight)
      self._goal_reward = float(config.manager.goal_reward)
      self._reach_bonus = float(config.manager.reach_bonus)
      self._mgr_bonus = float(config.manager.mgr_bonus)
      self._reach_threshold = float(config.manager.reach_threshold)
      self.mgr = embodied.jax.MLPHead(
          elements.Space(np.int32, (), 0, self._goals), 'categorical',
          **config.policy, name='mgr')
      if self._shared:
        # The goal stream's own return statistics: normalising it separately
        # is what lets goal_weight mean what it says.
        self.gretnorm = embodied.jax.Normalize(
            **config.retnorm, name='gretnorm')
        self.gvalnorm = embodied.jax.Normalize(
            **config.valnorm, name='gvalnorm')
        self.gadvnorm = embodied.jax.Normalize(
            **config.advnorm, name='gadvnorm')
      else:
        self.mval = embodied.jax.MLPHead(scalar, **config.value, name='mval')
        self.mslowval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name='mslowval'),
            source=self.mval, **config.slowvalue)
        self.mretnorm = embodied.jax.Normalize(
            **config.retnorm, name='mretnorm')
        self.mvalnorm = embodied.jax.Normalize(
            **config.valnorm, name='mvalnorm')
        self.madvnorm = embodied.jax.Normalize(
            **config.advnorm, name='madvnorm')
      if self._learned_goals:
        assert self._shared, 'learned goals build on v1.2 (critic shared)'
      else:
        self.gphi = embodied.jax.MLPHead(
            elements.Space(np.float32, (self._goals,)), **config.gphihead,
            name='gphi')
    # The replay value loss needs every replay step's goal, its position in the
    # segment and its goal reward, and a bootstrap from an imagination that
    # started with that same goal at that same position. v1/v1.1 imagination
    # starts every rollout on a fresh manager decision, so the two would
    # disagree and the critic learns from imagination alone. v1.2 resumes the
    # replay's goal and segment step, so its game head gets the replay loss
    # back (the goal head stays imagination-only).
    self._shared = self._use_mgr and self._shared
    self._learned_goals = self._use_mgr and self._learned_goals

    # --- v2: curiosity -----------------------------------------------------
    # Replaces the tech-tree and survival potentials. An ensemble of small
    # networks predicts the next latent from (latent, action); where they
    # disagree the world model does not understand the game yet, and the
    # disagreement is paid as reward (Plan2Explore, Simulus). It is a third
    # reward stream with its own critic head and normalisation; the bottom
    # actor and the manager both see it.
    self._curio = bool(config.curiosity.enabled)
    if self._curio:
      assert self._shared, 'curiosity uses the shared critic (manager v1.2)'
      cc = config.curiosity
      r = config.dyn[config.dyn.typ]
      self._curio_dim = int(r.stoch) * int(r.classes)
      tspace = elements.Space(np.float32, (self._curio_dim,))
      self.curio = [
          embodied.jax.MLPHead(tspace, 'mse', layers=int(cc.layers),
                               units=int(cc.units), name=f'curio{i}')
          for i in range(int(cc.members))]
      self._curio_weight = float(cc.weight)
      self._curio_mgr = float(cc.mgr_weight)
      # retlimit: the floor under the explore stream's return spread. The
      # game's floor of 1 keeps a sparse reward from being blown up into
      # noise, but curiosity returns span ~0.02, so under that floor they were
      # never rescaled and curiosity carried ~2% of the actor's signal (v2's
      # first run). normalize also puts the manager's explore value on the
      # same footing: in units of its own spread, not raw.
      self._curio_norm = bool(cc.normalize)
      eret = dict(config.retnorm)
      if self._curio_norm:
        eret['limit'] = float(cc.retlimit)
      self.eretnorm = embodied.jax.Normalize(**eret, name='eretnorm')
      self.evalnorm = embodied.jax.Normalize(**config.valnorm, name='evalnorm')
      self.eadvnorm = embodied.jax.Normalize(**config.advnorm, name='eadvnorm')

    self._repval = bool(config.repval_loss) and (
        not self._use_mgr or self._shared)

    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    # --- Phase 1: compositional reward-cause head -----------------------------
    # Predicts the factored feature vector phi of the achievement event at each
    # step (binary per-feature), so common tiers teach the model unseen ones.
    self._use_rewcause = bool(getattr(config, 'rewcause', False))
    if self._use_rewcause:
      phi_space = elements.Space(bool, (cf.PHI_DIM,), 0, 2)
      self.rewcause = embodied.jax.MLPHead(
          phi_space, **config.rewcausehead, name='rewcause')
      self._phi_table = cf.build_phi_table()          # np [A, PHI_DIM]
      self._weight_table = cf.build_weight_table()    # np [A]
      hv = np.zeros(cf.NUM_ACHIEVEMENTS, np.float32)
      for tok in str(config.rewcause_holdout).split(','):
        tok = tok.strip()
        if tok:
          idx = int(tok) if tok.isdigit() else cf.ACHIEVEMENT_NAMES.index(tok)
          hv[idx] = 1.0
      self._holdout_vec = hv
      self._rewcause_none_weight = float(config.rewcause_none_weight)

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    if self._shared:
      # One critic body, two heads: 'game' (achievements, potential, health)
      # judges both actors; 'goal' (progress + reach bonus) only the bottom
      # actor. One scalar could not do both: holding the goal reward, it
      # would pay the manager for its own goals; without it, nothing would
      # tie the bottom actor to the goal.
      vkw = dict(config.value)
      vout = vkw.pop('output')
      vspace = {'game': scalar, 'goal': scalar}
      if self._curio:
        vspace['explore'] = scalar      # v2: curiosity's own value head
      vouts = {k: vout for k in vspace}
      self.val = embodied.jax.MLPHead(vspace, vouts, **vkw, name='val')
      self.slowval = embodied.jax.SlowModel(
          embodied.jax.MLPHead(vspace, vouts, **vkw, name='slowval'),
          source=self.val, **config.slowvalue)
    else:
      self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
      self.slowval = embodied.jax.SlowModel(
          embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
          source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    if self._use_rewcause:
      self.modules.append(self.rewcause)
    if self._valid_mask:
      self.modules.append(self.feas)
    if self._use_map:
      self.modules.append(self.mapmodel)
      if self._map_memory:
        self.modules += [self.memfut, self.memrec]
    if self._use_mgr:
      self.modules.append(self.mgr)
      # The codebook (goalbook) moves by moving average in the loss, like the
      # normalisers, so it is not handed to the optimizer.
      self.modules.append(self.goalnet if self._learned_goals else self.gphi)
      if not self._shared:
        self.modules.append(self.mval)
    if self._curio:
      self.modules += self.curio
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    if not self._use_rewcause:
      scales.pop('rewcause', None)  # keep losses/scales keys in sync
    if not self._use_map or self._map_memory:
      scales.pop('map', None)       # ditto -- agent asserts the keys match
      scales.pop('mappos', None)
    if not (self._use_map and self._map_memory):
      scales.pop('memfut', None)
      scales.pop('memrec', None)
    if not self._learned_goals:
      scales.pop('goalvq', None)
    if not self._curio:
      scales.pop('curio', None)
    if not self._valid_mask:
      scales.pop('feas', None)
    if not self._use_mgr:
      for key in ('gphi', 'mpolicy', 'mvalue'):
        scales.pop(key, None)
    elif self._shared:
      scales.pop('mvalue', None)      # the manager has no critic of its own
    if self._learned_goals:
      scales.pop('gphi', None)        # progress comes from the codebook
    if not self._repval:
      scales.pop('repval', None)
    self.scales = scales

  @property
  def policy_keys(self):
    # The reward-cause head is queried by the probe (mode='probe') through the
    # policy path, so when enabled its params must be part of the policy param
    # set (which the base agent syncs to the policy device).
    keys = ['enc', 'dyn', 'dec', 'pol']
    if self._use_rewcause:
      keys.append('rewcause')
    if self._use_map and self._map_to_actor:
      # The acting path decodes the map itself, so RSSM-2's weights have to
      # ship to the policy device alongside the actor's.
      keys.append('mapmodel')
    if self._use_mgr:
      keys.append('mgr')            # the manager picks goals while acting
      if self._learned_goals:
        keys += ['goalnet', 'goalbook']   # reached-checks run while acting
    if self._valid_learned:
      keys += ['feas', 'feasclock']  # B3: the learned mask acts
    return '^(' + '|'.join(keys) + ')/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self._use_mgr:
      # The goal the actor was pursuing at each step. Not used in training
      # (see _repval); kept so tools can read what the manager chose.
      spaces['goal'] = elements.Space(np.int32, (), 0, self._goals)
      if self._shared:
        # v1.2 resumes imagination from the replay's goal and segment step.
        spaces['gphase'] = elements.Space(np.int32, (), 0, self._hold)
    if self.config.replay_context:
      entries = dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)
      if self._use_map:
        # Without this RSSM-2 restarts from zeros every batch and never sees
        # more than batch_length/tick = 8 ticks -- far too few for a map that
        # accumulates over hundreds of steps.
        entries['map'] = self.mapmodel.entry_space
      spaces.update(elements.tree.flatdict(entries))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        self._map_initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def _map_truncate(self, entries, carry):
    """Resume RSSM-2 from the replay context at a mid-episode chunk boundary.

    deter2 and the window step (count) come back exactly, so the first window
    closes on the same step it did while acting. The partial feature and move
    sums of that window are not stored (5120 floats a step), so _map_step
    averages the steps it does have and rescales the move sum -- an
    approximation confined to the first window of each chunk.
    """
    if not self._use_map:
      return {}
    out = {**carry, 'deter2': nn.cast(entries['deter2'][:, -1])}
    if 'count' in entries:
      out['count'] = nn.cast(entries['count'][:, -1])
    for key in ('featsum', 'movesum', 'n'):
      out[key] = jnp.zeros_like(out[key])
    return out

  def _map_initial(self, batch_size):
    """RSSM-2 carry plus the accumulators the single-step acting path needs.

    observe() consumes whole 8-step windows; policy() sees one step at a time,
    so it sums features and movement until a window closes and then ticks.
    """
    if not self._use_map:
      return {}
    carry = dict(self.mapmodel.initial(batch_size))
    carry['featsum'] = jnp.zeros((batch_size, self._feat1_dim), f32)
    carry['movesum'] = jnp.zeros(
        (batch_size, self._n_act if self._map_memory else 2), f32)
    carry['count'] = jnp.zeros((batch_size, 1), f32)   # step within window
    carry['n'] = jnp.zeros((batch_size, 1), f32)       # steps summed so far
    if self._use_mgr:
      # The manager's current goal and the step within its segment. They ride
      # in the map carry because the manager ticks with RSSM-2.
      carry['goal'] = jnp.zeros((batch_size,), i32)
      carry['gphase'] = jnp.zeros((batch_size,), i32)
      if self._learned_goals:
        # Where the latent stood when the current goal was set: progress is
        # the change since then, compared with the goal's code.
        carry['gstart'] = jnp.zeros((batch_size, self._feat1_dim), f32)
    return nn.cast(carry)

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    # mode is static (traced once per value). 'greedy' takes the most likely
    # action and goal everywhere the agent acts; every other mode samples.
    self._greedy = (mode == 'greedy')
    (enc_carry, dyn_carry, dec_carry, map_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    mapfeat = None
    if self._use_map:
      prev_map = map_carry
      map_carry, mapfeat = self._map_act(map_carry, feat, prevact, reset)
    ainp = self.actor2tensor(feat, mapfeat)
    if self._use_mgr:
      goal, phase, map_carry = self._mgr_act(
          prev_map, map_carry, ainp, reset, obs, self.feat2tensor(feat))
      ainp = jnp.concatenate([ainp, self._goalfeat(goal, phase)], -1)
    policy = self.pol(ainp, bdims=1)
    if self._valid_learned:
      allowed = self._dream_valid(self.feat2tensor(feat), 1)
      explore = jax.random.uniform(nj.seed(), (allowed.shape[0], 1))
      allowed = allowed | (explore < self._valid_explore)
      if self._valid_rule_mix:
        # Label calibration only (never in a comparison run): some steps act
        # behind the rules, so batches hold valid AND invalid special actions.
        mix = jax.random.uniform(nj.seed(), (allowed.shape[0], 1))
        allowed = jnp.where(mix < self._valid_rule_mix, obs['valid'] > 0.5,
                            allowed)
      policy = self._masked(policy, allowed)
    elif self._valid_mask:
      policy = self._masked(policy, obs['valid'] > 0.5)
    act = jax.tree.map(self._choose, policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    if self._use_rewcause and mode == 'probe':
      # Probe only: expose the reward-cause head's per-feature P(phi_j = 1) so
      # the holdout probe can score compositional extrapolation at held-out
      # states. Gated on the dedicated 'probe' mode (not 'train'/'eval') because
      # every other caller feeds policy outputs into a replay buffer that later
      # asserts data.keys() == spaces.keys() (train.py and train_eval.py both do
      # driver.on_step(replay.add)); the probe's driver never stores to replay.
      # `mode` is a static arg to the compiled policy, so this is a clean
      # compile-time branch. The head wraps a Binary output in an Agg that sums
      # over phi, so read the raw per-feature logit rather than Agg.prob.
      rc = self.rewcause(self.feat2tensor(feat), bdims=1)
      out['rewcause_prob'] = jax.nn.sigmoid(rc.output.logit)
    if mode == 'probe':
      # Inspection only (tools/watch_agent.py): the action distribution the
      # actor actually produced, and the critic's value for this state. Gated
      # on 'probe' for the same reason rewcause_prob is -- every other caller
      # feeds policy outputs into a replay buffer that asserts the key set.
      # Only heads listed in policy_keys have their params on the policy
      # device, so `pol` is available here and `val` is not.
      out['policy_prob'] = jax.nn.softmax(
          policy['action'].logits, -1)
      if self._use_map and self._map_to_actor and not self._map_memory:
        # RSSM-2's map belief, for tools/map_eval.py to score against the true
        # map. mapmodel's params are only on the policy device when to_actor is
        # set (see policy_keys), which every map run uses.
        out['map_pred'] = jax.nn.sigmoid(
            self.mapmodel.decode(map_carry['deter2'])[0])
    if self._use_mgr:
      out['goal'] = goal
      if self._shared:
        out['gphase'] = phase
    carry = (enc_carry, dyn_carry, dec_carry, map_carry, act)
    if self.config.replay_context:
      entries = dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)
      if self._use_map:
        entries['map'] = dict(deter2=map_carry['deter2'],
                              count=map_carry['count'])
      out.update(elements.tree.flatdict(entries))
    return carry, act, out

  def _choose(self, dist):
    """An acting-time draw: the argmax under policy(mode='greedy')."""
    if getattr(self, '_greedy', False):
      return dist.pred()
    return dist.sample(nj.seed())

  def _masked(self, policy, valid):
    """Policy with impossible actions removed: zero probability, no gradient.

    The basic actions (NOOP, moves, DO) are always allowed. The mask is added
    after the categorical's unimix, so no uniform floor leaks back onto a
    masked action.
    """
    if valid is None:
      return policy
    allowed = valid
    if self._exempt_basic:
      basic = jnp.zeros(valid.shape[-1], bool).at[
          jnp.asarray(cvalid.BASIC)].set(True)
      allowed = valid | basic
    logits = policy['action'].logits + jnp.where(allowed, 0.0, -1e4)
    return {**policy, 'action': jouts.Categorical(logits)}

  def _feas_learned(self, inp, repfeat, prevact, reset, obs, training):
    """B3: train the effect head on the world model's own counterfactual.

    The action taken at step t-1 (prevact[:, t]) moved the agent from the
    posterior state at t-1 to the observation at t. One prior step from that
    state under the real action, and one under NOOP: if NOOP explains the real
    posterior at t about as well, the action did nothing. Labels and the
    counterfactual carry no gradient; only the head learns from them.
    """
    B, T = reset.shape
    flat = lambda x: x.reshape((B * (T - 1), *x.shape[2:]))
    state = {k: sg(flat(repfeat[k][:, :-1])) for k in ('deter', 'stoch')}
    act = flat(prevact['action'][:, 1:])
    _, (real, _) = self.dyn.imagine(state, {'action': act}, 1, training,
                                    single=True)
    if self._valid_reference == 'random':
      # v2: no assumption that action 0 does nothing -- compare with a
      # randomly chosen other action instead.
      shift = jax.random.randint(nj.seed(), act.shape, 1, self._n_act)
      ref = (act + shift) % self._n_act
    else:
      ref = jnp.zeros_like(act)
    _, (noop, _) = self.dyn.imagine(state, {'action': ref}, 1,
                                    training, single=True)
    rules = obs.get('valid')
    if self._valid_label == 'obs':
      def err(feat):
        # Decode the prior's most likely state, so sampling noise does not
        # differ between the two predictions being compared.
        logit = sg(f32(feat['logit']))
        stoch = jax.nn.one_hot(jnp.argmax(logit, -1), logit.shape[-1])
        fd = {'deter': sg(feat['deter']).reshape((B, T - 1, -1)),
              'stoch': nn.cast(stoch).reshape((B, T - 1, *stoch.shape[1:]))}
        _, _, rec = self.dec({}, fd, reset[:, 1:], training)
        return sum(f32(out.loss(sg(f32(obs[k][:, 1:]))))
                   for k, out in rec.items())
      # Absolute nats, not relative: a real effect (one new tile, one item)
      # is a few nats inside a ~100-nat reconstruction error, so a relative
      # margin almost never fired and labels came out at chance.
      evidence = (sg(err(noop)) - sg(err(real))).reshape(-1)
      label = f32(evidence > self._valid_margin).reshape((B, T - 1))
    else:
      dist = self.dyn._dist
      post = dist(sg(flat(repfeat['logit'][:, 1:])))
      evidence = sg(f32(post.kl(dist(sg(noop['logit'])))
                        - post.kl(dist(sg(real['logit'])))))
      label = f32(evidence > self._valid_evidence).reshape((B, T - 1))
    act = act.reshape((B, T - 1))
    # Only real transitions (not across an episode start) and actions that
    # can differ from NOOP at all.
    keep = ~reset[:, 1:]
    if self._valid_reference != 'random':
      keep = keep & (act > 0)
    keep = f32(keep)
    logit = f32(self.feas(inp, 2).output.logit)[:, :-1]           # (B, T-1, A)
    taken = jnp.take_along_axis(logit, act[..., None], -1)[..., 0]
    bce = jax.nn.softplus(taken) - label * taken
    loss = jnp.concatenate([bce * keep, jnp.zeros_like(bce[:, :1])], 1)
    mets = {'feas/label_rate': (label * keep).sum() / jnp.maximum(keep.sum(), 1),
            'feas/evidence': (evidence.reshape((B, T - 1)) * keep).sum()
                             / jnp.maximum(keep.sum(), 1)}
    if rules is not None:
      # Measuring stick only: did the learned label agree with the rules for
      # the special actions actually taken?
      truth = jnp.take_along_axis(f32(rules[:, :-1] > 0.5), act[..., None],
                                  -1)[..., 0]
      special = keep * f32(act >= len(cvalid.BASIC))
      tp = (label * truth * special).sum()
      mets['feas/label_precision'] = tp / jnp.maximum((label * special).sum(), 1)
      mets['feas/label_recall'] = tp / jnp.maximum((truth * special).sum(), 1)
      # Threshold-free: how well the evidence ranks rule-valid special
      # actions above invalid ones (0.5 = chance), and its mean on each side.
      ev = evidence.reshape(-1)
      pos, neg = (truth * special).reshape(-1), ((1 - truth) * special).reshape(-1)
      pairs = pos[:, None] * neg[None, :]
      wins = f32(ev[:, None] > ev[None, :]) * pairs
      mets['feas/evidence_auc'] = wins.sum() / jnp.maximum(pairs.sum(), 1)
      mets['feas/evidence_valid'] = (ev * pos).sum() / jnp.maximum(pos.sum(), 1)
      mets['feas/evidence_invalid'] = (ev * neg).sum() / jnp.maximum(neg.sum(), 1)
      mets['feas/valid_rate'] = pos.sum() / jnp.maximum(special.sum(), 1)
      # The mask blocks an action when the head's P(effect) < threshold, so
      # what matters per margin is how often valid actions get labelled
      # (must stay well above threshold) and invalid ones (well below).
      for m in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
        hit = f32(ev > m)
        mets[f'feas/tpr_{m:g}'] = (hit * pos).sum() / jnp.maximum(pos.sum(), 1)
        mets[f'feas/fpr_{m:g}'] = (hit * neg).sum() / jnp.maximum(neg.sum(), 1)
    return loss, mets

  def _dream_valid(self, x, bdims):
    """Validity inside imagination, from the latent -- or None if unmasked."""
    if not self._valid_mask:
      return None
    logit = f32(self.feas(x, bdims).output.logit)
    # A low threshold errs toward allowing: a false "impossible" blocks an
    # action in every dream, a false "possible" merely costs a no-op.
    allowed = sg(jax.nn.sigmoid(logit)) > self._valid_threshold
    if self._valid_learned:
      allowed = allowed | (self.feasclock.read() < self._valid_warmup)
    return allowed

  def _goalfeat(self, goal, phase):
    """What the actor and its critic see of the manager: goal and segment step.

    The step matters to the critic: the same state is worth more with seven
    steps left to reach the goal than with one.
    """
    return nn.cast(jnp.concatenate([
        jax.nn.one_hot(goal, self._goals),
        jax.nn.one_hot(phase, self._phases)], -1))

  def _mgr_input(self, ainp, deter2, flags):
    """Manager input: the actor's view, RSSM-2's state, goals reached so far.

    sg on deter2 for the same reason mapfeat stops gradients: the manager's
    objective must not reshape RSSM-2 into whatever raises its return. The
    flags say which goals were already reached this episode -- the manager's
    bonus is paid only on a goal's first reach, so without them the same state
    would sometimes pay and sometimes not.
    """
    deter2 = nn.cast(sg(deter2))
    if deter2.ndim == ainp.ndim - 1:
      deter2 = jnp.repeat(deter2[:, None], ainp.shape[-2], -2)
    return jnp.concatenate([ainp, deter2, nn.cast(f32(flags))], -1)

  def _mgr_dist(self, x, bdims, reached):
    """The manager's choice, with goals already reached right now ruled out.

    Proposing a goal that already holds would let the manager collect its
    bonus for nothing and teach the actor nothing. NONE is always allowed.
    """
    logits = self.mgr(x, bdims).logits
    blocked = reached & (jnp.arange(self._goals) > 0)
    return jouts.Categorical(logits + jnp.where(blocked, -1e4, 0.0))

  def _reached(self, feat, bdims):
    """Which goals hold in an imagined state, from the progress head."""
    phi = f32(sg(self.gphi(self.feat2tensor(feat), bdims).pred()))
    return (phi > self._reach_threshold) & (jnp.arange(self._goals) > 0)

  def _mgr_act_learned(self, prev, carry, ainp, reset, feat1):
    """v2 acting: hold a learned goal until the latent has moved its way.

    Reached when the change since the goal was set is classified as the
    goal's code (its nearest code, cosine above code_reach), or after `hold`
    steps. No
    observation-derived flags or masks: a learned change-type is never
    "already true", and which codes were reached this episode is not tracked.
    """
    B = ainp.shape[0]
    flags = jnp.zeros((B, self._goals), bool)
    feat1 = f32(sg(feat1))
    _, hit = goalcodes.progress(
        self.goalnet, self.goalbook, feat1 - f32(prev['gstart']), prev['goal'],
        self._code_reach)
    hit = hit & (prev['gphase'] > 0)
    decide = reset | hit | (prev['gphase'] >= self._hold)
    x = self._mgr_input(ainp, carry['deter2'], flags)
    pick = self._choose(self._mgr_dist(x, 1, flags))
    phase = jnp.where(decide, 0, prev['gphase'])
    goal = jnp.where(decide, pick, prev['goal'])
    gstart = jnp.where(decide[:, None], feat1, f32(prev['gstart']))
    carry = {**carry, 'goal': goal, 'gphase': phase + 1, 'gstart': gstart}
    return goal, phase, carry

  def _mgr_act(self, prev, carry, ainp, reset, obs, feat1=None):
    if self._learned_goals:
      return self._mgr_act_learned(prev, carry, ainp, reset, feat1)
    """One acting step of the manager: pick a goal at the start of a segment.

    Segments are counted from the episode start, so a decision at step 8k
    reads the RSSM-2 tick that closed at step 8k - 1. Branchless like
    _map_act: the manager runs every step and its pick is kept only on a
    decision step. Flags and the mask come from the observation itself.
    """
    flags = obs['goalreach'] > 0.5
    reached = (obs['goalphi'] >= 1.0 - 1e-3) & (jnp.arange(self._goals) > 0)
    x = self._mgr_input(ainp, carry['deter2'], flags)
    pick = self._choose(self._mgr_dist(x, 1, reached))
    if self._shared:
      # v1.2: hold the goal until the observation shows it reached, or for
      # `hold` steps; gphase counts steps since it was set.
      hit = jnp.take_along_axis(reached, prev['goal'][:, None], -1)[:, 0]
      decide = reset | hit | (prev['gphase'] >= self._hold)
      phase = jnp.where(decide, 0, prev['gphase'])
      goal = jnp.where(decide, pick, prev['goal'])
      carry = {**carry, 'goal': goal, 'gphase': phase + 1}
      return goal, phase, carry
    phase = jnp.where(reset, 0, prev['gphase'])
    goal = jnp.where(phase == 0, pick, prev['goal'])
    carry = {**carry, 'goal': goal, 'gphase': (phase + 1) % self._every}
    return goal, phase, carry

  def _imagine_mgr(self, starts, first, frozen, deter2, flags0, H, training):
    """Imagination with the manager in the loop.

    Every rollout opens on a manager decision; after each `every` steps the
    manager picks again from the state reached. Goal, segment step and the
    reached-this-episode flags ride in the scan carry next to (deter, stoch),
    so the rollout samples each action and goal from exactly the distribution
    the loss later scores -- the consistency imag_shift lacked. Inside a dream
    "reached" is read from the progress head; the flags start from the
    observation-derived ones at the imagination start.
    """
    actor = lambda feat: self.actor2tensor(feat, frozen)

    def mgr(feat, flags, reached):
      x = self._mgr_input(actor(feat), deter2, flags)
      return self._mgr_dist(x, 1, reached).sample(nj.seed())

    def act(feat, goal, phase):
      x = jnp.concatenate([actor(feat), self._goalfeat(goal, phase)], -1)
      return sample(self._masked(
          self.pol(x, 1), self._dream_valid(self.feat2tensor(feat), 1)))

    def step(carry, _):
      dc = {k: carry[k] for k in ('deter', 'stoch')}
      action = act(sg(dc), carry['goal'], carry['gphase'])
      dc, (feat, action) = self.dyn.imagine(
          dc, action, 1, training, single=True)
      reached = self._reached(sg(feat), 1)
      flags = carry['flags'] | reached
      phase = carry['gphase'] + 1
      decide = phase >= self._every
      goal = jnp.where(decide, mgr(sg(feat), flags, reached), carry['goal'])
      phase = jnp.where(decide, 0, phase)
      carry = {**dc, 'goal': goal, 'gphase': phase, 'flags': flags}
      return carry, (feat, action, goal, phase, flags, reached)

    first1 = jax.tree.map(lambda x: x[:, 0], first)
    reached0 = self._reached(sg(first1), 1)
    flags0 = flags0 | reached0
    goal0 = mgr(sg(first1), flags0, reached0)
    phase0 = jnp.zeros_like(goal0)
    carry = {**nn.cast(starts), 'goal': goal0, 'gphase': phase0,
             'flags': flags0}
    _, (feat, action, goals, phases, flags, reached) = nj.scan(
        step, carry, (), H, axis=1)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(feat)], 1)
    head = lambda x0, xs: jnp.concatenate([x0[:, None], xs], 1)
    goals, phases = head(goal0, goals), head(phase0, phases)
    flags, reached = head(flags0, flags), head(reached0, reached)
    last = act(jax.tree.map(lambda x: x[:, -1], imgfeat),
               goals[:, -1], phases[:, -1])
    imgact = concat([action, jax.tree.map(lambda x: x[:, None], last)], 1)
    return imgfeat, imgact, goals, phases, flags, reached

  def _held(self, x, goals):
    """x[:, t, goal held at t-1] for t = 1..H: (N, H+1, G) -> (N, H)."""
    held = goals[:, :-1]
    return jnp.take_along_axis(x[:, 1:], held[..., None], -1)[..., 0]

  def _segments(self, H):
    idx = list(range(0, H + 1, self._every))
    if idx[-1] != H:
      idx.append(H)
    return idx

  def _actor_bonus(self, goals, reached, H):
    """+1 to the actor the first time its goal is reached within a segment.

    Only the first time: stepping away from a table and back would otherwise
    collect the bonus every two steps.
    """
    held = goals[:, :-1]
    now = self._held(reached, goals)
    before = jnp.take_along_axis(reached[:, :-1], held[..., None], -1)[..., 0]
    event = f32(now & ~before & (held > 0))                  # (N, H)
    idx = self._segments(H)
    parts = []
    for a, b in zip(idx[:-1], idx[1:]):
      seg = event[:, a:b]
      parts.append(seg * f32(jnp.cumsum(seg, 1) == 1))
    return jnp.concatenate(parts, 1)

  def _mgr_loss(self, imgfeat, goals, flags, reached, rew, con, frozen, deter2,
                H, training):
    """The manager as an actor-critic over its own decisions.

    Its time step is the segment: states 0, every, 2*every, ... and the last
    imagined state for bootstrapping. A segment's reward is the game reward
    plus mgr_bonus for each goal it set that was reached for the first time
    this episode, accumulated over the segment and discounted inside it
    exactly as lambda_return would; its continuation is the product of the
    segment's continuation probabilities. imag_loss then runs unchanged on
    that abstract trajectory -- 2 decisions per 15-step rollout at every=8,
    and a critic whose bootstrap reaches 8x further per step than the
    actor's.
    """
    held = goals[:, :-1]
    was = jnp.take_along_axis(flags[:, :-1], held[..., None], -1)[..., 0]
    first = f32(self._held(reached, goals) & ~was & (held > 0))   # (N, H)
    step_rew = rew + self._mgr_bonus * jnp.concatenate(
        [jnp.zeros_like(first[:, :1]), first], 1)
    idx = self._segments(H)
    mrew, mcon = [jnp.zeros_like(rew[:, 0])], [con[:, 0]]
    for a, b in zip(idx[:-1], idx[1:]):
      live = jnp.cumprod(con[:, a + 1:b + 1], 1)
      before = jnp.concatenate([jnp.ones_like(live[:, :1]), live[:, :-1]], 1)
      mrew.append((before * step_rew[:, a + 1:b + 1]).sum(1))
      mcon.append(live[:, -1])
    mrew, mcon = jnp.stack(mrew, 1), jnp.stack(mcon, 1)
    mfeat = jax.tree.map(lambda x: x[:, idx], imgfeat)
    minp = self._mgr_input(
        self.actor2tensor(mfeat, frozen), deter2, flags[:, idx])
    mact = {'goal': goals[:, idx]}
    kw = {**self.config.imag_loss, 'actent': float(self.config.manager.actent)}
    los, _, mets = imag_loss(
        mact, mrew, mcon,
        {'goal': self._mgr_dist(minp, 2, reached[:, idx])},
        self.mval(minp, 2),
        self.mslowval(minp, 2),
        self.mretnorm, self.mvalnorm, self.madvnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **kw)
    picks = jax.nn.one_hot(goals[:, idx[:-1]], self._goals).mean((0, 1))
    for i, name in enumerate(self._goal_names):
      mets[f'pick/{name.lower()}'] = picks[i]
    mets['first_reach'] = first.sum(1).mean()
    mets['masked_share'] = f32(reached[:, idx[:-1]]).mean()
    return los, mets

  # --- v1.2: goals held until reached, one two-headed critic -----------------

  def _imagine_hold(self, starts, first, frozen, deter2, flags0, goal0, phase0,
                    H, training):
    """Imagination for v1.2: goals are held until reached or `hold` steps.

    Each rollout resumes the goal and segment step the agent actually had at
    that replay state, so the critic sees every segment step the acting agent
    does (up to hold - 1) and the replay value loss is consistent again. A new
    goal is drawn after the state where the held goal is reached, or when the
    hold runs out -- the same rule _mgr_act applies to real observations.
    """
    actor = lambda feat: self.actor2tensor(feat, frozen)

    def mgr(feat, flags, reached):
      x = self._mgr_input(actor(feat), deter2, flags)
      return self._mgr_dist(x, 1, reached).sample(nj.seed())

    def act(feat, goal, phase):
      x = jnp.concatenate([actor(feat), self._goalfeat(goal, phase)], -1)
      return sample(self._masked(
          self.pol(x, 1), self._dream_valid(self.feat2tensor(feat), 1)))

    def step(carry, _):
      dc = {k: carry[k] for k in ('deter', 'stoch')}
      action = act(sg(dc), carry['goal'], carry['gphase'])
      dc, (feat, action) = self.dyn.imagine(
          dc, action, 1, training, single=True)
      reached = self._reached(sg(feat), 1)
      flags = carry['flags'] | reached
      hit = jnp.take_along_axis(reached, carry['goal'][:, None], -1)[:, 0]
      since = carry['gphase'] + 1
      decide = hit | (since >= self._hold)
      goal = jnp.where(decide, mgr(sg(feat), flags, reached), carry['goal'])
      phase = jnp.where(decide, 0, since)
      carry = {**dc, 'goal': goal, 'gphase': phase, 'flags': flags}
      return carry, (feat, action, goal, phase, flags, reached)

    first1 = jax.tree.map(lambda x: x[:, 0], first)
    reached0 = self._reached(sg(first1), 1)
    flags0 = flags0 | reached0
    carry = {**nn.cast(starts), 'goal': goal0, 'gphase': phase0,
             'flags': flags0}
    _, (feat, action, goals, phases, flags, reached) = nj.scan(
        step, carry, (), H, axis=1)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(feat)], 1)
    head = lambda x0, xs: jnp.concatenate([x0[:, None], xs], 1)
    goals, phases = head(goal0, goals), head(phase0, phases)
    flags, reached = head(flags0, flags), head(reached0, reached)
    last = act(jax.tree.map(lambda x: x[:, -1], imgfeat),
               goals[:, -1], phases[:, -1])
    imgact = concat([action, jax.tree.map(lambda x: x[:, None], last)], 1)
    return imgfeat, imgact, goals, phases, flags, reached

  def _imagine_learned(self, starts, first, frozen, deter2, goal0, phase0,
                       gstart0, H, training):
    """v2 imagination: learned goals held until the latent moves their way.

    Like _imagine_hold, but "reached" and the goal reward come from the
    codebook: progress is the cosine between the change since the goal was set
    and the goal's code, and the goal is reached when that change is
    classified as the goal's code. The goal reward for s_{t-1} -> s_t is
    goal_reward x the change in progress plus reach_bonus on reaching, and is
    computed in the scan because progress is measured from a per-rollout
    start point that moves whenever the goal changes.
    """
    actor = lambda feat: self.actor2tensor(feat, frozen)
    N = goal0.shape[0]
    noflags = jnp.zeros((N, self._goals), bool)

    def mgr(feat):
      x = self._mgr_input(actor(feat), deter2, noflags)
      return self._mgr_dist(x, 1, noflags).sample(nj.seed())

    def act(feat, goal, phase):
      x = jnp.concatenate([actor(feat), self._goalfeat(goal, phase)], -1)
      return sample(self._masked(
          self.pol(x, 1), self._dream_valid(self.feat2tensor(feat), 1)))

    def step(carry, _):
      dc = {k: carry[k] for k in ('deter', 'stoch')}
      action = act(sg(dc), carry['goal'], carry['gphase'])
      dc, (feat, action) = self.dyn.imagine(
          dc, action, 1, training, single=True)
      feat1 = f32(sg(self.feat2tensor(feat)))
      prog, hit = goalcodes.progress(
          self.goalnet, self.goalbook, feat1 - carry['gstart'], carry['goal'],
          self._code_reach)
      grew = (self._goal_reward * (prog - carry['gprog'])
              + self._reach_bonus * f32(hit))
      since = carry['gphase'] + 1
      decide = hit | (since >= self._hold)
      goal = jnp.where(decide, mgr(sg(feat)), carry['goal'])
      phase = jnp.where(decide, 0, since)
      carry = {**dc, 'goal': goal, 'gphase': phase,
               'gstart': jnp.where(decide[:, None], feat1, carry['gstart']),
               'gprog': jnp.where(decide, 0.0, prog)}
      return carry, (feat, action, goal, phase, grew, f32(hit))

    gstart0 = f32(sg(gstart0))
    first1 = jax.tree.map(lambda x: x[:, 0], first)
    prog0, _ = goalcodes.progress(
        self.goalnet, self.goalbook,
        f32(sg(self.feat2tensor(first1))) - gstart0, goal0, self._code_reach)
    carry = {**nn.cast(starts), 'goal': goal0, 'gphase': phase0,
             'gstart': gstart0, 'gprog': sg(prog0)}
    _, (feat, action, goals, phases, grew, hits) = nj.scan(
        step, carry, (), H, axis=1)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(feat)], 1)
    head = lambda x0, xs: jnp.concatenate([x0[:, None], xs], 1)
    goals, phases = head(goal0, goals), head(phase0, phases)
    grew = head(jnp.zeros_like(grew[:, 0]), sg(grew))
    last = act(jax.tree.map(lambda x: x[:, -1], imgfeat),
               goals[:, -1], phases[:, -1])
    imgact = concat([action, jax.tree.map(lambda x: x[:, None], last)], 1)
    return imgfeat, imgact, goals, phases, grew, hits

  def _curiosity(self, feat, action):
    """(..., F), (...) -> (...): disagreement of the ensemble's predictions."""
    x = jnp.concatenate([nn.cast(sg(feat)), nn.cast(
        jax.nn.one_hot(action, self._n_act))], -1)
    bd = x.ndim - 1
    preds = jnp.stack([f32(m(x, bd).pred()) for m in self.curio], 0)
    return sg(preds.var(0).mean(-1))

  def _goal_stream(self, inp, goals, reached):
    """The bottom actor's goal reward for each step s_{t-1} -> s_t.

    goal_reward x the change in the held goal's predicted progress, plus
    reach_bonus when it becomes reached. The goal is replaced right after the
    state where it is reached, so the bonus fires once per goal held.
    """
    phi = jnp.clip(f32(sg(self.gphi(inp, 2).pred())), 0.0, 1.0)
    held = goals[:, :-1]
    pick = lambda p: jnp.take_along_axis(p, held[..., None], -1)[..., 0]
    prog = jnp.where(held == 0, 0.0, pick(phi[:, 1:]) - pick(phi[:, :-1]))
    event = f32(pick(reached[:, 1:]) & ~pick(reached[:, :-1]) & (held > 0))
    grew = self._goal_reward * prog + self._reach_bonus * event
    grew = jnp.concatenate([jnp.zeros_like(grew[:, :1]), grew], 1)
    return grew, event

  def _mgr_loss_shared(self, imgfeat, goals, phases, flags, reached, weight,
                       frozen, deter2, rscale):
    """The manager actor, judged by the shared critic's game head.

    At imagined states 0, 8, ... the game head scores every goal as if set
    right now, Q(s, g) = V_game(s, g, step 0), and the manager's gradient is
    the exact expectation over all 13 goals, sum_g pi(g|s) A(s, g), with
    A = Q - sum_g pi Q. No sampled choice and no critic of its own: v1's
    manager learned from one sampled goal per decision and its advantage
    drowned in that noise. Goal-stream value never enters, so the manager
    cannot pay itself through the bottom actor's bonuses.
    """
    T = imgfeat['deter'].shape[1]
    idx = list(range(0, T - 1, 8))
    feat = jax.tree.map(lambda x: x[:, idx], imgfeat)
    base = self.actor2tensor(feat, frozen)                   # (N, I, D)
    N, I = base.shape[:2]
    G = self._goals
    allg = self._goalfeat(jnp.arange(G), jnp.zeros(G, i32))  # (G, G + P)
    d2 = nn.cast(sg(deter2))
    fl = nn.cast(f32(flags[:, idx]))
    tile = lambda x: jnp.broadcast_to(x[:, :, None], (N, I, G, x.shape[-1]))
    cinp = jnp.concatenate([
        tile(base), jnp.broadcast_to(allg, (N, I, G, allg.shape[-1])),
        tile(jnp.repeat(d2[:, None], I, 1)), tile(fl)], -1)
    voffset, vscale = self.valnorm.stats()
    heads = self.val(cinp, 3)
    q = f32(heads['game'].pred()) * vscale + voffset   # (N, I, G)
    if self._curio:
      # v2: the manager also values where a goal leads somewhere new.
      eoff, escale = self.evalnorm.stats()
      qe = f32(heads['explore'].pred()) * escale + eoff
      if self._curio_norm:
        # Divided by rscale below, like the game value: rescale so the
        # explore part ends up divided by its own spread instead.
        qe = qe * rscale / self.eretnorm.stats()[1]
      q = q + self._curio_mgr * qe
    dist = self._mgr_dist(self._mgr_input(base, deter2, flags[:, idx]), 2,
                          reached[:, idx])
    probs = jax.nn.softmax(f32(dist.logits), -1)
    adv = sg((q - (probs * q).sum(-1, keepdims=True)) / rscale)
    ent = dist.entropy()
    actent = float(self.config.manager.actent)
    loss = sg(weight[:, idx]) * -((probs * adv).sum(-1) + actent * ent)
    mets = {'ent/goal': ent.mean(),
            'q_spread': q.std(-1).mean(),
            'q_best_minus_mean': (q.max(-1) - (probs * q).sum(-1)).mean()}
    decided = (phases == 0)[:, 1:]
    picks = (jax.nn.one_hot(goals[:, 1:], G) * decided[..., None]).sum((0, 1))
    picks = picks / jnp.maximum(picks.sum(), 1)
    for i, name in enumerate(self._goal_names):
      mets[f'pick/{name.lower()}'] = picks[i]
    return loss, mets

  def _map_moves(self, actions):
    """What RSSM-2 is told about the actions in its window.

    v1: hand-computed (dy, dx) per movement action -- knowledge of what actions
    1-4 do. v2 (memory): the raw action one-hot, to be learned from.
    """
    if self._map_memory:
      return jax.nn.one_hot(actions, self._n_act)
    return mapmod.action_deltas(actions)

  def _map_step(self, carry, feat1, move, reset):
    """One env step of RSSM-2, shared by acting and training.

    Accumulate this step's features and movement; when the window's `tick`
    steps are in, run the GRU once. Windows count from the episode start. The
    state a step sees is the last CLOSED window -- possibly including the step
    itself, never a later one. Training scans this same function over the
    batch, so it learns on exactly the states the actor reads online.

    Branchless on purpose: the GRU runs every step and its result is kept only
    where a window closes. Returns (carry, closed, tick input).
    """
    tick = self._map_tick
    keep = f32(~reset)[:, None]
    # A new episode starts from an empty state at once, not when its first
    # window closes -- otherwise the old episode's state is read for up to
    # tick-1 steps and then fed into the new episode's first update.
    prev = jnp.where(reset[:, None], 0.0, f32(carry['deter2']))
    featsum = f32(carry['featsum']) * keep + f32(feat1)
    # On a reset step the action is the previous episode's last one: it moved
    # the old agent, not this one, so it stays out of the new window.
    movesum = (f32(carry['movesum']) + f32(move)) * keep
    count = f32(carry['count']) * keep + 1.0
    n = f32(carry['n']) * keep + 1.0
    # n < count only in the first window of a resumed chunk (_map_truncate).
    inp = jnp.concatenate(
        [featsum / n, movesum * (count / n), jnp.full_like(count, tick)], -1)
    new = f32(self.mapmodel.tick(nn.cast(prev), nn.cast(inp)))
    closed = count[:, 0] >= tick
    deter2 = jnp.where(closed[:, None], new, prev)
    zero = lambda x: jnp.where(closed[:, None], jnp.zeros_like(x), x)
    # The framework compiles the policy carry in the compute dtype (bf16);
    # counts up to `tick` are exact there.
    carry = {**carry, **nn.cast(dict(
        deter2=deter2, featsum=zero(featsum), movesum=zero(movesum),
        count=zero(count), n=zero(n)))}
    return carry, closed, inp

  def _map_act(self, carry, feat, prevact, reset):
    """One acting step of RSSM-2, then the actor's view of it."""
    feat1 = sg(self.feat2tensor(feat))
    move = self._map_moves(prevact['action'])
    carry, _, _ = self._map_step(carry, feat1, move, reset)
    if not self._map_to_actor:
      return carry, None
    mapfeat, _ = self.mapfeat(carry['deter2'])
    return carry, mapfeat[:, 0]

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    if self._use_mgr and not self._shared:
      self.mslowval.update()
    if self._valid_learned:
      self.feasclock.inc()
    outs = {}
    if self.config.replay_context:
      names = dict(stepid=stepid, enc=entries[0], dyn=entries[1],
                   dec=entries[2])
      if self._use_map:
        names['map'] = entries[3]
      updates = elements.tree.flatdict(names)
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry, map_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}
    map_entries = {}
    deter2_step = None

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    if self._use_mgr and not self._learned_goals:
      # Goal progress from the latent, so the actor's goal reward exists inside
      # imagination. Trained like 'rew': from the world-model feature, never
      # the actor's, so the goal cannot leak into what the model believes.
      losses['gphi'] = self.gphi(inp, 2).loss(sg(f32(obs['goalphi'])))
    if self._use_rewcause:
      ach = f32(obs['ach'])                              # (B, T, A)
      phi = jnp.asarray(self._phi_table)                # (A, D)
      wt = jnp.asarray(self._weight_table)              # (A,)
      hv = jnp.asarray(self._holdout_vec)               # (A,)
      # Target = union of unlocked achievements' feature rows.
      target = jnp.clip(ach @ phi, 0.0, 1.0)            # (B, T, D)
      any_ach = ach.sum(-1) > 0                          # (B, T)
      w_ach = (ach * wt[None, None, :]).max(-1)          # (B, T)
      weight = jnp.where(any_ach, w_ach, self._rewcause_none_weight)
      # Held-out steps contribute zero loss (iron-holdout extrapolation test).
      mask = 1.0 - jnp.clip((ach * hv[None, None, :]).sum(-1), 0.0, 1.0)
      rc = self.rewcause(inp, 2).loss(sg(target))        # (B, T)
      losses['rewcause'] = rc * sg(weight) * sg(mask)
    if self._valid_learned:
      losses['feas'], mets = self._feas_learned(
          sg(inp) if self._valid_detach else inp, repfeat, prevact, reset,
          obs, training)
      metrics.update(mets)
    elif self._valid_mask:
      # Named 'feas', not 'valid': with valid.input the decoder already owns
      # a 'valid' reconstruction loss.
      losses['feas'] = self.feas(inp, 2).loss(sg(f32(obs['valid'] > 0.5)))
    if self._valid_mask and 'valid' in obs:
      target = f32(obs['valid'] > 0.5)
      pred = self._dream_valid(inp, 2)
      special = jnp.arange(target.shape[-1]) >= len(cvalid.BASIC)
      pos = (target > 0) & special
      # Recall on the actions that are actually possible is the number that
      # matters: a missed one is an action the dreaming agent cannot press.
      metrics['valid/recall'] = (pred & pos).sum() / jnp.maximum(pos.sum(), 1)
      metrics['valid/false_pos'] = (pred & ~(target > 0) & special).mean()
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    if self._learned_goals:
      # v2: the goal codebook learns from latent changes over code_window
      # steps within one episode.
      w = self._code_window
      f1 = f32(sg(self.feat2tensor(repfeat)))
      delta = (f1[:, w:] - f1[:, :-w]).reshape((-1, f1.shape[-1]))
      starts = jnp.cumsum(f32(reset), 1)
      valid = f32((starts[:, w:] - starts[:, :-w]) == 0).reshape(-1)
      vq, vmets = goalcodes.vq_loss(
          self.goalnet, self.goalbook, delta, valid, self._code_beta,
          update=training)
      vq = vq.reshape((B, T - w))
      losses['goalvq'] = jnp.concatenate([vq, jnp.zeros((B, w), f32)], 1)
      metrics.update(prefix(vmets, 'goals'))
    if self._curio:
      # v2: each ensemble member predicts the next posterior stoch (as
      # probabilities) from (latent, action taken); the disagreement of the
      # trained members is the curiosity reward in imagination.
      f1 = sg(self.feat2tensor(repfeat))[:, :-1]
      a = prevact['action'][:, 1:]
      x = jnp.concatenate([nn.cast(f1), nn.cast(
          jax.nn.one_hot(a, self._n_act))], -1)
      tgt = sg(f32(jax.nn.softmax(f32(repfeat['logit'][:, 1:]), -1)))
      tgt = tgt.reshape((*tgt.shape[:2], -1))
      keep = f32(~reset[:, 1:])
      closs = sum(m(x, 2).loss(tgt) for m in self.curio) * keep
      losses['curio'] = jnp.concatenate([closs, jnp.zeros((B, 1), f32)], 1)
    if self._use_map:
      tick = self._map_tick
      # sg on the way IN: the map loss must never reach RSSM-1, or we recreate
      # the very gradient competition the two-model split exists to prevent.
      feat1 = sg(self.feat2tensor(repfeat))
      moves = self._map_moves(prevact['action'])
      # The acting step, scanned over the batch: each step carries the state
      # of the last closed window, exactly as the actor saw it online. (This
      # used to pool fixed 8-step windows and hand every step its own window's
      # result, so a step saw up to 7 steps of its future.)
      keys = ('deter2', 'featsum', 'movesum', 'count', 'n')
      mc = nn.cast({k: map_carry[k] for k in keys})   # as _map_step returns

      def mapstep(c, xs):
        c, closed, inp = self._map_step(c, *xs)
        return c, (c['deter2'], c['count'], closed, inp)

      mc, (deter2_step, count_step, closed, tickin) = nj.scan(
          mapstep, mc, (feat1, moves, reset), axis=1)
      map_carry = {**map_carry, **mc}
      map_entries = dict(deter2=deter2_step, count=count_step)
      wclose = f32(closed)                                    # (B, T)
      nclose = jnp.maximum(wclose.sum(), 1.0)
    if self._use_map and self._map_memory:
      # v2 memory: from RSSM-2's state where a window closes, predict RSSM-1's
      # mean latent over the window h ticks later (what is coming) and
      # reconstruct the observation at the close h ticks earlier (what was
      # seen), within one episode. Windows count from the episode start, so
      # the close h ticks away is exactly h*tick steps away.
      featw = f32(tickin[..., :self._feat1_dim])              # window means
      vec = f32(obs['vector'])
      cs = jnp.cumsum(f32(reset), 1)
      futs, recs = self.memfut(deter2_step, 2), self.memrec(deter2_step, 2)
      fl, fn, rl, rn = 0.0, 0.0, 0.0, 0.0
      for h in self._mem_ahead:
        d = h * tick
        if d >= T:
          continue
        ok = wclose[:, :-d] * wclose[:, d:] * f32(cs[:, d:] == cs[:, :-d])
        lh = futs[f'h{h}'].loss(
            sg(jnp.concatenate([featw[:, d:], featw[:, -d:]], 1)))
        fl = fl + (lh[:, :-d] * ok).sum()
        fn = fn + ok.sum()
      for h in self._mem_back:
        d = h * tick
        if d >= T:
          continue
        ok = wclose[:, d:] * wclose[:, :-d] * f32(cs[:, d:] == cs[:, :-d])
        lh = recs[f'h{h}'].loss(
            sg(jnp.concatenate([vec[:, :d], vec[:, :-d]], 1)))
        rl = rl + (lh[:, d:] * ok).sum()
        rn = rn + ok.sum()
      fut = fl / jnp.maximum(fn, 1.0)
      rec = rl / jnp.maximum(rn, 1.0)
      losses['memfut'] = jnp.full((B, T), fut, f32)
      losses['memrec'] = jnp.full((B, T), rec, f32)
      metrics['memory/future'] = fut
      metrics['memory/recall'] = rec
      metrics['memory/gate'] = self.mapmodel.gate()
    elif self._use_map:
      mtgt, ptgt = obs['map12'], obs['mappos']
      # Cells the agent has not observed weigh zero, so no gradient is ever
      # taken from terrain it could not have seen. Absent only on runs whose
      # env predates the key, which fall back to the old all-cells behaviour.
      wtgt = obs['mapknown'] if 'mapknown' in obs else None
      mweight = None
      if wtgt is not None:
        # Only TERRAIN is static, so only terrain may be graded against later
        # evidence, and only terrain is unknown until seen. The mob planes mean
        # "visible right now" and P_SEEN is the agent's own visitation record:
        # both are facts about step t, known everywhere (outside the window the
        # honest answer is "no mob visible", "not seen"), so they keep the
        # causal target at full weight. Hindsighting them would ask the model
        # where cows will wander and where it will walk; masking P_SEEN would
        # leave it unable to say "I have not seen this cell" -- the one thing
        # the actor needs to tell real terrain from a guess.
        # Position is not hindsighted either: where the agent stands at step t
        # is a fact about step t.
        P = mtgt.shape[-1]
        terrain = jnp.arange(P) < cmap.P_MOB_PASSIVE          # planes 0..12
        if self.config.mapmodel.hindsight:
          mtgt = jnp.where(terrain, mapmod.segment_last(mtgt, reset), mtgt)
          wtgt = mapmod.segment_last(wtgt, reset)
        mweight = jnp.where(terrain, wtgt[..., None], 1.0)
      mloss, ploss = self.mapmodel.loss(
          deter2_step, sg(mtgt), sg(ptgt),
          None if mweight is None else sg(mweight))
      # Graded where a window closes -- the state that just took in the
      # window, against the map as of that step -- about one step in `tick`,
      # so the mean over (B, T) keeps the old per-tick / tick scale.
      losses['map'] = mloss * wclose
      losses['mappos'] = ploss * wclose
      mean = lambda x: (f32(x) * wclose).sum() / nclose
      metrics['map/bce'] = mean(mloss)                  # per tick, over all cells
      metrics['map/bce_cell'] = mean(mloss) / (
          self._map_coarse ** 2 * int(self.config.mapmodel.planes))
      metrics['map/posce'] = mean(ploss)                # chance = ln(144) = 4.97
      if wtgt is not None:
        metrics['map/observed'] = wtgt.mean()   # share of cells carrying signal
      metrics['map/posacc'] = mean(
          self.mapmodel.decode(deter2_step)[1].argmax(-1) == ptgt)
      metrics['map/gate'] = self.mapmodel.gate()

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    # The rollout must sample from the SAME policy the loss differentiates, or
    # imag_loss's REINFORCE term scores actions drawn from a different
    # distribution. RSSM-1's imagine() carries only (deter, stoch) through its
    # scan, so a crop that slides step by step cannot be threaded in here --
    # the rollout therefore uses the crop frozen at the imagination start,
    # which needs no future actions to compute.
    frozen = None
    if self._use_map and self._map_to_actor:
      start2 = deter2_step[:, -K:].reshape((B * K, -1))
      startfeat, cell0 = self.mapfeat(start2)
      frozen = startfeat[:, 0]
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    if self._learned_goals:
      goal0 = obs['goal'][:, -K:].reshape((B * K,))
      phase0 = obs['gphase'][:, -K:].reshape((B * K,))
      # Where the latent stood when that goal was set: gphase steps back in
      # this replay window. A goal set before the window began is restarted
      # at the window's first step -- its start state AND its step count, so
      # the progress, the reached test and the hold limit all measure from
      # the same state (clamping only the state used to mismatch the count).
      f1 = f32(sg(self.feat2tensor(repfeat)))
      tpos = jnp.arange(T - K, T)[None, :]
      src, phase0 = goal_start(tpos, obs['gphase'][:, -K:])
      gstart0 = jnp.take_along_axis(f1, src[..., None], 1).reshape(
          (B * K, -1))
      phase0 = phase0.reshape((B * K,))
      imgfeat, imgact, goals, phases, grew, hits = self._imagine_learned(
          starts, first, frozen, start2, goal0, phase0, gstart0, H, training)
      flags = jnp.zeros((B * K, H + 1, self._goals), bool)
      reached = flags
    elif self._use_mgr:
      flags0 = obs['goalreach'][:, -K:].reshape((B * K, -1)) > 0.5
      if self._shared:
        goal0 = obs['goal'][:, -K:].reshape((B * K,))
        phase0 = obs['gphase'][:, -K:].reshape((B * K,))
        imgfeat, imgact, goals, phases, flags, reached = self._imagine_hold(
            starts, first, frozen, start2, flags0, goal0, phase0, H, training)
      else:
        imgfeat, imgact, goals, phases, flags, reached = self._imagine_mgr(
            starts, first, frozen, start2, flags0, H, training)
    else:
      policyfn = lambda feat: sample(self._masked(
          self.pol(self.actor2tensor(feat, frozen), 1),
          self._dream_valid(self.feat2tensor(feat), 1)))
      _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
      imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
      lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    # rew/con keep the stock tensor: a wrong map cell must never be able to
    # fabricate imagined reward. Only pol/val/slowval see the map, and slowval
    # must see exactly what val does or the value target drifts from the
    # estimate for good.
    ainp = inp
    if self._use_map and self._map_to_actor:
      if self._map_imag_shift:
        # Stage 5. The map content stays frozen -- imagining teaches you no
        # geography -- but the window slides, so imagining "walk north" finally
        # changes the actor's input and produces a gradient. This makes the
        # rollout slightly off-policy (see `frozen` above); over H=15 steps the
        # agent covers at most ~4 cells of a +/-4-cell crop, so the two inputs
        # overlap heavily. imag_shift False is the exactly-consistent ablation.
        cells = mapmod.dead_reckon(
            cell0, imgact['action'], self._map_coarse, self._map_cell_tiles)
        imgmapfeat, _ = self.mapfeat(start2, cells)
      else:
        imgmapfeat = jnp.repeat(frozen[:, None], imgfeat['deter'].shape[1], 1)
      ainp = self.actor2tensor(imgfeat, imgmapfeat)
    rew = self.rew(inp, 2).pred()
    con = self.con(inp, 2).prob(1)
    actor_rew = rew
    if self._shared:
      ainp = jnp.concatenate([ainp, self._goalfeat(goals, phases)], -1)
      cinp = self._mgr_input(ainp, start2, flags)
      if self._learned_goals:
        event = hits
      else:
        grew, event = self._goal_stream(inp, goals, reached)
      value, slow = self.val(cinp, 2), self.slowval(cinp, 2)
      streams = {
          'game': (rew, value['game'], slow['game'], self.retnorm,
                   self.valnorm, self.advnorm, 1.0),
          'goal': (grew, value['goal'], slow['goal'], self.gretnorm,
                   self.gvalnorm, self.gadvnorm, self._goal_weight)}
      if self._curio:
        # Curiosity for s_t -> s_{t+1}, paid on arrival like the reward.
        cur = self._curiosity(inp[:, :-1], imgact['action'][:, :-1])
        cur = jnp.concatenate([jnp.zeros_like(cur[:, :1]), cur], 1)
        streams['explore'] = (cur, value['explore'], slow['explore'],
                              self.eretnorm, self.evalnorm, self.eadvnorm,
                              self._curio_weight)
        metrics['curiosity/reward'] = cur.mean()
      los, imgloss_out, mets = imag_loss_streams(
          imgact, con,
          self._masked(self.pol(ainp, 2), self._dream_valid(inp, 2)),
          streams,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
      metrics.update(mets)
      imgloss_out['ret'] = imgloss_out['ret/game']
      metrics['manager/reach_rate'] = event.sum(1).mean()
      metrics['manager/goal_rew'] = grew.mean()
      mloss, mmets = self._mgr_loss_shared(
          imgfeat, goals, phases, flags, reached, imgloss_out['weight'],
          frozen, start2, imgloss_out['rscale/game'])
      losses['mpolicy'] = mloss.mean(1).reshape((B, K))
      metrics.update(prefix(mmets, 'manager'))
    elif self._use_mgr:
      ainp = jnp.concatenate([ainp, self._goalfeat(goals, phases)], -1)
      # Goal reward for the step s_{t-1} -> s_t: the change in the predicted
      # progress of the goal that was active when the action was taken.
      phi = jnp.clip(f32(sg(self.gphi(inp, 2).pred())), 0.0, 1.0)
      held = goals[:, :-1]
      pick = lambda p: jnp.take_along_axis(p, held[..., None], -1)[..., 0]
      grew = jnp.where(held == 0, 0.0, pick(phi[:, 1:]) - pick(phi[:, :-1]))
      bonus = self._actor_bonus(goals, reached, H)
      grew = self._goal_reward * grew + self._reach_bonus * bonus
      actor_rew = rew + jnp.concatenate([jnp.zeros_like(grew[:, :1]), grew], 1)
      metrics['manager/goal_rew'] = grew.mean()
      metrics['manager/goal_rew_pos'] = (grew > 0.05).mean()
      metrics['manager/reach_rate'] = bonus.sum(1).mean()
      mlos, mmets = self._mgr_loss(
          imgfeat, goals, flags, reached, rew, con, frozen, start2, H,
          training)
      losses['mpolicy'] = mlos['policy'].mean(1).reshape((B, K))
      losses['mvalue'] = mlos['value'].mean(1).reshape((B, K))
      metrics.update(prefix(mmets, 'manager'))
    if not self._shared:
      los, imgloss_out, mets = imag_loss(
          imgact,
          actor_rew,
          con,
          self._masked(self.pol(ainp, 2), self._dream_valid(inp, 2)),
          self.val(ainp, 2),
          self.slowval(ainp, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
      metrics.update(mets)

    # Replay
    if self._repval:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      if self._use_map and self._map_to_actor:
        d2 = deter2_step[:, -K:].reshape((B * K, -1))
        mf, _ = self.mapfeat(d2)
        inp = self.actor2tensor(feat, mf.reshape((B, K, -1)))
      value, slow = self.val, self.slowval
      if self._shared:
        # The game head on replay, with the goal and segment step the agent
        # really had there -- the same input the imagined rollouts resume.
        inp = jnp.concatenate([inp, self._goalfeat(
            obs['goal'][:, -K:], obs['gphase'][:, -K:])], -1)
        rflags = (jnp.zeros((B, K, self._goals), bool)
                  if self._learned_goals else obs['goalreach'][:, -K:] > 0.5)
        inp = self._mgr_input(inp, deter2_step[:, -K:], rflags)
        value = lambda x, b: self.val(x, b)['game']
        slow = lambda x, b: self.slowval(x, b)['game']
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          value(inp, 2),
          slow(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry, map_carry)
    entries = (enc_entries, dyn_entries, dec_entries, map_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry, map_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    metrics.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, map_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry, map_carry)
    stepid = data['stepid']
    keys = list(self.obs_space)
    if self._use_mgr and self._shared:
      keys += ['goal', 'gphase']      # what the agent pursued at each step
    obs = {k: data[k] for k in keys}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec', 'map')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry),
        self._map_truncate(lhs(entries[3]), map_carry))
    rep_obs = {k: rhs(data[k]) for k in keys}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def imag_loss_streams(
    act, con, policy, streams,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  """imag_loss with several reward streams, one value head per stream.

  ``streams`` maps a name to (rew, value, slowvalue, retnorm, valnorm, advnorm,
  weight). Each stream gets its own lambda-return, its own value loss and its
  own return normalisation; the policy follows sum_k weight_k * adv_k of the
  NORMALISED advantages. So each stream counts by its weight, not by its raw
  scale -- the fix for v1.1, where the goal bonuses and the game reward were
  summed into one return at sizes nobody had chosen on purpose.
  """
  losses, metrics, outs = {}, {}, {}
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  total_adv = 0.0
  value_loss = 0.0
  for name, (rew, value, slowvalue, retnorm, valnorm, advnorm, w) in (
      streams.items()):
    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)
    roffset, rscale = retnorm(ret, update)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    total_adv = total_adv + w * (adv - aoffset) / ascale
    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
    value_loss = value_loss + sg(weight[:, :-1]) * (
        value.loss(sg(tar_padded)) +
        slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]
    outs[f'ret/{name}'] = ret
    outs[f'rscale/{name}'] = rscale
    metrics[f'{name}/adv_mag'] = jnp.abs(adv).mean()
    metrics[f'{name}/rew'] = rew.mean()
    metrics[f'{name}/ret'] = ((ret - roffset) / rscale).mean()
    metrics[f'{name}/val'] = val.mean()
    metrics[f'{name}/rscale'] = rscale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  losses['policy'] = sg(weight[:, :-1]) * -(
      logpi * sg(total_adv) + actent * sum(ents.values()))
  losses['value'] = value_loss
  metrics['con'] = con.mean()
  metrics['weight'] = weight.mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)
  outs['weight'] = weight
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
