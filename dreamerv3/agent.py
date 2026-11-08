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
from . import craftax_valid as cvalid
from . import mapmodel as mapmod
from . import rssm
from embodied.jax import outs as jouts

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
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

    # 'ach' (Craftax) is the multi-hot of achievements unlocked this step. It
    # is kept in replay for the scoring tools but is not an observation: it
    # must never reach the encoder or be reconstructed.
    # 'map12'/'mappos'/'mapseen'/'mapknown' are RSSM-2 TARGETS -- like 'ach'
    # they are supervision only and must never reach the encoder or decoder.
    # 'valid' (experiment B) is an INPUT only when valid.input is set (B1);
    # otherwise it is the label for the feasibility head, like 'ach'.
    exclude = ('is_first', 'is_last', 'is_terminal', 'reward', 'ach',
               'map12', 'mappos', 'mapseen', 'mapknown')
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
    # A second, slower world model holding a coarse map of the level, trained
    # on the env's map targets. Off by default: with mapmodel.enabled False
    # this file is plain DreamerV3.
    self._use_map = bool(config.mapmodel.enabled) and 'map12' in obs_space
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
    # obs['valid'] transcribes the game's rules, so B1 (valid.input) and B2
    # (valid.mask) count as cheating.
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
    # the actor while acting.
    #   label    for each real step (s, a, s'): would NOOP have explained the
    #            outcome as well? evidence = KL(post' || prior(s, NOOP))
    #            - KL(post' || prior(s, a)), the world model's own judgement of
    #            how much better the action explains what really happened.
    #            Above valid.evidence nats, the action did something.
    #   partial  only the action actually taken gets a label each step.
    #   explore  a masked action is never taken, so a wrong "impossible" would
    #            never be corrected; with probability valid.explore a step
    #            ignores the mask.
    # obs['valid'] (the rules) is kept only as a measuring stick in metrics --
    # but B3 still needs env.craftax.valid_obs True to build the head, and it
    # keeps two pieces of hand-written action knowledge: the basic actions
    # (cvalid.BASIC: NOOP, moves, DO) are exempt from the mask, and action 0
    # is assumed to be NOOP (the reference). See README.
    self._valid_learned = self._valid_mask and bool(config.valid.learned)
    # exempt_basic False learns the mask for NOOP, the moves and DO too;
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
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    if self._valid_mask:
      self.modules.append(self.feas)
    if self._use_map:
      self.modules.append(self.mapmodel)
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    if not self._use_map:
      scales.pop('map', None)       # the agent asserts losses/scales match
      scales.pop('mappos', None)
    if not self._valid_mask:
      scales.pop('feas', None)
    self.scales = scales

  @property
  def policy_keys(self):
    keys = ['enc', 'dyn', 'dec', 'pol']
    if self._use_map and self._map_to_actor:
      # The acting path decodes the map itself, so RSSM-2's weights have to
      # ship to the policy device alongside the actor's.
      keys.append('mapmodel')
    if self._valid_learned:
      keys += ['feas', 'feasclock']  # B3: the learned mask acts
    return '^(' + '|'.join(keys) + ')/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
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
    carry['movesum'] = jnp.zeros((batch_size, 2), f32)
    carry['count'] = jnp.zeros((batch_size, 1), f32)   # step within window
    carry['n'] = jnp.zeros((batch_size, 1), f32)       # steps summed so far
    return nn.cast(carry)

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    # mode is static (traced once per value). 'greedy' takes the most likely
    # action; every other mode samples.
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
      map_carry, mapfeat = self._map_act(map_carry, feat, prevact, reset)
    policy = self.pol(self.actor2tensor(feat, mapfeat), bdims=1)
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
    if mode == 'probe':
      # Inspection only (tools/watch_agent.py): the action distribution the
      # actor actually produced. Gated on 'probe' because every other caller
      # feeds policy outputs into a replay buffer that asserts the key set.
      out['policy_prob'] = jax.nn.softmax(policy['action'].logits, -1)
      if self._use_map and self._map_to_actor:
        # RSSM-2's map belief. mapmodel's params are only on the policy device
        # when to_actor is set (see policy_keys), which every map run uses.
        out['map_pred'] = jax.nn.sigmoid(
            self.mapmodel.decode(map_carry['deter2'])[0])
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

    The basic actions (NOOP, moves, DO) are always allowed unless
    valid.exempt_basic is off. The mask is added after the categorical's
    unimix, so no uniform floor leaks back onto a masked action. ``valid``
    None (no mask) returns the policy unchanged.
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
      # No assumption that action 0 does nothing -- compare with a randomly
      # chosen other action instead.
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
    move = mapmod.action_deltas(prevact['action'])
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

    if self._use_map:
      tick = self._map_tick
      # sg on the way IN: the map loss must never reach RSSM-1, or we recreate
      # the very gradient competition the two-model split exists to prevent.
      feat1 = sg(self.feat2tensor(repfeat))
      moves = mapmod.action_deltas(prevact['action'])
      # The acting step, scanned over the batch: each step carries the state
      # of the last closed window, exactly as the actor saw it online.
      keys = ('deter2', 'featsum', 'movesum', 'count', 'n')
      mc = nn.cast({k: map_carry[k] for k in keys})   # as _map_step returns

      def mapstep(c, xs):
        c, closed, inp = self._map_step(c, *xs)
        return c, (c['deter2'], c['count'], closed, inp)

      mc, (deter2_step, count_step, closed, _) = nj.scan(
          mapstep, mc, (feat1, moves, reset), axis=1)
      map_carry = {**map_carry, **mc}
      map_entries = dict(deter2=deter2_step, count=count_step)
      wclose = f32(closed)                                    # (B, T)
      nclose = jnp.maximum(wclose.sum(), 1.0)
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
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1),
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
    if self.config.repval_loss:
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
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
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
    obs = {k: data[k] for k in self.obs_space}
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
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
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
