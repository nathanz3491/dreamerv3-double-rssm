"""Craftax (Symbolic) adapter for the embodied framework  --  Phase 0.

Wraps a single (non-batched) Craftax env behind embodied.Env, mirroring the
repo's own crafter.py. Two additions beyond a plain wrapper, both load-bearing
for the compositional-reward / frontier-exploration plan:

  0. Optionally emits the map-model TARGETS ``obs['map12'|'mappos'|'mapseen']``
     when ``mapmodel=True`` -- privileged supervision for RSSM-2, excluded from
     the encoder/decoder exactly like ``ach``. See ``design-map-model.md``.
  1. Emits ``obs['ach']`` -- a float32 multi-hot of achievements newly unlocked
     *this step* (length = NUM_ACHIEVEMENTS). This is the training target for the
     reward-cause head (Phase 1) and the trigger signal for frontier saving
     (Phase 4). It is a LABEL: exclude it from the encoder/decoder in agent.py.

  2. ``save_state()`` / ``reset_to()`` -- checkpoint and restore the functional
     Craftax EnvState, which is what makes Go-Explore-style frontier return
     essentially free (Phase 4).

dreamerv3's embodied.jax installs a global ``jax_transfer_guard='disallow'`` to
catch stray host<->device transfers inside the agent. The env legitimately
moves data across that boundary every step (build an RNG key, run the functional
JAX env, pull the observation back to numpy), so every such region is wrapped in
``jax.transfer_guard('allow')`` -- scoped, so the agent still gets guarded. This
mirrors the reference crafter/gymnax adapters.

Notes / verify on the GPU box:
  - Uses the NoAutoReset env variant so the adapter owns reset (like crafter.py).
  - `python -c "from craftax_features import validate_against_craftax as v; v()"`
    to assert the embedded achievement schema matches your installed Craftax.
"""

import elements
import embodied
import numpy as np

# JAX is imported lazily inside __init__ so that non-JAX tooling can import this
# module's symbols without a JAX/GPU install.


class Craftax(embodied.Env):

  def __init__(self, task='symbolic', size=None, seed=0, logs=False,
               mapmodel=False, seen_decay=0.99, survival='none',
               surv_alive=0.005, surv_death=5.0, surv_restore=0.3,
               surv_threshold=3.0, surv_kill=0.5, surv_idle=1.0,
               surv_idle_steps=30, phi_scale=4.0, phi_gamma=0.997):
    assert task in ('symbolic',), task  # pixels: add 'Craftax-Pixels-v1' below
    import jax
    from craftax.craftax_env import make_craftax_env_from_name
    from dreamerv3.craftax_features import NUM_ACHIEVEMENTS

    self._jax = jax
    self._env = make_craftax_env_from_name(
        'Craftax-Symbolic-v1', auto_reset=False)
    self._num_ach = NUM_ACHIEVEMENTS

    self._mapmodel = bool(mapmodel)
    self._seen_decay = float(seen_decay)
    if self._mapmodel:
      from dreamerv3 import craftax_map
      self._M = craftax_map
    self._seen = None
    self._prev_level = None

    # --- survival shaping ----------------------------------------------------
    # Craftax pays for COLLECT_DRINK and EAT_COW exactly ONCE. Every drink after
    # the first is worth zero, so maintenance is unrewarded and -- because a
    # plateaued agent's future is worth little -- dying is nearly free. Measured
    # at 689k steps: 45% of deaths are thirst, 23% hunger, and the agent dies
    # with mean drink 2.45 / food 2.90 still in the tank while knowing exactly
    # where the water is (map position accuracy 0.73).
    #
    # Craftax's own reward is `new_achievements + 0.1 * health_delta` and has NO
    # terminal term, so dying costs at most -0.9, paid gradually as health
    # drains. Measured at 1.1M steps: 55% of deaths are thirst, COLLECT_DRINK
    # fires in only 43% of episodes, and MAKE_WOOD_PICKAXE has never fired in 60
    # episodes despite a table being built in 35% of them.
    #
    #   'none'    stock Craftax reward, bit-for-bit.
    #   'shaped'  the five terms below, each independently scalable.
    #
    # SCALE THE TERMS AGAINST ACHIEVEMENTS, which stay the largest single share
    # of the budget -- DreamerV3 normalises returns before the policy gradient,
    # so a term an order of magnitude larger than the rest does not merely
    # dominate, it scales everything else down into noise. At these defaults, a
    # ~294-step episode earns roughly:
    #
    #   achievements  ~4.3  (48%)   <- still the biggest term
    #   restore       ~2.4  (26%)   <- the one that answers thirst deaths
    #   alive         ~1.5  (16%)
    #   kill          ~0.3
    #   idle          ~-0.5
    #   death         -5.0  once
    #
    # `alive` at 0.005/step is worth 0.005/(1-gamma) = ~1.7 in value terms at
    # horizon 333, so being alive is worth about 1.7 achievements at any moment
    # -- a real pull toward survival that cannot outvote the tech tree.
    #
    # `death` at 5.0 is deliberately mild. It makes risking death for an
    # achievement rational whenever P(death) < ~20%, so the agent will still
    # fight a zombie it expects to beat; a large penalty buys survival by making
    # the agent refuse to explore, mine or fight, which is where the remaining
    # achievements live.
    #
    # Shaping changes `reward` only, never `log/achievements`, which is read
    # straight from the env state -- so achievement counts stay comparable to
    # unshaped runs and to the leaderboard.
    # 'potential' is the successor to 'shaped': one potential-based term over
    # tech-tree progress instead of five hand-written bonuses. It provably
    # cannot change the optimal policy, so it cannot invent the cheap optima
    # that made 'shaped' collect HALF the achievements of no shaping at all
    # (1.73 vs 3.49 at 500k). 'shaped' is kept only to reproduce that result.
    assert survival in ('none', 'shaped', 'potential'), survival
    self._survival = survival
    self._phi_scale = float(phi_scale)
    self._phi_gamma = float(phi_gamma)
    self._prev_phi = None
    self._ach_names = None
    if survival == 'potential':
      from dreamerv3 import craftax_potential
      from craftax.craftax.constants import Achievement
      self._P = craftax_potential
      self._ach_names = [a.name for a in Achievement]
    self._surv = dict(
        alive=float(surv_alive),          # per step; 0.1 == 1 point per 10 steps
        death=float(surv_death),          # subtracted once, on death not timeout
        restore=float(surv_restore),      # per meter refilled while low
        threshold=float(surv_threshold),  # what counts as "low"
        kill=float(surv_kill),            # per hostile mob killed with DO
        idle=float(surv_idle),            # penalty when nothing has changed
        idle_steps=int(surv_idle_steps))
    self._prev_meters = None
    self._prev_hostiles = None
    self._prev_sig = None
    self._idle_for = 0

    # Everything here touches the host<->device boundary (params pytree, space
    # bounds, RNG key), so it runs under an allow scope; see module docstring.
    with jax.transfer_guard('allow'):
      self._params = self._env.default_params
      self._max_timesteps = int(self._params.max_timesteps)
      obs_space = self._env.observation_space(self._params)
      self._obs_dim = int(np.prod(obs_space.shape))
      self._num_actions = int(self._env.action_space(self._params).n)
      self._key = jax.random.PRNGKey(seed)

    # Jit the pure env transitions once; params captured as a closure constant.
    # (Defining a jit does not transfer, so it stays outside the allow scope.)
    self._reset_fn = jax.jit(lambda key: self._env.reset(key, self._params))
    self._step_fn = jax.jit(
        lambda key, state, act: self._env.step(key, state, act, self._params))
    self._get_obs_fn = jax.jit(lambda state: self._env.get_obs(state))

    self._state = None
    self._prev_ach = np.zeros(self._num_ach, np.float32)
    self._done = True
    self._logs = logs
    self._episode = 0
    self._length = 0
    self._reward = 0.0

  # --- embodied.Env interface -------------------------------------------------
  @property
  def obs_space(self):
    spaces = {
        'vector': elements.Space(np.float32, (self._obs_dim,), 0.0, 1.0),
        'ach': elements.Space(np.float32, (self._num_ach,), 0.0, 1.0),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    if self._mapmodel:
      C, P = self._M.COARSE, self._M.N_PLANES
      # Privileged TARGETS, never inputs -- agent.py excludes them from
      # enc/dec alongside 'ach'.
      spaces['map12'] = elements.Space(np.float32, (C, C, P), 0.0, 1.0)
      spaces['mappos'] = elements.Space(np.int32, (), 0, self._M.N_CELLS)
      spaces['mapseen'] = elements.Space(np.float32, (C, C), 0.0, 1.0)
    if self._logs:
      spaces['log/reward'] = elements.Space(np.float32)
      spaces['log/achievements'] = elements.Space(np.int32)
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._num_actions),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      return self._reset()
    with self._jax.transfer_guard('allow'):
      key, subkey = self._jax.random.split(self._key)
      self._key = key
      act = self._jax.numpy.asarray(
          int(action['action']), self._jax.numpy.int32)
      obs, self._state, reward, done, info = self._step_fn(
          subkey, self._state, act)
      self._done = bool(done)
      reward = float(reward)
      obs = np.asarray(obs, np.float32)
      # Craftax's is_game_over (hence `done`) fires on death, boss-defeat AND
      # timeout, and its `discount` is 0 for all three. For correct value
      # bootstrapping we mark only non-timeout endings as terminal -- a timeout
      # is a truncation, not a true absorbing state (the intent the reference
      # crafter.py encodes via `info['discount'] == 0`; Craftax does not expose
      # timeout separately in `info`, so we derive it from the state timestep).
      timeout = bool(np.asarray(self._state.timestep) >= self._max_timesteps)
      reward += self._survival_reward(
          self._state, int(action['action']), self._done and not timeout)
    self._reward += reward
    self._length += 1
    return self._obs(
        obs, reward, self._state,
        is_last=self._done,
        is_terminal=self._done and not timeout)

  def _meters(self, state):
    return np.array([
        float(state.player_food), float(state.player_drink),
        float(state.player_energy)], np.float32)

  ACTION_DO = 5                              # constants.Action.DO

  def _hostiles(self, state):
    """Alive hostile mobs on the current level."""
    level = int(state.player_level)
    total = 0
    for name in ('melee_mobs', 'ranged_mobs'):
      mobs = getattr(state, name, None)
      if mobs is None:
        continue
      mask = np.asarray(mobs.mask)
      total += int(mask[level].sum() if mask.ndim > 1 else mask.sum())
    return total

  def _progress_signature(self, state):
    """Scalars that change iff the agent did something that mattered."""
    inv = state.inventory
    fields = [getattr(inv, f) for f in (
        'wood', 'stone', 'coal', 'iron', 'diamond', 'sapling',
        'pickaxe', 'sword', 'bow', 'arrows', 'armour', 'torches')]
    return (
        tuple(np.asarray(state.player_position).reshape(-1).tolist()),
        int(np.asarray(state.achievements).sum()),
        tuple(int(np.asarray(f).sum()) for f in fields))

  def _survival_reward(self, state, action, is_terminal):
    """Shaping reward. Caller already holds the transfer guard.

    Five terms, each independently scalable; see the ctor for the scaling
    warning. Every term is repeatable on purpose -- Craftax's own achievements
    fire once, which is exactly why maintenance never became worth doing.
    """
    if self._survival == 'none':
      return 0.0
    if self._survival == 'potential':
      phi = self._P.potential(state, self._ach_names, self._phi_scale)
      bonus = self._P.shaped(self._prev_phi, phi, self._phi_gamma)
      self._prev_phi = phi
      return bonus
    bonus = self._surv['alive']            # paid per step, not per 10, so the
                                           # signal is smooth rather than a
                                           # sawtooth the critic has to model
    meters = self._meters(state)
    if self._prev_meters is not None:
      # Pay only for a restore that happened while the meter was LOW, so
      # drinking at full thirst earns nothing. Energy covers sleeping: sleep is
      # the only thing that refills it.
      was_low = self._prev_meters <= self._surv['threshold']
      restored = meters > self._prev_meters
      bonus += self._surv['restore'] * float((was_low & restored).sum())
    self._prev_meters = meters

    # Kills: a hostile count that drops on a DO action. Mobs also despawn, but
    # despawns are not correlated with the player swinging, so gating on DO
    # keeps the misattribution small.
    hostiles = self._hostiles(state)
    if (self._prev_hostiles is not None and action == self.ACTION_DO
        and hostiles < self._prev_hostiles):
      bonus += self._surv['kill'] * (self._prev_hostiles - hostiles)
    self._prev_hostiles = hostiles

    # Idle: nothing moved, nothing was gained, nothing was unlocked. Charged
    # once per window rather than every step, so standing still is costly but
    # not unboundedly so. Sleeping is explicitly NOT idle -- it moves nothing by
    # design, and it is the only thing that refills energy, so counting it would
    # punish the behaviour the `restore` term pays for.
    sig = self._progress_signature(state)
    asleep = bool(np.asarray(state.is_sleeping))
    if sig != self._prev_sig or asleep:
      self._idle_for = 0
    else:
      self._idle_for += 1
    self._prev_sig = sig
    if self._idle_for >= self._surv['idle_steps']:
      bonus -= self._surv['idle']
      self._idle_for = 0

    if is_terminal:
      bonus -= self._surv['death']
    return bonus

  def _reset(self):
    with self._jax.transfer_guard('allow'):
      key, subkey = self._jax.random.split(self._key)
      self._key = key
      obs, self._state = self._reset_fn(subkey)
      obs = np.asarray(obs, np.float32)
    self._done = False
    self._episode += 1
    self._length = 0
    self._reward = 0.0
    self._prev_ach = np.zeros(self._num_ach, np.float32)
    self._prev_meters = None
    self._prev_hostiles = None
    self._prev_sig = None
    self._idle_for = 0
    self._prev_phi = None          # first step of an episode shapes to zero
    return self._obs(obs, 0.0, self._state, is_first=True)

  # --- achievement featurization ---------------------------------------------
  def _newly_unlocked(self, state):
    """Multi-hot of achievements that flipped False->True since last step."""
    with self._jax.transfer_guard('allow'):
      curr = np.asarray(state.achievements, np.float32).reshape(-1)
    assert curr.shape[0] == self._num_ach, (curr.shape, self._num_ach)
    new = (curr.astype(bool) & ~self._prev_ach.astype(bool)).astype(np.float32)
    self._prev_ach = curr
    return new

  def _obs(self, vector, reward, state,
           is_first=False, is_last=False, is_terminal=False):
    if is_first:
      ach = np.zeros(self._num_ach, np.float32)
    else:
      ach = self._newly_unlocked(state)
    obs = dict(
        vector=vector,
        ach=ach,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
    if self._mapmodel:
      obs.update(self._map_targets(state, is_first))
    if self._logs:
      obs['log/reward'] = np.float32(reward)
      with self._jax.transfer_guard('allow'):
        obs['log/achievements'] = np.int32(
            np.asarray(state.achievements).sum())
    return obs

  # --- map-model targets (privileged; training supervision only) -------------
  def _map_targets(self, state, is_first):
    """Coarse map, coarse position and cumulative visitation for RSSM-2.

    The visitation mask is per-episode AND per-level: Craftax has 9 levels,
    each its own 48x48 map, and v1 models only the level the agent is on
    (design SS7.1), so descending a ladder resets the mask.
    """
    with self._jax.transfer_guard('allow'):
      level = int(state.player_level)
      if is_first or level != self._prev_level:
        self._seen = None
      self._prev_level = level
      self._seen = self._M.update_seen(self._seen, state, self._seen_decay)
      return dict(
          map12=self._M.coarse_map(state, self._seen),
          mappos=self._M.coarse_pos(state),
          mapseen=self._seen.astype(np.float32),
      )

  # --- Phase 4: frontier checkpoint/restore ----------------------------------
  def save_state(self):
    """Return an opaque snapshot (EnvState pytree + achievement history)."""
    return (self._state, self._prev_ach.copy())

  def reset_to(self, snapshot):
    """Restore a snapshot saved by save_state() and return the matching obs.

    The restored obs is is_first=True so the agent starts a fresh carry from a
    real, previously-reached state (Go-Explore return). Kept a real state -- do
    NOT idealize (e.g. refill health): that would create a train/test mismatch.
    """
    state, prev_ach = snapshot
    self._state = state
    self._prev_ach = np.asarray(prev_ach, np.float32).copy()
    self._done = False
    self._length = 0
    self._reward = 0.0
    with self._jax.transfer_guard('allow'):
      obs = np.asarray(self._get_obs_fn(state), np.float32)
      ach_sum = np.int32(np.asarray(state.achievements).sum())
    result = dict(
        vector=obs,
        ach=np.zeros(self._num_ach, np.float32),
        reward=np.float32(0.0),
        is_first=True, is_last=False, is_terminal=False,
    )
    # Must match obs_space exactly -- the agent asserts on the key set, and a
    # restored obs that omits the map targets fails that assert.
    if self._mapmodel:
      result.update(self._map_targets(state, is_first=True))
    if self._logs:
      result['log/reward'] = np.float32(0.0)
      result['log/achievements'] = ach_sum
    return result
