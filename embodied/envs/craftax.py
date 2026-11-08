"""Craftax (Symbolic) adapter for the embodied framework.

Wraps a single (non-batched) Craftax env behind embodied.Env, mirroring the
repo's own crafter.py. Beyond a plain wrapper:

  1. Emits ``obs['ach']`` -- a float32 multi-hot of achievements newly unlocked
     *this step* (length = NUM_ACHIEVEMENTS). The scoring tools read it from
     replay. It is not an observation: agent.py excludes it from the encoder
     and decoder.
  2. A timeout is a truncation (``is_last``), not a terminal state; only death
     and the boss ending are ``is_terminal``.
  3. ``reseed(seed)`` builds the next world from the seed alone, so evaluation
     can score every agent on the same fixed worlds (tools/death_eval.py).
  4. ``raw_reward`` holds Craftax's own reward for the last step, before any
     shaping.
  5. ``save_state()`` / ``reset_to()`` checkpoint and restore the functional
     Craftax EnvState (used by tools/craft_probe.py).

The goal-manager family adds, each behind its own flag and excluded from the
encoder and decoder by agent.py:

  - ``mapmodel``: RSSM-2's map targets ``obs['map12'|'mappos'|'mapseen'|
    'mapknown']`` (craftax_map), built from what the agent has seen.
  - ``valid_obs``: ``obs['valid']``, the rule mask's flags (craftax_valid).
  - ``goals_obs``: ``obs['goalphi'|'goalreach']`` for the named goals
    (craftax_goals).
  - ``survival``: potential-based reward shaping (craftax_potential).

dreamerv3's embodied.jax installs a global ``jax_transfer_guard='disallow'`` to
catch stray host<->device transfers inside the agent. The env legitimately
moves data across that boundary every step (build an RNG key, run the functional
JAX env, pull the observation back to numpy), so every such region is wrapped in
``jax.transfer_guard('allow')`` -- scoped, so the agent still gets guarded. This
mirrors the reference crafter/gymnax adapters.

Notes:
  - Uses the NoAutoReset env variant so the adapter owns reset (like crafter.py).
  - dreamerv3.craftax_features.validate_against_craftax() asserts that the
    embedded achievement schema matches the installed Craftax.
"""

import elements
import embodied
import numpy as np

# JAX is imported lazily inside __init__ so that non-JAX tooling can import this
# module's symbols without a JAX/GPU install.


class Craftax(embodied.Env):

  def __init__(self, task='symbolic', size=None, seed=0, logs=False,
               mapmodel=False, seen_decay=0.99, map_privileged=False,
               valid_obs=False, goals_obs=False, goals_survival=False,
               survival='none', phi_scale=4.0, phi_gamma=0.997,
               surv_w0=0.75, surv_kappa=1.0, surv_ach_ref=10.0):
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
    # False (default): the map target is a mosaic of the agent's own lit 9x11
    # windows, so no gradient ever reaches cells it has not observed. True
    # reinstates full-map ground truth on every cell, kept only to reproduce the
    # original runs as an ablation -- run it with agent.mapmodel.hindsight False
    # too, since the old targets were never hindsighted. See craftax_map.
    self._map_privileged = bool(map_privileged)
    # Experiment B: obs['valid'], which of the 43 actions can do anything now,
    # computed from the observation vector alone (craftax_valid).
    self._valid_obs = bool(valid_obs)
    if self._valid_obs:
      from dreamerv3 import craftax_valid
      self._V = craftax_valid
    # Two-level agent: obs['goalphi'], progress toward each manager goal,
    # computed from the observation vector alone (craftax_goals) -- a training
    # target for the agent's goal-progress head, never an encoder input -- and
    # obs['goalreach'], which goals have read as reached at any step of this
    # episode. The second is the manager's memory of what it has done: its
    # bonus pays only on a goal's first reach, and the game's own achievement
    # flags are not in the observation. Built from goalphi alone, so it is
    # what an observer of the screen could have kept.
    self._goals_obs = bool(goals_obs)
    if self._goals_obs:
      from dreamerv3 import craftax_goals
      self._G = craftax_goals
      self._goals_survival = bool(goals_survival)
      self._n_goals = len(craftax_goals.names(self._goals_survival))
      self._goalreach = np.zeros(self._n_goals, bool)
    if self._mapmodel:
      from dreamerv3 import craftax_map
      self._M = craftax_map
    self._seen = None
    self._prev_level = None
    self._last_action = 0
    if self._mapmodel:
      self._observed = self._M.ObservedTargets(self._seen_decay)

    # --- survival shaping ----------------------------------------------------
    # Craftax pays for COLLECT_DRINK and EAT_COW exactly ONCE. Every drink after
    # the first is worth zero, so maintenance is unrewarded and -- because a
    # plateaued agent's future is worth little -- dying is nearly free.
    # Craftax's own reward is `new_achievements + 0.1 * health_delta` and has NO
    # terminal term, so dying costs at most -0.9, paid gradually as health
    # drains.
    #
    #   'none'              stock Craftax reward, bit-for-bit.
    #   'potential'         one potential-based term over tech-tree progress
    #                       (craftax_potential). It provably cannot change the
    #                       optimal policy, so it cannot invent cheap optima.
    #   'potential+meters'  (v1.3) adds craftax_potential.survival_potential:
    #                       food and drink as a potential whose weight grows
    #                       with tech progress and with achievements unlocked,
    #                       so thirst costs something every step instead of
    #                       only at a death ~200 steps later.
    #
    # Shaping changes `reward` only, never `log/achievements` or `raw_reward`,
    # so achievement counts stay comparable to unshaped runs. The older
    # five-term 'shaped' mode (run map_survive) is not on this branch: no run
    # of this family used it.
    assert survival in ('none', 'potential', 'potential+meters'), survival
    self._survival = survival
    self._phi_scale = float(phi_scale)
    self._phi_gamma = float(phi_gamma)
    self._prev_phi = None
    self._ach_names = None
    self._meter_kw = dict(w0=float(surv_w0), kappa=float(surv_kappa),
                          ach_ref=float(surv_ach_ref))
    if survival in ('potential', 'potential+meters'):
      from dreamerv3 import craftax_potential
      from craftax.craftax.constants import Achievement
      self._P = craftax_potential
      # By index, not iteration order: Craftax's Achievement enum is not
      # declared in value order (positions 25-66 are shuffled), and the state's
      # achievements array is indexed by value.
      self._ach_names = [Achievement(i).name for i in range(len(Achievement))]

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
        # Not bounded by 1: Craftax divides health, food, drink, energy and mana
        # by 10, and their maxima grow with attributes (max health 8 + str,
        # max mana 6 + 3 * int, up to 13 and 21), and XP is unbounded. The old
        # high=1.0 held only until an agent first levelled up -- which needs
        # XP, which only a new floor gives -- and then failed the space check.
        'vector': elements.Space(np.float32, (self._obs_dim,), 0.0, np.inf),
        'ach': elements.Space(np.float32, (self._num_ach,), 0.0, 1.0),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    if self._mapmodel:
      C, P = self._M.COARSE, self._M.N_PLANES
      # TARGETS, never inputs -- agent.py excludes them from enc/dec
      # alongside 'ach'.
      spaces['map12'] = elements.Space(np.float32, (C, C, P), 0.0, 1.0)
      spaces['mappos'] = elements.Space(np.int32, (), 0, self._M.N_CELLS)
      spaces['mapseen'] = elements.Space(np.float32, (C, C), 0.0, 1.0)
      # Supervision WEIGHT: fraction of each cell's 16 tiles the agent has
      # observed. Cells at 0.0 carry no gradient.
      spaces['mapknown'] = elements.Space(np.float32, (C, C), 0.0, 1.0)
    if self._valid_obs:
      spaces['valid'] = elements.Space(np.float32, (self._num_actions,), 0.0, 1.0)
    if self._goals_obs:
      spaces['goalphi'] = elements.Space(
          np.float32, (self._n_goals,), 0.0, 1.0)
      spaces['goalreach'] = elements.Space(
          np.float32, (self._n_goals,), 0.0, 1.0)
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

  def reseed(self, seed):
    """Build the next world (and its rollout) from `seed` alone.

    reset and every step draw from one key stream, so without this the world
    an episode starts in depends on how many steps earlier episodes took --
    two agents with the same seed would meet different worlds after their
    first episode. Evaluation reseeds before each episode.
    """
    with self._jax.transfer_guard('allow'):
      self._key = self._jax.random.PRNGKey(int(seed))

  def step(self, action):
    if action['reset'] or self._done:
      return self._reset()
    with self._jax.transfer_guard('allow'):
      key, subkey = self._jax.random.split(self._key)
      self._key = key
      self._last_action = int(action['action'])
      act = self._jax.numpy.asarray(self._last_action, self._jax.numpy.int32)
      obs, self._state, reward, done, info = self._step_fn(
          subkey, self._state, act)
      self._done = bool(done)
      reward = float(reward)
      self.raw_reward = reward           # Craftax's own, before any shaping
      obs = np.asarray(obs, np.float32)
      # Craftax's is_game_over (hence `done`) fires on death, boss-defeat AND
      # timeout, and its `discount` is 0 for all three. For correct value
      # bootstrapping we mark only non-timeout endings as terminal -- a timeout
      # is a truncation, not a true absorbing state (the intent the reference
      # crafter.py encodes via `info['discount'] == 0`; Craftax does not expose
      # timeout separately in `info`, so we derive it from the state timestep).
      timeout = bool(np.asarray(self._state.timestep) >= self._max_timesteps)
      reward += self._survival_reward(
          self._state, self._done and not timeout)
    self._reward += reward
    self._length += 1
    return self._obs(
        obs, reward, self._state,
        is_last=self._done,
        is_terminal=self._done and not timeout)

  def _potential(self, state):
    """PHI(s) for the potential shaping modes. Caller holds the guard."""
    phi = self._P.potential(state, self._ach_names, self._phi_scale)
    if self._survival == 'potential+meters':
      # In achievement units already: not divided by phi_scale.
      phi += self._P.survival_potential(
          state, self._ach_names, **self._meter_kw)
    return phi

  def _survival_reward(self, state, is_terminal):
    """Shaping reward: gamma * PHI(s') - PHI(s). Caller holds the guard."""
    if self._survival == 'none':
      return 0.0
    phi = self._potential(state)
    bonus = self._P.shaped(
        self._prev_phi, phi, self._phi_gamma, terminal=is_terminal)
    self._prev_phi = phi
    return bonus

  def _reset(self):
    with self._jax.transfer_guard('allow'):
      key, subkey = self._jax.random.split(self._key)
      self._key = key
      obs, self._state = self._reset_fn(subkey)
      obs = np.asarray(obs, np.float32)
      # PHI of the start state, so the first real step is shaped like every
      # other: gamma * PHI(s1) - PHI(s0). Leaving it unset paid that step
      # nothing, which over an episode telescopes to -gamma * PHI(s1) -- a
      # sum that depends on the first action instead of being a constant.
      self._prev_phi = (
          self._potential(self._state)
          if self._survival in ('potential', 'potential+meters') else None)
    self._done = False
    self.raw_reward = 0.0
    self._episode += 1
    self._length = 0
    self._reward = 0.0
    self._prev_ach = np.zeros(self._num_ach, np.float32)
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
      obs.update(self._map_targets(state, vector, is_first))
    if self._valid_obs:
      obs['valid'] = self._V.valid_actions(vector).astype(np.float32)
    if self._goals_obs:
      obs.update(self._goal_obs(vector, is_first))
    if self._logs:
      obs['log/reward'] = np.float32(reward)
      with self._jax.transfer_guard('allow'):
        obs['log/achievements'] = np.int32(
            np.asarray(state.achievements).sum())
    return obs

  def _goal_obs(self, vector, is_first):
    phi = self._G.progress(vector, self._goals_survival)
    if is_first:
      self._goalreach[:] = False
    self._goalreach |= phi >= 1.0
    return dict(goalphi=phi, goalreach=self._goalreach.astype(np.float32))

  # --- map-model targets (training supervision only) --------------------------
  def _map_targets(self, state, vector, is_first):
    """Coarse map, coarse position and cumulative visitation for RSSM-2.

    Honest mode (default) hands ``ObservedTargets`` the observation vector and
    the action just taken -- and nothing else. It never sees ``state``, so no
    label can contain terrain the agent has not seen or a coordinate it could
    not have worked out. Privileged mode is the old setup, kept as an ablation:
    true map on every cell, true coordinates.

    Both are per-episode AND per-level: Craftax has 9 levels, each its own 48x48
    map, and v1 models only the level the agent is on (design SS7.1).
    """
    if not self._map_privileged:
      return self._observed.step(vector, self._last_action, is_first)
    with self._jax.transfer_guard('allow'):
      level = int(state.player_level)
      if is_first or level != self._prev_level:
        self._seen = None
      self._prev_level = level
      self._seen = self._M.update_seen(self._seen, state, self._seen_decay)
      # Ground truth everywhere AND every cell supervised. Emitting the observed
      # fraction here would mask unseen cells away again and quietly turn the
      # ablation into a copy of the honest run.
      return dict(
          map12=self._M.coarse_map(state, self._seen),
          mappos=self._M.coarse_pos(state),
          mapseen=self._seen.astype(np.float32),
          mapknown=np.ones((self._M.COARSE, self._M.COARSE), np.float32),
      )

  # --- checkpoint/restore ----------------------------------------------------
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
      result.update(self._map_targets(state, obs, is_first=True))
    if self._valid_obs:
      result['valid'] = self._V.valid_actions(obs).astype(np.float32)
    if self._goals_obs:
      result.update(self._goal_obs(obs, is_first=True))
    if self._logs:
      result['log/reward'] = np.float32(0.0)
      result['log/achievements'] = ach_sum
    return result
