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
               death_penalty=1.0, maintain_scale=0.1,
               maintain_threshold=3.0):
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
    #   'death'    one penalty at death. Minimal: makes dying cost something
    #              absolute without saying how to avoid it, so whether the agent
    #              learns to walk to water is a real result rather than a
    #              behaviour we paid for directly.
    #   'maintain' pay for restoring a meter that is already low. Dense and
    #              faster, but it pays for the ACT, so an agent can farm it by
    #              letting a meter drain in order to refill it.
    #   'both'     both.
    #
    # Off by default: 'none' reproduces stock Craftax reward exactly. Shaping
    # changes `reward`, never `log/achievements`, which is read straight from
    # the env state -- so achievement counts stay comparable to unshaped runs
    # and to the leaderboard.
    assert survival in ('none', 'death', 'maintain', 'both'), survival
    self._survival = survival
    self._death_penalty = float(death_penalty)
    self._maintain_scale = float(maintain_scale)
    self._maintain_threshold = float(maintain_threshold)
    self._prev_meters = None

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
      reward += self._survival_reward(self._state, self._done and not timeout)
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

  def _survival_reward(self, state, is_terminal):
    """Shaping reward. Caller already holds the transfer guard."""
    if self._survival == 'none':
      return 0.0
    meters = self._meters(state)
    bonus = 0.0
    if self._survival in ('maintain', 'both') and self._prev_meters is not None:
      # Pay only for a restore that happened while the meter was actually low,
      # so ordinary topping-up at full health earns nothing.
      was_low = self._prev_meters <= self._maintain_threshold
      restored = meters > self._prev_meters
      bonus += self._maintain_scale * float((was_low & restored).sum())
    self._prev_meters = meters
    if is_terminal and self._survival in ('death', 'both'):
      bonus -= self._death_penalty
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
    if self._logs:
      result['log/reward'] = np.float32(0.0)
      result['log/achievements'] = ach_sum
    return result
