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
  4. ``raw_reward`` holds Craftax's own reward for the last step.
  5. ``save_state()`` / ``reset_to()`` checkpoint and restore the functional
     Craftax EnvState (used by tools/craft_probe.py).

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
               survival='none'):
    # Reward shaping lives on the family branches; main has the stock reward
    # only. The argument stays so `--env.craftax.survival none` still parses.
    assert survival == 'none', survival
    assert task in ('symbolic',), task  # pixels: add 'Craftax-Pixels-v1' below
    import jax
    from craftax.craftax_env import make_craftax_env_from_name
    from dreamerv3.craftax_features import NUM_ACHIEVEMENTS

    self._jax = jax
    self._env = make_craftax_env_from_name(
        'Craftax-Symbolic-v1', auto_reset=False)
    self._num_ach = NUM_ACHIEVEMENTS

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
      act = self._jax.numpy.asarray(
          int(action['action']), self._jax.numpy.int32)
      obs, self._state, reward, done, info = self._step_fn(
          subkey, self._state, act)
      self._done = bool(done)
      reward = float(reward)
      self.raw_reward = reward           # Craftax's own reward
      obs = np.asarray(obs, np.float32)
      # Craftax's is_game_over (hence `done`) fires on death, boss-defeat AND
      # timeout, and its `discount` is 0 for all three. For correct value
      # bootstrapping we mark only non-timeout endings as terminal -- a timeout
      # is a truncation, not a true absorbing state (the intent the reference
      # crafter.py encodes via `info['discount'] == 0`; Craftax does not expose
      # timeout separately in `info`, so we derive it from the state timestep).
      timeout = bool(np.asarray(self._state.timestep) >= self._max_timesteps)
    self._reward += reward
    self._length += 1
    return self._obs(
        obs, reward, self._state,
        is_last=self._done,
        is_terminal=self._done and not timeout)

  def _reset(self):
    with self._jax.transfer_guard('allow'):
      key, subkey = self._jax.random.split(self._key)
      self._key = key
      obs, self._state = self._reset_fn(subkey)
      obs = np.asarray(obs, np.float32)
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
    if self._logs:
      obs['log/reward'] = np.float32(reward)
      with self._jax.transfer_guard('allow'):
        obs['log/achievements'] = np.int32(
            np.asarray(state.achievements).sum())
    return obs

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
    # Must match obs_space exactly -- the agent asserts on the key set.
    if self._logs:
      result['log/reward'] = np.float32(0.0)
      result['log/achievements'] = ach_sum
    return result
