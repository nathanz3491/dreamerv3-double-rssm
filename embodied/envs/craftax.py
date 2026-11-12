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
  6. ``valid_obs``: ``obs['valid']``, which of the 43 actions can do anything
     now, computed from the observation vector alone (craftax_valid). It is the
     label of the rule mask (agent.valid.mask, counted as cheating) and never
     an encoder input.
  7. ``goals_obs``: ``obs['goalphi']``, progress toward each of the manager's
     13 named goals, and ``obs['goalreach']``, which goals have read as
     reached at any step of this episode (craftax_goals). Both are computed
     from the observation vector alone; neither reaches the encoder. With
     ``goals_survival`` (v1.3, ported from family/v) the two survival goals
     DRINK and EAT follow the 13, so both vectors have 15 entries.
  8. ``quota``: the market agent's quota (dreamerv3/quota.py, design §4)
     adds a penalty to ``reward``: ``mode window`` (v1) when a window passes
     with no new achievement, charging the skipped windows at death;
     ``mode log`` (v2) when the count of achievements is below a logarithmic
     floor fitted to the agent's recent lives. It reads ``obs['ach']`` and
     the step count only. ``raw_reward`` and ``log/reward`` stay Craftax's
     own. With ``logs`` it also emits ``log/quota_pen``, ``_death``,
     ``_floor`` (and ``_a``, ``_c`` in v2): metrics only.
  9. ``survival potential``: v1.2's tech-tree potential shaping (ported from
     family/v), sent as its own key ``obs['shape']`` and never added to
     ``reward``. The agent's ``shp`` head learns it (market12, README).

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
               valid_obs=False, goals_obs=False, goals_survival=False,
               survival='none', phi_scale=4.0, phi_gamma=0.997, quota=None):
    # Reward shaping. 'none' is the stock reward. 'potential' is v1.2's
    # tech-tree potential (family/v, craftax_potential.potential): F =
    # gamma * PHI(s') - PHI(s), with PHI handed back at a true death. Unlike
    # family/v it is NOT added to `reward`: it goes out as its own key
    # `shape`, so the market's score Q-head can learn from the game reward
    # plus the quota alone while the critic and the manager get both
    # (agent.py, `shp`). The meters mode (v1.3) is not ported.
    assert survival in ('none', 'potential'), survival
    assert task in ('symbolic',), task  # pixels: add 'Craftax-Pixels-v1' below
    import jax
    from craftax.craftax_env import make_craftax_env_from_name
    from dreamerv3.craftax_features import NUM_ACHIEVEMENTS

    self._jax = jax
    self._env = make_craftax_env_from_name(
        'Craftax-Symbolic-v1', auto_reset=False)
    self._num_ach = NUM_ACHIEVEMENTS

    self._valid_obs = bool(valid_obs)
    if self._valid_obs:
      from dreamerv3 import craftax_valid
      self._V = craftax_valid
    # goalreach is the manager's memory of what this episode has done: the
    # game's own achievement flags are not in the observation. Built from
    # goalphi alone, so it is what an observer of the screen could have kept.
    self._goals_obs = bool(goals_obs)
    if self._goals_obs:
      from dreamerv3 import craftax_goals
      self._G = craftax_goals
      self._goals_survival = bool(goals_survival)
      self._n_goals = len(craftax_goals.names(self._goals_survival))
      self._goalreach = np.zeros(self._n_goals, bool)
    self._survival = survival
    self._phi_scale = float(phi_scale)
    self._phi_gamma = float(phi_gamma)
    self._prev_phi = None
    if survival == 'potential':
      from dreamerv3 import craftax_potential
      from craftax.craftax.constants import Achievement
      self._P = craftax_potential
      # By index, not iteration order: Craftax's Achievement enum is not
      # declared in value order, and the state's achievements are by value.
      self._ach_names = [Achievement(i).name for i in range(len(Achievement))]
    # The slope quota keeps its running statistics across this env's
    # episodes; `quota` is the config group env.craftax.quota.
    self._quota = None
    if quota and quota.get('enabled', False):
      from dreamerv3 import quota as quotalib
      kw = {k: v for k, v in dict(quota).items() if k != 'enabled'}
      self._quota = quotalib.make(**kw)

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
    if self._valid_obs:
      spaces['valid'] = elements.Space(np.float32, (self._num_actions,), 0.0, 1.0)
    if self._survival == 'potential':
      spaces['shape'] = elements.Space(np.float32)
    if self._goals_obs:
      spaces['goalphi'] = elements.Space(
          np.float32, (self._n_goals,), 0.0, 1.0)
      spaces['goalreach'] = elements.Space(
          np.float32, (self._n_goals,), 0.0, 1.0)
    if self._logs:
      spaces['log/reward'] = elements.Space(np.float32)
      spaces['log/achievements'] = elements.Space(np.int32)
      if self._quota is not None:
        for name in self._quota.LOGS:
          spaces[f'log/quota_{name}'] = elements.Space(np.float32)
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
      obs = self._finite(np.asarray(obs, np.float32))
      # Craftax's is_game_over (hence `done`) fires on death, boss-defeat AND
      # timeout, and its `discount` is 0 for all three. For correct value
      # bootstrapping we mark only non-timeout endings as terminal -- a timeout
      # is a truncation, not a true absorbing state (the intent the reference
      # crafter.py encodes via `info['discount'] == 0`; Craftax does not expose
      # timeout separately in `info`, so we derive it from the state timestep).
      timeout = bool(np.asarray(self._state.timestep) >= self._max_timesteps)
      shape = self._shaping(self._state, self._done and not timeout)
    self._reward += reward
    self._length += 1
    obs = self._obs(
        obs, reward, self._state,
        is_last=self._done,
        is_terminal=self._done and not timeout, shape=shape)
    if self._quota is not None:
      obs['reward'] = np.float32(reward + self._quota.step(
          int(obs['ach'].sum()), terminal=obs['is_terminal'],
          last=obs['is_last']))
      obs.update(self._quota_logs(self._quota.info))
    return obs

  def _reset(self):
    with self._jax.transfer_guard('allow'):
      key, subkey = self._jax.random.split(self._key)
      self._key = key
      obs, self._state = self._reset_fn(subkey)
      obs = self._finite(np.asarray(obs, np.float32))
      # PHI of the start state, so the first real step is shaped like every
      # other (family/v: leaving it unset made the episode's shaping sum
      # depend on the first action).
      self._prev_phi = self._phi(self._state)
    self._done = False
    self.raw_reward = 0.0
    self._episode += 1
    self._length = 0
    self._reward = 0.0
    self._prev_ach = np.zeros(self._num_ach, np.float32)
    if self._quota is not None:
      self._quota.begin()
    return self._obs(obs, 0.0, self._state, is_first=True)

  def _phi(self, state):
    """PHI(s) under 'potential', else None. Caller holds the guard."""
    if self._survival != 'potential':
      return None
    return self._P.potential(state, self._ach_names, self._phi_scale)

  def _shaping(self, state, is_terminal):
    """gamma * PHI(s') - PHI(s), PHI = 0 at a true death; 0.0 under 'none'.
    Caller holds the guard."""
    if self._survival != 'potential':
      return 0.0
    phi = self._phi(state)
    bonus = self._P.shaped(
        self._prev_phi, phi, self._phi_gamma, terminal=is_terminal)
    self._prev_phi = phi
    return bonus

  def _finite(self, vector):
    """Replace NaN in Craftax's observation with 0 and say where it was.

    market_fixed_q1 died at 100k env steps on the trainer's NaN assert with a
    NaN in a sampled `vector`, while every chunk saved to disk was finite. If
    Craftax itself emits a NaN, this keeps one bad step from killing a 19 h
    run and prints the dims, so the source shows in the run's log. Finite
    observations pass through unchanged."""
    bad = np.isnan(vector)
    if bad.any():
      print(f'[craftax] NaN in obs: episode {self._episode} step {self._length}'
            f' dims {np.flatnonzero(bad)[:20].tolist()} ({int(bad.sum())})',
            flush=True)
      vector = np.where(bad, np.float32(0), vector)
    return vector

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
           is_first=False, is_last=False, is_terminal=False, shape=0.0):
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
    if self._valid_obs:
      obs['valid'] = self._V.valid_actions(vector).astype(np.float32)
    if self._survival == 'potential':
      obs['shape'] = np.float32(shape)
    if self._goals_obs:
      obs.update(self._goal_obs(vector, is_first))
    if self._logs:
      obs['log/reward'] = np.float32(reward)
      with self._jax.transfer_guard('allow'):
        obs['log/achievements'] = np.int32(
            np.asarray(state.achievements).sum())
      if is_first:
        obs.update(self._quota_logs())
    return obs

  def _quota_logs(self, info=None):
    """The quota's ``log/quota_*`` metrics (zeros at a life's first step).
    Read-only: they reach the run's metrics, not the reward or the agent."""
    if not (self._logs and self._quota is not None):
      return {}
    info = info or {}
    return {f'log/quota_{name}': np.float32(info.get(name, 0.0))
            for name in self._quota.LOGS}

  def _goal_obs(self, vector, is_first):
    phi = self._G.progress(vector, self._goals_survival)
    if is_first:
      self._goalreach[:] = False
    self._goalreach |= phi >= 1.0
    return dict(goalphi=phi, goalreach=self._goalreach.astype(np.float32))

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
    if self._quota is not None:
      self._quota.begin()
    with self._jax.transfer_guard('allow'):
      obs = self._finite(np.asarray(self._get_obs_fn(state), np.float32))
      ach_sum = np.int32(np.asarray(state.achievements).sum())
      self._prev_phi = self._phi(state)
    result = dict(
        vector=obs,
        ach=np.zeros(self._num_ach, np.float32),
        reward=np.float32(0.0),
        is_first=True, is_last=False, is_terminal=False,
    )
    # Must match obs_space exactly -- the agent asserts on the key set.
    if self._valid_obs:
      result['valid'] = self._V.valid_actions(obs).astype(np.float32)
    if self._survival == 'potential':
      result['shape'] = np.float32(0.0)
    if self._goals_obs:
      result.update(self._goal_obs(obs, is_first=True))
    if self._logs:
      result['log/reward'] = np.float32(0.0)
      result['log/achievements'] = ach_sum
      result.update(self._quota_logs())
    return result
