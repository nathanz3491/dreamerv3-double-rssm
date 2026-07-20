"""Craftax (Symbolic) adapter for the embodied framework  --  Phase 0.

Wraps a single (non-batched) Craftax env behind embodied.Env, mirroring the
repo's own crafter.py. Two additions beyond a plain wrapper, both load-bearing
for the compositional-reward / frontier-exploration plan:

  1. Emits ``obs['ach']`` -- a float32 multi-hot of achievements newly unlocked
     *this step* (length = NUM_ACHIEVEMENTS). This is the training target for the
     reward-cause head (Phase 1) and the trigger signal for frontier saving
     (Phase 4). It is a LABEL: exclude it from the encoder/decoder in agent.py.

  2. ``save_state()`` / ``reset_to()`` -- checkpoint and restore the functional
     Craftax EnvState, which is what makes Go-Explore-style frontier return
     essentially free (Phase 4).

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

  def __init__(self, task='symbolic', size=None, seed=0, logs=False):
    assert task in ('symbolic',), task  # pixels: add 'Craftax-Pixels-v1' below
    import jax
    from craftax.craftax_env import make_craftax_env_from_name
    from dreamerv3.craftax_features import NUM_ACHIEVEMENTS

    self._jax = jax
    self._env = make_craftax_env_from_name(
        'Craftax-Symbolic-v1', auto_reset=False)
    self._params = self._env.default_params
    self._max_timesteps = int(self._params.max_timesteps)
    self._num_ach = NUM_ACHIEVEMENTS

    obs_space = self._env.observation_space(self._params)
    self._obs_dim = int(np.prod(obs_space.shape))
    self._num_actions = int(self._env.action_space(self._params).n)

    # Jit the pure env transitions once; params captured as a closure constant.
    self._reset_fn = jax.jit(lambda key: self._env.reset(key, self._params))
    self._step_fn = jax.jit(
        lambda key, state, act: self._env.step(key, state, act, self._params))
    self._get_obs_fn = jax.jit(lambda state: self._env.get_obs(state))

    self._key = jax.random.PRNGKey(seed)
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
    key, subkey = self._jax.random.split(self._key)
    self._key = key
    act = self._jax.numpy.asarray(int(action['action']), self._jax.numpy.int32)
    obs, self._state, reward, done, info = self._step_fn(
        subkey, self._state, act)
    self._done = bool(done)
    reward = float(reward)
    self._reward += reward
    self._length += 1
    # Craftax's is_game_over (hence `done`) fires on death, boss-defeat AND
    # timeout, and its `discount` is 0 for all three. For correct value
    # bootstrapping we mark only non-timeout endings as terminal -- a timeout is
    # a truncation, not a true absorbing state (this is the intent the reference
    # crafter.py encodes via `info['discount'] == 0`; Craftax does not expose
    # timeout separately in `info`, so we derive it from the state timestep).
    timeout = bool(np.asarray(self._state.timestep) >= self._max_timesteps)
    return self._obs(
        np.asarray(obs, np.float32), reward, self._state,
        is_last=self._done,
        is_terminal=self._done and not timeout)

  def _reset(self):
    key, subkey = self._jax.random.split(self._key)
    self._key = key
    obs, self._state = self._reset_fn(subkey)
    self._done = False
    self._episode += 1
    self._length = 0
    self._reward = 0.0
    self._prev_ach = np.zeros(self._num_ach, np.float32)
    return self._obs(np.asarray(obs, np.float32), 0.0, self._state,
                     is_first=True)

  # --- achievement featurization ---------------------------------------------
  def _newly_unlocked(self, state):
    """Multi-hot of achievements that flipped False->True since last step."""
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
      obs['log/achievements'] = np.int32(
          np.asarray(state.achievements).sum())
    return obs

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
    obs = np.asarray(self._get_obs_fn(state), np.float32)
    return dict(
        vector=obs,
        ach=np.zeros(self._num_ach, np.float32),
        reward=np.float32(0.0),
        is_first=True, is_last=False, is_terminal=False,
        **({'log/reward': np.float32(0.0),
            'log/achievements': np.int32(np.asarray(state.achievements).sum())}
           if self._logs else {}),
    )
