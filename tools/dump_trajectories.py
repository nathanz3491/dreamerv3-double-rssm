"""Stage 0 -- dump Craftax trajectories as (seed, actions) pairs.

Craftax is a pure function of (seed, action sequence), so we store only that:
~278 int8s per episode instead of ~9MB of observations. Everything else --
observations, true maps, positions -- is regenerated exactly by replay().

Usage:
  python tools/dump_trajectories.py --episodes 2000 --out data/traj.npz
  python tools/dump_trajectories.py --episodes 20 --verify      # determinism gate
"""
import argparse, pathlib
import jax, numpy as np
from craftax.craftax_env import make_craftax_env_from_name


_ENV = _PARAMS = _STEP = _RESET = _NACT = None


def _env():
    """Build the env and jitted fns exactly once (recompiling per episode is slow)."""
    global _ENV, _PARAMS, _STEP, _RESET, _NACT
    if _ENV is None:
        _ENV = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
        _PARAMS = _ENV.default_params
        _STEP, _RESET = jax.jit(_ENV.step), jax.jit(_ENV.reset)
        _NACT = int(_ENV.action_space(_PARAMS).n)
    return _ENV, _PARAMS, _STEP, _RESET, _NACT


def rollout(seed, policy=None, max_steps=1200):
    """Run one episode. Returns (actions, achievements, length)."""
    _, params, step_fn, reset_fn, n_actions = _env()
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    obs, state = reset_fn(k, params)
    acts = []
    for _ in range(max_steps):
        rng, ka, ks = jax.random.split(rng, 3)
        act = policy(obs) if policy else int(jax.random.randint(ka, (), 0, n_actions))
        acts.append(act)
        obs, state, _, done, _ = step_fn(ks, state, act, params)
        if bool(done):
            break
    return np.array(acts, np.int8), np.array(state.achievements, bool), len(acts)


def replay(seed, actions):
    """Regenerate an episode exactly from (seed, actions). Yields states."""
    _, params, step_fn, reset_fn, _ = _env()
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    obs, state = reset_fn(k, params)
    for act in actions:
        # Must mirror rollout()'s 3-way split exactly -- a 2-way split here
        # hands the env a different key and the episode silently diverges.
        rng, _ka, ks = jax.random.split(rng, 3)
        obs, state, _, done, _ = step_fn(ks, state, int(act), params)
        yield obs, state
        if bool(done):
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--out", default="data/traj.npz")
    p.add_argument("--verify", action="store_true",
                   help="Stage 0 gate: replay must reproduce achievements exactly")
    a = p.parse_args()

    seeds, acts, lens = [], [], []
    for ep in range(a.episodes):
        A, ach, n = rollout(ep)
        seeds.append(ep); acts.append(A); lens.append(n)
        if a.verify:
            *_, (_, st) = ((None, s) for _, s in replay(ep, A))
            ok = np.array_equal(np.array(st.achievements, bool), ach)
            print(f"ep {ep:>3}  len {n:>4}  achievements match: {ok}")
            assert ok, f"NON-DETERMINISTIC at ep {ep} -- Stage 0 gate FAILED"

    if not a.verify:
        out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        flat = np.concatenate(acts)
        offs = np.cumsum([0] + lens)
        np.savez_compressed(out, seeds=np.array(seeds), actions=flat, offsets=offs)
        mb = out.stat().st_size / 1e6
        print(f"{a.episodes} episodes -> {out}  ({mb:.2f} MB, "
              f"mean length {np.mean(lens):.0f})")
    else:
        print(f"\nSTAGE 0 GATE PASSED -- replay is deterministic over {a.episodes} episodes")
