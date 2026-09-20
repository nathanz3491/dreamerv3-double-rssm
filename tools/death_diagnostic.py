"""Why does the agent die? Logs cause-of-death over N Craftax episodes.

Random policy by default (no checkpoint needed). Pass --policy to plug in a
trained one. Records step of death, terminal stats, and which necessity failed
first -- the three things that distinguish starvation, thirst, exhaustion and
mob damage.
"""
import argparse, collections
import jax, jax.numpy as jnp, numpy as np
from craftax.craftax_env import make_craftax_env_from_name


def run(n_episodes, seed, max_steps):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params
    step_fn, reset_fn = jax.jit(env.step), jax.jit(env.reset)
    rng = jax.random.PRNGKey(seed)
    n_actions = env.action_space(params).n

    rows, causes = [], collections.Counter()
    for ep in range(n_episodes):
        rng, k = jax.random.split(rng)
        _, state = reset_fn(k, params)
        first_zero, t = {}, 0
        while t < max_steps:
            rng, ka, ks = jax.random.split(rng, 3)
            act = int(jax.random.randint(ka, (), 0, n_actions))
            _, state, _, done, _ = step_fn(ks, state, act, params)
            t += 1
            for name, val in (("food", state.player_food),
                              ("drink", state.player_drink),
                              ("energy", state.player_energy)):
                if float(val) <= 0 and name not in first_zero:
                    first_zero[name] = t
            if bool(done):
                break

        hp = float(state.player_health)
        if first_zero:
            cause = min(first_zero, key=first_zero.get)
            causes[f"{cause} ran out first"] += 1
        else:
            causes["no necessity hit zero -> mob damage"] += 1
        rows.append((t, hp, float(state.player_food), float(state.player_drink),
                     float(state.player_energy), first_zero.get("drink"),
                     first_zero.get("food"), first_zero.get("energy")))
    return rows, causes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1200)
    a = p.parse_args()

    rows, causes = run(a.episodes, a.seed, a.max_steps)
    L = np.array([r[0] for r in rows])
    print(f"\n=== {a.episodes} episodes, RANDOM policy ===")
    print(f"lifespan   mean {L.mean():6.1f}   median {np.median(L):6.1f}"
          f"   min {L.min()}   max {L.max()}")
    print(f"terminal   health {np.mean([r[1] for r in rows]):.2f}"
          f"   food {np.mean([r[2] for r in rows]):.2f}"
          f"   drink {np.mean([r[3] for r in rows]):.2f}"
          f"   energy {np.mean([r[4] for r in rows]):.2f}")
    for k in ("drink", "food", "energy"):
        i = {"drink": 5, "food": 6, "energy": 7}[k]
        v = [r[i] for r in rows if r[i] is not None]
        if v:
            print(f"{k:<8} hit zero in {len(v):>3}/{len(rows)} eps, "
                  f"first at mean step {np.mean(v):.0f}")
    print("\ncause of death:")
    for c, n in causes.most_common():
        print(f"  {n:>3}  {c}")
    print(f"\nreference: closed-form neglect ceiling = 333 steps")
