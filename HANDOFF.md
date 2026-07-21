# HANDOFF

## Yesterday (state before this session)

Phases 0 & 1 of the compositional **reward-cause head** for DreamerV3 (design/rationale in [`REWCAUSE_INTEGRATION.md`](REWCAUSE_INTEGRATION.md)) had just landed in commit `c9a07f1`:

- `dreamerv3/craftax_features.py` — achievement→φ feature table, tech-tree DAG, depth-based loss reweighting.
- `embodied/envs/craftax.py` — Craftax-Symbolic-v1 env adapter with per-step unlocked-achievement observations, `save_state()`/`reset_to()` for later frontier-return work, and the `is_terminal`/timeout-vs-death fix.
- `dreamerv3/agent.py`, `dreamerv3/configs.yaml`, `dreamerv3/main.py` — `rewcause` head wired in, plus the new `craftax` config preset.
- `dreamerv3/rewcause_eval.py` + `embodied/run/rewcause_probe.py` — the iron-holdout go/no-go probe runner.
- `dreamerv3/test_craftax_features.py` — 8 unit tests passing; schema validated offline against Craftax v1.6.1.

Repo ran end-to-end for the `craftax` suite but nothing had been executed on GPU yet. Planned next steps were: confirm the achievement schema on the GPU box, smoke-test `--configs craftax size50m`, then run the actual iron-pickaxe holdout probe.

## Today

- Fixed a JAX transfer-guard violation in `embodied/envs/craftax.py`: `embodied.jax` installs a global `jax_transfer_guard='disallow'` to catch stray host↔device transfers inside the agent, but the Craftax adapter legitimately crosses that boundary every step (building RNG keys, running the functional JAX env, pulling observations back to numpy). Every such region (`__init__` param/space setup, `step`, `_reset`, `_newly_unlocked`, `_obs` logging, and the Phase-4 `reset_to` checkpoint restore) is now wrapped in a scoped `jax.transfer_guard('allow')` block, mirroring the reference crafter/gymnax adapters, so the guard stays active everywhere else in the agent.
- No functional/behavioral change to the env logic itself — this only unblocks running under the agent's transfer guard.
- Change was still uncommitted at session start; committed now along with this HANDOFF update.

## Tomorrow

- On the GPU box, confirm the achievement schema still matches the installed Craftax build:
  `cd dreamerv3/dreamerv3 && python -c "from craftax_features import validate_against_craftax as v; v(); print('schema OK')"`
- Now that the transfer-guard fix is in, run the smoke test of `--configs craftax size50m` — this is likely what surfaced the guard violation in the first place, so re-run it first before anything else.
- Run the actual go/no-go experiment: train with `MAKE_IRON_PICKAXE` held out of the reward-cause loss, then run `rewcause_probe` against the checkpoint.
  - **PASS** → head recovers verb/tool/material/tier for iron states it never trained on → proceed to Phases 2-4.
  - **FAIL** → φ feature space doesn't extrapolate linearly → revisit `craftax_features.py` before building further.
  - Watch for the flagged **reachability caveat**: the earlier vanilla-PPO run only reached the wood tier by ~1.1M steps, so a fresh agent may rarely/never unlock iron — an `inconclusive (0 states)` probe result should be read as "collect more/longer runs," not as FAIL.
