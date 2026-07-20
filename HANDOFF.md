# HANDOFF

_First run of the automated handoff log for this repo (no prior HANDOFF.md existed) — "Yesterday" below reflects repo state before this session, reconstructed from git history and the research docs in `../` (mcAI root)._

## Yesterday (state before this session)

- Repo was a clean, unmodified clone of upstream `danijar/dreamerv3` (single commit: `e3f0224` "Fix Atari frame maxpooling on reset (#213)"), no local changes.
- No Craftax integration existed in this repo — upstream ships a built-in `crafter` config only.
- Prior project work (see `../dreamer-roadmap-and-findings.md`, `../experiment-plan-dreamerv3-vs-ppo.md`) had established: a working Craftax-Symbolic-v1 PPO baseline (100M steps, mean return 17.16, 21/67 achievements — ~7.6% of max reward, in line with published PPO numbers) and the case for bringing in DreamerV3 as a model-based comparison, since Dreamer doesn't appear on the Craftax leaderboard at all.
- Open question carried in from that research: whether a compositional reward-feature approach could let the model generalize to unseen high-tier achievements (e.g. iron tools) from lower-tier experience alone.

## Today

Implemented and locally verified Phases 0 & 1 of a compositional **reward-cause head** for DreamerV3, targeting the iron-tool generalization question above (full design/rationale in [`REWCAUSE_INTEGRATION.md`](REWCAUSE_INTEGRATION.md)):

- Added `dreamerv3/craftax_features.py` — achievement→φ feature table (verb/material/tier/tool-line), tech-tree DAG, depth-based loss reweighting.
- Added `embodied/envs/craftax.py` — Craftax-Symbolic-v1 env adapter emitting per-step unlocked-achievement observations, plus `save_state()`/`reset_to()` for later frontier-return work; fixed a bug where `is_terminal` read a non-existent `info['timeout']` key (every episode was misread as terminal) — now derives truncation from `state.timestep >= max_timesteps`.
- Edited `dreamerv3/agent.py` — wired the new `rewcause` head (excluded from encoder/decoder, rebalanced + held-out-masked BCE loss, probe-mode output, policy-key wiring).
- Edited `dreamerv3/configs.yaml` / `dreamerv3/main.py` — new `craftax` config preset and env/script registration.
- Added `dreamerv3/rewcause_eval.py` (iron-holdout scorer) and `embodied/run/rewcause_probe.py` (+ `run/__init__.py` update) — the go/no-go probe runner.
- Added `dreamerv3/test_craftax_features.py` — 8 unit tests, all passing; schema also validated offline against the installed Craftax v1.6.1 (exact match).
- Craftax repo itself (`../craftax`) untouched — working tree clean, no changes needed there.

Net: repo now runs end-to-end for the `craftax` suite (adapter, config, head, probe script all present); nothing has been executed on GPU yet.

## Tomorrow

- On the GPU box, first confirm the achievement schema still matches the installed Craftax build:
  `cd dreamerv3/dreamerv3 && python -c "from craftax_features import validate_against_craftax as v; v(); print('schema OK')"`
- Run a smoke test of `--configs craftax size50m`.
- Run the actual go/no-go experiment: train with `MAKE_IRON_PICKAXE` held out of the reward-cause loss, then run `rewcause_probe` against the checkpoint.
  - **PASS** → head recovers verb/tool/material/tier for iron states it never trained on → proceed to Phases 2-4.
  - **FAIL** → φ feature space doesn't extrapolate linearly → revisit `craftax_features.py` before building further.
  - Watch for the flagged **reachability caveat**: the earlier vanilla-PPO run only reached the wood tier by ~1.1M steps, so a fresh agent may rarely/never unlock iron — an `inconclusive (0 states)` probe result should be read as "collect more/longer runs," not as FAIL.
