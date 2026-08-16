# HANDOFF

## Yesterday (state before this session)

Phases 0 & 1 of the compositional **reward-cause head** for DreamerV3 (design/rationale in [`REWCAUSE_INTEGRATION.md`](REWCAUSE_INTEGRATION.md)) are in place (`c9a07f1`), the JAX transfer-guard violation in `embodied/envs/craftax.py` is fixed (`70a89b8`), and `embodied/run/rewcause_probe.py` has been reworked and committed (`7f6d285`): checkpoint-dir resolution now handles a `latest` pointer, `collect()` reports a full achievement-reach profile plus mean episode return (not just the held-out target), progress logs live per driver chunk, and the probe budget was lowered (`PROBE_MAX_EPISODES` 300→150, `PROBE_MIN_STATES` 50→30) for a faster read given the iron-tier reachability caveat.

No development session ran on 2026-07-24 through 2026-07-29 (six consecutive idle days); working tree stayed clean at `417222d`, matching the code state described above. Planned next steps carried forward unchanged: on the GPU box, confirm the achievement schema, smoke-test `--configs craftax size50m`, then run the actual iron-pickaxe holdout probe.

## Today

No development session ran today (2026-07-30). Working tree is clean and matches the `417222d` commit — no new commits, no uncommitted changes, and no file modifications anywhere in `mcAI/` (including `craftax/`, still pinned at `c3c2e0d`) since 2026-07-29 05:02.

## Tomorrow

- On the GPU box, confirm the achievement schema still matches the installed Craftax build:
  `cd dreamerv3/dreamerv3 && python -c "from craftax_features import validate_against_craftax as v; v(); print('schema OK')"`
- Run the smoke test of `--configs craftax size50m` (transfer-guard fix should unblock it).
- Run the actual go/no-go experiment: train with `MAKE_IRON_PICKAXE` held out of the reward-cause loss, then run the now-updated `rewcause_probe` against the checkpoint and check the new achievement-reach profile/mean-return output.
  - **PASS** → head recovers verb/tool/material/tier for iron states it never trained on → proceed to Phases 2-4.
  - **FAIL** → φ feature space doesn't extrapolate linearly → revisit `craftax_features.py` before building further.
  - Watch the reachability caveat: the earlier vanilla-PPO run only reached the wood tier by ~1.1M steps, so a fresh agent may rarely/never unlock iron. With `PROBE_MIN_STATES` now lowered to 30, an `inconclusive (0 states)` result should still be read as "collect more/longer runs," not FAIL — use the new reach profile to judge whether the agent is even close.

No development session ran on 2026-07-31 (seventh consecutive idle day). Working tree stayed clean at `a54c722`; no files changed anywhere under `mcAI/` (including `craftax/`, still pinned at `c3c2e0d`) since the 2026-07-30 05:02 update. Planned next steps above carry forward unchanged.

No development session ran on 2026-08-01. Working tree stayed clean at `6e3bd4a`; no files changed anywhere under `mcAI/` since the 2026-07-31 05:01 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today. Planned next steps above carry forward unchanged.

No development session ran on 2026-08-02. Working tree stayed clean at `f176f20`; no files changed anywhere under `mcAI/` since the 2026-08-01 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (ten consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-03. Working tree stayed clean at `eba58b6`; no files changed anywhere under `mcAI/` since the 2026-08-02 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (eleven consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-04. Working tree stayed clean at `91c2b81`; no files changed anywhere under `mcAI/` since the 2026-08-03 05:01 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (twelve consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-05. Working tree stayed clean at `ebb59a1`; no files changed anywhere under `mcAI/` since the 2026-08-04 05:01 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (thirteen consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-06. Working tree stayed clean at `c98c9a9`; no files changed anywhere under `mcAI/` since the 2026-08-05 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (fourteen consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-07. Working tree stayed clean at `c41d262`; no files changed anywhere under `mcAI/` since the 2026-08-06 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (fifteen consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-08. Working tree stayed clean at `c2a8363`; no files changed anywhere under `mcAI/` since the 2026-08-07 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (sixteen consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-09. Working tree stayed clean at `4bb7961`; no files changed anywhere under `mcAI/` since the 2026-08-08 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (seventeen consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-10. Working tree stayed clean at `d03e200`; no files changed anywhere under `mcAI/` since the 2026-08-09 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (eighteen consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-11. Working tree stayed clean at `7ce2c42`; no files changed anywhere under `mcAI/` since the 2026-08-10 05:01 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (nineteen consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-12 or 2026-08-13 (the 2026-08-12 run of this task appears to have been skipped — no HANDOFF entry was recorded for it, but the file state confirms no activity occurred). Working tree stayed clean at `cd8ccdd`; no files changed anywhere under `mcAI/` since the 2026-08-11 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (twenty-one consecutive idle days). Planned next steps above carry forward unchanged.

No development session ran on 2026-08-14. Working tree stayed clean at `f51d71b`; no files changed anywhere under `mcAI/` since the 2026-08-13 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (twenty-two consecutive idle days, three-plus weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.

No development session ran on 2026-08-15. Working tree stayed clean at `0e4e744`; no files changed anywhere under `mcAI/` since the 2026-08-14 05:03 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (twenty-three consecutive idle days, over three weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.

No development session ran on 2026-08-16. Working tree stayed clean at `45e8868`; no files changed anywhere under `mcAI/` since the 2026-08-15 05:01 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (twenty-four consecutive idle days, over three weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.
