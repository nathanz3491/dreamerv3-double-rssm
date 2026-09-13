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

No development session ran on 2026-08-17. Working tree stayed clean at `d9030a0`; no files changed anywhere under `mcAI/` since the 2026-08-16 10:25 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (twenty-five consecutive idle days, over three-and-a-half weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.

No development session ran on 2026-08-18, 2026-08-19, or 2026-08-20 (the 2026-08-18 and 2026-08-19 runs of this task appear to have been skipped — no HANDOFF entries were recorded for them, but the file state confirms no activity occurred). Working tree stayed clean at `6f77fc0`; no files changed anywhere under `mcAI/` since the 2026-08-17 05:01 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (twenty-eight consecutive idle days, four weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.

No development session ran on 2026-08-21. Working tree stayed clean at `6434232`; no files changed anywhere under `mcAI/` since the 2026-08-20 07:05 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (twenty-nine consecutive idle days, over four weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.

No development session ran on 2026-08-22. Working tree stayed clean at `1a2a8bd`; no files changed anywhere under `mcAI/` since the 2026-08-21 05:02 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (thirty consecutive idle days, over four weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.

No development session ran on 2026-08-23. Working tree stayed clean at `eb8b81e`; no files changed anywhere under `mcAI/` since the 2026-08-22 05:03 update, and `craftax/` remains pinned at `c3c2e0d`. The idle streak now spans every day from 2026-07-24 through today (thirty-one consecutive idle days, over four weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.

No development session ran in this `dreamerv3` repo on 2026-08-24; working tree stayed clean at `98ffb31` and `craftax/` remains pinned at `c3c2e0d`. However, later on 2026-08-23 (after that day's 05:02 update above), three files were added directly under the parent `mcAI/` workspace — outside both this repo and `craftax/` — for manual Craftax play: `play_craftax.ps1`, `play_craftax_fixed.py`, and `craftax-keybindings.md` (a Windows launcher that fixes a JAX-compile window-focus freeze, so the wood-tier plateau can be explored by hand). That workspace now has its own [`../HANDOFF.md`](../HANDOFF.md) and [`../README.md`](../README.md) tracking it, since it isn't part of this git repo. The idle streak for `dreamerv3` itself now spans every day from 2026-07-24 through today (thirty-two consecutive idle days, over four-and-a-half weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.

No development session ran in this `dreamerv3` repo on 2026-08-25; working tree stayed clean at `418b7d7` (this file's prior 2026-08-24 entry, already committed) and `craftax/` remains pinned at `c3c2e0d`. No files changed anywhere under `mcAI/`, including the parent workspace's own `HANDOFF.md`/`README.md`. The idle streak for `dreamerv3` itself now spans every day from 2026-07-24 through today (thirty-three consecutive idle days, over four-and-a-half weeks). Planned next steps above (`## Tomorrow`) carry forward unchanged.

**Gap notice (2026-08-26 through 2026-09-04, folded in on 2026-09-05):** this file went 11 days without an update. Root cause, confirmed today: the parent `mcAI/` workspace's single scheduled `handoffmd-generation` task (there is no separate scheduled task for this subrepo) kept running daily and updating `../HANDOFF.md`, but no prior run of it reached into this `dreamerv3/` subrepo to update this file — it was never a stalled/broken task, this file was just outside what those runs touched. Confirmed via mtimes that no file anywhere under `mcAI/` (this repo, `craftax/`, or the parent workspace) changed at any point in that 11-day window beyond the parent's own daily `HANDOFF.md` edits. Working tree here stayed clean at `f770262` (2026-08-25) the entire time; `craftax/` remained pinned at `c3c2e0d`; the local `main` branch remains 32 commits ahead of `origin/main`, still unpushed. Planned next steps above (`## Tomorrow`) carry forward unchanged.

## Today (2026-09-05)

No development session ran. Working tree is clean, still matching `f770262` (2026-08-25) — no code changes, only this HANDOFF update. `craftax/` remains pinned at `c3c2e0d`. This run of the parent workspace's scheduled task explicitly checked this subrepo (closing the gap noted above) and will keep doing so going forward. Housekeeping: `README.md` (the upstream DreamerV3 reimplementation README) and `REWCAUSE_INTEGRATION.md` are unchanged and still accurate for the current code state; no `AGENTS.md` needed — no multi-agent roles exist in this codebase.

## Today (2026-09-06)

No development session ran in this `dreamerv3` repo on 2026-09-05 or 2026-09-06; working tree stays clean at `3084fde`. (The parent workspace *did* have a real session on 2026-09-05 evening — it settled the teaching-agents channel architecture in `../DECISIONS.md` and `../proposal-teaching-agents.md`, planning-only, no code here — see `../HANDOFF.md` for details.) `craftax/` remains pinned at `c3c2e0d`. Note for later: the parent proposal's D5 architecture change (`feat2tensor → concat([deter, stoch, goal_embed])`) will touch this codebase and void existing checkpoints once implemented — worth sequencing after, not during, the iron-pickaxe probe below.

## Today (2026-09-07)

No development session ran in this `dreamerv3` repo on 2026-09-06 or 2026-09-07; working tree stays clean at `f42d38e`. The parent workspace *did* have a real session on 2026-09-06 midday — it started the map-model project (P1, the plateau itself) as its own effort: `../design-map-model.md` (a second, slower world model holding a coarse level map), `../plan-map-model-training.md` (six gated stages), and Stage 0 tooling under `../tools/` (`dump_trajectories.py`, `death_diagnostic.py`, `death_math.py`) — all planning/tooling in the parent workspace, no code changes here yet. See `../HANDOFF.md` for the full account. `craftax/` remains pinned at `c3c2e0d`. Note for later: per `../design-map-model.md` §11.5, the map model and this repo's own pending D5 goal-channel change (`feat2tensor → concat([deter, stoch, goal_embed])`) both extend the same `feat2tensor` concatenation and need to be planned together, not built independently — on top of the existing sequencing note that D5 should land after, not during, the iron-pickaxe probe below.

## Today (2026-09-08)

No development session ran in this `dreamerv3` repo on 2026-09-07 or 2026-09-08; working tree stays clean at `61e3897`. The parent workspace also had no session on 2026-09-07 (confirmed idle — see `../HANDOFF.md`) or 2026-09-08. `craftax/` remains pinned at `c3c2e0d`. Nothing new to note beyond continued idleness; all prior sequencing notes (D5/map-model `feat2tensor` overlap, iron-pickaxe probe sequencing) carry forward unchanged.

## Today (2026-09-09)

No development session ran in this `dreamerv3` repo on 2026-09-08 or 2026-09-09; working tree stays clean at `3433682`. The parent workspace also had no session on 2026-09-08 or 2026-09-09 (confirmed idle — see `../HANDOFF.md`). `craftax/` remains pinned at `c3c2e0d`. Nothing new to note beyond continued idleness; all prior sequencing notes (D5/map-model `feat2tensor` overlap, iron-pickaxe probe sequencing) carry forward unchanged.

## Today (2026-09-10 / 2026-09-11)

No development session ran in this `dreamerv3` repo on 2026-09-09, 2026-09-10, or 2026-09-11; working tree stays clean at `dafedf0`. The parent workspace also had no session across that span (confirmed idle — see `../HANDOFF.md`). `craftax/` remains pinned at `c3c2e0d`. Nothing new to note beyond continued idleness; all prior sequencing notes (D5/map-model `feat2tensor` overlap, iron-pickaxe probe sequencing) carry forward unchanged.

## Today (2026-09-12)

No development session ran in this `dreamerv3` repo on 2026-09-11 or 2026-09-12; working tree stays clean at `2051d2b`. The parent workspace also had no session across that span (confirmed idle — see `../HANDOFF.md`). `craftax/` remains pinned at `c3c2e0d`. Nothing new to note beyond continued idleness; all prior sequencing notes (D5/map-model `feat2tensor` overlap, iron-pickaxe probe sequencing) carry forward unchanged.

## Tomorrow

- Carry forward the unchanged plan from above: on the GPU box, confirm the achievement schema, smoke-test `--configs craftax size50m`, then run the actual iron-pickaxe holdout probe with the reworked `rewcause_probe.py` and read the achievement-reach profile / mean-return output (PASS/FAIL criteria as described above).
- Watch the reachability caveat noted above: prior vanilla-PPO only reached the wood tier by ~1.1M steps, so an `inconclusive (0 states)` probe result means "collect more/longer runs," not FAIL.
- Decide whether/when to push the 34 unpushed local commits to `origin/main`.
- Keep this file updated on every future run of the parent workspace's scheduled task, not just when this subrepo itself changes — that's what let the 11-day gap happen.
- Once the map model reaches Stage 3+ in the parent plan, this repo's `agent.py`/`configs.yaml`/`rssm.py` are the files that will actually change — see `../plan-map-model-training.md` §9 for the touch list.
