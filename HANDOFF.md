# HANDOFF (family/next)

The full working log of the pre-split project is `HANDOFF.md` at tag
`archive/full-2026-11-08` (commit `49a6bc3`). This file keeps the entries that
concern the next agent (2026-11-04 to 2026-11-08), newest first.

## 2026-11-08 split

**State.** Branch `family/next` was created from `main` (`490a7c9`, plain
DreamerV3) and carries the next family's code, config, env options, tests and
docs (see `CHANGELOG.md`). Runs of this family were trained on `~/vtmgr` =
`49a6bc3` on the GPU box and are not retrained here.

**Not training.** `next_nomask`, the next agent **without** the rule mask
(`--agent.valid.mask False`, `valid_obs` still True; script `~/run_nomask.sh`),
started 2026-11-08 09:26 CST. At 11:35 the box ran out of memory (parallel
CPU check jobs of all five branches) and the kernel killed it at ~74k env
steps (exit 137); its logdir is `~/logdir/next_nomask_oom`. A fresh restart
at 11:43 was stopped by request at 11:46. It has to be relaunched (after
`~/train_guard.sh`; side jobs only through `~/safe_job.sh`). It is the honest
run: it decides whether the next agent is any good without cheating. The masked runs (`next_lam0` 11.20 ± 0.41 on 30 worlds, 11.30
± 0.27 on 60) are counted as cheating and stay out of headline rankings.
`next_notaste` crashed three times on GPU faults (Xid 69, 13; logdirs
`next_notaste`, `next_notaste_crash1`, `_crash2`); `next_lam0` is its stand-in.

**Checks done (12:10, after the box came back).** Equivalence with
`49a6bc3` on CPU (each process through `~/safe_job.sh`, one at a time):
single-threaded, `plain`, `next_lam0` and `next_nomask` are bit-identical
(parameter tree and initial values, a 65-step rollout in two worlds, probe and
greedy acting, parameters and metrics after four train steps, report
metrics); so are the init-only `size50m` trees, and `next_lam0` vs
`next_notaste` on this branch outside taste's own state. Details in
`CHANGELOG.md` and `README.md`. Tests: `pytest dreamerv3/` 83 passed (after
copying the game oracle into `test_craftax_valid.py`); `test_replay.py` has
only the 53 known `Replay.dataset` failures. `~/equiv_next` is deleted.

**Next steps.**
1. Relaunch `next_nomask` (see above). When it finishes (1.1M env steps),
   score it from this branch on
   60 fixed worlds (`tools/death_eval.py --config-from-run --episodes 60`) and
   compare with `next_lam0` (11.30) and `v12_nomask` (6.03, family/v). Add the
   entry to `docs/eval_results.json`, its curve to `docs/curves/` and
   `tools/plot_training_curves.py`, regenerate the report, update the README
   table. A 30-world score is ±0.55: gaps under ~1.1 are not evidence.
2. Decide taste's fate on `family/taste` (fix its goal-lock or drop it). It
   hurt here: `next_lam0` beats `next_taste` by +1.92 ± 0.41
   (`docs/taste-on-manager.md`). Taste stays in this branch only for the
   `lam_mgr 0` flags.
3. Carry forward (not specific to this branch): re-sync `../localplay/`;
   `tools/action_suppression.py` on the v2 checkpoint lives on the branches
   that have v2's runs. The old `tomain` to `main` merge decision is settled by
   the split.

---

## 2026-11-08 run

**Yesterday:** `7be4d51`, clean, `feat/taste-manager`.

**Today:** the 2026-11-07 session committed `de105e0`: no-taste control
`next_lam0` = 11.30 ± 0.27 on 60 test worlds (best run; +1.92 ± 0.41 over
next-taste); `docs/taste-prior.md` explains the failure (taste outweighed the
critic 15-300x, manager locked on `place_table`, 60% no-goal steps near
tables); 29-run comparison report regenerated; CHANGELOG updated.

**Tomorrow:** fix taste's goal-lock or drop it; `tomain` -> `main`;
`action_suppression.py` on v2.

---

## 2026-11-06 run

**Yesterday:** `c7ccad5` on `feat/taste-manager`; `v12_nomask`, second
ablation seeds and the no-taste control were queued on the box.

**Today (2026-11-05 evening work):**
- `874a461` (20:11, self-committed): `v12_nomask` and `abl_manager_s1`
  recorded. The mask is worth +2.90 under the v1.2 manager (6.13 vs 9.03 on 60
  worlds), same as without a manager; the manager's second ablation seed ends
  below plain, so its own share was not established. Curves
  `docs/curves/v12_nomask.json`, `abl_manager_s1.json`.
- `abl_plain_s1` scores 4.77 (same as seed 0), so with two seeds the manager
  is +0.77 ± 0.22 over plain and RSSM-2 alone +0.57 ± 0.26
  (`docs/ablations.md`, family/v). Comparison report regenerated with 28 runs;
  `tools/eval_entry.py` gets `--new`.

**Tomorrow:** fix taste's goal-lock or drop taste-on-manager; read the
requeued no-taste control. Carry forward: `tomain` -> `main` decision,
`action_suppression.py` on v2, `../localplay/` re-sync.

---

## 2026-11-05 run

**Yesterday:** `68acdea` on `feat/taste-manager` (taste on the manager's goal
choice, ablations, `eval_entry.py`); `v12_nomask` and second ablation seeds
queued on the box.

**Today (2026-11-04 afternoon work, self-committed):**
- `e45bf99` (14:16): `next_taste` (taste on manager) trains to 9.2, above
  fixed v1.2 and B2, but the manager's goal entropy was 0 from 90k steps:
  centered evidence from chosen goals only freezes the winner and starves the
  rest. The no-taste control crashed on a GPU fault and was requeued. Added
  `docs/curves/next_taste.json`, a `docs/taste-prior.md` section, CHANGELOG.
- `9b7519d` (15:50): `next_taste` scored on 60 worlds: 9.38, level with fixed
  v1.2, +0.85 over B2, +2.78 over v2; never reaches the stone tier.
- Tree clean; `plastic-reward/` (`a7a4df9`) and `craftax/` (`c3c2e0d`)
  unchanged.

**Tomorrow:** fix taste's goal-lock (evidence for unchosen goals) or drop it;
read the requeued no-taste control, `v12_nomask`, second ablation seeds.

---

## 2026-11-04 run

**Yesterday:** `e66d97e`, branch `feat/taste-prior`.

**Today (11-01..03 work):** branch `feat/taste-manager`. Taste on the manager's
goal choice (`agent.taste.target manager`); `docs/ablations.md` with ablation
curves and a ±0.55 scoring-noise check (only RSSM-2 + the v1.2 manager clearly
beats plain; action-taste does nothing); `tools/eval_entry.py`; regenerated
report and `eval_results.json`.

**Tomorrow:** read `v12_nomask` + second ablation seeds; decide on a full
taste-on-manager run.

---

## Carry-forward items from earlier entries that still apply

- Scoring noise: a 30-world score is ±0.55; one training seed per run.
  Second seeds of the leaders are still missing for the next family.
- RSSM-2 train/act alignment fixes (2026-10-11,
  `docs/rssm2-step-alignment.md`) are in every run of this family; runs from
  before that date used buggy RSSM-2 code that no branch reproduces exactly.
- Evaluation worlds are fixed by seed (`env.reseed`, audit B05): compare runs
  only when the world hash matches (`9067302a2635` for the default 30).
