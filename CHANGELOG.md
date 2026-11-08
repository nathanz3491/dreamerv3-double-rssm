# Changelog

All notable changes to `family/masked` (the rule-validity experiments B1-B3)
are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The branch is `main` (plain DreamerV3 on Craftax) plus this family's code.
Changes to `main` itself are in `main`'s changelog; the history before the
split, all families together, is in `CHANGELOG.md` at tag
`archive/full-2026-11-08`.

## [Unreleased]

### Changed
- **The whole family is counted as cheating, and no new work goes here.**
  Every run reads the game's action rules (`craftax_valid`):
  - B2 masks impossible actions by them;
  - B1 gets them as encoder input;
  - B3 learns its mask, but still needs `valid_obs` to build it, hand-picks
    the never-masked basic actions and takes action 0 as NOOP.

  The mask was worth +2.6 to +2.9 achievements wherever it was measured, more
  than anything else in the project, and the published baselines have no such
  knowledge. The README says so first; results list the honest runs first
  and mark every family row.
- `docs/eval_results.json`, `docs/curves/`, the comparison report and the
  training curves hold the plain baselines and this family's runs only;
  `tools/plot_training_curves.py` lists them.
- `docs/entropy-and-action-suppression.md` gains a branch note and loses the
  section on directions that belong to no family.

### Added
- **Split: `family/masked` branches off `main`** and adds back from `49a6bc3`
  what the B1-B3 runs need. Everything is default-off. Config keys, module
  and parameter names, RNG order and loss order match `49a6bc3`.
  - Experiment B:
    - `craftax_valid.py`;
    - the env's `valid_obs`;
    - `agent.valid` and `agent.validhead`: B1 `input`, B2 `mask`, and B3
      `learned` with `evidence`, `explore`, `warmup`, `exempt_basic`,
      `reference`, `detach`, `label`, `margin` and `rule_mix`;
    - the `feas` head and the `feasclock` counter;
    - `_masked`, `_dream_valid` and `_feas_learned`;
    - the `valid/*` and `feas/*` metrics.
  - RSSM-2 with the map target, as used by every run here:
    - `mapmodel.py` and `craftax_map.py`;
    - the env's `mapmodel`, `map_privileged` and `seen_decay`;
    - `agent.mapmodel` (`enabled`, `to_actor`, `imag_shift`, `hindsight`,
      sizes);
    - the actor's map crop and the train/act step `_map_step`, with the
      2026-10-11 fixes;
    - `replay.save_skip` also skips `map/`.
  - Potential shaping: `craftax_potential.py` and the env's
    `survival potential` with `phi_scale` and `phi_gamma`. The start-state
    potential is set at reset (audit B07).
  - Tests:
    - `test_craftax_valid.py`, `test_craftax_map.py`,
      `test_craftax_potential.py` and `test_mapmodel_step.py`;
    - in `test_audit_fixes.py`, the first-shaped-step test.
  - Tools:
    - `action_suppression.py` (experiment A);
    - `diagnose_run.py`;
    - `episode_cases.py` and `make_casebook.py`;
    - the "Where B2 falls short" page of `make_comparison_report.py`.
  - Docs:
    - `entropy-and-action-suppression.md`;
    - `experiment_a/`, `cases/`, `b2-casebook.pdf` and `diagnosis.json`;
    - `rssm2-step-alignment.md`.
- Checked on the box against `49a6bc3`:
  - every run's flags parse on both trees (expB_input, expB_mask,
    expB_mask_fix and s1/s2, expB3, expB3fix, calib_obs, calib_latent, plain);
  - the env builds and steps with each of them.
- CPU equivalence of the built agents against `49a6bc3` (2026-11-08, size50m,
  batch 2 x 16, seed 0; parameters at init, policy outputs in train, greedy
  and probe modes, metrics and parameters after 3 and 4 train steps, report
  metrics). Each flag set ran `49a6bc3` twice (refA, refB) and this branch
  once or twice, because two CPU runs of the same code differ in a few
  parameter arrays by up to about 1e-3 after training:
  - `expB_input`: bit-identical to `49a6bc3` in everything, parameters
    included;
  - `expB_mask_fix`, `expB3fix`, `b3fix_stress`: policy outputs, train
    outputs and metrics bit-identical; parameters after training differ by
    at most 4.9e-4, 5.5e-5 and 2.2e-6 (ref vs ref: 4.9e-4, 0, 1.5e-5; the
    `expB3fix` refs happened to agree, and 5.5e-5 is the size of the noise
    between `49a6bc3` runs in the other flag sets);
  - `expB3`: the same, with parameters within 9.8e-4 (ref vs ref: 3.3e-5;
    one branch run differs from the other by 9.8e-4 as well);
  - `plain`: the same. One first-pass run differed in the third train step's
    metrics (loss, advantage) and in parameters by up to 0.2; that did not
    recur in two reruns (metrics bit-identical, parameters 1.3e-4 from
    refA, against 2.4e-4 between the two refs), so it was a bad recording.
- `test_craftax_valid.py::test_agrees_with_the_game` passes with `tools/`
  present. The rest of `dreamerv3/` and `test_replay.py`: 113 passed and 53
  failed, all of them in `test_replay.py` (`'Replay' object has no attribute
  'dataset'`), which fail the same way at `49a6bc3`.

### Removed
- Left on other branches, because no run here used them:
  - RSSM-2's memory target (`mapmodel.target`, `ahead`, `back`, `memhead`,
    `memfut`/`memrec`): on `family/rssm2`;
  - the `shaped` and `potential+meters` survival modes, with their `surv_*`
    keys and `survival_potential`;
  - the manager, goals, curiosity and taste.

### History of the family before the split (summary)
- 2026-10-01: experiment A (`tools/action_suppression.py`). Every run kept
  exactly one wood craft alive, and frontier keys sat at 0% where they would
  work: valid-action suppression.
- 2026-10-01: experiment B built.
  - `craftax_valid.valid_actions` agrees with the game in 77,700 of 77,700
    checks.
  - B1 (`valid.input`) and B2 (`valid.mask`) run on the honest-map
    configuration.
- 2026-10-03: B1 and B2 results. Original evaluation: honest map 5.87, B1
  6.90, B2 7.60. Frontier keys come off zero only under the mask.
- 2026-10-05: B3 learns the mask from the world model's counterfactual
  (`valid.learned`, with `evidence`, `explore` and `warmup`).
- 2026-10-07: B3 scored 5.60. Its latent labels fed back into the world model
  (87% wrong).
  - The fix: `valid.detach` and `valid.label obs` with `valid.margin`,
    calibrated on B2's world model (AUC 0.83-0.88), plus `valid.rule_mix` for
    calibration only.
  - B3-fix was stopped at ~254k steps: its mask allowed only 45% of the
    possible special actions. The later agents kept the rule mask.
- 2026-10-11: the outside audit found three RSSM-2 bugs
  (`docs/rssm2-step-alignment.md`). Every B run predates the fixes.
  `expB_mask_fix` and seeds 1-2 were retrained on the fixed code
  (8.27 / 7.70 / 9.03).
- 2026-10-15: every run re-scored on the fixed worlds:
  - B1 6.30, B2 8.40, B3 5.97, honest map 6.17;
  - B2 over the honest map: +2.63 ± 0.38 on the fixed code (three seeds).
