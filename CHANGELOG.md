# Changelog

All notable changes to `family/taste` (taste on DreamerV3 / Craftax) are
recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This branch is `main` plus the taste family. The entries marked *(main)* are
`main`'s, shared by every family branch. The family's history before the
split is in `CHANGELOG.md` at tag `archive/full-2026-11-08`; it is summarised
below.

## [Unreleased]

### Added
- **The taste family on its own branch** (`family/taste`): `main` plus
  everything the taste runs (`abl_taste`, `v2_taste`, `next_taste`) need. Why
  a branch: the families had grown `agent.py` to 2126 lines behind
  default-off flags; each family now carries only its own code.
  - Taste (`dreamerv3/taste.py`, `agent.taste.*`): over actions (a logit
    bias in `Agent._masked`, evidence from the replay advantage) and on the
    manager (an internal reward on the manager's goal choice in
    `_mgr_loss_shared`, evidence from `_taste_mgr_evidence`). `lam_mgr 0` is
    the no-taste control.
  - The agents underneath: RSSM-2 as a memory (`mapmodel.target memory`,
    `mapmodel.py`), the manager with the shared two-headed critic and held
    goals over named (`craftax_goals.py`, env `goals_obs`) or learned goals
    (`goalcodes.py`), curiosity, and the rule mask (`valid.mask`, env
    `valid_obs`; default off, counted as cheating). `craftax_map.py`,
    `craftax_potential.py` and `craftax_valid.py` come along as libraries of
    the goals and the mask.
  - Their tests: `test_taste.py`, `test_mapmodel_step.py`,
    `test_craftax_goals.py`, `test_craftax_valid.py`, `test_craftax_map.py`,
    `test_craftax_potential.py`, and the codebook and goal-start tests in
    `test_audit_fixes.py`. `test_craftax_valid.py` carries its own copy of
    the game oracle (`make_oracle`), which lived in a `family/masked` tool.
  - Checked against the pre-split code (`49a6bc3`): the flag sets of
    `abl_taste`, `v2_taste`, `next_taste`, `next_lam0` and the plain run all
    parse, and every config key both trees have gets the same value (the
    keys removed here are listed under Changed).
  - Equivalence with `49a6bc3` on CPU (single-threaded XLA; batch 2 × 16,
    `size50m`, seed 0), for the flag sets of `abl_taste`, `v2_taste`,
    `next_taste` and the plain run, and the three taste runs again with
    `taste.every 1` so taste folds on every update: the parameter tree
    (names, shapes, dtypes) and the initial values, the policy outputs
    (train, greedy and probe modes), the parameters after 3 train steps,
    the train metrics of 4 steps and the report metrics are all
    bit-identical (max abs diff 0). The parameters after the 4th step are
    bit-identical too for every taste set; for plain they differ by up to
    4.9e-4 in three tensors, and 49a6bc3 against itself differs by the same
    4.9e-4, so that is run-to-run noise in the 4th step, not the code.
  - Tests (`pytest dreamerv3/ embodied/tests/test_replay.py`, on the box):
    99 passed; the 53 failures are the known `Replay.dataset` ones in
    `test_replay.py`, the same as on `main` and at 49a6bc3.
  - Docs: `docs/taste-prior.md` (design, every result, why taste hurt, with
    the `abl_taste` result added and masked runs marked), `docs/taste-run.json`,
    `docs/agent.md` (the agents underneath, trimmed from `family/v`'s design
    notes), `docs/rssm2-step-alignment.md`; `docs/eval_results.json`, the
    curves, the training-curve figure and the comparison report hold the
    plain baseline, the three taste runs and their no-taste controls
    (`next_lam0`, `v2_norules`), masked runs marked.
- Family history up to the split (`archive/full-2026-11-08`):
  - Taste over actions, ported from the plastic-reward toy as a
    knowledge-free stand-in for v2's rule mask. `v2_taste` (trained on
    `cbec67b`) scored 5.33 on the 30 fixed worlds: −1.69 against fixed v2
    (masked), +0.70 against vanilla, level in training with its no-taste
    control. Uncentered, its bias followed how often each button was
    pressed (rank correlation −0.98).
  - `taste.center` (each slot against the fold's average) and sparse counts
    (`observe(..., steps=)`); the defaults reproduce v2-taste.
  - `abl_taste` (plain DreamerV3 + centered taste over actions): 4.63
    against plain's 4.77, no effect.
  - Taste on the manager (`target manager`), with the code-review fixes
    before its first run: fast critic for the evidence, taste relative to
    the goals' mean, its own `lam_mgr`, only decision rows scored
    (`taste.rows`), `q_spread` logged before taste, the step counter only
    for the manager target (v2-taste checkpoints still load).
  - `next_taste` (masked): 9.80 on 30 worlds, 9.38 on 60; its no-taste
    control `next_lam0` (masked, `family/next`) 11.30 on 60, +1.92 ± 0.41.
    Taste outweighed the critic 15-300 times from its first update and
    locked the manager onto `place_table` (docs/taste-prior.md, "Why taste
    hurt").

### Changed
- The manager has one kind of critic here, `shared` with held goals: the
  v1/v1.1 manager with its own critic (`_imagine_mgr`, `_mgr_loss`,
  `mval`, `manager.every`, `manager.mgr_bonus`) stays on `family/v`. The
  config defaults are now `manager.critic shared`, `manager.hold 32` and
  `mapmodel.target memory`; every run that uses the manager or RSSM-2
  passes these explicitly, so none changes.
- RSSM-2 has the memory target only. The map target (`mapmodel.target map`,
  the map decoder, crop, `imag_shift`, `hindsight`, `coarse`, `planes`,
  `crop`, the env's map targets and `map_privileged`) stays on
  `family/rssm2`; `mapmodel.py` keeps the GRU, its state and the gate, with
  the same param names.
- The rule mask is B2 only: `valid.mask` and `valid.threshold`. B1
  (`valid.input`) and B3 (the learned mask and its options) stay on
  `family/masked`. The basic actions are always exempt.
- The env has `valid_obs` and `goals_obs`; reward shaping (`survival`
  other than `none`, the `surv_*` and `phi_*` options) and the v1.3
  survival goals (`goals_survival`) stay on `family/rssm2` and `family/v`.
- `curiosity.normalize` and `retlimit` (v2-cur) stay on `family/v`.
- `replay.save_skip` includes `map/` again (RSSM-2's replay-context
  entries).
- `tools/plot_training_curves.py` lists this family's runs and controls.
- `docs/audit-2026-10-11.md` says which items apply to this branch.

### Changed (main)
- **Split: `main` is now plain DreamerV3 on Craftax.** Every experiment family
  moved out to a branch of its own (indexed in `README.md`); `main` keeps
  upstream DreamerV3, the general fixes, the plain parts of the Craftax adapter
  and the scoring tools. Why: the families had piled into one `agent.py`
  (2126 lines) behind default-off flags, so a fix to the shared parts could not
  be told apart from a change to one family, and the plain baseline was hard
  to read. The full pre-split code, with every run reproducible, is the tag
  `archive/full-2026-11-08`.
  - The plain agent is unchanged: built from `main` and from the pre-split
    code with `--configs craftax size50m --env.craftax.survival none`, the
    parameter tree (names, shapes, dtypes), the initial values, the policy's
    actions and outputs, every parameter after four train steps, and the train
    and report metrics are bit-identical on CPU. Old plain checkpoints
    (`abl_plain`) therefore load on `main`.
  - Kept for that reason: the craftax preset's `replay.size 4e5`, `obs['ach']`
    with its exclusion from the encoder and decoder, and the env's `survival`
    option (only `none` is accepted).
- `docs/eval_results.json`, `docs/curves/`, the training curves and the
  comparison report hold the plain runs only (vanilla and `abl_plain`);
  `tools/plot_training_curves.py` lists them.
- `tools/make_comparison_report.py` drops the B2 diagnosis page and makes the
  checkpoint-variance panel optional.
- `test_craftax_features.py` imports through the package, so
  `python -m pytest dreamerv3/` collects it.

### Removed (main)
- Family code: RSSM-2 / map model with its targets, potential and survival
  shaping and privileged maps (`mapmodel.py`, `craftax_map.py`,
  `craftax_potential.py`); the goal manager v1-v1.3 and v2 with learned goals,
  curiosity and the shared critic (`goalcodes.py`); named goals
  (`craftax_goals.py`); taste (`taste.py`); the rule mask, `valid_obs` and
  B1-B3 (`craftax_valid.py`); the reward-cause head and its probe
  (`rewcause_eval.py`, `embodied/run/rewcause_probe.py`, the phi tables in
  `craftax_features.py`). Their config keys, env options, tests, tools and
  docs went with them.
- `HANDOFF.md`, `REWCAUSE_INTEGRATION.md`, `reports/` and the family docs
  (design notes, ablations, casebooks, experiment A/B data).

### Added (main)
- Craftax (symbolic) adapter `embodied/envs/craftax.py` and the
  `--configs craftax` preset (single env, `train_ratio 512`, 1.1M steps).
  - `obs['ach']`: achievements unlocked this step, read from replay by the
    scoring tools.
  - A timeout is a truncation, not a terminal state.
  - `env.reseed(seed)` builds the next world from the seed alone.
  - `raw_reward` holds Craftax's own reward.
  - `save_state()` / `reset_to()` restore a functional EnvState.
- `policy(mode='greedy')` takes the argmax action; `policy(mode='probe')`
  also returns the action distribution (`policy_prob`).
- Scoring and inspection tools: `death_eval.py` (fixed worlds, world hash,
  endings, cause of death, raw return, `--config-from-run`), `eval_entry.py`,
  `curves_from_replay.py`, `plot_training_curves.py`, `compare_at_steps.py`,
  `make_comparison_report.py`, `watch_agent.py`, `action_audit.py`,
  `craft_probe.py`, `death_diagnostic.py`, `death_math.py`,
  `strip_replay.py`.
- `replay.save_skip` keeps the replay_context latents out of saved chunks
  (~1.4 GB a run instead of ~9.3 GB); a resumed run gets zeros back that the
  agent overwrites as it trains (`test_restore_save_skip`).

### Fixed (main)
- Audit 2026-10-11 items that apply to the plain agent
  (`docs/audit-2026-10-11.md`, tests in `dreamerv3/test_audit_fixes.py`):
  - B05: evaluation worlds depended on how long earlier episodes ran;
    `death_eval.py` now reseeds each episode and prints a world hash.
  - B09: `--greedy` sampled; it now takes the argmax.
  - B10: evaluator cutoffs were counted as deaths; endings are split into
    death, timeout and cutoff.
  - 6.1: `report()` dropped its loss metrics (`mets.update(mets)`).
  - 6.4: raw and shaped return are reported separately.
- The adapter declared `obs['vector']` in [0, 1], which crashed the first
  agent to level up; the upper bound is now unbounded.
- Achievements past index 24 were reported under the wrong names; they are
  named by enum value everywhere.
- `run.from_checkpoint_regex` was read by the run scripts but missing from
  `configs.yaml`, so `--run.from_checkpoint` crashed.
