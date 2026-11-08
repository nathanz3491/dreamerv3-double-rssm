# Changelog

All notable changes to `main` (plain DreamerV3 on Craftax) are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The history of the experiment families up to the split (map model, goal
manager, named and learned goals, taste, rule mask, reward-cause head) is in
`CHANGELOG.md` at tag `archive/full-2026-11-08`. From now on each family keeps
its own changelog on its own branch.

## [Unreleased]

### Changed
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

### Removed
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

### Added
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

### Fixed
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
