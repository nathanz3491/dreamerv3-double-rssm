# Changelog

All notable changes to `family/rssm2` (the map-model family: DreamerV3 with a
second, slower RSSM on Craftax) are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The branch is `main` plus this family. Entries for `main`'s own parts are
kept as on `main`. The full history before the split, every family together,
is in `CHANGELOG.md` at tag `archive/full-2026-11-08`; the map-model entries
from it are summarised under "Added" and "Fixed" below.

## [Unreleased]

### Changed
- **`family/rssm2`: the map-model family as a branch of its own.** It is
  `main` plus RSSM-2 (map and memory targets, honest and privileged map
  labels, hindsight, `imag_shift`), the potential and survival reward shaping
  of the map runs, their tests, tools and docs, taken from `49a6bc3` (tag
  `archive/full-2026-11-08`) with config keys, module and parameter names, RNG
  order and loss order unchanged, so old checkpoints load and behaviour
  matches. Nothing from the other families is here: no rule mask or validity
  flags, goal manager, named or learned goals, curiosity, taste or
  reward-cause head. Why: each family is reproducible and readable on its own;
  see `main`'s README for the index.
  - CPU equivalence check against `49a6bc3` (size50m, batch 2x16; records
    the parameter tree, initial values, policy outputs and carry in train,
    greedy and probe modes, train outputs and metrics, parameters after 3
    and 4 train steps, report metrics) for the flag sets plain, `abl_rssm2`,
    `honest_map_fix` and `map_stage5`. Each set was run twice on `49a6bc3`
    (ref A, ref B) and once here, and the new-vs-ref comparison was judged
    against ref-vs-ref, because CPU runs of the same code are not bit-exact
    in a few parameters (changes of 1e-5 to 6e-5 in `dyn/` kernels and
    optimizer slots, and in one initial `rew` kernel). In all four sets the
    parameter tree, dtypes, policy outputs and carry, train outputs, train
    metrics and report metrics are bit-identical (max abs diff 0), and the
    initial parameters are identical except as noted:
    - `map_stage5`: parameters after 3 and 4 train steps bit-identical. The
      one differing array is the initial `rew/mlp/linear0/kernel` (6.1e-5),
      which also differs between the two runs of `49a6bc3`. Match.
    - `abl_rssm2`: a single kernel, `dyn/dynhid0/kernel`, differs by 1.9e-6
      in the initial values and carries that difference unchanged into the
      later parameters; nothing else differs. The two `49a6bc3` runs had
      identical initial values but differ by up to 5.9e-5 after training in
      six arrays, so this is inside that noise. Match, with a stray
      initial-value difference.
    - plain: parameters after 3 and 4 train steps differ by at most 7.6e-6
      in 6 arrays (`dyn/dyngru/kernel`, optimizer slots of `dyn/dynin0`),
      against up to 5.5e-5 in 7 arrays between the two `49a6bc3` runs.
      Within the ref-vs-ref noise, not bit-identical.
    - `honest_map_fix`: after 4 train steps the largest difference is 1.3e-5
      in 7 arrays, the same as between the two `49a6bc3` runs (1.3e-5, 7
      arrays). After 3 steps one parameter, `slowval/mlp/linear2/kernel`,
      differs by 9.8e-4 against 6.2e-6 between the two `49a6bc3` runs, and
      is identical again after step 4, which the 0.02 slow-value update
      should not produce. All train and report metrics are identical, so this
      is not understood. Not resolved by the records kept (the per-element
      data was deleted with the run); a rerun of this one set is needed to
      call it noise or a difference.

    Tests on the box (CPU, 8 GB cap): `pytest dreamerv3/
    embodied/tests/test_replay.py` gives 111 passed, 53 failed; all 53 are
    `embodied/tests/test_replay.py` (`'Replay' object has no attribute
    'dataset'`, the same at `49a6bc3`); `test_mapmodel_step.py` and
    `test_audit_fixes.py` pass.
  - Every run flag set of the family (`map_stage5`, `map_survive`,
    `map_potential`, `map_noshift`, `map_pot_fixed`, `priv_map`,
    `honest_map`, `honest_map_truepos`, `honest_map_fix`, `abl_rssm2`, and
    `abl_plain`) parses on this branch, and builds the env and the agent.
- `docs/eval_results.json`, `docs/curves/`, the training curves and the
  comparison report hold this family's runs and the plain baselines;
  `tools/plot_training_curves.py` lists them. `docs/ablations.md` keeps the
  plain-vs-RSSM-2 part and the scoring-noise result;
  `docs/audit-2026-10-11.md` and `docs/rssm2-step-alignment.md` are annotated
  for this branch.
- `tools/death_eval.py` examples use a map run.

- **Split: `main` is now plain DreamerV3 on Craftax.** Every experiment family
  moved out to a branch of its own (indexed in `main`'s `README.md`); `main` keeps
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
- `tools/make_comparison_report.py` drops the B2 diagnosis page and makes the
  checkpoint-variance panel optional.
- `test_craftax_features.py` imports through the package, so
  `python -m pytest dreamerv3/` collects it.

### Removed
- Not on this branch (they live on their own family branches, or only at tag
  `archive/full-2026-11-08`): the goal manager v1-v1.3 and v2 with learned
  goals, curiosity and the shared critic (`goalcodes.py`); named goals
  (`craftax_goals.py`, `env.craftax.goals_obs`, `goals_survival`); taste
  (`taste.py`); the rule mask, `valid_obs` and B1-B3 (`craftax_valid.py`);
  the reward-cause head and its probe; the survival potential over food and
  drink (`survival potential+meters`, `surv_w0`, `surv_kappa`,
  `surv_ach_ref`, `craftax_potential.survival_potential`), which only the
  manager v1.3 run used.
- `HANDOFF.md`, `REWCAUSE_INTEGRATION.md` and the other families' docs
  (design notes, casebooks, experiment A/B data).

### Added
- **RSSM-2, the map model** (`dreamerv3/mapmodel.py`, `agent.mapmodel.*`): a
  deterministic GRU that ticks once per 8 env steps on the window's mean
  RSSM-1 feature and its moves, behind a stop-gradient each way, read by the
  actor and critic only (`actor2tensor`) through a learned gate. Its state is
  stored in the replay context (`map/deter2`, `map/count`) so chunks resume
  mid-episode. Loss scales `map`, `mappos`, `memfut`, `memrec`; head config
  `memhead`.
  - Map target (`target map`): a 12x12x16 coarse map of the level and the
    agent's coarse cell (`dreamerv3/craftax_map.py`; env
    `env.craftax.mapmodel`, keys `map12`, `mappos`, `mapseen`, `mapknown`,
    excluded from the encoder and decoder). The actor reads a 9x9-cell crop
    around RSSM-2's predicted position; `imag_shift` slides it through
    imagination (broken, default off).
  - Honest targets (default): built from the observation vector and the
    agent's own actions only (`ObservedTargets`), graded only on observed
    cells, with `hindsight` (an early tick graded against the mosaic at the
    end of the segment). `env.craftax.map_privileged True` reinstates the
    true map on every cell.
  - Memory target (`target memory`): RSSM-2 predicts RSSM-1's latent 1/2/4
    ticks ahead and recalls the observation 1/2/4 ticks back, fed raw
    actions; no labels.
  - `out['map_pred']` in `policy(mode='probe')` for `tools/map_eval.py`.
- **Reward shaping** (`env.craftax.survival`): `potential`, a potential over
  the tech-tree spine (`dreamerv3/craftax_potential.py`, `phi_scale`,
  `phi_gamma`); `shaped`, five hand-written survival terms (`surv_*`), kept
  to reproduce `map_survive`. `raw_reward` and `log/achievements` stay
  unshaped.
- Tests: `test_craftax_map.py` (targets, including against the game),
  `test_craftax_potential.py` (the ramp, spine recipes against the game,
  death and timeout), `test_mapmodel_step.py` (RSSM-2's step), and the first
  shaped step in `test_audit_fixes.py`.
- Tools: `map_eval.py`, `coverage_audit.py`, `mapwatch.py`,
  `make_architecture.py` (`docs/architecture.svg` / `.pdf`),
  `dump_trajectories.py`, `make_report.py`, `make_onepager.py`
  (`reports/craftax-result-2026-09-20.pdf`).
- Docs: `docs/rssm2-step-alignment.md`, `docs/raising-achievements.md`,
  `docs/ablations.md` (RSSM-2 alone, scoring noise); results of every map run
  in `docs/eval_results.json` and `docs/curves/`.
- Results (30 fixed worlds): honest map 6.17 ± 0.34 (pre-fix) and
  5.70 ± 0.21 (`honest_map_fix`, fixed code); privileged-map control 6.17;
  map + potential 5.47; map alone 4.73; RSSM-2 as a memory on plain
  DreamerV3 5.33 (plain 4.77). See `README.md`.
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
  (`dyn/`, `enc/`, `dec/` and, on this branch, RSSM-2's `map/`; ~1.4 GB a
  run instead of ~9.3 GB); a resumed run gets zeros back that the
  agent overwrites as it trains (`test_restore_save_skip`).

### Fixed
- RSSM-2 (audit 2026-10-11, `docs/rssm2-step-alignment.md`,
  `test_mapmodel_step.py`):
  - B01: `action_deltas` sent actions 5-42 to DOWN; only 1-4 move now.
  - B02: an episode reset kept the old RSSM-2 state until the window closed;
    it is cleared at once, and the reset step's action leaves the move sum.
  - B03/B04: training pooled fixed windows and handed each window's state to
    its earlier steps (up to 7 future steps). Training now scans the acting
    step (`Agent._map_step`), windows count from the episode start, and the
    replay context stores the window step.
- B07: the first step of each episode was not shaped; the start state's
  potential is set at reset.
- The potential kept its value on death, so shaping subsidised dying at high
  tech; a real death now pays back -PHI. `PLACE_TABLE` asked for one log
  where the game needs two, and the iron recipes lacked stone.
- RSSM-2 was trained on map cells the agent had never seen; honest targets
  and the `mapknown` weight replaced that (privileged kept as an ablation).
- `imag_shift True` let the rollout and the loss read different crops and
  collapsed the policy; it defaults to False.
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
