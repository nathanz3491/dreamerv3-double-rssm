# Changelog

All notable changes to `family/v` (the goal-manager family: manager v1-v1.3
and v2 on DreamerV3 for Craftax) are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The branch is `main` plus this family; `main`'s own entries (the plain
Craftax adapter, the scoring tools, the general fixes) are in `main`'s
`CHANGELOG.md`. The full history up to the split, all families together, is in
`CHANGELOG.md` at tag `archive/full-2026-11-08`.

## [Unreleased]

### Added
- **The goal-manager family, re-added on top of the plain `main`** from the
  pre-split code (`49a6bc3`). Config keys, module and parameter names, RNG
  order and loss order are the pre-split ones, so old checkpoints load and the
  agent behaves the same:
  - RSSM-2 (`dreamerv3/mapmodel.py`, `agent.mapmodel.*`): map target
    (`craftax_map.py`, env `mapmodel`, `map_privileged`, `seen_decay`) and
    memory target (`memhead`), with the 2026-10-11 step-alignment fixes;
  - manager v1/v1.1 (separate critic, reach bonuses), v1.2 (shared two-headed
    critic, held goals, `imag_loss_streams`), v1.3 (survival goals,
    `survival potential+meters`), v2 (learned goal codebook, `goalcodes.py`,
    with the B06/B08 fixes);
  - named goals (`craftax_goals.py`, env `goals_obs`, `goals_survival`,
    `gphihead`);
  - curiosity (`agent.curiosity.*`, incl. `normalize`/`retlimit`);
  - potential shaping (`craftax_potential.py`, env `survival potential` and
    `potential+meters`, `phi_*`, `surv_w0/kappa/ach_ref`), with the B07
    first-step fix;
  - the B2 rule mask (`craftax_valid.py`, env `valid_obs`, `agent.valid.mask`,
    `threshold`, `validhead`), **default off and counted as cheating**.
- Tests: `test_craftax_goals`, `test_craftax_map`, `test_craftax_potential`,
  `test_craftax_valid`, `test_mapmodel_step`, and the family parts of
  `test_audit_fixes` (first shaped step, goal codebook, goal starts).
- Docs: `docs/design-manager.md`, `docs/design-v2.md`,
  `docs/architecture-v2.svg`, `docs/rssm2-step-alignment.md`, the family's
  part of `docs/ablations.md`, and the whole `docs/audit-2026-10-11.md`.
- Results: `docs/eval_results.json` holds the plain baselines and this
  family's scored runs (honest first, masked ones marked `[masked]`);
  `docs/curves/` adds every family run; `tools/plot_training_curves.py` lists
  them (masked runs as grey context); the training curves and the comparison
  report are regenerated.

### Changed
- **Split: this branch holds the goal-manager family only.** Checked against
  the pre-split code on CPU (parameter names, shapes, dtypes and initial
  values; policy outputs in train, greedy and probe mode; parameters and
  optimizer state after three and four train steps on fixed batches; train
  and report metrics):
  - bit-identical for the plain flags and those of `v2_fix` and
    `abl_manager`;
  - `v12_nomask`, `mgr`, `mgr4`, `v2cur`: identical except 1-5 recorded
    elements of the initial arrays (a ~1e-310 denormal in a zero-initialised
    optimizer slot; or a few weights out of millions, by up to 6e-5). Every
    value computed from those arrays is identical, including the same weights
    after training, and the denormal also differs between two runs of the
    pre-split code, so these are faults in the recorded copy, not code
    differences. The box segfaulted in `xla_extension.so` in about one run in
    four during these checks.

  All 16 runs' flags parse, and the parsed config equals each run's saved
  `config.yaml` on every key both have. Tests: `pytest dreamerv3
  embodied/tests` (without the two `zerofun` modules) gives 259 passed, 55
  failed; the 55 (53 in `test_replay.py`, `Replay.dataset`; 2 in
  `test_driver.py`, a `KeyError`) fail the same way on the pre-split code.
- The rule mask always leaves NOOP, the moves and DO allowed (the pre-split
  `valid.exempt_basic True`, which every run used).
- `test_craftax_valid` carries its own game oracle (`make_oracle`) instead of
  importing it from `tools/action_suppression.py`, which is on
  `family/masked`.
- `tools/death_eval.py --greedy` is documented as taking the argmax goal as
  well as the action.

### Removed
- Code of other families, kept out of this branch: taste and its hooks
  (`_masked` taste bias, the replay evidence, the manager's taste reward,
  `repl_loss`'s `outs['val']`); the B1 encoder input (`valid.input`) and the
  learned mask B3 (`valid.learned`, `feasclock`, `_feas_learned` and its
  options); the five-term `shaped` survival reward and its `surv_*` keys; the
  reward-cause head. This family's runs used none of them; their saved configs
  have them at the defaults, which `--config-from-run` drops.

## History before the split (summary)

From the pre-split changelog; numbers as reported then, before the
evaluation worlds were fixed unless stated.
- **v1** (`mgr`): a manager over 13 tech-tree goals on top of B2. It never
  trained its manager: after 900k steps it still chose uniformly.
- **v1.1** (`mgr2`): reaching the goal pays both levels; the manager reads
  the reached-this-episode flags and cannot choose goals that already hold.
- **v1.2** (`mgr3`): one two-headed critic shared by both actors, the exact
  manager gradient over all goals, goals held until reached; imagination
  resumes the replay's goal and step (`gphase` stored in replay).
- **v1.3** (`mgr4`): survival goals and the survival potential: same score,
  73 steps longer lives.
- **v2** (`v2`, `v2cur`): learned goals, curiosity, RSSM-2 as a memory; kept
  B2's mask after the learned mask B3 failed. `curiosity.normalize` added for
  v2-cur after curiosity carried ~2% of the signal in v2.
- **2026-10-11 audit**: RSSM-2 fixes (B01-B04) and the v2 goal fixes (B06,
  B08); `mgr3_fix` and `v2_fix` retrained with three seeds each.
- **2026-11**: `abl_manager` (two seeds) and `v12_nomask` measured the
  manager without the mask (`docs/ablations.md`).
