# Changelog

All notable changes to the double-RSSM Craftax work are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- **RSSM-2 is no longer trained on ground truth it could not have observed.**
  The map target is now a mosaic accumulated from the agent's own lit 9x11
  windows (`craftax_map.coarse_map_observed`, `update_known`), so an outside
  observer watching only the agent's screen could reconstruct every label. The
  BCE is weighted by `obs['mapknown']` -- the fraction of each coarse cell
  actually observed -- so unseen cells contribute no gradient. `coarse_map` and
  `env.craftax.map_privileged: True` are kept to reproduce the old runs as an
  ablation: it supervises every cell (`mapknown` = 1 everywhere) and should be
  run with `agent.mapmodel.hindsight False`, which together reproduce the old
  loss exactly.

  The capability that motivated full-map supervision is preserved by HINDSIGHT
  (`agent.mapmodel.hindsight`, default True): an early tick is graded against
  the mosaic as it stood at the end of the episode segment, so the model is
  still asked to predict terrain it has not reached, and is marked once the
  agent gets there. Every label remains an observation, just a later one.

  Both the masking and the hindsight apply to the 13 TERRAIN planes only. The
  mob planes ("visible right now") and P_SEEN (the agent's own visitation
  record) are facts about the current tick that are known everywhere, so they
  keep their causal target at full weight -- otherwise the model would be asked
  where cows will wander, and could never learn to say "not seen yet".

  Measured: 25.9% of the 12x12 grid is observed per episode, so ~74% of the old
  gradient came from cells with no observational basis. On a fresh map every
  episode that signal cannot be learnable, and the eval below shows it was not
  merely useless but harmful.
- Map BCE is normalised by observed-cell mass and rescaled by the cell count
  instead of summed over all cells. With weights all ones the value is
  bit-identical to the old sum; without the rescale, masking to ~26% coverage
  would silently shrink the map loss ~4x against every other term.

### Fixed
- **The potential kept its value on death, so shaping subsidised dying at high
  tech.** Potential-based shaping is only policy-invariant with PHI = 0 at
  absorbing states; ours returned before ever looking at `is_terminal`, so over
  an episode the shaping telescoped to `gamma^T * PHI(s_T) - PHI(s_0)` and an
  agent that climbed to a pickaxe and died kept the reward for the climb. A real
  death now pays `-PHI(s)` on its final step; timeouts, which bootstrap, keep
  their potential (`craftax_potential.shaped(terminal=...)`). Episode length had
  never moved off the random band, which this is consistent with.
- **Tech-tree potential asked for one log too few to place a crafting table.**
  `craftax_potential.SPINE` listed `PLACE_TABLE` at `wood=1`; Craftax's
  `place_block` requires `inventory.wood >= 2` and spends both. The ramp
  therefore read "fully prepared" one log early, the keypress silently failed,
  and no rung on the ramp paid for chopping the second log. Because the table
  gates every craft in the game, the whole spine stalled behind this off-by-one
  (`dreamerv3/craftax_potential.py`).
- `MAKE_IRON_PICKAXE` and `MAKE_IRON_SWORD` were missing `stone` from their
  ingredient lists; both recipes consume wood, stone, coal and iron.

### Added
- `docs/architecture.svg` / `.pdf` -- the current model on one page: where the
  potential sits (env wrapper, reaching the agent only through reward), what
  RSSM-2 reads (sg(feat) and its own actions), which heads see the map (pi and
  V only), and every stop-gradient. Regenerate with
  `python tools/make_architecture.py`.
- `tools/map_eval.py` -- scores RSSM-2's map belief against the true map, split
  into seen / unseen / marginal-prior floor, per plane. Ground truth is used
  here and only here: measuring against it was never the problem, learning from
  it was. Under observation-only training the unseen cells become a genuine
  held-out set, which the training run previously had none of -- every cell the
  model was scored on, it had also studied.
- `out['map_pred']` in the agent's `probe` mode, so the eval can read RSSM-2's
  belief without a training run.
- `tools/action_audit.py` and `tools/coverage_audit.py` -- what the agent spends
  actions on (impossible presses, inert steps, opportunities skipped) and how
  much of the coarse map it observes.
- Six tests in `dreamerv3/test_craftax_map.py` pinning the observation-only
  targets, including `test_mosaic_ignores_terrain_outside_the_window`, which
  scrambles every unseen cell and asserts the target does not move.
- `test_spine_ingredients_match_the_game` and
  `test_one_log_is_not_enough_for_a_table` in
  `dreamerv3/test_craftax_potential.py`, pinning every spine recipe against the
  costs read out of `craftax.craftax.game_logic`, so the table cannot drift away
  from the game again.
