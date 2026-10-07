# Changelog

All notable changes to the double-RSSM Craftax work are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **Learned action mask (B3) labels.** B3 scored 5.60 achievements against
  B2's 7.60, because its labels came from latent predictions the head's own
  loss was reshaping (87% disagreed with the rules). There are two new
  options. `valid.detach` stops the head's gradient reaching the world model.
  `valid.label obs` labels "did something" when the decoded real-action
  prediction explains the real next observation better than NOOP's by more
  than `valid.margin` nats. Calibrated on B2's world model: AUC 0.83-0.88
  against 0.65 for the latent label; margin 0.5 with threshold 0.15 separates
  valid actions (TPR ~0.5) from invalid ones (FPR ~0.04). Defaults are
  unchanged (`label latent`), so old runs reproduce. The metrics
  `feas/evidence_auc` and `feas/tpr_*`/`fpr_*` are added. The calibration-only
  option `valid.rule_mix` masks that share of acting steps by the rules, so the
  check sees both valid and invalid actions.
- `run.from_checkpoint_regex` was read by the run scripts but missing from
  `configs.yaml`, so `--run.from_checkpoint` crashed.

### Changed
- **Honest RSSM-2 targets are now built from the observation vector and the
  agent's own actions only** (`craftax_map.ObservedTargets`); the env's honest
  path never receives the game state. Terrain is a mosaic of the lit 9x11
  windows decoded from `obs['vector']`; position is dead-reckoned from the fixed
  spawn, judging each move by the tile the agent could see in front of it.
  Previously the position label and the mosaic's placement read the true
  coordinates -- recoverable on the surface, but after a ladder the game
  teleports the agent somewhere it cannot know, so below the surface both
  leaked. Each level now gets its own frame anchored where the agent arrived.
  Checked against the real game: 100% position agreement and 0 wrong tiles of
  337k over random surface rollouts; through the env wrapper the label equals
  the true cell on 2489/2489 surface steps. `tools/map_eval.py` uses the same
  observer for its seen/unseen split, on the surface only.
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
- **Achievements past index 24 were reported under the wrong names.** Craftax's
  `Achievement` enum is not declared in value order -- positions 25-66 are
  shuffled -- and `death_eval`, `watch_agent`, `play_agent`, the env wrapper and
  a test named the state's achievements array by enum iteration order. Index 29
  (`ENTER_DUNGEON`) printed as `MAKE_IRON_ARMOUR`. Every earlier evaluation
  only unlocked indices 0-24, which happen to line up, so no previously
  reported per-achievement number changes; the potential's 13 spine
  achievements are all below 25. Now named by index everywhere, with
  `test_achievement_names_follow_indices`.
- **The Craftax adapter declared `obs['vector']` in [0, 1], which crashes the
  first agent to level up.** Craftax divides health, food, drink, energy and
  mana by 10, and their maxima grow with attributes (up to 13 and 21); XP is
  unbounded. Levelling needs XP, and XP comes only from reaching a new floor,
  so no run hit this until experiment B2 (masked policy) entered the dungeon at
  ~306k steps and the space check raised on a value of 1.1. The upper bound is
  now unbounded.
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
- Manager v1.3 in the comparison report and the training-curve figure, and
  normalized return now *measured*: each run is evaluated with Craftax's
  unshaped reward (env `survival none`) over the 30 worlds, instead of being
  estimated from achievements.
  - v1.2 and v1.3 both score 3.91% (mean return 8.83).
  - v1.3 ties v1.2 on achievements (9.67 vs 9.73), lives 73 steps longer,
    dies to mobs far less (5 vs 14 of 30) and reaches the dungeon (3%), but
    gives up the stone tools. Hunger becomes its main killer.
- v2, the two-level agent without game knowledge (`docs/design-v2.md`,
  `docs/architecture-v2.svg`). It is built and debug-tested but not run yet.
  Each piece is a flag, and the defaults keep every earlier run reproducible:
  - **Learned mask for all actions** against a random-action reference
    (`valid.exempt_basic False`, `valid.reference random`).
  - **Learned goal codebook** (`manager.goals learned`, new
    `dreamerv3/goalcodes.py`): 16 types of 8-step latent change, trained as a
    VQ-VAE with moving-average codes. A goal is reached when the change since
    it was set is classified as its code.
  - **Curiosity** (`curiosity.enabled`): ensemble disagreement as a third
    reward stream and critic head.
  - **RSSM-2 memory** (`mapmodel.target memory`): raw actions in; predicts
    RSSM-1's latent 1/2/4 ticks ahead and recalls the observation 1/2/4 ticks
    back, with no map labels.

  The published 1M-step agents use none of the knowledge v1.2 does, so this is
  the comparable line. The honest map stays as an ablation
  (`mapmodel.target map`).
- Manager v1.2 in the comparison report and the training-curve figure: the
  new best run. It scores 9.73 achievements per episode on the 30 evaluation
  worlds (v1.1 8.53, B2 7.60), 3.9% normalized return. It is the first run to
  make the stone pickaxe regularly (23%), and it also mines coal (10%). Its
  manager favours iron-tier goals it cannot complete yet, which pay the bottom
  actor for gathering every ingredient. It dies earliest of the leading runs
  (221 steps), half of those deaths to mobs or lava.
- Manager v1.1 and v1 in the comparison report and the training-curve figure
  (`docs/eval_results.json`, `docs/curves/mgr2.json`). v1.1 is the new best
  run: 8.53 achievements per episode on the 30 evaluation worlds (B2 7.60,
  v1 7.13), 3.4% normalized return. It places the furnace and stone in 67% of
  episodes (B2 40%) and makes a stone sword (23%) and arrows (10%) for the
  first time. The report's page-1 table now sizes itself to the number of runs.
- Experiment B3 (`--agent.valid.learned True`): the action mask without the
  game's rules. The `feas` head is trained on the world model's own
  counterfactual: an action did something when it explains the real next
  latent better than NOOP by more than `valid.evidence` nats. That head, not
  `obs['valid']`, masks the actor while acting. There is 1% exploration past
  the mask, and the mask stays off for the first 50k train updates, because
  an untrained world model labels almost nothing as having an effect. B2's
  hand-transcribed rules were game knowledge used at test time, which the
  published 1M-step baselines do not use. See
  `docs/entropy-and-action-suppression.md`.
- Manager v1.3 (`--env.craftax.survival potential+meters
  --env.craftax.goals_survival True`, on top of v1.2):
  - A survival potential over food and drink. A meter's value is
    1 − (1 − m/9)², so it is steep when empty. Its weight grows with tech
    progress and achievements unlocked: w0 · (1 + T + n/10), with w0 = 0.75.
  - DRINK and EAT goals for the manager, reached at meter ≥ 8.

  B2 runs either in survival mode (400+ steps, 7 drinks, little tech) or in
  tech mode (5 stages, about 2 drinks, dead of thirst near step 330), never
  both. Thirst only cost it a death ~200 steps later, which it never learned
  to anticipate. Tests: `test_craftax_potential` (survival potential) and
  `test_craftax_goals` (survival goals against the game's meters).
- Manager v1 (`mgr`) on the training-curve figure: 7.5 achievements per
  episode over its last 100k steps, against B2's 7.9, with its manager still
  choosing goals uniformly at the end.
- Runs take ~1.4 GB on disk instead of ~9.3 GB. `replay.save_skip`
  (default `['dyn/', 'enc/', 'dec/', 'map/']`) keeps the replay_context
  latents out of the saved chunks. `dyn/deter` alone was 89% of a run's disk
  and barely compresses. Their shapes go to `replay/skipped.json`, and a
  resumed run gets zeros back that the agent overwrites as it trains
  (`test_restore_save_skip`). `tools/strip_replay.py` does the same to
  finished runs. It refuses a live run or one short of its `run.steps`. It
  freed ~69 GB across the ten finished runs.
- Manager v1.2 (`--agent.manager.critic shared --agent.manager.hold 32`):
  - One critic with two heads: 'game' judges both actors, 'goal' only the
    bottom actor. Each stream is normalised separately and the two are
    combined by `goal_weight`, so the reward sizes are chosen rather than
    accidental.
  - The manager drops its own critic and bonus. It takes the exact gradient
    over all 13 goals from the game head's Q(s, g).
  - Goals are held until reached or for 32 steps.
  - Imagination resumes the replay's goal and step (`gphase` is now stored in
    replay), which brings the replay value loss back.

  v1.1's once-per-episode manager bonus stopped paying for repeated logistics,
  and its single combined return paid one stone pickaxe five ways at
  unchosen sizes. Defaults keep v1/v1.1.
- Manager v1.1: reaching the goal pays both levels. The actor gets +1 the
  first time its goal is reached in a segment. The manager gets +0.5 the first
  time in the episode a goal it set is reached; it reads a new observation-only
  input, `obs['goalreach']` (13 reached-this-episode flags), and goals that
  already hold are masked from its choice. v1 never trained its manager: after
  900k steps it still picked uniformly. Most goals cannot be reached in 8
  steps, so the actor ignored them, and then the choice changed nothing. See
  `docs/design-manager.md`.
- Two-level agent (`agent.manager.enabled`, `env.craftax.goals_obs`;
  design in `docs/design-manager.md`). A manager actor-critic picks one of
  13 tech-tree goals (`craftax_goals.GOALS`) every 8 steps, reading RSSM-1's
  state and RSSM-2's slow state. The actor sees the goal and is paid the change
  in its predicted progress (`gphi` head, trained on observation-only targets)
  on top of the game reward. The manager is paid the game reward only, so it
  learns the order of the tech tree instead of being told it, and its critic
  bootstraps 8 steps at a time. B2 unlocks almost nothing after step ~160;
  the iron tier needs plans far longer than the actor's 15-step imagination,
  and imagining further had already failed. The replay value loss is off in
  this mode (see the design doc). `test_craftax_goals` checks goal progress
  against the game state.
- `docs/b2-casebook.pdf`: B2's shortfalls measured step by step from replay
  (`tools/episode_cases.py`, data in `docs/cases/`), compared with B1, the
  honest run and vanilla, then shown as frames from B2's own episodes
  (`tools/make_casebook.py`, drawn with Craftax's textures from the decoded
  observation). It measured the cause of the table-to-pickaxe leak, which had
  been inferred: B2 places a table the moment it holds two logs (98% of its
  2.2 tables per episode), so every table-but-no-pickaxe episode is left with
  0 wood. A stone-pickaxe chance lasts a median of one step, and 36% end with
  the only log going into the wood sword. 41% of episodes never drink. The
  report's "Where B2 falls short" page now states these measurements instead
  of the earlier guesses.
- Training curves for every run, rebuilt from replay
  (`tools/curves_from_replay.py`, `tools/plot_training_curves.py`,
  `docs/training-curves.png`/`.pdf`, data in `docs/curves/`). Seven of the ten
  runs never logged achievements, but each replay chunk stores the
  achievements unlocked per step, so the episode-final count is recoverable for
  the whole run. Checked against vanilla's own logged curve: r = 1.000, same
  mean. B2 leads from the first 50k steps and is still rising at 1.1M.
- `tools/diagnose_run.py` and a "Where B2 falls short" page in the comparison
  report (`docs/diagnosis.json`): over B2's last ~500 training episodes it
  reaches the furnace in 46% but the stone pickaxe in 3%. Of episodes that mined
  stone, 73% held wood and stone together but only 22% ever stood at a table
  with both. Thirst is still the first meter to run out in 45% of deaths.
- Experiment B results in `docs/comparison-report.pdf` and
  `docs/entropy-and-action-suppression.md`. Masking impossible actions (B2)
  reaches 7.60 achievements against 5.87 for the same model without it, and
  places the furnace (40%), places stone (40%) and mines coal (20%) -- all at 0%
  before. Validity flags as an input (B1) reach 6.90. Experiment A on the final
  checkpoints shows the frontier keys coming off 0% where they work only under
  masking. Raw files in `docs/experiment_a/expB_*.json`.
- Experiment B, two switchable arms on top of the honest-map configuration
  (`env.craftax.valid_obs True` plus one of):
  - `agent.valid.input True` (B1): the 43 validity flags enter the encoder, so
    RSSM-1's latent -- and every imagined step -- carries them.
  - `agent.valid.mask True` (B2): impossible actions get zero probability and
    zero gradient, so their logits are never pushed down where they cannot
    work. Acting uses the true flags; imagination uses a learned `feas` head on
    the latent, read by both the rollout and the loss so they cannot disagree.
  `dreamerv3/craftax_valid.py` computes the flags from the observation vector
  alone; `test_craftax_valid.py` checks it against the game (0 disagreements in
  77,700 action-state checks).
- `tools/action_suppression.py` (experiment A): rolls a checkpoint and, at each
  step, asks the game which of the 43 actions would change anything (state
  copied, every action stepped with one key, compared with doing nothing).
  Reports valid-action mass, conditional vs marginal entropy, opportunity
  probabilities and each key's probability where valid vs invalid. Results for
  four checkpoints in `docs/experiment_a/` and
  `docs/entropy-and-action-suppression.md`: only 36-40% of probability lands on
  actions that do anything; exactly one wood craft survives per run (the
  control gave the pickaxe 0.0% across 150 craftable states); the furnace was
  never placed in 386 (old record) and 133 (honest run) states where it could
  have been. The tech tree stalls at suppressed frontier keys, not unreached
  states.
- `honest_map` scored and added to `docs/comparison-report.pdf`: the
  observation-only map model reaches 5.87 achievements, level with the
  privileged control's 5.93, so dropping privileged supervision cost nothing
  measurable. With the same recipe and death fixes it makes a pickaxe in 20% of
  episodes, which the control never did -- the fixes are unlikely to be what
  removed the control's pickaxe.
- `docs/comparison-report.pdf` (from `tools/make_comparison_report.py` and
  `docs/eval_results.json`) -- every finished run compared on the same fixed
  evaluation worlds, with the unlock rate of every achievement any run reached.
  Two findings: the control run with the recipe and death fixes scores the
  highest total (5.93) but never makes a pickaxe (0/30 vs 9/30 for the old
  record); and WAKE_UP (sleeping) accounts for the map runs' whole lead over
  vanilla -- without it the fixed map model scores 4.20 against vanilla's 4.44.
  Re-scoring two old checkpoints with the current code reproduced them exactly.
- `docs/entropy-and-action-suppression.md` -- reference note on reading policy
  entropy: competence-conditioned concentration vs action-support collapse, the
  valid-action-suppression mechanism (Zabounidis et al. 2026, verified), our
  audit numbers, better metrics than raw 43-action entropy, and experiments A-D.
  Also records the status of the LLM-memory and LLM-reward directions, with the
  verified SCALAR citation.
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
