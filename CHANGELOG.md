# Changelog

All notable changes to the double-RSSM Craftax work are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Comparison report: 28 runs** (`docs/comparison-report.pdf`,
  `docs/eval_results.json`). Adds the four ablations (plain and manager with
  two seeds), v1.2 without the mask and next-taste, with updated summary and
  conclusions; the training-curve page adds them too. `abl_plain_s1` scores
  4.77, the same as seed 0.
  - Bold table rows now mark runs new since the last report (`new` in the
    entry, `--new` in `tools/eval_entry.py`), no longer the fixed-code runs.
- **v1.2 without the mask, and the manager ablation's second seed**
  (`docs/ablations.md`, curves `docs/curves/v12_nomask.json` and
  `abl_manager_s1.json`).
  - `v12_nomask` scores 6.13 on 60 test worlds: the mask is worth
    +2.90 ± 0.26 under the v1.2 manager, as much as without one.
  - `abl_manager_s1` ends below plain in training (5.03 against 5.41) and
    scores 5.30 on worlds 0–29. With two seeds the manager is +0.20 ± 0.24
    over RSSM-2 alone: not established.
  - `next_notaste` crashed a second time on its first policy step (Xid 13).
    `~/run_notaste3.sh` tries it once more after `abl_plain_s1`, and if it
    dies again, runs `next_lam0` (taste on the manager with `lam_mgr 0`) as
    the no-taste control.
- **next_taste result** (`docs/taste-prior.md`, curve in
  `docs/curves/next_taste.json`). It trains to 9.2 achievements per episode
  at 1.08M steps, above fixed v1.2 (8.74) and B2 (8.79). On 60 test worlds
  it scores 9.38 ± 0.34: +0.35 ± 0.35 over fixed v1.2, +0.85 ± 0.39 over
  B2 and +2.78 ± 0.48 over v2 (30 worlds). But taste locked
  the manager onto `place_table` by 90k steps (goal entropy 0 from then on),
  so the run measures a fixed goal, not taste. `next_notaste` crashed on a
  GPU fault (Xid 69) 90 s in; it reruns after `v12_nomask` via
  `~/run_ablseeds2.sh`, followed by the two ablation seeds.
- **Ablation results and a scoring-noise check** (`docs/ablations.md`,
  curves in `docs/curves/abl_*.json`).
  - On plain DreamerV3, only RSSM-2 + the v1.2 manager comes out ahead in
    both measures: +1.00 ± 0.35 on the test worlds and +0.6 in training.
    RSSM-2 alone is unclear, and taste over actions does nothing.
  - Rescoring the same checkpoints on a second set of 30 worlds puts one
    30-world score's noise at about ±0.55.
  - Fixed v1.2 leads B2 by +0.50 ± 0.25 over 60 worlds but is level in
    training.
  - Old v1.2's 9.97 is 9.73 over 60 worlds. That is a real lead over the
    fixed v1.2 seeds, plus about +0.25 of world luck.
  - Queued in `~/run_ablseeds.sh`:
    - `v12_nomask`: fixed v1.2 minus the rule mask, to measure the mask
      under the v1.2 manager;
    - second seeds of `abl_manager` and `abl_plain`.
  - v2-norules' curve is in `docs/curves/v2_norules.json`.
- **Taste on the manager** (`agent.taste.target manager`, design in
  `docs/taste-prior.md`). Taste becomes an internal reward on the manager's
  choice of goal. It is added to Q(s, g) in `_mgr_loss_shared` and learned
  from the real return after each choice, compared with the manager's
  expected value Σ π Q.
  - The bottom actor is untouched. It needs the named goals and the shared
    critic.
  - Queued on the box (`~/run_ablq.sh`) after the ablations below:
    `next_taste`, then its no-taste control `next_notaste`. Both use the
    next agent: B2 mask, curiosity, v1.2 named goals, the rest as fixed v2.
- **Ablation queue: each addition alone on plain DreamerV3** (no rule mask,
  no shaping, no curiosity, same code):
  - `abl_rssm2`: RSSM-2 as a memory;
  - `abl_manager`: RSSM-2 + the v1.2 manager (the manager reads RSSM-2, so
    its own share is `abl_manager` − `abl_rssm2`);
  - `abl_taste`: taste over actions, centered, equal weights, w0 0;
  - `abl_plain`: plain DreamerV3 itself. The old vanilla run used other
    code.
- `taste.center`: each slot's evidence is compared with the fold's average,
  the fix for v2-taste's negative bias. `Taste.observe(..., steps=)` counts
  sparse evidence (manager choices) as a share of all env steps. The
  defaults (`center False`, `target actions`) reproduce v2-taste.
- Tests: centering ignores a constant shift (two different batches) and
  undoes v2-taste's most-pressed-sinks-most failure; sparse counts over two
  folds; the all-goals critic input matches the replay layout.
  - The manager evidence files ret[t] − Σπ Q at exactly the chosen steps,
    skipping episode ends and the cut last step.
  - The row cap drops and counts any choices beyond it.
  - Relative taste is neutral to a shared level.

### Fixed
- Manager taste, from code review before the first run:
  - **The evidence used the slow critic** while the return and the manager
    use the fast one. It now uses the fast one.
  - **A level shared by the 12 goals handicapped NONE.** Taste now enters Q
    relative to the goals' mean.
  - **The manager reused the actions target's `lam` (4.92 logits).** It now
    has its own key, `lam_mgr` (1.0).
  - **The evidence scored all replay rows for 13 goals.** Only the decision
    rows are scored now, capped by `taste.rows`.
  - **`q_spread` mixed taste into the game spread.** It is now logged before
    taste, with `q_spread_taste` after.
  - **The new step counter broke v2-taste checkpoints.** It now exists only
    for the manager target.
  - The queued run uses equal good and bad evidence weights (1.43). The
    toy's 3:1 ratio pins every above-average goal at +1 once evidence is
    centered.
- **v2-taste scored: 5.33 achievements on the 30 fixed worlds** (1.96%
  return). That is −1.69 ± 0.25 against fixed v2 (three-seed mean 7.02),
  level with the honest map (5.70), and +0.70 against vanilla. It never makes
  a wood pickaxe.
  - Taste turned against nearly every special action. The bias tracks how
    often an action is pressed (rank correlation −0.98): placing a table ends
    as low as an impossible iron sword, at −4.5 logits.
  - Write-up in `docs/taste-prior.md`. The data (`taste/` metrics and the
    final per-action taste) is in `docs/taste-run.json`. The training curve
    is in `docs/curves/v2_taste.json`.
- `tools/eval_entry.py` turns a `death_eval.py` log into a
  `docs/eval_results.json` run entry. It reproduces the existing entries
  exactly.
- **Taste over actions** (`dreamerv3/taste.py`, `agent.taste.*`,
  `docs/taste-prior.md`): a knowledge-free stand-in for v2's rule mask,
  ported from the plastic-reward toy.
  - It is a per-action logit bias, learned from the critic's advantage at the
    real replay steps where each action was taken.
  - The rule constants are the toy's evolved champion. In the toy it helped
    in all 11 worlds (+11 to +37 lifespan), where the same taste as a reward
    lost up to −81.
  - Acting, imagination and the actor loss share the bias through
    `Agent._masked`. `repl_loss` also returns its value baseline.
  - Tests are in `test_taste.py`.
- `tools/compare_at_steps.py`: achievements per training episode at matched
  env steps, from `curves_from_replay` JSONs, with seeds averaged and an SE
  over seeds. It compares a run still training with finished runs at the
  same step.
- Training curve for fixed v2, seed 2 (`docs/curves/v2_fix_s2.json`).
- Training curve for fixed B2, seed 1 (`docs/curves/expB_mask_fix_s1.json`): 7.70 ± 0.59 achievements on the 30 fixed worlds vs 8.27 for seed 0.

### Changed
- **Comparison report adds v2-taste and the three-seed results** (22 runs).
  The seeds of fixed B2, v1.2 and v2 had been scored but were never reported.
  Their means on the test worlds are B2 8.33, v1.2 8.78 and v2 7.02.
  - Paired comparisons now use the per-world mean over the three seeds.
    These hold up: the mask over the honest map (+2.63 ± 0.38), v2 over
    vanilla (+2.39 ± 0.28) and B2 over v2 (+1.31 ± 0.32).
  - Not established: v1.2 over B2 (+0.44 ± 0.33).
  - The training-curve plot colours v2-taste and greys the old-code manager
    runs.
- **Comparison report on fixed evaluation worlds.** All 17 old checkpoints
  were re-scored, plus the four reruns on the fixed code (honest map, B2,
  v1.2, v2). Each is evaluated on the same 30 worlds, with one world hash
  across all runs. The table now shows each score ± its standard error over
  the worlds, and the raw normalized return.
  - Holds up on the fixed code: the action mask (+2.57 ± 0.48), and v2 over
    vanilla (+2.53 ± 0.44).
  - Not established: the manager over B2 (+0.50 ± 0.57), and the order among
    v1.1/v1.2/v1.3.
  - v1.2 fell from 9.97 to 8.77 after the fixes; the other three reruns are
    within noise of the old runs.
- `death_eval.py --config-from-run` rebuilds a checkpoint from its saved
  `config.yaml`, layered over the code's defaults. Older envs without
  `reseed()` are reseeded through their key.
- The training-curve plot adds the fixed runs.

### Fixed
- **RSSM-2: training no longer sees what acting cannot, and episodes no
  longer leak** (found in an outside code review; details in
  `docs/rssm2-step-alignment.md`). Every run with RSSM-2 so far (honest map,
  B1, B2, B3, v1.x, v2) predates these fixes.
  - `mapmodel.action_deltas` clipped the action to 0..4, so DO, SLEEP and every
    crafting action (5-42) counted as a step DOWN in RSSM-2's movement input
    (map mode only; v2's memory mode uses raw one-hots).
  - On an episode reset mid-window, the acting step kept the old episode's
    RSSM-2 state until the window closed, then fed it into the new episode's
    first update. It is now cleared at once.
  - The batch path pooled fixed 8-step windows and gave every step its own
    window's result, so imagination starts and the manager saw up to 7 steps
    of their future. Acting saw only the last closed window, which could be up
    to 7 steps stale. Windows also followed the replay chunk rather than the
    episode start, and a reset inside a window was missed.

    Training now scans the acting step (`Agent._map_step`) over the batch, so
    both paths read identical states. Windows count from the episode start;
    the replay context stores the window step (`map/count`).
  - Tests: `dreamerv3/test_mapmodel_step.py` covers all 43 actions, a reset
    mid-window, the batch path matching the online path, window timing,
    causality under future perturbation, and a resumed chunk.
- **Audit round 2** (`docs/audit-2026-10-11.md`). Tests in
  `dreamerv3/test_audit_fixes.py`.
  - **Fixed evaluation worlds.** Reset and every env step drew from one key
    stream, so from the second episode on an evaluation world depended on how
    long the agent's earlier episodes ran. `death_eval.py` now reseeds each
    episode (`env.reseed(seed + i)`) and prints a world hash that must match
    across compared runs. Past 30-episode comparisons were therefore paired
    only on the first world.
  - The first step of each episode is now shaped: the start-state potential is
    set at reset (`craftax.py`).
  - The RSSM-2 movement sum drops the reset step's action, which belongs to
    the previous episode.
  - v2: a goal set before the replay window now restarts at the window's
    first step with a matching step count (`agent.goal_start`). The goal
    codebook skips batches with no valid change, and revives dead codes only
    onto valid ones.
  - `death_eval.py`:
    - `--greedy` now really takes the argmax action and goal, through the new
      `policy(mode='greedy')`;
    - endings are split into death, timeout and cutoff, and only deaths get a
      cause;
    - raw Craftax return and shaped return are reported separately, with the
      normalised return.
  - `report()` now includes its loss metrics (it ran `mets.update(mets)`).

### Added
- `tools/death_eval.py --config-from-run` rebuilds a checkpoint from its logdir `config.yaml` (this code's defaults fill options added later); other flags override.
- **v2, v2-cur and B3 in the comparison report.** On the 30 evaluation worlds:
  - v2 scores 6.90 achievements, a real normalized return of 2.65%.
  - v2-cur scores 7.00, a normalized return of 2.70%.
  - B3 scores 5.60.
  - For comparison: B2 7.60 and v1.2 9.73 (3.91%).

  Neither v2 run makes a stone tool. The training-curve plot adds the three
  runs and greys out the older context runs. When the summary and the table
  no longer fit on one page, the report moves the table to its own page.
- **`curiosity.normalize` / `curiosity.retlimit`** turn curiosity on
  properly. In v2's first run the explore stream shared the game's return
  floor of 1. Its returns span only ~0.02, so they were never rescaled, and
  curiosity carried ~2% of the actor's signal. The manager's explore value was
  just as faint. With `normalize True` the stream is divided by its own spread
  (floor `retlimit`, 1e-3), and the manager's explore value is put in the same
  units. v2-cur (weights 0.1/0.1) is queued after v2. The default (`False`)
  reproduces v2.

### Changed
- **v2 keeps B2's rule-based action mask** (`valid.mask True`, env
  `valid_obs True`) instead of the learned one. B3 scored 5.60 against B2's
  7.60. B3-fix, at 254k steps, allowed only 45% of the rule-valid special
  actions: it pressed PLACE_TABLE in 1.3% of the states where it was possible
  (B2: 45%) and MAKE_WOOD_PICKAXE in 0.1% (B2: 100%). Its label cannot tell
  "does nothing" from "the world model has not learned this effect yet", and
  the mask then starves the world model of exactly those examples. The rules
  are now v2's only game knowledge. `docs/design-v2.md` and
  `docs/architecture-v2.svg` are updated.

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
