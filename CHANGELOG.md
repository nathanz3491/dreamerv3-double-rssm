# Changelog

All notable changes to `family/market` (the market agent on DreamerV3 for
Craftax) are recorded here. The branch starts from `family/next`, the base
agent's code, so `family/next`'s entries and `main`'s are kept below. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The history of every family up to the split is in `CHANGELOG.md` at tag
`archive/full-2026-11-08`; the next family's part is summarised under
[Family history](#family-history-from-49a6bc3s-changelog).

## [Unreleased]

### Added
- **`market_quota`'s score** (quota v1 only): 6.03 ± 0.24 on the 30 fixed
  worlds, 6.05 ± 0.18 on 60. That is level with `next_nomask` (5.82 ± 0.17),
  a paired gap of +0.23 ± 0.23, so the floor alone neither helps nor hurts. The
  README gains a Results section. `docs/eval_results.json` now holds
  `next_nomask` and `market_quota`; its text was still `family/next`'s.

### Fixed
- `test_market.py::test_fixed_price_is_price0_everywhere`: the agent stand-in
  lacked `_shift` (added with the moving price), so `Agent._price` raised
  AttributeError. A test-only fix; no agent code changed.

### Changed
- `market_fixed` and `market_full` now run on quota v2 (their README launch
  commands add `--env.craftax.quota.mode log`); `market_quota` stays v1.
- The market's consumers now read the price: `log(P*/price0)` (stop-gradient)
  joins the actor's, the critic's (and slow critic's) and the manager's input
  whenever `agent.market.enabled`; constant 0 under the fixed price. Needs
  `mapmodel.enabled` and `to_actor`. Before this the actor never saw P*, so the
  price net's per-step credit was mostly noise. Off changes nothing.

### Added
- **Quota v2** (design §4): `--env.craftax.quota.mode log` (`LogQuota` in
  `dreamerv3/quota.py`). A logarithmic floor F(t) = frac * a * ln(1 + t/c),
  (a, c) refitted at each life's end by least squares (closed-form a on a
  log-spaced c grid, no scipy) to the unlock points of the last `fit_lives`
  lives; every `every` steps a count below F(t) costs -k, with no restart on an
  achievement (so no pacing), and a death below F(t) costs one capped
  -k * `death_cap` instead of v1's lump. No floor before `min_lives` lives. New
  keys `mode` (`window`, the default: v1 unchanged), `frac 0.5`, `every 100`,
  `death_cap 1`, `fit_lives 50`, `min_lives 5`. Tests in `test_quota.py`.
- Quota logging, both modes: `log/quota_pen`, `log/quota_death`,
  `log/quota_floor` (and `log/quota_a`, `log/quota_c` in v2) with `logs`.
  Metrics only: the reward and the observations are unchanged.
- Launch `market_quota2` (market_quota's flags + `--env.craftax.quota.mode log`).
- **The moving price** (design §3): `--agent.market.shift True` (with
  `agent.market.enabled`; needs the RSSM-2 memory). A small net reads the
  observation's change over 8 steps (non-view subtracted, the view as 1 - cos)
  and draws two curve shifts; log-linear supply and demand cross at P*. P* is
  computed in `policy()`, stored in replay with the draw, held through each
  imagined rollout, and logged with Q* (`market/logp`, `market/logq`). The net
  learns by REINFORCE on the achievements in the next 32 steps. `dreamerv3/market.py`
  (pure numpy maths); tests `test_market_math.py` (no JAX) and
  `test_market.py` (JAX). The consumers read log(P*/price0), see Changed.
- **Q-heads and a fixed price P\*** (design §1, §2): `--agent.market.enabled
  True` adds Q_score and Q_life (43 scalar outputs each, their own small MLP on
  the actor's input, trained on imagined lambda-returns of the action taken).
  The critic, actor and manager value at `r_score + P* x alive` (P* =
  `agent.market.price0`, 0.01), and the critic gets each action's economic
  profit as an input. Default off. Tests: `dreamerv3/test_market.py` (needs
  JAX), `dreamerv3/test_market_config.py`.
- **Slope quota** (design §4): `--env.craftax.quota.enabled True` adds a
  penalty `k` to the reward when a window (`margin` x the running mean gap at
  the current achievement count) passes without a new achievement, and at death
  charges `k x ceil((L_nom - t) / N)`. `dreamerv3/quota.py` is pure numpy;
  tests in `dreamerv3/test_quota.py` (the env ones need Craftax). Default off.
- **Survival goals for the manager** (design §5), ported from `family/v`'s
  v1.3: `--env.craftax.goals_survival True` appends DRINK and EAT to the 13
  named goals in `obs['goalphi']` and `obs['goalreach']`; the agent reads its
  goal names off the observation (13 or 15). Default off. Test:
  `test_env_emits_the_survival_goals_with_goals_survival` (needs Craftax).
- **Branch `family/market`: the design of the market agent**
  ([docs/design-market.md](docs/design-market.md)); no code yet. The agent
  trades score against survival at a price P\*, which a small network moves
  when the world changes.
  - The base is `next_nomask`. The rule mask stays off, because it counts as
    cheating.
  - Two goods, score and survival, in one market. Per-action Q-heads for score
    and for life. The critic is given each action's economic profit (its value
    minus the best alternative's).
  - The price acts on the critic, the actor and the manager. Curves are
    log-linear with tanh-bounded shifts. The shifts are learned from the
    observation's change over 8 steps: the non-view part subtracted, the view as
    1 − cosine.
  - Slope quota: at least one achievement per window, or a moderate penalty.
    The window grows with the agent's progress, and dying charges the remaining
    windows. The health term stays in the score.
  - The manager gets v1.3's DRINK and EAT goals.
  - Why it is layer 1.5 (valuation between the world model and the decision
    layers), not a slower layer on top: lives are too short for a slower
    manager, and nothing but the weights may carry over between lives.

### Added (from `family/next`)
- **Branch `family/next`: the next agent**, re-added on top of `main` from the
  pre-split code (`49a6bc3`, tag `archive/full-2026-11-08`): v1.2's named-goal
  manager on v2's parts.
  - `dreamerv3/agent.py`: the manager over 13 named goals, held until reached
    or 32 steps (`_mgr_act`, `_imagine_hold`, `_mgr_loss_shared`); one shared
    critic with game, goal and explore heads; curiosity as a third reward
    stream; RSSM-2 as a memory with `_map_step`, the one step function that
    acting and training share; taste on the manager; the rule mask (default
    off).
  - `dreamerv3/mapmodel.py` (RSSM-2, trimmed to the memory target) and
    `dreamerv3/taste.py` (verbatim).
  - `dreamerv3/craftax_goals.py`, `craftax_map.py`, `craftax_potential.py` and
    `craftax_valid.py`, verbatim, used as goal and mask libraries.
  - Env options `valid_obs` (`obs['valid']`, the rule flags) and `goals_obs`
    (`obs['goalphi']`, `obs['goalreach']`) in `embodied/envs/craftax.py`.
  - Config keys in `dreamerv3/configs.yaml`: `valid`, `validhead`, `manager`,
    `taste`, `curiosity`, `memhead`, `gphihead`, `mapmodel`, the extra
    `loss_scales`, and `'map/'` in `replay.save_skip`.
  - Tests: `test_craftax_{goals,map,potential,valid}.py`,
    `test_mapmodel_step.py` (the map-delta test is dropped with the map
    target), `test_taste.py` (the action-taste agent hook test is dropped with
    taste over actions).
  - Docs: `design-manager.md` (v1.2 only), `design-v2.md` (RSSM-2 memory and
    curiosity only), `rssm2-step-alignment.md`, `taste-on-manager.md`, a note
    in `audit-2026-10-11.md`; `HANDOFF.md`.
  - `docs/eval_results.json`, `docs/curves/next_lam0.json`, the training
    curves and the comparison report now hold vanilla, `abl_plain` and
    `next_lam0`; `tools/plot_training_curves.py` lists them.
- **The honest-versus-masked statement.** The rule mask (B2: `valid_obs` plus
  `agent.valid.mask`) reads the game's action rules and is counted as cheating,
  as are rule flags used as an input. Masked results are marked as such in the
  README and the report. `next_nomask`, the next agent without the mask, is the
  run that decides whether the agent is good without cheating; it was training
  on 2026-11-08 and is not scored yet.
- From `main` (Craftax adapter, tools, replay):
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

### Changed
- **Config defaults narrowed to what the family used.** `manager.critic` is
  `shared` (was `separate`), `manager.hold` is 32 (was 0), `taste.target` is
  `manager` (was `actions`), `mapmodel.target` is `memory` (was `map`). They
  are the only values this branch accepts; the assertions name the branch that
  holds the others. `manager.goals` accepts only `named`.
- Taste is in the code only so the flags of `next_lam0` and `next_nomask`
  (`taste.lam_mgr 0.0`) still run: at 0 taste adds exactly zero to the
  manager's values and only produces `taste/*` metrics. Taste itself, and
  `next_taste` (`lam_mgr 1`), are studied on `family/taste`.
- `tools/death_eval.py --greedy` help text: it takes the argmax goal too.
- README, results table and report: honest runs first, masked runs marked.
- **Verified against `49a6bc3`** on CPU (seed 0, `size1m`, single-threaded
  XLA): parameter tree and initial values, a 65-step rollout in two Craftax
  worlds with every policy output, probe and greedy acting, parameters and
  metrics after four train steps, and report metrics are bit-identical for
  `plain`, `next_lam0` and `next_nomask`; the `size50m` initial parameter
  trees too. `next_notaste` matches through train step 1 and the metrics of
  steps 1-3, then differs by float rounding (optimizer state 2.9e-4 at step
  2); on this branch it equals `next_lam0` bit for bit outside taste's own
  state, so `lam_mgr 0` adds nothing. Multi-threaded XLA adds compile-
  dependent rounding (up to 0.069 in optimizer state after four steps) that
  vanishes single-threaded. Tests: `pytest dreamerv3/` 83 passed;
  `embodied/tests/test_replay.py` only the 53 known `Replay.dataset`
  failures.

- From `main`:
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
- Relative to `49a6bc3`, moved to other branches (or left in the archive tag):
  - the map target, map labels, `imag_shift`, potential and survival shaping
    (`family/rssm2`, `family/v`);
  - the v1/v1.1 separate-critic manager and learned goals (`goalcodes.py`,
    `family/v`);
  - B1 and B3 (`family/masked`);
  - taste over actions (`family/taste`);
  - the reward-cause head (archive tag only);
  - `curiosity.normalize` / `retlimit` (`family/v`, run `v2cur`);
  - the env's `goals_survival` (v1.3; `family/v`).
- From `main`:
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

### Fixed
- `test_craftax_valid.py::test_agrees_with_the_game` imported its game
  oracle from `tools/action_suppression.py`, which is not on this branch
  (family/masked); the oracle is now a function in the test.
- From `main`:
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

## Family history (from 49a6bc3's CHANGELOG)

What the pre-split changelog recorded for the next agent, oldest first. The
runs were trained on `~/vtmgr` = `49a6bc3`.

- **Taste on the manager** (`agent.taste.target manager`, 2026-11-02/03).
  Taste becomes an internal reward on the manager's choice of goal, added to
  Q(s, g) in `_mgr_loss_shared` and learned from the real return after each
  choice against the manager's expected value. The bottom actor is untouched.
  Code-review fixes before the first run: the evidence uses the fast critic,
  taste enters Q relative to the goals' mean (so a shared level cannot
  handicap NONE), the manager has its own `lam_mgr` (1.0) instead of the
  actions target's `lam` (4.92), only the decision rows are scored (capped by
  `taste.rows`), `q_spread` is logged before taste with `q_spread_taste`
  after, equal good and bad weights (`pos = neg = 1.43`). `taste.center` and
  `Taste.observe(..., steps=)` (sparse evidence counted as a share of all
  steps) were added with tests.
- **The next agent** was defined as B2 mask + curiosity + v1.2 named goals,
  the rest as fixed v2 (RSSM-2 memory, shared critic, hold 32, survival none);
  queued as `next_taste`, then its no-taste control `next_notaste`.
  `next_notaste` crashed on a GPU fault (Xid 69) 90 s in, and again (Xid 13)
  on its first policy step; `~/run_notaste3.sh` was to try once more and, if it
  died, run `next_lam0` (`lam_mgr 0`) as the control.
- **next_taste result** (2026-11-04). Trains to 9.2 achievements per episode
  at 1.08M steps, above fixed v1.2 (8.74) and B2 (8.79); 9.38 ± 0.34 on 60
  test worlds. But taste locked the manager onto `place_table` by 90k steps
  (goal entropy 0 from then on), so the run measures a fixed goal, not taste.
- **next_lam0 result and why taste hurt** (2026-11-07). The no-taste control
  trains to 9.92 against 9.22 and scores 11.30 ± 0.27 on 60 test worlds (11.20
  ± 0.41 on the 30), the best of any run: +1.92 ± 0.41 over `next_taste`, +2.27
  ± 0.33 over fixed v1.2. Taste outweighed the critic 15-300 times from its
  first update, the manager locked onto `place_table`, relative taste then made
  NONE the choice near a table (60% of steps with no goal), and the actor built
  tables instead of stone tools (`docs/taste-on-manager.md`).
- 2026-11-08: the pre-split project was split into `main` (plain DreamerV3) and
  one branch per family; this branch is the next family. `next_nomask` (no
  rule mask) started training on the box.
