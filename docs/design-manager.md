# Two-level agent: a manager over tech-tree goals

**Status:** v1 implemented (`agent.manager.enabled`), first run `mgr` training
on top of B2. Code: `dreamerv3/agent.py` (`_mgr_act`, `_imagine_mgr`,
`_mgr_loss`), `dreamerv3/craftax_goals.py`, `embodied/envs/craftax.py`
(`goals_obs`).

## Why

B2 (honest map + potential + masking, 7.60 achievements) unlocks 3.6
achievements in its first 50 steps and 0.5 per 50 steps after step 200. Its
last unlock comes at a median step of 163, and it lives another 86 steps
without one. Almost everything left after step 100 sits behind the stone
pickaxe, and the iron tier is a plan of 100+ steps: find iron, carry wood,
stone and coal, stand at a table beside a furnace, craft.

The actor learns from 15-step imagined rollouts. Anything beyond 15 steps
reaches it only through the critic, which has to chain ~100 one-step
bootstraps to see iron. Imagining further does not work: 50 steps was tried
and reverted (`configs.yaml`). The fix is to decide in slower time.

## What it is

A manager policy over goals (feudal / Director family). It is not an
explicit planner (it never writes out a sequence of goals) and it does no
search when acting. Like everything in Dreamer, it is a learned network,
trained in imagination and run reactively.

```
RSSM-2 deter2 (every 8 steps) ─┐
RSSM-1 feat + map crop ────────┴─► MANAGER (actor-critic) ─► goal, held 8 steps
                                                               │
RSSM-1 feat + map crop + goal one-hot + segment step one-hot ─► ACTOR ─► action
```

- **Goals** (`craftax_goals.GOALS`, 13): NONE plus the potential's
  tech-tree spine except PLACE_STONE. Each goal is a state to reach, judged
  from the observation alone, e.g. "pickaxe tier >= 2 in the inventory" or
  "a table in the 8-neighbourhood". An existing table counts, so the goal
  never asks for a second table.
- **Manager** (`mgr`, `mval`, `mslowval`): a categorical MLP head over 13
  goals, plus its own critic and slow critic. Input: the actor's view (RSSM-1
  state and the gated map crop) and RSSM-2's `deter2` (stop-gradient). It
  decides at steps 0, 8, 16, ... of each episode, on the RSSM-2 tick that has
  just closed.
- **Actor**: unchanged network. Its input gains the goal one-hot (13) and the
  step within the segment (8). The critic `val` / `slowval` sees the same,
  because a state's value depends on the goal being pursued and the time left.
  `rew` and `con` never see the goal, the same rule as the map crop.

## Rewards

- **Actor**: game reward plus `goal_reward` (1.0) times the change in the
  predicted progress of the goal that was active when the action was taken.
  Progress comes from the `gphi` head, which predicts `obs['goalphi']` from the
  world-model feature, so it exists inside imagination. NONE pays nothing.
- **Manager**: game reward only, summed and discounted over its segment. It
  learns which goal to set. It is never paid for the goal reward itself, so
  it cannot learn to set easy goals for their own sake.

## Training (v1)

- Imagination is a custom scan (`_imagine_mgr`) whose carry holds
  `(deter, stoch, goal, gphase)`. Each rollout opens on a manager decision
  and the manager decides again every 8 steps. The actions and goals the loss
  scores are exactly the ones the rollout sampled, which is the consistency
  `imag_shift` lacked.
- The actor's loss is `imag_loss` unchanged, on the augmented input and reward.
- The manager's loss is `imag_loss` on the abstract trajectory: states
  0, 8, ..., plus the last imagined state for bootstrapping. Segment reward is
  `sum_t prod_{j<t} con_j * rew_t`, and segment continuation is the product of
  `con`. With H = 15 that gives 2 decisions per rollout, and each bootstrap
  step covers 8 real steps.
- The manager's entropy bonus is `manager.actent` (1e-3), not the actor's
  3e-4. It picks among 13 options only every 8 steps and must keep trying
  goals that have not paid yet.
- **The replay value loss is off** in this mode (`_repval`). It would need each
  replay step's goal, segment step and goal reward, plus a bootstrap from an
  imagination started on that same goal and step. v1 imagination always starts
  on a fresh decision, so the two would disagree.
- RSSM-2 has no dynamics of its own in v1. During a rollout it stays at its
  value at the imagination start (as the map crop already did), so the
  manager's two in-rollout decisions read the same slow state.

## Honesty

| | source | verdict |
|---|---|---|
| goal list (13 milestones) | game knowledge, same as the potential's spine | disclosed; the order is never given |
| `goalphi` (progress) | observation vector only (inventory and 8-neighbourhood) | target only, excluded from the encoder; `test_craftax_goals` checks it against the game state |
| achievement flags, true map, game state | not used | |

At test time the agent uses only the observation, as before. The goal
vocabulary is part of the architecture, so results must be compared against
methods that use similar game knowledge, not tabula-rasa ones.

## v1 result, and v1.1

**v1 (`mgr`) failed to train the manager.** At 900k steps its goal entropy
was still the maximum, ln 13 = 2.56: every goal picked about 7.7% of the
time. The actor made progress on its goal on only 1.5% of imagined steps.
Most goals cannot be reached within 8 steps, so a random goal almost never
paid, and the actor learned to ignore it. With the actor ignoring the goal,
the manager's choice changed nothing (advantage about 0.02) and it had nothing
to learn from. The score trailed B2 by about 0.9 at 800k (5.21 vs 6.11).

**v1.1 (`mgr2`)** pays both levels when the goal is reached:

| change | why |
|---|---|
| actor: +`reach_bonus` (1.0) the first time its goal is reached in a segment | a reached goal stands out from noise; "first time" stops step-away-and-back farming |
| manager: +`mgr_bonus` (0.5) the first time in the episode a goal it set is reached | links the payment to its own choice; below an achievement's +1, so reaching easy goals never outbids playing |
| manager input: `obs['goalreach']`, 13 reached-this-episode flags | the bonus is once per episode, so the manager has to know which bonuses are still available |
| goals that already hold are masked from the manager's choice | otherwise the manager learns to propose what is already true, or what the actor would do anyway |

Inside imagination, "reached" means the `gphi` prediction is above
`reach_threshold` (0.6). Progress is exactly 1.0 when a goal is reached and at
most 0.25 otherwise. The flags carried through the rollout start from the
observation-derived ones at the imagination start. `goalreach` is the running
OR of `goalphi == 1` since the episode began, so it is something an observer
of the screen could keep. `test_craftax_goals` checks it.

Watch `train/manager/reach_rate` (actor bonus events per rollout),
`first_reach` (manager bonus events), and `masked_share`. Imagined reach
events are only as honest as `gphi`. If `reach_rate` in imagination runs far
above how often goals are really reached in replay, the actor is exploiting
the progress head.

## v1.1 result: the new best run

`mgr2`, 1.1M steps, scored with `tools/death_eval.py` on the same 30
evaluation worlds as every other run:

| run | achievements | normalized return | furnace | place stone | stone | stone sword | arrow |
|---|---|---|---|---|---|---|---|
| B2 | 7.60 | 3.0% | 40% | 40% | 43% | 0% | 0% |
| v1 | 7.13 | — | 40% | 43% | 47% | 10% | 0% |
| **v1.1** | **8.53** | **3.4%** | **67%** | **67%** | **73%** | **23%** | **10%** |

- **The manager learned to choose.** Its entropy fell from ln 13 = 2.56 to
  1.65, and the goals it set were reached 0.31 times per imagined rollout by
  the end. It settled on the start of the tech tree in order: wood pickaxe,
  table, stone, wood sword, then coal.
- **It overtook B2 late.** v1.1 trailed B2 until ~550k steps, overtook it,
  and finished 0.9 achievements ahead (8.8 vs 7.9 per training episode over
  the last 100k steps).
- **Still missing:** the stone pickaxe (0%; the stone sword takes the
  ingredients), and coal and torches, which B2 got. Thirst is still the first
  meter to run out in 16 of 30 deaths.
- **One seed per run,** so the size of the gap is uncertain. Its direction
  matches the manager metrics.

## v1.2: one two-headed critic, goals held until reached

Built behind flags: `--agent.manager.critic shared --agent.manager.hold 32`.
Defaults keep v1/v1.1.

Two v1.1 problems it addresses:
- **Muddled reward sizes.** v1.1 summed achievements, potential, goal progress,
  the reach bonus and the manager bonus into one return, at sizes nobody had
  chosen on purpose. One stone pickaxe was paid five ways.
- **Once-per-episode manager bonus.** Repeated logistics (stone, then table,
  then craft) earned the manager nothing after the first time.

```
              ┌─► head 'game' (achievements + potential + health) ──► BOTH actors
critic body ──┤
              └─► head 'goal' (progress + reach bonus)              ──► bottom actor only
input: RSSM-1 feat, map crop, goal one-hot, steps-since-set one-hot (32),
       RSSM-2 deter2, reached-this-episode flags
```

- **Bottom actor:** advantage = Â_game + `goal_weight` (0.5) × Â_goal. Each
  stream has its own lambda-return and return normalisation
  (`imag_loss_streams`), so each counts by its weight rather than its raw
  scale.
- **Manager actor:** no critic of its own (`mval` is gone) and no bonus
  (`mgr_bonus` is unused). At imagined states 0 and 8 the game head scores all
  13 goals as if set right now, Q(s, g) = V_game(s, g, step 0), and the
  manager follows the exact gradient sum_g π(g|s) A(s, g). That replaces v1's
  single sampled goal per decision, whose advantage drowned in noise. It is
  judged on the game head only, so it cannot pay itself through the goal
  bonuses.
- **Goal duration:** a goal is held until the observation shows it reached
  (then the next step decides again), or for 32 steps. Reaching it pays the
  bottom actor once, naturally. Iron-tier goals now fit inside one goal.
- **Imagination resumes the replay's goal and step** (`gphase` is stored in
  replay), instead of opening on a fresh decision. The critic therefore sees
  every step value 0..31 that acting produces, and the replay value loss is
  back on, for the game head.
- Masking of goals that already hold, and the reached-this-episode flags as
  input, carry over from v1.1.

New metrics: `game/*` and `goal/*` (per-stream return, advantage, scale),
`manager/q_spread` (how much the game head distinguishes goals; ~0 means the
manager has nothing to learn from), `manager/q_best_minus_mean`, and
`manager/pick/*` counted at real decisions inside imagination.

Known risk: Q for a goal never set in a state is the head's extrapolation. An
over-optimistic one would draw the manager toward it until trying it corrects
the estimate. Watch `q_spread` alongside the pick distribution.

## v1.2 result: the new best run

`mgr3`, 1.1M steps, scored on the same 30 evaluation worlds:

| run | achievements | normalized return | stone pickaxe | coal | wood sword | stone sword | lifespan |
|---|---|---|---|---|---|---|---|
| B2 | 7.60 | 3.0% | 0% | 20% | 20% | 0% | 246 |
| v1.1 | 8.53 | 3.4% | 0% | 0% | 33% | 23% | 259 |
| **v1.2** | **9.73** | **3.9%** | **23%** | 10% | **57%** | **37%** | 221 |

- **The early manager collapse did not sink it.** Its entropy fell to 0.55-0.8
  within 20k steps, and its favourite goal kept moving (table, stone pickaxe,
  diamond, furnace, stone sword). It trailed v1.1 until ~350k steps, then
  pulled ahead. It finished at 9.85 achievements per training episode over the
  last 100k steps (v1.1 8.81, B2 7.91).
- **It favours goals it cannot complete yet:** iron pickaxe, iron sword and
  diamond took 50-60% of its picks around 800k. Their progress counts every
  ingredient (wood, stone, coal, iron, a table and a furnace nearby), so
  under them the bottom actor is paid for stocking up rather than for
  spending. This plausibly explains the first regular stone pickaxe, but it is
  not yet checked in its episodes.
- **It dies earliest of the leading runs** (221 steps). 14 of its 30 deaths
  are mobs or lava and 14 are thirst: it does more, and is more exposed.
  v1.3 adds survival goals and a survival potential on top of v1.2.

## v1.3: survival as well as tech

B2 never does both. Over its last 507 training episodes:
- The ones that live 400+ steps drink 7.3 times and reach 2.5 tech stages.
- The ones that reach 5 tech stages drink 1.8 times and die near step 330,
  mostly of thirst.

Episode length tracks drinking (r = 0.57), not tech (r = 0.12). Nothing in
the reward makes a water trip in the middle of the tech climb worth taking:
Craftax pays for the first drink only, and death lands ~200 steps after the
last drink, far past the 15-step imagination.

v1.3 = v1.2 + two changes (`--env.craftax.survival potential+meters
--env.craftax.goals_survival True`):

- **Survival potential** (`craftax_potential.survival_potential`), added to
  the tech potential:
  Φ_surv = w(s) · (u(food) + u(drink)), with u(m) = 1 − (1 − m/9)² and
  w(s) = w0 · (1 + κ·T + n/10).
  - T is tech progress (Φ_tech / max) and n is the number of achievements
    unlocked; w0 = 0.75 and κ = 1.
  - u is steep when a meter is empty: a 1 → 5 refill pays 3× a 5 → 9 one.
  - w grows with what has been built, so survival is worth more the more is
    at stake. Φ is zeroed on death as before, so a death hands all of it back.
  - A full drink refill is worth ~+0.6 achievement at the start of an episode.
  - It is a function of the state only, so it stays potential-based.
- **Survival goals** DRINK and EAT for the manager (15 goals instead of 13).
  Each is reached when the meter is at 8 or more, read from the observation.
  The manager can now say "drink, then go back to mining".

Success means episodes that both reach 4+ tech stages and drink 4+ times;
B2 has almost none.

## v2 (not built)

- Give RSSM-2 its own action-conditioned dynamics
  (slow state, goal) → next slow state, so imagination can run in slow time:
  15 slow steps ≈ 120 real steps.
- A replay value loss with stored goals and phase-aware imagination.
- A where-goal (map cell, 144-way) alongside the what-goal.

## What to watch

- `train/manager/pick/*`: does the distribution move away from uniform, and
  toward which goals?
- `train/manager/ent/goal`: entropy collapse toward 0 would mean the manager
  has fixed on one goal.
- `train/manager/goal_rew_pos`: how often the actor actually makes progress on
  its goal.
- `train/loss/gphi`: how accurately the latent predicts progress. If this is
  poor, the actor's goal reward is noise.
- The score itself, against B2's curve (same settings except the manager and
  the replay value loss).
