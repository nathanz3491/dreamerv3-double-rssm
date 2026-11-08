# The manager over named goals (v1.2), as used by the next agent

**Status:** implemented and trained (`next_lam0`, `next_nomask`; see the
README). Code: `dreamerv3/agent.py` (`_mgr_act`, `_imagine_hold`,
`_mgr_loss_shared`, `_goal_stream`), `dreamerv3/craftax_goals.py`,
`embodied/envs/craftax.py` (`goals_obs`). Config: `agent.manager.*`.

This is the v1.2 design from the pre-split project, trimmed to what this
branch runs. The earlier steps (v1, v1.1: a separate manager critic and
once-per-episode manager bonus; v1.3: survival goals and a survival potential;
learned goals in v2) and their results live on `family/v` and in the full
`docs/design-manager.md` at tag `archive/full-2026-11-08`. This branch accepts
only `manager.critic shared` and `manager.goals named`.

## Why

An agent that unlocks the first few achievements quickly then stalls: almost
everything left sits behind the stone pickaxe, and the iron tier is a plan of
100+ steps (find iron, carry wood, stone and coal, stand at a table beside a
furnace, craft). The actor learns from 15-step imagined rollouts, so anything
beyond 15 steps reaches it only through the critic, which has to chain ~100
one-step bootstraps. Imagining further was tried (50 steps) and did not work.
The fix is to decide in slower time.

## What it is

A manager policy over goals (feudal / Director family). It is not a planner
and does no search when acting: like everything in Dreamer it is a learned
network, trained in imagination and run reactively.

```
RSSM-1 feat + RSSM-2 state + reached-so-far flags ─► MANAGER ─► goal, held until reached or 32 steps
                                                                  |
RSSM-1 feat + RSSM-2 state + goal one-hot + steps-since-set ───► ACTOR ─► action
```

- **Goals** (`craftax_goals.GOALS`, 13): NONE plus 12 tech-tree milestones
  (place table, wood pickaxe and sword, collect stone, place furnace, stone
  pickaxe and sword, coal, iron, iron pickaxe and sword, diamond). Each is a
  state to reach, judged from the observation vector alone, e.g. "pickaxe
  tier >= 2 in the inventory" or "a table in the 8-neighbourhood". The manager
  is told which goals exist, never in what order to pursue them.
- **Progress** (`obs['goalphi']`, 13 numbers): the share of a goal's
  prerequisites met, exactly 1.0 when the goal is reached, at most 0.25
  otherwise. It is a training target only: the `gphi` head predicts it from
  the world-model latent, so progress exists inside imagination.
  `obs['goalreach']` (13 flags) is the running OR of `goalphi == 1` since the
  episode began, which an observer of the screen could keep.
- **Manager** (`mgr`): a categorical MLP head over the 13 goals. Input: the
  actor's view (RSSM-1 state plus RSSM-2's state, stop-gradient) and the
  reached-this-episode flags. A goal that already holds right now is blocked
  from the choice (NONE is always allowed), since proposing it teaches
  nothing.
- **Bottom actor**: the stock actor network. Its input gains the goal one-hot
  (13) and the steps since the goal was set (32). The critic sees the same,
  because a state's value depends on the goal and the time left. `rew` and
  `con` never see the goal or RSSM-2.
- **Goal duration.** A goal is held until the observation shows it reached
  (the next step decides again) or for `manager.hold` = 32 steps, so iron-tier
  goals fit inside one goal. Imagination resumes the goal and step the replay
  had (`gphase` is stored in replay), so the critic sees every step value
  0..31 that acting produces and the replay value loss stays on.

## One critic body, two or three heads

```
              +-> head 'game'    (the environment's reward) ---> BOTH actors
critic body --+-> head 'goal'    (progress + reach bonus)   ---> bottom actor only
              +-> head 'explore' (curiosity, if enabled)     ---> both (weights)
```

Why shared: one scalar could not do both jobs. Holding the goal reward, it
would pay the manager for setting easy goals for their own sake; without it,
nothing would tie the bottom actor to the goal. Each stream has its own
lambda-return and its own return, value and advantage normalisation
(`imag_loss_streams`), so each counts by its weight, not by its raw scale.

- **Bottom actor:** advantage = A_game + `goal_weight` (0.5) x A_goal
  (+ `curiosity.weight` x A_explore).
- **Goal stream:** `goal_reward` (1.0) x the change in the predicted progress
  of the goal that was active when the action was taken, plus `reach_bonus`
  (1.0) when it is reached. In imagination "reached" means the `gphi`
  prediction is above `reach_threshold` (0.6). NONE pays nothing.
- **Manager gradient:** at imagined states 0 and 8 the game head scores all 13
  goals as if set right now, Q(s, g) = V_game(s, g, step 0), and the manager
  follows the exact expectation sum_g pi(g|s) A(s, g) with A = Q - sum_g pi Q.
  No sampled goal (whose advantage drowned in noise in v1), no critic of its
  own, and no goal-stream value: it is judged by the game head (plus, with
  curiosity, `mgr_weight` x the explore head), so it cannot pay itself through
  the bottom actor's bonuses. Its entropy bonus is `manager.actent` (1e-3).
- **Taste** (`agent.taste`, `taste.target manager`) adds
  `lam_mgr * rscale * relative_taste(g)` to Q. The next runs here use
  `lam_mgr 0`, which adds exactly zero; see
  [`taste-on-manager.md`](taste-on-manager.md).

## Metrics to watch

`manager/pick/*` (distribution over goals at real decisions inside
imagination), `manager/ent/goal` (entropy; ln 13 = 2.56 is uniform, a collapse
to 0 means one goal), `manager/q_spread` (how much the game head tells goals
apart; about 0 means the manager has nothing to learn from),
`manager/q_best_minus_mean`, `manager/reach_rate` (reach bonuses per imagined
rollout), `manager/goal_rew`, `loss/gphi` (if the progress head is poor, the
goal reward is noise), and `game/*`, `goal/*` (per-stream return, advantage,
scale). Imagined reach events are only as honest as `gphi`: a `reach_rate`
far above how often goals are really reached in replay would mean the actor
exploits the progress head.

Known risk: Q for a goal never set in a state is the head's extrapolation. An
over-optimistic one draws the manager toward it until trying it corrects the
estimate. Read `q_spread` alongside the pick distribution.

## Honesty

| piece | source | verdict |
|---|---|---|
| goal list (12 milestones) | game knowledge: the tech-tree spine | disclosed; the order is never given |
| `goalphi`, `goalreach` | observation vector only (inventory, 8-neighbourhood) | training target / manager input, excluded from the encoder; `test_craftax_goals` checks them against the game state |
| achievement flags, true map, game state | not used | |
| rule mask (`valid.mask`, B2) | the game's action rules | **counted as cheating**; off by default, on in the masked runs only |

At test time the agent reads only the observation. The goal vocabulary is part
of the architecture, so the manager is compared against methods that use
similar game knowledge, not tabula-rasa ones.

## Results of the design (pre-split, 30 fixed worlds)

v1.2 with the mask scored 9.97 on the original code, 8.77 after the RSSM-2
fixes (`mgr3_fix`; 9.03 over 60 worlds, 3 seeds), 6.03 without the mask
(`v12_nomask`). It favours goals it cannot complete yet (iron pickaxe, iron
sword, diamond took 50-60% of its picks): their progress counts every
ingredient, so the bottom actor is paid for stocking up. Its deaths are mostly
mobs, lava and thirst. These runs are on `family/v`. The next agent's runs are
in the README.
