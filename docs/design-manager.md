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
