# Design: the market agent (`family/market`)

Status: **being built, not trained yet.** Decided with the user on 2026-11-08.
Points marked *proposed* are implementation choices still open to change. The
[Open questions](#open-questions) section lists what is undecided; where the
code had to choose, the choice is recorded there as "chosen: …".

| part | status | flag (default off) |
|---|---|---|
| §1, §2 Q-heads, fixed P\* | built | `agent.market.enabled` |
| §4 slope quota | built | `env.craftax.quota.enabled` |
| §5 survival goals | built | `env.craftax.goals_survival` |

## The idea in one paragraph

The agent has two goods, **score** (achievements) and **survival**, and one life
to spend on them. A single market sets their exchange rate, the **price P\***:
how much survival one unit of score is worth right now.

- When the world changes (night falls, a mob appears, food runs low), a small
  network shifts the supply and demand curves, and the price moves.
- Every decision level values things at that price: the critic, the actor and
  the manager.
- A **slope quota** makes sure the agent never stops pursuing score: at least
  one achievement per window, or pay a penalty.

None of this uses the game's rules. The mask stays off; it counts as cheating.

## Where it sits

The market is part of a valuation layer, "layer 1.5", between the world model
and the decision layers. It is not a slower layer on top. Agents live too short
a life for a slower manager-of-the-manager, and nothing but the weights may
carry over from one life to the next.

| layer | what it does |
|---|---|
| 1 | world model (RSSM, plus RSSM-2 as memory) |
| **1.5** | **valuation: the market sets P\*; per-action Q-heads price each action at P\*** |
| 2 | actor-critic, valuing at P\* |
| 2.5 | goal → action translation (named goals' progress and reach, as in `next`) |
| 3 | manager: chooses goals, valued at P\* |

## Base agent

`next_nomask` (see `family/next`): next_lam0's flags with
`--agent.valid.mask False`. This branch starts from `family/next`'s code. Its
own design docs (`design-manager.md`, `design-v2.md`, `taste-on-manager.md`)
describe the base. Taste stays at `lam_mgr 0`, where it is inert.

## Components

### 1. Two goods, two value heads

- **Score.** Craftax's own reward, **health term kept**: achievements are worth
  1, 3, 5 or 8 by tier, plus 0.1 × the health change. The quota penalty (§4) is
  added to it. The health term is not split out, because the quota already
  enforces pursuit.
- **Survival.** Time alive. Its reward is 1 per step alive; the continue head
  ends it at death.
- **Value of an action:** Q(s, a) = Q_score(s, a) + P\* · Q_life(s, a).

### 2. Per-action Q-heads

- **Heads.** Two heads, Q_score and Q_life, each with 43 outputs (one per
  action), beside DreamerV3's critic. They are trained on imagined λ-returns of
  the action actually taken.
  - *Proposed:* scalar outputs per action, since DreamerV3's 255-bin output
    times 43 actions is heavy.
- **Economic profit** of action a: Q(s, a) − max over b ≠ a of Q(s, b).
  - It is positive only for the best action, and its size is how far that
    action beats the runner-up.
  - Example: with Q = [10, 2, 1], the profits are [+8, −8, −9].
  - A super-good action therefore has a *low* opportunity cost (its best
    alternative is poor). The runner-up carries the high cost.
- **The critic** gets the 43-dim profit vector as an input, with gradients
  stopped. It is relative, so the critic cannot copy it as its value. What it
  carries is "how much the choice matters here".
- **Built** (`--agent.market.enabled True`, `agent.py`: `_q_loss`, `_profit`,
  `_alive`, `_price`; config `agent.market`, `agent.qhead`, loss `qhead`).
  Here P\* is the constant `agent.market.price0` (0.01); the moving price is
  §3. Chosen, where the design left it open:
  - *Q-head output: scalar, own trunk.* One small MLP (`qhead`: 2 layers, 512
    units) on the actor's input (stop-gradient) with two heads of 43 scalar
    outputs. It does not share the critic's trunk: the critic reads the Q-heads'
    profit, and a shared body would let it copy them. The two heads do share
    this one small trunk. Targets are divided by each head's running return
    scale (`qsnorm`, `qlnorm`, `retnorm`'s settings, floor 1).
  - *Targets.* λ-returns on the imagined rollout, per head: score reward = the
    reward head's prediction (quota included); life reward = 1 per step alive,
    cut by the continue head (`con` rescaled by 1 − 1/horizon, so 1 while
    alive and 0 at death). Each head bootstraps on itself at the next state and
    the next action the dream sampled (SARSA), so Q(s, a) is the return of
    taking a then following the actor. Weighted by the survival of the dream,
    like the critic.
  - *Every consumer values at P\*.* The reward the critic and actor see in
    imagination is `r_score + P* · alive`, and the real reward in the replay
    value loss is `reward + P* · (1 − is_terminal)`. The manager is judged by
    the critic's game head, so it inherits the price with no further change. The
    critic's input gains the 43-dim profit (computed at Q = Q_score + P\* Q_life
    in return units, divided by the game return scale, stop-gradient); the
    manager's all-goals critic input and taste's evidence carry it too (the
    Q-heads read under each goal at step 0).
  - *Why `price0` 0.01.* Survival pays 1 per step against achievements worth
    1 to 8 every few hundred steps. At 0.01, a few hundred steps alive are worth
    about one basic achievement, so staying alive matters but cannot swamp the
    score signal (a price of 1 would make dawdling worth more than any
    achievement). Untuned; sweep it if the Q-heads run is interesting.
  - *One price per rollout:* the value at the rollout's starting state, held
    through the dream (constant here anyway).

### 3. The market

- **Sensing (Δenv).** Every step, compare the observation with the one k = 8
  steps ago, split in two:
  - **Non-view part** (inventory, vitals, light, …): subtracted directly,
    x_t − x_{t−8}.
  - **View** (the local tile grid): one number, 1 − cos(view_t, view_{t−8}).
    A raw difference would mostly measure "I walked", because the view is
    centred on the agent.
  - The split uses the observation's layout (`craftax_map` constants), not the
    game's rules.
- **Curve shift.** A small network takes [Δ non-view, 1 − cos]. It has a linear
  projection in front, and outputs two shifts: Δ log A_supply and
  Δ log A_demand. Each passes through a bounded S-shape (tanh with bound B),
  with a small dead zone so world-model and observation noise do not move the
  price.
- **Curves.** Log-linear (constant elasticity):
  - supply: log Q = log A_s + ε_s · log P
  - demand: log Q = log A_d − ε_d · log P
  - equilibrium: log P\* = (log A_d − log A_s) / (ε_s + ε_d)
  - Baselines A and elasticities ε are constants (*proposed:* swept).
  - P\* stays positive. Q\* is logged as a diagnostic; behaviour changes only
    through P\*.
- **Consumers (all three):**
  - the critic and actor value at Q_score + P\* · Q_life;
  - the manager values goals the same way.
- **Training signal: the achievement slope.** This is the external score, not
  the critic's loss (which would collapse the price into the critic) and not an
  LLM (game knowledge, too slow).
  - *Proposed:* treat the two shifts as a stochastic action of a tiny policy.
    Train it by REINFORCE on real experience, rewarded by achievements over the
    next window, with a running baseline.

### 4. Slope quota

- **Rule.** At least one new achievement per window of N steps; otherwise
  penalty k. Achievements unlock once per episode, so the quota cannot be
  farmed.
- **Size.** Moderate: *proposed* start k = 1 (one basic achievement), sweep
  {0.5, 1, 2}. A large penalty would widen DreamerV3's return scale and drown
  the achievements' own signal, the way taste took over the manager.
- **The window grows with progress.** N(n) = the agent's running mean of steps
  between achievements, at its current count n of unlocked achievements, times a
  margin (*proposed* 2). It is a running statistic like the return normaliser,
  and uses no game knowledge. Late in life a fixed N would demand the
  impossible and push the agent into reckless risks.
- **No escape by dying.** At death, the remaining windows up to a nominal
  lifespan L_nom are charged as failed: penalty k · ceil((L_nom − t) / N).
  Dying never saves penalty.
- **Watch for pacing.** The quota rewards spreading achievements out, so the
  agent may hold back easy ones. Track the time to the first easy achievement
  against `next_nomask`.
- **No soft floor.** The Cobb-Douglas share α was dropped; the quota is the
  only floor.
- **Built:** `--env.craftax.quota.enabled True` (`dreamerv3/quota.py`, pure
  numpy; the env adds the term to `reward`, so it is part of the score good the
  world model predicts; `raw_reward` and `log/reward` stay Craftax's own).
  Group `{enabled, k 1.0, margin 2.0, lnom mean, init 200, rate 0.05,
  min_obs 5}`. Chosen, where the design left it open:
  - *N before statistics exist.* A count n with fewer than `min_obs` recorded
    gaps borrows the window of the nearest lower count that has them, never
    below `init` = 200 steps; with no statistics at all N = `init`. Why: a
    lenient window early (when the agent is random and achievements are rare)
    keeps the penalty from being a constant drip that the return normaliser
    then absorbs; 200 is about one wood-gathering time.
  - *The running mean* is an EMA (rate 0.05, debiased) over the gaps of the
    last ~20 episodes at that count, so the window follows the agent as it
    improves. Only completed gaps count (an episode ending before the next
    achievement adds none); simultaneous unlocks record one gap.
  - *L_nom* defaults to the running mean lifespan (the proposal; an old agent
    can then escape the charge by living short, which is the stated risk);
    `lnom 3000` gives a fixed one. With `lnom mean` no death is charged until
    the first episode has ended. Timeouts (`is_last` without terminal) are
    never charged.
  - *Pacing.* The window restarts after every penalty, so a quiet agent pays
    once per N steps, not every step.
  - The statistics live in the env process and are not checkpointed; a resumed
    run relearns them in a few episodes.

### 5. Survival goals for the manager

- v1.3's survival goals, **DRINK and EAT**, are added to the 13 named goals.
  v1.3 had no SLEEP goal.
- The code is on `family/v` (`goals_survival`) and must be ported here;
  `family/next` left it out.
  - **Built:** `--env.craftax.goals_survival True` (with `goals_obs True`).
    `craftax_goals.py` already carried the two goals; the env flag and the
    agent's goal names (13 or 15, read off `obs['goalphi']`) are ported.
    A goal reads as reached when its meter is at least 8 of 9; below that its
    progress is 0.25 × meter / 8.
- When P\* is high, the manager's goal values favour these goals. Without them,
  it could only choose "no goal".
- Like the named goals, they are hand-written: game knowledge accepted for now,
  as in `next`.

## How it will be judged

- **Headline.** Achievements on the 30 fixed test worlds
  (`tools/death_eval.py`) against `next_nomask`, the honest base. Noise is
  ±0.55 per score, so gaps under about 1.1 between single runs are not
  evidence.
- **Ablation order** (*proposed*), each one run, about 16 h:
  1. **Quota only.** Does the floor alone help or hurt?
  2. **Q-heads plus a fixed P\*** (no shifts). Does valuing survival help?
  3. **Full market.** Does a *moving* price beat a fixed one?
- **Adaptation.** A moving price should pay off most when the world changes in
  the middle of a life. The fixed test worlds may not show it (see open
  questions).

## Open questions

1. **Market-net training.** Is REINFORCE on real experience enough, or does it
   need imagination? What window and baseline does its reward use?
2. **L_nom** for the death charge (chosen: the running mean, §4):
   - a fixed number;
   - or the agent's running mean lifespan, which lets old agents escape
     (proposed).
3. **Constants.** The baselines A_s and A_d, elasticities ε_s and ε_d, tanh
   bound B, and dead-zone width.
4. **Q-head output:** scalar or distributional? Shared trunk with the critic?
   Chosen: scalar, own trunk (§2).
5. **Mid-life change test:**
   - Craftax's own changes (night, going down a floor);
   - or injected ones (food made scarce partway through a life), which need a
     new test harness.
6. **Seeds and budget** at about 16 h per run.
