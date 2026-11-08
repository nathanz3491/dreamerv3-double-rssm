# Taste: a learned prior over actions, then over the manager's goals

Status: implemented (`dreamerv3/taste.py`, `agent.taste.*`), tests in
`dreamerv3/test_taste.py`. Three runs, all scored on the 30 fixed worlds:

| run | taste on | rule mask | result |
|---|---|---|---|
| `v2_taste` | actions, v2 without the mask | no | 5.33: did not replace the mask ([Result](#result-2026-11-01)) |
| `abl_taste` | actions, plain DreamerV3 | no | 4.63: no effect ([abl_taste](#taste-over-actions-on-plain-dreamerv3-abl_taste-2026-11-03)) |
| `next_taste` | the manager's goals | **yes — masked, counted as cheating** | 9.80: locked the manager on one goal and cost −1.92 against its no-taste control ([Why taste hurt](#why-taste-hurt-2026-11-07)) |

The rule mask reads the game's action rules, so a masked run is not
comparable with knowledge-free agents; masked runs are marked wherever they
appear below. The runs and docs of the other families named here (v2, v1.2,
B2, `next_lam0`) live on their own branches (`family/v`, `family/masked`,
`family/next`); the pre-split code and results are at tag
`archive/full-2026-11-08`.

## Why

v2 still uses the game's rules: `valid.mask True` blocks every action that
`craftax_valid` says is impossible. That is game knowledge, so v2's scores
cannot be compared fairly with knowledge-free agents. The learned
replacement, B3, failed. Its mask could not tell "this action does nothing"
from "not learned yet", it blocked 55% of the valid special actions, and it
starved the world model (`docs/entropy-and-action-suppression.md` on
`family/masked`).

The plastic-reward toy (`../plastic-reward`, `docs/results-buttons.md` there)
tested a different idea on the same problem: a *taste* per action, revised
from experience. It reached B3's failure in miniature, then fixed it:

| version | lifespan vs no taste (11 worlds × 16 paired lives) |
|---|---|
| taste as an intrinsic **reward** | −10 to −81 (farmed, then trapped) |
| taste as a **logit prior**, advantage evidence, evolved rule | **+11 to +37 in every world** |
| same rule, evidence shuffled (attitude only) | −11 to +19 |

## What carried over

1. **A logit bias, never a reward.** A reward on an action that can be taken
   at any time can be farmed. A bias on the actor's logits cannot. The bias is
   added in `Agent._masked`, so acting, the imagined rollouts and the actor
   loss all see the same prior.
2. **The critic's advantage as evidence.** For each real step in the replay
   value loss, the evidence is the return that really followed minus the
   critic's value there, filed under the action taken. Each state is its own
   baseline, so an action is not credited for the situations it tends to be
   taken in.
3. **The evolved rule** (`btn2adv`), transplanted unit for unit:
   - birth taste +0.25;
   - good evidence weighted 3.1× bad (4.45 vs 1.43);
   - 40% of the old evidence kept per fold;
   - learning speed 0.41;
   - loudness 4.92 logits per unit of taste.
4. **Confidence fades with evidence.** Until an action has evidence, the
   policy sees `optimism` (0), not the taste. A rarely taken action therefore
   starts neutral rather than being blocked.

The **time**, **count** and **scale** are matched to the toy as described in
the `taste.py` docstring:
- one fold every 2000 train updates (4000 new env steps, about 275 folds per
  run);
- counts re-spread over the fresh env steps;
- evidence in units of the per-step advantage spread, × 0.288 (the toy's 9.2
  / 32).

`actions: special` leaves NOOP, the moves and DO untouched, as the toy's
taste covered its buttons but not its moves. That uses the order of the action
list, not the game's rules.

## Not carried over

- **Evolution itself.** The rule's constants come from the toy. There is no
  evolution over DreamerV3 runs, which would cost about 15 GPU-hours per
  genome.
- **The toy's lopsided "once bitten" rule.** It belonged to taste as a
  reward. As a prior, evolution reversed it: good news counts more, because
  the actor's own reward already teaches it to avoid harm.

## Open question

RL already learns which actions pay. Is a second, fast, context-free learner
redundant? The toy says no (on top of PPO it added +11 to +37). Craftax
decides here, with two runs:
- **v2-taste:** v2 flags, `valid_obs False`, `valid.mask False`,
  `taste.enabled True`;
- **v2-norules:** the same without taste, the control.

Both are compared on the 30 fixed evaluation worlds with v2 (rules) and
vanilla.

Metrics are logged under `taste/`:
- `adv_std`;
- `shown_mean`, `shown_min`, `shown_max`;
- `liked` (share of special actions with a positive shown taste);
- `conf_mean`.

## Result (2026-11-01)

v2-taste trained for 1.1M env steps (one seed) and was scored on the same 30
fixed worlds as every other run (`docs/comparison-report.pdf`).

| run | test achievements | normalized return | wood pickaxe |
|---|---|---|---|
| **v2-taste** (no mask) | **5.33 ± 0.19** | **1.96%** | **0%** |
| honest map fixed (no mask, no taste; `family/rssm2`) | 5.70 ± 0.21 | 2.12% | 7% |
| vanilla | 4.63 ± 0.22 | 1.65% | 23% |
| v2 fixed (3 seeds; **masked, counted as cheating**; `family/v`) | 7.02 (7.17 / 7.83 / 6.07) | 2.71% | 53% (seed 0) |

Paired by world: −1.69 ± 0.25 against v2, −0.37 ± 0.26 against the honest
map, +0.70 ± 0.28 against vanilla. In training it stayed about 2 below v2
fixed at every step (5.12 against 7.36 at 1.08M).

**What taste learned** (`docs/taste-run.json`: the `taste/` metrics over
training, and each action's taste in the final checkpoint):
- It started curious and liked every special action at 100k steps.
- It then turned against almost all of them. From 250k steps on, at most 2
  of the 37 had a positive bias, both almost never pressed. The mean shown
  taste ended at −0.53, or −2.6 logits.
- The bias follows how often an action was pressed (rank correlation −0.98
  between evidence count and bias). Placing a table, which v2 needs, ended at
  −4.5 logits. Making an iron sword, which is impossible this early, ended at
  the same value.

**Why:**
1. *The evidence was not compared with a baseline.* The advantage
   (λ-return minus the critic's value) averaged below zero for every action.
   Each press therefore counted a little against the button pressed, and the
   most-pressed buttons collected the most of it.
2. *Taste ignores the situation.* Crafting pays only next to a table with
   the right items. A single number per button cannot say "good here".
3. *It replaced the mask instead of sitting on top of it.* The run therefore
   tested whether taste can stand in for the game's rules, a question the
   research proposal never asked. It confounds everything else.

**No-taste control.** v2-norules (the same agent without taste, `family/v`)
was never scored on the test worlds; its training curve is in
`docs/curves/v2_norules.json`. In training the two are level within one seed
each: v2-taste was 0.0 to 0.5 ahead at matched steps up to 523k, inside the
seed-to-seed spread (up to 1.8).

**Next.** The design the proposal intended:
- keep the rule mask and put taste on top, with the same agent without taste
  as the control;
- compare the evidence with the average step;
- let taste weigh observable events, such as the hand-written goals' reach
  bonuses, instead of buttons.

## Taste over actions on plain DreamerV3 (`abl_taste`, 2026-11-03)

The fixes above, tested where nothing else is in the way: plain DreamerV3
(no RSSM-2, no manager, no mask, no shaping) plus taste over actions with
**centered** evidence, `pos = neg = 1.43` (see Settings below for why) and
`w0 0` (birth taste neutral). One seed.

| run | test achievements | return | vs plain (paired) | training, last 100k steps |
|---|---|---|---|---|
| plain (`abl_plain`, main) | 4.77 ± 0.28 | 1.71% | — | 5.43 |
| **+ taste over actions** | **4.63 ± 0.29** | **1.65%** | **−0.13 ± 0.27** | **5.19** |

No effect in either measure. A 30-world score moves by about ±0.55 between
world sets, so a gap this size between single runs is not evidence either
way.

## Taste on the manager (`next_taste`, `agent.taste.target manager`)

**The agent** is the next agent with taste added (**masked, counted as
cheating**: it keeps the rule mask):
- B2's rule mask;
- curiosity as in v2;
- v1.2's 13 hand-written goals;
- everything else as fixed v2: RSSM-2 as a memory, the shared critic, hold
  32, survival none.

Taste acts **only on the manager**. The bottom actor, its goal reward and
the reach bonus are untouched.

**Where it writes.** `_mgr_loss_shared` already scores every goal g at a
manager decision, using Q(s, g) from the critic's game head. Taste adds an
internal reward for choosing g:

    Q'(s, g) = Q(s, g) + lam_mgr x rscale x (taste(g) − mean taste of the 12 goals)

This is the proposal's r_learn = r_env + λ·r_int, with "the manager chose
goal g" as the event. Details:
- rscale is the game retnorm scale, floored at 1. At 1M steps it was 1.0
  against a Q spread of about 0.1, so lam_mgr is a gain on that scale, not
  "one return spread".
- The manager's advantage ignores any amount added equally to every goal.
  Only taste *relative* to the other goals can move it.
- NONE (goal 0) never gets a taste. Without the mean subtraction, a level
  shared by the 12 goals (optimism, drift) would act as a permanent handicap
  on NONE only. Code review found that optimism 0.25 alone gave a 0.23
  offset, twice the game's Q spread.
- No bias is added anywhere: the manager's choices change only as it learns
  from Q'.
- `q_spread` is logged before taste is added and `q_spread_taste` after, so
  a run can tell the two apart.

**Where it reads** (`_taste_mgr_evidence`, in the replay block). At every
real step where the manager chose a goal (gphase 0), the evidence is:
- the λ-return that really followed;
- minus Σ_g π(g|s) Q(s, g), what the manager expected from its own choice
  there.

Only the game head counts: the proposal judges by external reward alone, so
curiosity and the goal stream stay out. The evidence is filed under the goal
chosen.

Q here comes from the *fast* critic. That is the one the manager's loss uses
and the one that bootstraps the λ-return (`repl_loss slowtar False`). The
slow critic lags by about 50 updates, and that lag would add a goal-dependent
error which centering cannot remove.

Only the decision rows (about 3% of replay) are scored, at most `taste.rows`
(256) per update. Scoring every row for all 13 goals would mean 13,000 critic
rows per update, of which about 400 are used. Any dropped choices are
counted in `choices_dropped`.

**The baseline fix.** `center True` subtracts the fold's average over all
choices from each goal's evidence. A constant error in the critic then
cannot drag every goal down, which is what sank v2-taste.

**Counts.** Choices happen at only a few steps, so `observe(..., steps=)`
gives each goal its share of all the fresh env steps, not of the choices.
Otherwise a fold of about 300 choices would count as 4000 pieces of
evidence. These are replay samples, so n is a rate rather than a count of
distinct choices. Early folds rest on few distinct decisions replayed many
times, a property inherited from v2-taste.

The step counter exists only for the manager target. Taste over actions
keeps its old variables, so v2-taste checkpoints still load.

**Settings.** The update rule keeps the toy's evolved `w0`, `alpha`, `rho`,
`decay` and `n0`. Three settings differ from the toy:
- **`pos = neg = 1.43`.** The toy weighted good evidence 3.1 times bad. With
  centered evidence that turns taste into a sign detector: a goal slightly
  above average pins at +1. In a simulation by the reviewer, every good goal
  ended at 1.00 and every bad one at about 0. Equal weights make taste a
  graded comparison with the average goal, which is what centering means.
- **`lam_mgr 1.0`**, a separate key from the actions target's `lam` (4.92
  logits), so a launch that forgets the flag cannot apply a reward 4.92 times
  too large.
- **`optimism 0`.** A level shared by every unseen goal cancels out of the
  manager's advantage, so optimism cannot make new goals "slightly
  favoured". The proposal's "curious at first" is left to v2's curiosity
  stream, which the manager also values.

**Runs** (`~/run_ablq.sh` on the box): `next_taste`, then `next_notaste`,
the same agent without taste. The control is the proposal's "fixed internal
reward": the manager judged by game value alone. `next_notaste` crashed on
GPU faults (Xid 69/13) three times; the control that finished is
`next_lam0`, the same agent with taste computed but `lam_mgr 0`, so it never
acts (it lives on `family/next`; on this branch it runs with the
`next_lam0` flags in `README.md`).

Metrics:
- `taste/<goal>`: the shown taste per goal, logged from the manager loss;
- the evidence fold's own: `adv_std`, `shown_mean`, `liked`, `conf_mean`,
  and `choices` (decisions per batch).

## Result: next_taste (2026-11-04)

Every run in this section uses the rule mask: **masked, counted as
cheating**.

**Training is the best of the fixed-code runs.** Achievements per training
episode, one seed, against three-seed means:

| env steps | next-taste | v1.2 fixed | B2 fixed | v2 fixed |
|---|---|---|---|---|
| 700k | 8.10 | 8.22 | 7.57 | 6.70 |
| 800k | 8.99 | 8.26 | 8.13 | 6.91 |
| 900k | 8.79 | 8.53 | 8.56 | 7.22 |
| 1000k | 9.21 | 8.67 | 8.63 | 7.31 |
| 1080k | 9.22 | 8.74 | 8.79 | 7.36 |

**Test worlds** (`tools/death_eval.py`, 60 worlds, seeds 0–59, paired gaps
by world):

| run | worlds 0–29 | worlds 30–59 | 60 worlds | next_taste − run |
|---|---|---|---|---|
| next_taste | 9.80 | 8.97 | 9.38 ± 0.34 | — |
| v1.2 fixed (3 seeds) | 8.78 | 9.29 | 9.03 ± 0.22 | +0.35 ± 0.35 |
| B2 fixed (3 seeds) | 8.33 | 8.73 | 8.53 ± 0.24 | +0.85 ± 0.39 |
| v2 fixed (3 seeds) | 7.02 | — | — | +2.78 ± 0.48 (worlds 0–29) |

Level with v1.2, likely above B2, far above v2. It never makes a stone
pickaxe or sword and never collects iron on any of the 60 worlds; its score
comes from the wood tier plus placing stone and furnaces, arrows and coal.

**But taste did not do what it was designed to do. It locked the manager
onto one goal by 90k steps:**
- `place_table`'s relative taste rose to 0.66 by 90k and 0.82 by 610k. Every
  other goal sat at −0.075, unchanged after that.
- The manager's goal entropy was 0.0 from 90k to the end.
- The critic barely separates goals (`q_spread` 0.05–0.13). Taste, scaled by
  `rscale` (about 7.6), separates them by about 4 (`q_spread_taste`). Taste
  outweighs the critic 40 to 80 times over.

**Why it locks in:** evidence comes only from goals the manager chooses.
Once one goal wins, the others get no evidence and keep their taste. The
winner's own evidence is centered over the fold, and the fold holds only
the winner, so its centered evidence is about 0 and its taste stops moving.
Nothing pulls it back.

**What this means for the score:** next_taste is effectively "the next
agent with the manager always saying `place_table`". Its 9.2 says a fixed
goal costs nothing on top of the mask; it does not show that taste helps.
The no-taste control (`next_lam0`, the same agent with a learned manager)
answered whether the manager's choice matters: it does (next section).

Possible fixes, if taste on the manager is kept:
- scale taste by the critic's own goal spread, not by `rscale`;
- center over the goals, not over the fold's choices;
- keep some exploration of goals, so every goal keeps getting evidence.

## Why taste hurt (2026-11-07)

The comparison is with `next_lam0`, the no-taste control: the same agent,
the same flags, `--agent.taste.lam_mgr 0.0`. Taste is then computed and
logged but adds nothing to the manager's Q (taste has no RNG and is not in
the acting path), so next_lam0 is the next agent without taste. The run
lives on `family/next`; it is cited here. Both runs use the rule mask:
**masked, counted as cheating**.

**Test worlds** (60 worlds, paired by world): next_lam0 scores
**11.30 ± 0.27**, the best of any run (old v1.2: 9.73 over the same 60).

| next_lam0 − run | gap |
|---|---|
| next-taste | +1.92 ± 0.41 |
| v1.2 fixed (3 seeds) | +2.27 ± 0.33 |
| B2 fixed (3 seeds) | +2.77 ± 0.32 |
| v2 fixed (3 seeds, worlds 0–29) | +4.18 ± 0.49 |

On worlds 0–29 it makes the stone sword in 50% of episodes and the stone
pickaxe in 30%; next-taste made neither in any of 60. One seed.

The control, `next_lam0` (the same agent with taste on the manager but
`lam_mgr 0`, so taste is computed and never used), trains to 9.92 at 1.08M
steps against next-taste's 9.22, and leads at every checkpoint from 700k.
Taste hurt through four links, each measured.

**1. Taste swamps the critic from its first update.** The manager's
advantage is `(Q − Σ π Q) / rscale`. The critic barely separates goals:
`q_spread` 0.02–0.1, so about 0.003–0.013 after dividing by `rscale`
(about 7.6). Taste adds `lam_mgr × relative taste` directly: 0.2 after the
first fold, 0.9 by the end. That is 15 to 300 times the game signal, against
an entropy bonus of only `actent` 1e-3. Goal entropy fell from 2.55 to 0.004
at the first fold, **6.7k env steps** into training.

**2. The lock is permanent.** Only a chosen goal gets evidence. The fold
holds only the winner, so its centered evidence is about 0. The others keep
their taste. Nothing can move it back (`place_table` 0.20 → 0.82; the rest
frozen at −0.075).

**3. "Relative" taste turns NONE into the second choice.** A goal that
already holds is blocked from the manager's choice, and `PLACE_TABLE` holds
whenever a table is nearby. NONE has no taste (0), and every unfavoured goal
sits at −0.075, so whenever the agent is near a table the manager picks NONE.
Over the last 160k steps of replay:

| goal share | next-taste | next_lam0 |
|---|---|---|
| NONE (no goal) | 60% | 27% |
| PLACE_TABLE | 40% | 10% |
| the 11 other goals | 0% | 63% (no goal above 11%) |

**4. Being told "be near a table", the actor builds tables instead of
tools.** The goal reward pays for reaching the goal, and placing a table
reaches it at once. Presses per episode (replay, last 160k steps):

| press | next-taste | next_lam0 |
|---|---|---|
| place table | 4.63 (4.21 under the table goal) | 3.53 |
| DO (chop, mine, hit) | 87.7 | 58.7 |
| place stone | 1.10 | 2.39 |
| place furnace | 1.11 | 1.53 |
| make stone pickaxe | 0.00 | 0.06 |
| make stone sword | 0.00 | 0.32 |

Over the last 150k steps (`tools/diagnose_run.py` on `family/masked`),
next-taste collects
stone in 69% of episodes (next_lam0 82%) and places a furnace in 61% (78%).
Of the episodes that held wood and stone next to a table, it made the stone
pickaxe in 0.3% (8.2%). The goals that point past the table (stone, iron,
the stone tools) are never proposed. The control's manager proposes all
eleven, and that is where its lead comes from.

**What this says about the design:**
- Taste must be on the critic's scale, not `rscale`'s: for example, in
  units of the critic's own spread between goals, or capped below it.
- Evidence must reach unchosen goals, or taste must decay back toward
  neutral; otherwise the first winner is permanent.
- NONE must be treated like the other goals (or excluded from the choice),
  so that centering does not make "no goal" the default.
- Not measured, but likely: easy goals win early. A rule that judges goals
  by the return right after choosing them favours whatever is achievable
  first, while the tech tree needs goals whose payoff comes later.
