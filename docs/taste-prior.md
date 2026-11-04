# Taste over actions: replacing v2's rule mask

Status: implemented (`dreamerv3/taste.py`, `agent.taste.*`), tests in
`dreamerv3/test_taste.py`. v2-taste trained and scored (2026-11-01): it did
not replace the rule mask. See [Result](#result-2026-11-01).

## Why

v2 still uses the game's rules: `valid.mask True` blocks every action that
`craftax_valid` says is impossible. That is game knowledge, so v2's scores
cannot be compared fairly with knowledge-free agents. The learned
replacement, B3, failed. Its mask could not tell "this action does nothing"
from "not learned yet", it blocked 55% of the valid special actions, and it
starved the world model (`docs/entropy-and-action-suppression.md`).

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
| v2 fixed (3 seeds) | 7.02 (7.17 / 7.83 / 6.07) | 2.71% | 53% (seed 0) |
| honest map fixed (no mask, no taste) | 5.70 ± 0.21 | 2.12% | 7% |
| **v2-taste** | **5.33 ± 0.19** | **1.96%** | **0%** |
| vanilla | 4.63 ± 0.22 | 1.65% | 23% |

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

**No-taste control.** v2-norules (the same agent without taste) is still
training. At matched steps up to 523k, v2-taste is 0.0 to 0.5 ahead. That is
one seed each, within the seed-to-seed spread (up to 1.8).

**Next.** The design the proposal intended:
- keep the rule mask and put taste on top, with the same agent without taste
  as the control;
- compare the evidence with the average step;
- let taste weigh observable events, such as the hand-written goals' reach
  bonuses, instead of buttons.

## Taste on the manager (next run, `agent.taste.target manager`)

**The agent** is the next agent with taste added:
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

**Runs** (`~/run_tmgrq.sh` on the box, after v2-norules): `next_taste`,
then `next_notaste`, the same agent without taste. The control is the
proposal's "fixed internal reward": the manager judged by game value alone.

Metrics:
- `taste/<goal>`: the shown taste per goal, logged from the manager loss;
- the evidence fold's own: `adv_std`, `shown_mean`, `liked`, `conf_mean`,
  and `choices` (decisions per batch).
