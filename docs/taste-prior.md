# Taste over actions: replacing v2's rule mask

Status: implemented (`dreamerv3/taste.py`, `agent.taste.*`), tests in
`dreamerv3/test_taste.py`; v2-taste training queued.

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
