# v2: the two-level agent with almost no game knowledge

**Status:** training since 2026-10-07 (`~/logdir/v2`, 1.1M steps) after a
CPU smoke test of this exact configuration. Diagram: [`architecture-v2.svg`](architecture-v2.svg).

## Why

v1.2 (9.73 achievements) uses game knowledge in several places:
- **At test time:** action-masking rules, the hand-written goal list, the
  goal-reached checks, and hand-computed movement vectors fed to RSSM-2.
- **In training:** the tech-tree and survival potentials, and the map labels.

The published 1M-step agents it would be compared with (ITC 7.09%, Simulus
6.59%, DreamerV3) use none. v2 replaces three of the four pieces with learned
parts. It **keeps B2's rule-based action mask** (decided 2026-10-07). The
learned mask failed twice: B3 scored 5.60 against B2's 7.60, and B3-fix
blocked crafting actions whose effect the world model had not yet learned
([`entropy-and-action-suppression.md`](entropy-and-action-suppression.md)).
So v2 is close to, but not strictly, like-for-like with the published agents;
quote it with that caveat. The honest baseline for v2 is vanilla DreamerV3
(4.44 achievements, ~2%), not v1.2.

## The four pieces

| knowledge in v1.x | v2 replacement | code |
|---|---|---|
| action rules (`craftax_valid`) | **kept**: B2's rule mask while acting, a head trained on the rules in imagination | `valid.mask True`; env `valid_obs True` |
| goal list, reached-checks, progress formula (`craftax_goals`) | **learned goal codebook**: 16 types of latent change | `manager.goals learned`, `goalcodes.py` |
| tech-tree and survival potentials | **curiosity**: ensemble disagreement as a third reward stream | `curiosity.enabled True`; env `survival none` |
| map labels (block groupings, movement rules, spawn), movement vectors | **RSSM-2 memory**: raw actions in; predict the future and recall the past | `mapmodel.target memory` |

### Learned goals (`goalcodes.py`)
- **Codebook training:** Δ = feat(t+8) − feat(t) from replay, within one
  episode, with gradients stopped. An encoder (5120 → 256 → 64 numbers on the
  unit sphere) and 16 codes are trained as in a VQ-VAE: the decoder
  reconstructs Δ from the quantised code, plus a commitment term. The codes
  move by moving average of their assigned encodings, and dead codes are moved
  onto random recent changes. The codes live in `goalbook`, outside the
  optimizer, like the return normalisers.
- **Manager:** picks one of the 16 codes. The machinery is v1.2's: shared
  critic, exact gradient over all goals, goals held until reached or 32 steps.
- **Progress:** the cosine between encode(feat_now − feat_at_goal_set) and the
  goal's code. The goal reward is the change in progress plus +1 on reaching.
- **Reached** means the change since the goal was set is *classified* as the
  goal's code (its nearest code, with cosine above the 0.3 floor). A fixed
  cosine threshold of 0.7 was tried first. Real changes matched their nearest
  code at only ~0.38, so goals were almost never reached (0.0001 per rollout);
  classification gave 0.16-0.25.
- **Dropped:** masking of goals that already hold, and the reached-this-episode
  flags (passed as zeros). A learned change-type is never "already true".

### Curiosity
Five MLP heads each predict the next posterior stoch (probabilities, 1024
numbers) from (feat, action taken). They are trained on replay with gradients
stopped. In imagination, the variance of their predictions, averaged over
dimensions, is the curiosity reward for that step. It feeds a third critic
head, `explore`, normalised separately:
- the bottom actor learns from game + 0.5 × goal + 0.2 × explore;
- the manager's goal values are game + 0.2 × explore.

**Curiosity was nearly off in the first v2 run.** Each stream's return is
divided by its spread (95th minus 5th percentile), but never by less than 1.
That floor suits the sparse game reward. Curiosity returns span only ~0.02, so
they were never scaled up. At 313k steps the actor's normalised advantages
were game 0.017, goal 0.028 and explore 0.0027. After the weights, curiosity
was ~2% of the signal. The manager added the raw explore value, which was
just as faint.

**v2-cur** (queued after v2; the curiosity changes only):
`--agent.curiosity.normalize True --agent.curiosity.weight 0.1
--agent.curiosity.mgr_weight 0.1`.
- The explore stream gets its own floor (`retlimit`, 1e-3), so it is divided
  by its real spread.
- The manager's explore value is put in the same units.
- The weight drops to 0.1: at 0.2, curiosity would push about as hard as game
  and goal together; at 0.1 it is about a fifth of the signal.

### RSSM-2 memory
- **Input per 8-step tick:** RSSM-1's mean latent, plus the 43-way action
  one-hots summed over the window (raw actions), plus the step count.
- **Targets:**
  - predict RSSM-1's mean latent 1, 2 and 4 ticks ahead (8/16/32 steps; MSE);
  - reconstruct the observation vector 1, 2 and 4 ticks back (MSE);
  - both only within one episode.
- **Into the actor:** RSSM-2's state (1024, gradient-stopped) times the learned
  gate replaces the map crop.
- **Removed:** map targets, position tracking and the egocentric crop.

## Running it

```bash
python dreamerv3/main.py --logdir ~/logdir/v2 --configs craftax size50m \
  --env.craftax.survival none --env.craftax.valid_obs True \
  --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
  --agent.mapmodel.target memory \
  --agent.valid.mask True \
  --agent.manager.enabled True --agent.manager.critic shared --agent.manager.hold 32 \
  --agent.manager.goals learned --agent.curiosity.enabled True
```

`valid_obs True` gives the env's rule-based action flags, which the mask
reads while acting, exactly as in B2.

**Ablation, honest map instead of memory:** replace
`--agent.mapmodel.target memory` with `--agent.mapmodel.target map
--agent.mapmodel.hindsight True --env.craftax.mapmodel True`. The map is built
from the agent's own observations, but with hand-written grouping and
movement rules.

## What to watch
- `goals/perplexity`: how many of the 16 codes are in use. It was 7.7-11 early
  in the debug run.
- `goals/match`: how well changes fit their nearest code.
- `manager/reach_rate`, `manager/ent/goal`: whether the manager learns to
  choose. In v1, uniform entropy (ln 16 = 2.77) for the whole run meant
  failure.
- `memory/future`, `memory/recall`, `memory/gate`: whether RSSM-2 learns, and
  whether the actor opens the gate to read it.
- `curiosity/reward`: should fall where the model has learned and rise in new
  places.

## Expectation
Below v1.2 at first, since the goal list and potentials were worth a lot. Success means beating
vanilla DreamerV3 clearly; reaching ITC's 7.09% is the stretch goal.
