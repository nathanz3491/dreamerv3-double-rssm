# v2 parts used by the next agent: RSSM-2 as a memory, and curiosity

**Status:** implemented and trained as part of `next_lam0` and `next_nomask`.
This is the pre-split `docs/design-v2.md`, trimmed to what this branch runs.
v2's learned goal codebook (`goalcodes.py`, `manager.goals learned`) lives on
`family/v`; the next agent uses v1.2's 13 named goals instead
([`design-manager.md`](design-manager.md)). The full text, with v2's results
(6.90 achievements on the original code, 7.02 as a three-seed mean on the
fixed code) is at tag `archive/full-2026-11-08`.

## Why

v1.2's manager used game knowledge in several places: action-masking rules,
the hand-written goal list, hand-computed movement vectors fed to RSSM-2, and
(in training) tech-tree and survival potentials and map labels. The published
1M-step agents it would be compared with use none. v2 replaced the map labels
and the potentials with learned parts:

| knowledge in v1.x | replacement | code |
|---|---|---|
| map labels (block groupings, movement rules, spawn), movement vectors | **RSSM-2 memory**: raw actions in; predict the future and recall the past | `mapmodel.target memory` |
| tech-tree and survival potentials | **curiosity**: ensemble disagreement as a third reward stream | `curiosity.enabled True`; env `survival none` |
| action rules (`craftax_valid`) | kept in v2 as the B2 rule mask; **off by default here and counted as cheating** | `valid.mask` |
| goal list | v2 learned it; the next agent keeps the hand-written list | `manager.goals named` |

## RSSM-2 as a memory (`mapmodel.py`)

A slow GRU that ticks once per 8 env steps and is trained with no game labels.

- **Input per tick:** RSSM-1's mean latent over the window, the 43-way action
  one-hots summed over the window (raw actions), and the step count.
- **Targets:** predict RSSM-1's mean latent 1, 2 and 4 ticks ahead (8/16/32
  steps; MSE; `memfut`) and reconstruct the observation vector 1, 2 and 4
  ticks back (MSE; `memrec`), both only within one episode.
- **Into the actor:** RSSM-2's state (1024, gradient-stopped) times a learned
  scalar gate (`mapmodel.to_actor`) is appended to the actor, critic and
  manager input. Only the gate learns from the actor.
- **Train/act alignment:** one step function, `Agent._map_step`, is used by
  acting and scanned over the batch in training, so the actor never trains on
  a state it could not see while acting
  ([`rssm2-step-alignment.md`](rssm2-step-alignment.md)). RSSM-2 does not tick
  inside a dream: the actor reads the state it had at the imagination start.
- Metrics: `memory/future`, `memory/recall`, `memory/gate`.

## Curiosity (`curiosity.*`)

Five MLP heads each predict the next posterior stoch (probabilities, 1024
numbers) from (feat, action taken), trained on replay with gradients stopped.
In imagination, the variance of their predictions, averaged over dimensions,
is the curiosity reward for that step (Plan2Explore). It feeds a third head of
the shared critic, `explore`, normalised separately:

- the bottom actor learns from game + `goal_weight` x goal + `weight` (0.2) x
  explore;
- the manager's goal values are game + `mgr_weight` (0.2) x explore.

Known weakness: each stream's return is divided by its spread (95th minus 5th
percentile) but never by less than 1. That floor suits the sparse game reward;
curiosity returns span only ~0.02 and are never scaled up, so at the weights
above curiosity is a small share of the signal (about 2% in the first v2 run).
A fix (`curiosity.normalize`, `retlimit`, tried as `v2cur`) did not change the
score and lives on `family/v`. Watch `curiosity/reward`: it should fall where
the model has learned and rise in new places.

## Running

The next agent's flags are in the README. RSSM-2 as a memory alone is
`--agent.mapmodel.enabled True --agent.mapmodel.to_actor True
--agent.mapmodel.target memory` on top of the plain flags; curiosity needs
the manager's shared critic, so it cannot run without `agent.manager`.
