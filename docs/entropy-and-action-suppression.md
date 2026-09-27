# Low entropy: skill, or suppressed actions?

A reference note on how to read policy entropy in this project, what our data
says so far, and which experiments tell the explanations apart. Written
2026-09-27 after reviewing the "human experts collapse too" hypothesis
(`创新点_gpt5.6.md`) against our own measurements.

## The question

Human players settle into a fixed routine for the parts of a game they have
mastered and explore only at the edge of what they can do. So is a policy whose
entropy drops *expert-like* rather than *broken*?

Half right. A good policy **should** be near-deterministic in states it has
mastered — at a table with two logs, `MAKE_WOOD_PICKAXE` should be close to
certain. But "the entropy went down" is compatible with two very different
things, and the average entropy we log cannot tell them apart:

| | **competence-conditioned concentration** (healthy) | **action-support collapse** (pathological) |
|---|---|---|
| mastered states | low entropy, high success | may look identical |
| opportunity states (table + wood, water adjacent) | the right action is likely | the right action is still near zero |
| novel states | directed exploration, or diversity at a higher level | repeats old actions |
| small map changes | adapts closed-loop | follows the old path and fails |
| a rare action becomes valid for the first time | used quickly | already suppressed |
| achievement / state coverage | keeps expanding | stops early |

Don't call the healthy case "entropy collapse", and don't use the human analogy
as evidence. It is a hypothesis; the table above is how to test it.

## The mechanism that makes the pathological case likely here

*Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms*
(Zabounidis et al., arXiv 2603.09090, March 2026 — verified; tested on Craftax,
Craftax-Classic and MiniHack). When an action is invalid in most states the
agent visits, the policy gradient pushes its probability down there, and shared
network parameters carry that suppression into the rare unvisited states where
the action is valid. The paper bounds the probability of such an action by an
exponential decay and proposes **feasibility classification** as the practical
fix.

In Craftax every one of the 43 actions can always be pressed; without its
preconditions an action is a silent no-op. Crafting, drinking and descending
are invalid almost everywhere the agent goes, which is exactly the condition
the paper describes.

## What our data already shows

From `tools/action_audit.py` on the map + potential checkpoint (8 episodes,
1978 steps, before the spine fix):

| measurement | value |
|---|---|
| presses of permanently impossible actions (potions, magic, enchanting, attributes, diamond tier) | 18.2% |
| steps where nothing changed (position, inventory, achievements) | 61.6% |
| move pressed but the agent did not move (wall, placed plant) | 15.6% |
| faced a tree, pressed DO | 20.6% of 165 chances |
| faced stone, pressed DO | 40.0% of 90 chances |
| faced water, pressed DO | 46.7% of 15 chances — water reached on 0.8% of steps |
| mean normalised entropy | 0.118 |

A policy that is confident at the tree and still doesn't chop 79% of the time
is confident in the *wrong* place. That fits suppression, not mastery. The
spine off-by-one (`PLACE_TABLE` at 1 wood instead of 2) is a second, compounding
cause, fixed in `7a7530e`.

Two cautions about older numbers:

* The entropy of **0.004** in the Sept 16 report came from runs with the
  `imag_shift` policy-gradient bug. It is evidence of that bug, not of anything
  about skill.
* Normalised entropy is `H / ln 43`, so 0.12 is about 43^0.12 ≈ 1.6 "effective
  actions" and 0.004 about 1.0. Both are concentrated; "0.12 healthy vs 0.004
  collapsed" was too coarse.

## Why raw 43-action entropy misleads

Pressing six different impossible crafts has high action entropy and produces
six identical no-ops. Action entropy and the diversity of what actually happens
diverge whenever many actions do nothing. Report instead:

* **valid-action mass** — probability on actions whose preconditions hold now;
* **entropy within the valid set** — the policy renormalised over valid actions;
* **conditional vs marginal entropy** — `H(A|state)` averaged over states, and
  `H(A)` of the actions actually taken. An expert has `H(A|state) ≈ 0` with
  `H(A) > 0` (certain, but different in different states); a stuck policy has
  both near zero. The gap is the mutual information `I(A; state)`;
* **key-action probability and rank in opportunity states**;
* **entropy by situation** — facing a tree, at a table with ingredients, open
  grass, next to water;
* **effect coverage** — how often an action changes position, inventory,
  meters, terrain or achievements.

## Experiments that discriminate

**A. Valid-action analysis (no training).** Compute validity for every step of a
rollout, derived from the observation (every Craftax precondition is in it:
inventory, XP, mana, learned spells, and the 9×11 window for tables and
furnaces). Measure the metrics above on each checkpoint. If crafting is pushed
down across ordinary states *and* stays low at a table with wood, that is
suppression.

**B. Tell the agent what is possible.** Add the 43 validity flags as an
observation key into the encoder, so RSSM-1's latent carries them and every
imagined step inherits them (see `docs/architecture.svg`; not via
`actor2tensor`, where a frozen input would be stale inside the dream). Compare
against a hard logit mask applied in both acting and imagination. If deep
achievements rise without raw entropy rising, the problem was suppression, not
"too little entropy". Report either as a labelled arm — published Craftax
baselines do not mask.

**C. Exploration above the action level.** Hold one exploration choice for a
whole episode or frontier stage rather than injecting per-step noise
(Bootstrapped DQN; Agent57). Low per-step entropy can coexist with diverse
episodes.

**D. Return, then explore.** Reliably reach the current frontier, then explore
from it — Go-Explore. The env already has `save_state()` / `reset_to()`
scaffolding (Phase 4). This is the most direct, falsifiable form of the "repeat
what you've mastered, explore what's next" intuition.

Not recommended: **Instant Episode Repetition** (arXiv 2608.17347, read in
full). It replays a successful episode's exact action list open-loop. In MuJoCo
the start state is the same pose plus noise, so the replay stays meaningful; in
Craftax every episode is a new map and the same keypresses walk into water. It
is also an exploitation mechanism — the paper itself reports reduced diversity
and premature convergence at high repetition.

## Related directions and their status

* **LLM-consolidated long-term memory ("sleep").** Heavily covered (Generative
  Agents, ExpeL, Reflexion, Voyager, wake–sleep replay). The missing piece in
  any proposal is how the consolidated memory changes the actor. An LLM summary
  is a hypothesis, not a fact: keep raw episodes, never summarise a summary,
  verify rules in the environment.
* **LLM-written reward functions.** Covered by Auto MC-Reward (CVPR 2024),
  Eureka (ICLR 2024), CARD, ELLM, OMNI, and on Craftax itself by **SCALAR**
  (Zabounidis et al., arXiv 2603.09036, March 2026 — verified; reports 88.2%
  diamond collection on Craftax-Classic and 9.1% Gnomish Mines on full Craftax).
  The safe form is an LLM-proposed *potential* Φ added to an unchanged task
  reward — the slot our hand-written `craftax_potential.py` fills. Our own
  history backs its warnings: SCALAR found the LLM overestimated resource costs
  from Minecraft priors; we hand-wrote an underestimate (table = 1 wood). The
  game must check the recipe — hence `test_spine_ingredients_match_the_game`.
  Φ must also be zero on death (`33ad395`), or shaping pays for dying at high
  tech.
* **Dreamer-specific caveat for changing rewards mid-training:** replay stores
  rewards, and the reward head is trained on them. A new Φ makes every stored
  reward stale; it needs relabelling (possible — Φ is a function of state) or a
  fresh buffer.

## What not to claim

Not: "human experts also collapse their entropy, so ours is fine."

Instead: low action entropy is not failure in itself. An effective agent should
form reliable, low-entropy closed-loop skills where it is competent, keep rare
valid actions available at its frontier, and explore coherently at a higher
level. Mean action entropy cannot distinguish that from action-support
collapse; state-conditioned probes, validity analysis and causal interventions
can.

## Next step

Experiment A: an observation-derived validity function (checked against the
real game, like the recipe test) plus the metrics above in `action_audit.py`,
run on the privileged checkpoint and on `honest_map` / `priv_map` when they
finish. The validity function is also the core of experiment B.
