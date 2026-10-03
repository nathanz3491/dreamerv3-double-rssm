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

## Experiment A: results (2026-10-01)

`tools/action_suppression.py`, 20 episodes per checkpoint on the same fixed
evaluation worlds as `death_eval`. Validity comes from the game itself: at
every step the state is copied, all 43 actions are stepped with one random key,
and an action counts as valid if the result differs from doing nothing.
Measurement only. Raw numbers: `docs/experiment_a/*.json`.

| | vanilla | old record | control | honest map |
|---|---|---|---|---|
| valid-action mass | 35.7% | 39.8% | 37.2% | 39.4% |
| actions that do something, per step | 5.2 | 4.9 | 4.4 | 4.6 |
| entropy H(A\|s) | 0.153 | 0.118 | 0.129 | 0.099 |
| entropy of the mean policy | 0.732 | 0.678 | 0.758 | 0.688 |
| I(A; s) | 0.579 | 0.560 | 0.628 | 0.589 |

**Not a one-key collapse.** The mean policy's entropy is five to seven times
the per-state entropy: the agents are confident in each state and choose
differently in different states. On that measure the low entropy looks like
skill. But only 36-40% of the probability lands on actions that do anything;
the rest goes to no-ops, mostly DO with nothing in front (14-19% of all
presses) and keys that can never work there.

**The frontier is suppressed.** Probability of each key in the states where it
would work, with the number of such states:

| key | vanilla | old record | control | honest map |
|---|---|---|---|---|
| make wood pickaxe | 6.3% (22) | 25.0% (32) | **0.0% (150)** | 15.6% (34) |
| make wood sword | 0.1% (23) | 1.5% (71) | 21.5% (58) | 1.4% (80) |
| place table | 1.1% (798) | 3.9% (823) | 6.2% (643) | 3.5% (866) |
| place stone | — | 0.4% (386) | — | **0.0% (133)** |
| place furnace | — | **0.0% (386)** | — | **0.0% (133)** |
| make stone pickaxe | — | 0.0% (4) | — | 0.0% (17) |

Three findings:

1. **In every run exactly one wood craft survives.** It is pressed far more
   where it works than where it doesn't (old record: pickaxe 25.0% when valid
   vs 0.8% when not). The other stays flat or dead. Which one survives
   varies: the pickaxe in vanilla, the old record and the honest run; the
   sword in the control. The control stood at a table able to craft a pickaxe
   in 150 states and gave it 0.0% in every one. The honest run carries the same
   recipe and death fixes and kept the pickaxe, so those fixes are not what
   killed it. This is the pattern valid-action suppression predicts, with an
   arbitrary winner per run.
2. **The tech tree stops because the next keys are dead where they would
   work.** The old record could have placed a furnace in 386 states and never
   did; the honest run had 133 such states and never placed one. The stone
   pickaxe was never pressed in any state where it was craftable. The agent is
   reaching the frontier; it is not pressing the button there.
3. **Some keys are pressed more where they cannot work.** Place plant in all
   four runs, the stone pickaxe wherever it was ever craftable, and place table
   in the honest run all have higher probability where they are invalid. The
   policy has not learned those preconditions at all.

### What this implies for experiment B

A mask applied to an already-trained policy will not help: renormalising a
probability of 0.0% over the valid set leaves it at 0.0%. The fix has to act
during training. With a mask in training, an invalid action has zero
probability and so receives no gradient, and its logit is never pushed down in
the states where it cannot work. That is the mechanism the Zabounidis et al.
paper identifies. It needs the mask in both the real env and imagination, or
the alternative: give the agent the 43 validity flags as an encoder input so
RSSM-1's latent carries them. The game oracle built here is the test for any
observation-derived validity function that B needs.

## Experiment B: built (2026-10-01)

Two arms on the honest-map configuration, each a separate 1.1M-step run
compared against `honest_map` (5.87):

| arm | flags | what changes |
|---|---|---|
| B1, tell it | `--env.craftax.valid_obs True --agent.valid.input True` | the 43 flags are an encoder input; the decoder must reconstruct them |
| B2, mask it | `--env.craftax.valid_obs True --agent.valid.mask True` | impossible actions get zero probability and zero gradient |

`dreamerv3/craftax_valid.py` reads only the observation vector (a test pins its
signature) and agrees with the game in 77,700 of 77,700 action-state checks
over random play plus states seeded with random inventories, stations,
potions, mana and XP. NOOP, the moves and DO are never masked.

In B2 the real-game mask is the true flags. Inside imagination there is no
observation, so a learned head (`feas`) predicts validity from the latent. The
imagined rollout and the actor loss read the same head on the same states, so
they stay consistent -- the property `imag_shift` broke. Its threshold (0.1)
errs toward allowing: a missed possible action would be unpressable in every
dream; a false "possible" only costs a no-op. Logged as `valid/recall` and
`valid/false_pos`.

What would count as a result: deep achievements (stone pickaxe, furnace) rising
without raw entropy rising, and experiment A on the new checkpoints showing the
frontier keys' probability where valid lifting off zero.

## Experiment B: results (2026-10-03)

Both arms trained 1.1M steps on the honest-map configuration, scored with
`death_eval` (30 episodes) and experiment A (20 episodes) on the same worlds as
every other run. Raw experiment A files: `docs/experiment_a/expB_*.json`.

| | honest map | B1, told | B2, masked |
|---|---|---|---|
| achievements | 5.87 | 6.90 | **7.60** |
| without WAKE_UP | 4.90 | 6.10 | **6.80** |
| place furnace (episodes) | 0% | 23% | **40%** |
| place stone (episodes) | 0% | 0% | **40%** |
| collect coal (episodes) | 0% | 0% | **20%** |
| probability on actions with an effect | 39.4% | 46.6% | **57.1%** |
| furnace, probability where valid | 0.0% (133) | 1.2% (452) | **3.2% (599)** |
| place stone, probability where valid | 0.0% (133) | 0.0% (452) | **2.4% (599)** |
| entropy H(A\|s) | 0.099 | 0.127 | 0.062 |

Reading these against the four outcomes set out above: **both helped, masking
far more.** The frontier keys that sat at exactly 0% where they worked come off
zero only once a useless press can no longer push them down, which is the
suppression mechanism. Telling the agent what is possible helps too (both wood
crafts survive in B1 for the first time), but placing stone stays dead there.

In B2 every no-effect press is a basic action (DO with nothing in front, NOOP,
a blocked move), and per-state entropy halves while the mean policy stays
varied: more decisive, not collapsed.

**The next wall is sequencing.** The stone pickaxe was craftable in only 5 of
B2's 5,167 evaluation steps: it mines stone but rarely returns to a table with
both wood and stone. That is planning, not suppression, and it gates iron and
the 3-5-point tier. Survival did not improve (B2 lifespan 246; half of its
deaths are mobs or lava).

Caveats: one seed per arm; masking changes the action space relative to
published Craftax baselines and is reported as a separate arm.
