# DreamerV3 with a second, slower RSSM

A fork of [DreamerV3][upstream] that adds a **second recurrent state-space model
dedicated to spatial memory**, and feeds its output to the actor and critic —
but never to the world model.

The original README is preserved at [`README-dreamerv3.md`](README-dreamerv3.md);
installation and the general run interface are unchanged.

[upstream]: https://github.com/danijar/dreamerv3

---

## The problem

In [Craftax][craftax], DreamerV3 plateaus at the wood tier. Diagnosing a trained
1.1M-step agent turned up something more specific than "it stops improving":

| | trained agent | random policy |
|---|---|---|
| episode length | 278 steps | 261–284 steps |
| achievements | ~5 | 1.63 |

**1.1M steps of training bought zero survival gain.** The agent unlocks more
achievements, but it dies just as fast — and a do-nothing agent survives 333
steps, longer than either. It never learned to stay alive at all.

The mechanism: Craftax achievements fire **once**. Drinking water the first time
pays; drinking water the twentieth time, when you are actually about to die of
thirst, pays nothing. Maintenance behaviour is unrewarded, so under the agent's
own learned model, dying is free.

Anything that takes more than one step to reach is hard for it, and the reason is
spatial. The trees, the water, the crafting table and the cows are in different
places. Reaching one means a sequence of moves, each of which the agent chooses
without knowing where it is or where anything else is — Craftax's 8,268-dim
observation contains **no absolute position**, only a 9×11 egocentric window.

[craftax]: https://github.com/MichaelTMatthews/Craftax

## The approach

RSSM-1 is stock DreamerV3, ticking every step, modelling one-step dynamics.
RSSM-2 ticks once per **8** steps and predicts a coarse map of the whole level.

```
                RSSM-1  (every step, 4096 deter + 32x32 stoch)
                   |
                  sg()                      <- map loss never reaches RSSM-1
                   v
                RSSM-2  (every 8 steps, 1024 deter, deterministic)
                   |
             decode 12x12x16 map + 144-way position
                   |
              crop 9x9 around the agent, flatten -> 1296
                   |
                  sg()  x  learned gate     <- policy grads never reach RSSM-2
                   v
     concat([deter1, stoch1, mapfeat]) -> pol, val   ONLY
     concat([deter1, stoch1])          -> rew, con, dyn
```

### Four design choices worth stating

**The map reaches the actor, never the world model.** Stock DreamerV3 has one
`feat2tensor` feeding five heads. Two of them (`rew`, `con`) *are* the world
model, so a wrong map cell there would fabricate imagined reward and corrupt
every plan built on it. The map goes through a separate `actor2tensor` that only
`pol` and `val` read.

**No CNN.** The crop is egocentric, so *position is the meaning* — stone in the
top-left means "walk up-left". A convolution computes the same function
everywhere by construction, which would deliberately blur "above me" into "below
me". The crop is flattened and concatenated raw.

**Position is a target, not an input.** The observation has no absolute (x, y),
so RSSM-2 must dead-reckon from its own movement. Asking it to predict its own
coarse cell is what makes path integration actually get learned.

**A learned gate instead of a warm-up schedule.** The actor's map input is
multiplied by one scalar, initialised at 0. At 0 the actor is map-blind, so an
untrained RSSM-2 cannot poison the policy while everything trains at once — but
the gate still receives gradient, so the channel opens exactly as fast as it
starts paying. It doubles as a continuous ablation: if `map/gate` stays near
zero, the actor found the map useless.

### Pooling is fixed per plane

Each coarse cell covers 4×4 tiles and carries 16 independent values. How those
4×4 tiles collapse into one number was measured over 8 fresh worlds:

| pooling | planes | why |
|---|---|---|
| **mean** | water, stone, sand, plant | fill fraction spans 0.05–0.95 across cells; a scattered pebble field and a solid mountain must not both read 1.0 |
| **max** | tree, lava, coal, iron, diamond, gem, table, furnace, path, mobs | 91% of tree cells are below 0.25 fill and none exceed 0.75 — a mean would read ~0.1 everywhere, indistinguishable from empty |
| decay | `seen` | recency, not a flag: a cow seen 5 steps ago is still there, one seen 300 steps ago is not |

Coal (8.2% of cells), iron (5.6%) and diamond (0.3%) get separate planes — they
gate different tech tiers and occur at rates differing by 25×.

## Running it

Stages are config flags on one binary:

```bash
# baseline — stock DreamerV3, byte-for-byte
python dreamerv3/main.py --logdir ~/logdir/base --configs craftax size50m

# RSSM-2 trains, actor untouched
python dreamerv3/main.py --logdir ~/logdir/s3 --configs craftax size50m \
  --env.craftax.mapmodel True --agent.mapmodel.enabled True

# + map reaches actor and critic, crop frozen through imagination
  --agent.mapmodel.to_actor True --agent.mapmodel.imag_shift False

# + crop slides through imagination
  --agent.mapmodel.to_actor True --agent.mapmodel.imag_shift True
```

`env.craftax.mapmodel` emits the targets; `agent.mapmodel.enabled` trains RSSM-2
on them. Both are needed. The targets are **supervision only** and never enter
the encoder — the agent's observation stays the stock 8,268 floats. By default
the target is a mosaic built only from the agent's own lit 9x11 windows
(`craftax_map.coarse_map_observed`), weighted by `mapknown` so unseen cells
carry no gradient; `--env.craftax.map_privileged True` reinstates the original
full-map ground truth as an ablation.

The whole model on one page — where the potential sits, what RSSM-2 reads,
which heads see the map, every stop-gradient — is
[`docs/architecture.svg`](docs/architecture.svg) /
[`.pdf`](docs/architecture.pdf), regenerated with
`python tools/make_architecture.py`.

**Disk.** A finished 1.1M-step run takes ~1.4 GB: ~0.8 GB of replay and a
~0.7 GB checkpoint. The latents the agent caches in replay for
`replay_context` (`dyn/`, `map/`, ...) used to be ~90% of a run's 9.3 GB. They
barely compress and nothing after training reads them, so
`replay.save_skip` keeps them in memory but off disk, and a resumed run gets
zeros back that are refreshed as it trains. To shrink a run that predates
this: `python tools/strip_replay.py ~/logdir/<run>`. It refuses a run that is
still training or stopped short of its `run.steps`; after stripping, that run
can no longer be resumed.

Tests (pure numpy, no GPU):

```bash
python -m pytest dreamerv3/test_craftax_map.py -q
```

## Status

> **2026-10-11:** every RSSM-2 result below predates three fixes. Non-movement
> actions counted as a step down. A new episode inherited the old episode's
> RSSM-2 state. Training saw up to 7 steps of the future that acting never
> saw. Details are in
> [`docs/rssm2-step-alignment.md`](docs/rssm2-step-alignment.md). B2, honest
> map, v1.2 and v2 are being rerun with the fixes. Vanilla DreamerV3 is
> unaffected.
>
> A second audit found that evaluation worlds were only identical across runs
> for the first episode. Later worlds depended on how long earlier episodes
> ran. The tool now reseeds every episode, and every run is to be re-scored.
> All findings are in [`docs/audit-2026-10-11.md`](docs/audit-2026-10-11.md).

**Current best: 9.73 achievements per episode** (manager v1.2: B2 plus a
manager that picks a tech-tree goal and holds it until reached, with one
two-headed critic shared by both levels), against 8.53 for v1.1, 7.60 for B2
and 4.44 for vanilla DreamerV3. **Normalized return, measured with Craftax's
unshaped reward: 3.91%** (mean return 8.83 of 226 over 30 evaluation episodes;
B2 about 3.0%; best published 1M-step agent, ITC, 7.09%). It is the first run
to make the stone pickaxe regularly (23% of episodes). Manager v1.3 (v1.2 plus
survival goals and a survival potential) ties it at 3.91% and lives 73 steps
longer. The
masking and the manager's goals use game knowledge at test time, which the
published agents do not. Every run, compared on the same evaluation worlds with
each achievement's unlock rate:
[`docs/comparison-report.pdf`](docs/comparison-report.pdf); all training
curves: [`docs/training-curves.png`](docs/training-curves.png).

What holds B2 back, measured step by step from its replay and shown in its
own episodes: [`docs/b2-casebook.pdf`](docs/b2-casebook.pdf)
(`tools/episode_cases.py` on the box, then `tools/make_casebook.py` from an
environment with Craftax installed). In short, it spends wood as fast as it
gets it: it builds a table the moment it holds two logs (2.2 tables per
episode), and a stone-pickaxe chance lasts one step before its only log goes
into the wood sword. 41% of episodes never drink.

Next: a two-level agent. A manager picks a tech-tree goal every 8 steps
from RSSM-2's slow state, and the actor pursues it
([`docs/design-manager.md`](docs/design-manager.md)). v1 scored 7.5
achievements/episode against B2's 7.9, with the manager still choosing goals
uniformly after 900k steps — it was never trained, since most goals can't be
reached within 8 steps. v1.1-v1.3 (2026-10-04) fixed that in turn: v1.1 pays
both levels the first time a goal is reached per segment/episode; v1.2 shares
one two-headed critic between manager and actor so the manager takes the
exact gradient over all 13 goals instead of a noisy sampled advantage, and
holds a goal until reached or 32 steps; v1.3 adds a survival potential plus
DRINK/EAT goals, since B2's episodes either survive (7 drinks, little tech)
or progress (5 tech stages, ~2 drinks, dead of thirst near step 330) but
never both. Enable with `--env.craftax.goals_obs True
--agent.manager.enabled True --agent.manager.critic shared
--agent.manager.hold 32 --env.craftax.survival potential+meters
--env.craftax.goals_survival True` on top of the B2 flags. v1.1, v1.2 and v1.3
are trained and scored above.

Experiment B3 learns the action mask from the world model's own counterfactual
instead of reading it from the game's rules (`--agent.valid.learned True`). It
scored 5.60, below B2's 7.60, because its labels fed back on themselves. B3-fix
detaches the mask head and labels from decoded observations:
`--agent.valid.detach True --agent.valid.label obs --agent.valid.margin 0.5
--agent.valid.threshold 0.15`. It was stopped at 298k steps: by 254k it blocked
over half of the valid crafting actions, those whose effect the world model
had not learned yet, so the learned mask is set aside. Details are in
[`docs/entropy-and-action-suppression.md`](docs/entropy-and-action-suppression.md).

v2 ([`docs/design-v2.md`](docs/design-v2.md)) is the two-level agent with
learned goals, curiosity instead of reward potentials, and RSSM-2 as a memory.
It keeps B2's rule-based action mask, the only game knowledge left. On the 30
evaluation worlds it scores **6.90** achievements, and **7.00** for v2-cur
(curiosity actually switched on). That is about 3 above vanilla, 0.6 below B2
and 2.7 below v1.2. The real normalized return is 2.65% for v2 and 2.70% for
v2-cur, against 3.91% for v1.2 and ITC's 7.09%. See
[`docs/comparison-report.pdf`](docs/comparison-report.pdf).


The map model helps, and the tech-tree gate has opened for the first time:
**map + potential shaping reaches 5.80 achievements/episode at 1.1M steps**,
against 4.44 for vanilla DreamerV3. Full writeup, every number sourced from
`tools/death_eval.py`: [`docs/raising-achievements.md`](docs/raising-achievements.md).

**2026-09-28, re-scored against every finished run:** the 5.80 headline needs
two caveats. The newest run (recipe + death fixes, privileged map) scores
highest of all (**5.93**) but **never crafts a pickaxe** (0/30 episodes vs.
9/30 for the 5.80 run) — the total went up while the tech tree got shallower.
And **every map run's lead over vanilla is `WAKE_UP` (sleep)**: 83-98% unlock
rate vs. 0% for vanilla; strip it out and the fixed map model scores 4.20
against vanilla's 4.44. Only the potential shaping adds real tech-tree
progress on top. Full table and method:
[`docs/comparison-report.pdf`](docs/comparison-report.pdf) /
[`docs/eval_results.json`](docs/eval_results.json), from
`tools/make_comparison_report.py`. The honest (non-privileged) map run was
still training as of this report (an out-of-memory restart) and isn't in it
yet.

| metric | value | chance |
|---|---|---|
| map BCE / cell | 0.047 | 0.693 |
| position accuracy | 0.863 | 0.007 |

`map/gate` — whether the actor is genuinely reading the map rather than
ignoring a decorative channel — rose monotonically 0.095 → 0.526 over 1.1M
steps with no reversal.

These map numbers come from the **privileged** runs, trained against the true
map on every cell, and "chance" is not a fair floor: most planes are nearly
always empty. Scored with `tools/map_eval.py` against a per-plane prior, that
model beats the prior on cells it saw but is confidently *worse* than the prior
on cells it never saw. Current runs train only on what the agent observed
(`ObservedTargets`); their numbers will replace these.

How to read policy entropy — skill versus suppressed valid actions, the metrics
that separate them, and the experiments planned:
[`docs/entropy-and-action-suppression.md`](docs/entropy-and-action-suppression.md).

**2026-10-01:** the honest (non-privileged) map run finished training —
**5.87 achievements/episode, against the privileged control's 5.93** — and,
unlike the control, **crafts a pickaxe in 20% of episodes**, ruling out the
recipe/death fixes as the reason the control never crafts one.
**Experiment A** (`tools/action_suppression.py`) then answered why directly:
scored against a game-oracle validity function, every checkpoint spends only
36–40% of its action probability on actions that do anything, and the
control pressed a craftable wood pickaxe in 0 of 150 valid states while its
wood sword got 21.5% — action-support suppression at the exact frontier key,
not an unreached-state problem. **Experiment B** (`craftax_valid.py`, two
training arms — feed the validity flags to the encoder, or mask invalid
actions to zero probability/gradient during training) is built and tested
but **not yet trained**.

### Known limitation (fixed; kept for the ablation)

`rssm.imagine()` threads only `(deter, stoch)` through its scan, so a crop that
slides step by step cannot be handed to the rollout policy. With
`imag_shift: True` the rollout **picks** imagined actions reading the crop
frozen at the imagination start, while the loss **scores** those actions
reading the shifted crop. `imag_loss` is REINFORCE
(`logpi(a) x advantage`), so this takes the log-probability of actions drawn
from a different distribution — not a bounded ~4-cell gap as originally
assumed, but a runaway one: measured at 1.1M steps, `True` gives policy loss
-61.09 and action entropy 0.005 (collapsed); `False` gives 0.0001 and 0.138,
matching vanilla's 0.134. Every conclusion drawn before this fix — that the
map collapses entropy, trades depth for breadth, or blocks crafting — was a
conclusion about the bug. `imag_shift` now defaults to `False`.
Raising `imag_length` to 50 (to let the agent see thirst death inside
imagination) was tried and reverted: at `size50m` with `imag_last: 0` it
compiles but never completes a training step. 30 trains normally; the ceiling
between 30 and 50 hasn't been found.
Closing the shift bug properly means threading the coarse cell through the
scan.

## Files

| path | what |
|---|---|
| `dreamerv3/craftax_map.py` | coarse-map targets, pure numpy |
| `dreamerv3/mapmodel.py` | RSSM-2, the gate, dead reckoning |
| `dreamerv3/agent.py` | wiring: losses, actor path, imagination, replay context |
| `embodied/envs/craftax.py` | emits `map12`, `mappos`, `mapseen` |
| `dreamerv3/test_craftax_map.py` | target unit tests |

## Credit

DreamerV3 is by Danijar Hafner, Jurgis Pasukonis, Jimmy Ba and Timothy
Lillicrap; this fork keeps the original MIT licence and adds to it. Craftax is
by Michael Matthews et al.
