# What actually raised the achievement count

Craftax · DreamerV3 · measured on an RTX 4090, 1.1M steps per run, `size50m`.

Every number below is the **episode-final achievement count** from
[`tools/death_eval.py`](../tools/death_eval.py), 25–30 episodes per run, one
tool across all runs. It is *not* `episode/score`, which under shaping includes
the shaping term and is inflated by construction. It is also not
`epstats/log/achievements/avg`, which is a per-step mean of a total that climbs
0 → final and therefore reads ~35% low; use `epstats/log/achievements/max`.

| run | achievements | vs vanilla |
|---|---|---|
| **Map model, `imag_shift False`** | **5.03** | **+0.59** |
| Vanilla DreamerV3 | 4.44 | — |
| Map model, `imag_shift True` (bug) | 4.35 | −0.09 |
| Map + potential, `imag_shift True` (bug) | 3.87 | −0.57 |
| Map + five hand-written reward terms | 2.30 | −2.14 |
| Map + potential, `imag_shift False` | *pending* | — |

---

## 1. The single biggest change: `imag_shift: False`

**+0.68 achievements from one flag** (4.35 → 5.03, identical configs otherwise).
This was a bug fix, not a feature.

### What broke

The actor learns inside imagination: the world model is rolled forward 15 steps
with no real observations, and the policy gradient comes from that rollout. The
agent's map input is a 9×9 crop centred on where it believes it is.

`imag_shift: True` was meant to slide that crop as the agent imagines moving, so
that imagining "walk north" changes what the actor sees and therefore produces a
gradient about navigation. Without it, imagined movement changes nothing.

The implementation could not deliver that. `rssm.imagine()` threads only
`(deter, stoch)` through its `nj.scan`, so a crop that slides step by step
cannot reach the rollout policy. The result was two different policies:

- [`agent.py:434-436`](../dreamerv3/agent.py#L434) — the rollout **picks**
  imagined actions while reading `frozen`, the crop at the imagination start.
- [`agent.py:453-465`](../dreamerv3/agent.py#L453) — the loss **scores** those
  actions while reading the shifted crop.

`imag_loss` is REINFORCE (`logpi(a) × advantage`), so it takes the
log-probability of actions drawn from a *different* distribution. As the learned
gate opens and the two crops diverge, that log-probability runs away.

### Measured

Identical configs differing only in this flag:

| @1.1M | `True` | `False` | vanilla |
|---|---|---|---|
| `train/loss/policy` | **−61.09** | **0.0001** | 0.00 |
| action entropy | 0.005 | **0.138** | 0.134 |
| `dyn` KL | 1.76, falling | 2.58, rising | 3.04, rising |
| `COLLECT_DRINK` | 0.43 | **0.70** | 0.80 |
| `PLACE_TABLE` | 0.35 | **0.63** | 0.64 |
| `MAKE_WOOD_PICKAXE` | 0.00 | **0.07** | 0.24 |

The divergence scales with the gate, which is the signature: `map_stage5`
reached gate 0.53 and policy −61; the potential run reached gate 0.32 and
policy −30.

Rising `dyn` KL means the agent keeps reaching states the world model cannot
predict — it never stopped exploring. Under the bug it fell, because the policy
had collapsed onto a narrow loop.

### Status

Default is now `False` — [`configs.yaml:120`](../dreamerv3/configs.yaml#L120),
with the reasoning inline. Making `True` work means threading the coarse cell
through `nj.scan` in `rssm.py`; worth doing now that the map is known to help.

**Every conclusion drawn before this fix was a conclusion about the bug** — that
the map collapses entropy, trades depth for breadth, or prevents crafting. All
void.

---

## 2. The map model itself

A second, slower RSSM that ticks once per 8 env steps and predicts a 12×12×16
coarse map of the level plus the agent's own coarse cell.

| piece | where |
|---|---|
| RSSM-2 (deterministic GRU, no stochastic latent) | [`mapmodel.py:82`](../dreamerv3/mapmodel.py#L82) |
| supervision targets, pure numpy | [`craftax_map.py`](../dreamerv3/craftax_map.py) |
| egocentric crop | [`mapmodel.py:169`](../dreamerv3/mapmodel.py#L169) |

Three choices that matter:

**The map reaches the actor only, never the world model.** Stock DreamerV3 has
one `feat2tensor` feeding five heads, two of which (`rew`, `con`) *are* the world
model — a wrong map cell there would fabricate imagined reward.
[`agent.py:83`](../dreamerv3/agent.py#L83) adds a separate `actor2tensor` that
only `pol` and `val` read; [`agent.py:59`](../dreamerv3/agent.py#L59) leaves
`feat2tensor` untouched.

**Two stop-gradient boundaries.** [`agent.py:389`](../dreamerv3/agent.py#L389)
blocks the map loss from reaching RSSM-1; [`agent.py:116`](../dreamerv3/agent.py#L116)
blocks policy gradients from reaching RSSM-2. The map stays a description of the
world rather than becoming whatever raises return this batch.

**No CNN.** The crop is egocentric, so *position is the meaning* — stone in the
top-left means "walk up-left". A convolution computes the same function
everywhere by construction, which would blur "above me" into "below me". The
1,296 values are flattened and concatenated raw.

---

## 3. The learned gate

[`mapmodel.py:110`](../dreamerv3/mapmodel.py#L110) — one scalar, initialised at
**0**, that the actor's map input is multiplied by.

This replaced a planned warm-up schedule and teacher-forced crop positions. At 0
the actor is map-blind, so an untrained RSSM-2 cannot poison the policy while
everything trains jointly; the gate still receives gradient at 0, so the channel
opens exactly as fast as it starts paying.

It also doubles as a permanent ablation readout. `map/gate` near zero after
training would mean the actor found the map useless. It never has: 0.095 → 0.526
across 1.1M steps with no reversal, and the fixed run reached 0.39.

Sign is arbitrary — the actor's first layer absorbs a flip for free, so only
|gate| is meaningful.

---

## 4. Per-plane pooling, measured not guessed

Each coarse cell covers 4×4 tiles and carries 16 values. How those tiles collapse
into one number was measured over 8 fresh worlds:

| pooling | planes | why | code |
|---|---|---|---|
| **mean** | water, stone, sand, plant | fill spans 0.05–0.95 across cells; max would print 1.0 for a scattered pebble field and a solid mountain alike | [`craftax_map.py:60`](../dreamerv3/craftax_map.py#L60) |
| **max** | tree, lava, coal, iron, diamond, gem, table, furnace, path, mobs | 91% of tree cells are below 0.25 fill and none exceed 0.75; a mean would read ~0.1 everywhere | [`craftax_map.py:64`](../dreamerv3/craftax_map.py#L64) |
| decay | `seen` | recency, not a flag — a cow seen 5 steps ago is still there, one seen 300 steps ago is not | `update_seen` |

Coal (8.2% of cells), iron (5.6%) and diamond (0.3%) get separate planes: they
gate different tech tiers and occur at rates differing by 25×.

---

## 5. Potential-based reward shaping

[`craftax_potential.py`](../dreamerv3/craftax_potential.py), wired at
[`craftax.py:260`](../embodied/envs/craftax.py#L260) behind
`--env.craftax.survival potential`.

`F = γΦ(s′) − Φ(s)`, where Φ measures progress toward each achievement's
prerequisites plus capability currently held. Potential-based shaping provably
cannot change which policy is optimal, so it cannot invent new optima.

### Why the form matters

Three earlier hand-written bonus terms each created a cheaper way to earn than
the behaviour they encoded — that run scored **2.30**, half of no shaping at all:

| term | intent | measured outcome |
|---|---|---|
| `alive` 0.005/step | survive | agent stationary **86% of steps** |
| `restore` 0.3 | drink when low | collected **0.0 times in 50 episodes** |
| `idle` −1.0 | keep moving | defeated by jiggling in place |

The sizing error was calibrating by share of episode total rather than **points
per unit of effort**. `alive` was the smallest term (1.3/episode) but its effort
was zero, so its rate was unbounded and no magnitude would have made it safe.

Under a potential every loop cancels by construction — collect wood then drop it
nets zero, and standing still costs `(γ−1)Φ` per step. Asserted in
[`test_craftax_potential.py`](../dreamerv3/test_craftax_potential.py) rather than
trusted.

### Two things the tests caught

**Unlocked achievements must keep their full weight** — drop them from Φ on
success and Φ falls at the moment of victory, punishing the agent for winning.

**`PREREQ_CEIL` needed 0.25, not the obvious 0.6**
([`craftax_potential.py:61`](../dreamerv3/craftax_potential.py#L61)). Crafting
*consumes* its ingredients, so making a wood pickaxe spends the wood and lowers
progress on every other recipe needing wood. At 0.6 the keypress was worth +0.25
against +2.60 for merely walking to the table — the completing action was the
cheapest step on its own ramp. At 0.25 it is worth ~2× the last prerequisite.

**Capability is paid every step it is held**
([`craftax_potential.py:113`](../dreamerv3/craftax_potential.py#L113)). This
supplies the compounding return Craftax lacks: a tool is otherwise a receipt for
a reward already spent, which is why value-per-step falls with tier depth.

### Status

Measured only under the `imag_shift` bug (3.87, peak 4.68 at 477k). A clean run
is in flight; **the potential has no trustworthy measurement yet.**

---

## 6. Two config traps fixed along the way

**`replay.size`** — [`configs.yaml:231`](../dreamerv3/configs.yaml#L231), capped
at 4e5. The default 5e6 never evicts inside a 1.1M-step run, so peak memory is
the whole run. The map targets make each step ~40% bigger (33 KB → 47 KB), which
OOM-killed a run at 690k on a 62 GB box.

**`imag_length`** — [`configs.yaml:140`](../dreamerv3/configs.yaml#L140), back to
15. Raising it to 50 to cover thirst death (~50 steps out) gives 1024 rollouts ×
51 steps at `imag_last: 0`; it compiles, enters the training loop, and never
completes a single training step. 30 trains normally.

---

## What has not been fixed

**`MAKE_WOOD_PICKAXE` is 0.07** against vanilla's 0.24. The map still trails on
tech-tree depth even with the bug gone, and `COLLECT_STONE` remains at zero.

A probe that placed the trained agent beside its own crafting table, holding
wood, meters full — a state where one keypress yields a pickaxe, verified 10/10
seeds — got **zero presses across 25 trials using 6 of 43 actions**
([`tools/craft_probe.py`](../tools/craft_probe.py)). That was measured before the
fix and is worth re-running.

The remaining candidate is exploration: adaptive entropy targeting rather than a
fixed `actent`, or seeding replay with a handful of scripted crafting
trajectories so the transition exists in the buffer at all.

---

## Reproducing

```bash
python dreamerv3/main.py --logdir ~/logdir/map --configs craftax size50m \
  --env.craftax.mapmodel True \
  --agent.mapmodel.enabled True --agent.mapmodel.to_actor True
```

`imag_shift`, `imag_length` and `replay.size` now default correctly. Add
`--env.craftax.survival potential` for the shaped variant.

Watch it with [`tools/mapwatch.py`](../tools/mapwatch.py), which checks the
process is alive before quoting an ETA — reading `metrics.jsonl` alone reported
a run as healthy four hours after it had been OOM-killed.
