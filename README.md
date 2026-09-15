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

`env.craftax.mapmodel` emits the privileged targets; `agent.mapmodel.enabled`
trains RSSM-2 on them. Both are needed. The targets are **supervision only** and
never enter the encoder — the agent's observation stays the stock 8,268 floats.

Tests (pure numpy, no GPU):

```bash
python -m pytest dreamerv3/test_craftax_map.py -q
```

## Status

RSSM-2 learns quickly. Early in a 1.1M-step run:

| metric | value | chance |
|---|---|---|
| map BCE / cell | 0.047 | 0.693 |
| position accuracy | 0.863 | 0.007 |

**Whether this fixes the plateau is not yet known.** The number that decides it
is `map/gate`: if the actor keeps opening the channel, it is genuinely reading
the map; if the gate stalls near zero, the map is decoration and that is the
result. Runs are in progress.

### Known limitation

`rssm.imagine()` threads only `(deter, stoch)` through its scan, so a crop that
slides step by step cannot be handed to the rollout policy. With
`imag_shift: True` the rollout samples actions using the crop frozen at the
imagination start while the loss differentiates the shifted crop, making the
REINFORCE term slightly off-policy. At the original H=15 the gap was bounded
(the agent covers ~4 coarse cells inside a ±4-cell crop); **`imag_length` is now
50** (see `dreamerv3/configs.yaml`, changed to let the agent see thirst death
inside imagination), which likely widens this gap well past the crop — not yet
re-measured. `imag_shift: False` remains the exactly-consistent control.
Closing it properly means threading the coarse cell through the scan.

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
