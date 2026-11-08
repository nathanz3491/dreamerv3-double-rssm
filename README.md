# DreamerV3 on Craftax: the map-model family (RSSM-2)

`family/rssm2` is `main` (plain [DreamerV3][upstream] on symbolic
[Craftax][craftax]) plus a **second, slower recurrent state-space
model, RSSM-2**, whose state is fed to the actor and critic but never to the
world model. RSSM-2 is trained either to predict a coarse map of the level
(the *map target*) or, with no game labels at all, as a memory (the *memory
target*). The branch also holds the reward shaping the map runs used: a
potential over the tech tree, and the five hand-written survival terms it
replaced.

The plain baseline, its fixes and the scoring tools are documented on `main`;
this README covers what the family adds. The original DreamerV3 README is
kept at [`README-dreamerv3.md`](README-dreamerv3.md).

**No rule knowledge reaches the agent on this branch.** The rule mask and the
action-validity flags (`family/masked`) are not here, and none of these runs
used them. The map labels and the potential are built from game knowledge,
but only as training targets and shaped reward.

[upstream]: https://github.com/danijar/dreamerv3
[craftax]: https://github.com/MichaelTMatthews/Craftax

## Why

In Craftax, DreamerV3 plateaus at the wood tier, and its episodes last no
longer than a random policy's (~260-285 steps). Anything that takes more than
one step to reach is hard, and the reason is spatial: the trees, the water,
the table and the cows are in different places, and the 8,268-dim observation
holds **no absolute position**, only a 9x11 egocentric window. RSSM-2 is a
place for the agent to keep what it has seen.

## Design

RSSM-1 is stock DreamerV3, ticking every step. RSSM-2 ticks once per **8**
steps on the mean of RSSM-1's features over the window and the window's moves.

```
                RSSM-1  (every step, 4096 deter + 32x32 stoch)
                   |
                  sg()                      <- RSSM-2's loss never reaches RSSM-1
                   v
                RSSM-2  (every 8 steps, 1024 deter GRU, deterministic)
                   |
     map target:   decode 12x12x16 map + 144-way position,
                   crop 9x9 cells around the agent, flatten -> 1296
     memory target: RSSM-2's state itself -> 1024
                   |
                  sg()  x  learned gate     <- policy grads never reach RSSM-2
                   v
     concat([deter1, stoch1, mapfeat]) -> pol, val, slowval   ONLY
     concat([deter1, stoch1])          -> rew, con, dyn
```

- **The map reaches the actor, never the world model.** `rew` and `con` are
  the world model; a wrong map cell there would fabricate imagined reward. The
  map goes through a separate `actor2tensor` that only the actor and critic
  read.
- **A learned gate instead of a warm-up.** The actor's map input is scaled by
  one scalar, initialised at 0, so an untrained RSSM-2 cannot poison the
  policy, and the channel opens as fast as it pays (`map/gate`,
  `memory/gate`).
- **Position is a target, not an input** (map target). RSSM-2 must
  dead-reckon from its own moves; predicting its own coarse cell is what makes
  path integration get learned.
- **No CNN on the crop.** The crop is egocentric, so position is the meaning:
  stone in the top-left means "walk up-left".
- **One step function for acting and training** (`Agent._map_step`). Acting
  calls it once per env step; training scans it over the batch, so both read
  identical states: windows count from the episode start, a step sees only
  the last closed window, a reset clears the state at once
  ([`docs/rssm2-step-alignment.md`](docs/rssm2-step-alignment.md)).
- **Imagination reads the crop frozen at its start.** `rssm.imagine()`
  threads only RSSM-1's state through its scan, so the rollout policy and the
  loss both read the start crop. Sliding it (`imag_shift True`, "Stage 5")
  made the rollout and the loss disagree, and REINFORCE then ran away (policy
  loss -61, entropy 0.005 at 1.1M); it is kept only to reproduce the old runs.

### Targets

| `agent.mapmodel.target` | env flags | RSSM-2 learns to | fed |
|---|---|---|---|
| `map`, honest (default) | `env.craftax.mapmodel True` | predict a 12x12x16 coarse map and its own cell, graded only on cells the agent has seen (`mapknown`), with **hindsight**: an early tick is graded against the mosaic at the end of the episode segment, so it must predict terrain before reaching it | the window's mean feature and summed (dy, dx) moves |
| `map`, privileged | + `env.craftax.map_privileged True` | the same against the true map on every cell (the original runs; use with `agent.mapmodel.hindsight False`) | same |
| `memory` | none | predict RSSM-1's mean latent 1/2/4 ticks ahead and recall the observation 1/2/4 ticks back, within one episode | the window's mean feature and the raw action one-hots |

Honest targets are built from the observation vector and the agent's own
actions only (`craftax_map.ObservedTargets`): terrain is a mosaic of the lit
9x11 windows, and position is dead-reckoned from the spawn. Checked against
the game: 100% position agreement on the surface. The map planes and how each
4x4-tile cell is pooled are documented in `dreamerv3/craftax_map.py`.

### Reward shaping (`env.craftax.survival`)

- `none`: Craftax's own reward.
- `potential`: F = gamma * PHI(s') - PHI(s), with PHI the progress along the
  tech-tree spine plus tools held (`dreamerv3/craftax_potential.py`).
  Potential-based, so it cannot change the optimal policy; PHI is 0 at a real
  death, so a climb is handed back on dying; the start state's PHI is set at
  reset (audit B07).
- `shaped`: five hand-written survival terms (alive, death, restore, kill,
  idle). Each opened a cheaper way to earn than the behaviour it was meant to
  encourage, and the run collapsed (`map_survive`, 2.33). Kept only to
  reproduce that run.

`log/achievements` and `raw_reward` are never shaped, so scores stay
comparable. Write-up of the shaping and recipe fixes:
[`docs/raising-achievements.md`](docs/raising-achievements.md).

## Install

As on `main`:

```bash
pip install -U -r requirements.txt
pip install craftax==1.6.1      # the version every scored run used
python -c "from dreamerv3.craftax_features import validate_against_craftax as v; print(v())"
```

## Run

Every run: `python dreamerv3/main.py --logdir ~/logdir/<run> <flags>`, 1.1M
env steps (the `craftax` preset, replay capped at 4e5). The map target needs
**both** `env.craftax.mapmodel` (the env emits the targets) and
`agent.mapmodel.enabled` (the agent trains RSSM-2).

```bash
PLAIN="--configs craftax size50m --env.craftax.survival none"

# abl_rssm2: plain + RSSM-2 as a memory, no shaping
python dreamerv3/main.py --logdir ~/logdir/abl_rssm2 $PLAIN \
  --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
  --agent.mapmodel.target memory

# honest_map_fix (and honest_map): honest map target + potential shaping
MAP="--configs craftax size50m --env.craftax.mapmodel True \
  --env.craftax.survival potential --env.craftax.map_privileged False \
  --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
  --agent.mapmodel.imag_shift False --agent.mapmodel.hindsight True \
  --agent.imag_length 15"
python dreamerv3/main.py --logdir ~/logdir/honest_map_fix $MAP
```

The older map runs predate the honest targets and the `map_privileged` key;
their targets were the privileged ones, never hindsighted. On today's code
they are:

```bash
OLD="--configs craftax size50m --env.craftax.mapmodel True \
  --env.craftax.map_privileged True --agent.mapmodel.enabled True \
  --agent.mapmodel.to_actor True --agent.mapmodel.hindsight False"
# map_stage5     $OLD --env.craftax.logs True --env.craftax.survival none      --agent.mapmodel.imag_shift True
# map_survive    $OLD --env.craftax.logs True --env.craftax.survival shaped    --agent.mapmodel.imag_shift True
# map_potential  $OLD --env.craftax.survival potential --agent.mapmodel.imag_shift True --agent.imag_length 30
# map_noshift    $OLD --env.craftax.survival none      --agent.mapmodel.imag_shift False
# map_pot_fixed  $OLD --env.craftax.survival potential --agent.mapmodel.imag_shift False
# priv_map       $OLD --env.craftax.survival potential --agent.mapmodel.imag_shift False
```

`map_pot_fixed` and `priv_map` have the same flags on today's code; they
differ by code version (`priv_map` has the potential's recipe and death
fixes, see the CHANGELOG). Each flag set was parse-checked on this branch
(env and agent built, one policy step).

**Reproducibility.** Every run here except `honest_map_fix` and `abl_rssm2`
ran on code older than the 2026-10-11 fixes (RSSM-2 bugs B01-B04, the first
shaped step B07; [`docs/audit-2026-10-11.md`](docs/audit-2026-10-11.md)),
and `map_stage5`, `map_survive`, `map_potential`, `map_noshift` and
`map_pot_fixed` also on code older than the potential's recipe and death
fixes.
No branch reproduces those runs bit for bit: these flags give their
configuration on today's code. `honest_map_fix` ran on `7df2d36` (`~/vfix`)
and `abl_rssm2` on `49a6bc3` (`~/vtmgr`). The CPU equivalence check of this
branch against `49a6bc3` (see the CHANGELOG) gives bit-identical policy
outputs, train outputs and metrics for both, and parameters equal up to the
run-to-run noise of `49a6bc3` itself, except one transient 9.8e-4 difference
in `honest_map_fix` that is not yet explained. `honest_map_truepos` (config = `honest_map`,
never scored) is probably an earlier attempt at `honest_map`.

Disk: a finished run takes ~1.4 GB; `replay.save_skip` keeps the
replay_context latents (`dyn/`, `enc/`, `dec/`, `map/`) off disk.

## Test

```bash
python -m pytest dreamerv3/
python -m pytest embodied/tests/test_replay.py -k save_skip
```

- `test_craftax_map.py`, `test_craftax_potential.py`: the honest and
  privileged map targets and the potential, checked against the game (pure
  numpy; the game checks need Craftax).
- `test_mapmodel_step.py`: RSSM-2's step (all 43 actions, resets, train path
  equals act path, causality, resumed chunks). Needs JAX.
- `test_audit_fixes.py`: fixed evaluation worlds, the first shaped step, raw
  reward, greedy acting. Needs JAX; the env tests need Craftax.

The other tests in `embodied/tests/test_replay.py` call `Replay.dataset`,
which this `Replay` does not have; they fail the same way on `main` and on
the pre-split code.

## Score

As on `main` (`tools/death_eval.py` on the 30 fixed worlds, world hash
`9067302a2635`; `tools/eval_entry.py`; `tools/curves_from_replay.py`;
`tools/plot_training_curves.py`; `tools/make_comparison_report.py`). Score a
run with this branch's flags, or with `--config-from-run`, **from this
branch**: on another branch the map keys would be dropped and the run
rebuilt as a different agent.

```bash
python tools/death_eval.py --logdir ~/logdir/honest_map_fix --episodes 30 $MAP
```

Map-model tools:
- `tools/map_eval.py`: RSSM-2's map belief against the true map, split into
  seen / unseen cells and a per-plane prior floor (reads `map_pred` from
  `policy(mode='probe')`).
- `tools/coverage_audit.py`: how much of the coarse map the agent observes.
- `tools/mapwatch.py`: the latest RSSM-2 and actor metrics of a run, each
  against its chance level, while it trains.
- `tools/make_architecture.py`: draws [`docs/architecture.svg`](docs/architecture.svg)
  / `.pdf`, the model on one page (where the potential sits, what RSSM-2
  reads, every stop-gradient).
- `tools/dump_trajectories.py`: Craftax trajectories stored as (seed,
  actions), from which observations and true maps are regenerated exactly.
- `tools/make_report.py`, `tools/make_onepager.py`: the September map-model
  reports ([`reports/craftax-result-2026-09-20.pdf`](reports/craftax-result-2026-09-20.pdf));
  their inputs were dumped by hand and are not in the repo.

## Results

30 fixed worlds, 1.1M env steps, `size50m`, one training seed each
([`docs/eval_results.json`](docs/eval_results.json),
[`docs/comparison-report.pdf`](docs/comparison-report.pdf),
[`docs/training-curves.png`](docs/training-curves.png)). No run on this
branch uses the rule mask. A 30-world score moves by about ±0.55 between
world sets: a gap under ~1.1 between two single runs is not evidence
([`docs/ablations.md`](docs/ablations.md)).

| run | what | code | achievements ± SE | w/o WAKE_UP | return | wood pickaxe |
|---|---|---|---|---|---|---|
| `honest_map_fix` | honest map + potential | fixed (10-11) | 5.70 ± 0.21 | 5.17 | 2.12% | 7% |
| `abl_rssm2` | plain + RSSM-2 memory | fixed (11-03) | 5.33 ± 0.22 | 4.33 | 1.96% | 0% |
| `honest_map` | honest map + potential | pre-fix | **6.17 ± 0.34** | 5.20 | 2.33% | 40% |
| `priv_map` | privileged map + potential + recipe/death fixes | pre-fix | 6.17 ± 0.23 | 5.20 | 2.33% | 0% |
| `map_pot_fixed` | privileged map + potential | pre-fix | 5.47 ± 0.34 | 4.64 | 2.02% | 33% |
| `map_noshift` | privileged map | pre-fix | 4.73 ± 0.29 | 4.16 | 1.70% | 10% |
| `map_stage5` | privileged map, imag_shift bug | pre-fix | 4.40 ± 0.29 | 3.50 | 1.55% | 0% |
| `map_potential` | + potential, imag_shift bug | pre-fix | 4.10 ± 0.19 | 3.13 | 1.42% | 0% |
| `map_survive` | + 5 hand rewards, imag_shift bug | pre-fix | 2.33 ± 0.19 | 2.33 | 0.63% | 0% |
| `abl_plain` (main) | plain DreamerV3 | fixed | 4.77 ± 0.28 | 3.80 | 1.71% | 3% |
| `vanilla` (main) | plain DreamerV3 | Sept. | 4.63 ± 0.22 | 4.63 | 1.65% | 23% |

What the numbers say:

- **The map with the potential is the best of this family,** 6.17 on the old
  code and 5.70 rerun with every fix (one seed each; the gap is within
  noise). On the fixed code it leads vanilla by +1.07 ± 0.24, paired, but the
  potential is part of that.
- **Observation-only targets cost nothing:** the honest map matches the
  privileged control (6.17 each) and makes the wood pickaxe where the control
  never does.
- **The potential, not the map, moves the tech tree:** the map alone is level
  with plain (4.73 vs 4.77); the potential lifts the pickaxe from 10% to
  33-40% of episodes among the map runs.
- **RSSM-2 as a memory alone is not established:** +0.57 ± 0.31 over plain on
  the test worlds, but slightly below it in training (5.18 vs 5.43).
- **Two failures, kept on record:** sliding the crop through imagination
  (`imag_shift`) and the hand-written survival terms (`shaped`).
- WAKE_UP (sleeping through the night) is a large share of every recent
  score, plain included.
- `map_survive`'s training curve stops at 0.5M steps, where its replay
  record ends.

The goal managers and the rule mask that build on RSSM-2 (8-11 achievements)
are on `family/v`, `family/next` and `family/masked`; `main` indexes every
family.

## Files

| path | what |
|---|---|
| `dreamerv3/mapmodel.py` | RSSM-2: the GRU, map decoder, gate, loss, crop, dead reckoning, action deltas |
| `dreamerv3/craftax_map.py` | coarse-map targets: honest (`ObservedTargets`) and privileged; pure numpy |
| `dreamerv3/craftax_potential.py` | the tech-tree potential; pure numpy |
| `dreamerv3/agent.py` | wiring: RSSM-2 carry and step, map losses, actor input, imagination, replay context |
| `embodied/envs/craftax.py` | emits `map12`, `mappos`, `mapseen`, `mapknown`; reward shaping |
| `docs/rssm2-step-alignment.md` | the 2026-10-11 RSSM-2 fixes |
| `docs/raising-achievements.md` | the potential shaping, the recipe and death fixes, the imag_shift bug |
| `docs/ablations.md` | plain vs RSSM-2 alone, and scoring noise |
| `docs/audit-2026-10-11.md` | the outside audit, annotated for this branch |

## Credit

DreamerV3 is by Danijar Hafner, Jurgis Pasukonis, Jimmy Ba and Timothy
Lillicrap; this fork keeps the original MIT licence. Craftax is by Michael
Matthews et al.
