# RSSM-2 alone on plain DreamerV3, and how noisy our scores are

Updated 2026-11-08 (the runs are from 2026-11-03/05). Training ran on the box
with `~/run_ablq.sh`, code `~/vtmgr` (`49a6bc3`), size50m, 1.1M env steps.
The full ablation set (plain, + RSSM-2, + RSSM-2 + the v1.2 manager, + taste)
is at tag `archive/full-2026-11-08`; the manager and taste rows are on their
own family branches.

## The runs

Plain DreamerV3 with and without one addition. Neither has the rule mask,
survival shaping or curiosity.

| run | addition | flags beyond `--configs craftax size50m --env.craftax.survival none` |
|---|---|---|
| `abl_plain` (on `main`) | none | — |
| `abl_rssm2` | RSSM-2 as a memory | `--agent.mapmodel.enabled True --agent.mapmodel.to_actor True --agent.mapmodel.target memory` |

**Test scores** (30 fixed worlds, `tools/death_eval.py`, ± SE over the worlds;
the gap is paired by world):

| run | test achievements | return | vs plain | training, last 100k steps |
|---|---|---|---|---|
| plain | 4.77 ± 0.28 | 1.71% | — | 5.43 |
| + RSSM-2 | 5.33 ± 0.22 | 1.96% | +0.57 ± 0.31 | 5.18 |
| vanilla (old code, Sept.) | 4.63 ± 0.22 | 1.65% | −0.13 ± 0.31 | 4.17 |

- **RSSM-2 alone: no effect established.** It is +0.57 on the test worlds
  but −0.25 in training. That disagreement is what noise looks like (below).
  Against plain's two seeds (`abl_plain_s1` also scores 4.77) the gap is
  +0.57 ± 0.26.
- **Plain DreamerV3 on today's code is about the old vanilla on the test
  worlds**, though it trains higher (5.43 against 4.17 over the last 100k
  training steps).
- The map runs on this branch differ from `abl_rssm2` in three ways: the map
  target (hand-built labels, hindsight) instead of the memory target, the
  tech-tree potential, and (all but `honest_map_fix`) the pre-fix RSSM-2. The
  potential without RSSM-2 has not been run, so the two are not separated.

## Scoring noise

To measure noise directly, seven checkpoints from other families (old v1.2
and three fixed seeds each of v1.2 and B2) were scored again on a second set
of 30 worlds (seeds 30-59, world hash `9cda959f2838`); the table is in
`docs/ablations.md` at tag `archive/full-2026-11-08`.

**Scoring noise is about ±0.55 for one 30-world score.** The same
checkpoint moved by 0.8 on average between the two world sets (SD 0.77,
divided by √2). That is larger than the per-world SE (about 0.4) suggests, so:
- a gap under about 1.1 between two single runs, each scored on 30 worlds,
  is not evidence;
- per-world SEs understate the noise, so paired gaps need about 1.5 times the
  usual margin.

Most of the noise comes from one episode differing from the next, not from
which worlds were drawn: the variance per episode is about 4.5 to 7, against
about 0.2 to 2 for the world.
