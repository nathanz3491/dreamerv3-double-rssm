# What each addition contributes, and how noisy our scores are

Updated 2026-11-03. Training ran on the box with `~/run_ablq.sh`, code
`~/vtmgr` (branch `feat/taste-manager`), size50m, 1.1M env steps, one seed
each.

## The ablation runs

Each run is plain DreamerV3 plus one addition. None of them has the rule mask,
survival shaping or curiosity.

| run | addition |
|---|---|
| `abl_plain` | none |
| `abl_rssm2` | RSSM-2 as a memory (`mapmodel.target memory`, the v2 mode) |
| `abl_manager` | RSSM-2 + the v1.2 manager: 13 named goals, shared critic, hold 32 |
| `abl_taste` | taste over actions: centered evidence, pos = neg = 1.43, w0 0 |

The manager reads RSSM-2, so it cannot run without it. Its own share is
`abl_manager` − `abl_rssm2`.

**Test scores** (30 fixed worlds, `tools/death_eval.py`, ± SE over the worlds;
the gap columns are paired by world):

| run | test achievements | return | vs plain | training, last 100k steps |
|---|---|---|---|---|
| plain | 4.77 ± 0.28 | 1.71% | — | 5.43 |
| + RSSM-2 | 5.33 ± 0.22 | 1.96% | +0.57 ± 0.31 | 5.18 |
| + RSSM-2 + manager | **5.77 ± 0.24** | 2.15% | **+1.00 ± 0.35** | **6.01** |
| + taste | 4.63 ± 0.29 | 1.65% | −0.13 ± 0.27 | 5.19 |
| vanilla (old code, Sept.) | 4.63 ± 0.22 | 1.65% | −0.13 ± 0.31 | 4.17 |

**What the numbers say:**
- **The manager is the only addition that comes out ahead in both measures:**
  +1.00 on the test worlds and +0.6 in training. The manager alone, compared
  with RSSM-2 alone, is +0.43 ± 0.29 on the test worlds. It learns more slowly
  at first (behind plain until about 800k steps), then is still climbing at
  the end. The second seed did not repeat it (see "Second seed of the
  manager ablation" below).
- **RSSM-2 alone:** the two measures disagree. It is +0.57 on the test worlds
  but −0.25 in training. That disagreement is what noise looks like (see
  below). No effect is established.
- **Taste over actions: no effect** in either measure.
- **Plain DreamerV3 on today's code is about the old vanilla on the test
  worlds**, though it trains higher: 5.43 against 4.17 over the last 100k
  training steps. `abl_plain_s1` is also queued.
- **None of these comes close to the rule mask.** B2 fixed scores 8.33 on the
  test worlds (three-seed mean). The mask adds +2.6 to the honest map (test
  worlds) and +2.6 to v2 (v2-norules → v2 fixed, end of training).

**v1.2 is not "RSSM-2 + manager".** It is B2 + the manager, and B2 brings
three things that `abl_manager` lacks:
- the rule mask;
- RSSM-2 in map mode (map labels from the observation, hindsight) instead of
  memory mode;
- the tech-tree potential shaping (`survival potential`).

## What the mask is worth under the v1.2 manager (`v12_nomask`, 2026-11-05)

`v12_nomask` is fixed v1.2 with the `~/vfix` code and flags of its three
seeds, minus `valid.mask` and `valid_obs`. One seed.

| run | test, 60 worlds | training, 1.08M steps |
|---|---|---|
| v1.2 fixed (3 seeds) | 9.03 ± 0.22 | 8.74 |
| B2 fixed (3 seeds) | 8.53 ± 0.24 | 8.79 |
| **v1.2 without the mask** | **6.13 ± 0.22** | **5.90** |
| honest map fixed (worlds 0–29) | 5.70 ± 0.21 | 5.60 |

- **The mask is worth +2.90 ± 0.26 under the v1.2 manager** (paired over
  60 worlds), and +2.8 in training. That is the same as without a manager
  (+2.6), so the manager does not reduce the need for the mask.
- Without the mask, v1.2 is +0.33 ± 0.40 over the honest map (worlds 0–29)
  and +0.3 in training: the manager and the shaping on top of the map
  add little.

## Second seed of the manager ablation

`abl_manager_s1` ends training at 5.03, below plain (5.41), against 5.78 for
seed 0. On worlds 0–29 it scores 5.30 ± 0.23 (seed 0: 5.77).

| two-seed mean, worlds 0–29 | gap |
|---|---|
| abl_manager 5.53 ± 0.16 − abl_plain (s0) | +0.77 ± 0.28 |
| abl_manager 5.53 ± 0.16 − abl_rssm2 | +0.20 ± 0.24 |

The manager's own share (over RSSM-2 alone) is not established. Its lead
over plain on the test worlds is smaller with two seeds, and gone in
training.

`abl_plain_s1` scores 4.77 on worlds 0–29, exactly seed 0's score. With two
seeds each, the manager is +0.77 ± 0.22 over plain and RSSM-2 alone is
+0.57 ± 0.26.

## Scoring noise and training luck (the v1.2 question)

To measure noise directly, old v1.2 and the three fixed seeds each of v1.2
and B2 were scored again on a second set of 30 worlds (seeds 30–59, world
hash 9cda959f2838).

| checkpoint | worlds 0–29 | worlds 30–59 | change | 60 worlds | training, last 100k |
|---|---|---|---|---|---|
| old v1.2 (`mgr3`) | 9.97 | 9.50 | −0.47 | 9.73 ± 0.29 | 9.85 |
| v1.2 fixed s0 | 8.77 | 8.90 | +0.13 | 8.83 ± 0.35 | 8.96 |
| v1.2 fixed s1 | 8.83 | 9.50 | +0.67 | 9.17 ± 0.25 | 8.36 |
| v1.2 fixed s2 | 8.73 | 9.47 | +0.73 | 9.10 ± 0.34 | 8.94 |
| B2 fixed s0 | 8.27 | 9.17 | +0.90 | 8.72 ± 0.35 | 9.17 |
| B2 fixed s1 | 7.70 | 8.90 | +1.20 | 8.30 ± 0.40 | 8.59 |
| B2 fixed s2 | 9.03 | 8.13 | −0.90 | 8.58 ± 0.32 | 8.68 |

**Scoring noise is about ±0.55 for one 30-world score.** The same
checkpoint moved by 0.8 on average between the two world sets (SD 0.77,
divided by √2). It is larger than the per-world SE (about 0.4) suggests, so:
- a gap under about 1.1 between two single runs, each scored on 30 worlds,
  is not evidence;
- per-world SEs understate the noise, so paired gaps need about 1.5 times the
  usual margin.

Most of the noise comes from one episode differing from the next, not from
which worlds were drawn: the variance per episode is about 4.5 to 7, against
about 0.2 to 2 for the world.

**v1.2 against B2** (fixed code, three seeds each):
- 60 worlds: v1.2 9.03, B2 8.53, a paired gap of +0.50 ± 0.25;
- the three seed means: v1.2 8.83 / 9.17 / 9.10, B2 8.72 / 8.30 / 8.58;
- but over the last 100k training steps (about 360 episodes per run) they
  are level: 8.75 against 8.81.

A lead of about half an achievement, at best, and only on the test worlds.
The 30-world numbers that made v1.2 look uniquely consistent (8.77 / 8.83 /
8.73) were chance: on the new worlds its seeds spread 8.90 to 9.50.

**Old v1.2's record (9.97) was partly world luck and partly a better agent.**
- On the new worlds it scores 9.50. Over 60 worlds that is 9.73, +0.70 ± 0.32
  above the fixed seeds' mean.
- In training it was +1.1 above them (9.85 against 8.75).
- So the old agent really was better. Its first 30 worlds added about +0.25
  of luck on top.

With one old seed, the cause of the real part is open:
- **Luck:** about 3 times the training-seed spread of the fixed v1.2 seeds.
- **The pre-fix RSSM-2 leak:** training saw up to 7 future steps, and the
  manager plans from those states. The old B2, which had the same leak but no
  manager, trained 0.9 *below* its fixed reruns. That fits "the manager
  exploited it" better than "the bug helped everyone".

A second seed of the old code would settle it (about 15 GPU-hours).
