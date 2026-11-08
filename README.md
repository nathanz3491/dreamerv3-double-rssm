# DreamerV3 on Craftax: the goal-manager family (`family/v`)

A two-level agent on top of [DreamerV3][upstream] for symbolic
[Craftax][craftax]: a **manager** picks a goal, the **actor** pursues it, and
both are trained in imagination like the rest of Dreamer. This branch holds
manager versions v1, v1.1, v1.2, v1.3 and v2, with what they need: RSSM-2 (a
second, slower world model the manager reads), named and learned goals,
curiosity, potential reward shaping, and the rule mask (off by default).

This branch is `main` plus this family. The plain baseline, the shared
fixes and the index of all families are on the `main` branch (its
`README.md`). The original DreamerV3 README is kept at
[`README-dreamerv3.md`](README-dreamerv3.md).

**The rule mask counts as cheating.** `agent.valid.mask` (experiment B2)
removes actions that the game's hand-written rules say can do nothing. Most of
this family's headline runs used it. It is off by default and kept only so
those runs can be reproduced; masked results are marked as such and listed
apart from the honest ones.

[upstream]: https://github.com/danijar/dreamerv3
[craftax]: https://github.com/MichaelTMatthews/Craftax

## Design

Every 8-step tick, **RSSM-2** (`dreamerv3/mapmodel.py`) takes RSSM-1's mean
latent and the window's actions and updates a slow state. The manager reads
RSSM-1's state, RSSM-2's state and which goals were reached this episode, and
picks a goal; the actor sees the goal and its step within the goal. RSSM-2
has two targets:
- `map` (v1-v1.3): a coarse 12x12 map of the level, built only from what the
  agent has seen (`craftax_map.ObservedTargets`), graded in hindsight;
- `memory` (v2): predict RSSM-1's latent 1/2/4 ticks ahead and recall the
  observation 1/2/4 ticks back, from raw actions, with no game knowledge.

| version | what changes | flags on top of the previous row |
|---|---|---|
| v1 | manager over 13 hand-written tech-tree goals (`craftax_goals.py`), a new goal every 8 steps; the actor is paid the change in the goal's predicted progress (`gphi` head); the manager has its own critic | `--env.craftax.goals_obs True --agent.manager.enabled True` |
| v1.1 | reaching the goal pays both levels: actor +1 (`reach_bonus`), manager +0.5 the first time in an episode (`mgr_bonus`) | none (the defaults) |
| v1.2 | one two-headed critic (`game` for both actors, `goal` for the actor only); the manager takes the exact gradient over all goals from the game head; goals held until reached or 32 steps; imagination resumes the replay's goal | `--agent.manager.critic shared --agent.manager.hold 32` |
| v1.3 | DRINK and EAT goals, and food and drink in the potential | `--env.craftax.survival potential+meters --env.craftax.goals_survival True` |
| v2 | learned goals: a codebook of 16 kinds of latent change (`goalcodes.py`); curiosity (ensemble disagreement) as a third reward stream instead of the potentials; RSSM-2 as a memory instead of a map | see [Run](#run) |

Design notes: [`docs/design-manager.md`](docs/design-manager.md) (v1-v1.3),
[`docs/design-v2.md`](docs/design-v2.md) and
[`docs/architecture-v2.svg`](docs/architecture-v2.svg) (v2),
[`docs/rssm2-step-alignment.md`](docs/rssm2-step-alignment.md) (how RSSM-2's
training matches acting).

Game knowledge, by flag:
- `env.craftax.valid_obs` + `agent.valid.mask` (`craftax_valid.py`): the
  action rules. **Cheating**; default off.
- `env.craftax.goals_obs` (`craftax_goals.py`): the named goal list and its
  progress, read from the observation. A target and a manager input, never an
  encoder input. Disclosed game knowledge, like the tech-tree potential.
- `env.craftax.survival potential` / `potential+meters`
  (`craftax_potential.py`): potential-based shaping over the tech tree (and
  food and drink). It changes `reward` only; `raw_reward` and the scores use
  Craftax's own reward.
- `agent.mapmodel.target map` + `env.craftax.mapmodel`: map labels with
  hand-written block groupings, built from observed cells only.

Not on this branch: taste (`family/taste`), the B1 input flags and the
learned mask B3 (`family/masked`), the five-term `shaped` survival reward
(`family/rssm2`), and the reward-cause head (tag `archive/full-2026-11-08`).
Their config keys are gone too; this family's runs left them all at their
defaults.

## Install

```bash
pip install -U -r requirements.txt
pip install craftax==1.6.1      # the version every scored run used
python -c "from dreamerv3.craftax_features import validate_against_craftax as v; print(v())"
```

## Run

Every run: `python dreamerv3/main.py --logdir ~/logdir/<run> <flags>`, 1.1M
env steps on one GPU (about 15 GPU-hours at `size50m`). The flags are exactly
those of the original runs:

```bash
# Shared prefixes
MAP="--configs craftax size50m --env.craftax.mapmodel True --env.craftax.survival potential --env.craftax.map_privileged False --agent.mapmodel.enabled True --agent.mapmodel.to_actor True --agent.mapmodel.imag_shift False --agent.mapmodel.hindsight True --agent.imag_length 15"
B2="$MAP --env.craftax.valid_obs True --agent.valid.mask True"     # rule mask: cheating
PLAIN="--configs craftax size50m --env.craftax.survival none"

# Honest runs (no rule mask)
V12_NOMASK="$MAP --env.craftax.goals_obs True --agent.manager.enabled True --agent.manager.critic shared --agent.manager.hold 32"
MANAGER="$PLAIN --agent.mapmodel.enabled True --agent.mapmodel.to_actor True --agent.mapmodel.target memory --env.craftax.goals_obs True --agent.manager.enabled True --agent.manager.critic shared --agent.manager.hold 32 --agent.manager.goals named"
NORULES="$PLAIN --env.craftax.valid_obs False --agent.mapmodel.enabled True --agent.mapmodel.to_actor True --agent.mapmodel.target memory --agent.valid.mask False --agent.manager.enabled True --agent.manager.critic shared --agent.manager.hold 32 --agent.manager.goals learned --agent.curiosity.enabled True"

# Masked runs (counted as cheating)
V1="$B2 --env.craftax.goals_obs True --agent.manager.enabled True"
V12="$V1 --agent.manager.critic shared --agent.manager.hold 32"
V13="$V12 --env.craftax.survival potential+meters --env.craftax.goals_survival True"
V2="$PLAIN --env.craftax.valid_obs True --agent.mapmodel.enabled True --agent.mapmodel.to_actor True --agent.mapmodel.target memory --agent.valid.mask True --agent.manager.enabled True --agent.manager.critic shared --agent.manager.hold 32 --agent.manager.goals learned --agent.curiosity.enabled True"
V2CUR="$V2 --agent.curiosity.normalize True --agent.curiosity.weight 0.1 --agent.curiosity.mgr_weight 0.1"

python dreamerv3/main.py --logdir ~/logdir/v12_nomask $V12_NOMASK
```

| run | flags | seeds |
|---|---|---|
| `abl_manager`, `abl_manager_s1` | `$MANAGER` | `--seed 0`, `--seed 1` |
| `v12_nomask` | `$V12_NOMASK` | 0 |
| `v2_norules` | `$NORULES` | 0 |
| `mgr` (v1), `mgr2` (v1.1) | `$V1` | 0 |
| `mgr3` (v1.2), `mgr3_fix`, `_s1`, `_s2` | `$V12` | 0 / 0, 1, 2 |
| `mgr4` (v1.3) | `$V13` | 0 |
| `v2`, `v2_fix`, `_s1`, `_s2` | `$V2` | 0 / 0, 1, 2 |
| `v2cur` | `$V2CUR` | 0 |

`mgr` and `mgr2` share their flags: v1's code had no reach bonuses, and v1.1
added them as defaults. On this branch `$V1` therefore trains v1.1. v1 itself
cannot be rebuilt by flags: v1.1 also gave the manager the
reached-this-episode flags and blocked goals that already hold.
`--agent.manager.reach_bonus 0 --agent.manager.mgr_bonus 0` restores only
v1's rewards.

**Which runs this code reproduces.** The code is the pre-split tree
(`49a6bc3`) restricted to this family. The `*_fix` runs, `v12_nomask`
(`~/vfix`, commit `7df2d36`), `abl_manager` (`~/vtmgr` = `49a6bc3`) and
`v2_norules` (`~/vtaste`, `cbec67b`, taste off) were trained on code with the
2026-10-11 fixes, which this branch keeps. `mgr`-`mgr4`, `v2` and `v2cur`
were trained before 2026-10-11 on code with three RSSM-2 bugs (a step DOWN for
every non-move action, episodes leaking into the next, training seeing up to
7 future steps; [`docs/audit-2026-10-11.md`](docs/audit-2026-10-11.md)). No
branch reproduces those runs exactly; the same flags here train the fixed
version.

## Test

```bash
python -m pytest dreamerv3/
python -m pytest embodied/tests/test_replay.py -k save_skip
```

- `test_craftax_goals`, `test_craftax_map`, `test_craftax_potential`,
  `test_craftax_valid`: the env-side features against the game itself (the
  game comparisons need Craftax and are skipped without it).
- `test_mapmodel_step`: RSSM-2's step (all 43 actions, resets mid-window,
  the training path equal to the acting path, causality).
- `test_audit_fixes`: fixed evaluation worlds, the first shaped step, raw
  reward, the goal codebook on empty batches, goal starts at replay
  boundaries, greedy acting.

The other tests in `embodied/tests/test_replay.py` (53) call
`Replay.dataset`, which this `Replay` does not have, and two in
`embodied/tests/test_driver.py` fail with a `KeyError`; all of them fail the
same way on the pre-split code. `embodied/tests/test_parallel.py` and
`test_train.py` need the `zerofun` package (upstream). The whole suite
otherwise passes (259 tests, CPU, Craftax installed).

## Score

Scoring is `main`'s (fixed worlds, world hash `9067302a2635` for the default
seeds); run it from this branch so the agent is rebuilt with this family's
code:

```bash
python tools/death_eval.py --logdir ~/logdir/v12_nomask --episodes 30 --config-from-run > v12_nomask.log
python tools/eval_entry.py v12_nomask.log --id v12_nomask --label "v1.2 fixed, no mask" --short "v1.2 no mask"
python tools/curves_from_replay.py ~/logdir/v12_nomask --out docs/curves
python tools/plot_training_curves.py      # -> docs/training-curves.*
python tools/make_comparison_report.py    # -> docs/comparison-report.pdf
```

`--greedy` takes the argmax action and goal. A 30-world score moves by about
±0.55 between world sets: a gap under ~1.1 between two single runs is not
evidence ([`docs/ablations.md`](docs/ablations.md)).

## Results

30 fixed worlds, 1.1M env steps, `size50m`
([`docs/eval_results.json`](docs/eval_results.json),
[`docs/comparison-report.pdf`](docs/comparison-report.pdf),
[`docs/training-curves.png`](docs/training-curves.png)). "training" is the
mean achievements per training episode over the last 100k steps.

**Honest runs (no rule mask):**

| run | what | test achievements ± SE | return | training |
|---|---|---|---|---|
| `vanilla` | DreamerV3, old code (`main`) | 4.63 ± 0.22 | 1.65% | 4.17 |
| `abl_plain` | plain DreamerV3 (`main`) | 4.77 ± 0.28 | 1.71% | 5.43 |
| `abl_manager` | plain + RSSM-2 memory + v1.2 manager | 5.77 ± 0.24 | 2.15% | 6.01 |
| `abl_manager_s1` | same, seed 1 | 5.30 ± 0.23 | 1.95% | 5.05 |
| `v12_nomask` | fixed v1.2 without the mask | **6.03 ± 0.33** (60 worlds: 6.13 ± 0.22) | 2.27% | 5.75 |
| `v2_norules` | fixed v2 without the mask | not scored | | 4.78 |

**Masked runs (counted as cheating):**

| run | what | test achievements ± SE | return | training |
|---|---|---|---|---|
| `mgr` | v1, old code | 7.33 ± 0.53 | 2.85% | 7.48 |
| `mgr2` | v1.1, old code | 9.40 ± 0.54 | 3.76% | 8.81 |
| `mgr3` | v1.2, old code | 9.97 ± 0.36 (60 worlds: 9.73) | 4.04% | 9.85 |
| `mgr4` | v1.3, old code | 9.17 ± 0.49 | 3.66% | 9.44 |
| `mgr3_fix` ×3 | v1.2, fixed code | 8.77 / 8.83 / 8.73 (60 worlds: 9.03) | 3.48% | 8.96 / 8.36 / 8.94 |
| `v2` | v2, old code | 6.93 ± 0.36 | 2.67% | 7.12 |
| `v2cur` | v2 + scaled curiosity, old code | 6.53 ± 0.22 | 2.50% | 6.88 |
| `v2_fix` ×3 | v2, fixed code | 7.17 / 7.83 / 6.07 | 2.77% | 7.93 / 8.12 / 6.20 |

What they say:
- **Without the mask the manager adds little.** Over two seeds the v1.2
  manager is +0.77 ± 0.22 over plain but +0.20 ± 0.24 over RSSM-2 alone
  (`abl_rssm2`, 5.33, on `family/rssm2`), and its second seed ended training
  below plain. Fixed v1.2 without the mask is +0.33 ± 0.40 over the honest
  map underneath it. No honest run of this family makes a stone tool.
- **The masked scores are largely the mask's.** Removing it costs fixed v1.2
  2.90 ± 0.26 achievements over 60 worlds, and v2 about 2.6 in training.
- **Old v1.2's 9.97 was partly world luck** (9.73 over 60 worlds) and partly
  a better agent than its fixed reruns, possibly helped by the pre-fix RSSM-2
  leak. One old seed cannot tell ([`docs/ablations.md`](docs/ablations.md)).
- The next version of the manager (v1.2's named goals on v2) is on
  `family/next`.

## Files

- `dreamerv3/agent.py`: the agent; family code behind `mapmodel.*`,
  `manager.*`, `curiosity.*` and `valid.mask`, all off by default (then it
  is `main`'s plain agent).
- `dreamerv3/mapmodel.py` (RSSM-2), `craftax_map.py` (map targets),
  `craftax_potential.py` (shaping), `craftax_goals.py` (named goals),
  `goalcodes.py` (learned goals), `craftax_valid.py` (rule flags).
- `embodied/envs/craftax.py`: the adapter with `mapmodel`, `valid_obs`,
  `goals_obs`/`goals_survival` and `survival` options.
- `docs/`: design notes, ablations, audit, results, curves.

History up to the split: `CHANGELOG.md` at tag `archive/full-2026-11-08`;
this branch's changes: [`CHANGELOG.md`](CHANGELOG.md).
