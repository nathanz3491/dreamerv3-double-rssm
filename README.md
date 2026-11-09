# DreamerV3 on Craftax: the next agent (`family/next`)

This branch holds one agent family on top of plain [DreamerV3][upstream] on
[Craftax][craftax]: the **next agent**. It is `main` (plain DreamerV3, the
Craftax adapter, the scoring tools; see `main`'s README) plus this
family's code, config, tests and docs. Every other family lives on a branch of
its own, indexed in `main`'s README. The original DreamerV3 README is
[`README-dreamerv3.md`](README-dreamerv3.md).

[upstream]: https://github.com/danijar/dreamerv3
[craftax]: https://github.com/MichaelTMatthews/Craftax

## The next agent

The next agent is the goal manager of v1.2 on top of the parts of v2:

- **A manager over 13 named goals** (NONE plus 12 tech-tree milestones, from
  `env.craftax.goals_obs`) picks a goal every time the last one is reached or
  has been held for 32 steps. The bottom actor sees the goal and is paid for
  progress toward it. ([`docs/design-manager.md`](docs/design-manager.md))
- **RSSM-2 as a memory:** a second, slower recurrent model that predicts the
  future and recalls the past from raw actions, with no game labels; its
  state is read by the actor, the critic and the manager. Its train and act
  paths run the same step function
  ([`docs/design-v2.md`](docs/design-v2.md),
  [`docs/rssm2-step-alignment.md`](docs/rssm2-step-alignment.md)).
- **Curiosity:** the disagreement of an ensemble that predicts the next latent
  is a third reward stream.
- **One shared critic** with a game head, a goal head and an explore head.
- **Taste** (`agent.taste`) is in the code only so the flags of `next_lam0` and
  `next_nomask` still run: they set `lam_mgr 0.0`, which adds exactly zero to
  the manager's values. Taste itself is studied on `family/taste`; why it hurt
  is summarised in [`docs/taste-on-manager.md`](docs/taste-on-manager.md).

**The rule mask counts as cheating.** The B2 mask (`env.craftax.valid_obs` and
`agent.valid.mask`) reads the game's action rules, so a run with it is not an
honest agent; rule flags as an input would count as rule knowledge too. Its
code stays here, default off, because `next_lam0` was trained with it. **The
honest run is `next_nomask`** (mask off); results below list it first, and
masked results are marked and kept out of any headline ranking.

Defaults differ from `main` and from the pre-split code, and this branch
accepts only the values the family used (the assertions name the branch that
holds the others): `manager.critic shared`, `manager.hold 32`,
`manager.goals named`, `taste.target manager`, `mapmodel.target memory`.

## Install

```bash
pip install -U -r requirements.txt
pip install craftax==1.6.1      # the version every scored run used
python -c "from dreamerv3.craftax_features import validate_against_craftax as v; print(v())"
```

The last line checks that the achievement names embedded in
`dreamerv3/craftax_features.py` match the installed Craftax.

## Run

Every command is prefixed by
`python dreamerv3/main.py --logdir ~/logdir/<run> --configs craftax size50m`.
The plain baseline is `--env.craftax.survival none` (the default; kept so old
commands parse). Call that **PLAIN**; it is also what `main` trains, so
`abl_plain` is reproduced from `main`.

**NEXT** (= `next_notaste`) is PLAIN plus:

```
--env.craftax.valid_obs True --env.craftax.goals_obs True
--agent.mapmodel.enabled True --agent.mapmodel.to_actor True
--agent.mapmodel.target memory --agent.valid.mask True
--agent.manager.enabled True --agent.manager.critic shared
--agent.manager.hold 32 --agent.manager.goals named
--agent.curiosity.enabled True
```

1. **`next_nomask` (honest: no rule mask).** The `next_lam0` flags with
   `--agent.valid.mask False`; `valid_obs` stays True, so `obs['valid']`
   reaches nothing but replay and the run matches `next_lam0` everywhere else.
   Launcher on the box: `~/run_nomask.sh`.

   ```bash
   python dreamerv3/main.py --logdir ~/logdir/next_nomask \
     --configs craftax size50m --env.craftax.survival none \
     --env.craftax.valid_obs True --env.craftax.goals_obs True \
     --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
     --agent.mapmodel.target memory --agent.valid.mask False \
     --agent.manager.enabled True --agent.manager.critic shared \
     --agent.manager.hold 32 --agent.manager.goals named \
     --agent.curiosity.enabled True \
     --agent.taste.enabled True --agent.taste.target manager \
     --agent.taste.center True --agent.taste.pos 1.43 --agent.taste.neg 1.43 \
     --agent.taste.lam_mgr 0.0
   ```

2. **`next_lam0` (masked: counted as cheating).** The same with
   `--agent.valid.mask True` (taste at `lam_mgr 0.0`: no taste).
3. **`next_notaste` (masked; crashed, no result).** NEXT with no taste flags.
   It crashed three times on GPU faults (Xid 69 and 13; logdirs `next_notaste`,
   `next_notaste_crash1`, `_crash2`); `next_lam0` stands in for it, since
   `lam_mgr 0` adds exactly zero.

All of these were trained with the code directory `~/vtmgr` on the GPU box,
which is the pre-split code (commit `49a6bc3`, tag `archive/full-2026-11-08`).

A finished run takes ~1.4 GB: ~0.8 GB of replay and a ~0.7 GB checkpoint.
`replay.save_skip` keeps the replay_context latents (`dyn/`, `enc/`, `dec/`
and RSSM-2's `map/`) off disk; a resumed run gets zeros back and refreshes
them as it trains. `python tools/strip_replay.py ~/logdir/<run>` shrinks an
older run (it refuses a live or unfinished one, and a stripped run can no
longer be resumed).

## Test

```bash
python -m pytest dreamerv3/
python -m pytest embodied/tests/test_replay.py -k save_skip
```

`test_audit_fixes.py`, `test_taste.py` and `test_mapmodel_step.py` need JAX;
the craftax tests (`test_craftax_{features,goals,map,potential,valid}.py` and
the env tests) are skipped without Craftax. The other tests in
`embodied/tests/test_replay.py` call `Replay.dataset`, which this `Replay` does
not have: 53 of them fail with `'Replay' object has no attribute 'dataset'`,
and they fail the same way at `49a6bc3`.

## Score

Every result is scored on the same 30 fixed worlds (episode *i* is built from
seed *i*; world hash `9067302a2635` for the default seeds), so comparisons are
paired world by world.

```bash
# 1. Play 30 episodes; prints achievements, unlock rates, deaths, raw return
#    and the world hash, which must match across the runs being compared.
#    Rebuild the agent from the run's own config.yaml:
python tools/death_eval.py --logdir ~/logdir/next_nomask --episodes 30 --config-from-run > next_nomask.log
#    (60 worlds: --episodes 60)

# 2. Turn the log into an entry for docs/eval_results.json
python tools/eval_entry.py next_nomask.log --id next_nomask --label "Next, no mask" --short next-nomask

# 3. Curves, figure and report
python tools/curves_from_replay.py ~/logdir/next_nomask --out docs/curves
python tools/plot_training_curves.py                         # -> docs/training-curves.*
python tools/make_comparison_report.py                       # -> docs/comparison-report.pdf
```

`--config-from-run` drops keys that this branch's `configs.yaml` lacks, so a
run of another family would be rebuilt as a different agent. **Score each run
from the branch it belongs to**: runs of this family from here.
`tools/plot_training_curves.py` lists the curves it draws; add a run there when
its curve is in `docs/curves/`.

A 30-world score moves by about ±0.55 between world sets: a gap under ~1.1
between two single runs is not evidence. Use 60 worlds and several seeds.

## Results

30 fixed worlds, 1.1M env steps, `size50m`
([`docs/eval_results.json`](docs/eval_results.json),
[`docs/comparison-report.pdf`](docs/comparison-report.pdf),
[`docs/training-curves.png`](docs/training-curves.png)). Honest runs first.

| run | rule mask | achievements ± SE | 60 worlds | status |
|---|---|---|---|---|
| `next_nomask` | no (honest) | 5.63 ± 0.27 | 5.82 ± 0.17 | run 2026-11-09 00:22–17:42 CST (`~/run_market.sh`); level with `v12_nomask`, ~1 above plain |
| `vanilla` | no | 4.63 ± 0.22 | | original code, before the 2026-10 fixes |
| `abl_plain` | no | 4.77 ± 0.28 | | `main`'s plain agent (seed 1 also 4.77) |
| `next_lam0` | **masked: counted as cheating** | 11.20 ± 0.41 | 11.30 ± 0.27 | best run of the project, not a headline result |
| `next_notaste` | masked | none | | crashed three times on GPU faults; `next_lam0` stands in |

`next_nomask` is the honest measure of the next agent. On 60 worlds it scores
5.82 ± 0.17: level with `v12_nomask` (6.13 on the same 60 worlds), about one
achievement above plain (4.77), and 5.5 below `next_lam0`. Without the mask the
next agent's parts add nothing over manager v1.2. Its lives are short (mean 249
steps; 17 of 30 deaths from thirst).

Comparison points from other families (their runs and docs are on their own
branches; 30 worlds unless noted; a single-run gap under ~1.1 is not
evidence):

| run | branch | rule mask | achievements |
|---|---|---|---|
| `v12_nomask`: manager v1.2 without the mask | `family/v` | no | 6.03 ± 0.33 |
| `abl_manager`: RSSM-2 + the v1.2 manager, no curiosity, no mask | `family/v` | no | 5.77 |
| `abl_rssm2`: RSSM-2 alone | `family/rssm2` | no | 5.33 |
| `mgr3_fix`: manager v1.2 | `family/v` | masked | 8.77 |
| `next_taste`: the next agent with taste on the manager | `family/taste` | masked | 9.80 ± 0.42 (60 worlds: 9.38) |

Taste hurt: `next_lam0` leads `next_taste` by +1.92 ± 0.41 on 60 worlds
([`docs/taste-on-manager.md`](docs/taste-on-manager.md)).

## Equivalence with the pre-split code

Checked on CPU against `49a6bc3` (seed 0, `size1m`, batch 2 x 65 steps,
JAX 0.4.33). For each flag set the agent is built from both codes and
compared on: the parameter tree (names, shapes, dtypes) and initial values;
a 65-step rollout in two seeded Craftax worlds (observations, actions, every
policy output, including RSSM-2's state, goal and goal age); probe and greedy
acting; the parameters and train metrics after four train steps on that
batch; and the report metrics.

| flag set | result (single-threaded XLA) |
|---|---|
| `plain` | bit-identical everywhere (max abs diff 0) |
| `next_lam0` | bit-identical everywhere |
| `next_nomask` | bit-identical everywhere |
| `next_notaste` | identical through the rollout, acting, train step 1 and the train metrics of steps 1-3; afterwards float rounding: optimizer state differs by 2.9e-4 at step 2, 0.021 at step 4, step-4 `grad_norm` by 0.07 |
| `next_lam0` vs `next_notaste`, both on this branch | bit-identical everywhere outside taste's own variables and `taste/*` metrics: at `lam_mgr 0` taste adds exactly nothing |

At `size50m` the initial parameter trees (names, shapes, dtypes, values) of
`plain`, `next_lam0` and `next_nomask` are bit-identical too (144.6M, 314.7M
and 305.9M values). With XLA's default multi-threaded CPU backend the two
codes compile to slightly different thread partitions, so after the first
train step the optimizer state picks up float-rounding differences (up to
0.069 in `plain` after four steps); the same code run twice is bit-identical,
and single-threaded the differences vanish, so they are compilation rounding,
not behaviour. `next_notaste`'s residual difference is of the same kind:
`next_notaste` on this branch equals `next_lam0` on both codes bit for bit.
Checkpoints of the family's runs therefore load here unchanged.

Runs trained before 2026-10-11 used buggy RSSM-2 code that no branch
reproduces exactly (`docs/rssm2-step-alignment.md`). None of this family's
runs is affected: all were trained on `~/vtmgr` = `49a6bc3`, which has the
fixes.

## Docs

- [`docs/design-manager.md`](docs/design-manager.md): the manager, goals, the
  shared critic, the goal stream.
- [`docs/design-v2.md`](docs/design-v2.md): RSSM-2 as a memory, curiosity.
- [`docs/rssm2-step-alignment.md`](docs/rssm2-step-alignment.md): why training
  and acting read the same RSSM-2 state.
- [`docs/taste-on-manager.md`](docs/taste-on-manager.md): why `lam_mgr 0` is
  the no-taste control, and why taste hurt.
- [`docs/audit-2026-10-11.md`](docs/audit-2026-10-11.md): the outside audit.
- [`HANDOFF.md`](HANDOFF.md): working log of this family; the full log is in
  `HANDOFF.md` at tag `archive/full-2026-11-08`.
- [`CHANGELOG.md`](CHANGELOG.md).
