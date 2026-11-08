# DreamerV3 on Craftax: the taste family (`family/taste`)

**Taste** is a learned internal-reward / prior rule (`dreamerv3/taste.py`),
ported from the plastic-reward toy (`../plastic-reward`): a fast,
context-free judgement of each option, revised from the critic's advantage on
replay. This branch is [`main`][main] (plain DreamerV3 on symbolic
[Craftax][craftax]) plus taste and everything its three runs need:

- **taste over actions** (`--agent.taste.target actions`): a per-action logit
  bias on the actor, the knowledge-free stand-in for the rule mask. Runs
  `abl_taste` (on plain DreamerV3) and `v2_taste` (on v2 without the mask);
- **taste on the manager** (`--agent.taste.target manager`): an internal
  reward on the manager's choice of goal. Run `next_taste` (on the next
  agent, which uses the rule mask).

**Result: taste did not help anywhere it was tried** (see
[Results](#results)). The plain baseline and the general fixes are documented
on [`main`][main]; the other families are indexed in main's README.

[main]: ../../tree/main
[craftax]: https://github.com/MichaelTMatthews/Craftax

## Design

Taste keeps, per option (an action, or a manager goal), a running evidence
count and a mean, folded every 2000 train updates by the toy's evolved rule;
the policy sees `conf × taste`, where confidence grows with evidence
(`docs/taste-prior.md`, "What carried over").

- **Over actions.** Evidence: at each real replay step, the λ-return that
  followed minus the critic's value there, filed under the action taken. The
  bias (`lam` 4.92 logits per unit of taste) is added to the actor's logits
  in `Agent._masked`, so acting, the imagined rollouts and the actor loss all
  see the same prior. `actions special` leaves NOOP, the moves and DO alone.
  `center True` compares each action's evidence with the fold's average.
- **On the manager.** Evidence: at each real step where the manager chose a
  goal, the return that followed minus the manager's expected value of its
  own choice, Σ_g π(g|s) Q(s, g) (`_taste_mgr_evidence`). Taste then adds
  `lam_mgr × rscale × (taste(g) − mean taste)` to the manager's Q for each
  goal (`_mgr_loss_shared`); no bias acts. NONE never gets a taste.
  `lam_mgr 0` computes taste and never uses it: the no-taste control.

The agents underneath (`docs/agent.md`) are carried here in full: RSSM-2 as a
memory (`mapmodel.target memory`), the manager with a shared two-headed
critic and held goals over the 13 named goals or 16 learned goals, curiosity,
and the rule mask (`valid.mask`, default off). **The rule mask reads the
game's action rules, so masked runs count as cheating**; they are marked
below and kept out of the honest ranking.

Not on this branch (each lives on its own family branch, all of it at tag
`archive/full-2026-11-08`): RSSM-2's map target, potential and survival
shaping (`family/rssm2`); the v1/v1.1 manager with its own critic, v1.3's
survival goals and v2-cur's curiosity normalisation (`family/v`); B1 and B3
(`family/masked`); the reward-cause head (archive tag only).

## Install

As on `main`:

```bash
pip install -U -r requirements.txt
pip install craftax==1.6.1      # the version every scored run used
python -c "from dreamerv3.craftax_features import validate_against_craftax as v; print(v())"
```

## Run

Each command is the run's exact flags (from its `config.yaml`). Every run is
1.1M env steps, `size50m`, seed 0.

```bash
# abl_taste: plain DreamerV3 + taste over actions (centered, pos = neg, w0 0)
python dreamerv3/main.py --logdir ~/logdir/abl_taste --configs craftax size50m \
  --env.craftax.survival none \
  --agent.taste.enabled True --agent.taste.target actions \
  --agent.taste.center True --agent.taste.pos 1.43 --agent.taste.neg 1.43 \
  --agent.taste.w0 0.0

# v2_taste: v2 without the rule mask + taste over actions (the taste defaults:
# uncentered, pos 4.45, neg 1.43, w0 0.25)
python dreamerv3/main.py --logdir ~/logdir/v2_taste --configs craftax size50m \
  --env.craftax.survival none --env.craftax.valid_obs False \
  --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
  --agent.mapmodel.target memory --agent.valid.mask False \
  --agent.manager.enabled True --agent.manager.critic shared \
  --agent.manager.hold 32 --agent.manager.goals learned \
  --agent.curiosity.enabled True --agent.taste.enabled True

# next_taste (MASKED, counted as cheating): the next agent + taste on the
# manager (centered, pos = neg = 1.43, lam_mgr 1)
python dreamerv3/main.py --logdir ~/logdir/next_taste --configs craftax size50m \
  --env.craftax.survival none --env.craftax.valid_obs True \
  --env.craftax.goals_obs True \
  --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
  --agent.mapmodel.target memory --agent.valid.mask True \
  --agent.manager.enabled True --agent.manager.critic shared \
  --agent.manager.hold 32 --agent.manager.goals named \
  --agent.curiosity.enabled True \
  --agent.taste.enabled True --agent.taste.target manager \
  --agent.taste.center True --agent.taste.pos 1.43 --agent.taste.neg 1.43
```

The no-taste control of `next_taste` is `next_lam0`: the same command with
`--agent.taste.lam_mgr 0.0` (masked too). Its run and docs live on
`family/next`; the flags work here as well. The plain baseline (`abl_plain`)
is `main`'s command, `--configs craftax size50m --env.craftax.survival none`,
and builds the same agent here.

**Which code trained them.** `abl_taste` and `next_taste` were trained on
`49a6bc3` (`~/vtmgr` on the box, tag `archive/full-2026-11-08`), `v2_taste` on
`cbec67b` (`~/vtaste`). `cbec67b` had only taste over actions; `49a6bc3`'s
defaults (`target actions`, `center False`) reproduce it, so the `v2_taste`
command needs no taste flags beyond `enabled`. Every flag set above (and
`next_lam0` and the plain one) parses here and gives the same value as
`49a6bc3` for every config key both have. On CPU (single-threaded XLA), this
branch and `49a6bc3` build bit-identical agents for the `abl_taste`,
`v2_taste`, `next_taste` and plain flag sets, and for the three taste runs
again with a taste fold every update: the same parameter names, shapes,
dtypes and initial values, the same policy outputs in every mode, and the
same parameters, train metrics and report metrics after the train steps
(plain's 4th step shows the same 4.9e-4 run-to-run noise as `49a6bc3`
against itself). See `CHANGELOG.md`. Every run before
2026-10-11 used code with the RSSM-2 bugs (`docs/rssm2-step-alignment.md`)
and is not reproduced exactly by any branch; none of this family's runs
predates the fixes.

Config defaults that differ from the archive, none of which any run above
depends on: `manager.critic` is `shared` and `manager.hold` 32 (the only
manager here), and `mapmodel.target` is `memory` (the only target here).

## Test

```bash
python -m pytest dreamerv3/
python -m pytest embodied/tests/test_replay.py -k save_skip
```

- `test_taste.py`: the fold against the toy's rule, centering, sparse
  counts for the manager, relative taste, the manager's evidence and its row
  cap, and the taste bias with and without the mask;
- `test_mapmodel_step.py`: RSSM-2's step (train equals act, resets, no
  future leak);
- `test_audit_fixes.py`: fixed evaluation worlds, raw reward, the goal
  codebook on empty batches, goal starts at replay boundaries, greedy acting;
- `test_craftax_goals.py`, `test_craftax_valid.py`, `test_craftax_map.py`,
  `test_craftax_potential.py`, `test_craftax_features.py`: the goal, rule,
  observation-decoder and achievement libraries.

The tests need JAX; the env tests need Craftax (they are skipped without it).
The other tests in `embodied/tests/test_replay.py` call `Replay.dataset`,
which this `Replay` does not have; they fail the same way on the pre-split
code.

## Score

Every run is scored on the same 30 fixed worlds (episode *i* from seed *i*;
world hash `9067302a2635`), so comparisons are paired world by world. Score a
family run from this branch: `--config-from-run` drops the keys a branch's
`configs.yaml` lacks, so another branch would rebuild a different agent.

```bash
python tools/death_eval.py --logdir ~/logdir/next_taste --episodes 30 --config-from-run > next_taste.log
python tools/eval_entry.py next_taste.log --id next_taste --label "Next-taste" --short next-taste
python tools/curves_from_replay.py ~/logdir/next_taste --out docs/curves
python tools/plot_training_curves.py        # -> docs/training-curves.*
python tools/make_comparison_report.py      # -> docs/comparison-report.pdf
```

A 30-world score moves by about ±0.55 between world sets: a gap under ~1.1
between two single runs is not evidence. Use 60 worlds and several seeds.

## Results

30 fixed worlds, 1.1M env steps, `size50m`, one seed each
([`docs/eval_results.json`](docs/eval_results.json),
[`docs/comparison-report.pdf`](docs/comparison-report.pdf),
[`docs/training-curves.png`](docs/training-curves.png)).

**Honest runs** (no game rules reach the agent):

| run | what | achievements ± SE | return | lifespan | paired gap |
|---|---|---|---|---|---|
| `v2_taste` | v2 without the mask, taste over actions | 5.33 ± 0.19 | 1.96% | 280 | +0.70 ± 0.28 vs vanilla |
| `abl_plain` | plain DreamerV3, baseline (`main`) | 4.77 ± 0.28 | 1.71% | 265 | — |
| `abl_taste` | plain DreamerV3 + taste over actions | 4.63 ± 0.29 | 1.65% | 262 | −0.13 ± 0.27 vs plain |
| `vanilla` | DreamerV3, original code | 4.63 ± 0.22 | 1.65% | 295 | — |

**Masked runs — counted as cheating** (the rule mask reads the game's
action rules; not comparable with the rows above):

| run | what | achievements ± SE | 60 worlds | return | paired gap (60 worlds) |
|---|---|---|---|---|---|
| `next_lam0` | control: next agent, taste off (`lam_mgr 0`; `family/next`) | 11.20 ± 0.41 | 11.30 ± 0.27 | 4.56% | — |
| `next_taste` | next agent + taste on the manager | 9.80 ± 0.42 | 9.38 ± 0.34 | 3.94% | −1.92 ± 0.41 vs `next_lam0` |

- **Over actions, taste does nothing measurable.** On plain DreamerV3 it is
  level with plain on the test worlds and slightly lower in training (5.19
  against 5.43 over the last 100k steps). In place of v2's rule mask it beats
  vanilla but is level in training with the same agent without taste
  (`v2_norules`, not scored); uncentered, its bias came to follow how often
  a button was pressed.
- **On the manager, taste hurt.** It locked the manager onto one goal (place
  table) at its first update, 6.7k steps in, and outweighed the critic 15 to
  300 times over. The control, with taste off, proposes all goals and makes
  stone tools; `next_taste` never did. Why, link by link:
  [`docs/taste-prior.md`, "Why taste hurt"](docs/taste-prior.md#why-taste-hurt-2026-11-07).

## Docs

- [`docs/taste-prior.md`](docs/taste-prior.md): taste's design, what carried
  over from the toy, each run's result, and why taste hurt.
- [`docs/taste-run.json`](docs/taste-run.json): v2-taste's `taste/` metrics
  over training and each action's final taste.
- [`docs/agent.md`](docs/agent.md): the agents the taste runs are built on
  (RSSM-2 as a memory, the manager, named and learned goals, curiosity, the
  rule mask).
- [`docs/rssm2-step-alignment.md`](docs/rssm2-step-alignment.md) and
  [`docs/audit-2026-10-11.md`](docs/audit-2026-10-11.md): the 2026-10-11 fixes.
- [`README-dreamerv3.md`](README-dreamerv3.md): the original DreamerV3 README.
