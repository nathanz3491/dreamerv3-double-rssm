# DreamerV3 on Craftax — family `masked` (rule-validity experiments B1-B3)

> **Counted as cheating. No new work goes on this branch.**
>
> Every agent here reads the game's action rules.
> `dreamerv3/craftax_valid.py` copies each of the 43 actions' preconditions
> from Craftax's code. The env turns them into `obs['valid']`, and the agent
> uses it:
> - **B2** masks impossible actions while acting (`valid.mask`).
> - **B1** feeds the flags to the encoder (`valid.input`). That is rule
>   knowledge too, even though nothing is masked.
>
> The published 1M-step Craftax baselines use no such knowledge. The rules
> also pay off more than anything else in the project: the mask was worth
> **+2.6 to +2.9 achievements wherever it was measured** (B2 vs the same model
> without it, +2.63 ± 0.38 over three seeds; v1.2 with vs without it, +2.90 ±
> 0.26 over 60 worlds). That is why these scores are kept out of the headline
> rankings.
>
> The branch exists to reproduce and score the old runs. The one
> knowledge-free idea in it, the learned mask of **B3**, is still not free of
> the rules ([below](#is-b3-knowledge-free)).

This is plain DreamerV3 (the `main` branch) plus what the B1-B3 runs need.
For the plain baseline, the shared fixes and the index of all family branches,
see `main`'s README. The full pre-split code is the tag
`archive/full-2026-11-08`.

## What the family is

Experiment A (`tools/action_suppression.py`) found that unmasked policy
gradients push an action's logit down in every state where the action does
nothing. Shared weights carry that suppression into the rare states where the
action would work. For example, B2's predecessor never placed a furnace in any
of the 386 states where it could have. Experiment B tests the fix
(`docs/entropy-and-action-suppression.md` has the full record).

| arm | flags on top of the honest-map base | what changes |
|---|---|---|
| B1, tell it | `--env.craftax.valid_obs True --agent.valid.input True` | the 43 validity flags are an encoder input; the decoder reconstructs them |
| B2, mask it | `--env.craftax.valid_obs True --agent.valid.mask True` | impossible actions get zero probability and zero gradient: `obs['valid']` while acting, a learned `feas` head on the latent inside imagination |
| B3, learn the mask | B2 + `--agent.valid.learned True` | the `feas` head is trained on the world model's own counterfactual (did NOOP explain the outcome as well?) and masks the actor while acting too |

Every arm runs on the **honest-map configuration**:
- RSSM-2, a second, slower RSSM trained on a coarse map built from the
  agent's own observations (`mapmodel.py`, `craftax_map.py`). Its crop feeds
  the actor.
- Potential-based shaping over the tech tree (`craftax_potential.py`).

Both are hand-written game knowledge of their own. Their no-mask runs are on
the `family/rssm2` branch.

### Code on this branch (beyond `main`)

- `dreamerv3/craftax_valid.py`: `valid_actions(obs_vector)`, the rule mask.
  It reads only the observation (a test pins the signature) and agrees with
  the game in 77,700 of 77,700 checks (`test_craftax_valid.py`). It also
  holds `BASIC` (NOOP, the four moves, DO), which are never masked.
- `dreamerv3/agent.py`:
  - `valid.*` options, the `feas` head (`validhead`), and `_masked` (adds -1e4
    to masked logits after the unimix, so no floor leaks back).
  - `_dream_valid`: the mask inside imagination.
  - `_feas_learned`: B3's labels.
  - `Counter` (`feasclock`): B3's warm-up clock, kept with the parameters so
    acting can read it.
  - RSSM-2 with its map target, the actor's map crop and the
    acting/training step `_map_step`, with the 2026-10-11 fixes
    (`docs/rssm2-step-alignment.md`).
- `embodied/envs/craftax.py`:
  - `valid_obs`: emits `obs['valid']`.
  - `mapmodel`, `map_privileged`, `seen_decay`: the map targets.
  - `survival none|potential`, `phi_scale`, `phi_gamma`: the shaping.
- `dreamerv3/configs.yaml`: `agent.valid`, `agent.validhead`,
  `agent.mapmodel`, and the loss scales `map`, `mappos`, `feas`.
  - Every option defaults off. With all of them off the agent is `main`'s
    plain DreamerV3.
  - Key names, module names and parameter names, RNG order and loss order
    are those of `49a6bc3`, so old checkpoints load.
- Left out, because no run here used them:
  - RSSM-2's memory target (`mapmodel.target`, `memhead`): on `family/rssm2`.
  - The `shaped` and `potential+meters` survival modes.
  - The manager, goals, curiosity and taste: on `family/v`, `family/next`
    and `family/taste`.

### Is B3 knowledge-free?

No, though it comes closest.
- **The label** uses no rules. It is the world model's own judgement:
  - latent KL, B3: `valid.label latent`;
  - or the decoded-observation error against what really came next, B3-fix:
    `valid.label obs`, `valid.margin`.
- **Acting** reads the learned head, not `obs['valid']`.
  - `rule_mix` makes some acting steps follow the rules, but it was only
    used for label calibration. Every comparison run left it at 0.
- **But the rules are still in the setup:**
  - The `feas` head is only built when the env emits `obs['valid']`
    (`env.craftax.valid_obs True`). The rules then sit in replay and in
    metrics, although no gradient comes from them.
  - The six `BASIC` actions are hand-picked as never masked
    (`valid.exempt_basic True`).
  - Action 0 is assumed to be NOOP, the reference the label compares against
    (`valid.reference noop`).
  - `exempt_basic False` and `reference random` remove the last two, but no
    run used them.
- **The metrics depend on the rules:**
  - `feas/label_precision`, `feas/label_recall`, `feas/evidence_auc` and
    `feas/tpr_*` / `feas/fpr_*` score the labels against `obs['valid']`.
  - `valid/recall` and `valid/false_pos` score the mask.
- B3-fix's margin and threshold (0.5 nats, 0.15) were chosen from a
  calibration (`calib_obs` / `calib_latent`) that measured the labels
  against the rules.

A knowledge-free version would need the head without `valid_obs`, and some
other way to evaluate it. It is still an open idea. Block an action only when
a world-model ensemble agrees it does nothing, so an effect not yet learned
stays pressable. Its natural home is a new family branch, not this one.

## Install

As on `main`:

```bash
pip install -U -r requirements.txt
pip install craftax==1.6.1      # the version every scored run used
python -c "from dreamerv3.craftax_features import validate_against_craftax as v; print(v())"
```

## Run

Every run used `--configs craftax size50m` (1.1M env steps, replay 4e5). The
shared base (`MAP`, the honest-map configuration) is:

```bash
MAP="--configs craftax size50m --env.craftax.mapmodel True --env.craftax.survival potential \
  --env.craftax.map_privileged False --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
  --agent.mapmodel.imag_shift False --agent.mapmodel.hindsight True --agent.imag_length 15"
B2="$MAP --env.craftax.valid_obs True --agent.valid.mask True"

python dreamerv3/main.py --logdir ~/logdir/expB_input $MAP --env.craftax.valid_obs True --agent.valid.input True   # B1
python dreamerv3/main.py --logdir ~/logdir/expB_mask_fix $B2                  # B2 (also expB_mask)
python dreamerv3/main.py --logdir ~/logdir/expB_mask_fix_s1 $B2 --seed 1      # B2, seed 1 (s2: --seed 2)
python dreamerv3/main.py --logdir ~/logdir/expB3 $B2 \
  --agent.valid.learned True --agent.valid.threshold 0.02                     # B3
python dreamerv3/main.py --logdir ~/logdir/expB3fix $B2 --agent.valid.learned True \
  --agent.valid.detach True --agent.valid.label obs --agent.valid.margin 0.5 \
  --agent.valid.threshold 0.15                                                # B3-fix
```

Label calibration (`calib_obs`, `calib_latent`): load B2's trained world model
with a fresh `feas` head and optimizer, and train briefly:

```bash
python dreamerv3/main.py --logdir ~/logdir/calib_obs $B2 \
  --run.from_checkpoint ~/logdir/expB_mask/ckpt/<ckpt> --run.from_checkpoint_regex '^(?!.*(feas|opt)).*' \
  --agent.valid.learned True --agent.valid.threshold 0.02 --agent.valid.detach True \
  --agent.valid.label obs --run.train_ratio 64 --run.steps 8000 --run.log_every 60
```

Use `--agent.valid.label latent` for `calib_latent`. These are the flags of
the surviving `calib.sh`.
`docs/entropy-and-action-suppression.md` describes the calibration with
`--agent.valid.rule_mix 0.8`, which makes batches hold both valid and invalid
special actions. `calib.sh` does not pass that flag.

**Which code trained what.** Re-running a command on this branch gives the
code of `49a6bc3`. No branch reproduces the old runs bit for bit:
- `expB_input`, `expB_mask` and `expB3` trained before the 2026-10-11 RSSM-2
  fixes, on code with three RSSM-2 bugs.
- `expB_mask_fix` (s0-s2) trained on `7df2d36` (`~/vfix`), which has the
  fixes.
- `expB3fix` and the calibrations used a snapshot (`~/vb3fix`).
- This branch keeps every flag those runs passed.

## Test

```bash
python -m pytest dreamerv3/
python -m pytest embodied/tests/test_replay.py -k save_skip
```

The tests:
- `test_craftax_valid.py`: the rule mask against the game.
- `test_craftax_map.py`: map targets and decoders.
- `test_craftax_potential.py`: potential shaping.
- `test_mapmodel_step.py`: RSSM-2 train/act alignment and resets.
- `test_audit_fixes.py`: fixed worlds, first shaped step, raw reward, greedy
  acting.

Notes:
- The game checks need Craftax and are skipped without it.
- `test_mapmodel_step.py` and `test_audit_fixes.py` need JAX.
- The other tests in `embodied/tests/test_replay.py` call `Replay.dataset`,
  which this `Replay` lacks; they fail the same way at `49a6bc3`.

## Score

As on `main`, on the 30 fixed worlds (world hash `9067302a2635`). **Score each
run from this branch**: `--config-from-run` drops config keys a branch does not
have, so on `main` a B2 run would be rebuilt without its mask.

```bash
python tools/death_eval.py --logdir ~/logdir/expB_mask_fix --episodes 30 --config-from-run > b2.log
python tools/eval_entry.py b2.log --id expB_mask_fix --label "B2, fixed RSSM-2 [masked]" --short "B2 fix"
python tools/curves_from_replay.py ~/logdir/expB_mask_fix --out docs/curves
python tools/plot_training_curves.py         # -> docs/training-curves.*
python tools/make_comparison_report.py       # -> docs/comparison-report.pdf
```

A 30-world score moves by about ±0.55 between world sets, so a gap under ~1.1
between two single runs is not evidence.

Family analyses. Each reads a run's replay or plays a checkpoint:
- `tools/action_suppression.py` (experiment A): valid-action mass, entropy,
  and each key's probability where it would work.
- `tools/diagnose_run.py`: tech-tree funnel, stone-pickaxe ingredients and
  deaths, written to `docs/diagnosis.json`.
- `tools/episode_cases.py` and `tools/make_casebook.py`: B2's shortfalls
  shown step by step (`docs/cases/`, `docs/b2-casebook.pdf`).

## Results

30 fixed worlds, 1.1M env steps, `size50m` (`docs/eval_results.json`,
`docs/comparison-report.pdf`, `docs/training-curves.png`).

**Honest runs** (no rule knowledge), for reference; they live on `main`:

| run | achievements ± SE | normalized return | lifespan |
|---|---|---|---|
| `vanilla` (original code) | 4.63 ± 0.22 | 1.65% | 295 |
| `abl_plain` (fixed code, seed 0) | 4.77 ± 0.28 | 1.71% | 265 |

**This family: every run counted as cheating.** These rows are not
comparable with the honest runs above or with the published baselines.

| run | reads the rules how | achievements ± SE | normalized return | lifespan |
|---|---|---|---|---|
| `expB_input` (B1) | flags as encoder input (no mask) | 6.30 ± 0.33 | 2.39% | 295 |
| `expB3` (B3) | learned mask; rules in the setup (see above) | 5.97 ± 0.28 | 2.24% | 269 |
| `expB_mask` (B2) | **masked** by the rules | 8.40 ± 0.54 | 3.32% | 282 |
| `expB_mask_fix` (B2, fixed RSSM-2) | **masked** by the rules | 8.27 ± 0.44 (seeds 8.27 / 7.70 / 9.03, mean 8.33) | 3.32% | 274 |
| `expB3fix` (B3-fix) | learned mask; rules in the setup | stopped at ~254k steps, not scored | | |
| `calib_obs`, `calib_latent` | label calibration from B2's checkpoint, 8k steps | not scored (label metrics only) | | |

What the family showed:
- **Masking works, and it costs the rules.** B2 vs the same model without
  the mask (the honest map, `family/rssm2`):
  - +2.63 ± 0.38 on the fixed code, three seeds;
  - +2.23 on the old code.

  Frontier keys that sat at exactly 0% where they worked (furnace, placing
  stone) come off zero. This is valid-action suppression.
- **Telling is not masking.** B1 is level with the honest map on the fixed
  worlds (6.30 vs 6.17).
- **The learned mask failed.**
  - B3: its latent labels fed back into the world model, and 87% of its
    "did something" labels were wrong.
  - B3-fix: at 254k steps its mask let through only 45% of the possible
    special actions. Its label cannot tell "does nothing" from "not learned
    yet".
- **The RSSM-2 fixes did not move B2** (8.27 vs 8.40). Its three seeds
  spread by 1.3.

## Docs

- `docs/entropy-and-action-suppression.md`: the question, the mechanism,
  experiment A, and B1-B3 with their results and calibration.
- `docs/experiment_a/`: experiment A per run.
- `docs/b2-casebook.pdf` with `docs/cases/`, and `docs/diagnosis.json`:
  where B2 falls short.
- `docs/rssm2-step-alignment.md`: the three RSSM-2 bugs, fixed 2026-10-11.
- `docs/audit-2026-10-11.md`: the outside audit (shared with `main`).
- `docs/eval_results.json`, `docs/curves/`, `docs/comparison-report.pdf`,
  `docs/training-curves.*`: the scores and curves above.
