# DreamerV3 on Craftax

Plain [DreamerV3][upstream] trained on symbolic [Craftax][craftax], with the
fixes this project found along the way. This branch is the baseline and holds
no experiment code: every agent built on top of it lives on a family branch of
its own (see [Family branches](#family-branches)).

The original DreamerV3 README is kept at
[`README-dreamerv3.md`](README-dreamerv3.md); installation and the general run
interface are unchanged.

[upstream]: https://github.com/danijar/dreamerv3
[craftax]: https://github.com/MichaelTMatthews/Craftax

## What differs from upstream

- **Craftax adapter** (`embodied/envs/craftax.py`) and the `--configs craftax`
  preset: `craftax_symbolic`, one env, `train_ratio 512`, 1.1M steps, replay
  capped at 4e5 steps (replay lives in host RAM; every scored run used the
  cap, so it stays). The adapter:
  - emits `obs['ach']`, the achievements unlocked this step. The scoring tools
    read it from replay; the agent excludes it from the encoder and decoder;
  - treats a timeout as a truncation (`is_last`), not a terminal state;
  - rebuilds the next world from a seed alone (`env.reseed`), so evaluation
    meets the same fixed worlds whatever the earlier episodes did;
  - keeps Craftax's own reward in `raw_reward`;
  - leaves `obs['vector']` unbounded above (Craftax's meters grow past 1).
- **Acting modes:** `policy(mode='greedy')` takes the argmax action;
  `policy(mode='probe')` also returns the action distribution.
- **Fixes:** `report()` keeps its loss metrics; `run.from_checkpoint_regex`
  exists; `replay.save_skip` keeps the replay_context latents off disk (~1.4 GB
  a run instead of ~9.3 GB).
- **Scoring tools** in `tools/` (see [Score](#score)).

The world model, actor-critic, losses and hyperparameters are upstream's.

## Install

```bash
pip install -U -r requirements.txt
pip install craftax==1.6.1      # the version every scored run used
python -c "from dreamerv3.craftax_features import validate_against_craftax as v; print(v())"
```

The last line checks that the achievement names embedded in
`dreamerv3/craftax_features.py` match the installed Craftax.

## Run

The plain baseline, exactly as `abl_plain` was trained:

```bash
python dreamerv3/main.py --logdir ~/logdir/abl_plain \
  --configs craftax size50m --env.craftax.survival none
```

`--seed 1` gives `abl_plain_s1`. `--env.craftax.survival none` is the default
(main has no reward shaping) and is kept so the old commands still parse.

A finished run takes ~1.4 GB: ~0.8 GB of replay and a ~0.7 GB checkpoint. A
resumed run gets zeros back for the latents `replay.save_skip` left off disk,
and refreshes them as it trains. To shrink a run that predates this:
`python tools/strip_replay.py ~/logdir/<run>` (it refuses a live or unfinished
run; a stripped run can no longer be resumed).

## Test

```bash
python -m pytest dreamerv3/                                   # 6 tests
python -m pytest embodied/tests/test_replay.py -k save_skip
```

`dreamerv3/test_audit_fixes.py` needs JAX, and its env tests need Craftax (they
are skipped without it). The other tests in `embodied/tests/test_replay.py`
call `Replay.dataset`, which this `Replay` does not have; they fail the same
way on the pre-split code.

## Score

Every result is scored on the same 30 fixed worlds (episode *i* is built from
seed *i*), so comparisons are paired world by world.

```bash
# 1. Play 30 episodes; prints achievements, unlock rates, deaths, raw return
#    and a world hash, which must match across the runs being compared
#    (9067302a2635 for the default seeds).
python tools/death_eval.py --logdir ~/logdir/abl_plain --episodes 30 \
  --configs craftax size50m --env.craftax.survival none > abl_plain.log
#    or rebuild from the run's own config.yaml:
#    python tools/death_eval.py --logdir ~/logdir/abl_plain --config-from-run

# 2. Turn the log into an entry for docs/eval_results.json
python tools/eval_entry.py abl_plain.log --id abl_plain --label "Plain" --short plain

# 3. Training curves from replay, the curve figure and the PDF report
python tools/curves_from_replay.py ~/logdir/abl_plain --out docs/curves
python tools/plot_training_curves.py                         # -> docs/training-curves.*
python tools/make_comparison_report.py                       # -> docs/comparison-report.pdf

# 4. A run still training, against finished ones at the same env steps
python tools/compare_at_steps.py "plain=docs/curves/abl_plain.json" ...
```

`--config-from-run` drops the keys this branch's `configs.yaml` lacks, so a
family run would be rebuilt as the plain agent. **Score each run from its own
branch.**

A 30-world score moves by about ±0.55 between world sets: a gap under ~1.1
between two single runs is not evidence. Use 60 worlds and several training
seeds.

Inspection: `tools/watch_agent.py` (watch a checkpoint play),
`tools/action_audit.py` (what it spends its actions on),
`tools/craft_probe.py` (does it craft when crafting is one keypress away),
`tools/death_diagnostic.py` and `tools/death_math.py` (deaths of a random
policy, and survival bounds).

## Results

30 fixed worlds, 1.1M env steps, `size50m`
([`docs/eval_results.json`](docs/eval_results.json),
[`docs/comparison-report.pdf`](docs/comparison-report.pdf),
[`docs/training-curves.png`](docs/training-curves.png)):

| run | code | achievements ± SE | normalized return | lifespan |
|---|---|---|---|---|
| `vanilla` | original code, before the 2026-10 fixes | 4.63 ± 0.22 | 1.65% | 295 |
| `abl_plain` | this branch (seed 0) | 4.77 ± 0.28 | 1.71% | 265 |

The two are level. Both stay at the wood tier: no stone in any episode.
`abl_plain_s1` is recorded at 4.77 as well; its training curve stops at 0.9M
steps.

## Family branches

**Convention.** Each new agent version goes on one family branch, with its own
code, tests, docs, results and changelog. `main` holds only the plain agent and
what every family shares; its docs only index the families. Shared fixes are
made on `main` and merged into the family branches.

Best result = test achievements on the 30 fixed worlds
(`docs/eval_results.json` at `archive/full-2026-11-08`). The rule mask reads
the game's action rules, so **masked runs count as cheating**; for those the
best run without the mask is given too.

| branch | family | runs | best result |
|---|---|---|---|
| `family/rssm2` | Map model: a second, slower RSSM (RSSM-2) that predicts a coarse map of the level (honest or privileged targets) or serves as a memory (memory target), with potential and survival reward shaping | `map_stage5`, `map_survive`, `map_potential`, `map_noshift`, `map_pot_fixed`, `priv_map`, `honest_map`, `honest_map_fix`, `abl_rssm2` | 6.17 ± 0.34 `honest_map` (pre-fix code; `priv_map` also 6.17); fixed code 5.70 ± 0.21 `honest_map_fix` |
| `family/v` | Goal manager: v1-v1.3 (a manager over hand-written tech-tree goals; v1.2 adds a shared two-headed critic and held goals) and v2 (learned goal codebook, RSSM-2 as memory, curiosity) | `mgr`, `mgr2`, `mgr3`, `mgr4`, `mgr3_fix` ×3, `v12_nomask`, `v2`, `v2cur`, `v2_fix` ×3, `v2_norules`, `abl_manager` ×2 | 9.97 ± 0.36 `mgr3` (v1.2) — masked, counted as cheating; without the mask 6.03 ± 0.33 `v12_nomask` |
| `family/next` | The next agent: v1.2's named goals on v2 (memory RSSM-2, shared critic, curiosity); taste on the manager at `lam_mgr 0`, i.e. no taste | `next_notaste` (crashed), `next_lam0`, `next_nomask` | 5.63 ± 0.27 `next_nomask` (no mask; 5.82 ± 0.17 on 60 worlds); 11.20 ± 0.41 `next_lam0` — masked, counted as cheating |
| `family/taste` | Taste: a per-action logit bias learned from the critic's advantage (taste over actions), or an internal reward on the manager's goal choice (taste on the manager) | `abl_taste`, `v2_taste`, `next_taste` | 9.80 ± 0.42 `next_taste` — masked, counted as cheating; without the mask 5.33 ± 0.19 `v2_taste` |
| `family/masked` | B1/B2/B3 rule-validity experiments: B1 feeds the action-validity flags to the encoder, B2 masks impossible actions, B3 learns the mask | `expB_input`, `expB_mask`, `expB_mask_fix` ×3, `expB3`, `expB3fix`, `calib_obs`, `calib_latent` | 8.40 ± 0.54 `expB_mask` (B2) — masked, counted as cheating. The whole family reads the game's rules (B1 as input, 6.30 ± 0.33 `expB_input`), so all of it counts as cheating; no new work goes here |
| `family/market` | The market agent: score and survival traded at a price that a small network moves when the world changes; per-action Q-heads; a slope quota; on top of `next_nomask` | `market_quota` (training now), then `market_quota2`, `market_fixed`, `market_full` | not scored yet |
| tag `archive/full-2026-11-08` | The full pre-split code (`49a6bc3`): every family side by side, plus the reward-cause head (off in every run) | every run above | all results together in its `docs/eval_results.json` |

Each family branch runs its runs' exact flags (`config.yaml` in each logdir),
and so does the tag. Runs from before the 2026-10-11 fixes trained on buggy
code, which no branch reproduces exactly.
