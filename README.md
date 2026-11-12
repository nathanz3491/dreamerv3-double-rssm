# DreamerV3 on Craftax: the market agent (`family/market`)

**Status: the quota × market grid is training; three of its six runs are
scored (see Results).** Every part of the design is built: the survival goals
(`env.craftax.goals_survival`), the slope quota (`env.craftax.quota.enabled`,
v1 `mode window` or v2 `mode log`), the Q-heads at a fixed price
(`agent.market.enabled`) and the moving price (`agent.market.shift`). So is
`market12` (v1.2's tech-tree potential, `env.craftax.survival potential`; see
below). Every new part is behind a flag that defaults off, so with the
defaults the code is `family/next`'s agent. The design is
[docs/design-market.md](docs/design-market.md); its table says which parts are
built. The base, `next_nomask`, is the honest next agent from `family/next`. For
the plain baseline and the index of every family, see `main`.

## What it is

The agent trades two goods, **score** (achievements) and **survival**, in one
market:

- **The price.** A small network reads how the world just changed and shifts
  the market's supply and demand curves. That moves the price P\*, how much
  survival one unit of score is worth.
- **Who uses it.** The critic, the actor and the manager all value things at
  P\*, through per-action Q-heads for score and for life.
- **The floor.** A quota keeps the agent pursuing score. v1 (`mode window`):
  at least one achievement per window. v2 (`mode log`): at least a logarithmic
  floor fitted to the agent's own recent lives (design §4).
- **The manager's goals** get v1.3's survival goals (DRINK, EAT) beside the 13
  named goals, so that it can act on a high price for survival.

The game's rule mask stays off; it counts as cheating. Full design, the
decisions behind it and the open questions:
[docs/design-market.md](docs/design-market.md).

## Base agent

The code here is `family/next`'s (merged up to `f89e4d9`). Its docs describe the base:
[docs/design-manager.md](docs/design-manager.md),
[docs/design-v2.md](docs/design-v2.md) and
[docs/taste-on-manager.md](docs/taste-on-manager.md). The base run,
`next_nomask`:

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

Taste at `lam_mgr 0.0` adds exactly zero. Its runs and results live on
`family/next`.

## The ablation grid

Each adds to `next_nomask`'s flags (below as `$NEXT`) the survival goals and
one or more layers of the market. The rule mask stays off. The layers are
cumulative (quota only, then Q-heads at a fixed price, then a moving price).
Each layer runs on both quotas, v1 and v2, which gives six runs, one each,
about 17.5 h apiece:

```bash
NEXT="--configs craftax size50m --env.craftax.survival none   --env.craftax.valid_obs True --env.craftax.goals_obs True   --agent.mapmodel.enabled True --agent.mapmodel.to_actor True   --agent.mapmodel.target memory --agent.valid.mask False   --agent.manager.enabled True --agent.manager.critic shared   --agent.manager.hold 32 --agent.manager.goals named   --agent.curiosity.enabled True   --agent.taste.enabled True --agent.taste.target manager   --agent.taste.center True --agent.taste.pos 1.43 --agent.taste.neg 1.43   --agent.taste.lam_mgr 0.0 --env.craftax.goals_survival True"

# 1. Quota only: does the floor alone help or hurt?
python dreamerv3/main.py --logdir ~/logdir/market_quota $NEXT   --env.craftax.quota.enabled True

# 1b. The same on quota v2, the logarithmic floor (market_quota's flags + mode log)
python dreamerv3/main.py --logdir ~/logdir/market_quota2 $NEXT   --env.craftax.quota.enabled True --env.craftax.quota.mode log

# 2. Quota + Q-heads at a fixed price P* = 0.01: does valuing survival help?
python dreamerv3/main.py --logdir ~/logdir/market_fixed $NEXT   --env.craftax.quota.enabled True --env.craftax.quota.mode log --agent.market.enabled True

# 3. The full market: does a moving price beat a fixed one?
python dreamerv3/main.py --logdir ~/logdir/market_full $NEXT   --env.craftax.quota.enabled True --env.craftax.quota.mode log --agent.market.enabled True   --agent.market.shift True

# 2b, 3b. The same two on quota v1 (mode window, the default)
python dreamerv3/main.py --logdir ~/logdir/market_fixed_q1 $NEXT   --env.craftax.quota.enabled True --agent.market.enabled True
python dreamerv3/main.py --logdir ~/logdir/market_full_q1 $NEXT   --env.craftax.quota.enabled True --agent.market.enabled True   --agent.market.shift True
```

Runs without a suffix after `market_quota` use quota v2 (`--env.craftax.quota.mode
log`). `market_quota` and the `_q1` runs use v1 (`mode window`). The box's runs
also pass `--env.craftax.logs True`, which adds metrics only. The v2 and `_q1`
runs use code 93873d1; from `market_fixed_q1`'s resume on, with the NaN
guard (see Results). `market_quota` used 297635f, whose v1 quota code is the
same. The quota's per-life penalty, death charge, floor and fit are logged
as `log/quota_pen`, `_death`, `_floor` (and `_a`, `_c` in v2), each as
`<key>/sum`, `/avg`, `/max`.

Constants (`env.craftax.quota.*`, `agent.market.*`) are untuned defaults; each
is documented in `dreamerv3/configs.yaml` and chosen in
[docs/design-market.md](docs/design-market.md). With none of the flags the code
is `family/next`'s agent. The box's own launch rules (memory guard, detached
start) apply.

## market12: the market on v1.2's shaping

The grid suggests the market loses most where it stops sleeping and climbing
past wood, while v1.2 (`v12_nomask` on `family/v`, 6.13 on 60 worlds) is the
best honest agent so far. `market12` combines the parts chosen on 2026-11-12:

- **v1.2's tech-tree potential** (`--env.craftax.survival potential`),
  ported from `family/v` without its map labels or map target, so RSSM-2
  stays the honest memory model. It is potential-based shaping, F = γ·Φ(s′) −
  Φ(s) with Φ handed back at a true death, so it cannot change the optimal
  policy. Φ is built from the tech-tree spine, which is game knowledge.
- **The shaping is kept out of the score.** The env sends it as its own key,
  `shape`, and leaves `reward` (game reward plus quota) alone. The agent
  learns it with its own head, `shp`, and adds it to what the critic and the
  manager see. The market's score Q-head learns from the game reward plus the
  quota only.
- **Quota v1** (`mode window`) and **the Q-heads at a fixed price**
  (`agent.market.enabled`, P\* = 0.01).
- **No curiosity**, as in v1.2 (v2 used curiosity in place of the potential).
  Taste (inert at `lam_mgr 0`) and `valid_obs` (unused with the mask off)
  are dropped too.

```bash
python dreamerv3/main.py --logdir ~/logdir/market12 \
  --configs craftax size50m --env.craftax.survival potential \
  --env.craftax.goals_obs True --env.craftax.goals_survival True \
  --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
  --agent.mapmodel.target memory --agent.valid.mask False \
  --agent.manager.enabled True --agent.manager.critic shared \
  --agent.manager.hold 32 --agent.manager.goals named \
  --env.craftax.quota.enabled True --agent.market.enabled True
```

Against `market_fixed_q1` it differs in the shaping and in curiosity being off.
With `survival none` the code trains exactly as before: no `shape` key, no
`shp` head. Tests: `dreamerv3/test_shaping.py` (needs Craftax).

## Results

| run | quota | achievements ± SE | 60 worlds | status |
|---|---|---|---|---|
| `next_nomask` | none | 5.63 ± 0.27 | 5.82 ± 0.17 | the base agent (`family/next`) |
| `market_quota` | v1 | 6.03 ± 0.24 | 6.05 ± 0.18 | run 2026-11-09 17:42 – 2026-11-10 11:08 CST |
| `market_quota2` | v2 | 5.30 ± 0.21 | 5.38 ± 0.16 | run 2026-11-10 18:38 – 2026-11-11 12:13 CST |
| `market_fixed` | v2 | 5.10 ± 0.30 | 5.37 ± 0.19 | run 2026-11-11 12:13 – 2026-11-12 07:05 CST |
| `market_fixed_q1` | v1 | | | training, resumed at 90k after a NaN crash |
| `market12` | v1 | | | queued next (`~/run_market5.sh`): v1.2's potential, fixed price |
| `market_full_q1` | v1 | | | queued after `market12` |
| `market_full` | v2 | | | dropped 2026-11-12: quota v2 stopped the agent sleeping |

The quota alone is level with the base agent. Paired world by world on 60
worlds, `market_quota` is +0.23 ± 0.23 over `next_nomask`. Its lives are
slightly longer (272 steps against 256). Its deaths split evenly between food,
drink and mobs, where `next_nomask` mostly died of thirst. During training it
trailed `next_nomask` by 0.6–1.2 achievements per episode up to 800k env steps.
It caught up only in the last stretch.

Quota v2 gives no sign of helping. Paired on 60 worlds, `market_quota2` is
0.67 ± 0.23 below `market_quota` and 0.43 ± 0.23 below `next_nomask`. It led v1
slightly up to ~300k env steps. It then stayed near 4.2 achievements per
training episode until ~950k, while v1 climbed. These are single runs, so this
does not show that v2 is worse. Entries are in `docs/eval_results.json`.

The Q-heads at a fixed price buy a little life and no score on quota v2.
Paired on 60 worlds, `market_fixed` is −0.02 ± 0.27 against `market_quota2`
and −0.45 ± 0.22 against `next_nomask`. Its lives are the longest of the grid
so far (285 steps, +21 ± 15 over `market_quota2`). Its deaths are mostly
hunger (31 of 60), then mobs (16) and thirst (13).

`market_fixed_q1` died at 99,990 env steps on a NaN in a sampled observation;
the replay saved on disk was finite, so the source is unknown. The env now
zeroes a NaN in Craftax's observation and logs `[craftax] NaN in obs` with the
dims (the box's `~/vmarket3`, used from `market_fixed_q1`'s resume on).

## Install

```bash
pip install -U -r requirements.txt
pip install craftax==1.6.1      # the version every scored run used
python -c "from dreamerv3.craftax_features import validate_against_craftax as v; print(v())"
```

## Test

```bash
python -m pytest dreamerv3/
python -m pytest embodied/tests/test_replay.py -k save_skip
python -m pytest dreamerv3/test_shaping.py      # needs Craftax
```

`test_audit_fixes.py`, `test_taste.py`, `test_mapmodel_step.py` and
`test_market.py` need JAX. `test_quota.py`, `test_market_math.py` and
`test_market_config.py` run without it (the two env tests in `test_quota.py`
need Craftax).
The craftax tests are skipped without Craftax. The other tests in
`embodied/tests/test_replay.py` fail with
`'Replay' object has no attribute 'dataset'` here and at `49a6bc3` alike.

## Score

Every run is scored on the same 30 fixed worlds (world hash `9067302a2635`),
paired world by world against `next_nomask`:

```bash
python tools/death_eval.py --logdir ~/logdir/<run> --episodes 30 --config-from-run > <run>.log
python tools/eval_entry.py <run>.log --id <run> --label "<label>" --short <short>
```

A 30-world score moves by about ±0.55, so a gap under ~1.1 between two single
runs is not evidence.

`--config-from-run` drops keys that this branch's `configs.yaml` lacks, so score
each run from the branch it belongs to. `next_nomask`'s results live on
`family/next`.
