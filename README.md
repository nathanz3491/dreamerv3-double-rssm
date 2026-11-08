# DreamerV3 on Craftax: the market agent (`family/market`)

**Status: being built, not trained yet.** Built so far: the survival goals
(`env.craftax.goals_survival`) and the slope quota (`env.craftax.quota.enabled`). Every new part is behind a flag that defaults
off, so with the defaults the code is `family/next`'s agent. The design is
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
- **The floor.** A slope quota, at least one achievement per window, keeps the
  agent pursuing score.
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
```

`test_audit_fixes.py`, `test_taste.py` and `test_mapmodel_step.py` need JAX.
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
