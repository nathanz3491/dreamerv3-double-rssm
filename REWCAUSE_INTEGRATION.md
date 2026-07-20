# Compositional reward-cause head — Phases 0 & 1 (implemented)

This adds the de-risking core of the sparse-achievement plan to DreamerV3:
factor reward into achievement *features* so the wood/stone tiers teach the
model iron's reward-event **before iron is ever mined**. It's built so it can't
break the baseline (one additive head, gated by a config flag) and so the
central hypothesis is testable in one clean experiment (iron holdout).

Grounded against upstream `danijar/dreamerv3` (embodied.jax) and Craftax
`Craftax-Symbolic-v1`. **Status: fully integrated into this repo and runnable
end to end** — the `craftax` suite is registered, the config preset exists, the
head is wired, and the go/no-go experiment has a dedicated runner
(`rewcause_probe`). Offline-verified against the installed Craftax v1.6.1
(schema exact-match, 8/8 numpy tests). What remains is GPU execution: a smoke
run + the iron-holdout probe (see below).

## Files

New (drop in):
- `dreamerv3/dreamerv3/craftax_features.py` — achievement→φ feature table,
  depth-based rebalance weights, tech-tree DAG, subgoal decomposition. Pure
  numpy, self-contained (embeds the 67-achievement schema).
- `dreamerv3/embodied/envs/craftax.py` — **Phase 0** adapter: emits
  `obs['ach']` (multi-hot of newly-unlocked achievements) + `save_state()` /
  `reset_to()` for frontier return (Phase 4).
- `dreamerv3/dreamerv3/rewcause_eval.py` — iron-holdout scorer.
- `dreamerv3/dreamerv3/test_craftax_features.py` — unit tests (8, all pass;
  run: `python test_craftax_features.py` from that dir).

Edited (integrated):
- `dreamerv3/dreamerv3/agent.py` — **Phase 1**: `ach` excluded from enc/dec;
  `self.rewcause` head + φ/weight/holdout tables; rebalanced, held-out
  `losses['rewcause']`; module + loss-scale wiring. Also: `policy()` emits
  per-feature `rewcause_prob` under `mode='probe'`, and `policy_keys` includes
  the head so its params reach the policy device (both needed by the probe).
- `dreamerv3/dreamerv3/configs.yaml` — `rewcause` scale; `rewcausehead` block;
  `rewcause`/`rewcause_none_weight`/`rewcause_holdout` flags; **`craftax`
  config preset** (`task: craftax_symbolic`, single-env, `train_ratio: 512`) and
  `env.craftax` kwargs.
- `dreamerv3/dreamerv3/main.py` — registered the `craftax` env suite and the
  `rewcause_probe` script.
- `dreamerv3/dreamerv3/rewcause_eval.py` — import made package-safe.
- `dreamerv3/embodied/envs/craftax.py` — **bug fix**: `is_terminal` no longer
  reads a non-existent `info['timeout']` (which made every episode end look
  terminal); it now derives timeout from `state.timestep >= max_timesteps` so
  truncations bootstrap correctly (matches `crafter.py`'s `discount==0` intent).
- `dreamerv3/embodied/run/rewcause_probe.py` (+ `run/__init__.py`) — **new**:
  the iron-holdout go/no-go runner (loads a checkpoint, collects head
  predictions at held-out states, scores with `rewcause_eval`).

## First, on the GPU box: confirm the schema matches your Craftax

```bash
cd dreamerv3/dreamerv3
python -c "from craftax_features import validate_against_craftax as v; v(); print('schema OK')"
```
If this raises, your Craftax version renumbered achievements — update
`ACHIEVEMENT_NAMES` before training (the embedded list is enum-value order).

## The design in one paragraph

Each step the adapter emits `ach` (which achievements just unlocked). The head
predicts the **union of their φ rows** — a factored binary vector of
`[verb, material, ordered tier thermometer, tool-line]`. The loss is
per-feature BCE, summed over φ, then **re-weighted** (deep/rare achievements
up-weighted by tech-tree depth; the sea of no-achievement steps down-weighted
by `rewcause_none_weight`). Held-out achievements contribute **zero** loss, so
the head never trains on them — that's what makes the extrapolation test honest.
Because φ's tier axis is an *ordered thermometer*, seeing tier 1 (wood) and
tier 2 (stone) lets the head place tier 3 (iron) correctly.

## The experiment (this is the go/no-go for the whole direction)

**1. Train with iron-pickaxe held out.** The head is blinded to
`MAKE_IRON_PICKAXE` (masked out of its loss); everything else trains normally.
```bash
python dreamerv3/main.py --configs craftax size50m \
    --logdir ~/logdir/rewcause_ironholdout \
    --rewcause True --rewcause_holdout MAKE_IRON_PICKAXE
```

**2. Probe + score** (no manual glue — `rewcause_probe` does the collection and
scoring). It runs greedy eval episodes, reads the head's per-feature φ
prediction (`sigmoid(logit)`) at every step where the held-out achievement
actually unlocks, and prints/writes the report:
```bash
python dreamerv3/main.py --configs craftax size50m \
    --script rewcause_probe \
    --rewcause True --rewcause_holdout MAKE_IRON_PICKAXE \
    --run.from_checkpoint ~/logdir/rewcause_ironholdout/ckpt
```
- **PASS** = the head recovered verb=MAKE, tool=PICKAXE, mat=IRON, tier>=3 at
  held-out iron states → compositional transfer works → build Phases 2–4.
- **FAIL** = it predicted ~zeros there → the feature space isn't linearly
  extrapolable; revisit φ before anything downstream.

**Reachability caveat (important).** The probe can only score states the eval
policy actually reaches. The prior vanilla DreamerV3 run never got past the
wood tier at ~1.1M steps, so a fresh agent may unlock iron-pickaxe rarely or
never — in which case the probe reports `inconclusive (0 states)`, not FAIL.
Two ways through: (a) train longer / probe a stronger checkpoint; or (b) run the
**identical** tier-extrapolation test on a target the agent *does* reach —
`--rewcause_holdout MAKE_STONE_PICKAXE` tests tier 1→2 instead of →3 and is
executable immediately. Recommend running the stone-holdout probe first as a
cheap early read on whether φ extrapolates at all.

## Knobs

- `rewcause: False` — disables the whole thing; agent reverts to exact baseline
  (loss/scale keys stay in sync automatically).
- `rewcause_none_weight` (default 0.1) — lower = emphasize achievement steps
  more against the majority of zero-reward steps.
- `rewcause_holdout` — comma-separated achievement names or indices to hold out
  (e.g. `MAKE_IRON_PICKAXE,MAKE_IRON_SWORD`). Empty = train on all (production).

## Verification status

Verified off-GPU (no JAX needed):
- numpy suite 8/8: φ table, ordered-tier extrapolability, weight table,
  achievement diffing, tech-tree decomposition, holdout scorer.
- **Achievement schema exact-matches installed Craftax v1.6.1** (67 names, enum
  value order) — parsed from `constants.py` via AST, so `validate_against_craftax`
  will pass on the box.
- All touched files byte-compile; `configs.yaml` parses and exposes the
  `craftax` preset + all `rewcause*` keys.
- Traced by hand through the JAX head: `rewcause(inp,2).loss(target)` reduces
  `(B,T,PHI_DIM)→(B,T)` via the head's `Agg(sum)` wrapper; `Binary.logp` casts
  the soft φ target to f32 (genuine per-feature BCE); `set(losses)==set(scales)`
  holds in both `rewcause` on/off cases; `ach` is ignored by the encoder
  (it selects `veckeys` from `enc_space`, which excludes `ach`).

Remaining (needs JAX/GPU), in order:
1. **Schema check:** `cd dreamerv3/dreamerv3 && python -c "from craftax_features import validate_against_craftax as v; v(); print('schema OK')"`
2. **Smoke run** (~1k steps): `python dreamerv3/main.py --configs craftax debug --rewcause True --rewcause_holdout MAKE_IRON_PICKAXE` — confirm `loss/rewcause` appears in metrics and is finite, and that `train`/`eval_only` on a non-rewcause config are unaffected.
3. **Go/no-go probe:** the two commands above (stone-holdout first, then iron).
