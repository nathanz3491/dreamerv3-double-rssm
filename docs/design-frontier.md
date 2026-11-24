# Design: frontier dreaming (`family/frontier`)

Status: built 2026-11-24 (`dreamerv3/frontier.py`, `Agent._frontier_starts`,
`frontier_weighted`); not yet trained. Written 2026-11-24.

## 1. The idea in one paragraph

DreamerV3 teaches its actor and critic inside imagined rollouts. Every rollout
starts from a state in the current training batch, and the batch is drawn
uniformly from replay. Every episode starts in the easy early game and most
never get far, so almost all of that practice happens on parts the agent has
already mastered. Frontier dreaming keeps an archive of **frontier moments**:
points where an episode reached a progress level that replay rarely reaches.
About 20% of the imagined rollouts start from that archive. The world model's
training is unchanged; only *where the actor and critic practise* changes.

Pitch: **frontier practice without a resettable simulator; the world model is
the save file.**

## 2. Why now, and what is already taken

| work | what it does | how we differ |
|---|---|---|
| rekursiv.ai "frontier practice" (blog, 2026-10-07) | PPO on Craftax: saves the game and the agent's memory each time an episode reaches a new score level, and restarts 20% of training games from those saves, favouring rarely reached levels. With reward scaling it took the score from 53.7% to 72.5% at 20B steps. | Needs to save and restore the simulator. We restart inside the world model, so no simulator access is needed. |
| Anchor-and-Harvest Curriculum (arXiv 2610.15328) | Model-free Craftax-Full curriculum, unmasked; restarts from harvested progress states. | Same contrast: real-env restarts vs imagined ones. |
| Go-Explore (Ecoffet et al.), SCALAR (Zabounidis et al.) | Return to promising states, then explore from them. | Same contrast. |
| Prioritized replay (Schaul et al.; DreamerV3's own `replay.fracs.priority`) | Changes what the world model trains on. | We leave world-model training untouched and change only where imagination starts. |
| AGWM (arXiv 2605.06841) | Learns a prerequisite graph inside the world model. | No graph; we only reweight imagination starts. |

The scan (2026-11-23/24) found no model-based agent that chooses its
imagination starts by rarity of progress, and no model-based result on
Craftax-Full at all. Most 2026 papers were seen only as abstracts or
snippets; read the four closest in full before citing.

**Honesty rules kept:** no rule mask, no simulator resets, no privileged state.
The archive holds only the agent's own observations, actions and rewards.

## 3. Base models

1. **Main claim: this branch's plain DreamerV3** (`abl_plain`, 4.77 ± 0.28 on
   30 worlds; its 60-world score is in `~/logdir/archived_runs.txt` on the box).
   A clean comparison with no hand-written knowledge.
2. **Stacking: `market12_noq`** (6.47 ± 0.20 on 60 worlds, the best honest
   run). It carries v1.2's hand-written tech-tree potential and quota v1, so
   it only answers "does it stack?", not "does it work?". Code:
   `family/market` plus the frontier patch, in its own box code folder.

## 4. Mechanism

### 4.1 Progress level (reward-derived)

The level `L_t` of an episode at step `t` is the number of reward events so far
in the episode with `r >= 0.5`. In Craftax these are the achievement unlocks:
health pays 0.1 per point, so only a one-step gain of 5+ health (e.g. a potion)
could also cross 0.5. A test checks the count against `obs['ach']`, which is
used only as an oracle, not by the agent. If the two disagree often, use the
positive part of the reward net of health instead.

### 4.2 Rarity

The archive keeps, over the last `frontier.window` finished episodes (default
200, about 60k env steps at today's lifespans), the fraction `p(L)` of episodes
that reached level `L` or higher. A level is a **frontier level** while
`0 < p(L) <= frontier.pmax` (default 0.25). The best level ever reached always
counts as frontier, even if it was reached only once.

Early in training almost every level is rare, so the archive fills with
level-1 and level-2 moments. As the agent masters a level, its `p(L)` rises
above `pmax` and it stops being sampled. The frontier moves with the agent.

### 4.3 Archive

- **Event:** step `t` where an episode first reaches level `L` and `L` is a
  frontier level at that moment.
- **Window:** the 65 steps `[t-49, t+15]` (one train row: `batch_length +
  replay_context`) of observations, actions, rewards and flags, clipped to the
  episode. A window cut by the episode's end is padded,
  with a mask. Index 0 is marked `is_first`, so the model starts from a zero
  state there.
- **Storage:** copies, not pointers into replay. Replay evicts after 4e5 steps,
  and the rarest moments are the ones we most need to keep. One window is about
  2.2 MB (`vector` is 8268 float32), so the cap of 256 windows (at most 32 per
  level, FIFO within a level) is about 0.55 GB of host RAM. Training peaks at
  about 30 GB and `train_guard.sh` requires 54 GB free, so this fits.
- **Persistence:** saved next to the checkpoint as `frontier.npz` and reloaded
  on resume, because launches on this box do get restarted.
- **Fed by** a `driver.on_step` hook. It keeps a 64-step ring buffer per worker
  and finalises an event 31 steps later or at the episode's end.

### 4.4 Sampling

Each train step draws one window per batch row (16), so the frontier rows fit
the batch layout the JAX agent expects. It picks a frontier
level uniformly among the levels that have windows, then a window uniformly
within that level. This favours rare levels over common ones. Until the
archive holds any frontier window, the rows are passed with `valid = 0` and
contribute nothing, so jit shapes never change.

### 4.5 Re-encoding (no stale starts)

The frontier rows go through the **current** encoder and RSSM (`dyn.observe`)
under stop-gradient, starting from a zero state at index 0. The 32 steps before
the event rebuild the agent's memory of how it got there. This is the world
model's version of rekursiv's "save the agent's memory with the game". It never
uses latents cached at the time the window was recorded.

### 4.6 Imagination starts

- Normal starts are unchanged: `B x K = 16 x 64 = 1024`.
- Frontier starts are the posterior states at window positions 49..64 (the
  event and the 15 steps after it): `16 x 16 = 256`.
- Total `1280`, so **20% of starts come from the frontier**.
- Both sets are imagined together for `H = 15` steps with the same policy,
  reward, continue and value heads. Each start carries the same weight in the
  policy and value losses.
- Frontier rows give no world-model loss, no replay-value loss and no
  `replay_context` update. The world model's training data is exactly
  vanilla's.

### 4.7 Config

```yaml
agent:
  frontier:
    enabled: False
    starts: 16       # start positions per window (event and after); one
                     # window per batch row, 65 - 16 = 49 context steps
    window: 200      # episodes in the rarity estimate
    pmax: 0.25       # a level is frontier while reached by <= 25% of episodes
    cap: 256         # windows kept
    per_level: 32
    rmin: 0.5        # reward event threshold for a level step
    obs_keys: [vector]  # the encoder inputs stored in a window
```

### 4.8 Implementation notes

- **Plumbing.** `make_replay` attaches the archive to the train replay
  (`replay.frontier`). The train loop feeds it every env step, checkpoints it
  (`ckpt/frontier.npz`, compressed) and logs `frontier/*` stats. `make_stream`
  adds the rows to every batch as `fr/obs/*`, `fr/act/*`, `fr/reward`,
  `fr/is_first`, `fr/is_terminal`, `fr/valid` (extra spaces of the agent).
  Report batches (another length) get zero rows, and `report()` skips the
  frontier path.
- **Padding rows.** A row with no real frontier window (an empty archive, or
  positions after an episode's end) imagines from a copy of a normal start
  instead of from all-zero inputs. Its losses are masked out, but its rollouts
  do enter the return normalizers. So an enabled run with an empty archive is
  close to vanilla, not bit-identical; the RNG draws also differ.
- **Weighting.** `frontier_weighted` folds the frontier starts' losses into
  the (B, K) loss arrays so that the mean equals the per-start average over
  all real starts.

## 5. Diagnostics (logged every report)

- Archive: size; windows per level; `p(L)` table; current frontier levels; best
  level ever reached.
- Starts: frontier fraction actually used (should be 0.20 once filled).
- Imagined return and value at frontier starts vs normal starts.
- **World-model fit at the frontier.** The frontier rows are real data, so
  measure reward, continue and reconstruction error on them against normal
  rows. This is the main risk (§7) and the number we need to interpret any
  result.
- Achievement reach rates per level over training (from `obs['ach']` in
  replay). If the mechanism works, the levels the archive targets should be
  reached more often.

## 6. Runs and tests

### 6.1 Runs (1.1M env steps, `size50m`, one at a time, through `train_guard.sh`)

| order | run | base | what it answers |
|---|---|---|---|
| 1 | `frontier_plain` | plain | Does frontier dreaming beat plain DreamerV3? |
| 1b | `frontier_ctl` | plain | Only if run 1 wins: the same extra rows drawn uniformly from all archive windows, frontier or not. Is the gain from *where* the dreams start, or just from 25% more imagination? |
| 2 | `frontier_noq` | market12_noq | Does it stack on our best honest run? |
| 3 | `loopact_plain` | plain | The looped-actor side run (§8). |

Score each run on the 60 fixed worlds with `~/score_run.sh` and compare world
by world against its base. A gap under ~1.1 between two single runs is not
evidence on 30 worlds. On 60 worlds, use the paired SE.

**Success:** run 1 beats `abl_plain` by more than two paired SEs on 60
worlds, and run 1b falls short of run 1. Beyond the score, the reach rate of
at least one level above wood (stone pickaxe and up) should rise. Today no
plain run places stone at all.

### 6.2 Tests (CPU, through `safe_job.sh` on the box)

- Level counting: reward events `>= rmin`; health ticks don't count; resets at
  `is_first`.
- Rarity: `p(L)` over the window; the best level is always frontier; levels
  leave the frontier when mastered.
- Archive: window bounds, end-of-episode padding and mask, caps and FIFO,
  save/load round trip.
- Sampling: uniform over levels, then over windows; `valid = 0` before any
  window exists.
- Agent: `frontier.enabled: False` gives losses identical to `main`. The
  frontier rows get no world-model gradient (check gradient norms with frontier
  rows zeroed vs not). Start count is 1280 and the policy loss weights each
  start equally.
- Short CPU smoke run, then a GPU smoke run. If XLA faults the GPU (Xid 13),
  first try `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`.

## 7. Risks

- **The world model is weakest exactly at the frontier.** Rare states have few
  training samples, so the actor may learn to exploit model errors there.
  Mitigations: short horizon (`H = 15`); the frontier-fit diagnostics above;
  if the errors are large, a second arm that also gives the frontier rows a
  small world-model loss (this changes the data the model trains on, so it is
  reported separately).
- **Too little frontier early on.** In a 1.1M-step run, the levels above wood
  may be reached only a few times. The best level counts as frontier even when
  reached once, but if the archive holds only a handful of windows the actor
  may overfit to them. Watch windows per level.
- **Compute.** The frontier rows add a forward pass over 16 x 65 steps and 25%
  more imagination: expect +15-25% per train step. If memory is short on the
  4090, drop `starts` to 12 (about 16% of starts) before touching the base config.
- **Craftax's level-ups are coarse.** One level = one achievement, so "level 6"
  mixes very different progress (six easy items vs a stone pickaxe). If the
  archive fills with easy-item permutations, weight events by reward size
  (tiers 1/3/5/8) instead of counting. This stays reward-derived.

## 8. Side run: looped actor (not part of the family)

A patch, kept on `patch/looped-actor` off `main`. It tests whether more
computation per decision helps, with no extra parameters, in the style of
Thinking Machines' Inkling 2 Mini (2026-11-20), which loops its middle layers
with a gate that starts closed. Neighbours: Looped Actor (arXiv 2609.37432),
FlexLoop (2609.34488), Looped World Models (2606.18208), DRC (Guez et al.,
2019).

- The plain policy MLP is three layers (1024 units): `L1 -> L2 -> L3 -> head`.
- Looped: `h0 = L1(x)`; `m = 0`; three passes of `h = L2(h0 + g * m); m = h`;
  then `L3(h) -> head`. `g` is a learned per-unit gate, initialised to 0.
- With `g = 0` every pass gives `L2(h0)`, so at the start the actor is exactly
  plain. The only new parameters are the 1024 gate values.
- The world model and critic are unchanged. Fixed at 3 passes: no adaptive
  halting, which Looped Actor already covers.
- Test: `g = 0` gives the plain actor's outputs bit for bit.
- Expectation, stated in advance: little gain. rekursiv found the Craftax
  bottleneck is discovery (finding the ladder), not per-step reasoning.

## 9. Open questions before building

- Read in full: rekursiv's appendix (the frontier-practice ablations), AHC
  2610.15328, the Dreamer intrinsic-motivation study 2611.13113, AGWM
  2605.06841.
- Confirm that the original Craftax paper had no DreamerV3 baseline on
  Craftax-Full, and that no model-based Craftax-Full result exists.
- `abl_plain`'s 60-world score, from `~/logdir/archived_runs.txt` (the box
  was unreachable when this was written).
