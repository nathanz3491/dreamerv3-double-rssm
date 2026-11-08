# The agent the taste runs are built on

Taste (`docs/taste-prior.md`) is an add-on. The three taste runs put it on
three different agents, and this branch carries the code of all three:

| run | agent underneath | what that agent is |
|---|---|---|
| `abl_taste` | plain DreamerV3 | `main`'s agent, nothing added |
| `v2_taste` | v2 without the rule mask (`v2_norules`) | RSSM-2 as a memory, a manager over 16 learned goals, curiosity |
| `next_taste` | the next agent | RSSM-2 as a memory, a manager over the 13 named goals, curiosity, the rule mask |

The pieces are described here only as far as this branch has them. Their full
history (the v1-v1.3 managers, the map target, potential shaping, B1-B3)
lives on `family/v`, `family/rssm2` and `family/masked`, and all of it at tag
`archive/full-2026-11-08`. The design notes there are `docs/design-manager.md`
and `docs/design-v2.md` on `family/v`.

## RSSM-2 as a memory (`mapmodel.*`, `dreamerv3/mapmodel.py`)

A second, slower world model: a deterministic GRU (no KL, no sampling) that
ticks once per 8 env steps.

- **Input per tick:** RSSM-1's mean latent over the window, the 43-way action
  one-hots summed over the window (raw actions, no knowledge of what they
  do), and the step count.
- **Targets** (`target memory`, the only target here):
  - predict RSSM-1's mean latent 1, 2 and 4 ticks ahead (8/16/32 steps; MSE,
    head `memfut`);
  - reconstruct the observation vector 1, 2 and 4 ticks back (MSE, head
    `memrec`);
  - both only within one episode.
- **Into the actor:** RSSM-2's state (1024 numbers, gradient-stopped) times a
  learned gate that starts at 0 (`memory/gate`). `rew` and `con` never see it,
  so an error in RSSM-2 cannot fabricate imagined reward.
- **Inside imagination** RSSM-2 does not tick (it would be learning from
  imagined observations): every rollout reads its state frozen at the start.
- **Train equals act.** One step function, `Agent._map_step`, is called once
  per env step while acting and scanned over the batch in training, so both
  read identical states (`docs/rssm2-step-alignment.md`).

The module and its params are still called `mapmodel`: it began as a map
model (the map target is on `family/rssm2`), and the names are kept so old
checkpoints load.

## The manager (`manager.*`)

A manager policy over goals (feudal / Director family): a learned network,
trained in imagination and run reactively, never a planner.

```
              ┌─► head 'game' (achievements, health)   ──► BOTH actors
critic body ──┤
              └─► head 'goal' (progress + reach bonus) ──► bottom actor only
              (└─► head 'explore' (curiosity), when curiosity is on)
input: RSSM-1 feat, RSSM-2 state, goal one-hot, steps-since-set one-hot (32),
       reached-this-episode flags
```

- **Choosing.** The manager reads the actor's view plus RSSM-2's state and
  picks a goal. The goal is held until reached, or for `hold` (32) steps.
  Goals that already hold are masked out of the choice; NONE (goal 0) is
  always allowed and leaves the actor on the game reward alone.
- **Bottom actor.** Its input gains the goal one-hot and the steps since the
  goal was set. Its advantage is Â_game + `goal_weight` (0.5) × Â_goal
  (+ 0.2 × Â_explore with curiosity). Each stream has its own lambda-return
  and return normalisation (`imag_loss_streams`), so each counts by its
  weight, not its raw scale.
- **Manager actor.** No critic of its own: at imagined states 0 and 8 the
  game head scores every goal as if set right now, Q(s, g), and the manager
  follows the exact gradient Σ_g π(g|s) A(s, g) (`_mgr_loss_shared`). It is
  judged on the game head (plus 0.2 × the explore head with curiosity), so
  it cannot pay itself through the goal bonuses. Taste on the manager writes
  here.
- **Imagination resumes the replay's goal and step** (`goal` and `gphase`
  are stored in replay), so the replay value loss trains the game head on
  the same inputs the rollouts use. Taste reads its evidence from that
  replay loss.

`critic shared` is the only kind on this branch (`hold` must be > 0); the
v1/v1.1 manager with a critic of its own is on `family/v`.

### Named goals (`goals named`, `craftax_goals.py`; `next_taste`)

13 goals: NONE plus the tech-tree spine (place table, wood pickaxe and
sword, stone, furnace, stone pickaxe and sword, coal, iron, iron pickaxe and
sword, diamond). Each is a state to reach, judged from the observation
vector alone: for example "pickaxe tier ≥ 2 in the inventory" or "a table in
the 8-neighbourhood". The env emits `obs['goalphi']` (progress toward each
goal, a training target for the `gphi` head, so progress exists inside
imagination) and `obs['goalreach']` (which goals have read as reached this
episode). Needs `env.craftax.goals_obs True`. The goal list is game
knowledge; so is the progress formula.

### Learned goals (`goals learned`, `goalcodes.py`; `v2_taste`)

16 goals learned from experience: a codebook of latent-change types.
- Δ = feat(t+8) − feat(t) from replay, within one episode, gradient-stopped,
  is encoded (5120 → 256 → 64, on the unit sphere) and quantised as in a
  VQ-VAE; codes move by moving average, dead codes are revived onto recent
  valid changes (`goalbook`, outside the optimizer).
- Progress is the cosine between encode(feat_now − feat_at_goal_set) and the
  goal's code; the goal reward is the change in progress plus +1 on
  reaching. **Reached** means that change is classified as the goal's code
  (nearest code, cosine above 0.3).
- No goal masking and no reached flags: a learned change-type is never
  "already true".

## Curiosity (`curiosity.*`)

Five small MLP heads each predict the next posterior stoch (as
probabilities) from (latent, action taken), trained on replay with gradients
stopped. In imagination the variance of their predictions is the curiosity
reward, a third stream with its own critic head (`explore`) and
normalisation: weight 0.2 for the bottom actor, 0.2 on the manager's goal
values. It needs the manager (its critic). (v2-cur's `normalize` option is
on `family/v`.)

## The rule mask (`valid.mask`, `craftax_valid.py`) — counted as cheating

`env.craftax.valid_obs True` emits `obs['valid']`: which of the 43 actions
can do anything now, computed from the observation by the game's own rules.
With `agent.valid.mask True` an impossible action gets zero probability and
zero gradient: the true flags while acting, a head trained on them (`feas`)
inside imagination. The basic actions (NOOP, moves, DO) are never masked.
Because it reads the game's rules, **a masked run counts as cheating**.
Default off; `next_taste` used it (and `next_lam0`, its control). Taste over
actions was built as a knowledge-free stand-in for it (`v2_taste`).

## Taste (`taste.*`, `taste.py`)

See `docs/taste-prior.md`. In the code:
- `target actions`: a per-action logit bias, added in `Agent._masked` before
  the mask, so acting, the imagined rollouts and the actor loss all see the
  same prior. Evidence: the replay advantage at each real step, filed under
  the action taken.
- `target manager`: an internal reward `lam_mgr × rscale × relative taste`
  added to the manager's Q in `_mgr_loss_shared`; no bias acts. Evidence:
  `_taste_mgr_evidence`, the return after each real goal choice minus the
  manager's expected value there.
