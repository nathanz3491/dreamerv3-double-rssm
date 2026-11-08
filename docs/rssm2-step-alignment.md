# RSSM-2: training sees exactly what acting saw

Three bugs were found in an outside code review on 2026-10-11 and fixed the
same day. Every RSSM-2 run before that date had them (honest map, B1-B3,
v1.x, v2); no run of this branch is affected: `next_lam0`, `next_notaste` and
`next_nomask` were all trained on the pre-split code (`49a6bc3`), which has the
fixes. Runs from before 2026-10-11 used buggy RSSM-2 code that no branch
reproduces exactly. Vanilla DreamerV3 has no RSSM-2 and is not affected.
Reruns with the fixes: `expB_mask_fix`, `honest_map_fix`, `mgr3_fix`,
`v2_fix` (on the other families' branches).

## A. Every non-movement action counted as a step down (map target only)

`mapmodel.action_deltas` clipped the action to 0..4 before indexing the
movement table, so DO, SLEEP, crafting and every other action added a step
down to RSSM-2's movement input. This branch feeds the raw 43-way action
one-hots instead (memory target), so it never had the bug and the movement
table is not here; the fix (only actions 1-4 move) is on `family/rssm2`.

## B. A new episode inherited the old one's state

The acting step reset its window counters on `is_first` but kept the old
`deter2` until the window closed. For up to 7 steps the actor read the
previous episode's state, and the GRU started the new episode's first update
from it.

**Fix:** `deter2` is cleared on the reset step itself. The reset step's
action, the old episode's last, is also kept out of the new window.

## C. Training saw up to 7 steps of the future

Training used to pool fixed 8-step windows (steps 8j to 8j+7), run the GRU once
per window, and give each window's result to all 8 of its steps. Imagination
starts and the manager's input read that per-step value, so at step 8j they
saw a state built from steps up to 8j+7.

Acting worked differently. A step saw only the last window that had already
closed, so the state could be up to 7 steps old. On top of that, online
windows counted from the episode start while batch windows followed the replay
chunk, and the batch reset flag only looked at the last step of each window,
so a window could mix two episodes. The actor and critic therefore trained on
fresher state, part of it from the future, than they ever got while acting.

**Fix:** there is one step function, `Agent._map_step`. Acting calls it once
per env step; training scans the same function over the batch. Both paths
read identical states:
- windows count from the episode start;
- a step sees the last closed window, which may include the step itself but
  never a later one;
- resets take effect immediately.

The replay context stores the window step (`map/count`), so a chunk that
resumes mid-window closes its first window on the same step the actor did. The
partial feature and action sums from before the chunk are not stored (the
feature sum alone would cost 5120 floats per step), so that first window
averages the steps it has and its action sum is rescaled to a full window.
This only affects the first window of each chunk.

The memory losses moved from per window to per step, counted only where a
window closes (about one step in 8), which keeps the old loss scale. The
targets 1, 2 and 4 windows ahead or back sit exactly 8, 16 and 32 steps away,
because windows count from the episode start.

## Tests

`dreamerv3/test_mapmodel_step.py` (needs JAX) checks:
- a reset mid-window clears the state at once, the reset step drops the
  previous episode's last action, and a new episode's states do not depend on
  whatever came before;
- the batch path (scan) equals the online path (step by step) across episode
  boundaries;
- windows close every 8 steps from each episode start;
- no step's state changes when only later inputs change;
- a chunk resumed at window step 5 closes after 3 steps.

The map-delta test of the pre-split file (all 43 actions: only 1-4 move)
belongs to the map target and is on `family/rssm2`.
