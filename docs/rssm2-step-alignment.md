# RSSM-2: training sees exactly what acting saw

Three bugs were found in an outside code review on 2026-10-11 and fixed the
same day. Every run with RSSM-2 before that date has them: honest map, B1, B2,
B3, v1–v1.3, v2 and v2-cur. Vanilla DreamerV3 has no RSSM-2 and is not
affected. Reruns with the fixes: `expB_mask_fix`, `honest_map_fix`, `mgr3_fix`,
`v2_fix`.

## A. Every non-movement action counted as a step down

`mapmodel.action_deltas` clipped the action to 0..4 before indexing the
movement table, whose entry 4 is DOWN. DO, SLEEP, PLACE_*, MAKE_* and every
other action (5–42) therefore added a step down to RSSM-2's movement input,
the sum of (dy, dx) over the window. RSSM-2 still reached 70–83% position
accuracy, since it also reads RSSM-1's features, but its path integration was
fed wrong numbers.

v2's memory mode passes raw action one-hots instead and was not affected.

**Fix:** only actions 1–4 move; every other action is (0, 0).

## B. A new episode inherited the old one's state

The acting step reset its window counters on `is_first` but kept the old
`deter2` until the window closed. For up to 7 steps the actor read the
previous episode's state. Then the GRU started the new episode's first update
from it.

**Fix:** `deter2` is cleared on the reset step itself.

## C. Training saw up to 7 steps of the future

Training used to work like this:
- The batch path pooled fixed 8-step windows (steps 8j to 8j+7), ran the GRU
  once per window, and gave each window's result to all 8 of its steps.
- Imagination starts and the manager's input read that per-step value. So at
  step 8j they saw a state built from steps up to 8j+7.

Acting worked differently. A step saw only the last window that had already
closed, so the state could be up to 7 steps old. On top of that:
- Online windows counted from the episode start, while batch windows followed
  the replay chunk.
- The batch reset flag only looked at the last step of each window, so a
  window could mix two episodes.

So the actor and critic trained on fresher state, part of it from the future,
than they ever got while acting. The map and position losses themselves were
fine: they grade the state after a window closes against the map at that same
step.

**Fix:** there is now one step function, `Agent._map_step`. Acting calls it
once per env step. Training scans the same function over the batch. Both
paths therefore read identical states:
- windows count from the episode start;
- a step sees the last closed window, which may include the step itself but
  never a later one;
- resets take effect immediately.

The replay context stores the window step (`map/count`), so a chunk that
resumes mid-window closes its first window on the same step the actor did.
The partial feature and move sums from before the chunk are not stored: they
would cost 5120 floats per step. That first window therefore averages the
steps it has, and its move sum is rescaled to a full window. This
approximation only affects the first window of each chunk.

The losses moved from per window to per step, counted only where a window
closes, which is about one step in 8. That keeps the old loss scale. In memory
mode, the targets 1, 2 and 4 windows ahead or back sit exactly 8, 16 and 32
steps away, because windows count from the episode start.

## Tests

`dreamerv3/test_mapmodel_step.py` (run `python -m pytest
dreamerv3/test_mapmodel_step.py`) checks:
- all 43 actions: only 1–4 move;
- a reset mid-window clears the state at once, and a new episode's states do
  not depend on whatever came before;
- the batch path (scan) equals the online path (step by step) across episode
  boundaries;
- windows close every 8 steps from each episode start;
- no step's state changes when only later inputs change;
- a chunk resumed at window step 5 closes after 3 steps.

## What to expect from the reruns

- **Map runs (B2, honest map, v1.2):** probably moderate shifts. The actor's
  map gate ended at ±0.4–0.6 in those runs, so the map is used, and the map
  losses were mostly sound.
- **v2:** the likeliest to move. Its actor opened the memory gate fully
  (~1.0) without a matching gain, which fits an actor leaning on leaked
  future information.
