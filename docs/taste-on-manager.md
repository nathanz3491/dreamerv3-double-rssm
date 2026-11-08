# Taste on the manager: why `lam_mgr 0` is the no-taste control here

Taste itself (over actions, and on the manager with `lam_mgr` above 0) is
studied on `family/taste`; its full write-up is `docs/taste-prior.md` there and
at tag `archive/full-2026-11-08`. This note keeps only what the next agent
needs.

## Why taste is in this branch at all

The next agent has no taste. `taste.py` and the `agent.taste.*` keys are here
for one reason: the exact flags of `next_lam0` and `next_nomask` set
`--agent.taste.enabled True ... --agent.taste.lam_mgr 0.0`, and those runs must
still run on this branch. (`next_notaste`, the same agent without the taste
flags, crashed three times on GPU faults, Xid 69 and 13, and has no result;
`next_lam0` stands in for it.) With `lam_mgr 0` taste adds exactly zero:

- the manager's Q is `Q + lam_mgr * rscale * relative_taste`, so the term is
  `0 * rscale * rel`;
- taste draws no random numbers, so it cannot shift the random stream either;
- it only produces the `taste/*` metrics (the per-goal taste it would have
  applied), which is why the run is a control that also shows what taste
  would have said. Taste's own variables are extra checkpoint state, but
  nothing reads them.

Only `taste.target manager` is accepted on this branch; taste over actions
(which uses `lam`, `actions` and its own defaults) is on `family/taste`.

## What taste on the manager is

An internal reward for choosing each goal: `Q'(s, g) = Q(s, g) + lam_mgr x
rscale x (taste(g) - mean taste of the 12 goals)`, where taste is learned from
the real return after each choice, compared with the manager's expected value
(`_taste_mgr_evidence`). NONE never gets a taste. Settings of the study:
`center True`, `pos = neg = 1.43`, `lam_mgr 1.0`, `optimism 0`.

## Result: it hurt (`next_taste` vs `next_lam0`)

Test achievements, 60 fixed worlds, paired by world. Both runs are masked
(B2 rule mask), so both are counted as cheating; the comparison is of taste,
not of the agent's honesty.

| run | 30 worlds | 60 worlds |
|---|---|---|
| `next_lam0` (`lam_mgr 0`, no taste) | 11.20 ± 0.41 | 11.30 ± 0.27 |
| `next_taste` (`lam_mgr 1`) | 9.80 ± 0.42 | 9.38 ± 0.34 |

`next_lam0` leads by +1.92 ± 0.41 over 60 worlds, and trains to 9.92 against
9.22 at 1.08M steps, ahead at every checkpoint from 700k. One seed each, so
the size of the gap carries seed noise on top; the direction matches every
diagnostic below. `next_taste` never makes a stone pickaxe or sword on 60
worlds; `next_lam0` makes the stone sword in 50% and the stone pickaxe in 30%
of the first 30 worlds' episodes.

## Why taste hurt (four measured links)

1. **Taste swamps the critic from its first update.** The manager's advantage
   is `(Q - sum pi Q) / rscale`. The critic barely separates goals (`q_spread`
   0.02-0.1, so 0.003-0.013 after dividing by `rscale` of about 7.6), while
   taste adds `lam_mgr x relative taste` directly: 0.2 after the first fold,
   0.9 by the end, 15 to 300 times the game signal. Goal entropy fell from
   2.55 to 0.004 at the first fold, 6.7k env steps in.
2. **The lock is permanent.** Only a chosen goal gets evidence. Once one goal
   wins, the fold holds only the winner, so its centered evidence is about 0
   and the others keep their taste. `place_table` went 0.20 to 0.82 and the
   rest froze at -0.075.
3. **Relative taste makes NONE the second choice.** A goal that already holds
   is blocked from the choice, and `PLACE_TABLE` holds whenever a table is
   near. NONE has taste 0 and every other goal sits at -0.075, so near a table
   the manager picks NONE: over the last 160k steps, NONE took 60% of
   decisions in `next_taste` against 27% in `next_lam0`.
4. **Told "be near a table", the actor builds tables, not tools.** The goal
   reward is paid on reaching the goal, which placing a table does at once.
   Per episode (last 160k steps of replay): place table 4.63 vs 3.53; make
   stone pickaxe 0.00 vs 0.06; make stone sword 0.00 vs 0.32. The goals that
   point past the table never get proposed in `next_taste`; the control's
   manager proposes all eleven, and that is where its lead comes from.

What a working version would need (not built): taste in units of the critic's
own goal spread rather than `rscale`; evidence for unchosen goals, or decay
back toward neutral; NONE treated like any other goal. Whether taste should be
kept, fixed or dropped is decided on `family/taste`.
