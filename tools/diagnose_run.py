"""Where a run falls short: tech-tree funnel, stone-pickaxe ingredients, deaths.

Reads the last N steps of a run's replay chunks -- every training episode in
that window -- and decodes each step's observation with the same observation-
only readers the agent's targets use (craftax_valid.decode_stats,
craftax_map.decode_view). So everything here is what the agent itself could
see; nothing reads a game state.

  funnel       of the episodes that reached each tech-tree rung, how many
               reached the next
  stone pickaxe  the recipe is one wood + one stone at a table. Per episode:
               did it ever hold both at once, and was it ever at a table while
               holding both -- separating "never gathered both" from "never
               brought them to a table" from "stood there and didn't press"
  deaths       what ran out first (as death_eval), and at the moment of death:
               asleep, standing in lava, hostile mob adjacent, below surface

  python tools/diagnose_run.py ~/logdir/expB_mask --steps 150000
"""

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, '.')
from dreamerv3 import craftax_map as M
from dreamerv3 import craftax_valid as V

TABLE, LAVA = 11, 14
AROUND = V._AROUND
# Achievement indices (craftax.craftax.constants.Achievement values).
ACH = dict(COLLECT_WOOD=0, PLACE_TABLE=1, MAKE_WOOD_PICKAXE=5, COLLECT_STONE=9,
           PLACE_STONE=10, MAKE_STONE_PICKAXE=13, PLACE_FURNACE=16,
           COLLECT_COAL=17, COLLECT_IRON=18, ENTER_DUNGEON=29)
FUNNEL = ['COLLECT_WOOD', 'PLACE_TABLE', 'MAKE_WOOD_PICKAXE', 'COLLECT_STONE',
          'PLACE_FURNACE', 'MAKE_STONE_PICKAXE', 'COLLECT_IRON']


class Episode:

  def __init__(self):
    self.ach = np.zeros(67, bool)
    self.length = 0
    self.both = False            # ever held >= 1 wood and >= 1 stone at once
    self.both_at_table = False   # ... while a table was in the 8-neighbourhood
    self.first_zero = {}         # meter -> step it first hit zero
    self.end = None              # facts at the final step


def table_near(view):
  b = view['blocks']
  cy, cx = M.OBS_H // 2, M.OBS_W // 2
  return any(int(b[cy + dy, cx + dx]) == TABLE for dy, dx in AROUND)


def hostile_near(view):
  h = view['hostile']
  cy, cx = M.OBS_H // 2, M.OBS_W // 2
  return bool(h[cy - 1:cy + 2, cx - 1:cx + 2].any())


def episodes(logdir, steps):
  chunks = sorted((pathlib.Path(logdir).expanduser() / 'replay').glob('*.npz'))
  need, picked = steps, []
  for path in reversed(chunks):
    picked.append(path)
    need -= 1024
    if need <= 0:
      break
  done, ep = [], None
  for path in reversed(picked):
    with np.load(path) as z:
      vec, ach = z['vector'], z['ach'] > 0.5
      first, term = z['is_first'], z['is_terminal']
    for t in range(len(first)):
      if first[t]:
        if ep is not None and ep.end is not None:
          done.append(ep)
        ep = Episode()
      if ep is None:
        continue                       # window started mid-episode
      st = V.decode_stats(vec[t])
      ep.ach |= ach[t]
      ep.length += 1
      if st['wood'] >= 1 and st['stone'] >= 1:
        ep.both = True
        if not ep.both_at_table and table_near(M.decode_view(vec[t])):
          ep.both_at_table = True
      for meter in ('food', 'drink', 'energy'):
        if st[meter] <= 1e-3 and meter not in ep.first_zero:
          ep.first_zero[meter] = ep.length
      if term[t]:
        view = M.decode_view(vec[t])
        cy, cx = M.OBS_H // 2, M.OBS_W // 2
        ep.end = dict(sleeping=st['sleeping'], level=st['level'],
                      lava=int(view['blocks'][cy, cx]) == LAVA or LAVA in {
                          int(view['blocks'][cy + dy, cx + dx])
                          for dy, dx in AROUND},
                      hostile=hostile_near(view))
  return done


def report(name, eps):
  n = len(eps)
  out = dict(run=name, episodes=n)
  print(f'\n=== {name}: {n} terminal episodes in the window ===')
  print('funnel (share of all episodes; then of those that reached the previous rung)')
  prev = None
  out['funnel'] = {}
  for rung in FUNNEL:
    got = np.array([e.ach[ACH[rung]] for e in eps])
    cond = got[prev].mean() if prev is not None and prev.any() else float('nan')
    out['funnel'][rung] = dict(all=float(got.mean()), given_previous=float(cond))
    print(f'  {rung:<20} {100 * got.mean():5.1f}%   given previous {100 * cond:5.1f}%')
    prev = got
  dungeon = np.mean([e.ach[ACH['ENTER_DUNGEON']] for e in eps])
  print(f'  ENTER_DUNGEON        {100 * dungeon:5.1f}%')
  out['funnel']['ENTER_DUNGEON'] = dict(all=float(dungeon))

  stone = [e for e in eps if e.ach[ACH['COLLECT_STONE']]]
  both = np.mean([e.both for e in stone]) if stone else float('nan')
  table = np.mean([e.both_at_table for e in stone]) if stone else float('nan')
  made = np.mean([e.ach[ACH['MAKE_STONE_PICKAXE']] for e in stone]) if stone else float('nan')
  print(f'stone pickaxe, over the {len(stone)} episodes that collected stone:')
  print(f'  ever held wood and stone together   {100 * both:5.1f}%')
  print(f'  ... while next to a table           {100 * table:5.1f}%')
  print(f'  made the stone pickaxe              {100 * made:5.1f}%')
  out['stone_pickaxe'] = dict(episodes=len(stone), held_both=float(both),
                              both_at_table=float(table), made=float(made))

  causes = collections.Counter()
  facts = collections.Counter()
  for e in eps:
    if e.first_zero:
      causes[min(e.first_zero, key=e.first_zero.get) + ' ran out'] += 1
    else:
      causes['health lost, no meter at zero'] += 1
    for k in ('sleeping', 'lava', 'hostile'):
      facts[k] += int(bool(e.end[k]))
    facts['below surface'] += int(e.end['level'] > 0)
  lengths = np.array([e.length for e in eps])
  print(f'deaths: lifespan mean {lengths.mean():.0f}, median {np.median(lengths):.0f}')
  for k, v in causes.most_common():
    print(f'  {k:<32} {100 * v / n:5.1f}%')
  print('  at the moment of death:')
  for k in ('sleeping', 'hostile', 'lava', 'below surface'):
    print(f'    {k:<16} {100 * facts[k] / n:5.1f}%')
  out['deaths'] = dict(lifespan_mean=float(lengths.mean()),
                       lifespan_median=float(np.median(lengths)),
                       causes={k: v / n for k, v in causes.items()},
                       at_death={k: facts[k] / n for k in facts})
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logdirs', nargs='+')
  ap.add_argument('--steps', type=int, default=150_000)
  ap.add_argument('--out', default='~/logdir/diagnosis.json')
  args = ap.parse_args()
  results = [report(pathlib.Path(d).name, episodes(d, args.steps))
             for d in args.logdirs]
  pathlib.Path(args.out).expanduser().write_text(json.dumps(results, indent=1))
  print('\nwrote', args.out)


if __name__ == '__main__':
  main()
