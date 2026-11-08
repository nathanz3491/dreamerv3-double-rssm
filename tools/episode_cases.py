"""How B2's shortfalls show up inside its episodes: measured, with examples.

tools/diagnose_run.py says WHERE a run falls short (per-episode funnel, death
causes). This walks the same replay window step by step to say HOW: what the
agent does at the moments that matter, and keeps a few example episodes as
frame windows for tools/make_casebook.py to draw.

Everything is read from what the agent saw (the observation vector, decoded by
craftax_valid / craftax_map), the action it took, and the replay's own
observation-only ``valid`` mask. Inventory changes between step t and t+1 are
charged to the action taken at t.

  1 stone pickaxe   steps where MAKE_STONE_PICKAXE was possible (valid[12]):
                    how often it pressed it, and what it pressed instead.
                    Episodes that held wood + stone but never got there: was a
                    table in view, and where did the stone and wood go.
  2 wood pickaxe    episodes that placed a table but never made the pickaxe:
                    wood left after the table, later chances (valid[11]).
  3 survival        drinks and meals per episode; at death, how far the
                    nearest visible water / cow was; what it did while a
                    hostile mob was adjacent in its last steps.
  4 wasted presses  what DO was aimed at, and runs of a move repeated into a
                    blocked tile.

  python tools/episode_cases.py ~/logdir/expB_mask --steps 150000
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

ACTIONS = (
    'NOOP LEFT RIGHT UP DOWN DO SLEEP PLACE_STONE PLACE_TABLE PLACE_FURNACE '
    'PLACE_PLANT MAKE_WOOD_PICKAXE MAKE_STONE_PICKAXE MAKE_IRON_PICKAXE '
    'MAKE_WOOD_SWORD MAKE_STONE_SWORD MAKE_IRON_SWORD REST DESCEND ASCEND '
    'MAKE_DIAMOND_PICKAXE MAKE_DIAMOND_SWORD MAKE_IRON_ARMOUR '
    'MAKE_DIAMOND_ARMOUR SHOOT_ARROW MAKE_ARROW CAST_FIREBALL CAST_ICEBALL '
    'PLACE_TORCH DRINK_POTION_RED DRINK_POTION_GREEN DRINK_POTION_BLUE '
    'DRINK_POTION_PINK DRINK_POTION_CYAN DRINK_POTION_YELLOW READ_BOOK '
    'ENCHANT_SWORD ENCHANT_ARMOUR MAKE_TORCH LEVEL_UP_DEXTERITY '
    'LEVEL_UP_STRENGTH LEVEL_UP_INTELLIGENCE ENCHANT_BOW').split()
WOOD_PICK, STONE_PICK, PLACE_TABLE = 11, 12, 8
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
CY, CX = M.OBS_H // 2, M.OBS_W // 2
WATER, TABLE = 3, 11
BLOCK_NAMES = {0: 'invalid', 1: 'out of bounds', 2: 'grass', 3: 'water',
               4: 'stone', 5: 'tree', 6: 'wood', 7: 'path', 8: 'coal',
               9: 'iron', 10: 'diamond', 11: 'table', 12: 'furnace',
               13: 'sand', 14: 'lava', 15: 'plant', 16: 'ripe plant',
               M.UNKNOWN: 'dark'}
WINDOW = 10            # frames kept per example


def load(logdir, steps):
  """The last ``steps`` steps of a single-env replay, as one stream."""
  chunks = sorted((pathlib.Path(logdir).expanduser() / 'replay').glob('*.npz'))
  picked, need = [], steps
  for path in reversed(chunks):
    picked.append(path)
    need -= 1024
    if need <= 0:
      break
  keys = ('vector', 'action', 'ach', 'is_first', 'is_terminal')
  parts = {k: [] for k in keys + ('valid',)}
  for path in reversed(picked):
    with np.load(path) as z:
      for k in keys:
        parts[k].append(z[k])
      # Runs before experiment B did not record the mask; it is a function of
      # the observation, so recompute it.
      parts['valid'].append(z['valid'] if 'valid' in z.files else
                            np.stack([V.valid_actions(v) for v in z['vector']]))
  return {k: np.concatenate(v) for k, v in parts.items()}


def split(data):
  """[(start, end)] of complete episodes that end in a terminal step."""
  firsts = np.flatnonzero(data['is_first'])
  out = []
  for a, b in zip(firsts, list(firsts[1:]) + [len(data['is_first'])]):
    term = np.flatnonzero(data['is_terminal'][a:b])
    if len(term):
      out.append((a, a + term[0] + 1))
  return out


def nearest(blocks, ids, mask=None):
  """Chebyshev distance from the agent to the nearest visible tile of ``ids``."""
  hit = np.isin(blocks, ids) if mask is None else mask
  ys, xs = np.nonzero(hit)
  if not len(ys):
    return None
  return int(np.max(np.abs(np.stack([ys - CY, xs - CX])), 0).min())


def adjacent(view, key):
  return bool(view[key][CY - 1:CY + 2, CX - 1:CX + 2].any())


def facing(st):
  dy, dx = V._FACING[st['direction']]
  return CY + dy, CX + dx


class Episode:
  """Per-step decoded facts for one episode."""

  def __init__(self, data, a, b):
    self.a, self.b = a, b
    self.vec = data['vector'][a:b]
    self.act = data['action'][a:b].astype(int)
    self.valid = data['valid'][a:b] > 0.5
    self.ach = (data['ach'][a:b] > 0.5).any(0)
    self.st = [V.decode_stats(v) for v in self.vec]
    self.view = [M.decode_view(v) for v in self.vec]

  def __len__(self):
    return len(self.act)

  def delta(self, t, key):
    return self.st[t + 1][key] - self.st[t][key] if t + 1 < len(self) else 0


def frames(ep, ts, note):
  ts = [t for t in ts if 0 <= t < len(ep)]
  return dict(start=int(ep.a), steps=[int(t) for t in ts],
              actions=[ACTIONS[ep.act[t]] for t in ts],
              vectors=ep.vec[ts].astype(np.float16), note=note)


def stone_pickaxe(eps):
  out, examples = {}, []
  # (a) the moments it could have made one
  chances = collections.Counter()
  pressed, opp_eps = 0, 0
  wood_chances, wood_pressed = 0, 0
  for ep in eps:
    opp = np.flatnonzero(ep.valid[:, STONE_PICK])
    if len(opp):
      opp_eps += 1
    for t in opp:
      chances[ACTIONS[ep.act[t]]] += 1
    pressed += int((ep.act[opp] == STONE_PICK).sum())
    w = np.flatnonzero(ep.valid[:, WOOD_PICK])
    wood_chances += len(w)
    wood_pressed += int((ep.act[w] == WOOD_PICK).sum())
  total = sum(chances.values())
  out['chance_steps'] = total
  out['chance_episodes'] = opp_eps
  out['pressed_when_possible'] = pressed / max(total, 1)
  out['wood_pickaxe_pressed_when_possible'] = wood_pressed / max(wood_chances, 1)
  out['instead'] = {k: v / max(total, 1) for k, v in chances.most_common(8)}

  # (b) held both, never stood at a table with them
  stranded = [ep for ep in eps
              if any(s['wood'] >= 1 and s['stone'] >= 1 for s in ep.st)
              and not ep.valid[:, STONE_PICK].any()]
  table_seen = 0
  dist = []
  for ep in stranded:
    d = [nearest(ep.view[t]['blocks'], [TABLE]) for t in range(len(ep))
         if ep.st[t]['wood'] >= 1 and ep.st[t]['stone'] >= 1]
    d = [x for x in d if x is not None]
    if d:
      table_seen += 1
      dist.append(min(d))
  out['stranded_episodes'] = len(stranded)
  out['stranded_table_in_view'] = table_seen / max(len(stranded), 1)
  out['stranded_closest_table'] = float(np.median(dist)) if dist else None

  # (c) where the ingredients went, in every episode that collected stone
  sinks = {'stone': collections.Counter(), 'wood': collections.Counter()}
  for ep in eps:
    for t in range(len(ep) - 1):
      for k in sinks:
        if ep.delta(t, k) < 0:
          sinks[k][ACTIONS[ep.act[t]]] += -ep.delta(t, k)
  out['sinks'] = {k: {a: n / max(sum(c.values()), 1)
                      for a, n in c.most_common(6)} for k, c in sinks.items()}

  # (d) each unbroken run of possible steps is one chance; how did it end
  ends, lengths, sword = collections.Counter(), [], None
  for ep in eps:
    v, t = ep.valid[:, STONE_PICK], 0
    while t < len(ep):
      if not v[t]:
        t += 1
        continue
      s = t
      while t < len(ep) and v[t]:
        t += 1
      lengths.append(t - s)
      a = int(ep.act[t - 1])
      if a == STONE_PICK:
        ends['made it'] += 1
      elif t >= len(ep):
        ends['episode ended'] += 1
      elif ep.st[t]['wood'] < 1 or ep.st[t]['stone'] < 1:
        ends[f'ingredient spent on {ACTIONS[a]}'] += 1
        if sword is None and a == 14:
          sword = (ep, s)
      else:
        ends['walked away'] += 1
  n = max(len(lengths), 1)
  out['chances'] = len(lengths)
  out['chance_length_median'] = float(np.median(lengths)) if lengths else None
  out['chance_ended'] = {k: v / n for k, v in ends.most_common()}

  if sword:
    ep, s = sword
    examples.append(dict(flaw=1, title='At the table with wood and stone; '
                         'makes the wood sword instead', **frames(
                             ep, range(s - 6, s + WINDOW - 6),
                             'the sword spends its only wood; the pickaxe '
                             'chance is gone')))
  for ep in stranded:
    ts = [t for t in range(len(ep)) if ep.st[t]['wood'] >= 1
          and ep.st[t]['stone'] >= 1
          and (nearest(ep.view[t]['blocks'], [TABLE]) or 99) <= 3]
    if ts:
      t0 = ts[0]
      examples.append(dict(flaw=1, title='Holding both, a table in sight; '
                           'walks off', **frames(
                               ep, range(t0, t0 + 3 * WINDOW, 3),
                               'every third step shown')))
      break
  return out, examples


def wood_pickaxe(eps):
  # Every table, and whether one already stood in view when it was placed.
  held, per_ep, repeat_seen, repeats, example = [], [], 0, 0, None
  for ep in eps:
    ts = [t for t in range(len(ep) - 1)
          if ep.act[t] == PLACE_TABLE and ep.delta(t, 'wood') < 0]
    per_ep.append(len(ts))
    for i, t in enumerate(ts):
      held.append(ep.st[t]['wood'])
      if i:
        repeats += 1
        seen = nearest(ep.view[t]['blocks'], [TABLE]) is not None
        repeat_seen += seen
        if example is None and seen and not ep.ach[5]:
          example = dict(flaw=2, title='Places a second table beside the '
                         'first', **frames(ep, range(t - 6, t + WINDOW - 6),
                                           'both logs go into another table; '
                                           'nothing left for the pickaxe'))
  tables = dict(per_episode=float(np.mean(per_ep)),
                episodes_with_two_or_more=float(np.mean(np.array(per_ep) >= 2)),
                wood_held_when_placing=dict(collections.Counter(map(int, held))),
                repeats_with_a_table_in_view=repeat_seen / max(repeats, 1))

  leak = [ep for ep in eps if ep.ach[1] and not ep.ach[5]]
  after, again, chance = [], 0, 0
  for ep in leak:
    placed = np.flatnonzero((ep.act == PLACE_TABLE)
                            & np.array([ep.delta(t, 'wood') < 0
                                        for t in range(len(ep))]))
    if not len(placed):
      continue
    t0 = int(placed[0])
    after.append(ep.st[min(t0 + 1, len(ep) - 1)]['wood'])
    again += any(s['wood'] >= 1 for s in ep.st[t0 + 1:])
    chance += bool(ep.valid[t0 + 1:, WOOD_PICK].any())
  n = max(len(after), 1)
  out = dict(tables=tables, episodes=len(leak),
             wood_left_after_table=dict(collections.Counter(map(int, after))),
             zero_wood_after_table=sum(a == 0 for a in after) / n,
             got_wood_again=again / n, had_a_later_chance=chance / n)
  return out, [example] if example else []


def survival(eps):
  drinks, meals, examples = [], [], []
  thirst, mobs = [], collections.Counter()
  for ep in eps:
    drinks.append(sum(ep.delta(t, 'drink') > 0.5 for t in range(len(ep))))
    meals.append(sum(ep.delta(t, 'food') > 0.5 for t in range(len(ep))))
    zero = [t for t, s in enumerate(ep.st) if s['drink'] <= V.EPS]
    if zero and all(ep.st[t]['food'] > V.EPS for t in range(zero[0] + 1)):
      t0 = zero[0]
      d = nearest(ep.view[t0]['blocks'], [WATER])
      thirst.append(d)
      if not examples and d is not None and d <= 3:
        examples.append(dict(flaw=3, title='Drink at zero, water '
                             f"{d} tile{'s' * (d > 1)} away", **frames(
                                 ep, range(t0 - 6, t0 + WINDOW - 6),
                                 'it does not turn to the water')))
    tail = range(max(0, len(ep) - 10), len(ep))
    for t in tail:
      if adjacent(ep.view[t], 'hostile'):
        fy, fx = facing(ep.st[t])
        a = ep.act[t]
        if a == 5 and ep.view[t]['hostile'][fy, fx]:
          mobs['attacks it (DO facing the mob)'] += 1
        elif a in MOVES:
          mobs['moves'] += 1
        elif a == 5:
          mobs['DO aimed elsewhere'] += 1
        else:
          mobs[ACTIONS[a]] += 1
  for ep in eps:
    if adjacent(ep.view[-1], 'hostile') and len(examples) < 2:
      examples.append(dict(flaw=3, title='Killed beside a hostile mob',
                           **frames(ep, range(len(ep) - WINDOW, len(ep)),
                                    'last steps of the episode')))
  near = [d for d in thirst if d is not None]
  n = sum(mobs.values())
  # Drink starts full and only drains; if it is never topped up, the step it
  # reaches zero is a clock the policy is not managing.
  dry = [next((t for t, s in enumerate(ep.st) if s['drink'] <= V.EPS), None)
         for ep, k in zip(eps, drinks) if k == 0]
  dry = [t for t in dry if t is not None]
  out = dict(drinks_per_episode=float(np.mean(drinks)),
             meals_per_episode=float(np.mean(meals)),
             episodes_never_drinking=float(np.mean(np.array(drinks) == 0)),
             never_drinking_dry_at_step=float(np.median(dry)) if dry else None,
             never_drinking_lifespan=float(np.median(
                 [len(ep) for ep, k in zip(eps, drinks) if k == 0])),
             drinking_lifespan=float(np.median(
                 [len(ep) for ep, k in zip(eps, drinks) if k > 0])),
             thirst_episodes=len(thirst),
             water_in_view_when_drink_hit_zero=len(near) / max(len(thirst), 1),
             water_within_3=sum(d <= 3 for d in near) / max(len(thirst), 1),
             mob_adjacent_last10={k: v / max(n, 1)
                                  for k, v in mobs.most_common()})
  return out, examples


def wasted(eps):
  target = collections.Counter()
  streaks, steps, example = 0, 0, None
  for ep in eps:
    steps += len(ep)
    for t in range(len(ep)):
      if ep.act[t] == 5:
        fy, fx = facing(ep.st[t])
        v = ep.view[t]
        if v['hostile'][fy, fx] or v['passive'][fy, fx]:
          target['a mob'] += 1
        else:
          target[BLOCK_NAMES.get(int(v['blocks'][fy, fx]), 'other')] += 1
    run = 0
    for t in range(len(ep)):
      a = int(ep.act[t])
      blocked = False
      if a in MOVES:
        dy, dx = MOVES[a]
        ahead = int(ep.view[t]['blocks'][CY + dy, CX + dx])
        blocked = ahead in M.SOLID_IDS or ahead == WATER
      same = t and a == ep.act[t - 1]
      run = run + 1 if blocked and same else (1 if blocked else 0)
      if run == 8:
        streaks += 1
        if example is None:
          example = dict(flaw=4, title='Walks into the same blocked tile',
                         **frames(ep, range(t - 7, t + 3),
                                  'the move is repeated into a solid tile'))
  n = sum(target.values())
  out = dict(do_target={k: v / n for k, v in target.most_common(10)},
             do_share=n / steps,
             blocked_streaks_per_1000_steps=1000 * streaks / steps)
  return out, [example] if example else []


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logdir')
  ap.add_argument('--steps', type=int, default=150_000)
  ap.add_argument('--out', default='~/logdir/cases')
  args = ap.parse_args()
  data = load(args.logdir, args.steps)
  eps = [Episode(data, a, b) for a, b in split(data)]
  print(f'{len(eps)} episodes')
  result, examples = dict(run=pathlib.Path(args.logdir).name,
                          episodes=len(eps)), []
  for name, fn in (('stone_pickaxe', stone_pickaxe),
                   ('wood_pickaxe', wood_pickaxe),
                   ('survival', survival), ('wasted', wasted)):
    out, ex = fn(eps)
    result[name] = out
    examples += ex
    print(name, json.dumps(out, indent=1))
  outdir = pathlib.Path(args.out).expanduser()
  outdir.mkdir(parents=True, exist_ok=True)
  (outdir / 'cases.json').write_text(json.dumps(result, indent=1))
  np.savez_compressed(outdir / 'examples.npz', **{
      f'{i}_vectors': e.pop('vectors') for i, e in enumerate(examples)})
  (outdir / 'examples.json').write_text(json.dumps(examples, indent=1))
  print('wrote', outdir, len(examples), 'examples')


if __name__ == '__main__':
  main()
