"""B2 casebook: each shortfall measured across runs, then shown in an episode.

Reads docs/cases/ (tools/episode_cases.py run on the box: one <run>.json of
measurements per run, plus examples.json / examples.npz -- frame windows from
B2's replay) and writes docs/b2-casebook.pdf.

Frames are the agent's own 9x11 observation decoded back to tiles and drawn
with Craftax's textures, so they show exactly what the agent saw -- not the
true game state. Needs the craftax package for its texture files, so run it
from an environment that has it installed (e.g. localplay/.venv):

  python tools/make_casebook.py
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import make_comparison_report as R                      # palette, page, wrap
from dreamerv3 import craftax_map as M
from dreamerv3 import craftax_valid as V

plt = R.plt
CASES = R.ROOT / 'docs' / 'cases'
OUT = R.ROOT / 'docs' / 'b2-casebook.pdf'
ASSETS = pathlib.Path(importlib.util.find_spec('craftax').origin).parent / (
    'craftax/assets')
PX = 16

# constants.load_all_textures_given_size, in BlockType order.
BLOCKS = ('debug_tile grass water stone tree wood path coal iron diamond table '
          'furnace sand lava plant_on_grass ripe_plant_on_grass wall2 '
          'debug_tile wall_moss stalagmite sapphire ruby chest fountain '
          'fire_grass ice_grass gravel fire_tree ice_shrub '
          'enchantment_table_fire enchantment_table_ice necromancer grave '
          'grave2 grave3 necromancer_vulnerable').split()
BLOCKS = ['debug_tile'] + BLOCKS        # 0 INVALID, 1 OUT_OF_BOUNDS, ...
# OUT_OF_BOUNDS and DARKNESS, painted flat as the game's own renderer does.
FLAT = {1: (128, 128, 128), 18: (0, 0, 0)}
ITEMS = (None, 'torch_in_inventory', 'ladder_down', 'ladder_up',
         'ladder_down_blocked')
MOBS = (('zombie', 'gnome_warrior', 'orc_soldier', 'lizard', 'knight', 'troll',
         'pigman', 'frost_troll'),
        ('cow', 'bat', 'snail'),
        ('skeleton', 'gnome_archer', 'orc_mage', 'kobold', 'knight_archer',
         'deep_thing', 'fire_elemental', 'ice_elemental'),
        ('arrow-up', 'dagger', 'fireball', 'iceball', 'arrow-up', 'slimeball',
         'fireball', 'iceball'),
        ('arrow-up', 'dagger', 'fireball', 'iceball', 'arrow-up', 'slimeball',
         'fireball', 'iceball'))
PLAYER = {0: 'player-left', 1: 'player-right', 2: 'player-up', 3: 'player-down'}
RUNS = (('expB_mask', 'B2'), ('expB_input', 'B1'), ('honest_map', 'honest'),
        ('vanilla', 'vanilla'))

_tex = {}


def tex(name):
  if name not in _tex:
    im = Image.open(ASSETS / f'{name}.png').convert('RGBA').resize((PX, PX))
    _tex[name] = im
  return _tex[name]


def render(vec):
  """The agent's view as an RGB image, as the agent saw it (dark = black)."""
  vec = np.asarray(vec, np.float32)
  tiles = vec[:M.OBS_H * M.OBS_W * M.N_TILE].reshape(M.OBS_H, M.OBS_W, M.N_TILE)
  lit = tiles[..., -1] > 0.5
  blocks = tiles[..., :M.N_BLOCK].argmax(-1)
  items = tiles[..., M.N_BLOCK:M.N_BLOCK + M.N_ITEM].argmax(-1)
  mobs = tiles[..., M.N_BLOCK + M.N_ITEM:M.N_BLOCK + M.N_ITEM + M.N_MOB]
  mobs = mobs.reshape(M.OBS_H, M.OBS_W, 5, 8) > 0.5
  st = V.decode_stats(vec)
  img = Image.new('RGBA', (M.OBS_W * PX, M.OBS_H * PX), (0, 0, 0, 255))
  for y in range(M.OBS_H):
    for x in range(M.OBS_W):
      if not lit[y, x]:
        continue
      at = (x * PX, y * PX)
      b = int(blocks[y, x])
      img.alpha_composite(Image.new('RGBA', (PX, PX), FLAT[b] + (255,))
                          if b in FLAT else tex(BLOCKS[b]), at)
      if ITEMS[items[y, x]]:
        img.alpha_composite(tex(ITEMS[items[y, x]]), at)
      for c, k in zip(*np.nonzero(mobs[y, x])):
        if k < len(MOBS[c]):
          img.alpha_composite(tex(MOBS[c][k]), at)
  me = 'player-sleep' if st['sleeping'] else PLAYER[st['direction']]
  img.alpha_composite(tex(me), (M.OBS_W // 2 * PX, M.OBS_H // 2 * PX))
  return np.asarray(img.convert('RGB')), st


def strip(fig, ex, vectors, top, height):
  """Two rows of five frames with step, action and the meters that matter."""
  fig.text(0.07, top, f"Flaw {ex['flaw']}: {ex['title']}", fontsize=11,
           weight='bold')
  fig.text(0.07, top - 0.018, ex['note'], fontsize=8.5, color=R.INK2)
  w, gap = 0.165, 0.008
  h = w * (M.OBS_H / M.OBS_W) * (R.A4[0] / R.A4[1])
  for i, (t, a, v) in enumerate(zip(ex['steps'], ex['actions'], vectors)):
    row, col = divmod(i, 5)
    x = 0.07 + col * (w + gap)
    y = top - 0.03 - h - row * (h + 0.065)
    ax = fig.add_axes([x, y, w, h])
    img, st = render(v)
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
      s.set_visible(True)
      s.set_color(R.GRID)
    special = a not in ('NOOP', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'DO')
    fig.text(x, y - 0.009, f'{t}: {a.replace("_", " ").lower()}',
             fontsize=6.6, color=R.BLUE_DARK if special else R.INK,
             weight='bold' if special else 'normal')
    fig.text(x, y - 0.021, f"wood {st['wood']}  stone {st['stone']}",
             fontsize=6.2, color=R.INK2)
    fig.text(x, y - 0.032, f"hp {st['health']:.0f}  food {st['food']:.0f}  "
             f"drink {st['drink']:.0f}", fontsize=6.2, color=R.INK2)


def table(fig, top, rows, header):
  cell = fig.add_axes([0.07, top - 0.026 * (len(rows) + 1), 0.86,
                       0.026 * (len(rows) + 1)])
  cell.axis('off')
  t = cell.table(cellText=rows, colLabels=header, loc='upper left',
                 cellLoc='right', colLoc='right',
                 colWidths=[0.52] + [0.12] * (len(header) - 1))
  t.auto_set_font_size(False)
  t.set_fontsize(8)
  for (r, c), cl in t.get_celld().items():
    cl.set_edgecolor(R.GRID)
    cl.set_linewidth(0.6)
    cl.set_height(1 / (len(rows) + 1))
    if c == 0:
      cl.set_text_props(ha='left')
    if r == 0:
      cl.set_text_props(weight='bold', color=R.INK2)
    if c == 1 and r > 0:
      cl.set_text_props(weight='bold')
  return top - 0.026 * (len(rows) + 1)


def measured(c):
  sp, wp, sv, ws = (c['stone_pickaxe'], c['wood_pickaxe'], c['survival'],
                    c['wasted'])
  held = wp['tables']['wood_held_when_placing']
  n = c['episodes']
  pct = lambda v: f'{100 * v:.0f}%'
  return [
      f"{wp['tables']['per_episode']:.2f}",
      pct(held.get('2', 0) / max(sum(held.values()), 1)),
      pct(wp['zero_wood_after_table']) if wp['episodes'] else '-',
      f"{100 * sp['chances'] / n:.0f}",
      pct(sp['pressed_when_possible']),
      pct(sp['chance_ended'].get('ingredient spent on MAKE_WOOD_SWORD', 0)),
      pct(sp['wood_pickaxe_pressed_when_possible']),
      pct(sv['episodes_never_drinking']),
      f"{sv['never_drinking_dry_at_step']:.0f}",
      f"{sv['never_drinking_lifespan']:.0f} / {sv['drinking_lifespan']:.0f}",
      f"{sv['meals_per_episode']:.2f}",
      pct(sv['mob_adjacent_last10'].get('attacks it (DO facing the mob)', 0)),
      pct(ws['do_share']),
      pct(ws['do_target'].get('grass', 0)),
  ]


LABELS = [
    'tables placed per episode',
    'tables placed with exactly 2 wood (all of it)',
    'table-but-no-wood-pickaxe episodes left with 0 wood',
    'stone-pickaxe chances per 100 episodes',
    'pressed MAKE_STONE_PICKAXE when it was possible',
    'chances lost to MAKE_WOOD_SWORD',
    'pressed MAKE_WOOD_PICKAXE when it was possible',
    'episodes that never drink',
    '  ... drink reaches 0 at step (median)',
    '  lifespan: never drink / drink (median)',
    'meals per episode',
    'hostile adjacent in last 10 steps: attacks it',
    'share of all steps that are DO',
    '  ... aimed at grass',
]


def main():
  runs = [(name, label, json.loads((CASES / f'{name}.json').read_text()))
          for name, label in RUNS if (CASES / f'{name}.json').exists()]
  examples = json.loads((CASES / 'examples.json').read_text())
  frames = np.load(CASES / 'examples.npz')
  cols = [measured(c) for _, _, c in runs]
  rows = [[lab] + [col[i] for col in cols] for i, lab in enumerate(LABELS)]
  b2 = runs[0][2]
  sp, wp, sv = b2['stone_pickaxe'], b2['wood_pickaxe'], b2['survival']
  pct = lambda v: f'{100 * v:.0f}%'

  with R.PdfPages(OUT) as pdf:
    fig = R.page(pdf, "B2's flaws, and where they show in its episodes",
                 f"Last 150k training steps of each run ({b2['episodes']} "
                 'B2 episodes), decoded step by step from replay by '
                 'tools/episode_cases.py. Everything is read from the '
                 "agent's own observation and its own actions.")
    y = table(fig, 0.89, rows, ['measured'] + [lab for _, lab, _ in runs])
    text = (
        '1. It spends its wood as fast as it gets it. B2 places a table '
        f"{wp['tables']['per_episode']:.1f} times per episode, almost always "
        'the instant it holds two logs -- which is all of them. '
        f"{pct(1 - wp['tables']['repeats_with_a_table_in_view'])} of the "
        'repeat tables are built with no table in view, so it builds a new '
        'one rather than walk back. Every episode that placed a table but '
        'never made the wood pickaxe was left with 0 wood afterwards, and '
        'never again stood at a table holding wood. At a table with wood it '
        'makes the pickaxe every time (the other runs: 9-15% of such steps) -- '
        'masking taught the key; the order of '
        'operations is not. B2 is the most extreme of the runs here; masking '
        'makes PLACE_TABLE pressable exactly when the second log arrives.\n'
        '2. The stone-pickaxe chance lasts one step. B2 creates far more '
        'chances than any other run, but the median chance is '
        f"{sp['chance_length_median']:.0f} step long: in "
        f"{pct(sp['chance_ended'].get('ingredient spent on MAKE_WOOD_SWORD', 0))}"
        ' the single log goes into the wood sword (also valid, also rewarded '
        f"once), in {pct(sp['chance_ended'].get('walked away', 0))} it walks "
        f"off, and {pct(sp['chance_ended'].get('made it', 0))} succeed. Stone "
        f"goes to furnaces ({pct(sp['sinks']['stone'].get('PLACE_FURNACE', 0))})"
        ' and placed stone '
        f"({pct(sp['sinks']['stone'].get('PLACE_STONE', 0))}), not tools.\n"
        f"3. The meters run it out. {pct(sv['episodes_never_drinking'])} of "
        'episodes never drink once; their drink hits zero around step '
        f"{sv['never_drinking_dry_at_step']:.0f} and they die around "
        f"{sv['never_drinking_lifespan']:.0f}. It eats "
        f"{sv['meals_per_episode']:.2f} times per episode. When drink hits "
        f"zero, water is in view only {pct(sv['water_in_view_when_drink_hit_zero'])}"
        ' of the time -- it does not head for water it has seen before. With a '
        'hostile mob adjacent at the end it attacks it '
        f"{pct(sv['mob_adjacent_last10'].get('attacks it (DO facing the mob)', 0))}"
        ' of the time and NOOPs '
        f"{pct(sv['mob_adjacent_last10'].get('NOOP', 0))}.\n"
        f"4. A third of its steps are DO ({pct(b2['wasted']['do_share'])}), "
        f"over half of them at grass ({pct(b2['wasted']['do_target']['grass'])})"
        ' -- a 10% sapling lottery; another '
        f"{pct(b2['wasted']['do_target'].get('plant', 0))} hit plants it "
        'placed. Long walks into a solid tile are rare '
        f"({b2['wasted']['blocked_streaks_per_1000_steps']:.1f} per 1000 "
        'steps).')
    R.wrap(fig, 0.07, y - 0.02, text, width=108, size=8.4, line=0.0142)
    pdf.savefig(fig)
    plt.close(fig)

    for i in range(0, len(examples), 2):
      fig = R.page(pdf, 'In the episodes', 'Frames are what the agent saw '
                   '(its 9x11 view, dark tiles black), with the action it '
                   'took next. Special actions in blue. From B2 replay.')
      for j, ex in enumerate(examples[i:i + 2]):
        strip(fig, ex, frames[f'{i + j}_vectors'], 0.88 - j * 0.44, 0.40)
      pdf.savefig(fig)
      plt.close(fig)
  print('wrote', OUT)


if __name__ == '__main__':
  main()
