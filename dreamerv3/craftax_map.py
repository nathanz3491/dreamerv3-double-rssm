"""Coarse-map targets for the two-RSSM map model.

Turns a Craftax ``EnvState`` into the supervision RSSM-2 learns from: a 12x12x16
downsample of the current level, the agent's coarse cell, and a cumulative
visitation mask.

Two ways to build the targets, and the difference is the whole ballgame:

  ``coarse_map`` / ``coarse_pos``  read the TRUE state: the whole 48x48 map and
                           the true coordinates. Privileged. Kept for
                           evaluation and for reproducing the original runs.
  ``ObservedTargets``      reads nothing but the agent's own observation vector
                           and its own actions. Terrain is a mosaic of the lit
                           9x11 windows it has seen; position is dead-reckoned
                           from the fixed spawn, judging each move by the tile
                           the agent could see in front of it.

The second is the default. Supervising unseen cells against ground truth taught
RSSM-2 Craftax's world generator rather than teaching it to remember and infer,
and left no held-out set at all -- every cell it was ever scored on, it had also
studied. Reading true coordinates was a second, quieter leak: on the surface the
agent could have worked them out (the spawn is fixed), but after a ladder the
game teleports it somewhere it cannot know. Under ``ObservedTargets`` unobserved
cells carry no gradient and become a real exam (``tools/map_eval.py``), and each
level gets its own frame anchored where the agent arrived.

These are TRAINING TARGETS ONLY. Nothing here is ever fed to the encoder -- the
agent's observation stays the stock 8268-dim vector. Same posture as
``craftax_features.py``: pure numpy, no JAX/agent dependency, unit-testable
without a GPU.

Design: ``design-map-model.md`` SS3.3.
"""

import numpy as np

# --- geometry ---------------------------------------------------------------
MAP_SIZE = 48          # craftax_state.StaticEnvParams.map_size
COARSE = 12            # 12x12 grid => each cell covers CELL x CELL tiles
CELL = MAP_SIZE // COARSE
N_CELLS = COARSE * COARSE
OBS_H, OBS_W = 9, 11   # constants.OBS_DIM -- the agent's visible window

# --- plane layout (16 planes; keep in sync with mapmodel.planes) -------------
# Pooling is fixed PER PLANE here, at target-construction time. RSSM-2 never
# sees "max" or "mean" -- it only learns to predict whatever number sits in each
# plane. The choice is about dynamic range, measured over 8 fresh worlds:
#
#   stone fill spans 0.05..0.95 across cells (scattered pebbles vs solid
#   mountain) -> MEAN carries that, MAX would print 1.0 for both and lose it.
#   tree  fill is <0.25 in 91% of tree cells, never >0.75 -> MEAN would read
#   ~0.1 everywhere, indistinguishable from empty under gradient noise.
P_WATER, P_STONE, P_TREE, P_SAND, P_LAVA = 0, 1, 2, 3, 4
P_COAL, P_IRON, P_DIAMOND, P_GEM = 5, 6, 7, 8
P_TABLE, P_FURNACE, P_PLANT, P_PATH = 9, 10, 11, 12
P_MOB_PASSIVE, P_MOB_HOSTILE = 13, 14
P_SEEN = 15                       # recency: 1 = visible now, 0 = never seen
N_PLANES = 16

PLANE_NAMES = (
    'water', 'stone', 'tree', 'sand', 'lava',
    'coal', 'iron', 'diamond', 'gem',
    'table', 'furnace', 'plant', 'path',
    'mob_passive', 'mob_hostile', 'seen',
)

# BlockType ids (craftax.craftax.constants.BlockType)
_WATER = (3,)
_STONE = (4, 17, 19, 20, 27)                 # stone, wall, wall_moss, stalagmite, gravel
_TREE = (5, 28, 29)                          # tree, fire_tree, ice_shrub
_SAND = (13,)
_LAVA = (14,)
_COAL, _IRON, _DIAMOND = (8,), (9,), (10,)
_GEM = (21, 22)                              # sapphire, ruby
_TABLE, _FURNACE = (11,), (12,)
_PLANT = (15, 16)                            # plant, ripe_plant
_PATH = (7,)

# Dense enough for the fraction to be readable -> pool by mean.
_MEAN_GROUPS = (
    (P_WATER, _WATER), (P_STONE, _STONE), (P_SAND, _SAND), (P_PLANT, _PLANT),
)
# Sparse or all-or-nothing -> pool by presence.
_MAX_GROUPS = (
    (P_TREE, _TREE), (P_LAVA, _LAVA), (P_COAL, _COAL), (P_IRON, _IRON),
    (P_DIAMOND, _DIAMOND), (P_GEM, _GEM), (P_TABLE, _TABLE),
    (P_FURNACE, _FURNACE), (P_PATH, _PATH),
)


def _blocks_of_level(state, level=None):
  """(48, 48) int array of block ids for the level the agent is on."""
  blocks = np.asarray(state.map)
  if blocks.ndim == 3:                       # (num_levels, H, W)
    level = int(state.player_level) if level is None else level
    blocks = blocks[level]
  assert blocks.shape == (MAP_SIZE, MAP_SIZE), blocks.shape
  return blocks


def _pool_presence(mask):
  """(48,48) bool -> (12,12) float: does this 4x4 cell contain any of it?"""
  return mask.reshape(COARSE, CELL, COARSE, CELL).any(axis=(1, 3)).astype(np.float32)


def _pool_fraction(mask):
  """(48,48) bool -> (12,12) float: what fraction of this 4x4 cell is it?"""
  return mask.reshape(COARSE, CELL, COARSE, CELL).mean(axis=(1, 3)).astype(np.float32)


def _mob_counts(state):
  """(48,48) passive and hostile mob counts on the current level.

  Mob arrays are (num_levels, max_mobs, 2) positions plus a mask. Craftax names
  them by kind (melee/passive/ranged/...); we only need the passive/hostile
  split, so anything that is not passive counts as hostile.
  """
  level = int(state.player_level)
  passive = np.zeros((MAP_SIZE, MAP_SIZE), np.float32)
  hostile = np.zeros((MAP_SIZE, MAP_SIZE), np.float32)

  for name in ('passive_mobs', 'melee_mobs', 'ranged_mobs'):
    mobs = getattr(state, name, None)
    if mobs is None:
      continue
    pos = np.asarray(mobs.position)
    mask = np.asarray(mobs.mask)
    if pos.ndim == 3:                        # (levels, n, 2)
      pos, mask = pos[level], mask[level]
    target = passive if name == 'passive_mobs' else hostile
    for (y, x), alive in zip(pos.reshape(-1, 2), mask.reshape(-1)):
      if alive:
        yi, xi = int(y), int(x)
        if 0 <= yi < MAP_SIZE and 0 <= xi < MAP_SIZE:
          target[yi, xi] += 1.0
  return passive, hostile


def coarse_map(state, seen=None):
  """(12, 12, 16) float32 supervision target for RSSM-2's map decoder.

  ``seen`` is the cumulative visitation mask; pass the value returned by
  ``update_seen`` so the meta plane reflects history rather than this step.
  """
  blocks = _blocks_of_level(state)
  out = np.zeros((COARSE, COARSE, N_PLANES), np.float32)

  for plane, ids in _MEAN_GROUPS:
    out[:, :, plane] = _pool_fraction(np.isin(blocks, ids))
  for plane, ids in _MAX_GROUPS:
    out[:, :, plane] = _pool_presence(np.isin(blocks, ids))

  passive, hostile = _mob_counts(state)
  out[:, :, P_MOB_PASSIVE] = _pool_presence(passive > 0)
  out[:, :, P_MOB_HOSTILE] = _pool_presence(hostile > 0)

  if seen is not None:
    out[:, :, P_SEEN] = np.asarray(seen, np.float32)
  return out


def coarse_pos(state):
  """Agent's coarse cell as a flat index in [0, 144).

  This is a *prediction target*, never an input: the 8268-dim observation
  contains no absolute position, so RSSM-2 must dead-reckon it from its own
  movement. See design SS3.3(b).
  """
  y, x = np.asarray(state.player_position).reshape(2)
  cy = int(np.clip(int(y) // CELL, 0, COARSE - 1))
  cx = int(np.clip(int(x) // CELL, 0, COARSE - 1))
  return np.int32(cy * COARSE + cx)


def update_seen(seen, state, decay=1.0):
  """Recency of each coarse cell: 1.0 = visible now, 0.0 = never seen.

  ``decay`` is the per-step multiplier applied before the current window is
  stamped in. At 1.0 this is the old binary mask. Below 1.0 the value ages, so
  the actor can tell a cow seen five steps ago (still there) from one seen three
  hundred steps ago (long gone) -- one plane giving the right answer for both
  static terrain and moving mobs. tau steps of half-life is decay = 0.5**(1/tau).
  """
  seen = np.zeros((COARSE, COARSE), np.float32) if seen is None else (
      np.asarray(seen, np.float32) * np.float32(decay))
  y, x = np.asarray(state.player_position).reshape(2)
  y0, y1 = int(y) - OBS_H // 2, int(y) + OBS_H // 2
  x0, x1 = int(x) - OBS_W // 2, int(x) + OBS_W // 2
  cy0 = max(0, y0 // CELL)
  cy1 = min(COARSE - 1, y1 // CELL)
  cx0 = max(0, x0 // CELL)
  cx1 = min(COARSE - 1, x1 // CELL)
  if cy0 <= cy1 and cx0 <= cx1:
    seen[cy0:cy1 + 1, cx0:cx1 + 1] = 1.0
  return seen


UNKNOWN = -1           # a tile the agent has not observed this episode


def known_fraction(known):
  """(12, 12) float: what fraction of each cell's 16 tiles has been observed.

  This is the supervision WEIGHT. A cell at 0.0 contributes no gradient, so the
  model is never taught the contents of terrain it has not seen.
  """
  return (np.asarray(known) != UNKNOWN).reshape(
      COARSE, CELL, COARSE, CELL).mean((1, 3)).astype(np.float32)


# --- targets from the observation stream alone --------------------------------
# Everything below builds RSSM-2's honest targets from two inputs only: the
# agent's own 8268-dim observation vector and the action it took. It never sees
# an EnvState, so it cannot read terrain the agent has not seen or coordinates it
# has not worked out -- "an outside observer watching only the agent's screen
# could reconstruct every label" is enforced by the function signatures, not
# argued. test_craftax_map checks the reconstruction against the true state.
#
# Layout of Craftax-Symbolic's vector (renderer.render_craftax_symbolic):
# 9x11 tiles x 83 channels -- [37 block one-hot | 5 item one-hot | 40 mob | 1
# light], all but the last multiplied by the light mask -- then 51 scalars that
# end with 8 "special values" (light level, sleeping, resting, spells x2,
# level/10, level cleared, boss vulnerable).
N_BLOCK, N_ITEM, N_MOB = 37, 5, 40
N_TILE = N_BLOCK + N_ITEM + N_MOB + 1
OBS_LEN = OBS_H * OBS_W * N_TILE + 51
SPAWN = (MAP_SIZE // 2, MAP_SIZE // 2)   # world_gen.generate_world, every episode

# Blocks a land creature cannot enter: constants.SOLID_BLOCKS plus water and
# lava (COLLISION_LAND_CREATURE). Checked against the installed Craftax by
# test_blocked_set_matches_craftax.
SOLID_IDS = (4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 19, 20, 21, 22, 23, 24, 28,
             30, 31, 32, 33, 34, 35)
_BLOCKED = frozenset(SOLID_IDS) | {3, 14}
_MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}   # LEFT RIGHT UP DOWN


def decode_view(vec):
  """Observation vector -> what the agent can see this step.

  Returns ``blocks`` (9, 11) block ids with UNKNOWN where the tile is dark,
  ``passive`` / ``hostile`` (9, 11) visible-mob masks, and ``level``,
  ``sleeping``, ``resting`` from the special values. Out-of-bounds tiles are
  padded dark by the renderer, so they decode as UNKNOWN too.
  """
  vec = np.asarray(vec, np.float32).reshape(-1)
  assert vec.size == OBS_LEN, vec.size
  tiles = vec[:OBS_H * OBS_W * N_TILE].reshape(OBS_H, OBS_W, N_TILE)
  lit = tiles[..., -1] > 0.5
  blocks = np.where(lit, tiles[..., :N_BLOCK].argmax(-1), UNKNOWN)
  mobs = tiles[..., N_BLOCK + N_ITEM:N_BLOCK + N_ITEM + N_MOB].reshape(
      OBS_H, OBS_W, 5, 8) > 0.5
  special = vec[-8:]
  return dict(
      blocks=blocks.astype(np.int32),
      passive=mobs[:, :, 1].any(-1) & lit,                        # class 1
      hostile=(mobs[:, :, 0].any(-1) | mobs[:, :, 2].any(-1)) & lit,
      sleeping=bool(special[1] > 0.5),
      resting=bool(special[2] > 0.5),
      level=int(round(float(special[5]) * 10)))


def reckon(pos, view, action):
  """Dead-reckon one step, as an observer of the screen would.

  ``view`` is what the agent saw BEFORE acting. Craftax moves the player before
  any mob updates (game_logic.craftax_step), so whether a move succeeds is
  decided by the destination tile and whether a mob stands on it -- both
  adjacent to the centre of the view. A sleeping or resting agent's action is
  replaced by NOOP. The map is 48x48, a game rule the observer knows. A dark
  destination cannot be judged and is assumed passable; how often the
  reconstruction disagrees with the true position is measured in the tests.
  """
  a = int(action)
  if a not in _MOVES or view['sleeping'] or view['resting']:
    return pos
  dy, dx = _MOVES[a]
  y, x = pos[0] + dy, pos[1] + dx
  if not (0 <= y < MAP_SIZE and 0 <= x < MAP_SIZE):
    return pos
  cy, cx = OBS_H // 2 + dy, OBS_W // 2 + dx
  if int(view['blocks'][cy, cx]) in _BLOCKED:
    return pos
  if view['passive'][cy, cx] or view['hostile'][cy, cx]:
    return pos
  return (y, x)


def _stamp(canvas, window, pos, keep=None):
  """Write a (9, 11) window into a (48, 48) canvas centred at ``pos``, clipped.

  Where ``keep`` is False the canvas keeps its old value, so a dark tile never
  erases terrain seen earlier.
  """
  y0, x0 = int(pos[0]) - OBS_H // 2, int(pos[1]) - OBS_W // 2
  dy0, dx0 = max(0, y0), max(0, x0)
  dy1, dx1 = min(MAP_SIZE, y0 + OBS_H), min(MAP_SIZE, x0 + OBS_W)
  if dy0 >= dy1 or dx0 >= dx1:
    return canvas
  win = window[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
  if keep is not None:
    k = keep[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
    win = np.where(k, win, canvas[dy0:dy1, dx0:dx1])
  canvas[dy0:dy1, dx0:dx1] = win
  return canvas


def seen_at(seen, pos, decay=1.0):
  """``update_seen`` for a reckoned position instead of an EnvState."""
  seen = np.zeros((COARSE, COARSE), np.float32) if seen is None else (
      np.asarray(seen, np.float32) * np.float32(decay))
  y, x = int(pos[0]), int(pos[1])
  cy0 = max(0, (y - OBS_H // 2) // CELL)
  cy1 = min(COARSE - 1, (y + OBS_H // 2) // CELL)
  cx0 = max(0, (x - OBS_W // 2) // CELL)
  cx1 = min(COARSE - 1, (x + OBS_W // 2) // CELL)
  if cy0 <= cy1 and cx0 <= cx1:
    seen[cy0:cy1 + 1, cx0:cx1 + 1] = 1.0
  return seen


def cell_of(pos):
  """Coarse cell index in [0, 144) of a (y, x) tile position."""
  cy = int(np.clip(int(pos[0]) // CELL, 0, COARSE - 1))
  cx = int(np.clip(int(pos[1]) // CELL, 0, COARSE - 1))
  return np.int32(cy * COARSE + cx)


def _coarse_from(known, passive, hostile, seen):
  """(12, 12, 16) target from a mosaic and visible-mob maps -- no EnvState."""
  isknown = known != UNKNOWN
  out = np.zeros((COARSE, COARSE, N_PLANES), np.float32)
  seen_tiles = isknown.reshape(COARSE, CELL, COARSE, CELL).sum((1, 3))
  for plane, ids in _MEAN_GROUPS:
    hits = (np.isin(known, ids) & isknown).reshape(
        COARSE, CELL, COARSE, CELL).sum((1, 3))
    out[:, :, plane] = hits / np.maximum(seen_tiles, 1)
  for plane, ids in _MAX_GROUPS:
    out[:, :, plane] = _pool_presence(np.isin(known, ids) & isknown)
  out[:, :, P_MOB_PASSIVE] = _pool_presence(passive)
  out[:, :, P_MOB_HOSTILE] = _pool_presence(hostile)
  out[:, :, P_SEEN] = np.asarray(seen, np.float32)
  return out


class ObservedTargets:
  """RSSM-2 targets built from (observation vector, action) pairs only.

  Carries the observer's belief across steps: the reckoned position, the
  terrain mosaic, the visitation record, and the previous view (needed to judge
  whether the last move succeeded). A new episode or a change of level --
  visible in the observation -- starts a fresh frame anchored at SPAWN: after a
  ladder the agent genuinely does not know where on the new level it is, so the
  frame is its own rather than the game's.
  """

  def __init__(self, seen_decay=0.99):
    self.seen_decay = float(seen_decay)
    self.pos, self.known, self.seen = SPAWN, None, None
    self.view, self.level = None, None

  def step(self, vec, action, is_first):
    view = decode_view(vec)
    if is_first or self.view is None or view['level'] != self.level:
      self.pos, self.known, self.seen = SPAWN, None, None
    else:
      self.pos = reckon(self.pos, self.view, action)
    self.view, self.level = view, view['level']

    if self.known is None:
      self.known = np.full((MAP_SIZE, MAP_SIZE), UNKNOWN, np.int32)
    _stamp(self.known, view['blocks'], self.pos, keep=view['blocks'] != UNKNOWN)
    self.seen = seen_at(self.seen, self.pos, self.seen_decay)
    passive = _stamp(np.zeros((MAP_SIZE, MAP_SIZE), bool), view['passive'],
                     self.pos)
    hostile = _stamp(np.zeros((MAP_SIZE, MAP_SIZE), bool), view['hostile'],
                     self.pos)
    return dict(
        map12=_coarse_from(self.known, passive, hostile, self.seen),
        mappos=cell_of(self.pos),
        mapseen=self.seen.astype(np.float32),
        mapknown=known_fraction(self.known))


def crop_egocentric(map12, cell, size=9):
  """(size, size, 16) window of ``map12`` centred on ``cell``, zero-padded.

  Egocentric on purpose: with the agent at the centre, *position is the
  meaning* -- "stone up-left" is readable straight off the layout, with no
  coordinate transform for the actor to learn. This is also why no CNN is
  used downstream (design SS4).
  """
  map12 = np.asarray(map12)
  cy, cx = int(cell) // COARSE, int(cell) % COARSE
  r = size // 2
  out = np.zeros((size, size, map12.shape[-1]), np.float32)
  y0, y1 = max(0, cy - r), min(COARSE, cy + r + 1)
  x0, x1 = max(0, cx - r), min(COARSE, cx + r + 1)
  out[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)] = map12[y0:y1, x0:x1]
  return out
