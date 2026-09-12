"""Coarse-map targets for the two-RSSM map model.

Turns a privileged Craftax ``EnvState`` into the supervision RSSM-2 learns from:
a 12x12x13 downsample of the current level, the agent's coarse cell, and a
cumulative visitation mask.

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

# --- plane layout (13 planes; keep in sync with mapmodel.planes) -------------
# Terrain: prior-completable. Biomes cluster, so unseen cells are predictable.
P_WATER, P_STONE, P_TREE, P_SAND, P_LAVA, P_ORE = 0, 1, 2, 3, 4, 5
# Placed: memory-only. No prior can say where *you* put a table.
P_TABLE, P_FURNACE, P_PLANT, P_PATH = 6, 7, 8, 9
# Dynamic: density, not location -- a mob's position is stale within steps.
P_MOB_PASSIVE, P_MOB_HOSTILE = 10, 11
# Meta: the plane that lets the actor tell recall from guess.
P_SEEN = 12
N_PLANES = 13

PLANE_NAMES = (
    'water', 'stone', 'tree', 'sand', 'lava', 'ore',
    'table', 'furnace', 'plant', 'path',
    'mob_passive', 'mob_hostile', 'seen',
)

# BlockType ids (craftax.craftax.constants.BlockType)
_WATER = (3,)
_STONE = (4, 17, 19, 20, 27)                 # stone, wall, wall_moss, stalagmite, gravel
_TREE = (5, 28, 29)                          # tree, fire_tree, ice_shrub
_SAND = (13,)
_LAVA = (14,)
_ORE = (8, 9, 10, 21, 22)                    # coal, iron, diamond, sapphire, ruby
_TABLE = (11,)
_FURNACE = (12,)
_PLANT = (15, 16)                            # plant, ripe_plant
_PATH = (7,)

_TERRAIN_GROUPS = (
    (P_WATER, _WATER), (P_STONE, _STONE), (P_TREE, _TREE), (P_SAND, _SAND),
    (P_LAVA, _LAVA), (P_ORE, _ORE), (P_TABLE, _TABLE), (P_FURNACE, _FURNACE),
    (P_PLANT, _PLANT), (P_PATH, _PATH),
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


def _pool_density(counts):
  """(48,48) numeric -> (12,12) float in [0,1]: mean occupancy of the cell."""
  pooled = counts.reshape(COARSE, CELL, COARSE, CELL).sum(axis=(1, 3))
  return np.clip(pooled / (CELL * CELL), 0.0, 1.0).astype(np.float32)


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
  """(12, 12, 13) float32 supervision target for RSSM-2's map decoder.

  ``seen`` is the cumulative visitation mask; pass the value returned by
  ``update_seen`` so the meta plane reflects history rather than this step.
  """
  blocks = _blocks_of_level(state)
  out = np.zeros((COARSE, COARSE, N_PLANES), np.float32)

  for plane, ids in _TERRAIN_GROUPS:
    out[:, :, plane] = _pool_presence(np.isin(blocks, ids))

  passive, hostile = _mob_counts(state)
  out[:, :, P_MOB_PASSIVE] = _pool_density(passive)
  out[:, :, P_MOB_HOSTILE] = _pool_density(hostile)

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


def update_seen(seen, state):
  """Mark every coarse cell the agent's 9x11 window currently overlaps."""
  seen = np.zeros((COARSE, COARSE), bool) if seen is None else np.array(seen, bool)
  y, x = np.asarray(state.player_position).reshape(2)
  y0, y1 = int(y) - OBS_H // 2, int(y) + OBS_H // 2
  x0, x1 = int(x) - OBS_W // 2, int(x) + OBS_W // 2
  cy0 = max(0, y0 // CELL)
  cy1 = min(COARSE - 1, y1 // CELL)
  cx0 = max(0, x0 // CELL)
  cx1 = min(COARSE - 1, x1 // CELL)
  if cy0 <= cy1 and cx0 <= cx1:
    seen[cy0:cy1 + 1, cx0:cx1 + 1] = True
  return seen


def crop_egocentric(map12, cell, size=9):
  """(size, size, 13) window of ``map12`` centred on ``cell``, zero-padded.

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
