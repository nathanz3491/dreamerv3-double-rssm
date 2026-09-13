"""Unit tests for the coarse-map targets. Pure numpy -- no JAX, no GPU.

Run: python -m pytest dreamerv3/test_craftax_map.py -q
"""

import types

import numpy as np

from dreamerv3 import craftax_map as M


def _state(blocks=None, pos=(24, 24), level=0):
  """Minimal duck-typed EnvState carrying only what the targets read."""
  blocks = np.full((M.MAP_SIZE, M.MAP_SIZE), 2, np.int32) if blocks is None else blocks
  empty = types.SimpleNamespace(
      position=np.zeros((0, 2), np.int32), mask=np.zeros((0,), bool))
  return types.SimpleNamespace(
      map=blocks, player_position=np.array(pos, np.int32), player_level=level,
      passive_mobs=empty, melee_mobs=empty, ranged_mobs=empty)


def test_shapes_and_dtypes():
  m = M.coarse_map(_state())
  assert m.shape == (M.COARSE, M.COARSE, M.N_PLANES), m.shape
  assert m.dtype == np.float32
  assert len(M.PLANE_NAMES) == M.N_PLANES


def test_single_water_tile_lights_exactly_one_cell():
  blocks = np.full((M.MAP_SIZE, M.MAP_SIZE), 2, np.int32)
  blocks[10, 10] = 3                                  # BlockType.WATER
  m = M.coarse_map(_state(blocks))
  water = m[:, :, M.P_WATER]
  assert water[10 // M.CELL, 10 // M.CELL] > 0.0      # -> cell (2, 2)
  assert (water > 0).sum() == 1, 'must not leak into neighbours'


def test_mean_planes_carry_fill_not_just_presence():
  """stone/water/sand/plant are pooled by fraction: 1 tile != a full cell.

  This is the dynamic range the design buys by not using max everywhere -- a
  scattered pebble field and a solid mountain must not read identically.
  """
  one = np.full((M.MAP_SIZE, M.MAP_SIZE), 2, np.int32)
  one[0, 0] = 4                                       # STONE, 1 of 16 tiles
  full = np.full((M.MAP_SIZE, M.MAP_SIZE), 2, np.int32)
  full[0:M.CELL, 0:M.CELL] = 4                        # STONE, 16 of 16

  sparse = M.coarse_map(_state(one))[0, 0, M.P_STONE]
  dense = M.coarse_map(_state(full))[0, 0, M.P_STONE]
  assert sparse == 1.0 / (M.CELL * M.CELL), sparse
  assert dense == 1.0
  assert sparse < dense, 'mean pooling must distinguish these'


def test_max_planes_saturate_on_a_single_tile():
  """tree/ore/lava/table are sparse, so one tile lights the whole cell.

  Pooled by mean a tree cell would read ~0.1 -- indistinguishable from empty
  once the decoder is noisy. Presence keeps the signal at full scale.
  """
  blocks = np.full((M.MAP_SIZE, M.MAP_SIZE), 2, np.int32)
  blocks[0, 0] = 5                                    # TREE
  blocks[4, 0] = 8                                    # COAL
  m = M.coarse_map(_state(blocks))
  assert m[0, 0, M.P_TREE] == 1.0
  assert m[1, 0, M.P_COAL] == 1.0


def test_ores_do_not_share_a_plane():
  """Coal (8% of cells), iron (6%) and diamond (0.3%) are found and used at
  different tiers -- collapsing them loses exactly the distinction the actor
  needs when deciding where to dig."""
  blocks = np.full((M.MAP_SIZE, M.MAP_SIZE), 2, np.int32)
  blocks[0, 0] = 9                                    # IRON
  m = M.coarse_map(_state(blocks))
  assert m[0, 0, M.P_IRON] == 1.0
  assert m[0, 0, M.P_COAL] == 0.0
  assert m[0, 0, M.P_DIAMOND] == 0.0


def test_coarse_pos_matches_tile_position():
  for y, x in [(0, 0), (25, 23), (47, 47), (4, 8)]:
    cell = int(M.coarse_pos(_state(pos=(y, x))))
    assert (cell // M.COARSE, cell % M.COARSE) == (y // M.CELL, x // M.CELL)


def test_seen_accumulates_and_covers_the_window():
  st = _state(pos=(24, 24))
  seen = M.update_seen(None, st)
  assert seen[24 // M.CELL, 24 // M.CELL], 'own cell must be marked'
  n_first = seen.sum()
  seen = M.update_seen(seen, _state(pos=(4, 4)))
  assert seen.sum() > n_first, 'visitation must accumulate, not reset'
  assert seen[24 // M.CELL, 24 // M.CELL], 'old cells must stay marked'


def test_placed_blocks_land_on_their_own_planes():
  blocks = np.full((M.MAP_SIZE, M.MAP_SIZE), 2, np.int32)
  blocks[20, 20] = 11                                 # CRAFTING_TABLE
  blocks[20, 24] = 12                                 # FURNACE
  m = M.coarse_map(_state(blocks))
  assert m[5, 5, M.P_TABLE] == 1.0
  assert m[5, 6, M.P_FURNACE] == 1.0
  assert m[:, :, M.P_TABLE].sum() == 1.0, 'table must not leak to other cells'


def test_crop_is_egocentric_and_zero_padded():
  m = np.zeros((M.COARSE, M.COARSE, M.N_PLANES), np.float32)
  m[6, 5, M.P_STONE] = 1.0
  cell = 6 * M.COARSE + 5
  crop = M.crop_egocentric(m, cell, size=9)
  assert crop.shape == (9, 9, M.N_PLANES)
  assert crop[4, 4, M.P_STONE] == 1.0, 'agent cell belongs at the centre'

  corner = M.crop_egocentric(m, 0, size=9)            # cell (0, 0)
  assert corner[:4].sum() == 0.0, 'off-map must be zero-padded'


def test_direction_is_preserved_by_the_crop():
  """The reason no CNN is used: position in the crop *is* the instruction."""
  m = np.zeros((M.COARSE, M.COARSE, M.N_PLANES), np.float32)
  m[4, 5, M.P_STONE] = 1.0                            # two cells above (6, 5)
  crop = M.crop_egocentric(m, 6 * M.COARSE + 5, size=9)
  ys, xs = np.nonzero(crop[:, :, M.P_STONE])
  assert (ys[0], xs[0]) == (2, 4), (ys, xs)           # up from centre (4, 4)
