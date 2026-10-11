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
      light_map=np.ones((M.MAP_SIZE, M.MAP_SIZE), np.float32),
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


def test_privileged_map_does_move_when_unseen_terrain_changes():
  """The contrast: coarse_map reads everything, which is why it is the old one."""
  blocks = np.full((M.MAP_SIZE, M.MAP_SIZE), 2, np.int32)
  far = blocks.copy()
  far[2, 2] = 3                                     # water nowhere near (24,24)
  assert not np.array_equal(M.coarse_map(_state(blocks)), M.coarse_map(_state(far)))


# --- observation-only targets -----------------------------------------------
# The "are we cheating" question reduces to one property: every honest label is
# a function of what the agent saw and did. ObservedTargets takes nothing else,
# so these tests check that it reads the observation correctly and behaves like
# an observer should; the Craftax-backed ones at the bottom check it against the
# real game.

GRASS, STONE, WATER, LAVA = 2, 4, 3, 14


def _vec(blocks=None, lit=None, passive=(), hostile=(), level=0,
         sleeping=False, resting=False):
  """Build an observation vector the way render_craftax_symbolic lays it out."""
  blocks = np.full((M.OBS_H, M.OBS_W), GRASS) if blocks is None else blocks
  lit = np.ones((M.OBS_H, M.OBS_W), bool) if lit is None else lit
  tiles = np.zeros((M.OBS_H, M.OBS_W, M.N_TILE), np.float32)
  for y in range(M.OBS_H):
    for x in range(M.OBS_W):
      tiles[y, x, blocks[y, x]] = 1.0
  base = M.N_BLOCK + M.N_ITEM
  for y, x in passive:
    tiles[y, x, base + 1 * 8] = 1.0                   # class 1, type 0 (cow)
  for y, x in hostile:
    tiles[y, x, base + 0 * 8] = 1.0                   # class 0, type 0 (zombie)
  tiles[..., :-1] *= lit[..., None]                   # the renderer's light mask
  tiles[..., -1] = lit
  stats = np.zeros(51, np.float32)
  stats[-7], stats[-6], stats[-3] = sleeping, resting, level / 10
  return np.concatenate([tiles.reshape(-1), stats])


CY, CX = M.OBS_H // 2, M.OBS_W // 2                   # the agent, in view


def _ahead(block=None, action=2, **kw):
  """A view with ``block`` directly in the direction of ``action``."""
  blocks = np.full((M.OBS_H, M.OBS_W), GRASS)
  dy, dx = M._MOVES[action]
  if block is not None:
    blocks[CY + dy, CX + dx] = block
  return M.decode_view(_vec(blocks, **kw))


def test_decode_reads_blocks_mobs_and_flags():
  blocks = np.full((M.OBS_H, M.OBS_W), GRASS)
  blocks[0, 0] = WATER
  lit = np.ones((M.OBS_H, M.OBS_W), bool)
  lit[8, 10] = False
  v = M.decode_view(_vec(blocks, lit, passive=[(1, 1)], hostile=[(2, 2)],
                         level=3, sleeping=True))
  assert v['blocks'][0, 0] == WATER and v['blocks'][4, 4] == GRASS
  assert v['blocks'][8, 10] == M.UNKNOWN, 'a dark tile must decode as unknown'
  assert v['passive'][1, 1] and v['hostile'][2, 2] and not v['passive'][2, 2]
  assert v['level'] == 3 and v['sleeping'] and not v['resting']


def test_reckon_moves_onto_open_ground():
  assert M.reckon((24, 24), _ahead(None, 2), 2) == (24, 25)
  assert M.reckon((24, 24), _ahead(None, 3), 3) == (23, 24)


def test_reckon_is_blocked_by_what_the_agent_can_see():
  for block in (STONE, WATER, LAVA):
    assert M.reckon((24, 24), _ahead(block, 2), 2) == (24, 24), block
  cow = M.decode_view(_vec(passive=[(CY, CX + 1)]))
  assert M.reckon((24, 24), cow, 2) == (24, 24)


def test_reckon_respects_the_edge_and_sleep():
  assert M.reckon((24, 47), _ahead(None, 2), 2) == (24, 47)
  assert M.reckon((24, 24), _ahead(None, 2, sleeping=True), 2) == (24, 24)
  assert M.reckon((24, 24), _ahead(None, 2), 5) == (24, 24)   # DO is no move


def test_builder_never_sees_the_game_state():
  """Structural: the only way in is (observation vector, action, is_first)."""
  import inspect
  params = list(inspect.signature(M.ObservedTargets.step).parameters)
  assert params == ['self', 'vec', 'action', 'is_first'], params


def test_unobserved_cells_carry_no_weight():
  t = M.ObservedTargets().step(_vec(), 0, True)
  assert t['mapknown'].max() > 0.0, 'the window it stands in must be observed'
  assert t['mapknown'].min() == 0.0, 'the far side of the map must not be'
  assert t['mapknown'].mean() < 0.15
  assert t['mappos'] == M.cell_of(M.SPAWN)


def test_the_mosaic_accumulates_as_the_agent_walks():
  obs, before = M.ObservedTargets(), None
  t = obs.step(_vec(), 0, True)
  for _ in range(12):
    t = obs.step(_vec(), 2, False)                    # RIGHT over open grass
    now = t['mapknown'].sum()
    assert before is None or now >= before, 'coverage must never shrink'
    before = now
  assert obs.pos == (M.SPAWN[0], M.SPAWN[1] + 12)


def test_darkness_hides_tiles_from_the_mosaic():
  blocks = np.full((M.OBS_H, M.OBS_W), GRASS)
  blocks[CY, CX + 2] = WATER
  dark = np.zeros((M.OBS_H, M.OBS_W), bool)
  lit = M.ObservedTargets().step(_vec(blocks), 0, True)
  unlit = M.ObservedTargets().step(_vec(blocks, dark), 0, True)
  assert lit['map12'][:, :, M.P_WATER].sum() > 0
  assert unlit['map12'][:, :, M.P_WATER].sum() == 0
  assert unlit['mapknown'].sum() == 0


def test_a_new_level_starts_a_fresh_frame():
  """After a ladder the agent cannot know where it is; the frame restarts."""
  obs = M.ObservedTargets()
  obs.step(_vec(), 0, True)
  for _ in range(5):
    obs.step(_vec(), 2, False)
  t = obs.step(_vec(level=1), 18, False)              # DESCEND
  assert obs.pos == M.SPAWN
  assert t['mapknown'].sum() == M.ObservedTargets().step(
      _vec(), 0, True)['mapknown'].sum()


# --- against the real game (skipped where Craftax is not installed) ----------
def _craftax():
  try:
    import jax
    from craftax.craftax_env import make_craftax_env_from_name
    from craftax.craftax import constants as C
  except Exception:
    return None
  return jax, make_craftax_env_from_name, C


def test_blocked_set_matches_craftax():
  lib = _craftax()
  if lib is None:
    return
  _, _, C = lib
  assert set(M.SOLID_IDS) == {int(b) for b in C.SOLID_BLOCKS}
  assert C.BlockType.WATER.value == WATER and C.BlockType.LAVA.value == LAVA
  assert len(C.BlockType) == M.N_BLOCK and len(C.ItemType) == M.N_ITEM
  assert tuple(C.OBS_DIM) == (M.OBS_H, M.OBS_W)


def test_observer_tracks_the_true_state_in_the_real_game():
  """Reckoned position and mosaic vs the truth, over long random rollouts.

  The observer is honest by construction; this measures whether it is also
  RIGHT -- a label that is honest but wrong is noise. On the surface it should
  agree with the true position almost always, and every tile it claims to know
  should match the true map.
  """
  lib = _craftax()
  if lib is None:
    return
  jax, make, _ = lib
  env = make('Craftax-Symbolic-v1', auto_reset=False)
  params = env.default_params
  reset = jax.jit(lambda k: env.reset(k, params))
  step = jax.jit(lambda k, s, a: env.step(k, s, a, params))
  rng = np.random.default_rng(0)
  key = jax.random.PRNGKey(0)
  agree = total = wrong_tiles = checked = 0
  for _ in range(4):
    key, k = jax.random.split(key)
    vec, state = reset(k)
    obs, first, last = M.ObservedTargets(), True, 0
    for _ in range(1500):
      obs.step(np.asarray(vec), last, first)
      first = False
      if obs.level == 0:
        true = tuple(int(v) for v in np.asarray(state.player_position))
        total += 1
        if obs.pos == true:
          agree += 1
          truth = np.asarray(state.map)[0]
          known = obs.known != M.UNKNOWN
          checked += int(known.sum())
          wrong_tiles += int((obs.known[known] != truth[known]).sum())
      # Mostly moves, so the reckoner is exercised; some DO so terrain changes.
      last = int(rng.choice([1, 2, 3, 4, 5], p=[.22, .22, .22, .22, .12]))
      key, k = jax.random.split(key)
      vec, state, _, done, _ = step(k, state, last)
      if bool(done):
        break
  rate = agree / max(total, 1)
  print(f'position agreement {rate:.4f} over {total} surface steps; '
        f'{wrong_tiles} wrong of {checked} known tiles')
  assert rate > 0.99, rate
  # Tiles change after they are seen (trees felled, stone mined elsewhere is
  # not revisited), so the mosaic can be stale but should rarely be wrong.
  assert wrong_tiles <= 0.02 * max(checked, 1), (wrong_tiles, checked)
