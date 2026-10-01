"""Which of Craftax's 43 actions can do anything right now -- from the observation.

Every precondition in the game is visible to the agent: inventory, potions,
XP, mana, attributes and learned spells are in the 51 scalars; whether a table
or furnace is adjacent, what the agent is facing and whether it stands on a
ladder are in the 9x11 view. So validity is a function of the observation
vector alone, and this module takes nothing else -- like
``craftax_map.ObservedTargets``, it cannot read the game state.

Used for experiment B (docs/entropy-and-action-suppression.md): tell the agent
what is possible, or mask what is not, during training. Checked against the
game itself (``tools/action_suppression.make_oracle``) in
``test_craftax_valid.py``.

Rules transcribed from craftax.craftax.game_logic: do_crafting, place_block,
drink_potion, read_book, shoot_projectile, cast_spell, enchant,
level_up_attributes, change_floor, update_player_intrinsics (sleep, rest), and
craftax_step (a sleeping or resting agent's action becomes NOOP).

NOOP, the four moves and DO are always reported valid. They are the agent's
basic means of exploring, DO's outcome can depend on chance (saplings from
grass), and nothing is gained by masking them.
"""

import numpy as np

from dreamerv3 import craftax_map as M

N_ACTIONS = 43
BASIC = (0, 1, 2, 3, 4, 5)          # NOOP, LEFT, RIGHT, UP, DOWN, DO

# BlockType / ItemType ids (craftax.craftax.constants)
GRASS, WATER, TABLE, FURNACE = 2, 3, 11, 12
ENCHANT_FIRE, ENCHANT_ICE = 30, 31
CAN_PLACE_ITEM = (2, 13, 7, 25, 26)  # grass, sand, path, fire grass, ice grass
ITEM_NONE, LADDER_DOWN, LADDER_UP, LADDER_DOWN_BLOCKED = 0, 2, 3, 4
MAX_ATTRIBUTE = 5                   # EnvParams.max_attribute
MAX_LEVEL = 8                       # StaticEnvParams.num_levels - 1
EPS = 1e-3                          # float meters are /10-scaled in the obs

_FACING = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # direction one-hot
# constants.CLOSE_BLOCKS: the 8-neighbourhood behind is_near_block.
_AROUND = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _count(v):
  """Inverse of the renderer's sqrt(n) / 10 scaling."""
  return int(round((float(v) * 10.0) ** 2))


def decode_stats(vec):
  """The 51 scalars after the map, back in game units."""
  s = np.asarray(vec, np.float32).reshape(-1)[-51:]
  inv, pot, intr = s[0:16], s[16:22], s[22:31]
  return dict(
      wood=_count(inv[0]), stone=_count(inv[1]), coal=_count(inv[2]),
      iron=_count(inv[3]), diamond=_count(inv[4]), sapphire=_count(inv[5]),
      ruby=_count(inv[6]), sapling=_count(inv[7]), torches=_count(inv[8]),
      arrows=_count(inv[9]), books=int(round(inv[10] * 2)),
      pickaxe=int(round(inv[11] * 4)), sword=int(round(inv[12] * 4)),
      bow=int(round(inv[15])),
      potions=[_count(v) for v in pot],
      health=intr[0] * 10, food=intr[1] * 10, drink=intr[2] * 10,
      energy=intr[3] * 10, mana=intr[4] * 10, xp=intr[5] * 10,
      dex=int(round(intr[6] * 10)), str=int(round(intr[7] * 10)),
      int=int(round(intr[8] * 10)),
      direction=int(np.argmax(s[31:35])),
      armour=[int(round(v * 2)) for v in s[35:39]],
      sleeping=bool(s[44] > 0.5), resting=bool(s[45] > 0.5),
      spells=(bool(s[46] > 0.5), bool(s[47] > 0.5)),
      level=int(round(float(s[48]) * 10)), cleared=bool(s[49] > 0.5))


def _items(vec):
  """(9, 11) item ids, -1 where dark."""
  vec = np.asarray(vec, np.float32).reshape(-1)
  tiles = vec[:M.OBS_H * M.OBS_W * M.N_TILE].reshape(M.OBS_H, M.OBS_W, M.N_TILE)
  lit = tiles[..., -1] > 0.5
  items = tiles[..., M.N_BLOCK:M.N_BLOCK + M.N_ITEM].argmax(-1)
  return np.where(lit, items, -1)


def valid_actions(vec):
  """(43,) bool: can each action change the game, as far as the agent can see."""
  view = M.decode_view(vec)
  st = decode_stats(vec)
  blocks, items = view['blocks'], _items(vec)
  cy, cx = M.OBS_H // 2, M.OBS_W // 2
  out = np.zeros(N_ACTIONS, bool)
  out[list(BASIC)] = True
  if st['sleeping'] or st['resting']:
    return out                        # craftax_step: the action becomes NOOP

  around = {int(blocks[cy + dy, cx + dx]) for dy, dx in _AROUND}
  table, furnace = TABLE in around, FURNACE in around
  dy, dx = _FACING[st['direction']]
  face_block = int(blocks[cy + dy, cx + dx])
  face_item = int(items[cy + dy, cx + dx])
  under = int(items[cy, cx])
  known = face_block != M.UNKNOWN and face_item >= 0
  # place_block undoes any placement onto a tile a mob stands on.
  mob_ahead = bool(view['passive'][cy + dy, cx + dx]
                   or view['hostile'][cy + dy, cx + dx])
  placeable = known and not mob_ahead
  free = placeable and face_block not in M.SOLID_IDS and face_item == ITEM_NONE

  w, s, c, i, d = (st[k] for k in ('wood', 'stone', 'coal', 'iron', 'diamond'))
  pick, sword = st['pickaxe'], st['sword']

  out[6] = st['energy'] < 7 + 2 * st['dex'] - EPS                    # SLEEP
  out[7] = s >= 1 and placeable and (face_block == WATER or free)    # PLACE_STONE
  out[8] = w >= 2 and free                                           # PLACE_TABLE
  out[9] = s >= 1 and free                                           # PLACE_FURNACE
  out[10] = (st['sapling'] >= 1 and placeable and face_block == GRASS
             and face_item == ITEM_NONE)                             # PLACE_PLANT
  out[11] = table and w >= 1 and pick < 1                            # wood pickaxe
  out[12] = table and w >= 1 and s >= 1 and pick < 2                 # stone pickaxe
  iron_ok = table and furnace and min(w, s, i, c) >= 1
  out[13] = iron_ok and pick < 3                                     # iron pickaxe
  out[14] = table and w >= 1 and sword < 1                           # wood sword
  out[15] = table and w >= 1 and s >= 1 and sword < 2                # stone sword
  out[16] = iron_ok and sword < 3                                    # iron sword
  # Resting with food or drink at zero wakes the agent again in the same step
  # (update_player_intrinsics), so the press changes nothing.
  out[17] = (st['health'] < 8 + st['str'] - EPS and st['food'] > EPS
             and st['drink'] > EPS)                                  # REST
  out[18] = (under in (LADDER_DOWN, LADDER_DOWN_BLOCKED) and st['cleared']
             and st['level'] < MAX_LEVEL)                            # DESCEND
  out[19] = under == LADDER_UP and st['level'] > 0                   # ASCEND
  out[20] = table and w >= 1 and d >= 3 and pick < 4                 # diamond pickaxe
  out[21] = table and w >= 1 and d >= 2 and sword < 4                # diamond sword
  out[22] = (table and furnace and min(st['armour']) < 1
             and i >= 3 and c >= 3)                                  # iron armour
  out[23] = table and min(st['armour']) < 2 and d >= 3               # diamond armour
  out[24] = st['bow'] >= 1 and st['arrows'] >= 1                     # SHOOT_ARROW
  out[25] = table and s >= 1 and w >= 1 and st['arrows'] < 99        # MAKE_ARROW
  out[26] = st['mana'] >= 2 - EPS and st['spells'][0]                # FIREBALL
  out[27] = st['mana'] >= 2 - EPS and st['spells'][1]                # ICEBALL
  out[28] = (st['torches'] >= 1 and placeable
             and face_block in CAN_PLACE_ITEM
             and face_item == ITEM_NONE)                             # PLACE_TORCH
  for k in range(6):
    out[29 + k] = st['potions'][k] >= 1                              # potions
  out[35] = st['books'] >= 1                                         # READ_BOOK
  ench = face_block in (ENCHANT_FIRE, ENCHANT_ICE)
  gems = st['ruby'] if face_block == ENCHANT_FIRE else st['sapphire']
  can_ench = ench and st['mana'] >= 9 - EPS and gems >= 1
  out[36] = can_ench and sword > 0                                   # ENCHANT_SWORD
  out[37] = can_ench and sum(st['armour']) > 0                       # ENCHANT_ARMOUR
  out[38] = table and c >= 1 and w >= 1 and st['torches'] < 99       # MAKE_TORCH
  lvl = st['xp'] >= 1 - EPS
  out[39] = lvl and st['dex'] < MAX_ATTRIBUTE                        # LEVEL_UP_DEX
  out[40] = lvl and st['str'] < MAX_ATTRIBUTE                        # LEVEL_UP_STR
  out[41] = lvl and st['int'] < MAX_ATTRIBUTE                        # LEVEL_UP_INT
  out[42] = can_ench and st['bow'] > 0                               # ENCHANT_BOW
  return out
