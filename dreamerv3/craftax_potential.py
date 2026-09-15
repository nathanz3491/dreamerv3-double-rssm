"""Potential function over Craftax's tech-tree spine.

Craftax pays a flat one-shot bonus per achievement and nothing for the work in
between, so the value of a tier falls with its depth: MAKE_WOOD_PICKAXE is
weight 1 and roughly 30 steps away, ENTER_FIRE_REALM is weight 8 and a thousand.
Measured over 1.1M steps, our agent never pressed the craft key once.

This turns each achievement into a RAMP instead of a cliff. Every prerequisite
satisfied moves a potential upward, and the shaped reward is the difference

    F(s, s') = gamma * PHI(s') - PHI(s)

which is potential-based (Ng, Harada & Russell 1999) and therefore provably
cannot change which policy is optimal. That property is the whole reason for the
form: three earlier hand-written bonus terms each created a cheaper way to earn
than the behaviour they were meant to encourage -- `alive` paid for standing
still, `restore` was never collected once, `idle` was defeated by jiggling in
place. Under a potential, every loop cancels by construction:

    collect wood (+w) then drop it (-w)            -> exactly zero
    stand still in a good state                    -> (gamma-1)*PHI < 0, a bleed
    walk toward stone and back                     -> exactly zero

Only ENDING somewhere better than you started pays.

Two design points that are easy to get backwards:

* An unlocked achievement keeps contributing its FULL weight. Drop it from the
  sum on success and PHI falls at the moment of victory, i.e. the agent is
  punished for winning.
* Satisfying every prerequisite caps progress at PREREQ_CEIL (< 1), so the
  final keypress that actually completes the achievement is worth the remaining
  1 - PREREQ_CEIL in one jump -- the single largest increment on the ramp.

Pure numpy, no JAX, no agent dependency -- same posture as craftax_map.py, so it
unit-tests on a laptop.
"""

import numpy as np

# --- geometry ---------------------------------------------------------------
# constants.CLOSE_BLOCKS: the 8-neighbourhood Craftax itself uses for
# `is_near_block`, which gates every craft. Matching it exactly matters: a
# potential that disagrees with the game rewards approaches that cannot craft.
CLOSE = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))

# BlockType ids (craftax.craftax.constants.BlockType)
STONE, TREE, COAL, IRON, DIAMOND = 4, 5, 8, 9, 10
CRAFTING_TABLE, FURNACE, WATER = 11, 12, 3

# How much of an achievement's ramp the prerequisites can fill. The remainder is
# paid on the achievement actually firing, so pressing the key is worth more than
# any single prerequisite.
#
# Tuned rather than guessed, because crafting CONSUMES its ingredients: making a
# wood pickaxe spends the wood, which lowers progress on every other recipe that
# needs wood (sword, stone pickaxe, table). At the obvious 0.6 the keypress was
# worth +0.25 against +2.60 for merely walking to the table -- the completing
# action was the cheapest step on its own ramp. At 0.25 the keypress is worth
# ~2x the last prerequisite even after paying for the ingredients it burns.
PREREQ_CEIL = 0.25

# --- the spine --------------------------------------------------------------
# Only the tier chain that gates the plateau. The deep content (fire realm,
# dungeons) is unreachable while the agent is stuck at wood, and weighting
# things it cannot approach would only dilute the signal.
#
#   inv    minimum inventory counts (pickaxe/sword are TIERS: 1 wood, 2 stone,
#          3 iron, 4 diamond)
#   near   block ids that must be in the 8-neighbourhood
SPINE = {
    'PLACE_TABLE': dict(inv=dict(wood=1), near=()),
    'MAKE_WOOD_PICKAXE': dict(inv=dict(wood=1), near=(CRAFTING_TABLE,)),
    'MAKE_WOOD_SWORD': dict(inv=dict(wood=1), near=(CRAFTING_TABLE,)),
    'COLLECT_STONE': dict(inv=dict(pickaxe=1), near=(STONE,)),
    'PLACE_STONE': dict(inv=dict(stone=1), near=()),
    'PLACE_FURNACE': dict(inv=dict(stone=1), near=()),
    'MAKE_STONE_PICKAXE': dict(
        inv=dict(wood=1, stone=1), near=(CRAFTING_TABLE,)),
    'MAKE_STONE_SWORD': dict(
        inv=dict(wood=1, stone=1), near=(CRAFTING_TABLE,)),
    'COLLECT_COAL': dict(inv=dict(pickaxe=1), near=(COAL,)),
    'COLLECT_IRON': dict(inv=dict(pickaxe=2), near=(IRON,)),
    'MAKE_IRON_PICKAXE': dict(
        inv=dict(wood=1, coal=1, iron=1), near=(CRAFTING_TABLE, FURNACE)),
    'MAKE_IRON_SWORD': dict(
        inv=dict(wood=1, coal=1, iron=1), near=(CRAFTING_TABLE, FURNACE)),
    'COLLECT_DIAMOND': dict(inv=dict(pickaxe=3), near=(DIAMOND,)),
}

# Weight per achievement in the spine. Deeper tiers are worth more, mirroring
# Craftax's own 1/3/5/8 scale but applied to the ramp rather than the cliff.
TIER_WEIGHT = {
    'PLACE_TABLE': 1.0,
    'MAKE_WOOD_PICKAXE': 3.0,      # the gate everything else sits behind
    'MAKE_WOOD_SWORD': 1.0,
    'COLLECT_STONE': 2.0,
    'PLACE_STONE': 1.0,
    'PLACE_FURNACE': 1.5,
    'MAKE_STONE_PICKAXE': 2.5,
    'MAKE_STONE_SWORD': 1.5,
    'COLLECT_COAL': 1.5,
    'COLLECT_IRON': 2.5,
    'MAKE_IRON_PICKAXE': 3.0,
    'MAKE_IRON_SWORD': 2.0,
    'COLLECT_DIAMOND': 2.0,
}

# Holding a tool is worth something EVERY step, not once. This is the
# compounding return Craftax lacks: a pickaxe is currently a receipt for a
# reward already spent, so the middle of the tech tree has no instrumental
# value. Under this term, capability is an asset and losing it costs.
CAPABILITY = dict(pickaxe=2.0, sword=1.2)
CAPABILITY_MAX_TIER = 4.0


def _near(blocks, pos, wanted):
  """Is any wanted block in the 8-neighbourhood? Mirrors is_near_block."""
  h, w = blocks.shape
  y, x = int(pos[0]), int(pos[1])
  for dy, dx in CLOSE:
    yy, xx = y + dy, x + dx
    if 0 <= yy < h and 0 <= xx < w and int(blocks[yy, xx]) in wanted:
      return True
  return False


def _blocks_of_level(state):
  blocks = np.asarray(state.map)
  if blocks.ndim == 3:
    blocks = blocks[int(state.player_level)]
  return blocks


def progress(state, name, unlocked):
  """How far along this achievement's ramp the agent stands, in [0, 1].

  Unlocked -> 1.0. Otherwise the fraction of prerequisites met, scaled by
  PREREQ_CEIL so the completing action is always worth a further jump.
  """
  if unlocked:
    return 1.0
  spec = SPINE[name]
  conds = []
  inv = state.inventory
  for field, need in spec['inv'].items():
    conds.append(float(np.asarray(getattr(inv, field))) >= need)
  if spec['near']:
    blocks = _blocks_of_level(state)
    pos = np.asarray(state.player_position).reshape(2)
    for block in spec['near']:
      conds.append(_near(blocks, pos, (block,)))
  if not conds:
    return 0.0
  return PREREQ_CEIL * (sum(bool(c) for c in conds) / len(conds))


def potential(state, ach_names=None, scale=1.0):
  """PHI(s): tech-tree progress plus capability held, as a single scalar.

  ``ach_names`` maps achievement index -> name (constants.Achievement order);
  pass it once and reuse. ``scale`` divides the result, so the caller can set
  how large PHI is relative to Craftax's own achievement rewards.
  """
  achieved = np.asarray(state.achievements, bool).reshape(-1)
  total = 0.0
  for i, name in enumerate(ach_names or ()):
    if name in SPINE:
      total += TIER_WEIGHT[name] * progress(state, name, bool(achieved[i]))
  inv = state.inventory
  for field, weight in CAPABILITY.items():
    tier = float(np.asarray(getattr(inv, field)))
    total += weight * min(tier, CAPABILITY_MAX_TIER) / CAPABILITY_MAX_TIER
  return total / scale


def max_potential(scale=1.0):
  """Ceiling of PHI, for sizing `scale` against Craftax's own rewards."""
  return (sum(TIER_WEIGHT.values()) + sum(CAPABILITY.values())) / scale


def shaped(prev_phi, phi, gamma):
  """F = gamma * PHI(s') - PHI(s). Returns 0.0 on the first step of an episode.

  Potential-based, so it cannot change the optimal policy -- it only moves
  credit earlier in time.
  """
  if prev_phi is None:
    return 0.0
  return float(gamma * phi - prev_phi)
