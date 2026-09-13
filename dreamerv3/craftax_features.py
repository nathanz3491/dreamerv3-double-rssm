"""Compositional achievement featurization for Craftax (Phase 1 core).

Self-contained and dependency-light (numpy only) so it can be unit-tested
without JAX or a GPU. The achievement names are embedded in enum-VALUE order
(0..66) exactly matching craftax.craftax.constants.Achievement; call
`validate_against_craftax()` on the GPU box to assert they still match your
installed Craftax version.

The point of this module: turn each achievement into a *factored feature
vector* phi, so that having seen wood-pickaxe (tier 1) and stone-pickaxe
(tier 2) lets a model predict iron-pickaxe's (tier 3) reward-event features
*before iron is ever mined*. That compositional transfer is the whole thesis
of Phase 1.
"""

import numpy as np

# Achievement names in enum-VALUE order (index i == Achievement value i).
# Mirror of craftax.craftax.constants.Achievement (67 achievements, 0..66).
ACHIEVEMENT_NAMES = [
    "COLLECT_WOOD",          # 0
    "PLACE_TABLE",           # 1
    "EAT_COW",               # 2
    "COLLECT_SAPLING",       # 3
    "COLLECT_DRINK",         # 4
    "MAKE_WOOD_PICKAXE",     # 5
    "MAKE_WOOD_SWORD",       # 6
    "PLACE_PLANT",           # 7
    "DEFEAT_ZOMBIE",         # 8
    "COLLECT_STONE",         # 9
    "PLACE_STONE",           # 10
    "EAT_PLANT",             # 11
    "DEFEAT_SKELETON",       # 12
    "MAKE_STONE_PICKAXE",    # 13
    "MAKE_STONE_SWORD",      # 14
    "WAKE_UP",               # 15
    "PLACE_FURNACE",         # 16
    "COLLECT_COAL",          # 17
    "COLLECT_IRON",          # 18
    "COLLECT_DIAMOND",       # 19
    "MAKE_IRON_PICKAXE",     # 20
    "MAKE_IRON_SWORD",       # 21
    "MAKE_ARROW",            # 22
    "MAKE_TORCH",            # 23
    "PLACE_TORCH",           # 24
    "MAKE_DIAMOND_SWORD",    # 25
    "MAKE_IRON_ARMOUR",      # 26
    "MAKE_DIAMOND_ARMOUR",   # 27
    "ENTER_GNOMISH_MINES",   # 28
    "ENTER_DUNGEON",         # 29
    "ENTER_SEWERS",          # 30
    "ENTER_VAULT",           # 31
    "ENTER_TROLL_MINES",     # 32
    "ENTER_FIRE_REALM",      # 33
    "ENTER_ICE_REALM",       # 34
    "ENTER_GRAVEYARD",       # 35
    "DEFEAT_GNOME_WARRIOR",  # 36
    "DEFEAT_GNOME_ARCHER",   # 37
    "DEFEAT_ORC_SOLIDER",    # 38
    "DEFEAT_ORC_MAGE",       # 39
    "DEFEAT_LIZARD",         # 40
    "DEFEAT_KOBOLD",         # 41
    "DEFEAT_TROLL",          # 42
    "DEFEAT_DEEP_THING",     # 43
    "DEFEAT_PIGMAN",         # 44
    "DEFEAT_FIRE_ELEMENTAL", # 45
    "DEFEAT_FROST_TROLL",    # 46
    "DEFEAT_ICE_ELEMENTAL",  # 47
    "DAMAGE_NECROMANCER",    # 48
    "DEFEAT_NECROMANCER",    # 49
    "EAT_BAT",               # 50
    "EAT_SNAIL",             # 51
    "FIND_BOW",              # 52
    "FIRE_BOW",              # 53
    "COLLECT_SAPPHIRE",      # 54
    "LEARN_FIREBALL",        # 55
    "CAST_FIREBALL",         # 56
    "LEARN_ICEBALL",         # 57
    "CAST_ICEBALL",          # 58
    "COLLECT_RUBY",          # 59
    "MAKE_DIAMOND_PICKAXE",  # 60
    "OPEN_CHEST",            # 61
    "DRINK_POTION",          # 62
    "ENCHANT_SWORD",         # 63
    "ENCHANT_ARMOUR",        # 64
    "DEFEAT_KNIGHT",         # 65
    "DEFEAT_ARCHER",         # 66
]
NUM_ACHIEVEMENTS = len(ACHIEVEMENT_NAMES)  # 67

# --- Feature vocabularies (the factored axes of phi) ---------------------------
VERBS = [
    "COLLECT", "MAKE", "PLACE", "EAT", "DEFEAT", "ENTER",
    "DRINK", "OPEN", "ENCHANT", "LEARN", "CAST", "FIND", "FIRE",
    "WAKE", "DAMAGE",
]
# Ordered material tiers. The ORDER matters: a thermometer encoding of tier is
# what lets a model extrapolate tier 1 + tier 2 -> tier 3.
MATERIALS = ["WOOD", "STONE", "COAL", "IRON", "DIAMOND", "SAPPHIRE", "RUBY"]
MATERIAL_TIER = {
    "WOOD": 1, "STONE": 2, "COAL": 2, "IRON": 3,
    "DIAMOND": 4, "SAPPHIRE": 4, "RUBY": 4,
}
MAX_TIER = 4
TOOL_LINES = ["PICKAXE", "SWORD", "ARMOUR", "ARROW", "TORCH", "BOW"]

# Reward magnitude per achievement value, mirroring craftax's
# achievement_mapping(): <=24 -> 1, intermediate -> 3, very-advanced -> 8,
# else -> 5. Used as a rarity/importance proxy for loss re-balancing.
_INTERMEDIATE = {54, 59, 60, 25, 26, 27, 28, 29, 36, 37, 38, 39, 50, 51, 52, 53, 61, 62}
_VERY_ADVANCED = {33, 34, 35, 44, 45, 46, 47, 48, 49}


def _reward_tier(value):
    if value <= 24:
        return 1
    if value in _INTERMEDIATE:
        return 3
    if value in _VERY_ADVANCED:
        return 8
    return 5


def _onehot(index, size):
    v = np.zeros(size, np.float32)
    if index is not None and 0 <= index < size:
        v[index] = 1.0
    return v


def _thermometer(level, size):
    """level=3, size=4 -> [1,1,1,0]. Ordered => enables tier extrapolation."""
    v = np.zeros(size, np.float32)
    v[: max(0, min(level, size))] = 1.0
    return v


def achievement_phi(name):
    """Build the factored feature vector phi for one achievement name."""
    tokens = name.split("_")
    verb = tokens[0]
    # Some verbs are multi-token-safe; first token is the verb for all 67.
    verb_oh = _onehot(VERBS.index(verb) if verb in VERBS else None, len(VERBS))

    material = next((m for m in MATERIALS if m in tokens), None)
    material_oh = _onehot(
        MATERIALS.index(material) if material else None, len(MATERIALS))
    tier = MATERIAL_TIER.get(material, 0)
    tier_therm = _thermometer(tier, MAX_TIER)

    tool = next((t for t in TOOL_LINES if t in tokens), None)
    tool_oh = _onehot(
        TOOL_LINES.index(tool) if tool else None, len(TOOL_LINES))

    return np.concatenate([verb_oh, material_oh, tier_therm, tool_oh], 0)


# Human-readable names for each phi dimension (debugging / reports).
PHI_FEATURE_NAMES = (
    [f"verb:{v}" for v in VERBS]
    + [f"mat:{m}" for m in MATERIALS]
    + [f"tier>={i+1}" for i in range(MAX_TIER)]
    + [f"tool:{t}" for t in TOOL_LINES]
)
PHI_DIM = len(PHI_FEATURE_NAMES)


def build_phi_table():
    """[NUM_ACHIEVEMENTS, PHI_DIM] float32 feature table (row i = achievement i)."""
    return np.stack([achievement_phi(n) for n in ACHIEVEMENT_NAMES], 0).astype(np.float32)


def build_weight_table(scale=1.0):
    """[NUM_ACHIEVEMENTS] per-achievement loss weight.

    Craftax's own reward magnitude is a poor rarity proxy (it gives iron-pickaxe
    the same reward as wood). Instead weight by how deep the achievement sits in
    the crafting spine (longer prerequisite chain -> rarer -> upweight), falling
    back to the reward tier for achievements outside the modeled tech tree.
    """
    return np.array(
        [scale * max(1 + _tree_depth(i), _reward_tier(i))
         for i in range(NUM_ACHIEVEMENTS)],
        np.float32,
    )


# --- Tech-tree prerequisites (Phase 3 subgoal decomposition) -------------------
# value -> list of prerequisite achievement values. Minimal crafting spine;
# extend as needed. Absent keys mean "no modeled prerequisite".
def _v(name):
    return ACHIEVEMENT_NAMES.index(name)


TECH_TREE = {
    _v("MAKE_WOOD_PICKAXE"): [_v("COLLECT_WOOD"), _v("PLACE_TABLE")],
    _v("MAKE_WOOD_SWORD"): [_v("COLLECT_WOOD"), _v("PLACE_TABLE")],
    _v("PLACE_TABLE"): [_v("COLLECT_WOOD")],
    _v("COLLECT_STONE"): [_v("MAKE_WOOD_PICKAXE")],
    _v("PLACE_STONE"): [_v("COLLECT_STONE")],
    _v("MAKE_STONE_PICKAXE"): [_v("COLLECT_STONE"), _v("PLACE_TABLE")],
    _v("MAKE_STONE_SWORD"): [_v("COLLECT_STONE"), _v("PLACE_TABLE")],
    _v("PLACE_FURNACE"): [_v("COLLECT_STONE")],
    _v("COLLECT_COAL"): [_v("MAKE_WOOD_PICKAXE")],
    _v("COLLECT_IRON"): [_v("MAKE_STONE_PICKAXE")],
    _v("MAKE_IRON_PICKAXE"): [
        _v("COLLECT_IRON"), _v("COLLECT_COAL"),
        _v("PLACE_FURNACE"), _v("PLACE_TABLE"),
    ],
    _v("MAKE_IRON_SWORD"): [
        _v("COLLECT_IRON"), _v("COLLECT_COAL"),
        _v("PLACE_FURNACE"), _v("PLACE_TABLE"),
    ],
    _v("COLLECT_DIAMOND"): [_v("MAKE_IRON_PICKAXE")],
    _v("MAKE_DIAMOND_PICKAXE"): [_v("COLLECT_DIAMOND"), _v("PLACE_TABLE")],
    _v("MAKE_DIAMOND_SWORD"): [_v("COLLECT_DIAMOND"), _v("PLACE_TABLE")],
    _v("MAKE_IRON_ARMOUR"): [_v("COLLECT_IRON"), _v("PLACE_FURNACE")],
    _v("MAKE_DIAMOND_ARMOUR"): [_v("COLLECT_DIAMOND"), _v("PLACE_FURNACE")],
}


def _tree_depth(value, _stack=None):
    """Longest prerequisite chain length to reach `value` (0 if a root)."""
    _stack = _stack or set()
    prereqs = TECH_TREE.get(value, [])
    if not prereqs or value in _stack:
        return 0
    _stack = _stack | {value}
    return 1 + max(_tree_depth(p, _stack) for p in prereqs)


def nearest_unsatisfied_subgoal(goal_value, unlocked):
    """Depth-first walk of TECH_TREE to the shallowest unmet prerequisite.

    `unlocked` is a boolean array/list of length NUM_ACHIEVEMENTS. Returns the
    achievement value of the nearest subgoal to pursue toward `goal_value`
    (which may be `goal_value` itself if all its prereqs are met), or None if
    the goal is already unlocked.
    """
    unlocked = np.asarray(unlocked, bool)
    if unlocked[goal_value]:
        return None
    visiting = set()

    def walk(v):
        if unlocked[v]:
            return None
        if v in visiting:  # cycle guard
            return v
        visiting.add(v)
        for pre in TECH_TREE.get(v, []):
            if not unlocked[pre]:
                deeper = walk(pre)
                if deeper is not None:
                    return deeper
        return v  # all prereqs satisfied (or none) -> this is the frontier

    return walk(goal_value)


# --- Per-step target helpers (used by the agent loss) --------------------------
def newly_unlocked(prev_ach, curr_ach):
    """Multi-hot of achievements that flipped False->True this step."""
    prev_ach = np.asarray(prev_ach, bool)
    curr_ach = np.asarray(curr_ach, bool)
    return (curr_ach & ~prev_ach).astype(np.float32)


def validate_against_craftax():
    """On the GPU box (Craftax importable): assert embedded names/count match."""
    from craftax.craftax.constants import Achievement
    assert len(Achievement) == NUM_ACHIEVEMENTS, (
        len(Achievement), NUM_ACHIEVEMENTS)
    for a in Achievement:
        assert ACHIEVEMENT_NAMES[a.value] == a.name, (
            a.value, a.name, ACHIEVEMENT_NAMES[a.value])
    return True
