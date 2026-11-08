"""Craftax achievement schema (numpy only; testable without JAX or a GPU).

The achievement names are embedded in enum-VALUE order (0..66), exactly
matching craftax.craftax.constants.Achievement; call
`validate_against_craftax()` where Craftax is installed to assert they still
match the installed version. The env adapter (embodied/envs/craftax.py) reads
NUM_ACHIEVEMENTS for the length of obs['ach'].
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
