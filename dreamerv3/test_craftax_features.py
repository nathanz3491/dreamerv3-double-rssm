"""Unit tests for craftax_features (pure numpy; runs without JAX/GPU).

Run:  python dreamerv3/dreamerv3/test_craftax_features.py
"""

import numpy as np

import craftax_features as cf


def test_names_count():
    assert cf.NUM_ACHIEVEMENTS == 67
    assert len(set(cf.ACHIEVEMENT_NAMES)) == 67  # no duplicates


def test_phi_table_shape():
    tbl = cf.build_phi_table()
    assert tbl.shape == (67, cf.PHI_DIM), tbl.shape
    assert tbl.dtype == np.float32
    assert len(cf.PHI_FEATURE_NAMES) == cf.PHI_DIM


def test_pickaxe_line_is_ordered():
    """The whole thesis: wood/stone/iron pickaxes differ only by ordered tier."""
    tbl = cf.build_phi_table()
    wood = tbl[cf.ACHIEVEMENT_NAMES.index("MAKE_WOOD_PICKAXE")]
    stone = tbl[cf.ACHIEVEMENT_NAMES.index("MAKE_STONE_PICKAXE")]
    iron = tbl[cf.ACHIEVEMENT_NAMES.index("MAKE_IRON_PICKAXE")]

    def feat(vec, name):
        return vec[cf.PHI_FEATURE_NAMES.index(name)]

    # All three: verb=MAKE, tool=PICKAXE.
    for vec in (wood, stone, iron):
        assert feat(vec, "verb:MAKE") == 1.0
        assert feat(vec, "tool:PICKAXE") == 1.0
    # Tier is a strict, ordered thermometer 1 < 2 < 3.
    assert feat(wood, "tier>=1") == 1 and feat(wood, "tier>=2") == 0
    assert feat(stone, "tier>=2") == 1 and feat(stone, "tier>=3") == 0
    assert feat(iron, "tier>=3") == 1
    # Extrapolation sanity: iron's phi == stone's phi + one more tier bit
    # + material shifted. The tier thermometer strictly contains stone's.
    assert (iron[_tier_slice()] >= stone[_tier_slice()]).all()
    assert (stone[_tier_slice()] >= wood[_tier_slice()]).all()


def _tier_slice():
    lo = cf.PHI_FEATURE_NAMES.index("tier>=1")
    return slice(lo, lo + cf.MAX_TIER)


def test_iron_material_is_grounded_elsewhere():
    """Iron appears as a COLLECT material, so material=IRON is learnable even
    if MAKE_IRON_PICKAXE is held out."""
    tbl = cf.build_phi_table()
    ci = tbl[cf.ACHIEVEMENT_NAMES.index("COLLECT_IRON")]
    mi = tbl[cf.ACHIEVEMENT_NAMES.index("MAKE_IRON_PICKAXE")]
    idx = cf.PHI_FEATURE_NAMES.index("mat:IRON")
    assert ci[idx] == 1.0 and mi[idx] == 1.0


def test_weight_table():
    w = cf.build_weight_table()
    assert w.shape == (67,)
    # Deep achievements weigh more than early ones.
    assert w[cf.ACHIEVEMENT_NAMES.index("MAKE_IRON_PICKAXE")] > \
        w[cf.ACHIEVEMENT_NAMES.index("COLLECT_WOOD")]


def test_newly_unlocked():
    prev = np.zeros(67, bool)
    curr = np.zeros(67, bool)
    curr[9] = True  # COLLECT_STONE just unlocked
    prev[0] = curr[0] = True  # COLLECT_WOOD already had it -> not "new"
    nu = cf.newly_unlocked(prev, curr)
    assert nu[9] == 1.0 and nu[0] == 0.0
    assert nu.sum() == 1.0


def test_tech_tree_decomposition():
    unlocked = np.zeros(67, bool)
    iron_pick = cf.ACHIEVEMENT_NAMES.index("MAKE_IRON_PICKAXE")
    # From scratch, the nearest subgoal toward iron pickaxe is collecting wood.
    sub = cf.nearest_unsatisfied_subgoal(iron_pick, unlocked)
    assert cf.ACHIEVEMENT_NAMES[sub] == "COLLECT_WOOD", cf.ACHIEVEMENT_NAMES[sub]

    # Give it the whole wood tier -> next frontier should advance to stone.
    for n in ("COLLECT_WOOD", "PLACE_TABLE", "MAKE_WOOD_PICKAXE"):
        unlocked[cf.ACHIEVEMENT_NAMES.index(n)] = True
    sub = cf.nearest_unsatisfied_subgoal(iron_pick, unlocked)
    assert cf.ACHIEVEMENT_NAMES[sub] == "COLLECT_STONE", cf.ACHIEVEMENT_NAMES[sub]

    # Everything but the final craft -> subgoal is the goal itself.
    for n in ("COLLECT_STONE", "MAKE_STONE_PICKAXE", "COLLECT_IRON",
              "COLLECT_COAL", "PLACE_FURNACE"):
        unlocked[cf.ACHIEVEMENT_NAMES.index(n)] = True
    sub = cf.nearest_unsatisfied_subgoal(iron_pick, unlocked)
    assert sub == iron_pick, cf.ACHIEVEMENT_NAMES[sub]

    # Already unlocked -> None.
    unlocked[iron_pick] = True
    assert cf.nearest_unsatisfied_subgoal(iron_pick, unlocked) is None


def test_scorer_discriminates():
    import rewcause_eval as rce
    target = rce.iron_pickaxe_phi()

    # A perfect extrapolator predicts iron-pickaxe's true phi -> PASS.
    good = np.tile(target[None, :], (32, 1)).astype(np.float32)
    m = rce.score_predictions(good)
    assert m["extrapolated"] is True, rce.format_report(m)
    assert m["overall_feature_acc"] == 1.0

    # A memorizer that never saw iron predicts all-zeros -> FAIL (misses the
    # defining features like tier>=3, tool:PICKAXE).
    bad = np.zeros((32, cf.PHI_DIM), np.float32)
    m = rce.score_predictions(bad)
    assert m["extrapolated"] is False, rce.format_report(m)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
