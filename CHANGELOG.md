# Changelog

All notable changes to the double-RSSM Craftax work are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **Tech-tree potential asked for one log too few to place a crafting table.**
  `craftax_potential.SPINE` listed `PLACE_TABLE` at `wood=1`; Craftax's
  `place_block` requires `inventory.wood >= 2` and spends both. The ramp
  therefore read "fully prepared" one log early, the keypress silently failed,
  and no rung on the ramp paid for chopping the second log. Because the table
  gates every craft in the game, the whole spine stalled behind this off-by-one
  (`dreamerv3/craftax_potential.py`).
- `MAKE_IRON_PICKAXE` and `MAKE_IRON_SWORD` were missing `stone` from their
  ingredient lists; both recipes consume wood, stone, coal and iron.

### Added
- `test_spine_ingredients_match_the_game` and
  `test_one_log_is_not_enough_for_a_table` in
  `dreamerv3/test_craftax_potential.py`, pinning every spine recipe against the
  costs read out of `craftax.craftax.game_logic`, so the table cannot drift away
  from the game again.
