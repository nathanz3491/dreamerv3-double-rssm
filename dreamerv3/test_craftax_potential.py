"""Tests for the tech-tree potential. Pure numpy -- no JAX, no GPU.

The farming tests are the point. Three previous hand-written bonus terms each
created a cheaper way to earn than the behaviour they encoded; these assert that
the potential form cannot, by checking the algebra rather than trusting it.
"""

import types

import numpy as np

from dreamerv3 import craftax_potential as P

N_ACH = 67
NAMES = None  # filled by _names()


def _names():
  """Achievement index -> name. Falls back to a stub when craftax is absent."""
  global NAMES
  if NAMES is None:
    try:
      from craftax.craftax.constants import Achievement
      NAMES = [a.name for a in Achievement]
    except Exception:
      NAMES = [f'ACH_{i}' for i in range(N_ACH)]
      for i, n in enumerate(P.SPINE):     # put the spine somewhere findable
        NAMES[i] = n
  return NAMES


def _state(wood=0, stone=0, coal=0, iron=0, pickaxe=0, sword=0,
           near=(), pos=(24, 24), achieved=()):
  """Minimal duck-typed EnvState carrying only what the potential reads."""
  blocks = np.full((48, 48), 2, np.int32)          # GRASS
  y, x = pos
  for i, block in enumerate(near):                 # drop each adjacent
    dy, dx = P.CLOSE[i]
    blocks[y + dy, x + dx] = block
  ach = np.zeros(len(_names()), bool)
  for name in achieved:
    ach[_names().index(name)] = True
  inv = types.SimpleNamespace(
      wood=wood, stone=stone, coal=coal, iron=iron, diamond=0,
      pickaxe=pickaxe, sword=sword)
  return types.SimpleNamespace(
      map=blocks, player_position=np.array(pos, np.int32), player_level=0,
      inventory=inv, achievements=ach)


def phi(state):
  return P.potential(state, _names())


# --- the ramp ---------------------------------------------------------------
def test_prerequisites_raise_the_potential_one_at_a_time():
  """MAKE_WOOD_PICKAXE needs wood AND a table; each should pay separately."""
  none = phi(_state())
  wood = phi(_state(wood=1))
  both = phi(_state(wood=1, near=(P.CRAFTING_TABLE,)))
  assert wood > none, 'holding wood must be progress'
  assert both > wood, 'standing at a table with wood must be more progress'


def test_pressing_the_key_is_worth_more_than_any_single_prerequisite():
  """The completing action is the biggest single step on the ramp."""
  ready = _state(wood=1, near=(P.CRAFTING_TABLE,))
  done = _state(wood=0, pickaxe=1, near=(P.CRAFTING_TABLE,),
                achieved=('MAKE_WOOD_PICKAXE',))
  step_of_last_prereq = phi(ready) - phi(_state(wood=1))
  step_of_the_keypress = phi(done) - phi(ready)
  assert step_of_the_keypress > step_of_last_prereq, (
      step_of_the_keypress, step_of_last_prereq)


def test_unlocking_never_lowers_the_potential():
  """The inversion bug: drop unlocked achievements from PHI and success hurts."""
  for name in ('PLACE_TABLE', 'MAKE_WOOD_PICKAXE', 'COLLECT_STONE'):
    before = phi(_state(wood=1, stone=1, pickaxe=1, near=(P.STONE,)))
    after = phi(_state(wood=1, stone=1, pickaxe=1, near=(P.STONE,),
                       achieved=(name,)))
    assert after >= before, f'{name} unlocking lowered PHI'


def test_deeper_tiers_are_worth_more():
  assert P.TIER_WEIGHT['MAKE_IRON_PICKAXE'] > P.TIER_WEIGHT['PLACE_TABLE']
  assert P.TIER_WEIGHT['MAKE_WOOD_PICKAXE'] > P.TIER_WEIGHT['MAKE_WOOD_SWORD']


# --- the anti-farming properties -------------------------------------------
GAMMA = 1 - 1 / 333


def test_a_round_trip_pays_exactly_zero():
  """Collect wood then lose it: the two shaped rewards must cancel."""
  empty, holding = phi(_state()), phi(_state(wood=1))
  gain = P.shaped(empty, holding, GAMMA)
  loss = P.shaped(holding, empty, GAMMA)
  assert gain > 0 and loss < 0
  # Not exactly zero because gamma discounts; the residual must be tiny.
  assert abs(gain + loss) < 0.02 * max(abs(gain), 1e-9) + 0.01, (gain, loss)


def test_walking_to_a_resource_and_away_pays_zero():
  near = _state(pickaxe=1, near=(P.STONE,))
  away = _state(pickaxe=1)
  there = P.shaped(phi(away), phi(near), GAMMA)
  back = P.shaped(phi(near), phi(away), GAMMA)
  assert there > 0
  assert there + back < 0.01, 'a pointless walk must not pay'


def test_standing_still_bleeds():
  """The failure that killed `alive`: doing nothing must never be profitable."""
  for state in (_state(), _state(wood=3, pickaxe=1),
                _state(wood=1, pickaxe=2, near=(P.CRAFTING_TABLE,))):
    value = phi(state)
    assert P.shaped(value, value, GAMMA) <= 0, (
        'standing still paid a positive reward')


def test_capability_is_worth_holding_every_step():
  """A pickaxe must be an asset, not a spent receipt."""
  assert phi(_state(pickaxe=1)) > phi(_state(pickaxe=0))
  assert phi(_state(pickaxe=2)) > phi(_state(pickaxe=1))


def test_first_step_of_an_episode_is_neutral():
  assert P.shaped(None, 5.0, GAMMA) == 0.0


def test_scale_divides_the_whole_potential():
  plain = phi(_state(wood=1, pickaxe=1))
  scaled = P.potential(_state(wood=1, pickaxe=1), _names(), scale=4.0)
  assert abs(scaled * 4.0 - plain) < 1e-6
  assert P.max_potential(4.0) == P.max_potential(1.0) / 4.0


# --- the spine must agree with the game -------------------------------------
# The ramp is only useful if "prerequisites satisfied" means the same thing to
# us as to Craftax. It did not: PLACE_TABLE was listed at 1 wood while
# place_block spends 2, so the ramp told the agent it was ready one log early,
# the keypress silently failed, and nothing on the ramp asked for the second
# log. The table gates every craft, so the whole spine stalled there.
#
# Recipe costs read out of craftax.craftax.game_logic (do_crafting and
# place_block). Kept as literals so the test runs without craftax installed,
# and cross-checked against the installed package when there is one.
GAME_COSTS = {
    'PLACE_TABLE': dict(wood=2),
    'MAKE_WOOD_PICKAXE': dict(wood=1),
    'MAKE_WOOD_SWORD': dict(wood=1),
    'PLACE_STONE': dict(stone=1),
    'PLACE_FURNACE': dict(stone=1),
    'MAKE_STONE_PICKAXE': dict(wood=1, stone=1),
    'MAKE_STONE_SWORD': dict(wood=1, stone=1),
    'MAKE_IRON_PICKAXE': dict(wood=1, stone=1, coal=1, iron=1),
    'MAKE_IRON_SWORD': dict(wood=1, stone=1, coal=1, iron=1),
}


def test_spine_ingredients_match_the_game():
  for name, cost in GAME_COSTS.items():
    ours = {k: v for k, v in P.SPINE[name]['inv'].items()
            if k not in ('pickaxe', 'sword')}
    assert ours == cost, f'{name}: spine says {ours}, game wants {cost}'


def test_one_log_is_not_enough_for_a_table():
  """The rung that stalled the tech tree, asserted directly."""
  one = P.progress(_state(wood=1), 'PLACE_TABLE', False)
  two = P.progress(_state(wood=2), 'PLACE_TABLE', False)
  assert one < P.PREREQ_CEIL, 'one log must not read as fully prepared'
  assert two == P.PREREQ_CEIL
  assert phi(_state(wood=2)) > phi(_state(wood=1)), (
      'chopping the second log must pay, or the agent will not bother')


# --- death hands the potential back -----------------------------------------
def test_dying_pays_back_the_whole_climb():
  """Climb to a pickaxe, then die: net shaping must be ~0, not +PHI(pickaxe).

  Without PHI = 0 at the absorbing state the sum telescopes to
  gamma^T * PHI(s_T) - PHI(s_0), and dying with a pickaxe would keep the reward
  for earning it -- the potential would be paying the agent to die at high tech.
  """
  g = 0.997
  path = [_state(), _state(wood=1), _state(wood=2),
          _state(wood=1, near=(P.CRAFTING_TABLE,), pickaxe=1,
                 achieved=('PLACE_TABLE', 'MAKE_WOOD_PICKAXE'))]
  phis = [phi(s) for s in path]
  total, prev = 0.0, None
  for t, p_t in enumerate(phis):
    terminal = t == len(phis) - 1
    total += g ** t * P.shaped(prev, p_t, g, terminal=terminal)
    prev = p_t
  assert phis[-1] > 1.0, 'the climb must be worth something to hand back'
  # Every step telescopes away except the start, which is ~0 at spawn.
  assert abs(total + g * phis[0]) < 1e-6, total


def test_timeout_keeps_its_potential():
  """Timeouts bootstrap; they are not deaths and must not be charged."""
  a, b = phi(_state(wood=1)), phi(_state(wood=2))
  assert P.shaped(a, b, 0.997) == P.shaped(a, b, 0.997, terminal=False)
  assert P.shaped(a, b, 0.997, terminal=True) == -a
