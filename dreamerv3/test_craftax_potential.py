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
