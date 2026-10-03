"""The manager's goals: tech-tree milestones, and progress toward each one.

The two-level agent (docs/design-manager.md) has a slow manager that, every 8
steps, picks ONE of these goals for the fast actor to pursue. The list is the
potential's tech-tree spine (craftax_potential.SPINE) -- the same game
knowledge the training reward already carries -- plus NONE, which leaves the
actor on the game reward alone. The manager is told which goals exist, never
in what order to pursue them: that it learns.

Each goal is a STATE to reach, judged from the observation vector alone, like
craftax_valid -- no game state, no achievement flags (the agent cannot see
those). "Make the wood pickaxe" is done when the inventory shows a pickaxe;
"place a table" when a table stands in the 8-neighbourhood -- an existing
table counts, so the goal never asks for a second one. Before a goal is done,
progress is the share of its prerequisites met, scaled by PREREQ_CEIL as in the
potential, so reaching the goal is always worth more than any prerequisite.

PLACE_STONE is left out: once the stone is placed, nothing in the observation
says the agent did it.

``progress(vec)`` is emitted by the env as ``obs['goalphi']``. It is a
TRAINING TARGET only: the agent learns a head that predicts it from the
latent, so it exists inside imagination, where the actor's goal reward is the
change in the predicted progress of its current goal.
"""

import numpy as np

from dreamerv3 import craftax_map as M
from dreamerv3 import craftax_potential as P
from dreamerv3 import craftax_valid as V

GOALS = (
    'NONE',
    'PLACE_TABLE', 'MAKE_WOOD_PICKAXE', 'MAKE_WOOD_SWORD', 'COLLECT_STONE',
    'PLACE_FURNACE', 'MAKE_STONE_PICKAXE', 'MAKE_STONE_SWORD',
    'COLLECT_COAL', 'COLLECT_IRON', 'MAKE_IRON_PICKAXE', 'MAKE_IRON_SWORD',
    'COLLECT_DIAMOND',
)
N_GOALS = len(GOALS)

# What the observation shows once each goal is reached.
_DONE = {
    'PLACE_TABLE': ('near', P.CRAFTING_TABLE),
    'MAKE_WOOD_PICKAXE': ('inv', 'pickaxe', 1),
    'MAKE_WOOD_SWORD': ('inv', 'sword', 1),
    'COLLECT_STONE': ('inv', 'stone', 1),
    'PLACE_FURNACE': ('near', P.FURNACE),
    'MAKE_STONE_PICKAXE': ('inv', 'pickaxe', 2),
    'MAKE_STONE_SWORD': ('inv', 'sword', 2),
    'COLLECT_COAL': ('inv', 'coal', 1),
    'COLLECT_IRON': ('inv', 'iron', 1),
    'MAKE_IRON_PICKAXE': ('inv', 'pickaxe', 3),
    'MAKE_IRON_SWORD': ('inv', 'sword', 3),
    'COLLECT_DIAMOND': ('inv', 'diamond', 1),
}
assert set(_DONE) == set(GOALS[1:]) and set(_DONE) <= set(P.SPINE)


def _around(vec):
  """Block ids in the 8-neighbourhood, as is_near_block sees them."""
  blocks = M.decode_view(vec)['blocks']
  cy, cx = M.OBS_H // 2, M.OBS_W // 2
  return {int(blocks[cy + dy, cx + dx]) for dy, dx in P.CLOSE}


def progress(vec):
  """(N_GOALS,) float32 in [0, 1]: how far along each goal the agent stands."""
  st = V.decode_stats(vec)
  near = _around(vec)
  out = np.zeros(N_GOALS, np.float32)
  for i, name in enumerate(GOALS[1:], 1):
    kind, *what = _DONE[name]
    done = (what[0] in near) if kind == 'near' else st[what[0]] >= what[1]
    if done:
      out[i] = 1.0
      continue
    spec = P.SPINE[name]
    conds = [st[k] >= need for k, need in spec['inv'].items()]
    conds += [b in near for b in spec['near']]
    out[i] = P.PREREQ_CEIL * (sum(conds) / len(conds)) if conds else 0.0
  return out
