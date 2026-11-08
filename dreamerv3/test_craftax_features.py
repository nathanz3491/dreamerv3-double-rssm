"""Unit tests for craftax_features (pure numpy; runs without JAX/GPU).

Run:  python -m pytest dreamerv3/test_craftax_features.py
"""

import numpy as np

from dreamerv3 import craftax_features as cf


def test_names_count():
    assert cf.NUM_ACHIEVEMENTS == 67
    assert len(set(cf.ACHIEVEMENT_NAMES)) == 67  # no duplicates


def test_newly_unlocked():
    prev = np.zeros(67, bool)
    curr = np.zeros(67, bool)
    curr[9] = True  # COLLECT_STONE just unlocked
    prev[0] = curr[0] = True  # COLLECT_WOOD already had it -> not "new"
    nu = cf.newly_unlocked(prev, curr)
    assert nu[9] == 1.0 and nu[0] == 0.0
    assert nu.sum() == 1.0
