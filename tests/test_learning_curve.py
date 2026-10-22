"""
Tests for the learning curve helpers
====================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.learning_curve import schedule, subset_indices


def test_subsets_are_nested_within_a_seed():
    small = set(subset_indices(9681, 100, seed=42))
    large = set(subset_indices(9681, 1000, seed=42))
    assert len(small) == 100 and len(large) == 1000
    assert small <= large


def test_seeds_draw_different_subsets():
    assert subset_indices(9681, 100, seed=42) != subset_indices(9681, 100, seed=43)


def test_size_larger_than_data_uses_everything():
    assert subset_indices(50, 100, seed=1) == list(range(50))


def test_schedule_reaches_min_steps_for_small_sets():
    # 100 examples at batch size 16 is 7 steps per epoch, so 29 epochs reach 200 steps
    assert schedule(100, batch_size=16, epochs=4, min_steps=200) == 29
    # 9,681 examples already give 606 steps per epoch
    assert schedule(9681, batch_size=16, epochs=4, min_steps=200) == 4
