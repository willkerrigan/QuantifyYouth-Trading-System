import pytest
from optimizer.grid_search import GridSearchGenerator

def test_grid_search_generation():
    param_ranges = {"ma_short": [10, 20], "ma_long": [50, 100]}
    combos = list(GridSearchGenerator.generate_combinations(param_ranges))
    assert len(combos) == 4
    assert {"ma_short": 10, "ma_long": 50} in combos

def test_combination_count():
    param_ranges = {"ma_short": [10, 20, 30], "ma_long": [50, 100, 200], "exit_threshold": [0.02, 0.03]}
    count = GridSearchGenerator.count_combinations(param_ranges)
    assert count == 18
