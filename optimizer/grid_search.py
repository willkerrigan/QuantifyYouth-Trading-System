from itertools import product
from typing import Dict, Generator


class GridSearchGenerator:
    @staticmethod
    def generate_combinations(param_ranges: Dict) -> Generator[Dict, None, None]:
        keys = param_ranges.keys()
        value_lists = [param_ranges[key] for key in keys]
        for combo in product(*value_lists):
            yield dict(zip(keys, combo))

    @staticmethod
    def count_combinations(param_ranges: Dict) -> int:
        count = 1
        for values in param_ranges.values():
            count *= len(values)
        return count
