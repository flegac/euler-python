from typing import Tuple, Callable, List

import numpy as np

LParser = Callable[[str], List[int]]


def LINE_SPLITTER(line: str):
    return [int(_) for _ in line.split(' ')]


class Grid(object):
    def __init__(self, raw_data: str, line_parser: LParser = LINE_SPLITTER):
        self.grid = np.array(self._parse(raw_data, line_parser))

    @property
    def width(self):
        return self.grid.shape[1]

    @property
    def height(self):
        return self.grid.shape[0]

    def read(self, start: Tuple[int, int], direction: Tuple[int, int], n: int):
        res = []
        for i in range(n):
            x = start[0] + i * direction[0]
            y = start[1] + i * direction[1]
            res.append(self.grid[x, y])

        return res

    @staticmethod
    def _parse(raw_data: str, line_parser: LParser):
        lines = [line_parser(_) for _ in raw_data.split('\n')]
        return lines
