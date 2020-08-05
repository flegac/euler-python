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

    def solve_triangle(self, data: List[List[int]] = None):
        if data is None:
            data = self.grid.tolist()

        if len(data) == 1:
            return data[0][0]

        for i in range(len(data[-2])):
            a = data[-2][i] + data[-1][i]
            b = data[-2][i] + data[-1][i + 1]
            data[-2][i] = max(a, b)
        return self.solve_triangle(data[:-1])
